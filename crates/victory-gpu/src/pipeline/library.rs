//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
    /// Names already stored during this run, so concurrent workers compiling
    /// the same pipeline do not trigger duplicate `StorePipeline` errors.
    stored: HashSet<String>,
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 16384;
/// Minimum interval between background saves when the worker queue is quiet.
const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
/// Longer save interval while a heavy world-load compilation burst is running
/// so `Serialize` does not contend with worker PSO loads/stores.
const SAVE_BURST_EVERY: std::time::Duration = std::time::Duration::from_secs(60);
/// Queue depth above which background saves are deferred to [`SAVE_BURST_EVERY`].
const SAVE_BURST_QUEUE: u64 = 8;

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR
                .get()
                .and_then(|d| std::fs::read(d.join(LIBRARY_FILE)).ok())
                .unwrap_or_default();
            let library = open(&device, blob);
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    stored: HashSet::new(),
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: 0,
        stored: HashSet::new(),
    })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    if let Ok(pso) = load(&library.lock().unwrap_or_else(|e| e.into_inner()).library, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    if l.stored.insert(name.to_owned()) {
        // SAFETY: a live pipeline; skip `StorePipeline` if another worker
        // already stored the same name while we compiled outside the lock.
        if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
            l.added += 1;
        }
    }
    Ok(pso)
}

/// Required elapsed interval before triggering a background save, given the
/// number of pipelines currently waiting in the worker queue.
fn save_interval(waiting: u64) -> std::time::Duration {
    if waiting > SAVE_BURST_QUEUE {
        SAVE_BURST_EVERY
    } else {
        SAVE_EVERY
    }
}

/// Schedules a non-blocking background save of the pipeline library and early
/// list on a dedicated low-priority thread so compilation workers never stall
/// on serialization or disk I/O.
pub(crate) fn schedule_save() {
    static LAST_SCHEDULED: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    static SAVER: OnceLock<Option<std::sync::mpsc::SyncSender<()>>> = OnceLock::new();
    if DIR.get().is_none() || library().is_none() {
        return;
    }
    let interval = save_interval(super::make::waiting());
    {
        let mut last = LAST_SCHEDULED.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|t| t.elapsed() < interval) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    let sender = SAVER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
        match std::thread::Builder::new()
            .name("victory-pipeline-save".into())
            .spawn(move || {
                // SAFETY: a plain call on our own thread.
                unsafe {
                    use windows::Win32::System::Threading::*;
                    let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                while rx.recv().is_ok() {
                    save_inner(false);
                }
            }) {
            Ok(_) => Some(tx),
            Err(e) => {
                tracing::warn!(error = %e, "no pipeline cache saver thread");
                None
            }
        }
    });
    if let Some(tx) = sender {
        // Coalesce overlapping triggers without blocking the compiler worker.
        let _ = tx.try_send(());
    }
}

/// Writes the library and the early list if pipelines were added since the
/// last save and the interval has passed (or immediately when `now` is true,
/// e.g. on process shutdown).
pub(crate) fn save(now: bool) {
    save_inner(now);
}

fn save_inner(now: bool) {
    static SAVE_LOCK: Mutex<()> = Mutex::new(());
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    let _guard = SAVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let interval = save_interval(super::make::waiting());
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < interval) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    let blob = {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 && !USED_CHANGED.swap(false, Ordering::Relaxed) {
            return;
        }
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe {
            let mut bytes = vec![0u8; l.library.GetSerializedSize()];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                l.added = 0;
                bytes
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return;
            }
        }
    };
    let used: Vec<String> = USED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .flatten()
        .cloned()
        .collect();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| write_atomically(&dir.join(LIBRARY_FILE), &blob))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()));
    match written {
        Ok(()) => tracing::info!(
            bytes = blob.len(),
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes `bytes` to a unique temporary file beside `path`, then renames it
/// over `path`: a reader never sees a partial file and overlapping writers
/// never collide on the same temp path.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("cache");
    let tmp = path.with_file_name(format!("{name}.{}.{seq}.tmp", std::process::id()));
    if let Err(e) = std::fs::write(&tmp, bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Title pipeline keys drawn or dispatched with (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn save_interval_defers_during_compilation_bursts() {
        assert_eq!(save_interval(0), SAVE_EVERY);
        assert_eq!(save_interval(SAVE_BURST_QUEUE), SAVE_EVERY);
        assert_eq!(save_interval(SAVE_BURST_QUEUE + 1), SAVE_BURST_EVERY);
        assert!(SAVE_BURST_EVERY > SAVE_EVERY);
    }

    #[test]
    fn write_atomically_replaces_existing_file_cleanly() {
        let dir = std::env::temp_dir().join(format!("victory-lib-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        write_atomically(&path, b"first").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        write_atomically(&path, b"second-payload").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second-payload");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
