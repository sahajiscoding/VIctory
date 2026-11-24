//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline at the front of the work queue (ahead of
//! speculative early-start pipelines) and waits for it within an adaptive
//! budget per frame: generous when only a couple of pipelines are pending
//! (menus, one-off bakes) and tight during heavy streaming bursts so frame
//! pacing stays responsive and the GPU watchdog never trips. Past the budget,
//! a draw or dispatch whose pipeline is not ready is dropped for that frame
//! while workers finish compiling it in the background.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(
            graphics
                .as_ref()
                .map(|g| g.describe())
                .unwrap_or_default()
                .as_bytes(),
        );
        Pending {
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        self.made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Per-frame waiting budget adapted to the current background queue depth.
///
/// Small queues (menus, splash screens, one-off LUT/IBL bakes) get a generous
/// budget so first-frame draws land immediately; large world-streaming bursts
/// switch to a short budget so command recording stays responsive and the
/// title's GPU watchdog does not fire.
pub(crate) fn frame_budget(waiting: u64) -> Duration {
    match waiting {
        0..=2 => Duration::from_millis(200),
        3..=8 => Duration::from_millis(40),
        9..=32 => Duration::from_millis(12),
        _ => Duration::from_millis(4),
    }
}

/// Worker count for background shader translation and PSO compilation.
fn worker_count(logical_cpus: usize) -> usize {
    (logical_cpus.saturating_mul(3) / 4).clamp(2, 12)
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use with urgent priority (ahead of speculative early
/// pipelines).
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p, true);
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// Queues `p` for the workers, once. If `urgent` is true and `p` was already
/// queued speculatively, promotes it to the front of the work queue.
fn start(p: &Arc<Pending>, urgent: bool) {
    if !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        enqueue(p.clone(), urgent);
    } else if urgent && p.made.get().is_none() {
        promote(p);
    }
}

/// At the title's creation of a pipeline: starts it at once at background
/// priority if an earlier run drew with it (during loading, so the world's
/// first frames find it ready).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while frame `frame` has adaptive budget left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static WAITED_US: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        WAITED_US.store(0, Ordering::Relaxed);
    }
    let budget = frame_budget(WAITING.load(Ordering::Relaxed));
    let left = budget.saturating_sub(Duration::from_micros(WAITED_US.load(Ordering::Relaxed)));
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    WAITED_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Priority queue shared by the pipeline compilation worker pool.
struct WorkQueue {
    items: Mutex<VecDeque<Arc<Pending>>>,
    available: Condvar,
}

impl WorkQueue {
    fn new() -> Self {
        Self {
            items: Mutex::new(VecDeque::new()),
            available: Condvar::new(),
        }
    }

    fn push(&self, p: Arc<Pending>, urgent: bool) {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        if urgent {
            items.push_front(p);
        } else {
            items.push_back(p);
        }
        self.available.notify_one();
    }

    /// Moves `p` to the front of the queue if it is still waiting.
    fn promote(&self, p: &Arc<Pending>) {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = items.iter().position(|item| Arc::ptr_eq(item, p))
            && pos > 0
            && let Some(item) = items.remove(pos)
        {
            items.push_front(item);
        }
    }

    fn pop(&self) -> Arc<Pending> {
        let mut items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(next) = items.pop_front() {
                return next;
            }
            items = self
                .available
                .wait(items)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
}

fn work_queue() -> Option<&'static Arc<WorkQueue>> {
    static QUEUE: OnceLock<Option<Arc<WorkQueue>>> = OnceLock::new();
    QUEUE
        .get_or_init(|| {
            let queue = Arc::new(WorkQueue::new());
            let workers =
                std::thread::available_parallelism().map_or(2, |n| worker_count(n.get()));
            let mut started = 0usize;
            for i in 0..workers {
                let queue = queue.clone();
                let spawned = std::thread::Builder::new()
                    .name(format!("victory-pipelines-{i}"))
                    .spawn(move || {
                        // Below the title's own threads (loading, recording).
                        // SAFETY: a plain call on our own thread.
                        unsafe {
                            use windows::Win32::System::Threading::*;
                            let _ =
                                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                        }
                        loop {
                            let p = queue.pop();
                            p.make();
                            if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                                drained();
                            } else {
                                super::library::schedule_save();
                            }
                        }
                    });
                match spawned {
                    Ok(_) => started += 1,
                    Err(e) => tracing::warn!(error = %e, "no pipeline worker"),
                }
            }
            (started > 0).then_some(queue)
        })
        .as_ref()
}

/// Hands `p` to the workers, started on first use.
fn enqueue(p: Arc<Pending>, urgent: bool) {
    if let Some(q) = work_queue() {
        q.push(p, urgent);
    } else {
        // No worker: made here instead.
        p.make();
        WAITING.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Moves a speculatively queued pipeline to the front when a frame actually
/// needs it now.
fn promote(p: &Arc<Pending>) {
    if let Some(q) = work_queue() {
        q.promote(p);
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// schedules a non-blocking cache save.
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::schedule_save();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn adaptive_budget_shrinks_under_world_bursts() {
        assert_eq!(frame_budget(0), Duration::from_millis(200));
        assert_eq!(frame_budget(2), Duration::from_millis(200));
        assert_eq!(frame_budget(5), Duration::from_millis(40));
        assert_eq!(frame_budget(20), Duration::from_millis(12));
        assert_eq!(frame_budget(1200), Duration::from_millis(4));
        assert!(frame_budget(1200) < frame_budget(20));
        assert!(frame_budget(20) < frame_budget(5));
        assert!(frame_budget(5) < frame_budget(1));
    }

    #[test]
    fn work_queue_prioritizes_urgent_and_promotes_early_items() {
        let q = WorkQueue::new();
        let early_a = Arc::new(Pending::new(Vec::new(), None));
        let early_b = Arc::new(Pending::new(Vec::new(), None));
        let urgent_c = Arc::new(Pending::new(Vec::new(), None));
        q.push(early_a.clone(), false);
        q.push(early_b.clone(), false);
        q.push(urgent_c.clone(), true);
        // Promoting `early_b` moves it ahead of `urgent_c` and `early_a`.
        q.promote(&early_b);
        assert!(Arc::ptr_eq(&q.pop(), &early_b));
        assert!(Arc::ptr_eq(&q.pop(), &urgent_c));
        assert!(Arc::ptr_eq(&q.pop(), &early_a));
    }

    #[test]
    fn worker_count_scales_within_bounds() {
        assert_eq!(worker_count(2), 2);
        assert_eq!(worker_count(4), 3);
        assert_eq!(worker_count(8), 6);
        assert_eq!(worker_count(16), 12);
        assert_eq!(worker_count(32), 12);
    }
}
