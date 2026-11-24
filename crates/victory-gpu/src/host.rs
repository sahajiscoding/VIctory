//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::{
    GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Completed host presents and submissions, used to retire released GPU
/// resources only after multi-list and indirect frames referencing them have
/// finished running.
static COMPLETED_FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COMPLETED_SUBMISSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RETIRED_RESOURCES: Mutex<Vec<(u64, u64, ID3D12Resource)>> = Mutex::new(Vec::new());

/// Dynamic descriptor-slot allocations belonging to lists that have already
/// finished executing on the host GPU, kept reclaimable if a descriptor heap
/// runs short before those lists are reset.
type DynamicSlots = Mutex<Vec<(usize, u32, u32)>>;
type DynamicSlotList = std::sync::Arc<DynamicSlots>;
type WeakDynamicSlotList = std::sync::Weak<DynamicSlots>;
static EXECUTED_DYNAMIC: Mutex<Vec<WeakDynamicSlotList>> = Mutex::new(Vec::new());

/// Reclaims dynamic descriptor slots for `target_heap` held by command lists
/// that have already completed execution, returning the total slots freed.
pub(crate) fn reclaim_dynamic_slots(target_heap: usize) -> usize {
    let mut to_release = Vec::new();
    {
        let mut reg = EXECUTED_DYNAMIC.lock().unwrap_or_else(|e| e.into_inner());
        reg.retain(|weak| {
            let Some(list_slots) = weak.upgrade() else {
                return false;
            };
            let mut slots = list_slots.lock().unwrap_or_else(|e| e.into_inner());
            let mut i = 0;
            while i < slots.len() {
                if slots[i].0 == target_heap {
                    to_release.push(slots.swap_remove(i));
                } else {
                    i += 1;
                }
            }
            !slots.is_empty()
        });
    }
    let mut reclaimed = 0usize;
    for (heap, first, n) in to_release {
        crate::descriptor::release_dynamic(heap, first, n);
        reclaimed += n as usize;
    }
    reclaimed
}

/// Keeps a released host GPU resource alive until several presented frames
/// (or many completed submissions during loading) have elapsed, so indirect
/// draws and directly-indexed heap reads from earlier frames cannot fault on
/// a freed streaming texture or shadow buffer.
pub(crate) fn retire_resource(resource: ID3D12Resource) {
    let frame = COMPLETED_FRAMES.load(Ordering::Relaxed);
    let sub = COMPLETED_SUBMISSIONS.load(Ordering::Relaxed);
    RETIRED_RESOURCES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((frame, sub, resource));
}

/// Drops retired GPU resources old enough that no live command list or
/// multi-list indirect frame can still reference them.
fn drain_retired(frame_lag: u64, submission_lag: u64) {
    let frame = COMPLETED_FRAMES.load(Ordering::Relaxed);
    let sub = COMPLETED_SUBMISSIONS.load(Ordering::Relaxed);
    let mut retired = RETIRED_RESOURCES
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if retired.is_empty() {
        return;
    }
    let mut drop_now = Vec::new();
    let mut i = 0;
    while i < retired.len() {
        let (rf, rs, _) = &retired[i];
        if frame.saturating_sub(*rf) >= frame_lag || sub.saturating_sub(*rs) >= submission_lag {
            drop_now.push(retired.swap_remove(i).2);
        } else {
            i += 1;
        }
    }
    drop(retired);
    drop(drop_now);
}

/// Whether a diagnostics environment switch is enabled: unset, empty, and
/// `0`/`false`/`no`/`off` (ASCII case-insensitive) are off; anything else is
/// on.
pub(crate) fn env_flag(name: &str) -> bool {
    parse_env_flag(std::env::var_os(name).as_deref())
}

fn parse_env_flag(value: Option<&std::ffi::OsStr>) -> bool {
    let Some(v) = value else {
        return false;
    };
    let Some(s) = v.to_str() else {
        return !v.is_empty();
    };
    let trimmed = s.trim();
    !trimmed.is_empty()
        && !trimmed.eq_ignore_ascii_case("0")
        && !trimmed.eq_ignore_ascii_case("false")
        && !trimmed.eq_ignore_ascii_case("no")
        && !trimmed.eq_ignore_ascii_case("off")
}

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A CPU-only ring of UAV descriptors for `ClearUnorderedAccessView*`.
    uavs: ID3D12DescriptorHeap,
    uav_size: u32,
    next_uav: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && env_flag("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if env_flag("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && env_flag("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else if let Some((adapter, desc)) = preferred_adapter() {
            match D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) {
                Ok(()) => {
                    let end = desc
                        .Description
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(desc.Description.len());
                    let gpu = String::from_utf16_lossy(&desc.Description[..end]);
                    let vram_mb = (desc.DedicatedVideoMemory as u64) >> 20;
                    tracing::info!(gpu = %gpu, vram_mb, "host GPU selected");
                }
                Err(_) => {
                    D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
                }
            }
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let uavs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let uav_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            uavs,
            uav_size,
            next_uav: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

/// Picks the preferred hardware D3D12 adapter: `VICTORY_ADAPTER` (index or
/// case-insensitive name substring) when set, otherwise `IDXGIFactory6`'s
/// `HIGH_PERFORMANCE` adapter so hybrid laptops pick their discrete GPU
/// instead of an iGPU (#208 / #1132).
fn preferred_adapter() -> Option<(IDXGIAdapter, DXGI_ADAPTER_DESC1)> {
    // SAFETY: plain DXGI factory and adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        if let Ok(sel) = std::env::var("VICTORY_ADAPTER") {
            let sel = sel.trim();
            if !sel.is_empty() {
                if let Ok(idx) = sel.parse::<u32>() {
                    if let Ok(a1) = factory.EnumAdapters1(idx)
                        && let Ok(desc) = a1.GetDesc1()
                        && let Ok(adapter) = a1.cast::<IDXGIAdapter>()
                    {
                        return Some((adapter, desc));
                    }
                } else {
                    let needle = sel.to_ascii_lowercase();
                    let mut i = 0u32;
                    while let Ok(a1) = factory.EnumAdapters1(i) {
                        i += 1;
                        if let Ok(desc) = a1.GetDesc1() {
                            let end = desc
                                .Description
                                .iter()
                                .position(|&c| c == 0)
                                .unwrap_or(desc.Description.len());
                            let name = String::from_utf16_lossy(&desc.Description[..end]);
                            if name.to_ascii_lowercase().contains(&needle)
                                && let Ok(adapter) = a1.cast::<IDXGIAdapter>()
                            {
                                return Some((adapter, desc));
                            }
                        }
                    }
                }
            }
        }
        let f6: IDXGIFactory6 = factory.cast().ok()?;
        let a1: IDXGIAdapter1 = f6
            .EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            .ok()?;
        let desc = a1.GetDesc1().ok()?;
        let adapter: IDXGIAdapter = a1.cast().ok()?;
        Some((adapter, desc))
    }
}

impl Drop for HostList {
    fn drop(&mut self) {
        self.release_dynamic_slots();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Host buffer states the list assumes, by shadow buffer: at its start
    /// and at its end, reconciled with each buffer's real state in `prelude`.
    buffer_states: Mutex<BufferStates>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Dynamic descriptor slots `(title_heap, first, count)` allocated for
    /// this list; released on reset/drop or reclaimed once executed if a heap
    /// runs short.
    dynamic_slots: DynamicSlotList,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

type BufferStates =
    std::collections::HashMap<usize, (u32, u32, std::sync::Arc<crate::shadow::Shadow>)>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && env_flag("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// A 64-bit memory wait recorded on a command list and satisfied before the
/// list executes on the host GPU.
#[derive(Debug, Clone, Copy)]
struct MemoryWait {
    address: usize,
    comparison: u32,
    reference: u64,
}

/// Evaluates a D3D12 `D3D12_COMPARISON_FUNC` (`1..=8`) against `current` and
/// `reference`. Unknown/zero codes and `NEVER` return `true` so a malformed
/// comparison never hangs submission.
fn memory_wait_satisfied(current: u64, comparison: u32, reference: u64) -> bool {
    match comparison {
        2 => current < reference,
        3 => current == reference,
        4 => current <= reference,
        5 => current > reference,
        6 => current != reference,
        7 => current >= reference,
        _ => true,
    }
}

/// Waits outside the submission mutex for `waits` to become satisfied by
/// another queue or the CPU, with a bounded timeout so a dead producer logs
/// once rather than deadlocking forever.
fn wait_for_memory(waits: &[MemoryWait]) {
    if waits.is_empty() {
        return;
    }
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
    let read_u64 = |addr: usize| -> u64 {
        if addr.is_multiple_of(8) {
            // SAFETY: title memory address recorded by the command list.
            unsafe { (*(addr as *const std::sync::atomic::AtomicU64)).load(Ordering::Acquire) }
        } else {
            // SAFETY: title memory address recorded by the command list.
            let v = unsafe { (addr as *const u64).read_unaligned() };
            std::sync::atomic::fence(Ordering::Acquire);
            v
        }
    };
    let satisfied = |w: &MemoryWait| memory_wait_satisfied(read_u64(w.address), w.comparison, w.reference);
    if waits.iter().all(satisfied) {
        return;
    }
    let started = std::time::Instant::now();
    let mut spins = 0u32;
    while !waits.iter().all(satisfied) {
        if started.elapsed() >= TIMEOUT {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let first = waits[0];
                let current = read_u64(first.address);
                tracing::warn!(
                    address = format_args!("{:#x}", first.address),
                    comparison = first.comparison,
                    reference = first.reference,
                    current,
                    "command-list 64-bit memory wait timed out; continuing"
                );
            });
            return;
        }
        spins += 1;
        if spins <= 64 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// 64-bit memory waits that must be satisfied before the list runs.
    waits: Vec<MemoryWait>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Post-execution callbacks run right after the GPU finishes (e.g.
    /// copying resolved query readback data into title memory), before
    /// bottom-of-pipe writes signal completion.
    after: Vec<Box<dyn Fn() + Send>>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host supports depth-bounds testing (`OMSetDepthBounds`
    /// and `D3D12_DEPTH_STENCIL_DESC1`).
    pub(crate) fn depth_bounds(&self) -> bool {
        static SUPPORTED: OnceLock<bool> = OnceLock::new();
        *SUPPORTED.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds test support");
            supported
        })
    }

    /// Maximum shader wave lane count supported by the host GPU (`32` on
    /// NVIDIA/Intel, `64` on AMD RDNA/GCN).
    pub(crate) fn wave_lane_count_max(&self) -> u32 {
        static MAX_LANES: OnceLock<u32> = OnceLock::new();
        *MAX_LANES.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let max = if ok && options.WaveLaneCountMax > 0 {
                options.WaveLaneCountMax
            } else {
                32
            };
            tracing::info!(
                min = options.WaveLaneCountMin,
                max,
                wave_ops = options.WaveOps.as_bool(),
                "host GPU wave lane count"
            );
            max
        })
    }

    /// Host GPU direct queue timestamp frequency in ticks per second.
    pub(crate) fn timestamp_frequency(&self) -> Option<u64> {
        static FREQ: OnceLock<Option<u64>> = OnceLock::new();
        *FREQ.get_or_init(|| {
            // SAFETY: plain query on our live direct command queue.
            let freq = unsafe { self.queue.GetTimestampFrequency() }.ok()?;
            (freq > 0).then_some(freq)
        })
    }

    /// A mapped buffer in the readback heap (`D3D12_RESOURCE_STATE_COPY_DEST`)
    /// for GPU-to-CPU readbacks such as `ResolveQueryData`.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and persistent mapping of a readback buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                buffer_states: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                dynamic_slots: std::sync::Arc::new(Mutex::new(Vec::new())),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Runs closed lists and waits for them. Splits before any list carrying
    /// a 64-bit memory wait so earlier lists in the same submission flush
    /// their writes first, and waits outside `self.submitted` so concurrent
    /// producer queues can satisfy the wait without deadlocking.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        let mut start = 0;
        while start < runs.len() {
            let waits = runs[start].deferred().waits.clone();
            wait_for_memory(&waits);
            let mut end = start + 1;
            while end < runs.len() && runs[end].deferred().waits.is_empty() {
                end += 1;
            }
            self.execute_batch(&runs[start..end]);
            start = end;
        }
    }

    fn execute_batch(&self, runs: &[&HostList]) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        for list in runs {
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        let prepared = std::time::Instant::now();
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        WaitForSingleObject(self.event, INFINITE);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        COMPLETED_SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
        // During long loading screens without presents, reclaim resources
        // retired more than 128 completed (fence-waited) submissions ago so
        // VRAM stays bounded while still outliving multi-list work.
        drain_retired(4, 128);
        let _ = &firsts;
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len) = step
                    && !back
                        .iter()
                        .any(|(s, a, l)| std::sync::Arc::ptr_eq(s, shadow) && (*a, *l) == (*address, *len))
                {
                    back.push((shadow.clone(), *address, *len));
                }
            }
        }
        for (shadow, address, len) in &back {
            let (address, len) = (*address, *len);
            let written = shadow.copy_back(address, len);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        for list in runs {
            for work in &list.deferred().after {
                work();
            }
        }
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        // These lists have completed execution on the host GPU; register
        // their dynamic descriptor slots as reclaimable under heap pressure.
        let mut reg = EXECUTED_DYNAMIC.lock().unwrap_or_else(|e| e.into_inner());
        reg.retain(|w| w.strong_count() > 0);
        for list in runs {
            if !list.dynamic_slots.lock().unwrap_or_else(|e| e.into_inner()).is_empty() {
                let weak = std::sync::Arc::downgrade(&list.dynamic_slots);
                if !reg.iter().any(|existing| existing.ptr_eq(&weak)) {
                    reg.push(weak);
                }
            }
        }
        drop(reg);
        self.log_messages();
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // Packed 16-bit textures widened to `R8G8B8A8_UNORM` (#797) need their
        // RTV descriptor format matched to the host resource's format.
        let adjusted = desc.map(|d| {
            let mut copy = *d;
            if matches!(copy.Format.0, 85 | 86 | 115) {
                // SAFETY: a live resource.
                let own = unsafe { resource.GetDesc() }.Format;
                if own == DXGI_FORMAT_R8G8B8A8_UNORM {
                    copy.Format = own;
                }
            }
            copy
        });
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device.CreateRenderTargetView(
                resource,
                adjusted.as_ref().map(|d| d as *const _),
                handle,
            );
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A non-shader-visible CPU UAV descriptor of `resource` for
    /// `ClearUnorderedAccessView*`, valid until recorded.
    pub(crate) fn uav_cpu(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_uav.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our CPU-only heap and a live resource.
        unsafe {
            let mut handle = self.uavs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.uav_size) as usize;
            self.device.CreateUnorderedAccessView(
                resource,
                None::<&ID3D12Resource>,
                desc.map(std::ptr::from_ref),
                handle,
            );
            handle
        }
    }

    /// A GPU-only default-heap buffer of `size` bytes starting in `state`
    /// (used for rebasing unaligned placed-buffer copy offsets on the GPU).
    pub(crate) fn default_buffer(
        &self,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain committed default-heap buffer creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no default buffer");
                None
            }
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.release_dynamic_slots();
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.buffer_states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Current tracked host state for a title texture in this list (falling
    /// back to the texture's global state if this list has not touched it yet).
    pub(crate) fn tracked_texture_state(&self, resource: crate::object::This) -> Option<u32> {
        let copy = crate::resource::host_copy(resource)?;
        let key = std::sync::Arc::as_ptr(&copy) as usize;
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        Some(
            states
                .get(&key)
                .map_or_else(|| copy.state.load(Ordering::Relaxed), |&(_, current, _)| current),
        )
    }

    /// Moves a title buffer's host copy to `wanted` (a host buffer state) in
    /// this list. The list's first use of a buffer records nothing: its
    /// prelude brings the buffer there at submission.
    pub(crate) fn ensure_buffer(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        wanted: u32,
    ) {
        let before = {
            let mut states = self
                .buffer_states
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            match states.entry(std::sync::Arc::as_ptr(shadow) as usize) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert((wanted, wanted, shadow.clone()));
                    None
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    let prev = std::mem::replace(&mut o.get_mut().1, wanted);
                    (prev != wanted).then_some(prev)
                }
            }
        };
        if let Some(before) = before {
            let barrier = transition(
                &shadow.buffer,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list; `buffer_states` keeps the buffer alive.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// At submission: the list that moves each texture and buffer this list
    /// uses from its real state to the one the list starts with; their real
    /// states become those the list ends with.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let buffer_states = self
            .buffer_states
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            })
            .collect();
        barriers.extend(buffer_states.values().filter_map(|(first, last, shadow)| {
            let real = shadow.state.swap(*last, Ordering::Relaxed);
            (real != *first).then(|| {
                transition(
                    &shadow.buffer,
                    D3D12_RESOURCE_STATES(real as i32),
                    D3D12_RESOURCE_STATES(*first as i32),
                )
            })
        }));
        if barriers.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                list.ResourceBarrier(&barriers);
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len));
    }

    /// If tracked GPU writes earlier in this same command list (not
    /// superseded by a later CPU copy/run) cover all of `[address, address +
    /// len)` inside one host shadow buffer, returns that shadow so a same-list
    /// consumer (such as placed buffer-to-texture `CopyTextureRegion`) reads
    /// the GPU-written bytes directly instead of stale CPU memory (#785).
    pub(crate) fn latest_gpu_write_shadow(
        &self,
        address: usize,
        len: usize,
    ) -> Option<std::sync::Arc<crate::shadow::Shadow>> {
        let end = address.checked_add(len)?;
        if len == 0 {
            return None;
        }
        let deferred = self.deferred();
        let mut target: Option<std::sync::Arc<crate::shadow::Shadow>> = None;
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for step in deferred.copies.iter().rev() {
            match step {
                Step::GpuWrite(shadow, w_start, w_len) => {
                    let w_end = w_start.saturating_add(*w_len);
                    if *w_start >= end || w_end <= address {
                        continue;
                    }
                    let same = match &target {
                        Some(t) => std::sync::Arc::ptr_eq(t, shadow),
                        None => {
                            target = Some(shadow.clone());
                            true
                        }
                    };
                    if !same {
                        continue;
                    }
                    ranges.push((*w_start, w_end));
                    if gpu_writes_cover(&ranges, address, end) {
                        return target;
                    }
                }
                Step::Copy(dst, _, copy_len) => {
                    let c_end = dst.saturating_add(*copy_len);
                    if *dst < end && address < c_end {
                        return None;
                    }
                }
                Step::Run(_) => return None,
                Step::Fill(..) | Step::Refresh(..) | Step::Host(_) | Step::Texture(_) => {}
            }
        }
        None
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// Waits for `*(address as *const u64)` to satisfy `comparison` against
    /// `reference` before this list executes on the host GPU.
    pub(crate) fn defer_wait(&self, address: u64, comparison: u32, reference: u64) {
        self.deferred().waits.push(MemoryWait {
            address: address as usize,
            comparison,
            reference,
        });
    }

    /// Tracks dynamic descriptor slots allocated while recording this list.
    pub(crate) fn record_dynamic(&self, heap: usize, first: u32, n: u32) {
        self.dynamic_slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((heap, first, n));
    }

    fn release_dynamic_slots(&self) {
        let slots = std::mem::take(
            &mut *self.dynamic_slots.lock().unwrap_or_else(|e| e.into_inner()),
        );
        for (heap, first, n) in slots {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        if s.used + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
            name(&buffer, || "list scratch".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            s.chunks.push((buffer, data as usize, gpu));
            s.used = 0;
        }
        let (_, data, gpu) = s.chunks[s.current];
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Run(Box::new(work)));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Runs `work` right after the list's GPU submission completes, before
    /// bottom-of-pipe writes signal completion (used for query readbacks).
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + 'static) {
        self.deferred().after.push(Box::new(work));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data.
                    batch.flush();
                    shadow.writing(*address, *len);
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        staging.refresh();
                    }
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        match self
            .refreshes
            .iter_mut()
            .find(|(s, ..)| std::sync::Arc::ptr_eq(s, shadow))
        {
            Some((_, from, to)) => (*from, *to) = ((*from).min(address), (*to).max(end)),
            None => self.refreshes.push((shadow.clone(), address, end)),
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
    }

    fn flush(&mut self) {
        for (shadow, from, to) in self.refreshes.drain(..) {
            let started = std::time::Instant::now();
            let copied = shadow.update(from, to - from);
            let us = started.elapsed().as_micros() as u64;
            #[cfg(feature = "diagnostics")]
            crate::copystat::range(from, to - from, us, 0);
            PREPARING[0].fetch_add(us, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 || title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    // Something is unmapped: scan virtual memory regions once with
    // `VirtualQuery`, reading committed readable regions in a single call and
    // zeroing unmapped/inaccessible gaps directly without probing them page by
    // page (#783).
    for_virtual_regions(src as usize, len, |at, n, state, protect| {
        let to = dst.wrapping_add(at - src as usize);
        if is_readable_region(state, protect) {
            if !read(at as *const u8, to, n) {
                for_pieces(at, n, 4 << 10, |at, n| {
                    let to = dst.wrapping_add(at - src as usize);
                    if !read(at as *const u8, to, n) {
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                        // SAFETY: inside the destination, which is mapped.
                        unsafe { to.write_bytes(0, n) };
                    }
                });
            }
        } else {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: inside the destination, which is mapped.
            unsafe { to.write_bytes(0, n) };
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_virtual_regions(dst as usize, len, |at, n, state, protect| {
        if is_writable_region(state, protect) && !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

fn is_readable_region(state: u32, protect: u32) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD,
        PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
    };
    const READABLE: u32 = PAGE_READONLY
        | PAGE_READWRITE
        | PAGE_WRITECOPY
        | PAGE_EXECUTE_READ
        | PAGE_EXECUTE_READWRITE
        | PAGE_EXECUTE_WRITECOPY;
    state == MEM_COMMIT
        && (protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0
        && (protect & READABLE) != 0
}

fn is_writable_region(state: u32, protect: u32) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS,
        PAGE_READWRITE, PAGE_WRITECOPY,
    };
    const WRITABLE: u32 =
        PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    state == MEM_COMMIT
        && (protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0
        && (protect & WRITABLE) != 0
}

/// Walks `[start, start + len)` region by region via `VirtualQuery`, calling
/// `f(region_start, region_len, state, protect)` for every slice.
fn for_virtual_regions(start: usize, len: usize, mut f: impl FnMut(usize, usize, u32, u32)) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let end = start.saturating_add(len);
    let mut at = start;
    while at < end {
        // SAFETY: plain out-buffer filled by `VirtualQuery`.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid pointer and exact struct size.
        let queried = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if queried == 0 || info.RegionSize == 0 {
            f(at, end - at, 0, 0);
            break;
        }
        let region_end = (info.BaseAddress as usize)
            .saturating_add(info.RegionSize)
            .min(end);
        let n = region_end.saturating_sub(at);
        if n == 0 {
            f(at, end - at, 0, 0);
            break;
        }
        f(at, n, info.State, info.Protect);
        at = region_end;
    }
}

/// Returns whether the union of `[start, end)` intervals in `writes` covers
/// every byte of `[address, end)`.
fn gpu_writes_cover(writes: &[(usize, usize)], address: usize, end: usize) -> bool {
    if address >= end {
        return true;
    }
    let mut clipped: Vec<(usize, usize)> = writes
        .iter()
        .filter_map(|&(a, b)| {
            let lo = a.max(address);
            let hi = b.min(end);
            (lo < hi).then_some((lo, hi))
        })
        .collect();
    if clipped.is_empty() {
        return false;
    }
    clipped.sort_unstable_by_key(|&(lo, _)| lo);
    let mut covered = address;
    for (lo, hi) in clipped {
        if lo > covered {
            return false;
        }
        covered = covered.max(hi);
        if covered >= end {
            return true;
        }
    }
    covered >= end
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// Host display HDR capability and luminance bounds, queried once from DXGI
/// (`IDXGIOutput6::GetDesc1`) with optional `VICTORY_HDR` override.
pub fn hdr_info() -> crate::HdrInfo {
    static INFO: OnceLock<crate::HdrInfo> = OnceLock::new();
    *INFO.get_or_init(|| {
        if let Some(raw) = std::env::var_os("VICTORY_HDR") {
            if !parse_env_flag(Some(&raw)) {
                return crate::HdrInfo::default();
            }
            return crate::HdrInfo {
                supported: true,
                min_nits: 0.005,
                max_nits: 1000.0,
                max_full_frame_nits: 400.0,
            };
        }
        detect_hdr_output().unwrap_or_default()
    })
}

fn detect_hdr_output() -> Option<crate::HdrInfo> {
    // SAFETY: read-only DXGI adapter/output enumeration.
    unsafe {
        let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let adapter = factory.EnumAdapters1(0).ok()?;
        let output = adapter.EnumOutputs(0).ok()?;
        let output6: IDXGIOutput6 = output.cast().ok()?;
        let desc = output6.GetDesc1().ok()?;
        let supported = desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
        let info = crate::HdrInfo {
            supported,
            min_nits: desc.MinLuminance.max(0.0),
            max_nits: if desc.MaxLuminance > 0.0 {
                desc.MaxLuminance
            } else {
                1000.0
            },
            max_full_frame_nits: if desc.MaxFullFrameLuminance > 0.0 {
                desc.MaxFullFrameLuminance
            } else {
                400.0
            },
        };
        tracing::info!(?info, color_space = desc.ColorSpace.0, "host display HDR probe");
        Some(info)
    }
}

/// Maps a presented texture format to a concrete SRV format usable by the
/// present-scaling blitter (resolving typeless and sRGB swapchain formats).
fn present_srv_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27 | 28 => DXGI_FORMAT_R8G8B8A8_UNORM,
        29 => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        87 | 90 => DXGI_FORMAT_B8G8R8A8_UNORM,
        91 => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        88 | 92 => DXGI_FORMAT_B8G8R8X8_UNORM,
        93 => DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,
        23 | 24 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9 | 10 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

/// Whether a presented texture format belongs to a different DXGI typeless
/// family than its swap-chain back buffer (`B8G8R8X8*` vs `B8G8R8A8_UNORM`),
/// requiring the blitter instead of `CopyResource`.
fn needs_format_blit(format: DXGI_FORMAT) -> bool {
    matches!(format.0, 88 | 92 | 93)
}

/// RTV format used by the present blitter on the swap-chain back buffer.
/// Concrete sRGB source textures decode sRGB -> linear on SRV sampling, so the
/// blitter must write through a matching sRGB RTV to encode back to sRGB.
fn present_blit_rtv_format(swap: DXGI_FORMAT, src: DXGI_FORMAT) -> DXGI_FORMAT {
    match (swap, src.0) {
        (DXGI_FORMAT_R8G8B8A8_UNORM, 29) => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        (DXGI_FORMAT_B8G8R8A8_UNORM, 91 | 93) => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        _ => swap,
    }
}

/// Swap chain color space matching the presented back-buffer format.
fn swap_color_space(format: DXGI_FORMAT, hdr: bool) -> DXGI_COLOR_SPACE_TYPE {
    match format {
        DXGI_FORMAT_R10G10B10A2_UNORM if hdr => DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        DXGI_FORMAT_R16G16B16A16_FLOAT => DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
        _ => DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
    }
}

/// Maximum number of `PresentX` display planes composited in one present.
const BLITTER_PLANES: u32 = 8;

/// Fullscreen bilinear scaling & multi-plane compositing blit pipeline used
/// when presenting textures whose resolution or format differs from the swap
/// chain, or when `PresentX` supplies overlay planes (e.g. Series S 1080p
/// scene + 4K UI overlay, #211).
struct Blitter {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Premultiplied-alpha blend (`ONE`, `INV_SRC_ALPHA`) for planes `1..`.
    overlay: ID3D12PipelineState,
    pso_format: DXGI_FORMAT,
    srv_heap: ID3D12DescriptorHeap,
    srv_step: u32,
}

impl Blitter {
    fn new(device: &ID3D12Device, rtv_format: DXGI_FORMAT) -> windows::core::Result<Self> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::PCSTR;

        const HLSL: &[u8] = b"\
Texture2D<float4> src : register(t0);
SamplerState lin : register(s0);
struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
VSOut vs(uint id : SV_VertexID) {
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv;
    o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return o;
}
float4 ps(VSOut i) : SV_Target {
    return src.Sample(lin, i.uv);
}
";
        let compile = |entry: &std::ffi::CStr, target: &std::ffi::CStr| -> windows::core::Result<ID3DBlob> {
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: compiling static HLSL source via `d3dcompiler_47.dll`.
            unsafe {
                D3DCompile(
                    HLSL.as_ptr().cast(),
                    HLSL.len(),
                    PCSTR::null(),
                    None,
                    None,
                    PCSTR(entry.as_ptr().cast()),
                    PCSTR(target.as_ptr().cast()),
                    0,
                    0,
                    &mut blob,
                    None,
                )?;
            }
            blob.ok_or_else(windows::core::Error::empty)
        };
        let vs = compile(c"vs", c"vs_5_0")?;
        let ps = compile(c"ps", c"ps_5_0")?;

        let range = D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let param = D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            MaxLOD: f32::MAX,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..Default::default()
        };
        let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: &param,
                    NumStaticSamplers: 1,
                    pStaticSamplers: &sampler,
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                },
            },
        };
        let mut rs_blob: Option<ID3DBlob> = None;
        // SAFETY: serializing valid root signature description and creating root signature + PSO + heap.
        unsafe {
            D3D12SerializeVersionedRootSignature(&rs_desc, &mut rs_blob, None)?;
            let rs_blob = rs_blob.ok_or_else(windows::core::Error::empty)?;
            let rs_bytes = std::slice::from_raw_parts(
                rs_blob.GetBufferPointer().cast::<u8>(),
                rs_blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, rs_bytes)?;

            let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.GetBufferPointer(),
                    BytecodeLength: vs.GetBufferSize(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.GetBufferPointer(),
                    BytecodeLength: ps.GetBufferSize(),
                },
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pso_desc.RTVFormats[0] = rtv_format;
            let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&pso_desc)?;

            pso_desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                ..Default::default()
            };
            let overlay: ID3D12PipelineState = device.CreateGraphicsPipelineState(&pso_desc)?;

            let srv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: BLITTER_PLANES,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let srv_step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            Ok(Self {
                root,
                pso,
                overlay,
                pso_format: rtv_format,
                srv_heap,
                srv_step,
            })
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 88 | 90..=93 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    color_space: DXGI_COLOR_SPACE_TYPE,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    blitter: Option<Blitter>,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect",
                making = crate::pipeline::making()
            );
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows one or more `PresentX` planes (plane 0 opaque, planes `1..`
    /// composited on top with premultiplied alpha, #211), leaving each plane
    /// in its incoming host state, as frame `frame` (saved if captures are
    /// on). `None` if nothing was shown; else, in diagnostics builds, whether
    /// the frame was black.
    pub(crate) fn present_planes(
        &self,
        planes: &[(&ID3D12Resource, u32)],
        frame: u64,
    ) -> Option<bool> {
        let &(texture, state) = planes.first()?;
        let planes = &planes[..planes.len().min(BLITTER_PLANES as usize)];
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (tex_width, tex_height) = (desc.Width as u32, desc.Height);
        // Size the initial swap chain to the largest plane (e.g. Series S 1080p
        // game plane + 4K UI overlay plane).
        let (max_width, max_height) = planes.iter().fold((tex_width, tex_height), |(mw, mh), &(t, _)| {
            // SAFETY: live texture.
            let d = unsafe { t.GetDesc() };
            (mw.max(d.Width as u32), mh.max(d.Height))
        });
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(max_width, max_height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        // Track the window's client size so presented frames scale cleanly to
        // resized windows or dynamic render resolutions.
        let (target_width, target_height) = {
            let mut rect = RECT::default();
            // SAFETY: live window handle.
            let ok = unsafe { GetClientRect(p.hwnd, &mut rect) }.is_ok();
            let cw = rect.right.saturating_sub(rect.left) as u32;
            let ch = rect.bottom.saturating_sub(rect.top) as u32;
            if ok && cw > 0 && ch > 0 {
                (cw, ch)
            } else {
                (max_width, max_height)
            }
        };
        let can_scale = desc.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
            && present_srv_format(desc.Format).is_some();
        let (width, height) = if can_scale {
            (target_width, target_height)
        } else {
            (tex_width, tex_height)
        };
        let hdr = hdr_info().supported;
        let desired_space = swap_color_space(format, hdr);
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        if p.color_space != desired_space {
            // SAFETY: live swap chain; ignore unsupported color spaces gracefully.
            if unsafe { p.swap_chain.SetColorSpace1(desired_space) }.is_ok() {
                p.color_space = desired_space;
                tracing::info!(
                    format = format.0,
                    color_space = desired_space.0,
                    "swap chain color space updated"
                );
            }
        }
        let need_blit = planes.len() > 1
            || (tex_width, tex_height) != (width, height)
            || needs_format_blit(desc.Format);
        let blit_format = present_blit_rtv_format(format, desc.Format);
        if need_blit
            && p.blitter
                .as_ref()
                .is_none_or(|b| b.pso_format != blit_format)
        {
            p.blitter = match Blitter::new(&self.device, blit_format) {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::warn!(error = %e, "present scaling blitter unavailable");
                    None
                }
            };
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and live source textures.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            let used_blit = if need_blit
                && let Some(blitter) = &p.blitter
                && present_srv_format(desc.Format).is_some()
                && desc.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
            {
                let back_rtv_desc =
                    (blit_format != format).then_some(D3D12_RENDER_TARGET_VIEW_DESC {
                        Format: blit_format,
                        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_RTV::default(),
                        },
                    });
                let rtv = self.rtv(&back, back_rtv_desc.as_ref());
                let cpu_start = blitter.srv_heap.GetCPUDescriptorHandleForHeapStart();
                let gpu_start = blitter.srv_heap.GetGPUDescriptorHandleForHeapStart();
                let mut active: Vec<(&ID3D12Resource, D3D12_RESOURCE_STATES, u32)> = Vec::new();
                for (slot, &(plane_tex, plane_state)) in planes.iter().enumerate() {
                    let pd = plane_tex.GetDesc();
                    if pd.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 != 0 {
                        continue;
                    }
                    let Some(srv_fmt) = present_srv_format(pd.Format) else {
                        continue;
                    };
                    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: srv_fmt,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MipLevels: 1,
                                ..Default::default()
                            },
                        },
                    };
                    let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: cpu_start.ptr + slot * blitter.srv_step as usize,
                    };
                    self.device
                        .CreateShaderResourceView(plane_tex, Some(&srv_desc), cpu);
                    active.push((
                        plane_tex,
                        D3D12_RESOURCE_STATES(plane_state as i32),
                        slot as u32,
                    ));
                }
                // Transition unique source planes + back buffer before compositing.
                let mut to_srv = Vec::with_capacity(active.len() + 1);
                let mut transitioned: Vec<*mut std::ffi::c_void> = Vec::new();
                for &(plane_tex, plane_state, _) in &active {
                    let raw = plane_tex.as_raw();
                    if !transitioned.contains(&raw) {
                        transitioned.push(raw);
                        if plane_state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                            to_srv.push(transition(
                                plane_tex,
                                plane_state,
                                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                            ));
                        }
                    }
                }
                to_srv.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                ));
                list.ResourceBarrier(&to_srv);

                let vp = D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                };
                let scissor = RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                };
                list.OMSetRenderTargets(1, Some(&rtv), false, None);
                list.RSSetViewports(&[vp]);
                list.RSSetScissorRects(&[scissor]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                list.SetGraphicsRootSignature(&blitter.root);
                list.SetDescriptorHeaps(&[Some(blitter.srv_heap.clone())]);
                for (i, &(_, _, slot)) in active.iter().enumerate() {
                    list.SetPipelineState(if i == 0 {
                        &blitter.pso
                    } else {
                        &blitter.overlay
                    });
                    let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: gpu_start.ptr + u64::from(slot) * u64::from(blitter.srv_step),
                    };
                    list.SetGraphicsRootDescriptorTable(0, gpu);
                    list.DrawInstanced(3, 1, 0, 0);
                }

                // Restore unique source planes and prepare back buffer.
                let mut restore = Vec::with_capacity(active.len() + 1);
                let mut restored: Vec<*mut std::ffi::c_void> = Vec::new();
                for &(plane_tex, plane_state, _) in &active {
                    let raw = plane_tex.as_raw();
                    if !restored.contains(&raw) {
                        restored.push(raw);
                        if plane_state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                            restore.push(transition(
                                plane_tex,
                                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                plane_state,
                            ));
                        }
                    }
                }
                restore.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_PRESENT,
                ));
                list.ResourceBarrier(&restore);
                true
            } else {
                false
            };
            #[cfg(feature = "diagnostics")]
            let capture;
            #[cfg(feature = "diagnostics")]
            let probe;
            if !used_blit {
                list.ResourceBarrier(&[
                    transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition(
                        &back,
                        D3D12_RESOURCE_STATE_PRESENT,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    ),
                ]);
                if (tex_width, tex_height) == (width, height) {
                    list.CopyResource(&back, texture);
                } else {
                    let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(texture),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: 0,
                        },
                    };
                    let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&back),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: 0,
                        },
                    };
                    let copy_box = D3D12_BOX {
                        left: 0,
                        top: 0,
                        front: 0,
                        right: tex_width.min(width),
                        bottom: tex_height.min(height),
                        back: 1,
                    };
                    list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, Some(&copy_box));
                }
                #[cfg(feature = "diagnostics")]
                {
                    capture = crate::capture::wanted(frame)
                        .then(|| crate::capture::read_back(&self.device, list, texture))
                        .flatten();
                    probe = crate::capture::probe(&self.device, list, texture);
                }
                list.ResourceBarrier(&[
                    transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                    transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_PRESENT,
                    ),
                ]);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    list.ResourceBarrier(&[transition(
                        texture,
                        state,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    capture = crate::capture::wanted(frame)
                        .then(|| crate::capture::read_back(&self.device, list, texture))
                        .flatten();
                    probe = crate::capture::probe(&self.device, list, texture);
                    list.ResourceBarrier(&[transition(
                        texture,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        state,
                    )]);
                }
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint, total)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint, total);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            COMPLETED_FRAMES.fetch_add(1, Ordering::Relaxed);
            drain_retired(4, 128);
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let hdr = hdr_info().supported;
            let color_space = swap_color_space(format, hdr);
            let _ = swap_chain.SetColorSpace1(color_space);
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(
                width,
                height,
                format = format.0,
                color_space = color_space.0,
                "presenting to a window"
            );
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                color_space,
                copy,
                hwnd,
                meter: Meter::new(),
                blitter: None,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

/// Saved windowed placement while our fallback presentation window is in
/// borderless fullscreen (F11 / Alt+Enter).
static FULLSCREEN_RESTORE: Mutex<Option<(i32, RECT)>> = Mutex::new(None);

fn toggle_fullscreen(hwnd: HWND) {
    let mut saved = FULLSCREEN_RESTORE.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: valid top-level window handle on its owning message thread.
    unsafe {
        if let Some((style, rect)) = saved.take() {
            SetWindowLongW(hwnd, GWL_STYLE, style);
            let _ = SetWindowPos(
                hwnd,
                Some(HWND_NOTOPMOST),
                rect.left,
                rect.top,
                (rect.right - rect.left).max(1),
                (rect.bottom - rect.top).max(1),
                SWP_FRAMECHANGED | SWP_NOOWNERZORDER,
            );
            return;
        }
        let style = GetWindowLongW(hwnd, GWL_STYLE);
        let mut rect = RECT::default();
        if GetWindowRect(hwnd, &mut rect).is_err() {
            return;
        }
        let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        let mut info = MONITORINFO {
            cbSize: size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if !GetMonitorInfoW(monitor, &mut info).as_bool() {
            return;
        }
        let mr = info.rcMonitor;
        *saved = Some((style, rect));
        let borderless = style & !(WS_OVERLAPPEDWINDOW.0 as i32);
        SetWindowLongW(hwnd, GWL_STYLE, borderless);
        let _ = SetWindowPos(
            hwnd,
            Some(HWND_TOP),
            mr.left,
            mr.top,
            (mr.right - mr.left).max(1),
            (mr.bottom - mr.top).max(1),
            SWP_FRAMECHANGED | SWP_NOOWNERZORDER,
        );
    }
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    const VK_F11: usize = 0x7A;
    const VK_RETURN: usize = 0x0D;
    let repeat = (lparam.0 & (1 << 30)) != 0;
    let alt = (lparam.0 & (1 << 29)) != 0;
    if !repeat
        && ((message == WM_KEYDOWN && wparam.0 == VK_F11)
            || (message == WM_SYSKEYDOWN && wparam.0 == VK_RETURN && alt))
    {
        toggle_fullscreen(hwnd);
        return LRESULT(0);
    }
    if message == WM_CLOSE {
        // SAFETY: valid window handle on its owning message thread.
        unsafe {
            let _ = DestroyWindow(hwnd);
            PostQuitMessage(0);
        }
        return LRESULT(0);
    }
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn env_flag_respects_falsy_values() {
        use std::ffi::OsStr;
        for off in [None, Some(""), Some("   "), Some("0"), Some("false"), Some("FALSE"), Some("no"), Some("Off")] {
            assert!(!super::parse_env_flag(off.map(OsStr::new)), "expected off for {off:?}");
        }
        for on in ["1", "true", "yes", "on", "anything"] {
            assert!(super::parse_env_flag(Some(OsStr::new(on))), "expected on for {on:?}");
        }
    }

    #[test]
    fn memory_wait_comparisons_and_producer_wakeup() {
        assert!(super::memory_wait_satisfied(1, 2, 2)); // LESS
        assert!(!super::memory_wait_satisfied(2, 2, 2));
        assert!(super::memory_wait_satisfied(5, 3, 5)); // EQUAL
        assert!(!super::memory_wait_satisfied(4, 3, 5));
        assert!(super::memory_wait_satisfied(5, 4, 5)); // LESS_EQUAL
        assert!(super::memory_wait_satisfied(6, 5, 5)); // GREATER
        assert!(super::memory_wait_satisfied(4, 6, 5)); // NOT_EQUAL
        assert!(super::memory_wait_satisfied(5, 7, 5)); // GREATER_EQUAL
        assert!(!super::memory_wait_satisfied(4, 7, 5));
        assert!(super::memory_wait_satisfied(0, 8, 99)); // ALWAYS

        let cell = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let producer = cell.clone();
        let addr = std::sync::Arc::as_ptr(&cell) as usize;
        let t = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            producer.store(42, std::sync::atomic::Ordering::Release);
        });
        super::wait_for_memory(&[super::MemoryWait {
            address: addr,
            comparison: 7, // GREATER_EQUAL
            reference: 42,
        }]);
        t.join().unwrap();
        assert_eq!(cell.load(std::sync::atomic::Ordering::Acquire), 42);
    }

    #[test]
    fn region_protection_checks() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_FREE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
            PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        };
        assert!(super::is_readable_region(MEM_COMMIT, PAGE_READONLY));
        assert!(super::is_readable_region(MEM_COMMIT, PAGE_READWRITE));
        assert!(super::is_readable_region(MEM_COMMIT, PAGE_EXECUTE_READ));
        assert!(!super::is_readable_region(MEM_RESERVE, PAGE_READWRITE));
        assert!(!super::is_readable_region(MEM_FREE, PAGE_READWRITE));
        assert!(!super::is_readable_region(MEM_COMMIT, PAGE_NOACCESS));
        assert!(!super::is_readable_region(
            MEM_COMMIT,
            PAGE_READWRITE | PAGE_GUARD
        ));

        assert!(!super::is_writable_region(MEM_COMMIT, PAGE_READONLY));
        assert!(!super::is_writable_region(MEM_COMMIT, PAGE_EXECUTE_READ));
        assert!(super::is_writable_region(MEM_COMMIT, PAGE_READWRITE));
        assert!(super::is_writable_region(MEM_COMMIT, PAGE_EXECUTE_READWRITE));
        assert!(!super::is_writable_region(MEM_RESERVE, PAGE_READWRITE));
        assert!(!super::is_writable_region(
            MEM_COMMIT,
            PAGE_READWRITE | PAGE_GUARD
        ));
    }

    #[test]
    fn copy_mapped_and_write_mapped_skip_uncommitted_gaps() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const CHUNK: usize = 64 << 10;
        const TOTAL: usize = 3 * CHUNK;
        // Reserve 192 KiB and commit only the first and last 64 KiB, leaving an
        // uncommitted 64 KiB hole in the middle.
        // SAFETY: balanced VirtualAlloc / VirtualFree of our private test block.
        let base = unsafe { VirtualAlloc(std::ptr::null(), TOTAL, MEM_RESERVE, PAGE_READWRITE) };
        assert!(!base.is_null());
        // SAFETY: committing and initializing the first and third chunks of our reserved block.
        unsafe {
            let a = VirtualAlloc(base, CHUNK, MEM_COMMIT, PAGE_READWRITE);
            let b = VirtualAlloc(base.cast::<u8>().add(2 * CHUNK).cast(), CHUNK, MEM_COMMIT, PAGE_READWRITE);
            assert!(!a.is_null() && !b.is_null());
            std::slice::from_raw_parts_mut(base.cast::<u8>(), CHUNK).fill(0xA5);
            std::slice::from_raw_parts_mut(base.cast::<u8>().add(2 * CHUNK), CHUNK).fill(0x5A);
        }

        // `copy_mapped` copies both committed regions and zeroes the hole.
        let mut dst = vec![0xFFu8; TOTAL];
        super::copy_mapped(dst.as_mut_ptr(), base.cast::<u8>(), TOTAL, false);
        assert!(dst[..CHUNK].iter().all(|&b| b == 0xA5));
        assert!(dst[CHUNK..2 * CHUNK].iter().all(|&b| b == 0));
        assert!(dst[2 * CHUNK..].iter().all(|&b| b == 0x5A));

        // `write_mapped` updates both committed regions and ignores the hole.
        let mut src = vec![0x33u8; TOTAL];
        src[2 * CHUNK..].fill(0xCC);
        super::write_mapped(base.cast::<u8>(), src.as_ptr(), TOTAL);
        // SAFETY: reading back the two committed chunks before freeing the reservation.
        unsafe {
            assert!(std::slice::from_raw_parts(base.cast::<u8>(), CHUNK)
                .iter()
                .all(|&b| b == 0x33));
            assert!(
                std::slice::from_raw_parts(base.cast::<u8>().add(2 * CHUNK), CHUNK)
                    .iter()
                    .all(|&b| b == 0xCC)
            );
            VirtualFree(base, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn gpu_writes_cover_handles_single_merged_and_gapped_ranges() {
        assert!(super::gpu_writes_cover(&[(100, 300)], 100, 300));
        assert!(super::gpu_writes_cover(&[(50, 400)], 100, 300));
        assert!(super::gpu_writes_cover(&[(200, 300), (100, 200)], 100, 300));
        assert!(super::gpu_writes_cover(
            &[(100, 180), (160, 240), (240, 320)],
            100,
            300
        ));
        assert!(!super::gpu_writes_cover(&[(100, 199), (200, 300)], 100, 300));
        assert!(!super::gpu_writes_cover(&[(101, 300)], 100, 300));
        assert!(!super::gpu_writes_cover(&[], 100, 300));
    }

    #[test]
    fn present_blit_handles_b8g8r8x8_and_srgb_roundtrip() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
            DXGI_FORMAT_B8G8R8X8_TYPELESS, DXGI_FORMAT_B8G8R8X8_UNORM,
            DXGI_FORMAT_B8G8R8X8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_TYPELESS,
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        };
        // B8G8R8X8* cannot be copied directly into a B8G8R8A8_UNORM swap buffer.
        assert!(super::needs_format_blit(DXGI_FORMAT_B8G8R8X8_UNORM));
        assert!(super::needs_format_blit(DXGI_FORMAT_B8G8R8X8_TYPELESS));
        assert!(super::needs_format_blit(DXGI_FORMAT_B8G8R8X8_UNORM_SRGB));
        assert!(!super::needs_format_blit(DXGI_FORMAT_B8G8R8A8_UNORM));
        assert!(!super::needs_format_blit(DXGI_FORMAT_R8G8B8A8_UNORM));

        // Concrete sRGB sources decode on SRV sample and must re-encode via an sRGB RTV.
        assert_eq!(
            super::present_blit_rtv_format(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
            ),
            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        );
        assert_eq!(
            super::present_blit_rtv_format(
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
            ),
            DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        );
        assert_eq!(
            super::present_blit_rtv_format(
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_FORMAT_B8G8R8X8_UNORM_SRGB
            ),
            DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        );
        // UNORM and TYPELESS sources pass through untouched via the UNORM RTV.
        assert_eq!(
            super::present_blit_rtv_format(DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM),
            DXGI_FORMAT_B8G8R8A8_UNORM
        );
        assert_eq!(
            super::present_blit_rtv_format(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_TYPELESS
            ),
            DXGI_FORMAT_R8G8B8A8_UNORM
        );
    }
}
