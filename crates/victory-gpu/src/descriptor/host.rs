//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;
use victory_shader::ResourceType;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from,
    /// or one of the `NULL_*` sentinels when left null with a specific kind.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// D3D12's architectural cap for shader-visible sampler heaps across all
/// hardware tiers (`D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE`, #100).
const PORTABLE_SAMPLERS: u32 = 2_048;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run), reclaimed under heap
/// pressure, or released.
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): enough headroom for big world tables without
/// wrapping onto lists still in flight (#747).
const MIN_DYNAMIC: u32 = 1 << 18;

/// Sentinel `made` keys for null slots of each descriptor kind (real view
/// keys pack `generation << 32 | index` with small non-zero generations).
const NULL_SRV: u64 = 0;
const NULL_BUFFER_SRV: u64 = u64::MAX - 3;
const NULL_TEXTURE_UAV: u64 = u64::MAX - 2;
const NULL_BUFFER_UAV: u64 = u64::MAX - 1;
const NULL_CBV: u64 = u64::MAX;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) if count > PORTABLE_SAMPLERS => match make(PORTABLE_SAMPLERS) {
                    Ok(heap) => {
                        tracing::warn!(
                            error = %e,
                            count,
                            cap = PORTABLE_SAMPLERS,
                            "host sampler heap capped at {PORTABLE_SAMPLERS}"
                        );
                        (heap, PORTABLE_SAMPLERS, 0)
                    }
                    Err(e2) => {
                        tracing::warn!(error = %e2, count, sampler, "no host descriptor heap");
                        return None;
                    }
                },
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) => {
                    // Multi-tiered fallback for GPUs/drivers with tighter
                    // shader-visible heap caps (#902).
                    let mut fallback = None;
                    for cap in [PORTABLE_VIEWS, 500_000, 250_000, MIN_DYNAMIC] {
                        if let Ok(heap) = make(cap) {
                            let static_count = cap.saturating_sub(MIN_DYNAMIC);
                            let extra = cap - static_count;
                            tracing::warn!(
                                error = %e,
                                count,
                                cap,
                                static_count,
                                extra,
                                "host view heap capped at {cap}"
                            );
                            fallback = Some((heap, static_count, extra));
                            break;
                        }
                    }
                    match fallback {
                        Some(pair) => pair,
                        None => {
                            tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                            return None;
                        }
                    }
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, extra, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![NULL_SRV; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start — extending up to
        // `next + n - 1` so free runs straddling `next` are not missed (#871).
        let wrap_end = (*next as usize)
            .saturating_add(n - 1)
            .min(self.dynamic as usize);
        let starts = (*next as usize..self.dynamic as usize).chain(0..wrap_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let first = found?;
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// Emergency fallback when the dynamic descriptor region is exhausted
    /// mid-frame: clears the dynamic allocation bitmap and hands out slots from
    /// the start of the dynamic region so draws/dispatches continue instead of
    /// silently dropping (#902).
    pub(crate) fn recycle_dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        taken.fill(false);
        taken[..n as usize].fill(true);
        *next = n % self.dynamic;
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                dynamic = self.dynamic,
                needed = n,
                "dynamic descriptor heap exhausted; recycling slots"
            );
        });
        Some(self.count)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = at.saturating_add(n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// The GPU descriptor handle of slot `index` in this shader-visible heap.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: live shader-visible descriptor heap.
        let mut handle = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        handle.ptr += u64::from(index) * u64::from(self.step);
        handle
    }

    /// Writes a texture UAV into slot `index` and keeps `texture` alive.
    pub(crate) fn uav_at(
        &self,
        index: u32,
        texture: &ID3D12Resource,
        desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    ) {
        let Some(host) = host() else { return };
        // SAFETY: a live texture and a slot of our heap.
        unsafe {
            host.device.CreateUnorderedAccessView(
                texture,
                None::<&ID3D12Resource>,
                desc.map(std::ptr::from_ref),
                self.cpu(index),
            );
        }
        // Invalidate the `made` cache if this is a title slot so a future
        // draw mirroring another view here does not skip recreating it.
        if (index as usize) < self.count as usize {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(slot) = made.get_mut(index as usize) {
                *slot = NULL_TEXTURE_UAV;
            }
        }
        self.keep(index, Some(texture.clone()));
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Makes slot `index` a null descriptor matching the shader's expected
    /// resource kind and descriptor width (`4` dwords = buffer, `8` = image),
    /// so writing to an unbound UAV slot hits a hardware null UAV instead of
    /// a null SRV (which page-faults on RDNA3).
    pub(crate) fn make_null_for(&self, index: u32, kind: ResourceType, dwords: u8) {
        let at = self.cpu(index);
        if self.sampler || kind == ResourceType::Sampler {
            self.null_sampler(index);
        } else {
            match (kind, dwords == 4) {
                (ResourceType::Uav, true) => null_buffer_uav(at),
                (ResourceType::Uav, false) => null_texture_uav(4, at),
                (ResourceType::Cbv, _) => null_cbv(at),
                (ResourceType::Srv | ResourceType::Other(_), true) => null_buffer_view(at),
                (ResourceType::Srv | ResourceType::Other(_), false) => null_view(at),
                (ResourceType::Sampler, _) => unreachable!(),
            }
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    /// Any previously held host resource is retired across frames so in-flight
    /// multi-list or indirect draws reading this descriptor slot cannot fault.
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let old = {
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            kept.get_mut(index as usize)
                .and_then(|slot| std::mem::replace(slot, resource))
        };
        if let Some(old) = old {
            crate::host::retire_resource(old);
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        self.mirror_for(index, handle, for_draw, ResourceType::Srv, 8)
    }

    /// Like [`HostHeap::mirror`], but when the slot is unbound (`None`) makes
    /// a null descriptor of the shader's expected `kind` and descriptor width
    /// (`dwords == 4` for buffers, `8` for images).
    pub(crate) fn mirror_for(
        &self,
        index: u32,
        handle: u64,
        for_draw: bool,
        kind: ResourceType,
        dwords: u8,
    ) -> Option<View> {
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far. Still return the decoded view so callers
            // can mirror it into a dynamic slot (#902).
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return view;
        }
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): ensure the slot carries a matching null view.
            let want = if self.sampler || kind == ResourceType::Sampler {
                NULL_SRV
            } else {
                match (kind, dwords == 4) {
                    (ResourceType::Uav, true) => NULL_BUFFER_UAV,
                    (ResourceType::Uav, false) => NULL_TEXTURE_UAV,
                    (ResourceType::Cbv, _) => NULL_CBV,
                    (ResourceType::Srv | ResourceType::Other(_), true) => NULL_BUFFER_SRV,
                    (ResourceType::Srv | ResourceType::Other(_), false) => NULL_SRV,
                    (ResourceType::Sampler, _) => unreachable!(),
                }
            };
            if made[index as usize] != want {
                made[index as usize] = want;
                self.make_null_for(index, kind, dwords);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.is_buffer();
            made[index as usize] = if later { NULL_SRV } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.flatten());
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        if key == NULL_SRV || key >= NULL_BUFFER_SRV {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
/// Builds the host UAV description for a resource-backed texture, returning
/// `None` if the requested format does not fit `texture`. When the title
/// passed a null UAV desc for a texture whose host copy is typeless, synthesizes
/// a matching default UAV desc from the title's own format and texture dimension.
pub(crate) fn texture_uav_desc(
    resource: crate::object::This,
    texture: &ID3D12Resource,
    desc: Option<&[u32; 10]>,
) -> Option<Option<D3D12_UNORDERED_ACCESS_VIEW_DESC>> {
    let mut desc = match desc {
        Some(words) => {
            let mut d = uav_desc(words);
            if d.Format.0 == 0
                && let Some(vf) = crate::resource::view_format(resource)
            {
                d.Format = DXGI_FORMAT(crate::resource::pc_format(vf) as i32);
            }
            Some(d)
        }
        None => {
            // SAFETY: a live texture.
            let d = unsafe { texture.GetDesc() };
            let resolved = crate::resource::view_format(resource)
                .map(crate::resource::pc_format)
                .or_else(|| crate::resource::resolve_typeless_view(d.Format.0 as u32, true));
            resolved.and_then(|fmt| {
                let format = DXGI_FORMAT(fmt as i32);
                match d.Dimension {
                    D3D12_RESOURCE_DIMENSION_TEXTURE1D if d.DepthOrArraySize > 1 => {
                        Some(D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1DARRAY,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Texture1DArray: D3D12_TEX1D_ARRAY_UAV {
                                    MipSlice: 0,
                                    FirstArraySlice: 0,
                                    ArraySize: u32::from(d.DepthOrArraySize),
                                },
                            },
                        })
                    }
                    D3D12_RESOURCE_DIMENSION_TEXTURE1D => Some(D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE1D,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Texture1D: D3D12_TEX1D_UAV { MipSlice: 0 },
                        },
                    }),
                    D3D12_RESOURCE_DIMENSION_TEXTURE2D if d.DepthOrArraySize > 1 => {
                        Some(D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                                    MipSlice: 0,
                                    FirstArraySlice: 0,
                                    ArraySize: u32::from(d.DepthOrArraySize),
                                    PlaneSlice: 0,
                                },
                            },
                        })
                    }
                    D3D12_RESOURCE_DIMENSION_TEXTURE2D => Some(D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_UAV {
                                MipSlice: 0,
                                PlaneSlice: 0,
                            },
                        },
                    }),
                    D3D12_RESOURCE_DIMENSION_TEXTURE3D => Some(D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Texture3D: D3D12_TEX3D_UAV {
                                MipSlice: 0,
                                FirstWSlice: 0,
                                WSize: u32::MAX,
                            },
                        },
                    }),
                    _ => None,
                }
            })
        }
    };
    if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
        return None;
    }
    Some(desc)
}

fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        &View::Constant(address, size) => return Some((address, u64::from(size))),
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    let total = shadow.len / element;
    let first = buffer.FirstElement.checked_add(offset / element)?;
    if first >= total || buffer.NumElements == 0 {
        once!(
            first,
            total,
            "buffer shader view starts past the end of its host copy"
        );
        return None;
    }
    let avail = (total - first).min(u64::from(u32::MAX)) as u32;
    buffer.FirstElement = first;
    if buffer.NumElements > avail {
        buffer.NumElements = avail;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(offset, element, "unordered buffer view not on an element of its host copy");
        return None;
    }
    let total = shadow.len / element;
    let first = buffer.FirstElement.checked_add(offset / element)?;
    if first >= total || buffer.NumElements == 0 {
        once!(
            first,
            total,
            "buffer unordered view starts past the end of its host copy"
        );
        return None;
    }
    let avail = (total - first).min(u64::from(u32::MAX)) as u32;
    buffer.FirstElement = first;
    if buffer.NumElements > avail {
        buffer.NumElements = avail;
    }
    buffer.CounterOffsetInBytes = 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family, is_typeless, resolve_typeless_view};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // DSV-only formats (D32_FLOAT_S8X24_UINT, D32_FLOAT, D24_UNORM_S8_UINT,
    // D16_UNORM) are never valid SRV/UAV formats on D3D12 (#882).
    if matches!(view, 20 | 40 | 45 | 55) {
        if let Some(f) = format
            && let Some(resolved) = resolve_typeless_view(own, unordered)
        {
            f.0 = resolved as i32;
            return true;
        }
        return false;
    }
    if view == 0 {
        if !typeless {
            return true;
        }
        if let Some(f) = format
            && let Some(resolved) = resolve_typeless_view(own, unordered)
        {
            f.0 = resolved as i32;
            return true;
        }
        return false;
    }
    // Packed 16-bit textures (`B5G6R5_UNORM`, `B5G5R5A1_UNORM`, `B4G4R4A4_UNORM`)
    // that were widened to `R8G8B8A8_UNORM` (`28`) for RTV/UAV compatibility (#797).
    if own == 28 && matches!(view, 85 | 86 | 115) {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if typeless && is_typeless(view) {
        // Viewing a typeless texture with a typeless view format triggers
        // D3D12 device removal; resolve to a concrete format (#876).
        let Some(resolved) = resolve_typeless_view(own, unordered) else {
            return false;
        };
        let Some(f) = format else {
            return false;
        };
        f.0 = resolved as i32;
        return true;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Builds a default SRV description for a typeless host texture when the title
/// supplied a null `D3D12_SHADER_RESOURCE_VIEW_DESC` (#876, #882).
fn default_srv_desc(
    texture: &ID3D12Resource,
    format: DXGI_FORMAT,
) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    // SAFETY: a live texture.
    let d = unsafe { texture.GetDesc() };
    let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    match d.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D if d.DepthOrArraySize > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                FirstArraySlice: 0,
                ArraySize: u32::from(d.DepthOrArraySize),
                ResourceMinLODClamp: 0.0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            desc.Anonymous.Texture1D = D3D12_TEX1D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                ResourceMinLODClamp: 0.0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D if d.SampleDesc.Count > 1 && d.DepthOrArraySize > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
            desc.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_SRV {
                FirstArraySlice: 0,
                ArraySize: u32::from(d.DepthOrArraySize),
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D if d.SampleDesc.Count > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
            desc.Anonymous.Texture2DMS = D3D12_TEX2DMS_SRV::default();
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D if d.DepthOrArraySize > 1 => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                FirstArraySlice: 0,
                ArraySize: u32::from(d.DepthOrArraySize),
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            desc.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            };
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                ResourceMinLODClamp: 0.0,
            };
        }
        _ => return None,
    }
    Some(desc)
}

/// Converts a depth/stencil SRV format to its readable host DXGI format and
/// selects the matching plane (`0` for depth, `1` for stencil) on 2D and
/// 2D-array views of typeless depth/stencil host textures.
fn fix_depth_stencil_srv(own: u32, desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC) {
    let raw = desc.Format.0 as u32;
    let mapped = match (own, raw) {
        // D32_FLOAT_S8X24 family (typeless 19): DSV 20 -> depth SRV 21; stencil SRV 22.
        (19, 0 | 19 | 20) => 21,
        (19, 22) => 22,
        // R32 / D32 family (typeless 39): DSV 40 -> R32_FLOAT 41.
        (39, 0 | 39 | 40) => 41,
        // D24_UNORM_S8_UINT family (typeless 44): DSV 45 -> depth SRV 46; stencil SRV 47.
        (44, 0 | 44 | 45) => 46,
        (44, 47) => 47,
        // R16 / D16 family (typeless 53): DSV 55 -> R16_UNORM 56.
        (53, 0 | 53 | 55) => 56,
        (_, 20) => 21,
        (_, 40) => 41,
        (_, 45) => 46,
        (_, 55) => 56,
        _ => raw,
    };
    desc.Format = DXGI_FORMAT(mapped as i32);
    let plane = match mapped {
        22 | 47 => Some(1u32),
        21 | 46 if matches!(own, 19 | 44) => Some(0u32),
        _ => None,
    };
    if let Some(plane) = plane {
        match desc.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                desc.Anonymous.Texture2D.PlaneSlice = plane;
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                desc.Anonymous.Texture2DArray.PlaneSlice = plane;
            }
            _ => {}
        }
    }
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let padded = size.next_multiple_of(256);
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            if shadow.left(address) < u64::from(padded) {
                once!(
                    address = format_args!("{address:#x}"),
                    size,
                    left = shadow.left(address),
                    "constant buffer padded size reaches past its host copy"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: padded,
            };
            // SAFETY: a slot of our heap; the 256-byte-rounded range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // SAFETY: a live texture.
                let own = unsafe { texture.GetDesc() }.Format.0 as u32;
                let mut desc = match desc.as_ref() {
                    Some(words) => {
                        let mut d = srv_desc(words);
                        if d.Format.0 == 0
                            && crate::resource::is_typeless(own)
                            && let Some(vf) = crate::resource::view_format(resource)
                        {
                            d.Format = DXGI_FORMAT(crate::resource::pc_format(vf) as i32);
                        }
                        Some(d)
                    }
                    None if crate::resource::is_typeless(own) => {
                        let resolved = crate::resource::view_format(resource)
                            .map(crate::resource::pc_format)
                            .or_else(|| crate::resource::resolve_typeless_view(own, false))?;
                        default_srv_desc(texture, DXGI_FORMAT(resolved as i32))
                    }
                    None => None,
                };
                if let Some(d) = desc.as_mut() {
                    fix_depth_stencil_srv(own, d);
                }
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let desc = texture_uav_desc(resource, texture, desc.as_ref())?;
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader or unordered-access view of a texture the title placed in memory
/// itself, as a view of the host resource made for that memory (on first use,
/// #921).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource, *unordered)?;
    let words: Option<[u32; 10]> = desc[..10].try_into().ok();
    Some(if *unordered {
        View::Unordered(texture as usize, 0, words)
    } else {
        View::Shader(texture as usize, words)
    })
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (matching the buffer or texture dimension)
/// or a shader view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => {
            let dim = if desc[1] != 0 {
                Some(desc[1])
            } else {
                Some(match resource[0] {
                    2 => 2, // TEXTURE1D
                    4 => 8, // TEXTURE3D
                    _ => 4, // TEXTURE2D
                })
            };
            (resource[0] == 1, dim)
        }
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => return null_cbv(at),
        _ => return null_view(at),
    };
    if buffer {
        null_buffer_uav(at);
    } else {
        null_texture_uav(dimension.unwrap_or(4), at);
    }
}

/// A constant buffer view that reads zeros.
fn null_cbv(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
    // SAFETY: a slot of our heap; a zero location makes a null view.
    unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
}

/// A raw buffer UAV that drops writes and reads zeros.
fn null_buffer_uav(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 1,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A texture UAV of dimension `dim` (1D, 1DArray, 2D, 2DArray, or 3D; defaults
/// to 2D) that drops writes and reads zeros.
fn null_texture_uav(dim: u32, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = null_uav_desc(Some(dim));
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// A contiguous free run that straddles the scan cursor `next` must still
    /// be found on wraparound instead of spuriously returning `None` (#871).
    #[test]
    fn dynamic_slots_find_runs_straddling_next_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let d = heap.dynamic;
        // Take the whole heap as three blocks: [0..10], [10..20], [20..d].
        let a = heap.dynamic(10).unwrap();
        let b = heap.dynamic(10).unwrap();
        let c = heap.dynamic(d - 20).unwrap();
        assert_eq!((a, b, c), (4, 14, 24));
        // Free [0..10] and re-take [0..10] so `next` moves to 10; then free
        // [6..10] and [10..16] creating a 10-slot free run [6..16] straddling
        // `next == 10`, with nothing else free in the heap.
        heap.release(a, 10);
        assert_eq!(heap.dynamic(10), Some(4));
        // `next` is now 10.
        heap.release(4 + 6, 4); // free [6..10)
        heap.release(4 + 10, 6); // free [10..16)
        assert_eq!(
            heap.dynamic(10),
            Some(4 + 6),
            "10-slot run straddling next=10 must be allocated"
        );
    }

    #[test]
    fn depth_stencil_srv_maps_format_and_plane() {
        let mut depth = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(20), // D32_FLOAT_S8X24_UINT
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        depth.Anonymous.Texture2D.PlaneSlice = 1;
        fix_depth_stencil_srv(19, &mut depth);
        assert_eq!(depth.Format.0, 21); // R32_FLOAT_X8X24_TYPELESS
        // SAFETY: `ViewDimension` is `TEXTURE2D`.
        assert_eq!(unsafe { depth.Anonymous.Texture2D.PlaneSlice }, 0);

        let mut stencil = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(47), // X24_TYPELESS_G8_UINT
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            ..Default::default()
        };
        fix_depth_stencil_srv(44, &mut stencil);
        assert_eq!(stencil.Format.0, 47);
        // SAFETY: `ViewDimension` is `TEXTURE2DARRAY`.
        assert_eq!(unsafe { stencil.Anonymous.Texture2DArray.PlaneSlice }, 1);
    }

    #[test]
    fn typeless_resolution_covers_color_depth_and_uav_rules() {
        use crate::resource::{is_typeless, resolve_typeless_view};
        assert!(is_typeless(27));
        assert!(is_typeless(90));
        assert!(is_typeless(19));
        assert!(!is_typeless(28));
        // Color typeless resolves to concrete SRV and UAV formats.
        assert_eq!(resolve_typeless_view(27, false), Some(28));
        assert_eq!(resolve_typeless_view(27, true), Some(28));
        assert_eq!(resolve_typeless_view(90, false), Some(87));
        assert_eq!(resolve_typeless_view(90, true), Some(87));
        assert_eq!(resolve_typeless_view(92, false), Some(88));
        assert_eq!(resolve_typeless_view(92, true), None);
        // Depth-stencil typeless resolves to readable depth SRVs, never DSV or UAV.
        assert_eq!(resolve_typeless_view(19, false), Some(21));
        assert_eq!(resolve_typeless_view(19, true), None);
        assert_eq!(resolve_typeless_view(44, false), Some(46));
        assert_eq!(resolve_typeless_view(44, true), None);
        // R32 / R16 typeless resolve to concrete formats, never DSV 40 / 55.
        assert_eq!(resolve_typeless_view(39, false), Some(41));
        assert_eq!(resolve_typeless_view(53, false), Some(56));
    }

    #[test]
    fn buffer_views_of_no_elements_are_null() {
        let Some(host) = host() else { return };
        let Some(heap) = HostHeap::new(false, 8) else {
            return;
        };
        let mut backing = vec![0u8; 4096];
        let addr = backing.as_mut_ptr() as u64;
        // Zero-element SRV and UAV must return None without provoking D3D12
        // device removal (#912).
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 0,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 0,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        assert!(buffer_srv(addr, backing.len() as u64, srv, heap.cpu(0)).is_none());
        assert!(buffer_uav(addr, backing.len() as u64, uav, heap.cpu(1)).is_none());
        // SAFETY: live device.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn recycle_dynamic_recovers_exhausted_heap() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let d = heap.dynamic;
        assert_eq!(heap.dynamic(d), Some(4));
        assert_eq!(heap.dynamic(1), None);
        assert_eq!(heap.recycle_dynamic(8), Some(4));
        assert_eq!(heap.dynamic(4), Some(12));
    }
}
