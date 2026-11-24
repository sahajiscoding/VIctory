//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// `D3D12_DSV_FLAGS` (`0x1` read-only depth, `0x2` read-only stencil) of
    /// the bound depth view.
    depth_flags: u32,
    /// Whether the bound depth view's format includes a stencil plane.
    depth_has_stencil: bool,
    /// Set during [`Self::bind`] if the current graphics draw also binds
    /// [`Self::depth`] as a shader resource view.
    depth_sampled: bool,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

#[derive(Clone, Copy)]
enum RootViewKind {
    Cbv,
    Srv,
    Uav,
}

const BUFFER_READ: u32 = D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32;
const BUFFER_UAV: u32 = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
/// Maximum reach refreshed for a root SRV binding (32 MiB so mesh/geometry
/// shaders indexing megabytes past a root SRV do not fault, #951).
const ROOT_SRV_REACH: u64 = 32 << 20;

fn root_view(
    list: &HostList,
    address: u64,
    kind: RootViewKind,
    uav_buffers: &mut Vec<usize>,
) -> Option<u64> {
    let shadow = match kind {
        RootViewKind::Srv => crate::shadow::find_wide(address, ROOT_SRV_REACH)?,
        RootViewKind::Cbv | RootViewKind::Uav => crate::shadow::find(address, 1)?,
    };
    let len = shadow.left(address).min(match kind {
        RootViewKind::Cbv => 1 << 16,
        RootViewKind::Srv => ROOT_SRV_REACH,
        RootViewKind::Uav => 1 << 20,
    });
    let gpu = shadow.gpu(address);
    if !matches!(kind, RootViewKind::Cbv) || gpu.is_multiple_of(256) {
        shadow.refresh(list, address, len);
        let key = Arc::as_ptr(&shadow) as usize;
        if matches!(kind, RootViewKind::Uav) {
            shadow.gpu_writes(list, address, len);
            if !uav_buffers.contains(&key) {
                uav_buffers.push(key);
            }
            list.ensure_buffer(&shadow, BUFFER_UAV);
        } else if !uav_buffers.contains(&key) {
            list.ensure_buffer(&shadow, BUFFER_READ);
        }
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// Resets all command-list state to defaults while preserving any dynamic
    /// table cache already allocated on this open host list (`ClearState`, #848).
    pub(crate) fn clear(&mut self, list: &HostList, pipeline: This) {
        let fresh_tables = std::mem::take(&mut self.fresh_tables);
        *self = State::default();
        self.fresh_tables = fresh_tables;
        self.pipeline = pipeline as usize;
        // SAFETY: an open list; passing `None` resets host state (PSO, root
        // signatures, descriptor heaps, targets, viewports, scissors, topology,
        // stencil ref, blend factor, predication) to defaults. Our `host_*`
        // caches are now zeroed so the next draw rebinds `self.pipeline` and
        // any newly set heaps/roots.
        unsafe { list.list.ClearState(None::<&ID3D12PipelineState>) };
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(texture, record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_flags = 0;
        self.depth_has_stencil = false;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            let desc = record.depth_desc();
            self.depth_flags = desc.as_ref().map_or(0, |d| d.Flags.0 as u32);
            let fmt = desc
                .as_ref()
                .map(|d| d.Format.0 as u32)
                .filter(|&f| f != 0)
                .unwrap_or_else(|| {
                    // SAFETY: a live texture.
                    unsafe { texture.GetDesc() }.Format.0 as u32
                });
            self.depth_has_stencil = matches!(fmt, 19 | 20 | 44 | 45);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, desc.as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The host byte length of a bound root view for `slot` (0 if unset or
    /// outside shadow memory), matching [`root_view`]'s 64 KiB CBV / 1 MiB
    /// SRV/UAV cap so shaders never read or write past the host mapping.
    fn root_view_bytes(
        &self,
        stage: &crate::pipeline::host::Stage,
        compute: bool,
        slot: u8,
    ) -> u32 {
        let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
            return 0;
        };
        let is_cbv = stage.root_cbvs.iter().any(|&(s, _)| s == slot);
        let is_srv = stage.root_srvs.iter().any(|&(s, _)| s == slot);
        let Some(shadow) = (if is_srv {
            crate::shadow::find_wide(*address, ROOT_SRV_REACH)
        } else {
            crate::shadow::find(*address, 1)
        }) else {
            return 0;
        };
        let cap = if is_cbv {
            1u64 << 16
        } else if is_srv {
            ROOT_SRV_REACH
        } else {
            1u64 << 20
        };
        shadow.left(*address).min(cap) as u32
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value, or a root
    /// view's bound byte size.
    fn user_value(
        &self,
        stage: &crate::pipeline::host::Stage,
        compute: bool,
        slot: u8,
        root: RootType,
        dword: u32,
    ) -> u32 {
        if root == RootType::Descriptor {
            return self.root_view_bytes(stage, compute, slot);
        }
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        let Some(uav_buffers) = self.bind(list, p, false) else {
            return false;
        };
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            if !uav_buffers.contains(&(Arc::as_ptr(&shadow) as usize)) {
                list.ensure_buffer(&shadow, BUFFER_READ);
            }
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            if !uav_buffers.contains(&(Arc::as_ptr(&shadow) as usize)) {
                list.ensure_buffer(&shadow, BUFFER_READ);
            }
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            let state = depth_target_state(
                self.depth_flags,
                self.depth_has_stencil,
                self.depth_sampled,
            );
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// Ensures the shader-visible host view heap for `view_heap` is currently
    /// bound on `list` (required by `ClearUnorderedAccessView*`), keeping
    /// `host_heaps` and `host_roots` in sync for subsequent draws/dispatches.
    pub(crate) fn ensure_view_heap(&mut self, list: &HostList, view_heap: usize) {
        let wanted = if self.heaps[0] == view_heap {
            self.heaps
        } else {
            [view_heap, self.heaps[1]]
        };
        if self.host_heaps == wanted {
            return;
        }
        let heaps: Vec<Option<ID3D12DescriptorHeap>> = wanted
            .iter()
            .filter(|&&h| h != 0)
            .filter_map(|&h| crate::descriptor::heap(h as This).host())
            .map(|h| Some(h.heap.clone()))
            .collect();
        if heaps.is_empty() {
            return;
        }
        heaps.iter().flatten().for_each(|h| list.keep(h));
        // SAFETY: an open list and shader-visible heaps.
        unsafe { list.list.SetDescriptorHeaps(&heaps) };
        self.host_heaps = wanted;
        self.host_roots = [0; 2];
    }

    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold. Returns `Some(uav_buffers)` with
    /// shadow-buffer pointers bound as UAVs by this operation so callers do
    /// not transition them away from `UNORDERED_ACCESS`, or `None` if a
    /// required fresh table could not be made and the draw/dispatch must drop.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> Option<Vec<usize>> {
        self.depth_sampled = false;
        let mut uav_buffers = Vec::new();
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            let made = self.tables(list, &t.used, compute, &mut uav_buffers);
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                return None;
            }
            if let Some(binding) = stage.user_data {
                let dwords = match binding {
                    crate::pipeline::host::UserDataBinding::Constants { dwords, .. }
                    | crate::pipeline::host::UserDataBinding::RootCbv { dwords, .. } => dwords,
                };
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(stage, compute, u.slot, u.root, u.dword),
                    };
                }
                match binding {
                    crate::pipeline::host::UserDataBinding::Constants { param, dwords } => {
                        // SAFETY: an open list; `values` covers the parameter.
                        unsafe {
                            if compute {
                                l.SetComputeRoot32BitConstants(
                                    param,
                                    dwords,
                                    values.as_ptr().cast(),
                                    0,
                                );
                            } else {
                                l.SetGraphicsRoot32BitConstants(
                                    param,
                                    dwords,
                                    values.as_ptr().cast(),
                                    0,
                                );
                            }
                        }
                    }
                    crate::pipeline::host::UserDataBinding::RootCbv { param, dwords } => {
                        let byte_len = (dwords as usize) * 4;
                        let padded = u64::from(dwords * 4).next_multiple_of(256);
                        if let Some((gpu, to)) = list.scratch(padded) {
                            // SAFETY: `to` points to `padded >= byte_len` writable bytes of
                            // the list's upload ring; `values` is `byte_len` valid bytes.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    values.as_ptr().cast::<u8>(),
                                    to,
                                    byte_len,
                                );
                                if compute {
                                    l.SetComputeRootConstantBufferView(param, gpu);
                                } else {
                                    l.SetGraphicsRootConstantBufferView(param, gpu);
                                }
                            }
                        } else {
                            once!(param, "no scratch upload memory for spilled user-data CBV");
                        }
                    }
                }
            }
            for (views, kind) in [
                (&stage.root_cbvs, RootViewKind::Cbv),
                (&stage.root_srvs, RootViewKind::Srv),
                (&stage.root_uavs, RootViewKind::Uav),
            ] {
                for &(slot, param) in views {
                    let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        continue;
                    };
                    let Some(gpu) = root_view(list, *address, kind, &mut uav_buffers) else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, kind) {
                            (false, RootViewKind::Cbv) => {
                                l.SetGraphicsRootConstantBufferView(param, gpu)
                            }
                            (false, RootViewKind::Srv) => {
                                l.SetGraphicsRootShaderResourceView(param, gpu)
                            }
                            (false, RootViewKind::Uav) => {
                                l.SetGraphicsRootUnorderedAccessView(param, gpu)
                            }
                            (true, RootViewKind::Cbv) => {
                                l.SetComputeRootConstantBufferView(param, gpu)
                            }
                            (true, RootViewKind::Srv) => {
                                l.SetComputeRootShaderResourceView(param, gpu)
                            }
                            (true, RootViewKind::Uav) => {
                                l.SetComputeRootUnorderedAccessView(param, gpu)
                            }
                        }
                    }
                }
            }
            for (views, kind) in [
                (&stage.const_cbvs, RootViewKind::Cbv),
                (&stage.const_srvs, RootViewKind::Srv),
                (&stage.const_uavs, RootViewKind::Uav),
            ] {
                for &(slot, dword, param) in views {
                    let Some(Root::Constants(c)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(
                            slot,
                            "a shader reads a constant-pair root view the title did not set"
                        );
                        continue;
                    };
                    let (Some(&lo), Some(&hi)) = (
                        c.get(dword as usize),
                        c.get((dword as usize).saturating_add(1)),
                    ) else {
                        once!(
                            slot,
                            dword,
                            "a constant-pair root view reads past the set root constants"
                        );
                        continue;
                    };
                    let address = u64::from(lo) | (u64::from(hi) << 32);
                    let Some(gpu) = root_view(list, address, kind, &mut uav_buffers) else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, kind) {
                            (false, RootViewKind::Cbv) => {
                                l.SetGraphicsRootConstantBufferView(param, gpu)
                            }
                            (false, RootViewKind::Srv) => {
                                l.SetGraphicsRootShaderResourceView(param, gpu)
                            }
                            (false, RootViewKind::Uav) => {
                                l.SetGraphicsRootUnorderedAccessView(param, gpu)
                            }
                            (true, RootViewKind::Cbv) => {
                                l.SetComputeRootConstantBufferView(param, gpu)
                            }
                            (true, RootViewKind::Srv) => {
                                l.SetComputeRootShaderResourceView(param, gpu)
                            }
                            (true, RootViewKind::Uav) => {
                                l.SetComputeRootUnorderedAccessView(param, gpu)
                            }
                        }
                    }
                }
            }
        }
        Some(uav_buffers)
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if self.bind(list, p, true).is_none() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(stage, compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage
                .root_cbvs
                .iter()
                .chain(&stage.root_srvs)
                .chain(&stage.root_uavs)
            {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn readonly_depth(&self, compute: bool) -> usize {
        let ro_depth = self.depth_flags & READ_ONLY_DEPTH != 0;
        let ro_stencil = !self.depth_has_stencil || (self.depth_flags & READ_ONLY_STENCIL != 0);
        if !compute && self.depth != 0 && ro_depth && ro_stencil {
            self.depth
        } else {
            0
        }
    }

    fn table_entry(
        &mut self,
        list: &HostList,
        u: &UsedResource,
        compute: bool,
        uav_buffers: &mut Vec<usize>,
    ) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        let ro_depth = self.readonly_depth(compute);
        if let Some(view) = host.mirror_for(index, handle, true, u.kind, u.dwords)
            && readable(list, u, &view, uav_buffers, ro_depth)
        {
            self.depth_sampled = true;
        }
    }

    /// Whether a `View::Constant(address, size)` can be mirrored directly in
    /// place inside the title's own host descriptor heap slot (256-byte host
    /// GPU address alignment and padded size within its `Shadow` buffer),
    /// avoiding dynamic slot allocation and scratch upload copies (#840).
    fn can_mirror_cbv_in_place(address: u64, size: u32) -> bool {
        if size == 0 {
            return false;
        }
        let Some(padded) = size.checked_next_multiple_of(256) else {
            return false;
        };
        let Some(shadow) = crate::shadow::find(address, u64::from(size)) else {
            return false;
        };
        shadow.gpu(address).is_multiple_of(256) && shadow.left(address) >= u64::from(padded)
    }

    /// The descriptors a shader reads through tables. Constant-buffer tables
    /// whose views are 256-byte aligned and fit inside their host shadow buffer
    /// are mirrored directly in place (#840). Tables that hold unaligned or
    /// unshadowed constant buffers (the console requires only 16-byte alignment),
    /// CBV descriptor arrays, or entries reaching past a capped host heap are
    /// made afresh in dynamic slots, with unaligned constant data copied to
    /// upload memory on submission.
    ///
    /// Within a list, the same fresh table holding the same descriptors is made
    /// once so dynamic slots do not wrap onto lists still in flight.
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        compute: bool,
        uav_buffers: &mut Vec<usize>,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = Vec::new();
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let handle = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
                continue;
            }
            if u.kind == ResourceType::Cbv {
                let needs_fresh = u.stride > 0
                    || match crate::descriptor::view(handle) {
                        Some(View::Constant(address, size)) => {
                            !Self::can_mirror_cbv_in_place(address, size)
                        }
                        Some(_) => true,
                        None => false,
                    };
                if needs_fresh {
                    fresh.push(u.slot);
                }
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute, uav_buffers);
            }
        }
        struct Reserved<'a> {
            host: &'a crate::descriptor::host::HostHeap,
            heap_ptr: usize,
            key: (u64, u64),
            first: u32,
            n: u32,
        }
        let mut made = Vec::new();
        let mut reserved: Vec<Reserved<'_>> = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let (base, n) = fresh_table_span(&entries, host.count());
            // The table slice and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                base.hash(&mut h);
                n.hash(&mut h);
                for offset in 0..n {
                    let at = table + u64::from(base + offset) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            let ro_depth = self.readonly_depth(compute);
            if let Some(&table_base) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle)
                        && readable(list, u, &view, uav_buffers, ro_depth)
                    {
                        self.depth_sampled = true;
                    }
                }
                made.push((slot, table_base * 32));
                continue;
            }
            let heap_ptr = std::ptr::from_ref(heap) as usize;
            let Some(first) = host
                .dynamic(n)
                .or_else(|| {
                    let reclaimed = crate::host::reclaim_dynamic_slots(heap_ptr);
                    if reclaimed > 0 {
                        tracing::debug!(
                            reclaimed,
                            needed = n,
                            "reclaimed dynamic descriptor slots from executed lists"
                        );
                        host.dynamic(n)
                    } else {
                        None
                    }
                })
                .or_else(|| host.recycle_dynamic(n))
            else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: roll back slots reserved by this draw and drop (#59, #911).
                self.table_failed = true;
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
                });
                break;
            };
            // `first >= host.count() >= base`, so `first - base` never underflows
            // and shifting the shader's base back by `base` lands `base + offset`
            // directly on dynamic slot `first + offset`.
            let table_base = first - base;
            self.fresh_tables.insert(key, table_base);
            reserved.push(Reserved {
                host,
                heap_ptr,
                key,
                first,
                n,
            });
            // The whole range from the first used entry to the last: a shader may
            // index across intermediate entries (descriptor arrays).
            for offset in 0..n {
                let k = base + offset;
                let index = first + offset;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null_for(index, ResourceType::Cbv, 4);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used
                            && readable(list, u, &view, uav_buffers, ro_depth)
                        {
                            self.depth_sampled = true;
                        }
                    }
                    None => {
                        if let Some(u) = used {
                            host.make_null_for(index, u.kind, u.dwords);
                            once!(slot, "a table entry without a view");
                        } else {
                            host.make_null(index);
                        }
                    }
                }
            }
            made.push((slot, table_base * 32));
        }
        if self.table_failed {
            for r in reserved {
                self.fresh_tables.remove(&r.key);
                r.host.release(r.first, r.n);
            }
            made.clear();
        } else {
            for r in reserved {
                list.record_dynamic(r.heap_ptr, r.first, r.n);
            }
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES;
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let Some((from, src_bytes, dst_bytes)) =
                rect_vertex_range(v.address, v.size, v.stride, first, count)
            else {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            };
            if crate::shadow::find(from, src_bytes).is_none() {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(u64::from(dst_bytes)) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, v.stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: dst_bytes,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let ran = self.topology != RECTANGLES && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own, and
        // leaves `args` in `GENERIC_READ`.
        if !crate::indirect::bound(list, &args, args.gpu(arguments), (kind, stride, max)) {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return false;
        }
        if let Some((s, _)) = &counter {
            list.ensure_buffer(s, BUFFER_READ);
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW | INDEXED => self.topology != RECTANGLES && self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true).is_some(),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => crate::pipeline::note_drop(self.pipeline as This),
                Some(p) => why(&format!("prepare failed (compute {}, mesh {})", p.compute, p.mesh)),
            }
            return false;
        }
        // Ensure indirect argument and count buffers stay in `GENERIC_READ`
        // (includes `INDIRECT_ARGUMENT`) even if `bind` touched the same
        // backing shadow buffer.
        list.ensure_buffer(&args, BUFFER_READ);
        if let Some((s, _)) = &counter {
            list.ensure_buffer(s, BUFFER_READ);
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &args.buffer,
                args.offset(arguments),
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;
const READ_ONLY_DEPTH: u32 = 0x1;
const READ_ONLY_STENCIL: u32 = 0x2;

/// Host D3D12 resource state required for a bound depth-stencil target.
///
/// Read-only DSVs stay in `DEPTH_READ` (combined with shader-resource read
/// bits when simultaneously sampled as an SRV in deferred lighting, decal, or
/// atmosphere passes) instead of clobbering the depth surface into
/// `DEPTH_WRITE`.
fn depth_target_state(flags: u32, has_stencil: bool, sampled_as_srv: bool) -> u32 {
    let ro_depth = flags & READ_ONLY_DEPTH != 0;
    let ro_stencil = !has_stencil || (flags & READ_ONLY_STENCIL != 0);
    if ro_depth && ro_stencil {
        if sampled_as_srv {
            DEPTH_READ | SHADER_RESOURCE
        } else {
            DEPTH_READ
        }
    } else {
        DEPTH_WRITE
    }
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures and buffers moved to the state they need. Returns `true`
/// if `readonly_depth` was simultaneously bound as an SRV so the caller keeps
/// it in `DEPTH_READ | SHADER_RESOURCE`.
fn readable(
    list: &HostList,
    u: &UsedResource,
    view: &View,
    uav_buffers: &mut Vec<usize>,
    readonly_depth: usize,
) -> bool {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        let key = Arc::as_ptr(&shadow) as usize;
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
            if !uav_buffers.contains(&key) {
                uav_buffers.push(key);
            }
            list.ensure_buffer(&shadow, BUFFER_UAV);
        } else if !uav_buffers.contains(&key) {
            list.ensure_buffer(&shadow, BUFFER_READ);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            let sampled_depth = readonly_depth != 0 && *resource == readonly_depth;
            let wanted = if sampled_depth {
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, wanted);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
            sampled_depth
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            crate::texture::host_written(*resource as This);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            false
        }
        _ => false,
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        // Read via `host::read_into` (kernel-checked) to avoid TOCTOU races
        // against unmapping/guard pages (#913).
        Some(View::Constant(address, size)) if *size >= 16 => {
            let mut bytes = [0u8; 16];
            if crate::host::read_into(*address as *const u8, &mut bytes) {
                let v = [
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                    f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
                    f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
                ];
                format!("Constant({address:#x}, {size}) {v:?}")
            } else {
                format!("Constant({address:#x}, {size})")
            }
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    if crate::host::host().is_some_and(crate::host::Host::depth_bounds)
        && let Ok(l1) = list.list.cast::<ID3D12GraphicsCommandList1>()
    {
        // SAFETY: an open list on a host supporting depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}

/// Computes `(base, count)` for a fresh dynamic table covering `entries`,
/// trimming leading unused descriptor slots so tables starting deep in a
/// descriptor table do not waste dynamic heap slots (#747). `max_trim` is
/// `host.count()`, ensuring `first - base` never underflows.
fn fresh_table_span(entries: &[&UsedResource], max_trim: u32) -> (u32, u32) {
    let mut start = u32::MAX;
    let mut end = 0u32;
    for u in entries {
        let entry = u.table_offset_dwords / 8;
        start = start.min(entry);
        end = end.max(entry + 1);
    }
    if start == u32::MAX {
        return (0, 1);
    }
    let base = start.min(max_trim);
    (base, end - base)
}

/// Validates and computes `(from_address, src_bytes, dst_bytes)` for expanding
/// `count / 3` console rectangles starting at vertex `first` into triangles.
fn rect_vertex_range(
    address: u64,
    size: u32,
    stride: u32,
    first: u32,
    count: u32,
) -> Option<(u64, u64, u32)> {
    if address == 0 || stride == 0 {
        return None;
    }
    let stride = u64::from(stride);
    let end_vertex = u64::from(first).checked_add(u64::from(count))?;
    if end_vertex.checked_mul(stride)? > u64::from(size) {
        return None;
    }
    let from = address.checked_add(u64::from(first).checked_mul(stride)?)?;
    let rects = u64::from(count / 3);
    let src_bytes = rects.checked_mul(3)?.checked_mul(stride)?;
    let dst_bytes = u32::try_from(rects.checked_mul(6)?.checked_mul(stride)?).ok()?;
    Some((from, src_bytes, dst_bytes))
}

#[cfg(test)]
mod tests {
    use super::{
        DEPTH_READ, DEPTH_WRITE, READ_ONLY_DEPTH, READ_ONLY_STENCIL, SHADER_RESOURCE,
        depth_target_state, fresh_table_span, rect_vertex_range,
    };
    use victory_shader::{ResourceType, RootType, UsedResource};

    fn used_table_entry(offset_dwords: u32) -> UsedResource {
        UsedResource {
            root: RootType::Table,
            slot: 0,
            user_data: 0,
            table_offset_dwords: offset_dwords,
            kind: ResourceType::Cbv,
            dwords: 4,
            stride: 0,
            lower_bound: 0,
        }
    }

    #[test]
    fn fresh_table_span_trims_leading_unused_entries() {
        // Empty entry list falls back to 1 slot at base 0.
        assert_eq!(fresh_table_span(&[], 1000), (0, 1));

        // Single entry deep in the table allocates 1 slot instead of 101.
        let deep = used_table_entry(100 * 8);
        assert_eq!(fresh_table_span(&[&deep], 737_856), (100, 1));

        // Multiple entries span [min..=max] inclusively.
        let a = used_table_entry(40 * 8);
        let b = used_table_entry(45 * 8 + 4);
        assert_eq!(fresh_table_span(&[&a, &b], 737_856), (40, 6));

        // `base` is clamped to `max_trim` (`host.count()`) so `first - base`
        // can never underflow even on a tiny heap.
        let huge = used_table_entry(500 * 8);
        assert_eq!(fresh_table_span(&[&huge], 128), (128, 373));
    }

    #[test]
    fn rect_vertex_range_rejects_overflow_and_out_of_bounds() {
        assert_eq!(rect_vertex_range(0x1000, 96, 16, 0, 6), Some((0x1000, 96, 192)));
        assert_eq!(rect_vertex_range(0x1000, 96, 16, 3, 3), Some((0x1030, 48, 96)));
        // `first + count` wraps u32 if not widened to u64.
        assert_eq!(rect_vertex_range(0x1000, u32::MAX, 16, u32::MAX - 2, 3), None);
        // Zero address or stride or past buffer size.
        assert_eq!(rect_vertex_range(0, 96, 16, 0, 3), None);
        assert_eq!(rect_vertex_range(0x1000, 96, 0, 0, 3), None);
        assert_eq!(rect_vertex_range(0x1000, 47, 16, 0, 3), None);
        // Expanded triangle buffer would exceed u32 `SizeInBytes`.
        assert_eq!(rect_vertex_range(0x1000, u32::MAX, 16, 0, (1 << 28) - 1), None);
    }

    #[test]
    fn depth_target_state_preserves_read_only_and_srv_reads() {
        // Writable DSV always needs DEPTH_WRITE.
        assert_eq!(depth_target_state(0, false, false), DEPTH_WRITE);
        assert_eq!(depth_target_state(0, true, false), DEPTH_WRITE);
        assert_eq!(depth_target_state(0, false, true), DEPTH_WRITE);

        // Depth-only read-only DSV stays in DEPTH_READ (or DEPTH_READ|SRV when sampled).
        assert_eq!(depth_target_state(READ_ONLY_DEPTH, false, false), DEPTH_READ);
        assert_eq!(
            depth_target_state(READ_ONLY_DEPTH, false, true),
            DEPTH_READ | SHADER_RESOURCE
        );

        // Depth+stencil: both read-only stays in DEPTH_READ / DEPTH_READ|SRV;
        // writable stencil requires DEPTH_WRITE even when depth is read-only.
        assert_eq!(
            depth_target_state(READ_ONLY_DEPTH | READ_ONLY_STENCIL, true, false),
            DEPTH_READ
        );
        assert_eq!(
            depth_target_state(READ_ONLY_DEPTH | READ_ONLY_STENCIL, true, true),
            DEPTH_READ | SHADER_RESOURCE
        );
        assert_eq!(depth_target_state(READ_ONLY_DEPTH, true, false), DEPTH_WRITE);
        assert_eq!(depth_target_state(READ_ONLY_DEPTH, true, true), DEPTH_WRITE);
    }
}
