//! Persistent GPU buffers and the CPU→GPU upload path.
//!
//! * [`Uploader`] is a staging ring: equally sized mapped chunks that are only reused once the
//!   GPU has consumed them (frame-in-flight safe, no write-after-read hazards, no per-frame
//!   staging allocations in the steady state). Data is written straight into mapped staging
//!   memory; large copies are split across rayon workers on native.
//! * [`GrowBuffer`] is a persistent GPU buffer with amortized growth (1.5x, 256-byte aligned).
//! * [`TrackedBuffer`] adds a CPU shadow copy; writes are diffed against it so only changed
//!   byte ranges are uploaded (e.g. unchanged camera → 0 bytes).

use crate::dirty::DirtyRanges;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Copies at or above this size are parallelized (native only).
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_COPY_THRESHOLD: usize = 1 << 20;

/// Size of every staging chunk. Larger writes are split into pieces of at most this size, so
/// all chunks are interchangeable: any idle chunk serves any request, and the ring follows the
/// per-frame upload volume instead of keeping one chunk per record-sized write forever (which
/// `wgpu::util::StagingBelt` does: it never frees a chunk, and a write larger than every
/// existing chunk creates a new one of exactly that size).
pub(crate) const STAGING_CHUNK: u64 = 4 << 20;

/// Frames per window of the peak-usage tracker: idle chunks beyond what the last one to two
/// windows needed (twice the busiest frame, or the busiest frame for each frame the GPU may
/// still hold plus one) are released.
const TRIM_WINDOW: u32 = 128;

/// Frames per window of the queue-depth tracker: the allowance is the deepest queue of the last
/// one to two windows. It follows a deeper queue at once and keeps it for 512–1024 frames, so a
/// queue whose depth varies (frames recorded ahead of an unpaced GPU) keeps its chunks instead
/// of dropping and re-creating them, while the chunks of a one-off stall (the GPU fell far
/// behind once) are released within about 17 s at 60 Hz.
const DEPTH_WINDOW: u32 = 4 * TRIM_WINDOW;

/// Round `v` up to a multiple of `align` (power of two).
pub(crate) fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

/// Chunk states, written by the `map_async` callback.
const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const MAP_FAILED: u8 = 2;

/// One `MAP_WRITE | COPY_SRC` staging buffer of [`STAGING_CHUNK`] bytes.
struct Chunk {
    buffer: wgpu::Buffer,
    /// Frame (`Uploader::frames`) of its last submission.
    frame: u64,
    /// Bytes handed out since the chunk was (re)mapped.
    offset: u64,
    /// Set by the `map_async` callback once the GPU is done with the chunk.
    state: Arc<AtomicU8>,
}

/// Per-frame upload context: staging ring + counters.
///
/// Chunks move `free` → `active` (mapped, being filled) → `closed` (unmapped at
/// [`Self::finish`], read by the submitted copies) → `in_flight` (re-mapping after
/// [`Self::recall`]) → `free` once the GPU finished with them.
pub(crate) struct Uploader {
    device: wgpu::Device,
    active: Vec<Chunk>,
    closed: Vec<Chunk>,
    in_flight: Vec<Chunk>,
    free: Vec<Chunk>,
    /// Chunks taken into `active` since the last `recall`.
    used_this_frame: u32,
    /// Most chunks one frame used: in the current window, in the previous one.
    peak: [u32; 2],
    window_frames: u32,
    /// Frames submitted (`recall` calls).
    frames: u64,
    /// Deepest queue (frames whose chunks the GPU still held after a submission): in the current
    /// window of [`DEPTH_WINDOW`] frames, in the previous one.
    depth: [u32; 2],
    depth_frames: u32,
    /// Bytes written this frame.
    pub bytes: u64,
    /// Instance bytes written this frame.
    pub instance_bytes: u64,
    /// GPU buffers (re)allocated this frame (staging chunks not included).
    pub reallocations: u32,
    /// Staging chunks created this frame.
    pub chunks_created: u32,
    /// Whether the "device lost, skipping uploads" warning was already logged.
    lost_logged: bool,
    /// Staging chunks created so far.
    created: u64,
}

impl Uploader {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            device: device.clone(),
            active: Vec::new(),
            closed: Vec::new(),
            in_flight: Vec::new(),
            free: Vec::new(),
            used_this_frame: 0,
            peak: [0; 2],
            window_frames: 0,
            frames: 0,
            depth: [0; 2],
            depth_frames: 0,
            bytes: 0,
            instance_bytes: 0,
            reallocations: 0,
            chunks_created: 0,
            lost_logged: false,
            created: 0,
        }
    }

    pub fn reset_counters(&mut self) {
        self.bytes = 0;
        self.instance_bytes = 0;
        self.reallocations = 0;
        self.chunks_created = 0;
    }

    /// Copy `data` into `target` at `offset` through staging memory. `offset` and the length
    /// are padded to 4 bytes by the caller's layout (all our element sizes are multiples of 4).
    /// Writes larger than a chunk are split; the copies are recorded in order, so ordering
    /// against other commands in `encoder` is the same as for one copy.
    pub fn write(&mut self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.write_parts(encoder, target, offset, data.len(), [data]);
    }

    /// Like [`Self::write`] for `parts` (together `total` bytes) that go back to back from
    /// `offset`: one staging allocation and one copy per chunk-sized piece instead of one per
    /// part. Many small copies are costly on the GPU side (Metal tracks every blit's overlaps).
    pub fn write_parts<'a>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Buffer,
        offset: u64,
        total: usize,
        parts: impl IntoIterator<Item = &'a [u8]>,
    ) {
        debug_assert!(offset.is_multiple_of(4) && total.is_multiple_of(4), "unaligned upload");
        let mut parts = parts.into_iter();
        let mut cur: &[u8] = &[];
        let mut done = 0;
        while done < total {
            let (chunk, at, size) = self.allocate((total - done) as u64);
            let size = size as usize;
            let buffer = &self.active[chunk].buffer;
            encoder.copy_buffer_to_buffer(buffer, at, target, offset + done as u64, size as u64);
            let Ok(mut view) = buffer.slice(at..at + size as u64).get_mapped_range_mut() else {
                // Only a lost device leaves a fresh or recycled chunk unmapped. The copy just
                // recorded is invalid anyway (the frame is discarded); skip the rest of this
                // upload instead of crashing. Errors after loss are logged, not raised.
                if !self.lost_logged {
                    self.lost_logged = true;
                    tracing::warn!("staging memory unavailable (GPU device lost); skipping uploads");
                }
                return;
            };
            let mut filled = 0;
            while filled < size {
                if cur.is_empty() {
                    cur = parts.next().expect("write_parts: parts shorter than `total`");
                    continue;
                }
                let n = cur.len().min(size - filled);
                copy_to_staging(view.slice(filled..filled + n), &cur[..n]);
                cur = &cur[n..];
                filled += n;
            }
            done += size;
        }
        debug_assert!(
            cur.is_empty() && parts.all(|p| p.is_empty()),
            "write_parts: parts longer than `total`"
        );
        self.bytes += total as u64;
    }

    /// Reserve up to `want` (multiple of 4) bytes: `(index into active, offset, size)`. A
    /// write that does not fit whole takes what is left of an active chunk (if that is not a
    /// sliver) and continues in the next one, so large uploads pack chunks densely.
    fn allocate(&mut self, want: u64) -> (usize, u64, u64) {
        /// Smallest remainder of a chunk worth starting a split write in.
        const MIN_PIECE: u64 = 256 << 10;
        let mut best = None;
        for (i, c) in self.active.iter().enumerate() {
            let at = align_up(c.offset, wgpu::MAP_ALIGNMENT);
            let room = STAGING_CHUNK - at.min(STAGING_CHUNK);
            if room >= want {
                best = Some((i, at, want));
                break;
            }
            if room >= MIN_PIECE && best.is_none() {
                best = Some((i, at, room));
            }
        }
        if let Some((i, at, size)) = best {
            self.active[i].offset = at + size;
            return (i, at, size);
        }
        let size = want.min(STAGING_CHUNK);
        self.collect_returned();
        let (created, this_frame) = (&mut self.created, &mut self.chunks_created);
        let mut chunk = self.free.pop().unwrap_or_else(|| {
            *created += 1;
            *this_frame += 1;
            Chunk {
                buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("space3d staging"),
                    size: STAGING_CHUNK,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    // MAP_WRITE buffers are mapped directly (no hidden staging copy).
                    mapped_at_creation: true,
                }),
                offset: 0,
                state: Arc::new(AtomicU8::new(MAPPED)),
                frame: 0,
            }
        });
        chunk.offset = size;
        self.active.push(chunk);
        self.used_this_frame += 1;
        (self.active.len() - 1, 0, size)
    }

    /// Move chunks the GPU has finished with (mapped again) to `free`.
    fn collect_returned(&mut self) {
        let mut i = 0;
        while i < self.in_flight.len() {
            match self.in_flight[i].state.load(Ordering::Acquire) {
                MAPPED => {
                    let mut c = self.in_flight.swap_remove(i);
                    c.offset = 0;
                    self.free.push(c);
                }
                // Mapping failed (device lost): drop the chunk.
                MAP_FAILED => drop(self.in_flight.swap_remove(i)),
                _ => i += 1,
            }
        }
    }

    /// Close staging chunks before submission.
    pub fn finish(&mut self) {
        for c in self.active.drain(..) {
            c.buffer.unmap();
            self.closed.push(c);
        }
    }

    /// After submission: re-map the chunks just submitted (they come back once the GPU is
    /// done with them) and release idle chunks beyond what recent frames needed.
    pub fn recall(&mut self) {
        self.frames += 1;
        for mut c in self.closed.drain(..) {
            c.state.store(MAPPING, Ordering::Relaxed);
            c.frame = self.frames;
            let state = c.state.clone();
            c.buffer.slice(..).map_async(wgpu::MapMode::Write, move |r| {
                state.store(if r.is_ok() { MAPPED } else { MAP_FAILED }, Ordering::Release);
            });
            self.in_flight.push(c);
        }
        self.collect_returned();

        let used = std::mem::take(&mut self.used_this_frame);
        self.peak[0] = self.peak[0].max(used);
        self.window_frames += 1;
        if self.window_frames >= TRIM_WINDOW {
            self.peak = [0, self.peak[0]];
            self.window_frames = 0;
        }
        // Queue depth: the frames from the oldest submission the GPU still holds to this one.
        let oldest = self.in_flight.iter().map(|c| c.frame).min();
        let depth = oldest.map_or(0, |f| (self.frames - f + 1) as u32);
        self.depth[0] = self.depth[0].max(depth);
        self.depth_frames += 1;
        if self.depth_frames >= DEPTH_WINDOW {
            self.depth = [0, self.depth[0]];
            self.depth_frames = 0;
        }
        let depth = self.depth[0].max(self.depth[1]) as usize;
        // Dropping a chunk frees it (it is idle: not referenced by any submission). Kept: at
        // least twice the busiest frame, and the busiest frame for every frame of the queue
        // depth plus the next one — with frames queued, every frame's chunks are still in
        // flight when the next frame allocates, and trimming them would create and drop a
        // 4 MiB chunk every few frames even for a 16-byte upload. One spare stays, so an
        // occasional small edit after a quiet period does not create a chunk.
        let peak = self.peak[0].max(self.peak[1]) as usize;
        let resident = (2 * peak).max(peak * (depth + 1)).max(1);
        let keep = resident.saturating_sub(self.in_flight.len());
        self.free.truncate(keep);
    }

    /// Staging chunks created since the uploader was made (0 more in the steady state).
    #[cfg(test)]
    pub fn chunks_created(&self) -> u64 {
        self.created
    }

    /// Staging chunks currently allocated (in any state).
    #[cfg(test)]
    pub fn resident_chunks(&self) -> usize {
        self.active.len() + self.closed.len() + self.in_flight.len() + self.free.len()
    }
}

/// memcpy into write-combined staging memory, split across threads for large copies.
pub(crate) fn copy_to_staging(mut dst: wgpu::WriteOnly<'_, [u8]>, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(not(target_arch = "wasm32"))]
    if src.len() >= PARALLEL_COPY_THRESHOLD {
        /// Raw pointer into the mapped range, handed to exactly one worker.
        struct SendPtr(std::ptr::NonNull<u8>);
        // SAFETY: each task writes a disjoint sub-range of `dst`, which stays borrowed (and
        // therefore mapped and valid for writes) until `for_each` returns.
        unsafe impl Send for SendPtr {}
        // SAFETY: only used to derive the per-task pointers above; never dereferenced shared.
        unsafe impl Sync for SendPtr {}

        use rayon::prelude::*;
        let threads = rayon::current_num_threads().clamp(1, 8);
        let chunk = src.len().div_ceil(threads).next_multiple_of(4096);
        let base = SendPtr(dst.as_raw_element_ptr());
        let base = &base;
        // A parallel iterator splits with `join` (stack-allocated jobs), unlike `scope.spawn`,
        // which boxes every task: no heap allocation per frame.
        src.par_chunks(chunk).enumerate().for_each(|(i, part)| {
            // SAFETY: `i * chunk + part.len() <= dst.len()` because `src.len() == dst.len()`;
            // the range is exclusively owned by this task and stays mapped (see `SendPtr`).
            let mut w = unsafe {
                wgpu::WriteOnly::<[u8]>::new(std::ptr::NonNull::slice_from_raw_parts(
                    base.0.add(i * chunk),
                    part.len(),
                ))
            };
            w.copy_from_slice(part);
        });
        return;
    }
    dst.copy_from_slice(src);
}

/// Persistent GPU buffer with amortized growth.
pub(crate) struct GrowBuffer {
    label: &'static str,
    usage: wgpu::BufferUsages,
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
    /// Bumped on every reallocation so dependent bind groups can be rebuilt.
    pub generation: u32,
}

impl GrowBuffer {
    pub fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self {
            label,
            usage,
            buffer: None,
            capacity: 0,
            generation: 0,
        }
    }

    /// Ensure at least `size` bytes. Returns true if (re)allocated (contents lost).
    /// `exact` sizes the buffer tightly (static data), otherwise it grows 1.5x — but never
    /// beyond the device's `max_buffer_size`. Callers keep `size` itself within that limit
    /// (a larger buffer cannot exist and would fail validation).
    pub fn ensure(&mut self, device: &wgpu::Device, size: u64, exact: bool, up: &mut Uploader) -> bool {
        let size = align_up(size.max(16), 16);
        if self.buffer.is_some() && size <= self.capacity {
            return false;
        }
        let wanted = if exact {
            align_up(size, 256)
        } else {
            align_up(size.max(self.capacity + self.capacity / 2), 256)
        };
        let max = device.limits().max_buffer_size;
        debug_assert!(size <= max, "{}: {size} B exceed max_buffer_size ({max} B)", self.label);
        let capacity = wanted.min(max & !255).max(size);
        self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: capacity,
            usage: self.usage,
            mapped_at_creation: false,
        }));
        self.capacity = capacity;
        self.generation = self.generation.wrapping_add(1);
        up.reallocations += 1;
        true
    }

    /// The buffer; panics if never allocated (callers `ensure` first).
    pub fn get(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("GrowBuffer used before ensure()")
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

/// GPU buffer mirrored by a CPU shadow; writes are diffed and only changed ranges uploaded.
pub(crate) struct TrackedBuffer {
    pub gpu: GrowBuffer,
    shadow: Vec<u8>,
    len: usize,
    dirty: DirtyRanges,
    /// Set when the GPU copy must be fully rewritten (after reallocation).
    stale: bool,
    /// Changed runs per chunk of a parallel [`Self::write_gather`] (reused scratch).
    #[cfg(not(target_arch = "wasm32"))]
    chunk_runs: Vec<ChunkRuns>,
}

/// Elements per task of a parallel [`TrackedBuffer::write_gather`].
#[cfg(not(target_arch = "wasm32"))]
const GATHER_CHUNK: usize = 2048;
/// [`TrackedBuffer::write_gather`] of at least this many elements runs on all rayon workers.
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_GATHER_WRITE: usize = 32_768;

/// Up to four changed element runs of one chunk (a fifth extends the last one).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Default)]
struct ChunkRuns {
    n: u8,
    runs: [(u32, u32); 4],
}

#[cfg(not(target_arch = "wasm32"))]
impl ChunkRuns {
    fn add(&mut self, k: u32) {
        match self.n {
            0 => {}
            n if self.runs[n as usize - 1].1 == k => {
                self.runs[n as usize - 1].1 = k + 1;
                return;
            }
            4 => {
                self.runs[3].1 = k + 1;
                return;
            }
            _ => {}
        }
        self.runs[self.n as usize] = (k, k + 1);
        self.n += 1;
    }
}

impl TrackedBuffer {
    pub fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self {
            gpu: GrowBuffer::new(label, usage | wgpu::BufferUsages::COPY_DST),
            shadow: Vec::new(),
            len: 0,
            dirty: DirtyRanges::with_merge_gap(64),
            stale: true,
            #[cfg(not(target_arch = "wasm32"))]
            chunk_runs: Vec::new(),
        }
    }

    /// Start rewriting from offset 0 (content beyond the new length is ignored).
    pub fn begin(&mut self) {
        self.len = 0;
    }

    /// Bytes in use.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Make room for `len` bytes, counting CPU reallocations.
    fn reserve_to(&mut self, len: usize, cpu_reallocs: &mut u32) {
        if len > self.shadow.len() {
            if len > self.shadow.capacity() {
                *cpu_reallocs += 1;
                self.shadow
                    .reserve(len.max(self.shadow.capacity() * 3 / 2) - self.shadow.len());
            }
            // The GPU tail past the old shadow length was never written, so it is still
            // zero-initialized (wgpu guarantees this) and matches the zeroed shadow tail.
            self.shadow.resize(len, 0);
        }
    }

    /// Write `data` at `offset`, marking only bytes that differ from the shadow.
    pub fn write_at(&mut self, offset: usize, data: &[u8], cpu_reallocs: &mut u32) {
        let end = offset + data.len();
        self.reserve_to(end, cpu_reallocs);
        let dst = &mut self.shadow[offset..end];
        if dst != data {
            dst.copy_from_slice(data);
            self.dirty.mark(offset..end);
        }
        self.len = self.len.max(end);
    }

    /// Append at the current length; returns the offset written.
    pub fn push(&mut self, data: &[u8], cpu_reallocs: &mut u32) -> usize {
        let at = self.len;
        self.write_at(at, data, cpu_reallocs);
        at
    }

    /// Grow the used length to at least `n` bytes (zero-filled), so bindings of one full
    /// element stay valid even when nothing was written this frame.
    pub fn ensure_len(&mut self, n: usize, cpu_reallocs: &mut u32) {
        if self.len < n {
            self.reserve_to(n, cpu_reallocs);
            self.len = n;
        }
    }

    /// Write `count` elements (yielded by `items`) contiguously from `offset`, like one
    /// [`Self::write_at`] per element but reserving once and marking each run of changed
    /// elements with a single dirty range (instead of one mark per element).
    pub fn write_all<'a, T: bytemuck::Pod>(
        &mut self,
        offset: usize,
        count: usize,
        items: impl IntoIterator<Item = &'a T>,
        cpu_reallocs: &mut u32,
    ) {
        let size = std::mem::size_of::<T>();
        let end = offset + count * size;
        self.reserve_to(end, cpu_reallocs);
        let Self { shadow, dirty, .. } = self;
        let dst = &mut shadow[offset..end];
        let mut run: Option<usize> = None;
        let mut written = 0;
        for (k, (item, d)) in items.into_iter().zip(dst.chunks_exact_mut(size)).enumerate() {
            let src = bytemuck::bytes_of(item);
            if d != src {
                d.copy_from_slice(src);
                run.get_or_insert(k);
            } else if let Some(s) = run.take() {
                dirty.mark(offset + s * size..offset + k * size);
            }
            written = k + 1;
        }
        debug_assert_eq!(written, count, "write_all: item count mismatch");
        if let Some(s) = run {
            dirty.mark(offset + s * size..offset + written * size);
        }
        self.len = self.len.max(end);
    }

    /// [`Self::write_all`] of `items[order[k]]` for every `k`. Large gathers (native) diff and
    /// copy chunks of elements on all rayon workers; each chunk reports up to four changed
    /// runs (more are merged into the last), so dirty ranges stay exact for sparse changes.
    pub fn write_gather<T: bytemuck::Pod + Sync>(
        &mut self,
        offset: usize,
        order: &[u32],
        items: &[T],
        cpu_reallocs: &mut u32,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        if order.len() >= PARALLEL_GATHER_WRITE {
            use rayon::prelude::*;
            let size = std::mem::size_of::<T>();
            let end = offset + order.len() * size;
            self.reserve_to(end, cpu_reallocs);
            let chunks = order.len().div_ceil(GATHER_CHUNK);
            if self.chunk_runs.capacity() < chunks {
                *cpu_reallocs += 1;
            }
            self.chunk_runs.clear();
            self.chunk_runs.resize(chunks, ChunkRuns::default());
            let Self {
                shadow,
                dirty,
                chunk_runs,
                ..
            } = self;
            shadow[offset..end]
                .par_chunks_mut(GATHER_CHUNK * size)
                .zip(order.par_chunks(GATHER_CHUNK))
                .zip(chunk_runs.par_iter_mut())
                .for_each(|((dst, order), runs)| {
                    for (k, (d, &i)) in dst.chunks_exact_mut(size).zip(order).enumerate() {
                        let src = bytemuck::bytes_of(&items[i as usize]);
                        if d != src {
                            d.copy_from_slice(src);
                            runs.add(k as u32);
                        }
                    }
                });
            for (c, r) in chunk_runs.iter().enumerate() {
                for &(a, b) in &r.runs[..r.n as usize] {
                    let base = offset + c * GATHER_CHUNK * size;
                    dirty.mark(base + a as usize * size..base + b as usize * size);
                }
            }
            self.len = self.len.max(end);
            return;
        }
        self.write_all(
            offset,
            order.len(),
            order.iter().map(|&i| &items[i as usize]),
            cpu_reallocs,
        );
    }

    /// Ensure capacity; after a reallocation the whole shadow is uploaded. Returns the buffer
    /// when only dirty ranges remain to be uploaded, `None` when everything was.
    fn prepare_flush(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        up: &mut Uploader,
    ) -> Option<wgpu::Buffer> {
        let len = self.len.next_multiple_of(4);
        if len > self.shadow.len() {
            self.shadow.resize(len, 0);
        }
        if self.gpu.ensure(device, len as u64, false, up) {
            self.stale = true;
        }
        let buffer = self.gpu.get().clone();
        if self.stale {
            // Upload the whole shadow up to capacity-relevant length.
            let full = self.shadow.len().next_multiple_of(4).min(self.gpu.capacity() as usize);
            self.shadow.resize(full, 0);
            up.write(encoder, &buffer, 0, &self.shadow[..full]);
            self.stale = false;
            self.dirty.clear();
            return None;
        }
        Some(buffer)
    }

    /// Upload changed ranges (or everything after reallocation). Returns bytes uploaded.
    pub fn flush(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, up: &mut Uploader) -> u64 {
        let before = up.bytes;
        if let Some(buffer) = self.prepare_flush(device, encoder, up) {
            for r in self.dirty.ranges() {
                let start = r.start & !3;
                let end = r.end.next_multiple_of(4).min(self.shadow.len());
                if start < end {
                    up.write(encoder, &buffer, start as u64, &self.shadow[start..end]);
                }
            }
            self.dirty.clear();
        }
        up.bytes - before
    }

    /// Like [`Self::flush`], but upload only the dirty bytes inside `needed` (sorted,
    /// non-overlapping, 4-byte aligned byte ranges). Dirty bytes outside stay pending — the
    /// shadow keeps their current value — until a later flush needs them. Returns bytes uploaded.
    pub fn flush_needed(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        up: &mut Uploader,
        needed: &[Range<usize>],
    ) -> u64 {
        let before = up.bytes;
        if let Some(buffer) = self.prepare_flush(device, encoder, up) {
            let mut j = 0;
            for r in self.dirty.ranges() {
                let (start, end) = (r.start & !3, r.end.next_multiple_of(4).min(self.shadow.len()));
                while j < needed.len() && needed[j].end <= start {
                    j += 1;
                }
                for n in needed[j..].iter().take_while(|n| n.start < end) {
                    let (a, b) = (start.max(n.start), end.min(n.end));
                    if a < b {
                        up.write(encoder, &buffer, a as u64, &self.shadow[a..b]);
                    }
                }
            }
            self.dirty.subtract(needed);
        }
        up.bytes - before
    }

    /// Byte ranges written but not uploaded yet (tests).
    #[cfg(test)]
    pub fn pending(&self) -> &[Range<usize>] {
        self.dirty.ranges()
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.gpu.get()
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    /// A device limited to 1 MiB buffers (skips without an adapter).
    fn small_device() -> Option<wgpu::Device> {
        if crate::gpu_disabled_by_env() {
            return None;
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let limits = wgpu::Limits {
            max_buffer_size: 1 << 20,
            ..adapter.limits()
        };
        let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }))
        .ok()?;
        Some(device)
    }

    fn device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
        if crate::gpu_disabled_by_env() {
            return None;
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()
    }

    /// The parallel gather write matches the serial one: same shadow, same dirty ranges while
    /// changes are sparse (≤ 4 runs per chunk), and dirty ranges that still cover every
    /// changed element when they are dense.
    #[test]
    fn parallel_gather_writes_match_serial_writes() {
        let n = 100_000usize;
        let mut seed = 7u64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as usize
        };
        let order: Vec<u32> = (0..n as u32).rev().collect();
        let mut items: Vec<[f32; 4]> = (0..n).map(|i| [i as f32; 4]).collect();
        let usage = wgpu::BufferUsages::VERTEX;
        let (mut serial, mut parallel) = (TrackedBuffer::new("s", usage), TrackedBuffer::new("p", usage));
        let mut r = 0;
        for changes in [0usize, 10, 40, 3000, 60_000] {
            for _ in 0..changes {
                items[rand() % n][1] += 1.0;
            }
            serial.dirty.clear();
            parallel.dirty.clear();
            serial.begin();
            parallel.begin();
            serial.write_all(0, n, order.iter().map(|&i| &items[i as usize]), &mut r);
            parallel.write_gather(0, &order, &items, &mut r);
            assert_eq!(serial.shadow, parallel.shadow, "{changes} changes");
            assert_eq!(serial.len(), parallel.len());
            if changes <= 40 {
                assert_eq!(serial.pending(), parallel.pending(), "{changes} sparse changes");
            } else {
                let covered = |ranges: &[Range<usize>], x: usize| ranges.iter().any(|q| q.contains(&x));
                for q in serial.pending() {
                    assert!(covered(parallel.pending(), q.start) && covered(parallel.pending(), q.end - 1));
                }
            }
        }
    }

    /// The rayon split of large staging copies (disjoint chunks, a shorter last one) writes
    /// every byte to its own offset; checked without a GPU through `WriteOnly::from_mut`.
    #[test]
    fn parallel_staging_copies_are_exact() {
        for len in [
            PARALLEL_COPY_THRESHOLD - 4,
            PARALLEL_COPY_THRESHOLD,
            3_200_000,
            (5 << 20) + 12,
        ] {
            let src: Vec<u8> = (0..len).map(|i| (i * 7 + i / 4093) as u8).collect();
            let mut dst = vec![0u8; len];
            copy_to_staging(wgpu::WriteOnly::from_mut(&mut dst[..]), &src);
            let first_bad = dst.iter().zip(&src).position(|(a, b)| a != b);
            assert_eq!(first_bad, None, "len {len}");
        }
    }

    /// One "frame": the given writes into `target`, submit, recycle, wait for the GPU.
    fn frame(device: &wgpu::Device, queue: &wgpu::Queue, up: &mut Uploader, target: &wgpu::Buffer, writes: &[&[u8]]) {
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut offset = 0;
        for w in writes {
            up.write(&mut encoder, target, offset, w);
            offset += w.len() as u64;
        }
        up.finish();
        queue.submit([encoder.finish()]);
        up.recall();
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }

    /// Regression: `StagingBelt` never frees a chunk and creates a new one of exactly the
    /// request size whenever a write is larger than every chunk it has, so an upload that grows
    /// a little each frame (a growing particle `Stream` buffer, a reallocated ECS buffer)
    /// leaked one ever-larger mapped chunk per frame. Every chunk now has the same size, large
    /// writes are split, and the ring stays proportional to one frame's traffic.
    #[test]
    fn growing_uploads_do_not_accumulate_staging_memory() {
        let Some((device, queue)) = device_queue() else { return };
        let mut up = Uploader::new(&device);
        let max = 4 * STAGING_CHUNK as usize;
        let target = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: max as u64 + 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let data = vec![7u8; max];
        let mut peak = 0;
        for i in 0..64 {
            let n = (STAGING_CHUNK as usize + (64 << 10) + i * (192 << 10)).min(max);
            frame(&device, &queue, &mut up, &target, &[&[1u8; 256], &data[..n]]);
            peak = peak.max(up.resident_chunks());
        }
        // A frame needs at most 5 chunks; frames in flight, spares for the next frame and
        // one-frame latency of the returns fit well within 16. The belt reached 64+.
        assert!(peak <= 16, "resident staging chunks: {peak}");
    }

    /// Regression (review): with several frames queued on the GPU, each frame's chunk is still
    /// in flight when the next frame allocates. Trimming idle chunks to twice the per-frame peak
    /// dropped the returning ones, so a still HDR scene (16 bytes of post seeds per frame)
    /// created and dropped a 4 MiB chunk every few frames (0.5–1 ms of CPU each). The ring now
    /// keeps what the recent in-flight depth needs: nothing is created in the steady state.
    #[test]
    fn queued_frames_do_not_churn_staging_chunks() {
        let Some((device, queue)) = device_queue() else { return };
        let mut up = Uploader::new(&device);
        let target = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // The GPU is waited for only every `pace`th frame: up to `pace` frames (chunks) in
        // flight.
        let run = |up: &mut Uploader, frames: u32, pace: u32| {
            for i in 0..frames {
                let mut encoder = device.create_command_encoder(&Default::default());
                up.write(&mut encoder, &target, 0, &[7u8; 16]);
                up.finish();
                queue.submit([encoder.finish()]);
                up.recall();
                if i % pace == pace - 1 {
                    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
                }
            }
        };
        // The ring grows to the queue's depth (at most 7 chunks: a wait every 6th frame
        // returns them all; exactly when it gets there depends on the GPU's timing) and then
        // creates nothing, across depth windows (512 frames) while the depth recurs. Trimming
        // the in-flight chunks away created one every few frames: hundreds here.
        run(&mut up, 1260, 6);
        let created = up.chunks_created();
        assert!(
            created <= 8,
            "{created} chunks created in 1260 frames of a 6-deep queue"
        );
        assert!(up.resident_chunks() <= 8, "{}", up.resident_chunks());
        // A one-off stall (up to 40 frames unwaited: the ring grows to however far the GPU fell
        // behind); once frames are paced again its chunks are released within two depth
        // windows.
        run(&mut up, 40, 40);
        let grown = up.resident_chunks();
        run(&mut up, 2 * DEPTH_WINDOW + 8, 1);
        let kept = up.resident_chunks();
        assert!(kept <= 2, "{kept} of {grown} chunks kept after the stall");
    }

    /// A one-off burst (e.g. a big asset upload) is released once it is no longer the recent
    /// peak.
    #[test]
    fn idle_staging_chunks_are_released_after_a_burst() {
        let Some((device, queue)) = device_queue() else { return };
        let mut up = Uploader::new(&device);
        let big = vec![3u8; 10 * STAGING_CHUNK as usize];
        let target = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: big.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        frame(&device, &queue, &mut up, &target, &[&big]);
        assert!(up.resident_chunks() >= 10);
        for _ in 0..2 * TRIM_WINDOW + 2 {
            frame(&device, &queue, &mut up, &target, &[&[1u8; 64]]);
        }
        assert!(up.resident_chunks() <= 3, "{}", up.resident_chunks());
    }

    /// Writes split across chunks land intact and in order, and so do concatenated parts
    /// (of any length, empty ones included) that straddle chunk boundaries.
    #[test]
    fn split_writes_arrive_intact() {
        let Some((device, queue)) = device_queue() else { return };
        let mut up = Uploader::new(&device);
        let n = 2 * STAGING_CHUNK as usize + 1036;
        let data: Vec<u8> = (0..n).map(|i| (i * 7 + i / 4096) as u8).collect();
        let size = 2 * (n as u64 + 8);
        let target = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        up.write(&mut encoder, &target, 0, &[9u8; 8]);
        up.write(&mut encoder, &target, 8, &data);
        let k = STAGING_CHUNK as usize + 501;
        let parts: [&[u8]; 5] = [&[5u8; 8], &data[..101], &[], &data[101..k], &data[k..]];
        up.write_parts(&mut encoder, &target, n as u64 + 8, n + 8, parts);
        encoder.copy_buffer_to_buffer(&target, 0, &readback, 0, size);
        up.finish();
        queue.submit([encoder.finish()]);
        up.recall();
        readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = readback.slice(..).get_mapped_range().expect("mapped");
        assert_eq!(&view[..8], &[9u8; 8]);
        assert!(view[8..n + 8] == data[..], "split upload corrupted");
        assert_eq!(&view[n + 8..n + 16], &[5u8; 8]);
        assert!(view[n + 16..] == data[..], "concatenated upload corrupted");
        assert_eq!(up.bytes, size);
    }

    #[test]
    fn needed_flush_keeps_the_rest_pending() {
        let Some((device, queue)) = device_queue() else { return };
        let mut up = Uploader::new(&device);
        let mut t = TrackedBuffer::new("test", wgpu::BufferUsages::VERTEX);
        let mut r = 0;
        t.begin();
        t.write_all(0, 8, [[1u32; 16]; 8].iter(), &mut r);
        let mut encoder = device.create_command_encoder(&Default::default());
        assert_eq!(
            t.flush(&device, &mut encoder, &mut up),
            8 * 64,
            "first flush uploads all"
        );
        // Change elements 1, 2 and 6: two runs, one dirty range each.
        t.begin();
        let mut v = [[1u32; 16]; 8];
        v[1][0] = 2;
        v[2][3] = 2;
        v[6][15] = 2;
        t.write_all(0, 8, v.iter(), &mut r);
        assert_eq!(t.pending(), &[64..192, 384..448]);
        // Only element 2's range is needed now: 64 bytes go up, the rest stays pending.
        let n = t.flush_needed(&device, &mut encoder, &mut up, &[128..256]);
        assert_eq!(n, 64);
        assert_eq!(t.pending(), &[64..128, 384..448]);
        assert_eq!(t.flush(&device, &mut encoder, &mut up), 128);
        assert!(t.pending().is_empty());
        up.finish();
        queue.submit([encoder.finish()]);
        up.recall();
        assert_eq!(r, 1, "one shadow allocation");
    }

    /// Regression: 1.5x growth ignored `max_buffer_size`, so a buffer whose contents fit the
    /// limit could still request more and fail validation (a panic by default).
    #[test]
    fn growth_is_clamped_to_max_buffer_size() {
        let Some(device) = small_device() else { return };
        let mut up = Uploader::new(&device);
        let mut b = GrowBuffer::new("test", wgpu::BufferUsages::VERTEX);
        assert!(b.ensure(&device, 700_000, false, &mut up));
        assert!(b.ensure(&device, 800_000, false, &mut up));
        assert_eq!(b.capacity(), 1 << 20, "1.5x would be 1.05 MB");
        assert!(!b.ensure(&device, 1 << 20, false, &mut up));
    }
}
