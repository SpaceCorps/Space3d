# Post-processing: bloom, auto exposure, color grading, FXAA, lens effects, distortion, city effects

Post effects run in a camera's post group after its scene is drawn, in exposed HDR units, and
end in the one composite pass that writes the display target (see [`frame.md`](frame.md) for
frame modes, post groups and the composite). The normative design is
[`docs/design/graphics.md`](../design/graphics.md) §1.9 and §5.2 (P-POST).

This page covers bloom (milestone PO1, with the M1.1 position-stable prefilter), auto exposure
(PO2), the display-space effects FXAA, vignette, chromatic aberration and film grain (PO3), color
grading with 3D LUTs and WGSL hooks (PO4), screen-space distortion — heat haze, shockwaves —
(PO5) and the hardening of the whole chain: GPU timings, pass and memory counts, allocations,
resizes, device loss and the performance gate (PO6). The city effects (height/distance fog,
SSAO, SSR: opt-in, before bloom) are in "City effects". `examples/post_processing` shows every
effect on a top-down space scene, toggled with keys.

```rust
use space3d::prelude::*; // PostProcess, Bloom
use space3d::render::{FilmGrain, FxaaQuality, Vignette};

// Bloom is on by default; the display-space effects are opt-in.
PostProcess::default()
    .with_fxaa(FxaaQuality::Medium)
    .with_vignette(Vignette { enabled: true, intensity: 0.3, ..Vignette::default() })
    .with_chromatic_aberration(0.15)
    .with_film_grain(FilmGrain { enabled: true, intensity: 0.05, ..FilmGrain::default() });
// The same set: PostProcess::cinematic()
```

Order in a post group (normative, spec §1.9): [Distortion pass after the scene] → [auto-exposure
metering → bloom chain] → composite: the distortion offset → scene (+ chromatic aberration and
distortion split taps) → exposure (manual, or 2^(E + compensation)) → bloom (at the same
offset) → vignette → white balance × saturation, contrast → `post_hdr_hook` → tonemapper →
lift/gamma/gain, LUT → `post_ldr_hook` → **last pass**: grain → dither → output encoding. The
last pass is the composite, or the FXAA pass when FXAA is on (the composite then dithers the
8-bit intermediate it writes: "FXAA").

## Bloom

Bloom is on by default: `PostProcess::default()` carries `Bloom::glow()`.

```rust
use space3d::prelude::*; // PostProcess, Bloom
use space3d::render::BloomComposite;

app.world.spawn((
    Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    Camera3d::default(),
    PostProcess::default().with_bloom(
        Bloom::glow()
            .with_intensity(0.5)       // additive gain
            .with_threshold(1.0, 0.5)  // exposed radiance where bloom starts, knee fraction
            .with_scatter(0.8),        // 0 tight glow … 1 wide haze
    ),
));
// Or: PostProcess::default().without_bloom()
```

What blooms is exposed radiance above the threshold: emissive materials, bright lights on shiny
surfaces, unlit colors above 1, particles and `HdrScene` overlays. A material's `emissive` of 8
at exposure 0 glows; the same at `with_exposure(-3.0)` (exposed 1.0) does not.

| `Bloom` field | default (`glow()`) | meaning |
|---|---|---|
| `enabled` | `true` | |
| `intensity` | 0.35 | additive: bloom × intensity is added; energy conserving: mix factor 0..1 |
| `threshold` | 1.0 | exposed radiance (brightest channel) where bloom starts; 0 = everything blooms |
| `soft_knee` | 0.5 | knee width as a fraction of `threshold` on either side (0 = hard cut) |
| `scatter` | 0.7 | how much the coarse levels weigh: level k gets (1 − s)·s^k, the last one the rest |
| `composite` | `Additive` | `Additive` (adds light: glow) or `EnergyConserving` (`mix(scene, bloom, intensity)`: veil) |
| `mip_count` | 6 | levels of the reference chain, 1..=8 (see "Resolution independence"); fewer when the last level would drop below 2 px |
| `max_mip_height` | 540 | height of the reference chain's level 0: a rect of twice this height or more gets half its size scaled down to it (the cost stays that of 1080p at 1440p and 4K); smaller rects keep the reference heights |

Presets: `Bloom::glow()` (game emissives: the default) and `Bloom::natural()` (threshold 0,
energy conserving 0.15, scatter 0.85: a photographic veil that keeps the image's mean
brightness). `PostProcess::with_quality(GraphicsQuality)` sets `mip_count`/`max_mip_height`
(Low 4 / 270, Medium 6 / 540, High 7 / 720). `PostProcess::with_bloom(b)` sets `b` as given,
`enabled` included (`Bloom { enabled: false, .. }` stays off; the presets are enabled).

### How it works

Per post group (camera rect), after its scene views and `HdrScene` overlays, before the
composite:

1. **Prefilter** (scene → exactly half its size): every scene pixel is multiplied by the
   camera's exposure (with auto exposure, 2^(E + compensation)), soft-thresholded on its
   brightest channel (a quadratic knee, continuous at `threshold ± knee`, so hue is kept) and
   softly limited — unchanged up to 32, compressed toward 64 above (the **firefly guard**: a
   stray specular spike of 1000 glows like a pixel of about 63) — then a separable
   [1 2 1] / 4 tent over 3 × 3 pixels filters them. Pixels outside the post group's rect
   weigh nothing: split screens and picture-in-picture views never bleed into each other.
2. **Downsample** level i → i + 1 (`mip_count − 1` passes): 13 bilinear taps (Jimenez 2014),
   plain weights.
3. **Upsample** level i + 1 → i (`mip_count − 1` passes): a 3×3 tent, blended into level i as
   `src × scatter + dst × (1 − scatter)`. Level 0 ends up holding Σ (1 − s)·s^k · level_k.
4. **Composite:** level 0, magnified by a cubic B-spline (4 bilinear taps), is added after
   exposure (bloom is already exposed: never exposed twice) and before vignette, grading and
   the tonemapper: `hdr + bloom × intensity`, or `mix(hdr, bloom, intensity)` when energy
   conserving.

Every pass is a full-screen triangle with explicit-LOD sampling only (no derivatives), and each
reads one level view while writing another, which is portable to WebGL2.

The chain is a normalized weighted average: without a threshold its mean equals the scene's, so
`intensity` is the only gain (tested: energy-conserving bloom keeps the mean within 2 %).

### Stable under motion, proportional to energy (M1.1)

The prefilter's only non-linear steps (exposure, threshold, firefly limit) act on each pixel on
its own, and everything after them is linear: an even pixel weighs 1/2 in one level-0 texel per
axis, an odd one 1/4 in two, so every pixel's total weight is exactly 1/4 wherever it sits. The
composite then magnifies level 0 with a cubic B-spline, whose weights (unlike bilinear ones at a
non-integer ratio such as 720p's and 1440p's 2.67×) give every texel the same share of the frame.
Two consequences, both measured on level 0 read back in float and the composite's lookup
reproduced on the CPU (`post::bloom::tests`):

* **No shimmer.** A small bright source moving over the pixel grid adds the same light at every
  position: the spread over all phases is 0.02–0.1 % at 720p, 1080p and 1440p for sources of
  1–4 px (`bloom_is_stable_under_motion`, limit 0.3 %). M1 thresholded and Karis-weighted 2 × 2
  box averages of bilinear taps: a 2 px source's added light varied by 11 % and a 4 px source's
  by 4 % as it moved (the M1 review measured up to 16 % at 720p); stars, bolts and sparks pulsed
  while the camera panned.
* **Glow in proportion to energy.** The light bloom adds is the energy above the threshold ×
  `intensity`, whatever the source's size: 1.00 × for 1–32 px sources
  (`bloom_energy_is_proportional_and_limited`). M1 gave 1–2 px sources a third of it, 4 px ones
  43 % and 8 px ones 70 %, so small emissives (stars, sparks, laser heads, particles) now glow
  about 1.5–3× more than they did, and large ones the same. If a dense starfield now veils the
  image, keep its stars below the threshold (or the threshold above them) rather than lowering
  `intensity`, which also dims engines and lasers.

Karis averaging (M1's firefly guard) normalizes a pixel against its neighbours, which is exactly
what made the result depend on position; the per-pixel limit replaces it. A destination texel
of the [1 2 1] tent is centered on a scene pixel, half a pixel before the 2 × 2 block it stands
for; the composite looks bloom up half a pixel further. (A [1 3 3 1] tent over 4 × 4 pixels is as
stable but costs twice as much at 4K; a 2 × 2 box half as much but leaves blocky glows along
thin lines such as lasers.)

Cost, same GPU, loaded, A/B against the M1 prefilter in one session (`bloom_gpu_time_by_size`):
the same at 720p (0.13 ms) and 1080p (0.15 ms), 0.19 against 0.16 ms at 1440p and 0.29
against 0.20 ms at 4K — the prefilter loads 9 scene pixels per texel where M1 took 13 bilinear
taps, which matters where the scene texture is large. A compute prefilter (each pixel loaded
once through workgroup memory) is the follow-up with the compute chain (see "Cost"). The
composite's B-spline lookup takes 4 bilinear taps of level 0 instead of 1.

### The low tiers' cheap filters (`Bloom::fast`)

On an integrated GPU those two exact filters are a visible share of the frame: at 1080p on an
AMD Radeon iGPU (Vulkan) the 9-load prefilter is about 0.6 ms of the chain's 1.65 ms and the
B-spline lookup 0.15 to 0.35 ms of the composite's 1.9 ms. `Bloom::fast` (off by default;
`PostProcess::with_quality` sets it for Low and Medium and clears it for High and Ultra) trades
them:

* the prefilter takes 4 bilinear taps half a pixel around scene pixel 2i (the same
  [1 2 1] / 4 tent per axis) and applies exposure, threshold and firefly limit once, to their
  mean. A source of 1–2 px is averaged with its dark neighbours before the threshold: it glows
  less, and by an amount that depends on where it sits on the pixel grid (measured in
  `the_fast_prefilter_keeps_large_sources_and_dims_small_ones`: a 32 px source keeps its glow,
  smaller ones lose part of it, none gains). At the rect's edge the taps repeat the edge pixels;
* the composite reads level 0 with one bilinear tap: at 720p and 1440p (2.67×) texels weigh up
  to 3 % apart, so a small moving glow can pulse by that much; at 1080p level 0 is exactly half
  size and the difference is smallest.

Measured on that iGPU at Medium, 1080p (interleaved pairs against the build before): the
chain 1.65 → 1.05 ms, the composite 1.88 → 1.55 ms in an empty scene and 1.92 → 1.77 ms in a
busy one. On a discrete GPU both passes are under 0.1 ms either way.

(Tried and dropped in the same pass: FXAA reading its luma from the LDR intermediate's alpha,
with the centre and its four neighbours from two `textureGather` calls. On the same iGPU the
FXAA pass went from 1.83 to 2.11 ms in an empty scene and from 1.93 to 2.33 ms in a busy one,
two pairs each: gathers are no cheaper than bilinear fetches there.)

### Resolution independence

A level's blur spans a fixed number of its own texels, so the halo keeps its size relative to
the frame only if the level heights do not depend on the rect. The chain is therefore defined
by a **reference chain** of `mip_count` levels of heights `max_mip_height / 2^k` — what a 1080p
rect gets with the defaults (540, 270, … 17 px). A smaller rect keeps those heights: it drops the
`n` finest reference levels, where `2^n` is the power of two nearest to
`max_mip_height / (rect height / 2)`, and its level 0 (`max_mip_height / 2^n` high, within √2 of
half the rect, but **never above half the rect**: a larger level 0 would only be resampled up
from the prefilter's half-resolution output, a bigger target and a pass for no detail — High's
720 cap gave a 1280 × 720 level 0 at 1080p, more passes and memory than Ultra) stands for them; their weights merge into it
(the upsample into level 0 blends with `scatter^(n + 1)`), so every level keeps its reference
weight. The halo's size and energy relative to the frame are the same from 180p to 4K (only the
tight glow of the merged levels widens), and smaller rects cost fewer passes.

**The prefilter always halves the rect.** Its fixed 3 × 3 footprint weighs every scene pixel
alike only at exactly 2× reduction. Level 0 at the reference
height is 2.67× below the rect at 720p, 1440p and 360p and 4× at 4K: there the taps left gaps,
and a 1–2 px bright source (a star, spark, bolt head or running light) got a full halo, a double
one or none depending on its sub-pixel position, so a moving one blinked (the light bloom added
to a moving 1–2 px source went from none to 2.6× its mean at 720p, 1440p, 360p and 4K, within
1 % at 1080p). So the prefilter writes exactly half the rect (rounded up) and, when level 0 has
another size, an area resample makes level 0 from it: each level-0 pixel averages the
half-resolution pixels under its box, fractional ones by their overlap (2 × 2 bilinear taps per
pixel up to a 3-pixel box), so every pixel keeps its energy wherever it sits
(`post::bloom::tests::resample_keeps_every_pixel_energy`; on the GPU,
`bloom_halo_is_independent_of_position` in `tests/post.rs` and `tests/gl_post.rs`: a 1 px
source's halo stays within 5 % over 12 positions at every ratio, 0.5–3 % measured). It costs
one more render pass, a pooled texture (below) and GPU time where the half rect is larger than
level 0: nothing at 1080p (no resample), +4 % at 1440p, +25 % at the `GraphicsQuality::Low` rect
of T3 (1440 × 810: the prefilter writes 720 × 405 instead of 480 × 270) and +40 % at 4K (1920 ×
1080 instead of 960 × 540) — best-of chain times 0.152 → 0.158, 0.081 → 0.101 and 0.145 →
0.204 ms against the pre-change chain (M1 review round 3, interleaved sweeps; "Cost"), plus about
7 % more render-thread CPU for the extra pass at 1440p. That is the price of a halo that does not
blink.

| rect | chain (level 0, levels; prefilter target) | bloom-added mean luma, `Engine/Post/Bloom` (M1.1; M1; before the half-rect prefilter; before the reference heights) |
|---|---|---|
| 3840 × 2160 | 960 × 540, 6; 1920 × 1080 | +8.2 (+7.8; +7.6) |
| 2560 × 1440 | 960 × 540, 6; 1280 × 720 | +8.1 (+7.8; +7.7) |
| 1920 × 1080 | 960 × 540, 6; level 0 | +8.1 (+7.6; +7.6; +7.6) |
| 1600 × 900 | 960 × 540, 6; 800 × 450 | +8.2 (+7.6; +7.7) |
| 1280 × 720 | 480 × 270, 5; 640 × 360 | +8.2 (+7.4; +7.2; +9.3) |
| 640 × 360 | 240 × 135, 4; 320 × 180 | +8.1 (+7.0; +6.6; +11.7) |
| 320 × 180 (the goldens) | 121 × 68, 3; 160 × 90 | +8.5 (+6.7; +6.0; +14.1) |

With the per-pixel prefilter the added light is the same at every size within 5 %
(`engine::post::tests::bloom_added_luma_by_size` in space3d-stories, ignored: a measurement).
M1's fell below 1080p because its box averages thresholded small bright sources (stars), which
cover fewer scene pixels at lower resolutions, more harshly. Before the reference heights,
level 0 was simply half the rect with 6 levels at any size, so a 360p frame (a phone,
`render_scale` 0.5) got 1.5× and the 180p goldens about 2× the desktop veil.

### What gets bloomed

| drawn by | bloomed |
|---|---|
| scene views of the post camera (meshes, render nodes, particles) | yes |
| `OverlayStage::HdrScene` overlays | yes (they draw into the scene color before post) |
| `OverlayStage::AfterTonemap` overlays (egui, 249c874-era overlays) | no |
| display views (a later camera without post on the rect: gizmos, 3D HUD) | no |
| another post group (split screen, picture in picture) | no (taps clamp to the rect) |

A picture-in-picture post camera inside a full-screen post camera: the full group's bloom may
pick up PiP pixels at the PiP border (its scene texture holds them); the PiP's own group
overwrites its rect.

**Render targets with a transparent clear** (e.g. a ship preview shown in egui): the composite
un-premultiplies, adds bloom and re-multiplies by the scene alpha, so bloom is weighted by alpha
and no haze appears where nothing was drawn (alpha 0 stays exactly 0).

### Chain format, memory and platforms

| backend class (spec §4.1) | chain |
|---|---|
| A: Metal, Vulkan, DX12, WebGPU with `RG11B10UFLOAT_RENDERABLE` | `Rg11b10Ufloat` (4 B/px) for `Additive`; `Rgba16Float` for `EnergyConserving` |
| A without it; B: WebGL2 / GLES 3.0 with `EXT_color_buffer_float` (ANGLE) | `Rgba16Float` (8 B/px) |
| C: no float render targets (the Ldr fallback) | none: bloom is off (warned once) |

`Rg11b10Ufloat` has 6 mantissa bits (5 in blue) and converts by truncation on some GPUs (Apple):
over the chain's writes the additive glow loses about 5 % of its energy (7 % of blue), which
`intensity` absorbs. The energy-conserving mode promises the scene's mean, so its chain is
`Rgba16Float`. `HdrMode::Ldr` (or no float targets) turns bloom off: the LDR scene texture
holds tonemapped values.

The chain is one texture from the frame's pool (per-level views created once), plus the
prefilter's half-resolution target where level 0 is resampled from it. Post groups of the same
size in one frame share them (split screens cost one chain), and they are freed after 3 unused
frames. `RenderStats::target_bytes` counts them once per frame.

| rect | levels | `Rg11b10Ufloat` (chain + half-resolution target) | `Rgba16Float` |
|---|---|---|---|
| 1920 × 1080 (level 0 960 × 540) | 6 | 2.8 MB | 5.5 MB |
| 3840 × 2160 (capped to 960 × 540; half 1920 × 1080) | 6 | 2.8 + 8.3 = 11.1 MB | 22.1 MB |
| 2560 × 1440 (960 × 540; half 1280 × 720) | 6 | 2.8 + 3.7 = 6.5 MB | 12.9 MB |
| 1280 × 720 (480 × 270, the 540 level merged; half 640 × 360) | 5 | 0.7 + 0.9 = 1.6 MB | 3.2 MB |
| 1440 × 810, `GraphicsQuality::Low` (480 × 270; half 720 × 405) | 4 | 0.7 + 1.2 = 1.9 MB | 3.7 MB |

**Driver memory per render pass (Metal).** `target_bytes` counts textures only. Each of the
chain's render passes also holds GPU driver memory (IOAccelerator) while its frame is in flight —
on Metal about 2.8 MB per pass per frame in flight, whatever the attachment's size: the default
chain's 11–12 passes add about 60 MB to the process at 1080p with one frame in flight and 148 MB
with three (a window keeps two to three), per post group (split screen doubles it), against 2.7 MB
of `target_bytes` (`frame.md` "Cost"; `tests/footprint.rs` checks it). Budget it in addition to
the table above (spec §4.4's VRAM line counts textures only). Compute passes cost about 0.5 MB
each: a compute chain (one compute pass of dependent dispatches, below) would hold one pass's
worth.

### Cost

Render passes per post group: `2 × levels − 1`, one more for the resample where level 0 is
smaller than half the rect (11 for the default 6 levels at 1080p, 12 at 1440p and 4K, 10 at
720p, 6 at 180p; High's 7 levels 13 and Ultra's 8 levels 15 at 1080p) plus the composite. Their uniforms (one 64-byte slot per pass) are written when the settings or
the rect change, and the per-frame seeds of grain and dither a
window of 64 frames at a time: a still scene uploads nothing except that window (16 KiB per post
group, every 64th frame). The steady state creates no textures and reallocates nothing
(`tests/post.rs`, `tests/alloc_post.rs`); wgpu-hal on Metal allocates about 20 times per render
pass (234 allocations per frame with bloom at 320 × 180, 6 bloom passes, vs 106 for the
composite alone).

GPU time on the dev machine (Apple M4 Max, Metal; `bloom_gate_1080p` below): the GPU kept busy
(command buffers of 5 chains submitted back to back after a warm-up), each timed from the start
of its first pass to the end of its last, best of 40:

| chain | GPU (loaded) |
|---|---|
| 6 levels, 1080p (level 0 960 × 540), `Rg11b10Ufloat` | 0.14–0.17 ms |
| the same, `Rgba16Float` | 0.14–0.15 ms |
| 6 levels over a 128 × 128 scene (almost no pixels) | 0.15 ms |
| the prefilter alone (1 level), 1080p | 0.07 ms |
| `GraphicsQuality::Low` settings at 1080p (4 levels, level 0 480 × 270) | 0.08 ms |
| `GraphicsQuality::High` (7 levels; level 0 still 960 × 540 at 1080p) | 0.17 ms |

In real frames (`frame/ab`'s bloom sweep: the `post.bloom` scope of `GpuTimings`, medians of 48
frames back to back, machine load ≈ 30, so above the best-of figures): 0.196 ms at 720p (11
passes with the composite), 0.227 ms at 1080p, 0.254 ms at 1440p, 0.607 ms at 4K (the prefilter
reads the whole 4K scene) and 0.122 ms for the Low chain at 1440 × 810.

On this GPU the chain costs its 11 dependent render passes (about 14 µs each), not its pixels:
a 128² chain costs as much as a 1080p one, and the two chain formats cost the same. When
frames are isolated (the GPU idles between them and drops to low clocks) the same chain
measures about 0.45 ms; as for the composite (`frame.md` "Cost"), the gate uses the loaded
regime.

CPU (release build, same machine): bloom adds 0.15–0.17 ms of render-thread CPU per post group
over the composite alone at 1080p (the M1 record in `docs/performance.md`; an earlier estimate
here said 0.10 ms, measured on a machine busy with other builds). That is 11 render passes at
roughly 14 µs of wgpu/Metal pass overhead each (the renderer's own work per pass is a few map
lookups and allocates nothing); FXAA adds one more pass, about 0.01 ms. It is above the spec's
§4.4 line "post ≤ 0.05 ms" of the Battle CPU budget: **not met**. A compute chain (one compute
pass of dependent dispatches on class A backends, the render-pass chain kept for WebGL2) would
cut the CPU, the GPU pass overhead and the per-pass driver memory, and is the planned follow-up
(not in M2).

### Small views: UI previews and thumbnails

Every render-target camera with a `PostProcess` is a post group of its own, and a chain costs
its passes, not its pixels (above): a 320 × 192 preview pays about 9 bloom passes plus its
composite, nearly what the main view pays, for a glow a few pixels wide. Four such previews
next to a 1280 × 720 main view made the frame 47 render passes against 11 without them
(SpaceCorps2027's shop, game#15; the Battle's two 256² previews would add 22 passes and put its
CPU over the 1.5 ms gate, so they render without post). For views shorter than about 240 px:

* turn bloom off on them (`PostProcess::default().without_bloom()`), or keep one
  `PostProcess` for every camera and set `Bloom::min_rect_height`
  (`Bloom::glow().with_min_rect_height(240)`): groups whose rect (scene-texture pixels, after
  the render scale) is shorter get no chain and pay the composite only
  (`tests/post.rs::bloom_skips_small_rects`);
* or render them without `PostProcess` at all (Direct mode: no scene texture, no composite).

`min_rect_height` defaults to 0 (every group blooms), so nothing changes for existing cameras.
Level 0 is never larger than half the rect, so small groups also skip the resample pass they
used to pay (a 320 × 192 group: 8 passes where it had 9).

### Gate

Spec §4.4: "bloom 6 mips @1080p ≤ 0.20 ms" on the dev machine: **met** (0.14–0.17 ms). The test
`post::bloom::tests::bloom_gate_1080p` measures it; it is ignored by default because the other
GPU tests of the same binary run in parallel with it, so run it alone:

```sh
cargo test -p space3d-render --lib bloom_gate -- --ignored --nocapture
```

`SPACE3D_PERF_BUDGET_SCALE` relaxes the budget on slower machines; the test skips without
`TIMESTAMP_QUERY`. Each run takes the best of 40 rounds, and a run disturbed by other GPU work
is retried (best of 3 runs). The gate is only meaningful on an idle Dev GPU: at the M1 review,
with other apps holding the shared GPU at 50–60 % device utilization, runs measured 0.188–0.265 ms
and one of two invocations passed; on a quieter GPU the review measured 0.13–0.16 ms. It is not
part of `cargo test --workspace` or CI (hosted runners have software or paravirtual GPUs).

## Auto exposure

```rust
use space3d::prelude::*; // PostProcess
use space3d::render::{AutoExposure, Metering};

PostProcess::default()
    .with_auto_exposure(AutoExposure {
        min_ev: -4.0,               // darken a bright scene by 4 stops at most
        max_ev: 4.0,                // brighten a dark one by 4 at most
        metering: Metering::CenterWeighted,
        ..AutoExposure::on()
    })
    .with_exposure(-1.0);           // compensation: a darker key for space
// A cut or teleport: renderer.reset_post_history(camera) snaps on the next frame.
```

Each post group meters **its own rect** of the scene color (split screens and picture in
picture keep their own history) and writes the camera's 1 × 1 exposure texture `E` (stops); the
bloom prefilter and the composite multiply by 2^(E + `PostProcess::exposure`), so bloom's
threshold follows the metered exposure. The metered value maps the scene's log-average
luminance to middle grey (0.18) and is clamped to `min_ev..=max_ev`:

* **Black is not metered** (luminance below 2⁻¹⁶): a starfield on black is exposed for its ships,
  planets and nebulae, not brightened for the void. A frame with nothing metered holds the
  exposure.
* **A skybox is metered.** Only black is skipped: an HDR sky (a nebula cubemap) is scene
  luminance like any other, and where it covers most of the frame it sets the exposure. The
  Battle with `AutoExposure::on()` and no compensation meters its dim nebula to middle grey:
  mean luma 107.6 at 720p against 59.1 for its authored manual look (the hulls wash out pastel,
  the nebula turns milky; 77.4 with compensation −1). For sky scenes keep the manual exposure,
  or give auto exposure a negative compensation (−1 to −1.5 for a dim nebula) and a low
  `max_ev` (for example 1) so a frame of mostly sky cannot brighten past the authored look.
  Weighting the sky down needs scene depth in the metering, which post groups do not read yet:
  `AutoExposure` has no sky weight in this version (deferred).
* **Percentile window** (compute path): the darkest `low_percentile` (10 %) and the brightest
  `1 − high_percentile` (10 %) of the metered weight are cut, so a small very bright light or a
  deep shadow does not swing the exposure.
* **Metering:** `CenterWeighted` (default) weighs the center 4× the corners, linearly in the
  distance; `Average` weighs every pixel alike.
* **Adaptation:** each frame closes `1 − 2^(−dt · rate)` of the gap to the target, with
  `adapt_to_bright` (3/s: 7/8 of the gap in a second) when the scene got brighter and
  `adapt_to_dark` (1/s: half the gap in a second) when it got darker. `dt` is
  `FrameInfo::delta`: 0 when `Time::frame` did not advance (`render_to_image` twice, capture
  replays), so exposure is a pure function of the frames rendered (deterministic under
  `App::step_ticks`). `AutoExposure::instant()` (rates ∞) converges in the frame, even without
  time.
* **Snap:** a camera's first metered frame goes straight to the target (a one-shot capture is
  exposed correctly: headless rule 4 of the spec), as do the frame after auto exposure was turned
  on, after `Renderer::reset_post_history(camera)` and after a path change. Histories are freed
  60 frames after a camera stops being metered.
* **Readback:** with `FrameSettings::read_exposure`, `Renderer::camera_exposure(camera)` returns
  the exposure applied (metered + compensation, stops), 1–3 frames late and without ever waiting
  for the GPU; cameras without auto exposure report their manual exposure.
* Auto exposure needs Hdr mode: in the Ldr fallback (no float targets, `HdrMode::Ldr`) it is off
  (warned once) and the manual exposure applies. The passthrough group never meters.

### How it is metered

| path | when | passes (1080p) |
|---|---|---|
| **compute histogram** | `GraphicsCaps::compute_histogram` and an `R32Float` E (Metal, Vulkan, DX12, WebGPU; GLES 3.1+) | 1 compute pass (2 dispatches) + a 4-byte copy into E |
| **render passes** | otherwise (WebGL2, GLES 3.0, `DownlevelOverride::no_compute`) | 4 (log-luminance level, 2 reductions, adapt) |

* **Compute** (`shaders/post/exposure.wgsl`): 16 × 16 workgroups sample the rect every `stride`
  pixels per axis (1 up to 655 k samples, 2 at 1080p, 4 at 1440p and 4K). Each metered sample
  adds its weight (16 × the metering weight, an integer) to one of 255 log2-luminance bins over
  [−16, 16] (1/8 EV each) and its weight × its position inside the bin (6 bits) to a second
  counter, so a bin's samples count at their mean position, not the bin's center (no 1/8 EV
  steps as a scene brightens slowly). A second dispatch of 256 threads prefix-sums the bins,
  takes the part of each bin inside the percentile window, averages, clamps and adapts, writes
  the camera's 16-byte state and clears the histogram for the next group. Integer atomics and a
  fixed-order float reduction: bit-for-bit deterministic.
* **Render passes** (`shaders/post/exposure_dl.wgsl`, texel loads only): the rect → a level a
  quarter of its size holding Σ weight × log2 L and Σ weight of each 4 × 4 block (the same samples
  as the compute path), reductions by 8 × 8 while larger than 16 × 16, then an adapt pass that
  sums the last level and writes the camera's other E texture from the previous one (a ping-pong
  pair). No percentile window (it needs a histogram): the plain weighted log-average.

Accuracy against an exact reimplementation of the rule (every pixel, no bins,
`reference_ev` in `tests/common/post.rs`) on the test scene: the compute path within 0.01 EV
(spec: ±0.1), the render-pass path within 0.2 EV of the windowed rule (spec: ±0.5) and 0.002 EV
of the unwindowed one it computes. E is `R32Float` where renderable, else `R16Float` (about
0.004 EV of precision near ±8 EV).

Cost (Apple M4 Max, loaded, 1080p rect, `post::exposure::tests::exposure_gpu_time`): compute
0.008 ms of GPU, render passes 0.14 ms (mostly the passes' fixed overhead); spec §4.4 budgets
0.05 ms on T1. Memory: E (4 bytes, two on the render-pass path) and a 16-byte state per camera,
one shared 2 KiB histogram; the render-pass levels come from the frame's pool (1.05 MB of
`Rgba16Float` at 1080p, shared by same-size groups, counted in `RenderStats::target_bytes`). Uploads: the group's 80-byte parameters
when the settings or the rect change; per frame only the 16 bytes of adaptation factors when
`dt` changes (never with a fixed tick rate) — the steady state creates, reallocates and uploads
nothing (`auto_exposure_history_and_steady_state`, both paths).

## Color grading

```rust
use space3d::prelude::*; // PostProcess
use space3d::render::{ColorGrading, ColorLut, RenderAssets};
use glam::Vec3;

// A .cube LUT exported from Resolve, Photoshop, ... (display-referred, sRGB in and out).
let strip = ColorLut::from_cube(&std::fs::read_to_string("grade.cube")?)?;
let lut = app.resources.expect_mut::<RenderAssets>().add_texture(strip);
PostProcess::default().with_color_grading(ColorGrading {
    temperature: 0.15,                 // warmer
    saturation: 1.1,
    contrast: 1.1,                     // log2, around middle grey
    lift: Vec3::new(0.0, 0.01, 0.03),  // blue-ish blacks
    lut: Some(ColorLut { contribution: 0.8, ..ColorLut::new(lut, 33) }),
    ..ColorGrading::default()
});
```

Grading is fused into the composite (no extra pass) and costs nothing while neutral
(`ColorGrading::default()`; each part has its own effect bit). Order, and the space each step
works in:

| step | space | formula |
|---|---|---|
| white balance (`temperature`, `tint`, −1..1) | linear HDR, before the tonemapper | von Kries in LMS: the D65 white moved along the daylight locus (temperature; + is warmer) and across it (tint; + is magenta) |
| `saturation` | linear HDR | `luma + s · (c − luma)` (Rec. 709 luma; 0 grey, > 1 more saturated) |
| `contrast` | linear HDR | `0.18 · (c / 0.18)^contrast` per channel (log2, pivoting on middle grey) |
| `post_hdr_hook(c, uv)` | linear HDR | the hook shader's, if any |
| tonemapper | | |
| `gain`, `lift`, `gamma` | sRGB-encoded display color | ASC CDL: `(e · gain + lift)^(1 / gamma)` (gamma > 1 brightens the midtones) |
| LUT | sRGB-encoded display color | 3D LUT lookup, blended by `contribution` |
| `post_ldr_hook(c, uv)` | display-linear, 0..1 | the hook shader's, if any |

White balance and saturation are one 3 × 3 matrix (built on the CPU), so the HDR part costs a
matrix multiply and, with contrast, a `pow`. In the Ldr fallback everything runs on the already
tonemapped display colors (no tonemapper in the composite there).

**3D LUTs.** A `ColorLut` is an N²×N strip texture: blue slice `b` at x ∈ [b·N, (b + 1)·N), red
along x inside the slice, green along y; the composite indexes it with the encoded color and
blends two bilinear taps between the neighbouring blue slices. `ColorLut::identity(n)` gives a
neutral strip (`Rgba8Unorm`) to export, grade in an image editor and load back
(`RenderAssets::add_image`: an sRGB-format strip is re-encoded after sampling, so it behaves like
a linear one). `ColorLut::from_cube(text)` parses Adobe/Resolve `.cube` 3D LUTs (`LUT_3D_SIZE`
2..=64, `TITLE`, comments, `DOMAIN_MIN`/`DOMAIN_MAX` or `LUT_3D_INPUT_RANGE` — resampled onto
0..1 — and unknown keywords ignored; 1D LUTs are rejected with `LutError::Parse`) into an
`Rgba16Float` strip. A strip whose size is not `size² × size`, or an array texture, is ignored
(warned once). Accuracy: an identity strip changes nothing within 1/255 (`.cube`, 16-bit) or
2/255 (8-bit strips), 3/255 as an sRGB PNG (`lut_identity_is_neutral`).

**WGSL hooks.** `ColorGrading::shader` takes a `CustomShader` (registered with
`RenderAssets::add_shader`, hot-reloadable with `update_shader`) that defines `post_hdr_hook`
and/or `post_ldr_hook`, both `fn(c: vec3<f32>, uv: vec2<f32>) -> vec3<f32>` with `uv` 0..1 over
the camera's rect; missing ones are identity. The hook may use the composite's helpers
(`s3d_luma`, `s3d_linear_to_srgb`, `s3d_srgb_to_linear`, `s3d_tonemap`, …) but not declare
resources, workgroup memory or pipeline constants without a default. The composite with the hook
is validated with naga and built inside validation and internal error scopes; one that fails —
at once, or later where the scopes answer asynchronously (browser WebGPU) — grades without its
hooks, logs the error once and reports it through `Renderer::shader_error(handle)`
(`invalid_grading_hook_falls_back`). Hook pipelines are cached per (output format, hook
version); a new version compiles on first use (prewarm by rendering a frame with it on a loading
screen).

```rust
// Night vision: green monochrome with scanlines (the Engine/Post/Color Grading story).
let hook = assets.add_shader(CustomShader::new("night vision", r#"
fn post_ldr_hook(c: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let l = s3d_luma(c);
    let lines = 0.85 + 0.15 * step(0.5, fract(uv.y * 90.0));
    return vec3<f32>(0.15, 1.0, 0.25) * pow(l, 0.8) * 1.4 * lines;
}"#));
PostProcess::default().with_color_grading(ColorGrading { shader: Some(hook), ..Default::default() });
```

### Display gamma: a brightness setting for players

```rust
PostProcess::default().with_display_gamma(1.2) // shadows and midtones brighter; 1.0 = off
```

`PostProcess::display_gamma` raises the sRGB-encoded display color to `1 / display_gamma` as the
last step of the grading (after lift/gamma/gain, the LUT and `post_ldr_hook`; before FXAA and the
finish, so the anti-aliasing and the dither see the picture that is shown). Black and white stay
where they are; 0.5 encoded becomes 0.561 at 1.2 and 0.420 at 0.8. It is the control to give a
player as "Brightness":

* **Not exposure.** Exposure is applied before the bloom threshold and the tonemapper, so it
  changes how much of the scene blooms and where highlights roll off. The display gamma is
  applied to what the tonemapper produced: the glow, the highlights and the grading are the same
  at every setting.
* **Free at 1.** It is a uniform bit (`S3D_POST_DISPLAY_GAMMA`) and one float in the group's
  parameters: at 1 (the default) the bit is not set and the composite takes the path it took
  before the field existed (`display_gamma_is_off_at_one`; `display_gamma_matches_formula` only
  shows that setting 1 explicitly equals the default). That the picture did not change was
  checked once across commits: four composites without FXAA (the default, no effects, bloom
  without dither, graded; a dark ramp, a mid quad and a blooming quad at 256 × 128, DX12)
  rendered on the commit before the field and on the one with it are the same bytes. With FXAA
  the default picture differs on purpose (the intermediate's dither, above). Dragging a slider
  builds no pipeline. Moved, it costs one sRGB encode, a `pow`
  and a decode per pixel in the composite. One pair of runs on an AMD Radeon integrated GPU at
  1920 × 1080 (SpaceCorps, Medium, a busy PC): composite 2.21 ms at 1 against 2.61 ms at 1.3,
  composite + FXAA 3.54 against 4.13 ms, the whole GPU frame 7.51 against 7.98 ms. So roughly
  0.4 to 0.6 ms of post time there, n = 1: the other pair (frame p50 24 ms, another program on
  the GPU) pointed the other way (4.21 + 3.12 at 1, 3.71 + 2.33 at 1.3), and the FXAA pass does
  nothing different at 1.3, so part of the 0.6 is spread. Measure again on a quiet PC before
  quoting it; not measured on a discrete GPU. It is not fused with the finish's own encode: with FXAA the two are in
  different passes.
* **The scene only.** It belongs to a camera's post settings: UI drawn after post (egui, overlays
  after the tonemapper) is not touched, and a camera without it (a preview in a UI panel) keeps
  the UI's brightness. In Direct mode (no post) there is nothing to apply it, as with every other
  post effect; the Ldr fallback applies it.
* **Calibration UI.** `apply_display_gamma(encoded, gamma)` is the CPU mirror: paint a few dark
  patches (codes 4, 8, 16 on black) with it beside the slider and ask for the darkest to be barely
  visible. They are 1, 2, 5 at 0.7 and 16, 25, 40 at 1.5 (`display_gamma_golden_values`).

Values outside `DISPLAY_GAMMA_RANGE` (0.25 to 4) are clamped, NaN and infinities count as 1.
The step works on display colors in 0 to 1 and clamps to that range first, as the LDR grading
does: on a float output with `Tonemapper::None`, values above 1 pass through at 1 and are
clipped at any other setting (no test covers that combination). A
lifted gamma shows what 8 bits hold in the shadows (an 8-bit sky texture's own steps become two
or three codes apart at 1.5); the dither covers the composite's rounding, not the source's.

## FXAA

`PostProcess::with_fxaa(FxaaQuality)` (or `anti_aliasing: AntiAliasing::Fxaa(q)`): FXAA 3.11
(Timothy Lottes), the PC quality path, after tonemapping. The composite then writes an 8-bit
intermediate `L` (`Rgba8UnormSrgb`, the target's size, from the frame's texture pool: post groups
of one size share it) instead of the target, and one more full-screen pass reads `L`, finds
edges by local contrast (luma = √(Rec. 601 luma) of the display-linear color, about perceptual),
blends along them and writes the target — with grain, dither and the output encoding applied
there, after the anti-aliasing, so FXAA never smears the grain.

| `FxaaQuality` | Lottes preset | search steps | edge threshold | sub-pixel |
|---|---|---|---|---|
| `Low` | 10 | 3 | 1/4 | 0.75 |
| `Medium` (default) | 12 | 5 | 1/6 | 0.75 |
| `High` | 29 | 12 | 1/8 | 0.75 |
| `Ultra` | 39 | 12 | 1/8 | 1.0 |

Every tap is clamped to the post group's rect (split screens never bleed into each other), the
pass uses explicit-LOD sampling only (WebGL2), and it runs in the Ldr fallback too. Render
targets with a transparent clear keep their alpha: FXAA filters premultiplied colors and alpha
together, and grain and dither are weighted by alpha. `PostProcess::with_quality` sets the
FXAA quality to the tier's (Low, Medium, High) when FXAA is on; the Low tier (WebGL2) uses FXAA
Low instead of MSAA (spec §4.4). Cost: one pass, and `L` (8.3 MB at 1080p).

**`L` is dithered.** `L` has 8 bits, so the composite dithers what it writes there (triangular,
±1 LSB in the encoded space, `PostProcess::dither`, with its own hash of the frame's seeds:
`post_dither_intermediate`). Until 2026-11 it did not: the composite rounded a smooth dark
gradient into bands, and the last pass's dither only added noise to values that were whole codes
already, so every tier with FXAA (Low and Medium, and the higher ones without 4× MSAA) showed
the bands the tiers without it do not. Where FXAA finds no edge it copies the pixel of `L`, and
the FXAA pass leaves that pixel as it is (no second dither: it would add nothing but noise, σ 0.5
to 0.7 codes); where it blends, the pixel is fractional again and is dithered as before, and so
is everything under film grain and in a transparent group. ±1/255 of noise is far under FXAA's
edge thresholds (the lowest is 1/8 of the local maximum, with a floor of 0.03 in √luma), so it
finds the same edges. `fxaa_intermediate_is_dithered`: a ramp over 8 dark codes across 256 px in
16-px column blocks, each within 0.023 codes of the ramp with the dither (up to 0.19 without),
and a flat field between two codes has σ 0.498 codes with FXAA against 0.499 without (one dither
either way). Pure black keeps the sparkle any ±1 LSB dither gives it (one pixel in eight at code
1), as much as without FXAA.

Cost, measured in SpaceCorps on an AMD Radeon integrated GPU (Vulkan, Medium tier: FXAA Medium,
1920 × 1080, an empty sector, per-pass GPU timestamps, two interleaved pairs on a PC whose CPU
was at 100 % from other builds): the composite 1.59 and 1.92 ms before, 2.24 and 2.20 after (it
now encodes, hashes twice and decodes for every pixel); the FXAA pass 1.85 and 1.87 ms before,
1.36 and 1.33 after (it no longer does that for the pixels it copies, which is most of them).
Together 3.44 and 3.79 ms before, 3.60 and 3.53 after: the difference is inside the spread of
the two runs of one build. In that game's own picture the fix cannot be seen: its skies carry
star dust and grain of several codes everywhere, so no 8-px block of a dark sector is smooth
enough to band (a 4×-amplified before/after pair of its darkest sector shows no difference). It
shows on smooth dark content: a fog, a vignette over black, a UI-less fade.

## Vignette

`Vignette` darkens (or tints) toward the corners, in HDR before the tonemapper:
`color × mix(1, vignette.color, intensity × f)`, where `f` rises smoothly from 0 to 1 over the
last `smoothness` of the distance to the corners (1 at the corners of a centered vignette).
`roundness` 1 is circular whatever the aspect, 0 follows the rect's shape; `center` moves it;
`color` black darkens (a color tints). At the defaults (intensity 0.3) the corners are 30 %
darker.

## Chromatic aberration

`ChromaticAberration { enabled, intensity }` (or `with_chromatic_aberration(i)`): red and blue
are sampled apart, radially from the rect's center and growing with r²; at the corners the split
is `intensity × 2 %` of the rect's width (resolution-independent). The taps are bilinear and
clamped to the rect. Up to 1.5 px from green, red and blue are one tap each; a wider split is a
spectrum of taps at most 1.5 pixels apart (so a star smears into a small rainbow instead of
three dots: `wide_split_is_a_spectrum`), at most 17, so the cost is bounded whatever the
intensity (the composite at 1080p on Dev: 0.091 ms without CA, 0.119 ms at 0.15, 0.137 at 0.25,
0.172 at 0.5, 0.202 at 1.0; with taps a pixel apart and up to 25 of them it was 0.126 / 0.150 /
0.199 / 0.248). The distortion's
spectral split (`Distortion::chromatic`) uses the same taps. Off in `GraphicsQuality::Low`.

## Film grain

`FilmGrain`: value noise in the encoded (sRGB) space, one value for the three channels, cells of
`size` pixels at 1080p (scaled with the target's height, at least one pixel), amplitude
`intensity` fading in the highlights with `luminance_response` (1: no grain on white). It is
seeded (spec §4.3): `animated` grain uses `pcg(seed ^ Time::frame)` — a new pattern each frame,
the same pattern for the same frame (`render_to_image` twice is identical, capture replays
repeat it) — and still grain (`animated: false`) always `seed` (use it for goldens). The seeds
of animated grain and dither are uploaded for 64 frames at once (one aligned uniform slot per
frame and post group, selected with a dynamic offset); everything else of a still scene uploads
nothing.

## Distortion

Screen-space distortion bends the image under a *writer*: a heat haze behind an engine, a
shockwave ring expanding from a warp arrival or an explosion. Writers draw offsets into the post
group's distortion buffer, and the composite shows at each pixel the scene where the buffer
points. It is on by default (`PostProcess::distortion`) and free while nothing writes: without
a writer with work in the post camera's view there is no Distortion pass, no buffer write and
no composite tap (`no_distortion_pass_without_writers`: bit-identical images).

```rust
use space3d::prelude::*;
use space3d::render::{Distortion, DistortionMaterial, DistortionMode};

// A shockwave ring on a quad (MeshData::plane: its UV center in the middle, UV radius 0.5 at
// its edges); grow the transform's scale and fade `strength` to animate it.
world.spawn((ring_transform, Mesh3d(quad), DistortionMaterial {
    strength: 0.05, // view heights at the ring's crest
    mode: DistortionMode::Radial { inner: 0.3, outer: 0.5 },
    ..DistortionMaterial::default()
}));
// Engine heat haze: the built-in value noise, scrolling down the plume (UV per second). A
// shimmer is a few pixels: 0.004 view heights is 4 px at 1080p (and capped by the strip's size).
world.spawn((plume_transform, Mesh3d(strip), DistortionMaterial {
    strength: 0.004,
    scroll: Vec2::new(0.0, -1.2),
    soft_depth: 0.3,
    ..DistortionMaterial::default()
}));
// Per camera: a global multiplier, a spectral split and the buffer's resolution.
PostProcess::default().with_distortion(Distortion { chromatic: 0.4, ..Distortion::default() });
```

### How it works

* **The buffer.** One `Rgba8Unorm` texture per post group (`DISTORTION_FORMAT`), the group's
  rect in `S` at half resolution (`DistortionResolution::Half`, default) or full, persistent per
  settings camera (freed 60 frames after its last use; capture replays reuse it). Offsets are in
  *view heights* (1.0 = the view's height; +x right, +y up) with split signs — R = +x, G = +y,
  B = −x, A = −y — so writers blend additively (`DISTORTION_BLEND`, One/One) in a format every
  device renders and blends, WebGL2 without float extensions included. Each axis holds up to
  ±0.0625 view heights (`DISTORTION_RANGE`; overlapping writers saturate there, never wrap) in
  steps of 0.00025 view heights: 0.26 px at 1080p.
* **The Distortion pass** runs right after the settings camera's scene pass (and its SceneRead
  pass): the buffer is cleared to 0 and the camera's `RenderPhase::Distortion` node slots draw
  into it with that camera's view uniform, the view's single-sample scene depth bound in group 1
  where there is one (spec §1.6, §1.8).
* **The apply** is fused into the composite: it samples the buffer bilinearly at the pixel's
  position in the rect, looks up the scene that far away (taps clamped to the group's rect: split
  screens never pull each other's pixels in, `distortion_does_not_bleed_across_viewports`), with
  red and blue at (1 − c) and (1 + c) times the offset for `Distortion::chromatic` = c, and reads
  the bloom at the same offset. `Distortion::strength` scales every offset. Where the buffer is 0
  the pixel is exactly the undistorted one (a texel load at render scale 1), elsewhere a bilinear
  tap. HdrScene overlays are part of `S`, so they are distorted with the scene
  (`hdr_scene_overlays_are_distorted`); AfterTonemap overlays and display views are not.
* **Accuracy** (`distortion_shifts_the_background`, spec ±1 px): a constant writer of 0.04 view
  heights moves a 128-px-high view's edges by 5.114 px against 5.114 expected — within 0.003 px on
  Metal, ANGLE, with MSAA, at render scale 0.5 and under `DownlevelOverride::webgl2()`.

### `DistortionMaterial` meshes

The component draws the entity's mesh (`Mesh3d` + `GlobalTransform`, optional `Visibility`) into
the distortion buffer of every post camera that sees it. Without a `Material` the mesh is
invisible itself; with one it draws in color too. The built-in node `"space3d/distortion-meshes"`
does it through the public node API only (graphics spec §5.2 PO5):

* **Extraction and culling.** Once per frame the node gathers the writers, their meshes
  (`MeshRef`) and textures into draw order (texture, mesh, entity); per camera view it culls
  their bounding spheres against the view's frustum (only scene views of cameras whose
  `PostProcess` enables distortion) and appends the visible ones' 96-byte instances to one
  instance stream. A view's run is uploaded through the staging ring only when it differs from
  what the GPU buffer holds: a still scene uploads nothing (`distortion_steady_state`).
* **Drawing.** One instanced draw per run of equal mesh and texture, with the mesh arena bound at
  the mesh's first vertex and the instance stream at the run's first instance: neither
  `base_vertex` nor `first_instance`, so the WebGL2 path is the only path and every backend tests
  it. Pipelines are built per pass signature (`prewarm`), with and without scene depth.
* **`DistortionMode::Normal`**: the offset comes from `texture` (RG, 0.5 = none, 0 and 1 = −/+
  `strength` along U and V; sampled with the texture's own sampler, so use `AddressMode::Repeat`
  to scroll; sRGB formats are read back as their stored values) or, without one, the built-in
  value noise (two octaves, 3 and 7 cells per UV unit, the finer at a quarter, repeating every
  UV unit). `scroll` moves it by render time (`FrameInfo::time`: deterministic, replays and
  `render_to_image` twice give the same haze). The outer 0.2 of the UV square feathers out, so a
  quad has no visible edge.
* **Size cap.** The strength is in view heights, so a thin or distant strip could bend the
  image by more than its own width: at 1080p the old recipe (0.012 with 8- and 16-cell noise)
  asked a plume strip 82 px wide for 13 px and tore the straight seams under it into 10–15 px
  squiggles. Like the particles' writers, a `Normal` writer bends the image by at most a
  fraction of half the narrower on-screen extent of its UV square: 0.12 with the built-in noise
  (a shimmer: 4.9 px for that strip), 0.5 with a texture. `Radial` rings are not capped. Keep heat
  haze at a few pixels: `strength` 0.003–0.005.
* **`DistortionMode::Radial { inner, outer }`**: a smooth ring (sin², zero slope at both radii)
  between the two UV radii whose pixels look toward the UV center: the image under the ring is
  pushed outward, up to `strength` halfway between the radii
  (`radial_distortion_pushes_outward`).
* **In UV space.** Offsets are laid out in the mesh's UV space and turned onto the screen through
  the inverse of the UV Jacobian (screen-space derivatives), keeping their length: a rotated quad
  rotates its haze and its ring stays round in perspective; a mesh seen edge-on writes nothing.
  Vertex color alpha weights the offset (paint 0 where it should fade).
* **Occlusion.** A writer behind closer geometry writes nothing there: hard (`soft_depth` 0) or
  fading in over `soft_depth` world units, against the view's scene depth
  (`distortion_occluded_by_closer_geometry`: the closer geometry is bit-identical to the scene
  without the writer, with read-only depth, with the GL depth path forced, and with MSAA through
  the depth resolve). And the **foreground guard**: a writer fragment whose offset would look at
  scene depth closer than the writer writes nothing either, so a haze behind a ship never drags
  the ship's edge over the background (`foreground_is_not_dragged_over_the_background`).

### Writing distortion from a node (particles, custom effects)

Any `RenderNode` can write distortion: particle layers with `SpriteTarget::Distortion` (the
particles area) do, and so can a game's own effects. Register a slot in the Distortion phase,
report work only when something is visible, and draw into the pass described by `ctx.target`:

```rust
static SLOTS: [PhaseSlot; 1] = [PhaseSlot::new(RenderPhase::Distortion, 100)];
// Pipeline for ctx.target (DISTORTION_FORMAT, 1 sample, no depth attachment; scene depth in
// group 1 when ctx.target.depth_access != DepthAccess::None):
//   color target: ctx.target.color_target(Some(DISTORTION_BLEND))
//   shader: VIEW_WGSL + scene_wgsl(&ctx.target) + DISTORTION_WGSL + your code
// Draw: set_bind_group(0, ctx.view_bind_group, &[ctx.view.view_offset]),
//       set_bind_group(1, ctx.scene_bind_group, &[]); viewport = the buffer (ctx.target_size).
```

```wgsl
@fragment fn fs(in: Out) -> @location(0) vec4<f32> {
    let offset = in.dir * in.strength;                        // view heights, +y up
    let fade = distortion_depth_fade(in.pos, db_size, 0.2);  // db_size: ctx.target_size
    return distortion_encode(offset, fade * in.alpha);
}
```

`DISTORTION_WGSL` also has `s3d_distortion_decode` (the inverse of `distortion_encode`) and
`s3d_distortion_guard(px, offset, frag_view_depth)` (the foreground guard). The slot only draws
in a post group's settings camera whose `PostProcess::distortion` is enabled; everywhere else its
mask bits are ignored.

### Platforms

| class (spec §4.1) | distortion |
|---|---|
| A (Metal, Vulkan, DX12, WebGPU) | ✓, occluded by scene depth (with MSAA through the depth resolve) |
| B (WebGL2 / GLES 3.0 with float targets; ANGLE) | ✓, occluded at 1×; **not occluded with MSAA**: GLES 3.0 cannot sample multisampled depth (`gl_distortion_not_occluded_with_msaa`) |
| C (WebGL2 without float targets) | as B, in the Ldr fallback |
| D (GLES 3.1+ Android) | as B |

The writer shader uses no storage buffers, no integer `%` of negative values (undefined in GLSL
ES) and no `mix` of integer vectors; ANGLE runs every distortion test (`tests/gl_post.rs`).

### Cost

At 1080p on the dev machine (`benches/post.rs`): three writers covering about a third of the
frame (two shockwaves and a haze) add 27 µs of GPU throughput over bloom (`post/ab`: 236.8 → 264.0
µs over Direct); the Distortion pass's own timestamp reads 0.05 ms (0.52 ms at 4K with the same
coverage). One render pass, 2.1 MB of buffer at 1080p (8.3 MB at full resolution); wgpu's
bookkeeping adds 33 allocations per frame on Metal (`tests/alloc_post.rs`); the node itself
allocates nothing in the steady state.

### Limitations

* Transparent draws in front of a writer (particles, blended meshes, HdrScene overlays) are part
  of the scene color: they are distorted too (the depth test only sees opaque depth).
* Offsets are capped at ±6.25 % of the view height per axis; a writer that needs more saturates.
* Pixels bordering closer geometry blend across one buffer texel (half resolution): the
  occluder's outline moves by at most about a pixel; its interior is exact.
* No refraction of the scene color itself (a scene-color copy is deferred, spec D4).

## City effects: fog, SSAO, SSR

Three opt-in effects for photographic scenes (a city street, an interior): height/distance fog,
screen-space ambient occlusion and screen-space reflections. They are fields of the camera's
`PostProcess` (`fog`, `ssao`, `ssr`), disabled by default; a camera that enables none renders
pixel-identical to before and records no extra pass, texture or draw (tested:
`disabled_effects_are_pixel_identical_and_record_nothing`). The public surface is listed in
[`city-effects-api.md`](city-effects-api.md); `cargo run -p space3d-render --example
city_effects -- out_dir` renders a wet-street scene with each effect off and on.

```rust
use space3d::render::{Fog, Ssao, Ssr};

PostProcess::default()
    .with_ssr(Ssr::on())                    // reflections in wet asphalt and glass
    .with_ssao(Ssao::on().with_radius(1.5)) // contact shadows in creases
    .with_fog(Fog::city_haze());            // haze with a sun lobe, thinner with height
```

### Where they run

All three work on the linear HDR scene colour of the post group, **before bloom and exposure**,
in the order SSR, SSAO, fog (fog last, so distance hides what the other two did). They are one
extra stage of the post chain: the frame hands the settings camera's single-sample scene depth
(a multisampled depth is resolved for it, as for the SceneRead and Distortion passes) and its
`View` uniform to the effect passes, which write a pooled copy `S'` of the scene colour that
bloom, auto exposure and the composite then read instead of `S`.

| Condition | Result |
|---|---|
| `FrameMode::Hdr`, depth available | the effects run |
| `HdrMode::Ldr` / no float render targets (WebGL2) | all three disabled, warned once; the image equals the one without them |
| Direct mode (no post) | not applicable: effects need a `PostProcess` camera |
| GL, multisampled scene | no sampleable multisampled depth: disabled (`GraphicsCaps::msaa_depth_resolve`) |
| SSR without read-only depth attachments (GL) | SSR alone is disabled; fog and SSAO still run |
| capture replay that does not re-render the scene | the replayed image has no effects (depth is not kept) |

### Fog

`Fog` integrates the density `density * exp(-height_falloff * (y - base_height))` analytically
along the view ray (Y up), `opacity = min(1 - exp(-tau), max_opacity)`, and mixes the pixel
toward the fog colour. The colour is `color` plus a sun lobe `sun_color * sun_intensity *
pow(max(dot(view_dir, -sun_dir), 0), sun_exponent)`; the sun is the first directional light of
the view unless `sun_direction` is set. Sky pixels (depth at the far plane) take
`sky_amount * exp(-sky_falloff * max(view_dir.y, 0))`, so the haze meets the horizon.
`Fog::opacity` is the CPU mirror of the shader (tested against the rendered pixels).

* Cost: one full-screen pass over the group's rect (no extra textures except the pooled `S'`).
* Sky fog multiplies the sky radiance; with bloom on, keep `Bloom::threshold` above the fogged
  sky or the horizon blooms.

### SSAO

Ground-truth ambient occlusion (Jimenez et al., 2016, in XeGTAO's formulation): per pixel
`slices` directions, `steps` samples per side of each, a horizon angle search in view space and
the analytic cosine-weighted visibility integral, from the depth alone (view-space normals are
reconstructed from the closer depth neighbour on each axis). Interleaved gradient noise rotates
the slices and jitters the steps; a separable depth-aware blur (distance of the neighbour's
surface point from this pixel's tangent plane, so slanted ground blurs along itself) removes it.
Samples on the surface's own plane (within `bias` of its tangent) never occlude: depth
quantization would otherwise darken flat ground at grazing angles.

* The AO texture is `R8Unorm` at half resolution (`half_res`), pooled; two blur passes.
* The result multiplies the *finished* HDR colour by `mix(1, ao^power, intensity)`. A forward
  renderer has no ambient or diffuse-IBL buffer to scale instead, so direct light is darkened in
  creases as well; keep `intensity` moderate on strongly lit scenes, or lower `radius`.
* Perspective cameras only (the horizon search assumes `V = normalize(-P)`).

### SSR

A linear ray march of the reflected ray in view space, stepped in screen space (perspective-correct
depth by interpolating `1/w`), a thickness test (`thickness + 3 % of the ray depth`), a 5-step
binary refinement, a per-pixel jitter hiding the step banding. The hit takes the scene colour
(`S`, before this frame's effects) at the hit pixel. The reflection weight is
`F0 * A + B` from the existing BRDF table (the forward shader's split-sum terms) times a
roughness fade (`max_roughness`, from 60 % of it down to 0) times an edge fade (`edge_fade`) times
a fade over the last 40 % of the ray. On a hit the forward pass's cubemap specular is *replaced*:
`colour += weight * (hit - env_specular(reflect, roughness))`, so misses, rough surfaces and
the screen edge keep the IBL reflection the forward shader already produced.

**G-buffer-lite.** SSR needs a normal, a roughness and a reflectivity per pixel, which the forward
pass does not store. For a camera with `ssr.enabled` the frame records one extra pass after the
scene pass (`RenderStats::render_passes + 1`): the view's opaque draws again with
`PassKind::GBuffer` (`fs_gbuffer` of the forward template: the same material evaluation, no
lighting), depth-tested `LessEqual` read-only against the single-sample scene depth, into a
per-camera `Rgba8Unorm` target (octahedral world normal, roughness, F0 as the max channel of
`mix(0.04, base, metallic)`; `0` = no reflection: unlit and cleared pixels). Custom-shader
materials take part with their material parameters (their fragment hook is not run). Costs
4 bytes per scene pixel while enabled, and the opaque draw calls again.

Round 2 additions: `Ssr::{roughness_blur, stretch, hdr_boost, denoise}` (hit radiance averaged over
an ellipse stretched vertically for glare streaks on wet ground, a depth-aware reflection blur, no
history), hits on back-facing surfaces rejected, `Skybox::with_ground` (sky below the horizon
replaced by a color so a ground plane meets the sky), `MeshData::{cuboid_tiled, plane_tiled}` and
`Material::with_tiling` (real-world texture tiling through the existing KHR_texture_transform), see
`city-effects-api.md`. SSR GPU scopes: `gbuffer`, `post.ssr`, `post.ssr_blur`; SSAO `post.ssao`,
`post.ssao_blur`; `post.screenfx` is the apply pass.

Limitations (v1): blended (glass-as-alpha) materials are not in the G-buffer, so use an opaque
smooth, dark, metallic material for glass; reflections only show what is on screen (the rest
falls back to the cubemap); the reflection tint is the F0 maximum (metals reflect untinted);
a pixel's own G-buffer entry is the front-most opaque surface of the settings camera.

### Quality tiers

`GraphicsQuality::post(base)` caps what was authored (lossless: the authored look comes back on
a higher tier) and turns the expensive ones off:

| Tier | Fog | SSAO | SSR |
|---|---|---|---|
| Low | kept, sun lobe off | off | off |
| Medium | as authored | 1 slice x 4 steps, half res, blur 2 | off |
| High | as authored | up to 2 x 6 | up to 32 steps |
| Ultra | as authored | up to 3 x 8 | as authored (up to 128 steps) |

### Cost

Passes added per post group (`RenderStats::screen_fx_passes`): fog alone 1 (apply), SSAO 3 (AO +
two blurs) + apply, SSR 1 (apply) plus the G-buffer pass counted in `render_passes`. Several
effects share the one apply pass: all three on a 960x540 headless frame record 16 render passes
against 12 without (`city_effects` example). Memory: the pooled `S'` (one scene texture), two
`R8` AO textures (a quarter of the pixels each at half resolution), the camera's `Rgba8` G-buffer.

### Tests

Unit: parameter packing and the plan (`post/screenfx.rs`), the fog's CPU mirror
(`post/city.rs`), tier caps (`quality.rs`). GPU (`tests/city_effects.rs`, skipped without an
adapter): disabled effects are pixel-identical and record nothing; fog opacity against the CPU
mirror at two distances and its height falloff; the same fog through a resolved MSAA depth;
SSAO darkens a floor-wall crease and leaves open floor within 3 %; SSR puts a red box into a
glossy floor and changes nothing above the horizon; all three off in the Ldr fallback.

## Hardening (PO6)

### GPU timings

With `FrameSettings::gpu_timing` every post pass writes timestamps (F8's `GpuTimings`, 2–5 frames
late) under the spec §4.4 labels, with the post group's settings camera
(`post_gpu_timings_are_labelled`):

| label | scope |
|---|---|
| `post.exposure` | the histogram's compute pass, or one span over the render-pass metering chain |
| `post.bloom` | one span from the prefilter to the last upsample |
| `post.composite` | the composite |
| `post.fxaa` | the FXAA pass |
| `distortion` (the frame's) | the Distortion pass |

On Apple GPUs a pass's scope includes its overlap with the neighbouring passes, so the sum of the
scopes overstates the chain (bloom's own scope reads 0.17 ms alone and 0.30–0.39 ms inside the
full chain). The gate below therefore uses throughput.

### Passes and memory

Per post group at 1920 × 1080 (`post_pass_and_memory_counts`, on Metal and ANGLE):

| configuration | render passes | compute passes | memory (`target_bytes`) |
|---|---|---|---|
| composite alone (exposure, tonemapper, dither) | 1 | 0 | — |
| + bloom (6 levels) | +11 | 0 | +2.76 MB (`Rg11b10Ufloat`; 5.5 MB in `Rgba16Float`) |
| + auto exposure (compute) | 0 | +1 | 1×1 textures only |
| + auto exposure (render passes: WebGL2, `no_compute`) | +4 | 0 | +1.1 MB of levels |
| + FXAA | +1 | 0 | +8.3 MB (the LDR intermediate) |
| + distortion with a writer | +1 (the frame's Distortion pass) | 0 | +2.1 MB (half), +8.3 MB (full) |

The steady state creates no textures and reallocates nothing in any configuration; a change of
`DistortionResolution` recreates the buffer once. Allocations per frame (Metal, 320 × 180,
`tests/alloc_post.rs`): distortion meshes on top of bloom 267 (+33), every effect at once 351 (9
render passes and a compute pass), no growth, nothing retained.

### Resizes and device loss

`resize_reallocates_once_and_stays_bounded`: 30 resizes with every effect on; each new size
reallocates on its first frame only, the targets in use stay within 24 bytes per pixel (+64 KiB),
and back at the first size the frame uses exactly the memory it used before (pooled bloom levels
of old sizes are evicted after 3 idle frames; the distortion buffer is resized in place).
`a_lost_device_degrades_with_every_effect`: with auto exposure read back, bloom, FXAA, grading, a
LUT and distortion meshes, a destroyed device drops frames and fails captures without a panic,
also when writers, settings and the size change afterwards.

### Performance and the gate

`benches/post.rs` (release, dev machine: Apple M4 Max, Metal):

* `post/gate`, spec §4.4 "all post on (AE compute, bloom, FXAA, CA, grain, grading, LUT,
  distortion) ≤ 0.60 ms": **met, 0.387 ms**, measured as the paired throughput delta against
  Direct at 1080p (frames back to back, interleaved rounds), the way the frame's "Hdr minimal
  over Direct" gate is measured. The whole frame with every effect takes 0.49 ms of throughput,
  so even charging all of it but the scene pass (0.05 ms) to post stays under the gate. The
  per-pass timestamps sum to 1.0–1.3 ms (overlap, above).
* `post/ab` (1080p throughput per frame over Direct): composite +46 µs, + bloom +237 µs,
  + distortion +264 µs, every effect +388 µs.
* `post/gpu` (per-pass timestamps, loaded GPU, medians; reference only):

| mode | 1080p | 4K |
|---|---|---|
| composite | composite 0.16 | composite 1.16 |
| bloom | bloom 0.19, composite 0.18 | bloom 0.79, composite 0.67 |
| + auto exposure (compute) | exposure 0.04 | exposure 0.04 |
| + auto exposure (render passes) | exposure 0.53 | exposure 0.74 |
| + FXAA | fxaa 0.18 | fxaa 0.97 |
| + distortion | distortion 0.05 | distortion 0.52 |

* `post/cpu/<mode>` (criterion, the render stage's CPU at 1080p, load average 11–13 from other
  builds): composite 0.12 ms, bloom 0.26 ms, lens effects 0.25 ms, distortion 0.25 ms, every
  effect 0.25 ms, render-pass metering 0.29 ms (the auto-exposure and FXAA runs were disturbed by
  the load: 0.5–0.8 ms with wide intervals). Bloom's 11 render passes remain most of post's CPU
  cost; the spec's Battle line "post ≤ 0.05 ms" of CPU is **not met** (see Bloom, "Cost": the
  compute chain is the planned fix). Distortion adds one render pass and the node's extraction
  and culling, within the noise here.

## Display-space effects and overlays

Grain, dither, vignette, chromatic aberration and FXAA run inside the post group: `HdrScene`
overlays get them, `AfterTonemap` overlays (egui, 249c874-era overlays) and display views
(gizmo cameras without post) do not (tested:
`after_tonemap_overlays_untouched_by_grain_ca_vignette`). The passthrough group (render scale
without post) never applies them.

## The transfer function

sRGB is written down three times: on the CPU (`linear_to_srgb`, `srgb_to_linear`), in WGSL
(`s3d_linear_to_srgb`, `s3d_srgb_to_linear` in `OUTPUT_WGSL`, which the forward tail, the
overlays and the composite share) and in the GPU's own encoding of an sRGB-format target.
`tests/transfer.rs` holds them together: the constants and well-known greys (0.0031308 ↔ 0.04045,
0.18 → 118, 0.5 → 188, every code a round trip), the WGSL pair against the CPU pair over 256
entries within 1 LSB, and one grey ramp through every way a pass writes its output
(`OutputEncoding::Linear` into an sRGB target, `ManualSrgb` into an 8-bit target without one,
`Linear` into a float target encoded on the CPU): the same bytes. The composite's three output
kinds (`S3D_POST_OUT`) end in those same writes.

Which of them a display gets is decided once, from what the surface reports
(`surface_formats` in renderer.rs), the same on every platform: an sRGB surface format where
there is one (the GPU encodes, the UI and overlays blend in linear light); else an sRGB view of
the surface where the device has `SURFACE_VIEW_FORMATS` (browser WebGPU); else the shader
encodes and blending happens on encoded values. Read from wgpu-hal 30 (not run on a device):
Metal lists `Bgra8UnormSrgb` (iOS and macOS take the first path); Vulkan lists what the driver
does, which on Android includes an sRGB format in practice; GLES lists `Rgba8UnormSrgb` when
EGL is 1.5 or has `EGL_KHR_gl_colorspace` (WebGL2 always lists it), and only then: an Android
device on GLES with an older EGL takes the third path, where translucent UI and additive
overlays blend in gamma space and so look slightly different from every other device.

## Determinism

Distortion meshes are extracted in a fixed order (texture, mesh, entity) and scroll by render
time (`FrameInfo::time`), never the wall clock; capture replays draw the last frame's instances
again. Bloom and grading have no temporal state and no noise: the same scene gives bit-identical
images (tested with `render_to_image` twice). Capture replays re-run the chain from the scene
texture. Auto exposure adapts with `FrameInfo::delta` (0 when `Time::frame` did not advance) and
its histogram reduction is order-independent, so the same ticks give the same exposure and
capture replays never meter (`render_to_image_is_idempotent_with_ae_and_grain`: two apps
stepped through the same ticks give identical images). Grain and dither are seeded from
`Time::frame` (never the wall clock or the renderer's own frame count); their seeds are hashed on
the CPU (bit-identical to hashing per pixel), a window of frames at a time — each frame's seed is
still a pure function of that frame.

## Tests and stories

* `tests/transfer.rs`: `srgb_golden_values`, `every_output_kind_writes_the_same_bytes`,
  `wgsl_transfer_functions_match_the_cpu` ("The transfer function").

* `tests/post.rs` (and `tests/gl_post.rs` on GL/ANGLE): bloom — `bloom_only_above_threshold`,
  `bloom_is_in_exposed_units`, `energy_conserving_preserves_mean`,
  `bloom_does_not_bleed_across_viewports`, `hdr_scene_overlays_feed_bloom`,
  `bloom_off_in_ldr_fallback`, `transparent_rt_bloom_weighted_by_alpha`, `bloom_steady_state`,
  `bloom_downlevel_parity`, `bloom_halo_is_independent_of_position`; auto exposure —
  `auto_exposure_converges_and_matches_reference` (and `auto_exposure_downlevel_matches_reference`
  for the render-pass path on the default backend; on GL it is the only path),
  `instant_converges_frame_1`, `render_to_image_is_idempotent_with_ae_and_grain`,
  `auto_exposure_feeds_bloom`, `auto_exposure_per_post_group`,
  `auto_exposure_history_and_steady_state` (both paths); grading — `grading_matches_formulas`,
  `lut_identity_is_neutral`, `lut_changes_colors`, `invalid_grading_hook_falls_back`,
  `display_gamma_matches_formula`; FXAA and
  lens effects — `fxaa_softens_edges`, `fxaa_intermediate_is_dithered`, `film_grain_is_seeded`, `grain_after_fxaa`,
  `after_tonemap_overlays_untouched_by_grain_ca_vignette`, `vignette_darkens_corners`,
  `chromatic_aberration_splits_toward_corners`, `lens_effects_do_not_bleed_across_viewports`,
  `fxaa_ldr_fallback_and_steady_state`, `transparent_rt_lens_effects_keep_alpha`;
  `gl_bloom_matches_default_backend`; hardening — `post_gpu_timings_are_labelled` (default
  backend: GL has no timestamps), `post_pass_and_memory_counts`,
  `resize_reallocates_once_and_stays_bounded`, `a_lost_device_degrades_with_every_effect`.
* `tests/post_distortion.rs` (and the same cases in `tests/gl_post.rs`):
  `distortion_shifts_the_background` (also under `DownlevelOverride::webgl2()`, at render scale
  0.5 and with MSAA), `distortion_chromatic_split`, `distortion_occluded_by_closer_geometry`
  (read-only depth, forced GL depth path, MSAA; on GL with MSAA: not occluded),
  `foreground_is_not_dragged_over_the_background`, `hdr_scene_overlays_are_distorted`,
  `no_distortion_pass_without_writers`, `radial_distortion_pushes_outward`,
  `distortion_does_not_bleed_across_viewports`, `distortion_steady_state`.
* `tests/alloc_post.rs`: allocations per frame (Metal, 320 × 180 unless noted) with bloom on one
  post group (234) and on two (423), `PostProcess::cinematic()` (261: the FXAA pass on top of
  bloom), auto exposure with the readback on the compute path (289) and on render passes (308),
  bloom with every grading step, a LUT and a hook (234: grading adds nothing), distortion meshes
  moving every frame (267), every effect at once (351) and bloom at 1080p (332).
* Unit tests in `src/post/`: chain sizing, the soft threshold's and the firefly limit's CPU
  mirrors, pass uniforms, the prefilter's stability and energy on the GPU
  (`bloom_is_stable_under_motion`, `bloom_energy_is_proportional_and_limited`), WGSL layouts
  (`PostParams`, `BloomPass`, `ExposureParams`), the exposure histogram's CPU mirror, strides,
  adaptation factors and sanitizing, the grading matrices and CDL, hook composition, the `.cube`
  parser (`cube_parser_*`) and the strip lookup's CPU mirror, FXAA presets, the pre-hashed seeds,
  the distortion writer's WGSL (with and without scene depth) and instance layout, and GPU-time
  measurements run alone with `--ignored` (`bloom_gate_1080p`, `bloom_gpu_time_by_size`,
  `exposure_gpu_time`).
* `benches/post.rs`: `post/cpu/*`, `post/gpu`, `post/ab` and `post/gate` (above).
* Stories: `Engine/Post/Bloom` (engines, laser bolts, sparks, stars and a threshold ramp, with
  every `Bloom` field on a knob), `Engine/Post/Bloom Veil` (`Bloom::natural()`),
  `Engine/Post/Auto Exposure` (the same scene lit 16× dimmer, as authored and 16× brighter in
  three post groups, exposed alike; knobs for every `AutoExposure` field that matters and the
  compensation), `Engine/Post/FXAA` (the same view without and with FXAA side by side; quality
  knob), `Engine/Post/Lens` (vignette, chromatic aberration and still grain on knobs, FXAA on)
  and `Engine/Post/Color Grading` (ungraded and graded side by side: white balance, saturation,
  contrast, lift/gamma/gain, generated teal-and-orange and bleach-bypass `.cube` LUTs, and a
  night-vision `post_ldr_hook`), `Engine/Post/Distortion` (a shockwave ring and a heat haze over
  a checkerboard; strength, chromatic split and resolution knobs), `Engine/Post/Warp Shockwave`
  (a top-down warp arrival: the ring expanding under the ship over a nebula, stars and a planet;
  `progress` knob; a GL golden variant), `Engine/Post/Engine Heat Haze` (top-down fighters over a
  station deck, the haze scrolling down every plume; 20 ticks) and `Engine/Post/Tonemappers`
  (four tonemappers side by side over an HDR ramp, saturated emissives and lit materials).
* `examples/post_processing`: every effect on a top-down space scene, keys `1`–`9` toggle them,
  `T` cycles the tonemappers, `G` prints the post passes' GPU timings once per second;
  `SPACE3D_HEADLESS=1 cargo run -p post_processing` writes `post_processing.png` and exits.
