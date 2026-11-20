# Graphics

Space3d's renderer draws a space battle with HDR, bloom, PBR materials lit straight from glTF,
a skybox that is also the image-based lighting, a shadow-casting sun, 100k GPU particles with
ribbons, beams and screen-space distortion, and a post chain — at 1080p in about 1.5 ms of GPU
time on an Apple M4 Max, and on WebGL2 with a coherent Low preset.

![The SpaceCorps look](../crates/space3d-stories/tests/__screenshots__/Engine/Showcase/SpaceCorps_Look.png)

*`Engine/Showcase/SpaceCorps Look`: procedural hulls with base color, ORM, normal and emissive
maps; the nebula sky as skybox and image-based lighting; the sun's shadows on the asteroids;
engines, afterburners, a laser beam with hit sparks and shield ripples, bolts, a missile's smoke
ribbon, an explosion with its light flash and shockwave, a ship warping in; bloom, AgX punchy,
a vignette — `GraphicsQuality::High`.*

Nothing changes until you ask: a camera without `PostProcess`, a `Material` without maps and an
app without the new components render exactly as before, pixel for pixel (spec D20). Everything
here is additive to the API games already use ([`graphics/api.md`](graphics/api.md)).

| page | covers |
|---|---|
| this page | overview, the frame, a how-to per feature, WebGL2, budgets, quality presets, showcase |
| [`graphics/migration.md`](graphics/migration.md) | adopting every feature in SpaceCorps2027, step by step, with code |
| [`graphics/frame.md`](graphics/frame.md) | frame modes, HDR, tonemapping, render scale, dynamic resolution, levels of detail, several cameras, render nodes, overlay stages, depth views, GPU timings and CPU spans |
| [`graphics/materials.md`](graphics/materials.md) | material maps, blend modes, glTF import, tangents, cubemaps, skybox, image-based lighting, the directional shadow, prewarm |
| [`graphics/post.md`](graphics/post.md) | bloom, auto exposure, color grading and LUTs, FXAA, vignette, chromatic aberration, film grain, distortion |
| [`graphics/particles.md`](graphics/particles.md) | emitters, effects, sprites, ribbons, beams, flipbooks, light flashes, the built-in library, budgets, LOD |
| [`graphics/api.md`](graphics/api.md) | the compatibility policy and where each type lives |
| [`design/graphics.md`](design/graphics.md) | the normative design |

## Quick start: the SpaceCorps look

```rust
use space3d::particles::prelude::*;
use space3d::prelude::*; // PostProcess, Tonemapper, Skybox, EnvironmentLight, DirectionalShadow, …
use space3d::render::{GraphicsQuality, ShadowFit, Vignette};

fn build(app: &mut App) {
    app.add_plugin(DefaultPlugins::new().title("My space game"));
    app.add_plugin(ParticlesPlugin::default());
    app.add_system(Stage::Startup, setup);
}

fn setup(ctx: &mut SystemCtx) {
    // An HDR sky: a panorama, six faces or a `.skybox` manifest (materials.md).
    let sky = ctx.resources.expect_mut::<RenderAssets>().add_cubemap(my_sky_source());
    // The camera: HDR + the battle look, the sky as background and as image-based lighting.
    ctx.world.spawn((
        Transform::from_xyz(0.0, 400.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
        Camera3d::default(),
        PostProcess::default()
            .with_tonemapper(Tonemapper::AgXPunchy)
            .with_exposure(0.5)
            .with_vignette(Vignette { enabled: true, intensity: 0.24, ..Vignette::default() }),
        Skybox::new(sky),
        EnvironmentLight::new(sky).with_intensity(0.7, 1.0),
    ));
    // One sun that casts shadows over the flight plane.
    ctx.world.spawn((
        Transform::IDENTITY.looking_at(Vec3::new(-0.4, -0.8, -0.4), Vec3::Y),
        DirectionalLight { color: Vec3::new(1.0, 0.94, 0.86), intensity: 2.2 },
        DirectionalShadow::new().with_fit(ShadowFit::top_down(0.0, 60.0, 900.0)),
    ));
    // Ships: glTF models with their normal / ORM / emissive maps (`upload_model`), engines.
    // …
}

/// One quality tier for everything, picked for the device (Low on WebGL2). Windowed apps get
/// their renderer after `Startup`: run this once it exists (a `PreUpdate` system, as
/// `graphics_showcase` does).
fn choose_quality(ctx: &mut SystemCtx) {
    let Some(renderer) = ctx.resources.get::<Renderer>() else { return };
    let q = GraphicsQuality::recommended_for(renderer);
    let particles = ParticleSettings::for_quality_on(q, renderer.graphics_caps());
    q.apply(ctx.world, ctx.resources);
    *ctx.resources.expect_mut::<ParticleSettings>() = particles;
}
```

`examples/graphics_showcase` is this, complete: the Battle reference scene with every feature,
the tier chosen for the device and cycled with `Q`, content generated off the frame, an egui HUD
(`cargo run -p graphics_showcase --release`; `SPACE3D_HEADLESS=1` writes a PNG).

## The frame

A frame runs in one command encoder and one submit. Cameras are grouped by render target; a
camera with an enabled `PostProcess` makes its target group render through a scene texture `S`
(HDR `Rgba16Float` / `Rg11b10Ufloat`, or 8-bit in the Ldr fallback) that the composite pass
exposes, blooms, tonemaps and grades into the display target `D`. Without one, the group renders
straight into `D` as always (Direct mode).

```text
render_frame
  upload meshes, textures, cubemaps · extract and batch the ECS · plan views, groups, modes
  lighting: cubemap conversion, IBL bake, shadow light · post: plan each post group
  render nodes prepare once (particle simulation, uploads) · per view: node prepare, GpuView
  compute: GPU culling of every view (camera views and shadow maps) · particle sim / compaction
  depth views: the shadow map
  per camera view (camera order):
    scene pass → D (Direct) or S:  Opaque (meshes, alpha mask) → Skybox → Transparent
                                   (blended meshes by render_order, then node slots: particles)
    depth resolve + SceneRead (soft particles) · Distortion pass → DB (the post group's offsets)
  finish each intermediate group:
    HdrScene overlays → S (bloomed and tonemapped with the scene)
    per post group: auto exposure → bloom chain → composite → D (or → FXAA → D)
    display views (cameras after the post camera: gizmos, HUD) → D, LDR
  AfterTonemap overlays (egui) → D · submit · present
```

Details: [`frame.md`](graphics/frame.md); normative order: spec §1.4.

## How to …

**Turn on HDR, exposure and tonemapping.** Insert `PostProcess` next to a `Camera3d`. Pick the
look explicitly: `Tonemapper::AgXPunchy` with `.with_exposure(0.5)` keeps faction colors, lasers
and fire saturated in a dark space scene but skews dim saturated skies toward their strongest
channel; `AgX` (the default) is gentler, with `ColorGrading::vivid()` it keeps those skies' hues;
`KhronosPbrNeutral` keeps base colors but its toe crushes dim saturated ones, which
`KhronosPbrNeutralNoToe` passes unchanged; `AcesFitted` and `Reinhard` are there too (numbers
in frame.md, "Choose the look explicitly"). Clear colors pass through exposure
and the tonemapper like everything else. `FrameSettings::post = false` turns post off globally.
([frame.md](graphics/frame.md))

**Make things glow.** Bloom is on in `PostProcess::default()` (`Bloom::glow()`: threshold 1, a
soft knee): anything brighter than 1 after exposure glows — emissive above 1, additive
particles, stars in an HDR sky. `Bloom::natural()` is a photographic veil. Emissive maps ×
`emissive` factors above 1 make windows and engines glow. ([post.md](graphics/post.md#bloom))

**Expose automatically.** `.with_auto_exposure(AutoExposure::on())`; `exposure` becomes the
compensation. Black space is not metered, but an HDR skybox is: where a nebula fills most of the
frame it sets the exposure (the Battle brightens by about +0.9 EV and washes out), so give sky
scenes a negative compensation and a low `max_ev`, or keep a manual exposure.
([post.md](graphics/post.md#auto-exposure))

**Anti-alias.** MSAA on the primary target (`Renderer::set_msaa_samples(4)`, or the quality
preset) and on render targets (`RenderTargetDesc::with_samples(4)`); FXAA in the composite
(`.with_fxaa(FxaaQuality::Medium)`) where MSAA is too expensive (WebGL2).

**Grade and add lens effects.** `ColorGrading` (white balance, saturation, contrast,
lift/gamma/gain, a 3D LUT, WGSL hooks), `Vignette`, `.with_chromatic_aberration(x)`,
`FilmGrain`. `PostProcess::cinematic()` is a starting point. ([post.md](graphics/post.md))

**Bend space.** Distortion is on by default and costs nothing until something writes it:
explosions, warps, EMPs and engines of the particle library, `SpriteRender::distortion` in your
own effects, `DistortionMaterial` meshes (shockwave rings, heat haze), or a render node through
`DISTORTION_WGSL`. ([post.md](graphics/post.md#distortion))

**Use material maps.** `Material::with_normal_map`, `with_orm_map` (one texture for occlusion,
roughness, metallic), `with_metallic_roughness_map`, `with_occlusion_map`,
`with_emissive_map(texture, factor)`; maps multiply the factors. Normal maps need tangents:
glTF `TANGENT`, `MeshExtras::with_tangents` (MikkTSpace from
`space3d::assets::edit::mesh::compute_mikktspace_tangents`), or per-pixel frames when absent.
([materials.md](graphics/materials.md#maps))

**Load glTF materials as authored.** `space3d::assets::render::upload_model` (or
`prepare_model_async` + `PreparedModel::upload` off the frame) converts every slot — base color,
normal (+ scale), metallic-roughness, occlusion (+ strength, ORM packing), emissive (+ factor and
`KHR_materials_emissive_strength`), alpha mode, `texCoord` sets, `KHR_texture_transform` — with
sRGB for base color and emissive, linear for the rest, and generates tangents. Custom material
code adds the maps with `apply_material_maps`. ([materials.md](graphics/materials.md#gltf-import))

**Blend.** `BlendMode::{Opaque, AlphaMask, AlphaBlend, Premultiplied, Additive}`
(`Material::additive(color)`, `Material::premultiplied(color)`, `.with_alpha_mask(cutoff)`);
`render_order` orders transparent meshes; `LightingSettings::transparent_sort` sorts instances
back to front. ([materials.md](graphics/materials.md#blend-modes))

**Show a sky and light with it.** `RenderAssets::add_cubemap(CubemapSource::…)` from six faces,
three.js faces (`.skybox` manifests), an LDR or HDR panorama; `Skybox::new(cube)` on the camera
draws it behind everything; `EnvironmentLight::new(cube)` bakes irradiance and GGX-prefiltered
specular from it (once, on the GPU) so metal reflects the nebula. Use `ShadingModel::Standard`
(the presets do). ([materials.md](graphics/materials.md#image-based-lighting))

**Cast shadows.** `DirectionalShadow` next to one `DirectionalLight`; `ShadowFit::top_down(plane_y,
half_thickness, max_distance)` fits the camera's view of a slab (top-down games), `Bounds(aabb)` a
fixed box; `ShadowFilter::{Hard, Pcf4, Pcf9, Pcf16}`; `Material::shadows` opts meshes in or out.
([materials.md](graphics/materials.md#directional-shadow))

**Spawn effects.** `app.spawn_effect(builtin::EXPLOSION, EffectSpawn::at(p).scale(1.2))` — one
call per one-shot; `ParticleEmitter::new(builtin::ENGINE_EXHAUST)` (+ `EffectSpawn::follow`) for
engines; beams with `.beam_to(BeamEnd::Entity(target))`; ribbons follow bolts and missiles. 14
built-ins (explosion, hit spark, shield ripple, warps, engines, afterburner, laser trail, beam and
shot, missile trail, debris, EMP, repair beam), authored for HDR + bloom; your own in code or
`.pfx.ron`. ([particles.md](graphics/particles.md))

**Draw your own effects into the frame.** A `RenderNode` gets slots in the scene passes
(Opaque, Skybox, Transparent, SceneRead with the scene depth, Distortion), compute in `prepare`,
the view bind groups and `VIEW_WGSL` / `scene_wgsl` / `OUTPUT_WGSL`. A `RenderOverlay` with
`fn stage(&self) -> OverlayStage { OverlayStage::HdrScene }` draws into the HDR scene before
bloom and tonemapping (below). ([frame.md](graphics/frame.md#render-nodes-drawing-into-the-frames-passes))

**Render at a lower resolution.** `FrameSettings::render_scale` (or `CameraFrame::render_scale`
per camera): the scene renders smaller and the composite upscales; egui stays sharp.

**Hold the frame rate.** `DynamicResolution::on(60.0)` (a resource, off by default) multiplies
`FrameSettings::render_scale` by a step of 1, 0.85, 0.72, 0.6 or 0.5, chosen from the frame time
alone: down while frames are over budget and a lower resolution makes them faster, up again
when there is room. ([frame.md](graphics/frame.md#dynamic-resolution))

**Draw small models with fewer triangles.** `lod::simplify_to(&mesh, &extras, 0.25)` makes a
coarser mesh; `MeshLod::new(radius, &[full, mid, low])` on the entity lets the renderer set its
`Mesh3d` each frame from its size on screen (`LodSettings`). Shadow maps draw the same level. For
thin parts (fins, plates) that clustering holes, `lod::simplify_edges(&mesh, &extras,
&EdgeCollapse::to_ratio(&mesh, 0.3))` collapses edges by quadric error and keeps the outline.
([frame.md](graphics/frame.md#levels-of-detail))

**Preview a ship in the UI.** A render target (`add_render_target_with(w, h,
RenderTargetDesc::default().with_samples(4))`), a camera with a transparent clear and
`PostProcess` targeting it: the composite keeps alpha, egui shows it (`UiTextures`).

**Measure.** `FrameSettings::gpu_timing` fills `GpuTimings` (per-pass GPU ms, labels in
frame.md); `RenderStats` counts draws, passes, targets bytes; `ParticleStats` the particles.
The frame's CPU parts are trace spans named `render.*`, which the F4 profiler lists beside the
systems ([frame.md](graphics/frame.md#cpu-spans)).

**Build pipelines on the loading screen.** WebGL2 links a program in 50–150 ms: render one frame
of the loading scene, then `renderer.prewarm(&quality.prewarm_request(&materials))`.

## Overlay stages

| stage | when | target | gets |
|---|---|---|---|
| `OverlayStage::AfterTonemap` (default: every existing overlay, egui) | after every post group, FXAA and display view | the display target, display format | nothing: the old placement and pixels |
| `OverlayStage::HdrScene` (opt-in) | after the primary target's scene views, before its post groups (Direct mode: after its last view) | the scene color (`Rgba16Float`, `Rg11b10Ufloat`, or 8-bit in the Ldr fallback), scene size | distortion, exposure metering, bloom, tonemapping, grading, FXAA, grain |

An HdrScene overlay writes the encoding `prepare_scene` gives it (`OverlayScene::output`) and
reads the scene depth when `scene_reads()` contains `DEPTH`. Worked example — the game's additive
sprite overlay: [frame.md](graphics/frame.md#migrating-an-additive-overlay-the-games-additivers).

## WebGL2 and other downlevel devices

Every feature has a path without compute shaders and without storage buffers in vertex shaders.
Classes (spec §4.1): **A** native Vulkan/Metal/DX12 and WebGPU; **B** WebGL2 with float color
buffers (most desktop browsers; ANGLE); **C** WebGL2 without them; **D** GLES 3.1+ Android.

| feature | A | B (WebGL2 + float) | C (WebGL2, 8-bit) | D (GLES 3.1) |
|---|---|---|---|---|
| HDR scene, tonemapping | ✓ | ✓ | Ldr fallback: tonemapped per fragment, CPU-tonemapped clears | ✓ if float renderable |
| bloom | ✓ | ✓ | off (warned) | ✓ |
| auto exposure | compute histogram | render-pass chain | off | compute |
| FXAA, vignette, CA, grain, grading, LUT | ✓ | ✓ | ✓ | ✓ |
| distortion | ✓ | ✓ (unoccluded with MSAA) | ✓ | ✓ |
| material maps, blend modes, alpha mask | ✓ | ✓ | ✓ | ✓ |
| skybox, IBL bake | ✓ | ✓ (passes spread over frames) | ✓ (8-bit environment) | ✓ |
| directional shadow | ✓ | ✓ (CPU culling) | ✓ | ✓ |
| particles | GPU backend (compute) | CPU backend (16k native / 8k web) | CPU | GPU |
| ribbons and beams | CPU-built strips everywhere | ✓ | ✓ | ✓ |
| soft particles | ✓ | ✓ (fade off with MSAA) | ✓ | ✓ |
| GPU timings | ✓ | — | — | if offered |

`GraphicsQuality::recommended` picks **Low** on WebGL2 / GL, without float targets or without
compute, and `ParticleSettings::for_quality_on` gives the particles Low's CPU budget there whatever
the tier. Test any path on any GPU with `FrameSettings::downlevel` (`DownlevelOverride::webgl2()`
forces all three limits); `Engine/Showcase/WebGL2 *` are the showcase under it. On this Mac, ANGLE
runs the GL backend (`--features angle`, `WGPU_BACKEND=gl`; see the spec §5.0).

## Budgets

The reference scene **Battle** (spec §4.4): 300 ships of three hulls with ORM, normal and emissive
maps, a nebula skybox with IBL, one shadowed sun, ~20–40 point lights (the explosions' light
flashes), 100k particles with 64 ribbons, 16 beams and warp shockwaves, bloom, MSAA 4, two 256²
render-target previews — `space3d_stories::showcase::{spawn_battle, dress_battle}`, the
`Engine/Showcase/Battle` story, `benches/graphics.rs` and `tests/perf_graphics.rs` in
`space3d-stories`.

| gate (Dev: Apple M4 Max, Metal, 1080p, release) | budget | measured (High) |
|---|---|---|
| Battle GPU frame | ≤ 3.0 ms | 1.49 ms |
| Battle CPU `render_frame` | ≤ 1.5 ms | 1.45 ms (shared machine; see performance.md) |
| frame and post targets (T1/T2 VRAM line) | ≤ 220 MB | 133.5 MB |
| bloom 6 levels | ≤ 0.20 ms | 0.19–0.23 ms (shared GPU) |
| all post effects | ≤ 0.60 ms | 0.41 ms |
| particles 100k: sim + compaction / draw | ≤ 0.06 / 0.40 ms | 0.048 / 0.382 ms |
| shadow 2048², 100k casters | ≤ 0.30 ms | 0.28 ms |
| full-screen lit pixels, all maps + IBL + PCF9, over the old shader | ≤ +0.20 ms | +0.12 ms |
| IBL bake (LDR faces / HDR faces / HDR panorama) | ≤ 3 ms | 0.5–1.3 / 1.8 / 2.0–2.2 ms |

Per-tier Battle figures and the history of every gate: [performance.md](performance.md). Design
targets for other devices (T1 dGPU ≈ 5.8 ms, T2 Apple M1 ≈ 11 ms, T3 WebGL2 iGPU at Low ≈ 12.8
ms): spec §4.4. Steady state: no CPU reallocations, no textures created after warm-up, a static
scene uploads 0 bytes.

## Quality presets

`GraphicsQuality` sets the performance knobs of every area coherently; it never changes a game's
look otherwise (tonemapper, exposure, grading, bloom intensity, materials stay as authored).

| knob | Low (WebGL2 / C-class) | Medium (iGPU, Apple, mobile) | High (dGPU) | Ultra (opt-in) |
|---|---|---|---|---|
| render scale (primary) | 0.75 | 1 | 1 | 1 |
| MSAA / AA | 1 + FXAA Low | 4 on tile-based GPUs, else 1 + FXAA Medium | 4 | 4 |
| bloom mips / max mip height | 4 / 270 | 6 / 540 | 7 / 720 | 8 / 720 |
| auto exposure, chromatic aberration, film grain | off | as authored | as authored | as authored |
| shadow map / filter | 1024 / Pcf4 | 2048 / Pcf9 | 2048 / Pcf16 | 4096 / Pcf16 |
| IBL specular / irradiance / samples / passes per frame | 128 / 16 / 32 / 6 | 256 / 32 / 64 / all | 512 / 32 / 128 / all | 512 / 32 / 256 / all |
| shading model | Classic | Standard | Standard | Standard |
| particles (`ParticleSettings::for_quality`) | CPU 16k (8k web) | GPU 262k (65k mobile) | GPU 262k | GPU 262k |
| recommended sky face size | 512 | 1024 | 2048 | 2048 |

```rust
use space3d::render::GraphicsQuality;

let q = GraphicsQuality::recommended_for(renderer);   // Low on WebGL2/GL, no float or no compute;
                                                       // High on discrete GPUs; Medium otherwise
q.apply(world, resources);                            // FrameSettings, LightingSettings, MSAA,
                                                       // every PostProcess and DirectionalShadow
*resources.expect_mut::<ParticleSettings>() = ParticleSettings::for_quality_on(q, caps);
// Loading screen, after one frame of the scene:
renderer.prewarm(&q.prewarm_request(&materials));
```

* `apply` is lossless: the post settings a camera had before the first `apply` are remembered,
  so switching Low → High restores the chromatic aberration Low turned off. Cameras and lights
  spawned later take the tier with `q.post_for(post, msaa)` / `q.shadow(shadow)` (or `apply`
  again).
* The adapters (`frame_settings`, `msaa_samples`, `post`, `post_for`, `lighting`, `shadow`,
  `sky_face_size`) are there for games that store their own settings.
* **Ultra** is never recommended: a menu choice for big GPUs. It is High plus a 4096² PCF16
  shadow, one more (wider) bloom level (8 / 720) and a 256-sample environment bake. A level 0 is
  never larger than half the rect, so at 1080p High and Ultra both run their chains at 960 × 540
  (High 13 bloom passes, Ultra 15); before M2's fix High's 720 cap resampled level 0 up to
  1280 × 720 and cost more than Ultra.
* Map a game's own setting names to tiers (SpaceCorps2027: `low`/`medium`/`high` →
  `Low`/`Medium`/`High`): [migration.md](graphics/migration.md#2-quality-presets).

## Showcase, stories, tests

| where | what |
|---|---|
| `Engine/Showcase/SpaceCorps Look` | the close-up skirmish above; knobs: quality tier, shadows, IBL, bloom, distortion |
| `Engine/Showcase/Battle` | the reference scene at 320 × 180 (goldens), previews in the corner |
| `Engine/Showcase/WebGL2 SpaceCorps Look`, `WebGL2 Battle` | both under `DownlevelOverride::webgl2()` at Low |
| `examples/graphics_showcase` | the Battle, windowed (egui HUD, keys), headless PNG, web |
| `space3d_stories::showcase` | the content generators (hulls, hull maps, sky, asteroids) and scenes |
| `crates/space3d-stories/tests/perf_graphics.rs` | steady state and VRAM (always); the Dev gates (`--ignored`) |
| `crates/space3d-stories/benches/graphics.rs` | CPU per tier (criterion), `graphics/report` per tier |

Every area has its own stories (`Engine/Frame/*`, `Engine/Materials/*`, `Engine/Environment/*`,
`Engine/Lights/*`, `Engine/Post/*`, `Engine/Particles/*`) and goldens; the whole book runs as one
golden test (`cargo test -p space3d-stories --test storybook`), on Metal and on ANGLE (GL
variants `*.gl.png` where drivers differ).

![Battle](../crates/space3d-stories/tests/__screenshots__/Engine/Showcase/Battle.png)
![Battle on WebGL2](../crates/space3d-stories/tests/__screenshots__/Engine/Showcase/WebGL2_Battle.png)
