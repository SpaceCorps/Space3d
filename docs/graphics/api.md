# Graphics API: compatibility policy and ownership

This is the contract of the graphics upgrade (normative spec: [`docs/design/graphics.md`](../design/graphics.md),
§2 and §3.6). The public API of every area was frozen at milestone F2: every type, trait and
function existed and compiled from then on, with neutral stubs (`STUB(<owner>)` comments, never
panicking) until the owning area implemented it. Since M2 nothing is a stub any more: the tree
has no `STUB(` marker, `todo!` or `unimplemented!` in `crates/` or `examples/`, and every area
(frame, lighting, shadows, IBL, post, particles) runs.

## Semver policy (spec §2.7)

Code written against Space3d 80ab72b (what SpaceCorps2027 pins at its HEAD b120d9b; it pinned
249c874 when the spec was written, and `api_compat_249c874.rs` keeps that name) keeps compiling
and renders the same pixels by default — with up to 16 point lights. With more, each view is lit
by the 16 nearest to its camera instead of the first 16 in query order (spec D19, §2.7): a scene
with more than 16 point lights renders differently (the farthest lights go unlit instead of the
last queried). Migration notes must say so.
`crates/space3d/tests/api_compat_249c874.rs` compiles the game's own forms of every touched type;
`crates/space3d/tests/prelude_policy.rs` checks the prelude additions against every type name of
the game.

* **Config structs** (`FrameSettings`, `CameraFrame`, `RenderTargetDesc`, `DownlevelOverride`,
  `PostProcess` and its effect structs, `LightingSettings`, `IblSettings`, `DirectionalShadow`,
  `TextureTransform`, `ParticleSettings`, the particle data model, `ModelUploadOptions`) are
  plain `pub`-field structs with `Default`. Construct them with functional record update
  (`..Default::default()`) or the builders: new fields may be added. Exhaustive struct literals
  of them are not covered by the compatibility guarantee.
* **`Skybox` and `EnvironmentLight`** have public fields but no `Default` (they need a cubemap):
  build them with `Skybox::new(cubemap)` / `EnvironmentLight::new(cubemap)` and the builders, or
  with functional record update from one of those (`..Skybox::new(cubemap)`).
* **Enums** added by the upgrade are `#[non_exhaustive]` (new variants may be added): match them
  with a wildcard arm.
* **Read-only info and context structs** (`GraphicsCaps`, `GpuTimings`, `GpuScope`, `FrameInfo`,
  `ViewInfo`, `PassTarget`, `MeshRef`, `TextureRef`, `SceneDepthRef`, the node contexts,
  `FrameLayouts`, `OverlayScene`, `RenderModel`, `RenderPart`, `MeshExtras`, `EffectState`) are
  `#[non_exhaustive]`: read their fields, never construct them.
* **Bit sets** (`SceneReads`, `SlotMask`, `MapMask`) are newtypes with `const`s and `BitOr`.
* **Existing types that gain fields** — the only ones, and all "FRU-only" downstream:
  `Material`, `RenderStats`, `space3d_assets::MaterialData`, `space3d_assets::ModelMesh`. Their
  new fields default to the legacy behaviour (`MaterialData`/`ModelMesh` fields are
  `#[serde(default)]`, so old RON/JSON still loads). The evidence is a grep of the game at
  revisions 4159bb8, 2f160d1 and 7267c34 (its HEAD at the M1 review, which type-checks against
  `graphics`): none of them is built with an exhaustive literal.
* **`space3d_assets::Primitive` stays frozen** (`mesh`, `material`): the game builds it with
  exhaustive literals (`client/src/game/models.rs`, splitting double-sided panels). Its extra
  vertex streams (glTF `TANGENT`, `TEXCOORD_1`) live beside it in
  `ModelMesh::extras: Vec<PrimitiveExtras>` (`extras[i]` belongs to `primitives[i]`, matched by
  index). Code that removes, splits or reorders primitives must keep `extras` in step or clear
  it: a stream whose length is not the primitive's vertex count is ignored (per-pixel tangents
  then), but a shifted stream of the same length — a removed primitive followed by one with as
  many vertices, as the game's `resolve_coincident` does with `retain` + `extend` — is used for
  the wrong primitive. `ModelMesh::primitive_extras` checks the length only.
  `api_compat_249c874.rs` compiles the game's literal.
* **Frozen existing types** (unchanged): `Camera3d`, `Projection`, `RenderTarget`,
  `DirectionalLight`, `PointLight`, `AmbientLight`, `MeshData`, `InstancedMesh`,
  `InstanceLayout`, `RenderSettings`, `RendererConfig`, `RenderCapabilities`, `DrawPath`,
  `SamplerSettings`, `OverlayTarget`'s fields, `OverlayTargetKind`, `RenderPlugin`,
  `DefaultPlugins` / `MinimalPlugins`, `GltfLoader`, `space3d_assets::Primitive`. New
  configuration lives in new components and resources.
* **Traits**: `RenderOverlay` gains provided methods only (`stage`, `scene_reads`,
  `prepare_scene`): existing impls compile and draw exactly as before (`AfterTonemap`).
* **Preludes**: `space3d_render::prelude` (and so `space3d::prelude`) gains exactly `PostProcess`,
  `Tonemapper`, `Bloom`, `CameraFrame`, `Skybox`, `EnvironmentLight`, `DirectionalShadow`,
  `CubemapHandle`. Everything else is reached through `space3d_render::*` /
  `space3d::render::*`. Particles have their own prelude (`space3d::particles::prelude`, facade
  feature `particles`) and are never in the facade prelude.
* After F2, renames and removals of these items need the foundation maintainer's sign-off and an
  update of the compatibility fixtures; additions (fields on the FRU types above, variants of
  non-exhaustive enums, provided trait methods, new items in an area's hub) are free.

## Where the API lives

`space3d-render`'s crate root re-exports each area's hub, so areas never edit `lib.rs`:

| hub | area | contents |
|---|---|---|
| `settings.rs` | frame (F) | `FrameSettings`, `CameraFrame`, `HdrMode`, `DownlevelOverride`, `RenderTargetDesc`, `RenderTargetFormat`, `UploadBudget` |
| `caps.rs` | F | `GraphicsCaps` |
| `output.rs` | F | `OutputEncoding`, `Tonemapper`, `OUTPUT_WGSL` |
| `node.rs` | F | the render-node API (`RenderNode`, `RenderNodes`, contexts, `PassTarget`, `FrameLayouts`, `VIEW_WGSL`, `scene_wgsl`, `FULLSCREEN_WGSL`) |
| `unproject.rs` | F | `inverse_proj_rot`, `unproject_ray`: directions and rays from the `f64` inverse projection and view rotation, exact at any distance from the origin (`View::inv_proj_rot`, [`frame.md`](frame.md#the-sky-does-not-depend-on-the-cameras-position)). Added after F3 without touching any existing item |
| `overlay.rs` | F | `OverlayStage`, `OverlayScene`, `RenderOverlay` provided methods |
| `timing.rs`, `quality.rs` | F (quality: I) | `GpuTimings`, `GpuScope`, `GraphicsQuality` |
| `material.rs` | P-MAT | `BlendMode`, `ShadowMode`, `MapMask`, `TextureTransform`, `MeshExtras` |
| `lighting/api.rs` | P-MAT | cubemaps, `Skybox`, `EnvironmentLight`, `DirectionalShadow`, `LightingSettings` |
| `post/api.rs` | P-POST | `PostProcess` and every post type, `DISTORTION_WGSL` |
| `lod.rs` | F | `MeshLod`, `LodSettings`, the `lod` module (`simplify`, `simplify_to`, `simplify_edges`, `EdgeCollapse`, `UV_REACH`, `MAX_LOD_LEVELS`) |
| `dynamic_resolution.rs` | F | `DynamicResolution` |

Crate-internal contracts between the frame and the areas (spec §2.6) have fixed signatures:
`lighting::Lighting` (+ `DepthViewRequest`, `ViewResources`, `LightingPrepare`),
`post::PostState` (+ `PostGroupInput`, `PostGroupPlan`, `PostGroupNeeds`, `PostEncode`),
`frame::extract::source_casts`, `frame::targets::TexturePool`. They started as stubs at F2; all
are implemented (lighting and shadows by P-MAT, post by P-POST, the rest by F).

Additions made by the foundation since F2 (additive):

* F4: `PostState` has a working composite-only chain (exposure, tonemapper, dither,
  alpha, rect mapping, upscale). `PostState::plan_group(world, plan, reallocs)` writes the
  group's `GpuPost` slot (in call order) and `PostState::flush(device, encoder, uploader)`
  uploads the slots before any `encode_group`. `PostGroupPlan` carries unclipped `[i32; 4]`
  rects plus `scene_size`, `output_size`, `scene_alpha`, `transparent` and `passthrough`.
  `PostGroupInput` adds `index` (the plan slot), `scene_generation` (bind-group cache key) and
  `output_viewport` (the drawn rect; `output_rect` is its scissor). `PostEncode` adds `pool`
  (the frame's `TexturePool`, released after each group).

* F5: `PostGroupPlan::distortion` (the group's Distortion phase runs this frame: the composite's
  `DISTORTION` bit), `PostState::distortion_target` counts created buffers (reported through
  `end_frame` in `textures_created`), `PostState::distortion_view(camera)` returns the existing
  buffer (a cheap view clone, its size and generation) for the frame's passes and replays, and
  `PostState::prewarm(device, format, manual_srgb)` builds a composite pipeline ahead of use
  (`Renderer::prewarm`). The composite binds `PostGroupInput::distortion` at group 1 binding 4
  (input bind groups are keyed by `(scene_generation, distortion generation)`).
  `FrameLayouts::scene` has a third binding: a non-filtering nearest sampler (binding 2) through
  which `scene_wgsl`'s `s3d_scene_depth` samples: WebGL2 depth textures read as 0 through
  `textureLoad` (it uses the texture's own linear filter state there, and GLES 3.0 depth
  textures are incomplete unless filtered nearest). Node shaders declaring only bindings 0–1
  keep working.
* F5: `frame::extract::material_class(&Material) -> u8` is the one place pipeline class bits
  come from (P-MAT extends it for bits 2..=5; `Renderer::prewarm` uses it).
* F-core review (spec §1.8, §1.10, §2.2.4, §2.2.5 amended):
  * `scene_wgsl_for(DepthAccess)` (crate root) and `OverlayScene::{depth_access(), scene_wgsl()}`:
    HdrScene overlays compose the group-1 module of their pass (`scene_wgsl` needs a
    `PassTarget`, which only node contexts carry).
  * `PostState::distortion_target`'s `rect` is the post group's rect in `S` (`D` rect × render
    scale), as the frame has always passed it; the buffer is half (or all) of that.
  * Capture replays render the scene views of an intermediate primary group into `S` when the
    frame skipped them (no surface texture); `GroupTextures::scene_frame` records the frame `S`
    holds. `capture_scene_hdr` does the same, and errs when the last frame had no primary view.
* F-core review, round 2:
  * `Renderer::shader_warning(ShaderHandle) -> Option<&str>` is new and additive. A custom
    shader that uses what GLSL ES 3.00 lacks (`countOneBits`, `pack4x8unorm`,
    `textureNumLevels`, `textureGather*`, …) is rejected only on WebGL2 / GLES 3.0 devices (GL
    without compute shaders), where it cannot build. On every other device it builds and draws
    as on `main`, and `shader_warning` (plus one `tracing` warning) reports the portability
    problem. F0 had rejected such shaders on every backend, which changed pixels for shaders
    that ran on Metal, Vulkan, DX12 and WebGPU.
  * `FrameLayouts::view` holds binding 0 (`View`) only. The lighting bindings 1–6 of spec §3.4
    join it with the lighting wiring (F7 / P-MAT PM4), and their docs say so. *(Done at F7:
    bindings 1–6 are in the layout.)*

* F7 / F8 (additive):
  * `FrameLayouts::view` has the lighting bindings 1–6 (environment cubes, BRDF LUT and sampler,
    shadow map and comparison sampler); shaders declare only what they use.
  * `PrepareContext::{compute_timestamps, render_timestamps}` and
    `ViewPrepareContext::compute_timestamps` return writes of lifetime `'a` (the frame's query
    set), no longer borrowing the context: pass them to `ctx.encoder`. Existing callers compile
    unchanged.
  * `GpuTimings` is filled (`FrameSettings::gpu_timing`), 2–5 frames late (the spec said 1–3;
    see `docs/graphics/frame.md` "GPU timings").
  * `space3d_render::testing` (doc-hidden, not public API): the depth-view injector of the
    renderer's own tests.

* Graphics review (visual, perf, compat), all additive:
  * `space3d_core::App::add_system_ordered(stage, order, name, system)`: a stage's systems run
    by ascending `order`, ties in registration order (`add_system` / `add_system_named` use 0).
    `ParticlesPlugin::TICK_ORDER` (1000) runs the particle tick after gameplay.
  * `PrepareContext::count_compute_passes(n)` (→ `RenderStats::compute_passes`, which the
    particles' GPU backend now reports).
  * `Tonemapper::AgXPunchy` (id 5, `S3D_TM_AGX_PUNCHY`): AgX with Sobotka's punchy look. The
    default stays `AgX`.
  * `SpriteRender::max_screen_size` (per layer, 0 = the setting);
    `ParticleSettings::max_screen_size` defaults to 1.0 (was 0.6).
  * `space3d_assets::{ModelMesh::extras, ModelMesh::primitive_extras, PrimitiveExtras}` replace
    the `tangents` / `uvs1` fields F2 had added to `Primitive` (that broke the game's exhaustive
    `Primitive { mesh, material }` literals; `Primitive` is frozen again).
  * `PostProcess::with_bloom(b)` sets `b` as given: `enabled: false` stays off (it used to force
    `true`; the presets are enabled, so `with_bloom(Bloom::glow())` is unchanged).
  * `RenderStats::staging_chunks_created`: staging chunks the upload ring created this frame (0 in
    the steady state; the ring now keeps what the frames queued on the GPU need).
* Graphics review, round 1 (all additive or behavior-only):
  * `render_frame` keeps at most three frames in flight: without a window to pace it (headless,
    an unpaced `App::update` loop) it waits for the frame submitted three frames earlier, which
    is usually done already. No API change.
  * Particle emitters may move in any stage: the render node lays each moved emitter's
    world-space emission on to its render pose (the render-time bridge, particles.md "Time
    model"). `ParticlesPlugin::TICK_ORDER` is unchanged.
  * `space3d_test::TestApp::with_renderer(f)` is public: `f` runs with the renderer in the
    resources also under `render_on_demand`. `story_app` applies stories that way, so stories
    that size render targets from `Renderer::size` get the capture size.
  * `builtin::ENGINE_EXHAUST` is authored near-neutral and takes its hue from the tint (it was
    blue, which a warm or green tint multiplied to grey); `builtin::LASER_TRAIL` reserves 576
    slots (was 1024) and has a smaller head.

* Phase I (integration), all additive:
  * `GraphicsQuality` implements the §4.4 presets and gains `Ultra` (the enum is
    `#[non_exhaustive]`), `ALL`, `name()`, `Display`, `recommended_for(&Renderer)`,
    `post_for(PostProcess, msaa_samples)`, `sky_face_size()`, `prewarm_request(&[Material]) ->
    PrewarmRequest` and `apply(&mut World, &mut Resources)`; `recommended` also returns Low
    without compute. `PostProcess::with_quality` maps `Ultra` to 8 bloom levels / 1080 and FXAA
    Ultra.
  * `space3d_particles::ParticleSettings::for_quality_on(GraphicsQuality, &GraphicsCaps)`: Low's
    settings where the GPU backend is unavailable.
  * `space3d_stories::showcase` (feature `engine`): the showcase content and scenes
    (`ShowcaseContent`, `ShowcaseAssets`, `spawn_look`, `spawn_battle`, `dress_battle`,
    `battle_look`, `headless_battle`, `measure_battle`, …).
  * `space3d-editor`: scene reprs and inspector sections for `CameraFrame`, `PostProcess`,
    `DirectionalShadow` (saved), `Skybox`, `EnvironmentLight` (snapshot-only: session-local
    cubemap handles); the `Material` repr gains the scalar map and blend fields.

* Performance (all additive; nothing changes for an app that uses none of it):
  * Levels of detail: the `MeshLod` component, the `LodSettings` resource (the defaults apply
    without it) and the public module `space3d_render::lod` with `simplify(&MeshData,
    &MeshExtras, cell)` and `simplify_to(&MeshData, &MeshExtras, ratio)`, both returning
    `(MeshData, MeshExtras)`. `RenderPlugin` adds the system `render::select_lods` to the Render
    stage before `render::render_frame`; it writes `Mesh3d` of entities that have a `MeshLod`
    and of no other. `simplify` does not weld across texture seams or between vertices whose
    normals face different ways, and keeps a triangle and its reverse as two.
    ([frame.md](frame.md#levels-of-detail))
  * Edge collapse: `lod::simplify_edges(&MeshData, &MeshExtras, &EdgeCollapse)` and the settings
    `lod::EdgeCollapse { triangles, max_error, border_weight }` (`EdgeCollapse::to`,
    `to_ratio`, `with_max_error`, `with_border_weight`), in `lod/collapse.rs`: a quadric-error
    simplifier for thin parts that clustering holes. Additive; `simplify` and `simplify_to` are
    as they were. ([frame.md](frame.md#levels-of-detail))
  * Levels that draw nothing: `MeshHandle::NONE` (with `MeshHandle::is_none()`), the handle of no
    mesh, which no view draws, and `MeshLod::from_levels(radius, &[Option<MeshHandle>])`.
    `MeshLod` keeps the level last asked for, and its `PartialEq` compares the levels and the
    sphere only. With a renderer, `render::select_lods` draws the nearest uploaded level in
    place of one whose mesh is not on the GPU yet. `MeshLod::with_level_of(&other)` starts an
    entity added later at the level a sibling was last asked for.
    ([frame.md](frame.md#levels-that-draw-nothing))
  * Upload budget: the `UploadBudget` resource (`bytes`, `headless`; `per_frame(bytes)`),
    inserted by `RenderPlugin` with no limit; textures upload before meshes, so no new mesh
    arrives while a texture waits. `RenderAssets::is_texture_uploaded(handle)` and
    `pending_upload_bytes()` beside `is_mesh_uploaded`;
    `RenderStats::{pending_meshes, pending_textures, pending_upload_bytes}`. Behaviour-only: the
    ECS path draws an entity while its mesh has a GPU copy, the test the `InstancedMesh` path
    already made; a mesh or texture replaced before its upload is queued once, at its first
    place. ([frame.md](frame.md#asset-uploads-and-the-upload-budget))
  * Dynamic resolution: the `DynamicResolution` resource (`enabled`, `target_fps`, `min_scale`,
    `probe_secs`; `on(fps)`, `scale()`, `frame(ms)`), inserted by `RenderPlugin`, off by
    default, with the system `render::dynamic_resolution` in the Render stage. On, it
    multiplies `FrameSettings::render_scale` for the frame (the resource itself is not
    written), so it reaches the Primary target's cameras only.
    ([frame.md](frame.md#dynamic-resolution))
  * The frame's CPU parts are `tracing` spans named `render.*`
    ([frame.md](frame.md#cpu-spans)). No API.

## Ownership map (spec §3.6)

A builder edits only the files it owns; everything else goes through the foundation maintainer
(`F`), additively. `F→X` files are written by F and owned by X afterwards.

| files | owner |
|---|---|
| root `Cargo.toml` (workspace deps), `crates/space3d-core/src/data.rs` (`HdrImageData`) | F |
| `crates/space3d-render/Cargo.toml`, `ARCHITECTURE.md` | F (I merges area summaries) |
| `src/{lib, settings, caps, output, node, overlay, timing, renderer, stats, testing}.rs` | F |
| `src/quality.rs` | F→I |
| `src/components.rs` (`Material` fields and docs) | F |
| `src/material.rs` | F→P-MAT |
| `src/frame/{mod, plan, views, targets, passes, nodes, depth, tests}.rs` | F |
| `src/frame/{extract, draws}.rs` | F→P-MAT |
| `src/pipeline/{mod, view, depth}.rs` | F |
| `src/pipeline/{forward, instance}.rs`, `src/{arena, assets, texture, handles}.rs` | F→P-MAT |
| `src/lighting/**` | F→P-MAT |
| `src/post/{mod, api, chain}.rs` | F→P-POST |
| `src/post/{bloom, exposure, fxaa, lut, grading, distortion, distortion_mesh, readback, pipelines}.rs` | P-POST |
| `src/shaders/{cull, compact, blit, clear}.wgsl`, `src/shaders/common/*`, `src/shaders/frame/*` | F |
| `src/shaders/forward.wgsl` | F→P-MAT (F owns the `//#VIEW` and `//#OUTPUT` regions) |
| `src/shaders/{depth, sky, ibl_bake}.wgsl`, `brdf_lut.bin` | F→P-MAT |
| `src/shaders/post/{composite, distortion}.wgsl` | F→P-POST |
| `src/shaders/post/{bloom, exposure, exposure_dl, fxaa, distortion_mesh}.wgsl` | P-POST |
| `tests/{frame_modes, nodes, overlay_stages, depth_views, gpu_timings, footprint, capture_hdr, alloc_frame, gl, alloc, headless, custom_shader}.rs`, `tests/common/mod.rs` | F |
| `tests/{materials, environment, shadows, gl_materials, alloc_materials}.rs`, `tests/common/materials.rs` (glTF import renders: `space3d-assets/tests/gltf_materials.rs`) | P-MAT |
| `tests/{post, post_distortion, gl_post, alloc_post}.rs`, `tests/common/post.rs` | P-POST |
| `benches/{instancing, frame}.rs` / `materials.rs` / `post.rs`; `crates/space3d-stories/{benches/graphics.rs, tests/perf_graphics.rs}` (the Battle scene needs particles and the showcase content) | F / P-MAT / P-POST; I |
| `examples/frame_nodes.rs` | F |
| `crates/space3d-assets/**` (incl. the `render` feature) | F (skeleton) → P-MAT |
| `crates/space3d-particles/**` | F (skeleton) → P-PART |
| `crates/space3d-stories/src/engine/{mod, basics, frame}.rs`, `Cargo.toml` | F |
| `crates/space3d-stories/src/engine/{materials, lighting}.rs` / `post.rs` / `particles.rs` / `showcase.rs` | P-MAT / P-POST / P-PART / I |
| goldens `tests/__screenshots__/Engine/{Frame}` / `{Materials, Environment, Lights/Directional_Shadow}` / `Post` / `Particles` / `Showcase` | F / P-MAT / P-POST / P-PART / I |
| `crates/space3d/{Cargo.toml, src/lib.rs, tests/api_compat_249c874.rs, tests/prelude_policy.rs}` | F |
| `crates/space3d-editor/src/components.rs` | F → I (reprs) |
| `examples/{pbr_showcase}` / `{post_processing}` / `{particles, particles_stress}` / `{graphics_showcase}` | P-MAT / P-POST / P-PART / I |
| `docs/graphics/{api, frame}.md` / `materials.md` / `post.md` / `particles.md` / `migration.md`, `docs/graphics.md` | F / P-MAT / P-POST / P-PART / I |

Change requests to F files: `scratchpad/graphics/change-requests.md` (file, change, reason), or
through the orchestrator.
