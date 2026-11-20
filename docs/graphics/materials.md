# Materials and lighting: maps, blend modes, glTF import, skybox, image-based lighting, shadows

`Material` renders lit or unlit surfaces with a base color, metallic/roughness, emissive and — since
the graphics upgrade — normal, metallic-roughness, occlusion and emissive maps, five blend modes,
per-material UV transforms and a second UV set. glTF models "just light up": every standard
material slot is read and converted. The normative design is
[`docs/design/graphics.md`](../design/graphics.md) §2.2.8, §2.2.9, §2.4 and §5.2 (P-MAT); this page
covers milestones PM1 (blend modes and maps), PM2 (glTF import and tangents), PM3 (cubemaps and the
skybox), PM4 (image-based lighting), PM5 (the directional shadow) and PM6 (prewarm, budgets, the
example). `examples/pbr_showcase` puts everything together in the game's top-down view.

A material without maps, with `BlendMode::Opaque` (or the legacy `alpha_blend`), renders exactly —
bit for bit — as before the upgrade: every map sits behind a uniform flag, and new settings default
to their neutral values.

## Maps

```rust
use space3d::prelude::*;
use space3d::render::{MapMask, TextureTransform};

let assets = app.resources.expect_mut::<RenderAssets>();
let base = assets.add_image(albedo);            // sRGB (color)
let normal = assets.add_image_linear(normals);  // linear (data)
let orm = assets.add_image_linear(orm);         // linear: R occlusion, G roughness, B metallic
let glow = assets.add_image(emissive);          // sRGB (color)
let material = Material::color(Vec4::ONE)
    .with_texture(base)
    .with_normal_map(normal, 1.0)               // scale of the normal's XY
    .with_orm_map(orm)                          // = metallic-roughness + occlusion, one sample
    .with_emissive_map(glow, Vec3::splat(4.0))  // factor × map; > 1 feeds bloom with post
    .with_uv_transform(TextureTransform { scale: Vec2::splat(2.0), ..TextureTransform::IDENTITY });
```

| field | texture | meaning |
|---|---|---|
| `base_color_texture` | sRGB | × `base_color` (alpha too) |
| `normal_texture`, `normal_scale` | **linear**, tangent space, +Y green up (glTF / OpenGL) | XY × scale, Z reconstructed (two-channel maps work) |
| `metallic_roughness_texture` | **linear**, G roughness, B metallic | × `roughness`, × `metallic` |
| `occlusion_texture`, `occlusion_strength` | **linear**, R | darkens ambient (and image-based) light only: `1 + strength × (R − 1)` |
| `emissive_texture` | sRGB | × `emissive` (so `emissive` must be non-zero, as in glTF) |
| `uv_transform` | — | `KHR_texture_transform` (offset, rotation, scale) of every map, before the per-instance `UvRect` |
| `uv1_maps` | — | maps (`MapMask`) sampled with the mesh's second UV set (`MeshExtras::uvs1`; UV0 without one) |

* **ORM packing.** When `occlusion_texture` is the same handle as `metallic_roughness_texture`,
  the texture is sampled once (R occlusion, G roughness, B metallic).
* **Color spaces.** Upload color maps (base color, emissive) with `add_image` (`Rgba8UnormSrgb`),
  data maps (normal, metallic-roughness, occlusion) with `add_image_linear`
  (`TextureSource::from_image_linear`, `Rgba8Unorm`). A data map uploaded as sRGB by mistake still
  works — the shader re-encodes the sample, losing precision, and a warning is logged.
* **Texture arrays.** The color maps (base color, emissive) and the data maps (normal,
  metallic-roughness, occlusion) each share one texture kind: single-layer or array. Arrays sample
  the per-instance `TextureLayer`, so per-instance skins can share one normal/ORM array. A map of
  the other kind within a group is not sampled (warned once).
* **Missing textures** (not uploaded yet, removed, unsupported format) bind a white default and
  their map is switched off in the material uniform: the factors alone apply.
* `LightingSettings::normal_maps = false` ignores normal maps globally (low-end devices).

### Tangents

Normal maps need a tangent frame:

* **Vertex tangents** (MikkTSpace, as baked maps expect): register the mesh with
  `RenderAssets::add_mesh_with(data, MeshExtras::new().with_tangents(t))`. The tangents (and a
  second UV set) live in the arena's **extension stream**: 16 bytes per vertex (`Snorm16x4`
  tangent, `Float32x2` UV1) in lockstep with the mesh arena, created with the first mesh that has
  extras. `space3d_assets::edit::mesh::compute_mikktspace_tangents(&mesh)` computes them (glTF
  orientation: `cross(normal, tangent) × w` points up the image, toward −v), splitting vertices
  where mirrored UVs meet. Run it off the frame (`spawn_compute`).
* **Per-pixel frames**: without vertex tangents the frame comes from screen-space derivatives
  (a cotangent frame) — correct for most maps, slightly different from a MikkTSpace bake along
  seams.

Instance layouts with more than 10 vertex locations leave no room for the extension stream: their
draws use per-pixel frames and UV0 (warned once per layout). Double-sided materials flip the whole
frame on back faces (glTF).

## Blend modes

```rust
use space3d::render::BlendMode;
Material::unlit(color).with_blend(BlendMode::Additive);
Material::additive(Vec4::new(1.0, 0.5, 0.2, 1.0));      // unlit, additive, double-sided glow
Material::premultiplied(Vec4::new(0.4, 0.2, 0.1, 0.5)); // unlit; a = 0 adds light
Material::default().with_alpha_mask(0.5);               // cut-outs
```

| `BlendMode` | color | alpha | depth write | phase |
|---|---|---|---|---|
| `Opaque` (default) | src | src (written as is) | yes | opaque |
| `AlphaMask` | src where alpha ≥ `alpha_cutoff`, else discarded | 1 | yes | alpha mask (after opaque) |
| `AlphaBlend` (= legacy `alpha_blend: true`) | `src·a + dst·(1−a)` | `a + dst·(1−a)` | no | transparent |
| `Premultiplied` | `src + dst·(1−a)` (store color × alpha) | `a + dst·(1−a)` | no | transparent |
| `Additive` | `src·a + dst` | dst (kept) | no | transparent |

`BlendMode::blend_state()` is shared with the particle renderer, so meshes and particles blend
identically. With MSAA, `AlphaMask` becomes **alpha to coverage** (smooth, sorted-free edges);
without it, it discards.

**Draw order** of a view (spec §1.7): opaque batches, alpha-masked batches, (sky), then
transparent batches by `Material::render_order` (lower first), alpha-blended and premultiplied
before additive at equal order, then by pipeline and material. `render_order` orders meshes only
(particles have their own order and draw after meshes).

**Per-instance sorting** is opt-in: `LightingSettings::transparent_sort =
TransparentSort::PerInstance { max_instances }` sorts the instances of alpha-blended and
premultiplied ECS batches (up to `max_instances` each) back to front per view, on every draw path
(those batches are culled on the CPU). The default `TransparentSort::None` keeps batch (spawn)
order. Additive batches never need sorting; explicit `InstanceBuffer`s keep their order.

## Shading models

`LightingSettings::shading` selects the lighting model of lit materials, frame-wide:

* `ShadingModel::Classic` (default): the original Blinn-Phong-style model, pixel-identical.
* `ShadingModel::Standard`: GGX distribution, height-correlated Smith visibility, Schlick Fresnel,
  Lambert diffuse, and the energy the single-scattering lobe loses to multiple scattering added
  back (`1 + f0·(1 / (A + B) − 1)` from the BRDF LUT, so rough metals do not darken) — in the same
  light units (a white Lambert surface facing a light of intensity 1 reflects 1). Use it with
  image-based lighting and for the "High" look.

Image-based lighting works with both; with `Classic` (Blinn-Phong highlights next to GGX
reflections) it is an approximation.

## Cubemaps and the skybox

```rust
use space3d::prelude::*;
use space3d::render::CubemapSource;

let assets = app.resources.expect_mut::<RenderAssets>();
// An HDR panorama (.hdr through `space3d::assets::HdrImageLoader`), converted on the GPU:
let nebula = assets.add_cubemap(CubemapSource::from_equirect_hdr(&hdr)); // faces: width / 4, ≤ 2048
// or six faces (+X, −X, +Y, −Y, +Z, −Z), or three.js-authored ones (`.skybox` manifests):
let sky = assets.add_cubemap(CubemapSource::from_faces(faces).unwrap());
world.spawn((Transform::IDENTITY, Camera3d::default(), Skybox::new(nebula).with_intensity(1.0)));
```

* **`CubemapSource`**: `from_faces` (sRGB, wgpu / OpenGL face order and orientation),
  `from_faces_threejs` (faces made for three.js' `CubeTextureLoader`: ±X swapped and every face
  mirrored, so they look as they do in three.js), `from_equirect_image` (sRGB panorama),
  `from_equirect_hdr` (linear `Rgba16Float`), `with_face_size` for panoramas, or
  `CubemapSource::Faces(TextureSource)` in any filterable format (e.g. `Rgba16Float`, BC6H).
  `space3d::assets::SkyboxLoader` loads a `.skybox` JSON manifest
  (`{"faces": [...6 images], "convention": "threejs"}`) into `CubemapData`, whose `to_source()`
  applies the convention.
* **Orientation.** A panorama's center column (u = 0.5) looks down −Z (the default camera's
  forward), +X is to its right, its top row is straight up. Faces follow the cube-map face table
  of every graphics API, which is left-handed: seen from inside in this right-handed world, face
  images appear mirrored (text reads backwards) — author them for OpenGL/wgpu, or use
  `from_faces_threejs` for three.js assets. `Skybox::rotation` turns the sky (a right-handed
  rotation of the sky, e.g. `Quat::from_rotation_y(a)`).
* **Upload.** The renderer uploads a cubemap in the frame after `add_cubemap` / `update_cubemap`
  into one *radiance cube*: a 6-layer texture with a full mip chain (a `TEXTURE_CUBE_MAP` on GL).
  HDR sources become `GraphicsCaps::hdr_sky_format()`: `Rg11b10Ufloat` where the device renders
  and filters it and compute shaders can write it (`compact_hdr_storage`: Apple GPUs, most
  desktop Vulkan and D3D12 drivers, through `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`)
  or nothing is computed — half the memory of `Rgba16Float`; its 6- and 5-bit mantissas change a
  sky's pixels by at most 3–4 levels of 255 after tonemapping — else `Rgba16Float` where that
  renders (`GraphicsCaps::hdr_format`; GL and WebGL2 without `RG11B10UFLOAT_RENDERABLE`, and
  WebGPU, whose compute cannot write `Rg11b10Ufloat`: there the sky's conversion and mips stay
  one compute pass at twice the memory, rather than 66 render passes); without float targets
  they become `Rgba8UnormSrgb` (radiance above 1 clamps; warned once); LDR ones are
  `Rgba8UnormSrgb`. Panoramas are converted face by face, sampling the panorama mip that matches
  the face resolution; the mips are computed from each other (a 2×2 average). All of it runs in
  that frame (not budgeted: the sky needs the whole cube) — compute dispatches (one per mip, all
  six faces) where compute shaders can write the cube's format, render passes (one per face and
  mip) elsewhere (see [WebGL2](#webgl2-and-gles)).
* **Skybox** (camera component): drawn by the built-in render node `"space3d/skybox"` in the
  Skybox phase, after the opaque and alpha-masked meshes and before transparent ones, at the far
  plane (`LessEqual`, no depth write), so it covers only what nothing opaque drew. It writes linear
  radiance × `intensity` — unclamped in HDR, so bright stars bloom with post — through the view's
  output encoding (tonemapped per fragment in the Ldr fallback). Orthographic cameras see a
  single direction. Replace the node under its name to draw the sky yourself.
* `RenderAssets::remove_cubemap` frees it; `Renderer::environment_state(h)` reports where it is
  (below).

Story: `Engine/Environment/Skybox` (a procedural HDR nebula, panorama or faces; intensity,
rotation, yaw).

## Image-based lighting

```rust
world.spawn((
    Transform::IDENTITY,
    Camera3d::default(),
    Skybox::new(nebula),
    EnvironmentLight::new(nebula).with_intensity(1.0, 1.0), // diffuse, specular
));
app.insert_resource(LightingSettings { shading: ShadingModel::Standard, ..Default::default() });
```

`EnvironmentLight` (camera component) lights every lit material the camera draws from a cubemap,
on top of `AmbientLight` and the lights:

* **Bake** (per cubemap, on the GPU, once): a GGX-prefiltered specular cube
  (`IblSettings::specular_size`, default 256², down to 8²; mip `m` holds roughness
  `m / (mips − 1)`; filtered importance sampling with `specular_samples` fixed Hammersley points)
  and an irradiance cube (`irradiance_size`, 32², `irradiance_samples` cosine-weighted points,
  stored as the mean radiance E/π). `Rgba16Float`; `Rgba8UnormSrgb` without float targets (the
  LDR bake). The split-sum BRDF LUT (64² `Rg16Float`, GGX with height-correlated Smith) is
  precomputed and embedded (`shaders/brdf_lut.bin`, regenerated by the ignored
  `regenerate_brdf_lut` test).
* **Shading** (`s3d_ibl` in `forward.wgsl`): specular = prefiltered radiance at roughness ×
  (f0·A + B) plus the multiply scattered part (Fdez-Agüera 2019), diffuse = irradiance × the albedo
  the specular lobes leave: a white or metallic surface in a uniform white environment reflects
  exactly its radiance (`ibl_white_furnace`, ±3 %). Occlusion maps darken both (specular with
  Lagarde's specular occlusion). `EnvironmentLight::rotation` turns the environment like
  `Skybox::rotation`.
* **When.** A cubemap's environment is baked the first frame a camera's `EnvironmentLight` uses it
  (skybox-only cubemaps never pay for it) — every step at once on headless renderers and by
  default, or `IblSettings::bake_passes_per_frame` steps per frame (web: `Some(6)`); the camera uses
  it from the frame its last step is recorded. `Renderer::environment_state(cubemap)`: `Missing`
  (not seen by a frame yet, or removed), `Baking { passes_left }` (uploaded; the steps left), `Ready`,
  `Failed` (the upload failed, logged). A re-uploaded cubemap or changed `IblSettings` re-bake while
  the previous environment stays bound; environments unused for 600 frames are dropped.
* Views without an `EnvironmentLight` bind 1×1 black cubes and skip the image-based term (a
  uniform branch): materials render exactly as before.

Stories: `Engine/Environment/IBL Spheres` (gold and red rows, roughness 0 → 1, in a procedural
photo studio or the nebula) and `Engine/Environment/Nebula Ship` (the test ship lit by the nebula it
flies in: metal panels reflect the nebula's core, bloom on the engines).

### Fill light (dark skies)

A space sky is dark: away from the sun and its corona, a nebula's cosine-weighted mean radiance
(what the irradiance cube stores) is a few hundredths. With `EnvironmentLight` at intensity 1 the
sides the sun does not reach — and everything in a cast shadow — get 1–2 % of the light of a
sunlit surface (a sun of intensity 2, `DirectionalLight` intensity being the light a white
surface facing it reflects), which after tonemapping is black: the hull's shape and colors are
lost in every shadow. Plan the fill as a ratio of the key light:

* **Fill ratio.** `irradiance × diffuse_intensity + ambient + fill lights` on a side facing away
  from the sun, over `sun intensity × cos` on a lit top. 10–20 % keeps a top-down space scene's
  shadows dark but readable; below 5 % they read as holes. Measure the sky's irradiance (the
  mean radiance of a hemisphere, cosine-weighted, sun excluded) or read it back from the baked
  environment.
* **Diffuse image-based lighting** carries the fill's color: raise
  `EnvironmentLight::diffuse_intensity` until `irradiance × intensity` is about 7–10 % of the sun
  (the showcase's nebula, irradiance ≈ 0.03: `ENV_DIFFUSE = 6`). The specular intensity is tuned
  on its own (reflections of a dark zenith: `ENV_SPECULAR = 2.5`).
* **A fill light** gives the shadowed sides shape: a shadowless `DirectionalLight` from the side
  opposite the sun and a little from below (the nebula's glow bouncing up), in a muted version of
  the nebula's hue, at about 15 % of the sun (the showcase: 0.35 lavender, (0.59, 0.53, 0.82),
  against a 2.2 sun; `spawn_fill`, `fill_direction`). It costs one light in the forward loop. Keep
  its color muted and let the diffuse lighting carry the nebula's: from a top-down camera the fill
  reaches few visible faces but crease walls at grazing angles, and a saturated violet (0.62, 0.5,
  1.0) outlined them in thin violet slivers.
* **Per tier.** The lower tiers darken shadows on screen: they lose bloom's veil (Medium's is
  smaller, Low's smaller still, none without float render targets) and, at Low, GGX shading. With
  one rig a top in a cast shadow read 6.4 % of a sunlit one at High, 4.3 % at Medium, 3.4 % at Low
  and 2.4 % on WebGL2 (the Look, 1080p). The showcase raises the diffuse intensity, the ambient and
  the fill by `showcase::fill_lift` (× 1.3 at Medium, × 1.8 at Low): 6.2 %, 8.6 % and 5.9 %.
* **`AmbientLight`** stays faint (0.03 in the showcase): it is flat, and what lights a scene
  before its environment is baked.

The showcase's rig (`space3d_stories::showcase::{add_environment, spawn_sun, spawn_fill}`) gives
a side facing away from the sun 15 % of a sunlit top's light, a side facing the sun 72 % and a
top in a cast shadow 17 % at High (`showcase::tests::fill_keeps_shadowed_sides_readable`; before
the M2 final review 1.6 % and 3 %).

`EnvironmentLight::new` keeps both intensities at 1: radiance-true, right for a bright sky (a
studio or daylight HDR, whose irradiance already fills the shadows). The dark-sky tuning above
is the game's choice, per sky.

## Directional shadow

```rust
use space3d::prelude::*;
use space3d::render::{ShadowFilter, ShadowFit};

world.spawn((
    Transform::IDENTITY.looking_at(Vec3::new(-0.5, -0.8, -0.3), Vec3::Y),
    DirectionalLight { color: Vec3::ONE, intensity: 2.5 },
    // Top-down games: fit the map to what the camera sees within |y − 0| ≤ 60.
    DirectionalShadow::new().with_fit(ShadowFit::top_down(0.0, 60.0, 2000.0)),
));
```

One directional light casts: the one carrying a `DirectionalShadow` (a component next to the
unchanged `DirectionalLight`). With several, the lowest entity casts and a warning is logged once.

* **The map** is a `map_size`² `Depth32Float` texture (rounded up to a power of two, clamped to the
  device; default 2048 — use 1024 on the web), drawn as a *depth view* before the camera views: it
  culls its casters like a camera (on GPU draw paths as one more frustum of the same culling
  dispatches; on `CpuDirect` on the CPU) and draws them instanced, depth only. `RenderStats`
  counts `depth_views`, `shadow_casters` and `shadow_draw_calls`; its GPU time is the `shadow`
  scope of `GpuTimings`.
* **Fit** (`DirectionalShadow::fit`), recomputed every frame on the CPU:
  * `ShadowFit::Camera { camera, max_distance, receivers }` (default: the first primary camera,
    200 units, no box) covers that camera's frustum from its near plane to `max_distance`,
    intersected with `receivers` when given. `ShadowFit::top_down(plane_y, half_thickness,
    max_distance)` is that with the slab `|y − plane_y| ≤ half_thickness` as the box: a camera
    looking down at a playfield gets a map just over its footprint, whatever its far plane.
  * `ShadowFit::Bounds(aabb)` covers a fixed box (arenas, previews).
  * The map is square around the covered points in a light basis anchored at the world origin;
    its size changes only in steps of 1/8 of a power of two while zooming, and its center snaps to
    whole texels, so panning never makes shadow edges shimmer. It extends `caster_depth` (500)
    toward the light, so casters outside the view (above the camera, off screen toward the sun)
    still cast.
  * `min_caster_pixels` (default 0: everything casts) leaves an instance out of the map when
    its bounding sphere is under that many pixels in radius on the screen of the camera the
    map follows (`ShadowFit::Camera`'s, else the first primary scene view; perspective cameras
    only). The depth view's frustum carries that camera's eye and a radius ÷ distance ratio
    (`Frustum::small`), tested with the frustum planes on the CPU and in the GPU culling. It is
    per instance: a model made of parts loses its small parts' shadows first. No hysteresis: an
    instance crossing the size gains or loses its shadow at once. A cut for cheap tiers: on an
    AMD Radeon iGPU a wave of 200 ships of 18 parts went from 0.78 ms of shadow pass to 0.22 ms
    at 10 px.
  * Receivers outside the map are lit; shadows fade out over its outer 10 %.
* **Filter** (`ShadowFilter`, a pipeline constant, so it costs only what it uses): `Hard` (one
  bilinear comparison, 2×2 texels), `Pcf4` (3×3), `Pcf9` (default, 5×5), `Pcf16` (7×7), smooth
  kernels built from bilinear comparisons (Castaño 2013). A wider kernel is a softer edge; the
  texel size sets the scale, so a 1024² map with `Pcf9` looks like a 2048² one with `Pcf16`.
* **Biases** (`with_bias(depth, normal, slope)`): `depth_bias` (0.02, world units along the light)
  is subtracted from the receiver's light depth; `normal_bias` (1.0, in texels) moves the receiver
  along its geometric normal, more at grazing angles; `slope_bias` (1.5) is the raster slope bias
  of the depth pass, scaled with the filter's reach (a wider kernel compares texels farther along
  a sloped receiver). The defaults leave no acne on flat ground lit at 45° and at 80° with every
  filter and both a fine and a coarse map (`no_acne_at_grazing_angle`). Raise `slope_bias` for
  acne on steep surfaces; lower `depth_bias`/`normal_bias` when thin casters detach from their
  shadows ("peter-panning").
* **`strength`** (1.0) scales the shadow: 0.5 lets half of the light through; 0 skips the depth
  pass entirely.
* **Who casts and receives** (`Material::shadows`, `ShadowMode::Auto` by default): lit opaque and
  alpha-masked materials cast (masked ones their cut-out shape) and receive; lit transparent ones
  receive but do not cast; unlit ones neither (sky quads, glows, particles). `CastAndReceive`,
  `CastOnly`, `ReceiveOnly` and `None` override that (an unlit material with `CastOnly` casts).
  Explicit `InstanceBuffer`s cast like entities. Only the casting light is shadowed; the others,
  ambient and image-based light are not.
* The camera views get the light matrix and filter parameters in `GpuView` (`shadow_view_proj`,
  `shadow_params`, `shadow_info`), the casting light moves to `dir_lights[0]`, and group 0 binds
  the map (binding 5, `texture_depth_2d`) with a comparison sampler (binding 6). Custom shaders may
  call `s3d_shadow_factor(world_pos, geometric_normal, light_dir)` (1 = lit).
* **Parity and determinism.** The shadow is the same, pixel for pixel, on every draw path and
  under `DownlevelOverride::webgl2()` (`shadow_same_on_every_path`); captures replay the frame's
  map and never render it again. Without a `DirectionalShadow`, pipelines and pixels are exactly
  those of a renderer without shadows.

Story: `Engine/Lights/Directional Shadow` — three test ships over an asteroid field and a rocky
ground in the game's camera angle, with knobs for the map size, filter, fit, biases, strength and
sun elevation.

## glTF import

The glTF loader (`space3d::assets::GltfLoader`, `load_gltf`) reads every material slot into
`MaterialData`: base color, metallic-roughness, normal (+ `scale`), occlusion (+ `strength`),
emissive (+ `emissiveFactor` × `KHR_materials_emissive_strength`), `KHR_materials_unlit`, alpha
mode and cutoff, double-sidedness, and per slot its `texCoord`, sampler and
`KHR_texture_transform`. Primitives' `TANGENT` and `TEXCOORD_1` go to `ModelMesh::extras`.

The render bridge (`space3d::assets::render`, facade feature `render`) turns a model into render
assets:

```rust
use space3d::assets::render::{ModelUploadOptions, prepare_model_async, spawn_model};

// Off the frame: MikkTSpace tangents for normal-mapped primitives without TANGENT.
let task = prepare_model_async(model, ModelUploadOptions::default());
// Later, when `task.poll()` is ready, on the main thread (registration only):
let render_model = prepared.upload(&mut assets);
spawn_model(&mut world, &render_model, Transform::IDENTITY);
```

| glTF | `Material` |
|---|---|
| `baseColorFactor` / `baseColorTexture` (sRGB) | `base_color` / `base_color_texture` |
| `metallicFactor`, `roughnessFactor` / `metallicRoughnessTexture` (linear) | `metallic`, `roughness` / `metallic_roughness_texture` |
| `normalTexture` (+`scale`, linear) | `normal_texture`, `normal_scale` |
| `occlusionTexture` (+`strength`, linear); same image as metallic-roughness | `occlusion_texture` (= the same handle: ORM packed), `occlusion_strength` |
| `emissiveFactor` × `KHR_materials_emissive_strength` / `emissiveTexture` (sRGB) | `emissive` (unbounded, HDR) / `emissive_texture` |
| `alphaMode` OPAQUE / MASK (+`alphaCutoff`) / BLEND | `BlendMode::Opaque` / `AlphaMask` + `alpha_cutoff` / `AlphaBlend` |
| `doubleSided`, `KHR_materials_unlit` | `double_sided`, `unlit` |
| `texCoord: 1` on a slot | its bit in `uv1_maps` |
| `KHR_texture_transform` | `uv_transform`: the base color's, else the first slot's (different transforms per slot: warned, the first applies) |
| sampler wrap / filter | the texture's `SamplerSettings` (one upload per image × color space × sampler) |

Games that build their own materials (e.g. a paint-scheme material per role) keep them and add the
maps with `apply_material_maps(&mut material, &data, &mut upload)`: normal, metallic-roughness,
occlusion and emissive maps, their scales and strengths, ORM packing, `emissive_strength` × the
material's `emissive` (only where it is non-zero or the slot exists), `uv1_maps` and the UV
transform. Base color, blend mode and the surface factors are left alone (roughness and metallic
factors multiply the map). `upload` should use `add_image` for `TextureRole::is_srgb()` roles and
`add_image_linear` otherwise.

The storybook shows the path end to end: `Engine/Materials/glTF Import` loads
`crates/space3d-stories/assets/ship.glb`, a small hull with baked base color, normal, ORM and
emissive maps and no `TANGENT` (generated by `space3d-assets/tests/gltf_materials.rs`).

## WebGL2 and GLES

Everything above works on WebGL2 / GLES 3.0: a material binds 5 textures and the view 4 more
(environment specular and irradiance cubes, BRDF LUT, shadow map: 9 of the fragment stage's 16
texture units), no storage buffers, and the extension stream is rebound with the mesh arena where
`BASE_VERTEX` is missing. Alpha to coverage needs MSAA (usually off on the web). The shadow map is
a depth texture sampled as a `sampler2DShadow` with `textureLod` (bilinear comparisons are core
GLES 3.0), culled on the CPU (`tests/gl_materials.rs`: `shadow_on_gl_matches_default_backend`,
every filter, within 1 % of Metal's pixels).

Cubemaps are square 6-layer textures, which wgpu's GL backend creates as `TEXTURE_CUBE_MAP`s. The
cubemap conversion, mip chain and bake have two paths computing the same texels (same functions,
same sample sets; `compute_bake_matches_fragment_bake`):

* **Compute** (Vulkan, Metal, DX12, WebGPU, GLES 3.1+ with compute): one dispatch per mip for all
  six faces, into write-only storage views (`Rgba16Float`; `Rgba8UnormSrgb` cubes are
  `Rgba8Unorm` textures written through their linear view, with the sRGB encoding in the shader;
  `Rg11b10Ufloat` radiance cubes, which cannot be storage textures, take the fragment path for
  their conversion and mips); a frame's dispatches share one compute pass.
* **Fragment** (WebGL2 / GLES 3.0, or `DownlevelOverride::no_compute`): one render pass per face
  and mip into a single-layer view (`framebufferTexture2D` with the face target on GL). Set
  `IblSettings::bake_passes_per_frame` (e.g. `Some(6)`) to spread a bake over frames.

Without float render targets (WebGL2 without `EXT_color_buffer_float`, or `no_float_targets`),
radiance and environment cubes are `Rgba8UnormSrgb` (values clamp at 1; LDR environments are
lossless, `ibl_ldr_fallback_close_to_hdr`). Verified on ANGLE (GLES 3.0): `tests/materials.rs`
and `tests/environment.rs` with `WGPU_BACKEND=gl`, `tests/gl_materials.rs` (GL vs Metal, including
`cube_on_gl_is_texture_cube_map`), and the storybook.

## Prewarm

Pipelines are built on first use; on WebGL2 each program link takes 50–150 ms. Call
`Renderer::prewarm` on a loading screen, after one frame of the scene (so the frame's pass
signatures, texture kinds and frame-wide settings are known), with the materials the game will
spawn:

```rust
let request = space3d::render::PrewarmRequest { materials: vec![hull.clone(), shield.clone()], ..Default::default() };
app.resources.expect_mut::<Renderer>().prewarm(&request);
```

Each material's forward pipeline is built with the class bits a batch of it gets on each
instance layout (blend mode, double-sided, the extension stream where the layout has room), its
maps' texture kinds as the last frame resolved them (single-layer for maps no frame has drawn)
and the last frame's frame-wide settings (shading model, normal maps, the casting light's
filter); casters also get their depth pipeline when the last frame drew a shadow. A frame after
the prewarm builds no pipeline for them (`prewarm_builds_what_materials_draw_with`). Changing
`ShadingModel`, `normal_maps` or the shadow filter re-specializes every lit pipeline once.

## Performance

* Materials are interned per entity per frame; a material equal to the previous entity's is
  answered by one comparison (`Material: PartialEq`), others by a small cache. Texture sets get
  one bind group each, rebuilt only when a texture or the materials buffer changes.
* Maps cost only the materials that have them: the samples sit behind uniform branches.
* Steady state: no allocation in the renderer's code with every feature above in use
  (`tests/alloc_materials.rs`); a static scene uploads nothing.
* `benches/materials.rs`: `materials/ecs_pbr/100k` (100k static entities, 8 PBR materials with
  maps; gate ≤ 2.5 ms per frame on the dev machine), `materials/ecs_plain/100k`,
  `materials/transparent_sort/4096`. Measured values are in `docs/performance.md`.
* Image-based lighting costs 3 texture samples per lit pixel, the skybox one per uncovered pixel,
  the shadow 1/4/9/16 comparisons per lit pixel. A 1080p screen of mapped, IBL-lit, shadowed
  (PCF9) Standard pixels costs +0.12 ms of GPU over the legacy shader on the dev machine
  (`forward_full_screen_gate`, gate +0.20 ms). The shadow map of 100k cube entities costs
  0.28 ms at 2048² (`shadow_map_gate`, gate 0.30 ms; the same cubes from an explicit instance
  buffer cost the same); the depth view adds 0.1 ms of CPU at 100k entities on GPU draw paths and
  0.4 ms on `CpuDirect` (`materials/shadow_view/*`). A bake is one-time: 1024² faces → 256² specular + 32² irradiance
  in well under a millisecond of GPU on the dev machine with compute (`ibl_bake_gate`, gate ≤ 3 ms
  for LDR faces, HDR faces and an HDR panorama, the first of each kind included, on a loaded GPU:
  0.3–0.4, 0.5–0.7 and 0.6–1.0 ms; 2–5 times that on an idle GPU at a low clock), many ms with render passes (each
  pass has a fixed cost of tens of µs to a few hundred on Metal, and holds driver memory while its
  frame is in flight: 102 of them for LDR faces, 108 for an `Rg11b10Ufloat` sky, against 17–18
  dispatches in one compute pass).
* Memory: a radiance cube is `faces² × 6 × 4/3` texels (4 bytes in sRGB and `Rg11b10Ufloat`, 8 in
  `Rgba16Float`: a 1024² sky ≈ 34 MB, 67 MB as `Rgba16Float`; 512² ≈ 8 MB — use 512² faces on the
  web), an environment ≈ 4.2 MB at
  the default sizes, a shadow map 16.8 MB at 2048² (4.2 MB at 1024²; kept 600 frames after its
  light stops casting); a still camera with a skybox, an environment and a shadow uploads nothing
  and allocates nothing per frame (the depth pass adds wgpu's per-pass bookkeeping, 37
  allocations per frame on Metal).
