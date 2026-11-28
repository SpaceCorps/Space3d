# City effects API (fog, SSAO, SSR) - public surface

Status: implemented on branch `gfx/city-realism` exactly as listed here (the contract was written
first; nothing was renamed). Full documentation: `docs/graphics/post.md`, "City effects". Extras
beyond this list: `PostProcess::screen_effects()` (any of the three enabled), `Fog::opacity(camera_y,
point_y, distance)` (CPU mirror of the shader), `RenderStats::screen_fx_passes`, and the example
`cargo run -p space3d-render --example city_effects -- out_dir [w h]` (wet street, each effect on/off).

All three are opt-in fields of the camera's `PostProcess` component (same place as `bloom`,
`vignette`, ...). A camera whose `PostProcess` leaves them at their defaults (all
`enabled: false`) renders pixel-identical to before (spec D20). They run in the Hdr post chain,
on linear HDR radiance, **before bloom and exposure**, in the order SSR -> SSAO -> fog.
They need the scene depth, so they work only in `FrameMode::Hdr` on a camera with a
`PostProcess` (not in the Ldr fallback, not in Direct mode); elsewhere they are disabled
(warned once). All are Y-up (height = world Y).

```rust
use space3d_render::{Fog, Ssao, Ssr, PostProcess, GraphicsQuality};

let post = PostProcess::default()
    .with_fog(Fog::city_haze())          // or Fog { enabled: true, ..Fog::default() }
    .with_ssao(Ssao::on())               // Ssao::default() is disabled
    .with_ssr(Ssr::on());                // Ssr::default() is disabled
```

## `PostProcess` additions

```rust
pub struct PostProcess {
    // ... existing fields ...
    pub fog: Fog,     // default Fog::default() (disabled)
    pub ssao: Ssao,   // default Ssao::default() (disabled)
    pub ssr: Ssr,     // default Ssr::default() (disabled)
}
impl PostProcess {
    pub fn with_fog(self, f: Fog) -> Self;
    pub fn with_ssao(self, s: Ssao) -> Self;
    pub fn with_ssr(self, s: Ssr) -> Self;
    pub fn without_fog(self) -> Self;
    pub fn without_ssao(self) -> Self;
    pub fn without_ssr(self) -> Self;
}
```

## `Fog` (exponential height + distance fog, sun in-scatter)

Optical depth along the view ray with density `density * exp(-height_falloff * (y - base_height))`
integrated analytically; transmittance `T = exp(-tau)`; `opacity = min(1 - T, max_opacity)`.
`result = mix(scene, fog_color, opacity)`, where `fog_color = color + sun_color * sun_intensity *
pow(max(dot(view_dir, -sun_dir), 0), sun_exponent)`.

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    pub enabled: bool,          // false
    pub color: Vec3,            // linear in-scatter colour, default (0.60, 0.70, 0.82)
    pub density: f32,           // extinction per world unit at base_height, default 0.004
    pub height_falloff: f32,    // 1/units; 0 = uniform fog, default 0.05
    pub base_height: f32,       // world Y where `density` applies, default 0.0
    pub start_distance: f32,    // no fog before this camera distance, default 0.0
    pub max_opacity: f32,       // cap 0..=1, default 1.0
    pub sun_color: Vec3,        // default (1.0, 0.80, 0.55)
    pub sun_intensity: f32,     // 0 = no sun in-scatter, default 0.0
    pub sun_exponent: f32,      // lobe tightness, default 8.0
    pub sun_direction: Option<Vec3>, // direction the light TRAVELS; None = the view's first directional light
    pub sky_amount: f32,        // fog on sky pixels (depth = far): 0 none .. 1 full at the horizon, default 1.0
    pub sky_falloff: f32,       // how fast the sky fog fades with elevation (per unit of dir.y), default 3.0
}
impl Fog {
    pub fn city_haze() -> Self;                 // enabled, preset for the demo: light blue haze, sun_intensity 0.6
    pub fn with_density(self, d: f32) -> Self;
    pub fn with_height(self, base_height: f32, falloff: f32) -> Self;
    pub fn with_color(self, c: Vec3) -> Self;
    pub fn with_sun(self, color: Vec3, intensity: f32, exponent: f32) -> Self;
    pub fn with_start(self, d: f32) -> Self;
}
```

## `Ssao` (GTAO-style, half resolution, bilateral blur)

Reads the resolved scene depth; view-space normals are reconstructed from depth. The blurred
occlusion multiplies the HDR scene colour (ambient and direct alike are scaled by
`mix(1, ao, intensity)`; a forward renderer has no separate ambient buffer).

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ssao {
    pub enabled: bool,       // false
    pub radius: f32,         // world units, default 1.0
    pub intensity: f32,      // 0..=2, default 1.0
    pub power: f32,          // contrast exponent on the AO term, default 1.5
    pub bias: f32,           // horizon bias (fraction of radius), default 0.05
    pub max_distance: f32,   // fades out beyond this view distance, default 150.0
    pub slices: u32,         // 1..=4 direction slices, default 2
    pub steps: u32,          // 2..=12 samples per slice side, default 6
    pub half_res: bool,      // default true
    pub blur_radius: u32,    // bilateral blur taps each side, 0 = no blur, default 3
}
impl Ssao {
    pub fn on() -> Self;                         // enabled with defaults
    pub fn with_radius(self, r: f32) -> Self;
    pub fn with_intensity(self, i: f32) -> Self;
    pub fn with_quality(self, slices: u32, steps: u32) -> Self;
}
```

## `Ssr` (screen-space reflections)

Linear ray march in view space against the scene depth with a thickness test and a binary
refinement; the hit colour is the scene colour of the same frame. It needs surface normal,
roughness and reflectivity, which the renderer writes into a small G-buffer (`Rgba8Unorm`:
octahedral normal, roughness, F0) **only for cameras with `ssr.enabled`**, by an extra
depth-tested pass over the opaque draws after the scene pass. Reflection strength is Fresnel
(Schlick, from the material F0) x roughness fade x edge fade. Misses and the screen-edge fade
leave the pixel as the forward pass shaded it, i.e. with the existing IBL specular (cubemap)
reflection; on a hit the IBL specular term is replaced by the hit colour. Needs
`GraphicsCaps::read_only_depth` (not GL/WebGL2); disabled there.

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ssr {
    pub enabled: bool,        // false
    pub max_distance: f32,    // world units the ray travels, default 80.0
    pub steps: u32,           // 8..=128 march steps, default 48
    pub thickness: f32,       // view-space depth tolerance of a hit (world units), default 0.4
    pub intensity: f32,       // 0..=2, default 1.0
    pub max_roughness: f32,   // no reflection from rougher surfaces (fades from 0.6x of it), default 0.55
    pub edge_fade: f32,       // fraction of the screen near the border faded to IBL, default 0.1
}
impl Ssr {
    pub fn on() -> Self;                         // enabled with defaults
    pub fn with_distance(self, d: f32) -> Self;
    pub fn with_steps(self, s: u32) -> Self;
    pub fn with_intensity(self, i: f32) -> Self;
    pub fn with_max_roughness(self, r: f32) -> Self;
}
```

## `GraphicsQuality` integration (`GraphicsQuality::post(base)` / `post_for`)

Switching tiers is lossless (the authored values are kept, as for the other post settings).

| Tier | Fog | SSAO | SSR |
|---|---|---|---|
| Low | kept, sun in-scatter off | forced off | forced off |
| Medium | as authored | on if authored: `slices 1, steps 4`, half res, blur 2 | forced off |
| High | as authored | as authored, `slices 2, steps 6` (capped) | on if authored: `steps` capped at 32 |
| Ultra | as authored | as authored (capped at `slices 3, steps 8`) | as authored |

Ldr fallback / WebGL2 / Direct mode: all three disabled (warned once). SSR alone is also disabled
without read-only depth attachments; blended (alpha) materials are not in the SSR G-buffer, so model
glass as an opaque smooth dark metallic material.

## Notes for the demo

* Fog needs no extra textures; SSAO needs a half-res R8 pair; SSR adds one full-res Rgba8 G-buffer.
* Fog is applied last, so fogged distance hides SSR/SSAO as it should.
* Keep `Bloom::threshold` above the fog-lit sky value or the horizon haze will bloom.

---

# Round 2 (same branch): tiling, sky ground, better SSR

## 1. Tiling a wall texture at real-world scale (existing + two helpers)

The engine already has `Material::with_uv_transform(TextureTransform { scale, offset, rotation })`
(KHR_texture_transform): it applies to **every map** of the material (base, normal, ORM,
emissive) and the default sampler is `AddressMode::Repeat`, so a 1 k texture tiles without an
atlas. What was missing was UVs in world units. New:

```rust
MeshData::cuboid_tiled(size: Vec3, tile: f32) -> MeshData  // box, UV = metres / tile on every face
MeshData::plane_tiled(size: f32, tile: f32) -> MeshData    // XZ plane, UV = metres / tile
Material::with_tiling(u: f32, v: f32) -> Material          // uv_transform.scale = (u, v)
```

A facade of any size with `cuboid_tiled(size, 4.0)` and `.with_tiling(2.0, 2.0)` repeats the
texture every 2 m. Use `RenderAssets::add_image` / `add_texture` with the default (Repeat)
sampler; mips are generated, so it does not blur or alias with distance. Not applicable to
instanced atlas rects (those multiply after the transform).

## 2. Skybox ground / horizon

```rust
Skybox::new(cube).with_ground(color: Vec3 /*linear*/, blend_height: f32)
pub ground: Option<(Vec3, f32)>   // default None = unchanged
```

Below the horizon (view direction Y < 0, world up) the sky is replaced by `color`, blended over
`-blend_height..blend_height` of the direction's Y (0.05..0.1 is a soft horizon). Set it to the
fog colour (times the sky's brightness) so a flat ground plane fogged to the same colour meets
the sky with no seam. (`Fog::sky_*` still fogs the sky itself toward `Fog::color`.)

## 3. SSR for wet streets (new `Ssr` fields, defaults shown)

```rust
pub roughness_blur: f32, // 1.0  blur of the hit grows with roughness and ray length (0 = mirror)
pub stretch: f32,        // 3.0  vertical stretch of that blur: lights smear into glare streaks
pub hdr_boost: f32,      // 1.0  extra gain for reflected radiance above display white (up to 1+boost at 4x)
pub denoise: u32,        // 2    depth-aware blur taps each side of the reflection (0..=4), no history
Ssr::with_blur(roughness_blur, stretch), with_hdr_boost(b), with_denoise(taps)
```

Hit radiance is clamped at 64 before averaging (fireflies). `max_roughness` (default 0.55) still
gates the effect (fade from 60 % of it); raise it (0.7-0.8) together with `roughness_blur` for
rough asphalt. Reflections of back-facing surfaces (the ground the ray leaves, the back of a wall)
are rejected via the G-buffer normal. **Alpha-blended glass is not supported**: the G-buffer and
the depth are those of the opaque surface behind it. Model glass as an opaque smooth dark metallic
material (`with_metallic_roughness(0.9, 0.04)`); this is a documented limitation.

## 4. Vertex colours (baked AO): already supported

`MeshData::colors: Vec<[f32; 4]>` (linear RGBA per vertex; empty = white) multiplies the base
colour (`base = material.base_color * vertex_color * texture`), so a baked-AO multiplier in
rgb darkens diffuse and its ambient/IBL alike (not emissive, not the specular IBL). Just fill
`colors` before `add_mesh`; `MeshData::cuboid_tiled` leaves it empty for you to fill (see the
`city_effects` example).

## 5. GPU time (measured, 1080p, RTX-class adapter, debug build, `cargo run --example city_effects -- out 1920 1080`)

| scope | ms |
|---|---|
| SSAO (`post.ssao`, half res) + blur (`post.ssao_blur`, 2 passes) | 0.09 + 0.03 |
| SSR G-buffer pass (`gbuffer`) | 0.01 |
| SSR march (`post.ssr`) + denoise (`post.ssr_blur`, 2 passes) | 0.09 + 0.04 |
| apply pass (fog + AO + SSR add, `post.screenfx`) | 0.012 |
| scene pass, for scale | 0.03 |

(An easy scene; the numbers scale with pixel count and, for SSR, with ray length and roughness.)
Memory at 1080p: SSR adds an Rgba8 G-buffer (8 MB) and two Rgba16F reflection targets (33 MB each,
pooled and shared with other post users); SSAO two R8 half-res (0.5 MB each).

## Stale artifacts in the shared cargo target (important)

Two worktrees building the same crate in `D:/cache/cargo-target` can produce the SAME artifact
hash, and the later build overwrites the former's `libspace3d_render` without invalidating the
other's fingerprint: `cargo` then compiles your tests against the other worktree's library (symptoms:
"cannot find type `Fog`", missing struct fields). Fix: `touch crates/space3d-render/src/lib.rs`
(and any crate whose API the other tree changed) and rebuild. The last worktree that built
`space3d-render` from this branch is `D:/git/SpaceCorps/Space3d-wt-city`.
