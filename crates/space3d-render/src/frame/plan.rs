//! Frame planning (graphics spec §1.3): the frame's cameras in render order, their viewports in
//! whole pixels, and the frame plan — target groups, frame modes, post groups and view roles.

use super::FrameState;
use crate::assets::RenderAssets;
use crate::caps::GraphicsCaps;
use crate::components::{Camera3d, RenderTarget};
use crate::handles::TextureHandle;
use crate::node::{FrameMode, ViewRole};
use crate::output::{OutputEncoding, Tonemapper};
use crate::post::api::PostProcess;
use crate::settings::{CameraFrame, FrameSettings, HdrMode, RenderTargetFormat};
use glam::{Mat4, Vec4};
use space3d_core::{Entity, GlobalTransform, World};
use std::ops::Range;

#[derive(Clone, Copy)]
pub(super) struct CameraItem {
    pub(super) order: i32,
    pub(super) entity: u64,
    pub(super) id: Entity,
    pub(super) camera: Camera3d,
    pub(super) transform: Mat4,
    /// The camera's `PostProcess`, when it has one that is enabled.
    pub(super) post: Option<PostProcess>,
    pub(super) frame: Option<CameraFrame>,
}

impl FrameState {
    /// Active cameras, sorted by `(order, entity)` (the render order).
    pub(super) fn collect_cameras(&mut self, world: &World) {
        let f = self;
        f.cameras.clear();
        let mut query = world.query::<(
            Entity,
            &Camera3d,
            &GlobalTransform,
            Option<&PostProcess>,
            Option<&CameraFrame>,
        )>();
        for (e, cam, gt, post, frame) in query.iter() {
            if cam.active {
                f.cameras.push(CameraItem {
                    order: cam.order,
                    entity: e.to_bits().get(),
                    id: e,
                    camera: *cam,
                    transform: gt.0,
                    post: post.filter(|p| p.enabled).copied(),
                    frame: frame.copied(),
                });
            }
        }
        f.cameras.sort_unstable_by_key(|c| (c.order, c.entity));
    }
}

/// What a target group renders into: the primary target (window or offscreen) or a render
/// target texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GroupTarget {
    Primary,
    Texture(TextureHandle),
}

/// The display target `D` of the primary group this frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct PrimaryDisplay {
    pub(super) size: (u32, u32),
    pub(super) format: wgpu::TextureFormat,
    pub(super) samples: u32,
    /// The target has no sRGB format or view: shaders encode sRGB themselves.
    pub(super) manual_srgb: bool,
}

/// All views of one render target this frame (graphics spec §1.3).
#[derive(Clone, Debug)]
pub(crate) struct TargetGroup {
    pub(super) target: GroupTarget,
    pub(super) mode: FrameMode,
    /// Index (in `FrameState::cameras`) of the lead camera: the highest-order post camera, else
    /// the group's first camera.
    pub(super) lead: usize,
    /// The display target `D`: size, color format, samples of its own attachments, and whether
    /// shaders encode sRGB into it.
    pub(super) display_size: (u32, u32),
    pub(super) display_format: wgpu::TextureFormat,
    pub(super) display_samples: u32,
    pub(super) manual_srgb: bool,
    /// Scene color `S` of intermediate groups (`display_*` in Direct mode).
    pub(super) scene_format: wgpu::TextureFormat,
    pub(super) scene_size: (u32, u32),
    /// Samples of the scene passes (into `S`, or into `D` in Direct mode).
    pub(super) samples: u32,
    pub(super) scale: f32,
    /// Render-target group whose lead clears with alpha below 1: the composite keeps alpha.
    pub(super) transparent: bool,
    /// This frame's post groups (`FrameState::post_groups`), in settings-camera order.
    pub(super) post_groups: Range<usize>,
    /// Some view of the group is a display view.
    pub(super) display_views: bool,
    /// The view (index in `FrameState::views`) after whose pass the group is finished: its
    /// last scene view.
    pub(super) last_scene_view: Option<usize>,
    /// Views drawing into the group's scene-pass attachments this frame (every view in Direct
    /// mode, the scene views otherwise): with more than one, a multisampled group cannot
    /// resolve depth for single-sample passes between them (graphics spec §1.6).
    pub(super) scene_views: u32,
}

impl TargetGroup {
    pub(super) fn intermediate(&self) -> bool {
        self.mode != FrameMode::Direct
    }
}

/// One post group (graphics spec §1.3): a rect of an intermediate target group with a post
/// camera, or the implicit full-target group of an Ldr passthrough group.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FramePostGroup {
    pub(super) group: usize,
    /// Settings camera (index in `FrameState::cameras`).
    pub(super) camera: usize,
    pub(super) settings: PostProcess,
    pub(super) passthrough: bool,
    /// The rect in `D` and in `S`, unclipped.
    pub(super) display_rect: ViewportRect,
    pub(super) scene_rect: ViewportRect,
    /// How the group's scene views encode their color into `S`.
    pub(super) scene_output: OutputEncoding,
    /// For `Tonemapped` scene views: linear exposure multiplier and operator.
    pub(super) exposure: f32,
    pub(super) tonemapper: Tonemapper,
    /// The settings camera's view (index in `FrameState::views`), once prepared.
    pub(super) view: Option<usize>,
    /// The Distortion phase runs for this group this frame: `PostProcess::distortion` is
    /// enabled and a Distortion slot has work in the settings camera's view (graphics spec §1.8).
    pub(super) distortion: bool,
    /// The city effects (fog, SSAO, SSR) that run for this group this frame (`post::screenfx`).
    pub(super) fx: crate::post::FxPlan,
}

/// Plan of one camera this frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct CameraPlan {
    /// Index in `FrameState::groups`, `None` for a camera that is skipped (its render target is
    /// not created).
    pub(super) group: Option<usize>,
    pub(super) role: ViewRole,
    /// The post group a scene view renders for.
    pub(super) post_group: Option<usize>,
}

/// Warnings logged once per renderer.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PlanWarnings {
    hdr_target_post: bool,
    camera_frame_ignored: bool,
    hdr_msaa: bool,
    screen_fx_mode: bool,
}

/// The display target of a render-target camera: `(size, format, samples)`, `None` while the
/// target is not created.
fn texture_display(assets: &RenderAssets, h: TextureHandle) -> Option<((u32, u32), wgpu::TextureFormat, u32, bool)> {
    let e = assets.textures.get(h.raw())?;
    let g = e.gpu.as_ref()?;
    g.target_view.as_ref()?;
    g.depth_view.as_ref()?;
    let hdr = e.target_desc.is_some_and(|d| d.format == RenderTargetFormat::Hdr) && !g.format.is_srgb();
    Some(((e.width, e.height), g.format, g.samples, hdr))
}

/// The full rect of a `size` target.
pub(super) fn full_rect(size: (u32, u32)) -> ViewportRect {
    ViewportRect {
        x: 0,
        y: 0,
        w: size.0,
        h: size.1,
    }
}

/// A camera's rect on a `size` target (the whole target without a viewport).
pub(super) fn camera_rect(camera: &Camera3d, size: (u32, u32)) -> ViewportRect {
    camera.viewport.map_or(full_rect(size), |v| viewport_rect(v, size))
}

/// Scene size of a `display` target at render scale `scale`.
pub(super) fn scaled_size(display: (u32, u32), scale: f32) -> (u32, u32) {
    if scale >= 1.0 {
        return display;
    }
    let s = |v: u32| ((v as f32 * scale).round() as u32).max(1);
    (s(display.0), s(display.1))
}

impl FrameState {
    /// Plan the frame (graphics spec §1.3): assign every camera to its target group, choose each
    /// group's mode, scene format, samples and render scale, build the post groups and give
    /// every camera its role. Reuses the plan vectors (no allocation in the steady state).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_frame(
        &mut self,
        assets: &RenderAssets,
        primary: PrimaryDisplay,
        settings: &FrameSettings,
        graphics: &GraphicsCaps,
        max_msaa: u32,
        warnings: &mut PlanWarnings,
    ) {
        let f = self;
        f.groups.clear();
        f.post_groups.clear();
        f.post_depths.clear();
        f.camera_plans.clear();
        // Groups in order of their first camera; every camera's group.
        for ci in 0..f.cameras.len() {
            let target = match f.cameras[ci].camera.target {
                RenderTarget::Primary => Some((
                    GroupTarget::Primary,
                    primary.size,
                    primary.format,
                    primary.samples,
                    false,
                )),
                RenderTarget::Texture(h) => texture_display(assets, h)
                    .map(|(size, format, samples, hdr)| (GroupTarget::Texture(h), size, format, samples, hdr)),
            };
            let Some((target, size, format, samples, hdr)) = target else {
                f.camera_plans.push(CameraPlan {
                    group: None,
                    role: ViewRole::Scene,
                    post_group: None,
                });
                continue;
            };
            let group = match f.groups.iter().position(|g| g.target == target) {
                Some(g) => g,
                None => {
                    f.groups.push(TargetGroup {
                        target,
                        mode: FrameMode::Direct,
                        lead: ci,
                        display_size: size,
                        display_format: format,
                        display_samples: samples,
                        manual_srgb: target == GroupTarget::Primary && primary.manual_srgb,
                        scene_format: format,
                        scene_size: size,
                        samples,
                        scale: 1.0,
                        transparent: false,
                        post_groups: 0..0,
                        display_views: false,
                        last_scene_view: None,
                        scene_views: 0,
                    });
                    f.groups.len() - 1
                }
            };
            // Post on cameras rendering into HDR render targets is ignored (graphics spec D16).
            if hdr && f.cameras[ci].post.is_some() {
                if !warnings.hdr_target_post {
                    warnings.hdr_target_post = true;
                    tracing::warn!(
                        "PostProcess on a camera rendering into an HDR render target is ignored: HDR \
                         targets keep linear radiance (post runs in the camera that samples them)"
                    );
                }
                f.cameras[ci].post = None;
            }
            if !settings.post {
                f.cameras[ci].post = None;
            }
            // The highest-order post camera leads its group.
            if f.cameras[ci].post.is_some() {
                f.groups[group].lead = ci;
            }
            f.camera_plans.push(CameraPlan {
                group: Some(group),
                role: ViewRole::Scene,
                post_group: None,
            });
        }

        for g in 0..f.groups.len() {
            let lead = f.cameras[f.groups[g].lead];
            let primary_group = f.groups[g].target == GroupTarget::Primary;
            let hdr_target = match f.groups[g].target {
                GroupTarget::Texture(h) => texture_display(assets, h).is_some_and(|t| t.3),
                GroupTarget::Primary => false,
            };
            // Differing `CameraFrame`s of other cameras of the group are ignored.
            if !warnings.camera_frame_ignored {
                let ignored = (0..f.cameras.len()).any(|c| {
                    f.camera_plans[c].group == Some(g)
                        && c != f.groups[g].lead
                        && f.cameras[c].frame.is_some_and(|fr| Some(fr) != lead.frame)
                });
                if ignored {
                    warnings.camera_frame_ignored = true;
                    tracing::warn!(
                        "a CameraFrame differs from its target group's lead camera (the highest-order \
                         post camera, else the first): only the lead's applies"
                    );
                }
            }
            let has_post =
                (0..f.cameras.len()).any(|c| f.camera_plans[c].group == Some(g) && f.cameras[c].post.is_some());
            let scale = match lead.frame.and_then(|fr| fr.render_scale) {
                Some(s) => s,
                None if primary_group => settings.render_scale,
                None => 1.0,
            };
            let scale = if hdr_target || !scale.is_finite() {
                1.0
            } else {
                scale.clamp(0.25, 1.0)
            };
            let hdr_mode = lead.frame.map_or(HdrMode::Auto, |fr| fr.hdr);
            let lead_alpha = lead.camera.clear_color.map_or(1.0, |c| c.w);
            let group = &mut f.groups[g];
            group.scale = scale;
            group.transparent = !primary_group && lead_alpha < 1.0;
            let mut samples = group.display_samples.min(max_msaa.max(1));
            let (mode, format) = if !has_post && scale >= 1.0 {
                (FrameMode::Direct, group.display_format)
            } else if !has_post {
                (FrameMode::Ldr, wgpu::TextureFormat::Rgba8UnormSrgb)
            } else if hdr_mode != HdrMode::Ldr
                && let Some(hdr) = graphics.hdr_format
            {
                if samples > 1 && !graphics.hdr_msaa {
                    if !warnings.hdr_msaa {
                        warnings.hdr_msaa = true;
                        tracing::warn!(
                            "the HDR scene format cannot be multisampled on this device: HDR groups \
                             render with 1 sample"
                        );
                    }
                    samples = 1;
                }
                // Compact HDR has no alpha: opaque-cleared primary targets only, and single-sample
                // (its multisampling is not probed).
                let compact = hdr_mode == HdrMode::Compact && primary_group && lead_alpha >= 1.0 && samples == 1;
                match graphics.compact_hdr_format {
                    Some(c) if compact => (FrameMode::Hdr, c),
                    _ => (FrameMode::Hdr, hdr),
                }
            } else {
                (FrameMode::Ldr, wgpu::TextureFormat::Rgba8UnormSrgb)
            };
            group.mode = mode;
            if mode == FrameMode::Direct {
                continue;
            }
            group.scene_format = format;
            group.scene_size = scaled_size(group.display_size, scale);
            group.samples = samples;
            // Post groups: one per distinct rect with a post camera, its settings camera the
            // rect's highest-order post camera; in settings-camera order.
            let start = f.post_groups.len();
            let (display_size, scene_size) = (group.display_size, group.scene_size);
            if !has_post {
                // Ldr passthrough: one implicit full-target group, every view a scene view.
                let lead_index = group.lead;
                f.post_groups.push(FramePostGroup {
                    group: g,
                    camera: lead_index,
                    settings: PostProcess::passthrough(),
                    passthrough: true,
                    display_rect: full_rect(display_size),
                    scene_rect: full_rect(scene_size),
                    scene_output: OutputEncoding::Linear,
                    exposure: 1.0,
                    tonemapper: Tonemapper::None,
                    view: None,
                    distortion: false,
                    fx: crate::post::FxPlan::default(),
                });
            } else {
                for c in 0..f.cameras.len() {
                    if f.camera_plans[c].group != Some(g) {
                        continue;
                    }
                    let Some(post) = f.cameras[c].post else { continue };
                    let display_rect = camera_rect(&f.cameras[c].camera, display_size);
                    let (scene_output, exposure, tonemapper) = if mode == FrameMode::Hdr {
                        (OutputEncoding::Linear, 1.0, post.tonemapper)
                    } else {
                        (OutputEncoding::Tonemapped, post.exposure.exp2(), post.tonemapper)
                    };
                    let mut pg = FramePostGroup {
                        group: g,
                        camera: c,
                        settings: post,
                        passthrough: false,
                        display_rect,
                        scene_rect: camera_rect(&f.cameras[c].camera, scene_size),
                        scene_output,
                        exposure,
                        tonemapper,
                        view: None,
                        distortion: false,
                        fx: crate::post::FxPlan::default(),
                    };
                    if post.enabled && post.screen_effects() {
                        // The settings camera's depth must exist at one sample: a multisampled
                        // depth is resolved (not on GL, where it cannot be sampled).
                        let depth = samples <= 1 || graphics.msaa_depth_resolve;
                        pg.fx = crate::post::screen_fx_plan(&post, mode, depth, graphics.read_only_depth);
                        if mode != FrameMode::Hdr && !warnings.screen_fx_mode {
                            warnings.screen_fx_mode = true;
                            tracing::warn!(
                                "fog, SSAO and SSR are off in the Ldr fallback (no float render targets or                                  HdrMode::Ldr)"
                            );
                        }
                    }
                    match f.post_groups[start..]
                        .iter_mut()
                        .find(|p| p.display_rect == display_rect)
                    {
                        Some(existing) => *existing = pg,
                        None => f.post_groups.push(pg),
                    }
                }
                f.post_groups[start..].sort_unstable_by_key(|p| p.camera);
            }
            let end = f.post_groups.len();
            f.groups[g].post_groups = start..end;
            // Roles: a view of a post group's rect up to its settings camera renders into `S`
            // for it; every other view of the group is a display view.
            for c in 0..f.cameras.len() {
                if f.camera_plans[c].group != Some(g) {
                    continue;
                }
                let rect = camera_rect(&f.cameras[c].camera, display_size);
                let pg = (start..end).find(|&p| {
                    let p = &f.post_groups[p];
                    p.passthrough || (p.display_rect == rect && c <= p.camera)
                });
                let plan = &mut f.camera_plans[c];
                plan.post_group = pg;
                plan.role = if pg.is_some() {
                    ViewRole::Scene
                } else {
                    ViewRole::Display
                };
                if pg.is_none() {
                    f.groups[g].display_views = true;
                }
            }
        }
    }
}

/// A view's viewport in whole pixels of its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ViewportRect {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) w: u32,
    pub(super) h: u32,
}

impl ViewportRect {
    /// The rect covers every pixel of a `size` target.
    pub(super) fn covers(&self, size: (u32, u32)) -> bool {
        self.x <= 0
            && self.y <= 0
            && self.x as i64 + self.w as i64 >= size.0 as i64
            && self.y as i64 + self.h as i64 >= size.1 as i64
    }

    /// The rect to draw with, and a clip-space correction to apply after the projection.
    ///
    /// wgpu rejects viewports wider or higher than `max_texture_dimension_2d`, or reaching
    /// further than twice that from the origin: a validation error that loses the frame's
    /// commands. Camera viewports can be anything (`[-50.0, -50.0, 101.0, 101.0]` zooms into
    /// the center of a 101× larger image; non-finite values end up far outside). A rect within
    /// the limits is returned as is, with no correction — every in-range viewport draws exactly
    /// as before. Any other rect is drawn through its part within the target, with the
    /// projection remapped so that every pixel there shows exactly what the whole rect would
    /// (only the float rounding differs); a rect entirely outside the target draws nothing
    /// (a 1×1 rect just past the target's corner).
    pub(super) fn fitted(self, size: (u32, u32), max_dimension: u32) -> (ViewportRect, Mat4) {
        let max = max_dimension as i64;
        let (x, y, w, h) = (self.x as i64, self.y as i64, self.w as i64, self.h as i64);
        let valid = w <= max && h <= max && x >= -2 * max && y >= -2 * max && x + w < 2 * max && y + h < 2 * max;
        if valid {
            return (self, Mat4::IDENTITY);
        }
        let (tw, th) = (size.0.max(1) as i64, size.1.max(1) as i64);
        let (x0, y0) = (x.clamp(0, tw), y.clamp(0, th));
        let (x1, y1) = ((x + w).clamp(0, tw), (y + h).clamp(0, th));
        if x1 <= x0 || y1 <= y0 {
            let outside = ViewportRect {
                x: tw as i32,
                y: th as i32,
                w: 1,
                h: 1,
            };
            return (outside, Mat4::IDENTITY);
        }
        let clip = ViewportRect {
            x: x0 as i32,
            y: y0 as i32,
            w: (x1 - x0) as u32,
            h: (y1 - y0) as u32,
        };
        // NDC of the full rect → NDC of `clip` for the same pixel (y points up in NDC, down in
        // pixels).
        let (cw, ch) = ((x1 - x0) as f64, (y1 - y0) as f64);
        let (sx, sy) = (w as f64 / cw, h as f64 / ch);
        let tx = (2.0 * (x - x0) as f64 + w as f64 - cw) / cw;
        let ty = (ch - h as f64 - 2.0 * (y - y0) as f64) / ch;
        let fit = Mat4::from_cols(
            Vec4::new(sx as f32, 0.0, 0.0, 0.0),
            Vec4::new(0.0, sy as f32, 0.0, 0.0),
            Vec4::Z,
            Vec4::new(tx as f32, ty as f32, 0.0, 1.0),
        );
        (clip, fit)
    }

    /// `[x, y, width, height]` of the rect's part within a `size` target (at least one pixel,
    /// as a scissor rect must lie within the target): the scissor rect of the view's pass.
    pub(super) fn scissor(&self, size: (u32, u32)) -> [u32; 4] {
        let axis = |start: i32, len: u32, extent: u32| {
            let extent = extent.max(1) as i64;
            let a = (start as i64).clamp(0, extent - 1);
            let b = (start as i64 + len as i64).clamp(a + 1, extent);
            (a as u32, (b - a) as u32)
        };
        let (x, w) = axis(self.x, self.w, size.0);
        let (y, h) = axis(self.y, self.h, size.1);
        [x, y, w, h]
    }
}

/// Pixel rect of a normalized viewport `[x, y, width, height]` (top left origin) on a `size`
/// target: its edges rounded to the nearest pixel boundary, at least one pixel wide and high.
/// Rounded because GL takes integer viewports (wgpu's GL backend truncates position and size
/// separately, which would shift fractional viewports and open gaps between neighbours): every
/// backend draws the same pixels, and neighbouring viewports share their edges exactly.
///
/// A viewport whose edges round to the same boundary (thinner than about a pixel) gets the
/// pixel holding its center, kept within the target when the viewport overlaps the target:
/// growing its far edge instead would push a viewport in the last half pixel of the target
/// entirely outside it.
pub(super) fn viewport_rect([x, y, w, h]: [f32; 4], size: (u32, u32)) -> ViewportRect {
    let axis = |start: f32, len: f32, extent: u32| {
        let e = extent as f32;
        let a = (start * e).round();
        let b = ((start + len) * e).round();
        if b > a {
            return (a as i32, (b - a) as u32);
        }
        let center = ((start + 0.5 * len) * e).floor();
        let overlaps = start.min(start + len) < 1.0 && start.max(start + len) > 0.0;
        let p = if overlaps {
            center.clamp(0.0, (e - 1.0).max(0.0))
        } else {
            center
        };
        (p as i32, 1)
    };
    let (x, w) = axis(x, w, size.0);
    let (y, h) = axis(y, h, size.1);
    ViewportRect { x, y, w, h }
}
