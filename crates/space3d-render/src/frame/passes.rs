//! Render passes (graphics spec §1.4 step 12): every view's forward pass into its target
//! (clears, viewports, MSAA resolve) — `D` in Direct mode, the group's scene texture `S` for
//! scene views of intermediate groups — with the render nodes' phases inside it; after it the
//! depth resolve and the SceneRead pass of depth-reading node slots (§1.6), and the Distortion
//! pass of a post group's settings camera (§1.8); HdrScene overlays after the primary group's
//! last scene view (§1.5); the finish of intermediate groups (post groups' composites, then
//! display views into `D`), the clear of a window nothing drew to, and the windowed-capture
//! replay.

use super::nodes::{
    NodePass, Orders, ReadMode, SlotList, distortion_target, group_target, read_mode, read_target, scene_target, work,
};
use super::views::{ViewData, ViewTarget};
use crate::node::{FrameInfo, FrameKind, FrameMode, PassTarget, SceneDepthRef, SceneReads, ViewRole};
use crate::output::{OutputEncoding, Tonemapper};
use crate::overlay::{OverlayScene, OverlayStage, OverlayTarget, OverlayTargetKind, lock_overlay};
use crate::pipeline::OFFSCREEN_FORMAT;
use crate::post::{PostEncode, PostGroupInput};
use crate::renderer::Renderer;
use glam::Vec4;

pub(super) fn linear_to_srgb_color(c: Vec4) -> Vec4 {
    use crate::components::linear_to_srgb;
    Vec4::new(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z), c.w)
}

/// Encode state of one target group while recording a frame's passes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GroupEncode {
    /// Some pass drew into `D` this frame: later passes load it.
    drawn: bool,
    /// Some scene pass drew into `S` this frame: later ones load it.
    scene_started: bool,
    /// The group's post groups were composited into `D`.
    finished: bool,
    /// Scene views record into `S` in this encode: always in a frame; in a capture replay only
    /// when `S` does not hold the frame's scene views (the window's surface texture was
    /// unavailable, so the frame skipped them). Otherwise the replay composites the persisted
    /// `S` again.
    render_scene: bool,
}

/// The display target `D` of a view's group, as attachments.
struct DisplayTarget {
    /// Color attachment of Direct passes (multisampled when `samples > 1`).
    color: wgpu::TextureView,
    resolve: Option<wgpu::TextureView>,
    /// The single-sample texture of `D`: composites and display views draw into it.
    single: wgpu::TextureView,
    /// Depth of Direct passes (`samples` samples).
    depth: wgpu::TextureView,
    format: wgpu::TextureFormat,
    samples: u32,
    size: (u32, u32),
    manual_srgb: bool,
}

/// The attachments of one view's scene pass.
struct ViewPass<'a> {
    /// Color attachment (multisampled when `samples > 1`).
    color: &'a wgpu::TextureView,
    resolve: Option<&'a wgpu::TextureView>,
    /// The single-sample color (`resolve`, else `color`): SceneRead passes load it.
    single: &'a wgpu::TextureView,
    depth: &'a wgpu::TextureView,
    format: wgpu::TextureFormat,
    samples: u32,
    /// The encoding fragments apply in this pass.
    output: OutputEncoding,
}

/// The encoding a view's pipelines use in a pass: its own, except that only manual-sRGB
/// attachments get manual sRGB (a capture replay of a manual-sRGB window draws into the sRGB
/// offscreen target).
pub(super) fn pass_output(view: OutputEncoding, manual_srgb: bool) -> OutputEncoding {
    match view {
        OutputEncoding::ManualSrgb | OutputEncoding::Linear if manual_srgb => OutputEncoding::ManualSrgb,
        OutputEncoding::ManualSrgb => OutputEncoding::Linear,
        other => other,
    }
}

/// Viewport and scissor of a view's passes: its rect (`None`: the whole `extent`).
fn view_rects(view: &ViewData, extent: (u32, u32)) -> ([f32; 4], [u32; 4]) {
    match view.rect {
        Some(r) => ([r.x as f32, r.y as f32, r.w as f32, r.h as f32], r.scissor(extent)),
        None => (
            [0.0, 0.0, extent.0 as f32, extent.1 as f32],
            [0, 0, extent.0.max(1), extent.1.max(1)],
        ),
    }
}

fn extent(view: &wgpu::TextureView) -> (u32, u32) {
    (view.texture().width(), view.texture().height())
}

/// Where HdrScene overlays draw and what they are told.
struct HdrScenePass<'a> {
    target: &'a wgpu::TextureView,
    format: wgpu::TextureFormat,
    /// Scale factor on top of the overlays' own (the group's render scale).
    scale: f32,
    output: OutputEncoding,
    mode: FrameMode,
    depth: Option<wgpu::TextureView>,
    kind: OverlayTargetKind,
    /// Group 0 of the view the overlays follow.
    bind_group: usize,
}

impl Renderer {
    /// Record render passes for the prepared views. With `primary_to_offscreen`, only primary
    /// views are re-recorded into the offscreen target (windowed capture).
    ///
    /// Views sharing a target (split screen, picture-in-picture, overlays) draw in camera
    /// order into the same attachments: the first pass into a target in a frame clears it,
    /// later ones load what the earlier ones drew (see [`Self::encode_view`]). In intermediate
    /// groups, scene views draw into `S` the same way; after the group's last scene view its
    /// post groups are composited into `D` (the first composite clears `D`), then its display
    /// views draw into `D` in camera order.
    pub(crate) fn encode_views(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        surface_view: Option<&wgpu::TextureView>,
        primary_to_offscreen: bool,
    ) {
        let views = std::mem::take(&mut self.frame.views);
        let mut state = std::mem::take(&mut self.frame.group_encode);
        state.clear();
        state.resize(self.frame.groups.len(), GroupEncode::default());
        let frame_index = self.frame_index;
        for (s, group) in state.iter_mut().zip(&self.frame.groups) {
            s.render_scene = !primary_to_offscreen
                || self
                    .frame
                    .group_textures
                    .get(&group.target)
                    .is_some_and(|t| t.scene_frame != frame_index);
        }
        let mut presented = false;
        let kind = match (primary_to_offscreen, surface_view.is_some()) {
            (true, _) => OverlayTargetKind::Capture,
            (false, true) => OverlayTargetKind::Surface,
            (false, false) => OverlayTargetKind::Offscreen,
        };
        for (i, view) in views.iter().enumerate() {
            let is_primary = matches!(view.target, ViewTarget::Primary);
            if primary_to_offscreen && !is_primary {
                continue;
            }
            let Some(d) = self.display_target(view, surface_view, primary_to_offscreen) else {
                // Surface frame unavailable (minimized/occluded): skip.
                continue;
            };
            let g = view.group;
            let (intermediate, last_scene_view) = {
                let group = &self.frame.groups[g];
                (group.intermediate(), group.last_scene_view)
            };
            if !intermediate {
                let first = !state[g].drawn;
                state[g].drawn = true;
                presented |= is_primary;
                self.copy_feedback(encoder, view);
                let output = pass_output(view.output, d.manual_srgb);
                let pass = ViewPass {
                    color: &d.color,
                    resolve: d.resolve.as_ref(),
                    single: &d.single,
                    depth: &d.depth,
                    format: d.format,
                    samples: d.samples,
                    output,
                };
                let depth = self.encode_view(encoder, i, view, first, &pass, primary_to_offscreen);
                // HdrScene overlays of a Direct primary group: into `D` after its last view.
                if is_primary && self.frame.hdr_scene_view == Some(i) {
                    let hdr = HdrScenePass {
                        target: &d.single,
                        format: d.format,
                        scale: 1.0,
                        output,
                        mode: FrameMode::Direct,
                        depth,
                        kind,
                        bind_group: view.bind_group,
                    };
                    self.encode_hdr_scene_overlays(encoder, i, &hdr);
                }
                continue;
            }
            match view.role {
                ViewRole::Scene => {
                    // Replays keep `S` from the frame (only its composites run again), unless
                    // the frame skipped the scene views.
                    let mut depth = None;
                    if state[g].render_scene {
                        let first = !state[g].scene_started;
                        state[g].scene_started = true;
                        depth = self.encode_scene_view(encoder, i, view, first, primary_to_offscreen);
                    }
                    if last_scene_view == Some(i) {
                        self.finish_group(
                            encoder,
                            g,
                            i,
                            &views[..i],
                            &d,
                            primary_to_offscreen,
                            &mut state[g],
                            depth,
                            view.bind_group,
                        );
                        presented |= is_primary;
                    }
                }
                _ => {
                    // Display views before the group's last scene view wait for its finish.
                    if state[g].finished {
                        self.encode_display_view(encoder, i, view, &d, primary_to_offscreen);
                    }
                }
            }
        }
        // Nothing drew to the window: still clear it so stale content is not shown.
        if !presented
            && !primary_to_offscreen
            && let Some(sv) = surface_view
        {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("space3d clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: sv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        self.frame.views = views;
        self.frame.group_encode = state;
    }

    /// The display target of `view`'s group: the surface (windowed frames), the offscreen
    /// target (headless frames, capture replays) or the render target. `None` when the
    /// surface frame is unavailable.
    fn display_target(
        &self,
        view: &ViewData,
        surface_view: Option<&wgpu::TextureView>,
        to_offscreen: bool,
    ) -> Option<DisplayTarget> {
        match (&view.target, surface_view) {
            (ViewTarget::Primary, Some(sv)) if !to_offscreen => {
                let s = self.surface.as_ref().expect("surface");
                let a = &s.attachments;
                Some(DisplayTarget {
                    color: a.msaa_view.clone().unwrap_or_else(|| sv.clone()),
                    resolve: a.msaa_view.as_ref().map(|_| sv.clone()),
                    single: sv.clone(),
                    depth: a.depth_view.clone(),
                    format: a.format,
                    samples: a.samples,
                    size: (s.config.width, s.config.height),
                    manual_srgb: s.manual_srgb,
                })
            }
            (ViewTarget::Primary, _) => {
                if !to_offscreen && self.presents_to_window() {
                    return None;
                }
                let a = &self.offscreen;
                let target = a.color_view.clone().expect("offscreen color");
                Some(DisplayTarget {
                    color: a.msaa_view.clone().unwrap_or_else(|| target.clone()),
                    resolve: a.msaa_view.as_ref().map(|_| target.clone()),
                    single: target,
                    depth: a.depth_view.clone(),
                    format: OFFSCREEN_FORMAT,
                    samples: a.samples,
                    size: (a.width, a.height),
                    manual_srgb: false,
                })
            }
            (
                ViewTarget::Texture {
                    color,
                    depth,
                    msaa,
                    format,
                    samples,
                    ..
                },
                _,
            ) => Some(DisplayTarget {
                color: msaa.clone().unwrap_or_else(|| color.clone()),
                resolve: msaa.as_ref().map(|_| color.clone()),
                single: color.clone(),
                depth: depth.clone(),
                format: *format,
                samples: *samples,
                size: (color.texture().width(), color.texture().height()),
                manual_srgb: false,
            }),
        }
    }

    /// Snapshot a render target before a pass draws into it, for the view's materials that
    /// sample it (see `ViewData::feedback`).
    fn copy_feedback(&mut self, encoder: &mut wgpu::CommandEncoder, view: &ViewData) {
        if let ViewTarget::Texture { handle, color, .. } = &view.target
            && view.feedback
            && let Some(fb) = self.frame.feedback.get(handle)
        {
            encoder.copy_texture_to_texture(
                color.texture().as_image_copy(),
                fb.texture.as_image_copy(),
                wgpu::Extent3d {
                    width: fb.size.0,
                    height: fb.size.1,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// A scene view of an intermediate group: its forward pass into the group's `S` (MSAA
    /// resolved into it), with the viewport-clear rules applied to `S`, its SceneRead pass and,
    /// for a post group's settings camera, the Distortion pass. Returns the single-sample depth
    /// holding the view's depth afterwards, if a consumer needed one. `replay`: a capture
    /// replay re-rendering `S` (node draws get `FrameKind::CaptureReplay`).
    fn encode_scene_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        i: usize,
        view: &ViewData,
        first: bool,
        replay: bool,
    ) -> Option<wgpu::TextureView> {
        let group = &self.frame.groups[view.group];
        let (format, samples) = (group.scene_format, group.samples);
        let t = self.frame.group_textures.get(&group.target)?;
        let (color, resolve) = match &t.scene_msaa {
            Some(m) => (m.clone(), Some(t.scene_view.clone())),
            None => (t.scene_view.clone(), None),
        };
        let (single, depth) = (t.scene_view.clone(), t.depth.clone());
        let pass = ViewPass {
            color: &color,
            resolve: resolve.as_ref(),
            single: &single,
            depth: &depth,
            format,
            samples,
            output: view.output,
        };
        let depth_1x = self.encode_view(encoder, i, view, first, &pass, replay);
        if view.work & work::SCREEN_FX != 0
            && let Some(pg) = self
                .frame
                .post_groups
                .iter()
                .position(|p| p.view == Some(i) && p.fx.any())
        {
            let n = self.frame.post_groups.len();
            self.frame.post_depths.resize(n, None);
            let offset = self.frame.view_infos[i].view_offset;
            self.frame.post_depths[pg] = depth_1x.clone().map(|d| (d, view.bind_group, offset));
            if self.frame.post_groups[pg].fx.ssr
                && let Some(depth) = &depth_1x
            {
                self.encode_gbuffer(encoder, view, depth);
            }
        }
        if view.work & work::DISTORTION != 0
            && let Some(pg) = self
                .frame
                .post_groups
                .iter()
                .position(|p| p.view == Some(i) && p.distortion)
        {
            self.encode_distortion(encoder, i, view.bind_group, pg, depth_1x.as_ref(), replay);
        }
        depth_1x
    }

    /// The SSR G-buffer-lite pass of a scene view (`PassKind::GBuffer`): its opaque draws again,
    /// depth-tested read-only against the view's single-sample scene depth `depth`, writing
    /// normal, roughness and F0 into the camera's G-buffer.
    fn encode_gbuffer(&mut self, encoder: &mut wgpu::CommandEncoder, view: &ViewData, depth: &wgpu::TextureView) {
        let camera = self.frame.cameras[view.camera].id;
        let extent = extent(depth);
        let target = {
            let Renderer {
                device,
                queue,
                layouts,
                screen_fx,
                frame_index,
                stats,
                ..
            } = self;
            let fx = screen_fx.get_or_insert_with(|| crate::post::ScreenFx::new(device, queue, layouts, *frame_index));
            stats.target_bytes += 4 * extent.0 as u64 * extent.1 as u64;
            fx.gbuffer_target(device, camera, extent)
        };
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("space3d ssr g-buffer"),
                timestamp_writes: self.timer.render("gbuffer", Some(camera)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // Read-only: tested by hardware, never written.
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                ..Default::default()
            })
            .forget_lifetime();
        if let Some(r) = view.rect {
            pass.set_viewport(r.x as f32, r.y as f32, r.w as f32, r.h as f32, 0.0, 1.0);
            let [x, y, w, h] = r.scissor(extent);
            pass.set_scissor_rect(x, y, w, h);
        }
        self.encode_draws_as(
            &mut pass,
            view,
            view.draws.start..view.opaque_end,
            crate::post::GBUFFER_FORMAT,
            1,
            OutputEncoding::Linear,
            crate::pipeline::PassKind::GBuffer,
        );
        drop(pass);
        self.stats.render_passes += 1;
    }

    /// A display view of an intermediate group: drawn LDR, Direct-style, at 1 sample into `D`
    /// after the group's composites (with the group's display depth).
    fn encode_display_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        i: usize,
        view: &ViewData,
        d: &DisplayTarget,
        replay: bool,
    ) {
        let group = &self.frame.groups[view.group];
        let Some(depth) = self
            .frame
            .group_textures
            .get(&group.target)
            .and_then(|t| t.display_depth.clone())
        else {
            return;
        };
        self.copy_feedback(encoder, view);
        let pass = ViewPass {
            color: &d.single,
            resolve: None,
            single: &d.single,
            depth: &depth,
            format: d.format,
            samples: 1,
            output: pass_output(view.output, d.manual_srgb),
        };
        self.encode_view(encoder, i, view, false, &pass, replay);
    }

    /// Finish an intermediate group (spec §1.4 `finish(G)`): the primary group's HdrScene
    /// overlays into `S`, then each post group's composite, in settings-camera order, into `D`
    /// (the first write clears `D`), then the display views that came before the group's last
    /// scene view (`earlier`), in camera order. A capture replay composites again from the
    /// persisted `S` (no temporal update; HdrScene overlays are already in `S`) — or, when it
    /// re-rendered `S` (`state.render_scene`), calls the HdrScene overlays again (kind
    /// `Capture`) first.
    #[allow(clippy::too_many_arguments)]
    fn finish_group(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        g: usize,
        last: usize,
        earlier: &[ViewData],
        d: &DisplayTarget,
        replay: bool,
        state: &mut GroupEncode,
        depth: Option<wgpu::TextureView>,
        bind_group: usize,
    ) {
        let group = &self.frame.groups[g];
        let (target, mode, scene_format, scene_size, transparent, post_groups, scale) = (
            group.target,
            group.mode,
            group.scene_format,
            group.scene_size,
            group.transparent,
            group.post_groups.clone(),
            group.scale,
        );
        let frame_index = self.frame_index;
        let Some(t) = self.frame.group_textures.get_mut(&target) else {
            return;
        };
        if state.render_scene {
            // `S` now holds this frame's scene views.
            t.scene_frame = frame_index;
        }
        let (scene, generation) = (t.scene_view.clone(), t.generation);
        if state.render_scene && self.frame.hdr_scene_view == Some(last) {
            let hdr = HdrScenePass {
                target: &scene,
                format: scene_format,
                scale,
                output: self.frame.view_infos[last].output,
                mode,
                depth,
                kind: if replay {
                    OverlayTargetKind::Capture
                } else if self.surface.is_some() {
                    OverlayTargetKind::Surface
                } else {
                    OverlayTargetKind::Offscreen
                },
                bind_group,
            };
            self.encode_hdr_scene_overlays(encoder, last, &hdr);
        }
        let mut frame = self.frame.info;
        if replay {
            frame.kind = FrameKind::CaptureReplay;
            frame.delta = 0.0;
        }
        for p in post_groups {
            let pg = self.frame.post_groups[p];
            let (rect, _) = pg.display_rect.fitted(d.size, self.frame.max_texture_2d);
            let camera = self.frame.cameras[pg.camera].id;
            // The distortion buffer written by this frame's Distortion pass (it persists for
            // replays).
            let distortion = pg
                .distortion
                .then(|| self.post.distortion_view(camera))
                .flatten()
                .map(|(view, _, _)| view);
            // The city effects (fog, SSAO, SSR): S to S2, which the post chain then reads.
            let scene_rect = pg.scene_rect.scissor(scene_size);
            let fx_scene = self.encode_screen_fx(encoder, p, camera, (&scene, scene_format, scene_size), scene_rect);
            let (scene, generation) = match &fx_scene {
                Some((view, g)) => (view, *g),
                None => (&scene, generation),
            };
            let input = PostGroupInput {
                camera,
                settings: pg.settings,
                mode,
                scene_output: pg.scene_output,
                scene,
                scene_format,
                scene_size,
                scene_rect,
                scene_alpha: scene_format != wgpu::TextureFormat::Rg11b10Ufloat,
                transparent,
                depth: None,
                distortion: distortion.as_ref(),
                output: &d.single,
                output_format: d.format,
                output_size: d.size,
                output_rect: rect.scissor(d.size),
                output_manual_srgb: d.manual_srgb,
                output_clear: !state.drawn,
                passthrough: pg.passthrough,
                replay,
                frame,
                index: p,
                scene_generation: generation,
                output_viewport: [rect.x as f32, rect.y as f32, rect.w as f32, rect.h as f32],
            };
            let mut cx = PostEncode {
                device: &self.device,
                queue: &self.queue,
                encoder,
                stats: &mut self.stats,
                pool: &mut self.frame.pool,
                timing: &self.timer,
            };
            self.post.encode_group(&mut cx, &input);
            self.frame.pool.release_all();
            state.drawn = true;
        }
        state.finished = true;
        for (j, view) in earlier.iter().enumerate() {
            if view.group == g && view.role == ViewRole::Display {
                self.encode_display_view(encoder, j, view, d, replay);
            }
        }
    }

    /// The city effects of post group `p` (settings camera `camera`) over its scene color `S`:
    /// the pooled `S2` and its generation, or `None` when the group runs none (or lacks the
    /// settings camera's depth, e.g. in a capture replay that did not re-render the scene).
    fn encode_screen_fx(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        p: usize,
        camera: space3d_core::Entity,
        (scene, scene_format, scene_size): (&wgpu::TextureView, wgpu::TextureFormat, (u32, u32)),
        rect: [u32; 4],
    ) -> Option<(wgpu::TextureView, u64)> {
        let pg = self.frame.post_groups[p];
        if !pg.fx.any() {
            return None;
        }
        let (depth, bind_group, view_offset) = self.frame.post_depths.get(p)?.clone()?;
        let view_bind_group = self.frame.view_bind_group(bind_group)?.clone();
        let Renderer {
            device,
            queue,
            layouts,
            screen_fx,
            scene_inputs,
            frame: f,
            stats,
            timer,
            frame_index,
            ..
        } = self;
        let fx = screen_fx.get_or_insert_with(|| crate::post::ScreenFx::new(device, queue, layouts, *frame_index));
        let depth_bind_group = scene_inputs.with_depth(device, layouts, &depth).0.clone();
        let mut plan = pg.fx;
        plan.ssr &= fx.gbuffer(camera).is_some();
        let mut cx = crate::post::FxEncode {
            device,
            queue,
            encoder,
            stats,
            pool: &mut f.pool,
            timing: timer,
        };
        fx.encode(
            &mut cx,
            &crate::post::FxInput {
                camera,
                settings: pg.settings,
                plan,
                scene,
                scene_format,
                scene_size,
                rect,
                view_bind_group: &view_bind_group,
                view_offset,
                depth_bind_group: &depth_bind_group,
                frame: *frame_index,
            },
        )
    }

    /// Record one view's passes. `first`: no earlier pass of this frame drew into the view's
    /// target.
    ///
    /// The scene pass: a view covering its whole target clears it with its clear color (a load
    /// op). The first pass into a target otherwise clears it to black — never leaving the
    /// previous frame's content, and never painting a viewport's clear color outside the
    /// viewport — and later passes load it. A view with a viewport and a clear color then clears
    /// only its viewport, with a triangle in that color ([`Pipelines::viewport_clear`]). The
    /// viewport, in whole pixels ([`viewport_rect`]), also sets the pass's scissor rect (its
    /// part within the attachment). Depth is per view: every pass clears the whole depth
    /// attachment (views never test against each other) and discards it, unless a reader of
    /// the view's depth has work (graphics spec §1.6: its SceneRead pass, its post group's
    /// Distortion writers, HdrScene overlays reading depth).
    ///
    /// Inside the pass, per phase (graphics spec §1.4, §1.7): Opaque — node slots of negative
    /// order, the built-in opaque batches, the other Opaque slots; Skybox slots; Transparent —
    /// slots of negative order, the built-in blended batches, the other slots not reading
    /// depth (and, without a single-sample depth to read, the depth-reading ones). After it:
    /// the depth resolve (multisampled groups) and the SceneRead pass of the depth-reading
    /// slots. Returns the single-sample depth holding the view's depth afterwards, when a
    /// reader needed one.
    ///
    /// [`Pipelines::viewport_clear`]: crate::pipeline::Pipelines::viewport_clear
    /// [`viewport_rect`]: super::plan::viewport_rect
    fn encode_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        i: usize,
        view: &ViewData,
        first: bool,
        p: &ViewPass<'_>,
        replay: bool,
    ) -> Option<wgpu::TextureView> {
        let rect = view.rect;
        let full = rect.is_none_or(|r| r.covers(view.size));
        // Scissor rects must lie within the attachment itself. It has the size the view was
        // prepared for on every path (see `surface_lost`); should that ever not hold, a scissor
        // rect outside it would fail validation and lose the whole frame's commands.
        let extent = extent(p.color);
        let (viewport, scissor) = view_rects(view, extent);
        // Scene-depth readers of this view (graphics spec §1.6).
        let w = view.work;
        let single = self.frame.groups[view.group].scene_views <= 1 || view.role == ViewRole::Display;
        let resolved = (p.samples > 1)
            .then(|| self.frame.resolved_depth(group_target(&view.target), extent).cloned())
            .flatten();
        let (rods, can_resolve) = (
            self.frame_graphics.read_only_depth,
            self.frame_graphics.msaa_depth_resolve && resolved.is_some(),
        );
        let mode = read_mode(p.samples, rods, can_resolve, single);
        let reading = w & work::READING != 0;
        let scene_read = reading && mode != ReadMode::InPass;
        let overlay_depth = self.overlays.hdr_scene_depth() && self.frame.hdr_scene_view == Some(i);
        let wants_1x = scene_read || w & (work::DISTORTION | work::SCREEN_FX) != 0 || overlay_depth;
        let resolve = p.samples > 1 && can_resolve && wants_1x;
        let store = scene_read || resolve || (p.samples <= 1 && wants_1x);

        let clear = view.clear.map(|c| {
            let c = if p.output == OutputEncoding::ManualSrgb {
                linear_to_srgb_color(c)
            } else {
                c
            };
            wgpu::Color {
                r: c.x as f64,
                g: c.y as f64,
                b: c.z as f64,
                a: c.w as f64,
            }
        });
        let (load, rect_clear) = match clear {
            Some(c) if full => (wgpu::LoadOp::Clear(c), None),
            _ if first => (wgpu::LoadOp::Clear(wgpu::Color::BLACK), clear),
            _ => (wgpu::LoadOp::Load, clear),
        };
        let camera = Some(self.frame.cameras[view.camera].id);
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("space3d forward"),
                timestamp_writes: self.timer.render("scene", camera),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: p.color,
                    depth_slice: None,
                    resolve_target: p.resolve,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: p.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: if store {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            })
            .forget_lifetime();
        if let Some(r) = rect {
            pass.set_viewport(r.x as f32, r.y as f32, r.w as f32, r.h as f32, 0.0, 1.0);
            let [x, y, w, h] = r.scissor(extent);
            pass.set_scissor_rect(x, y, w, h);
        }
        if let Some(c) = rect_clear {
            pass.set_pipeline(self.pipelines.viewport_clear(&self.device, p.format, p.samples));
            pass.set_blend_constant(c);
            pass.draw(0..3, 0..1);
        }
        let target = scene_target(p.format, p.samples, p.output);
        let nodes = |r: &mut Self, pass: &mut wgpu::RenderPass<'static>, list: SlotList, orders: Orders| {
            r.draw_nodes(
                pass,
                (i, view.bind_group),
                list,
                orders,
                target,
                extent,
                viewport,
                scissor,
                None,
                replay,
            );
        };
        let restore = |pass: &mut wgpu::RenderPass<'static>| {
            let [x, y, w, h] = viewport;
            pass.set_viewport(x, y, w, h, 0.0, 1.0);
            let [x, y, w, h] = scissor;
            pass.set_scissor_rect(x, y, w, h);
        };
        if w & work::OPAQUE_PRE != 0 {
            nodes(self, &mut pass, SlotList::Opaque, Orders::BeforeBuiltins);
            restore(&mut pass);
        }
        if w & work::BETWEEN == 0 {
            self.encode_draws(&mut pass, view, view.draws.clone(), p.format, p.samples, p.output);
        } else {
            self.encode_draws(
                &mut pass,
                view,
                view.draws.start..view.opaque_end,
                p.format,
                p.samples,
                p.output,
            );
            if w & work::OPAQUE_POST != 0 {
                nodes(self, &mut pass, SlotList::Opaque, Orders::AfterBuiltins);
            }
            if w & work::SKYBOX != 0 {
                nodes(self, &mut pass, SlotList::Skybox, Orders::All);
            }
            if w & work::TRANSPARENT_PRE != 0 {
                nodes(self, &mut pass, SlotList::Transparent, Orders::BeforeBuiltins);
            }
            restore(&mut pass);
            self.encode_draws(
                &mut pass,
                view,
                view.opaque_end..view.draws.end,
                p.format,
                p.samples,
                p.output,
            );
        }
        if w & work::TRANSPARENT_POST != 0 {
            nodes(self, &mut pass, SlotList::Transparent, Orders::AfterBuiltins);
        }
        if reading && mode == ReadMode::InPass {
            nodes(self, &mut pass, SlotList::Reading, Orders::All);
        }
        if p.resolve.is_some() && (rect.is_some() || w != 0) {
            // wgpu's GL backend resolves MSAA with a framebuffer blit at the end of the pass,
            // and GL clips blits to the scissor rect: only the viewport (or a node's scissor)
            // would be resolved, and the rest of the target would keep its previous content.
            pass.set_scissor_rect(0, 0, extent.0, extent.1);
        }
        drop(pass);

        // The single-sample depth holding this view's depth.
        let depth_1x = if p.samples <= 1 {
            store.then(|| p.depth.clone())
        } else if resolve {
            let resolved = resolved.expect("can_resolve");
            self.encode_depth_resolve(encoder, p.depth, &resolved, p.samples, scissor);
            Some(resolved)
        } else {
            None
        };
        if scene_read && let Some(depth) = &depth_1x {
            let attached = mode != ReadMode::Sampled;
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("space3d scene read"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: p.single,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    // Read-only (no depth ops): tested by hardware and sampled in group 1.
                    depth_stencil_attachment: attached.then_some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth,
                        depth_ops: None,
                        stencil_ops: None,
                    }),
                    timestamp_writes: self.timer.render("scene_read", camera),
                    ..Default::default()
                })
                .forget_lifetime();
            let target = read_target(p.format, p.output, attached);
            self.draw_nodes(
                &mut pass,
                (i, view.bind_group),
                SlotList::Reading,
                Orders::All,
                target,
                extent,
                viewport,
                scissor,
                Some(depth),
                replay,
            );
            drop(pass);
            self.stats.scene_read_passes += 1;
            self.stats.render_passes += 1;
        }
        depth_1x
    }

    /// Resolve a multisampled depth (minimum of its samples) into `resolved`, within
    /// `scissor` (graphics spec §1.6).
    fn encode_depth_resolve(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        resolved: &wgpu::TextureView,
        samples: u32,
        scissor: [u32; 4],
    ) {
        let Renderer {
            device,
            pipelines,
            scene_inputs,
            stats,
            timer,
            ..
        } = self;
        let (pipeline, bgl) = pipelines.depth_resolve(device, samples);
        let bind_group = scene_inputs.resolve(device, bgl, depth);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("space3d depth resolve"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: resolved,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: timer.render("depth_resolve", None),
            ..Default::default()
        });
        let [x, y, w, h] = scissor;
        pass.set_scissor_rect(x, y, w, h);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
        stats.depth_resolves += 1;
        stats.render_passes += 1;
        stats.fullscreen_passes += 1;
    }

    /// The Distortion pass of post group `pg` (graphics spec §1.8): its settings camera's
    /// Distortion slots into the group's distortion buffer, cleared to 0, with the view's
    /// single-sample depth sampled when there is one.
    fn encode_distortion(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        i: usize,
        bind_group: usize,
        pg: usize,
        depth: Option<&wgpu::TextureView>,
        replay: bool,
    ) {
        let camera = self.frame.cameras[self.frame.post_groups[pg].camera].id;
        let Some((buffer, size, _)) = self.post.distortion_view(camera) else {
            return;
        };
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("space3d distortion"),
                timestamp_writes: self.timer.render("distortion", Some(camera)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &buffer,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        let viewport = [0.0, 0.0, size.0 as f32, size.1 as f32];
        let scissor = [0, 0, size.0, size.1];
        self.draw_nodes(
            &mut pass,
            (i, bind_group),
            SlotList::Distortion,
            Orders::All,
            distortion_target(depth.is_some()),
            size,
            viewport,
            scissor,
            depth,
            replay,
        );
        drop(pass);
        self.stats.distortion_passes += 1;
        self.stats.render_passes += 1;
    }

    /// Record the node slots of `list` with work in view `i` (group 0: its bind group
    /// `bind_group`) into `pass` (see [`super::NodeCache::draw`]), with scene depth `depth`
    /// bound in group 1 (the far dummy without).
    #[allow(clippy::too_many_arguments)]
    fn draw_nodes(
        &mut self,
        pass: &mut wgpu::RenderPass<'static>,
        (i, bind_group): (usize, usize),
        list: SlotList,
        orders: Orders,
        target: PassTarget,
        target_size: (u32, u32),
        viewport: [f32; 4],
        scissor: [u32; 4],
        depth: Option<&wgpu::TextureView>,
        replay: bool,
    ) {
        let Renderer {
            device,
            frame: f,
            nodes,
            scene_inputs,
            layouts,
            stats,
            ..
        } = self;
        let n = nodes.len();
        let Some(view_bind_group) = f.view_bind_group(bind_group) else {
            return;
        };
        let (scene_bind_group, scene_depth) = match depth {
            Some(view) => {
                let (bg, generation) = scene_inputs.with_depth(device, layouts, view);
                let depth = SceneDepthRef {
                    view,
                    size: extent(view),
                    generation,
                };
                (bg, Some(depth))
            }
            None => (&scene_inputs.none, None),
        };
        let frame = replay_info(f.info, replay);
        let np = NodePass {
            device,
            view: &f.view_infos[i],
            masks: &f.node_masks[i * n..(i + 1) * n],
            target,
            target_size,
            view_bind_group,
            scene_bind_group,
            scene_depth,
            frame,
            viewport,
            scissor,
        };
        stats.node_draws += nodes.draw(pass, list, orders, &np);
    }

    /// HdrScene overlays (graphics spec §1.5), after the last primary scene view `i`: each
    /// overlay's `prepare_scene` and `prepare`, then one single-sample pass (no depth, `Load`)
    /// into `hdr.target` with every overlay's `render`, viewport and scissor reset per overlay.
    fn encode_hdr_scene_overlays(&mut self, encoder: &mut wgpu::CommandEncoder, i: usize, hdr: &HdrScenePass<'_>) {
        if !self.overlays.has_stage(OverlayStage::HdrScene) {
            return;
        }
        let Renderer {
            device,
            queue,
            frame: f,
            overlays,
            scene_inputs,
            layouts,
            stats,
            timer,
            ..
        } = self;
        let Some(view_bind_group) = f.view_bind_group(hdr.bind_group) else {
            return;
        };
        let info = &f.view_infos[i];
        let size = extent(hdr.target);
        let (scene_bind_group, scene_depth) = match &hdr.depth {
            Some(view) => {
                let (bg, generation) = scene_inputs.with_depth(device, layouts, view);
                let depth = SceneDepthRef {
                    view,
                    size: extent(view),
                    generation,
                };
                (bg, Some(depth))
            }
            None => (&scene_inputs.none, None),
        };
        let (exposure, tonemapper) = if hdr.output == OutputEncoding::Tonemapped {
            (info.exposure, info.tonemapper)
        } else {
            (1.0, Tonemapper::None)
        };
        let target = OverlayTarget {
            size,
            format: hdr.format,
            scale_factor: overlays.scale_factor() * hdr.scale,
            kind: hdr.kind,
        };
        for (overlay, stage, reads) in overlays.iter() {
            if stage != OverlayStage::HdrScene {
                continue;
            }
            let scene = OverlayScene {
                view: info,
                mode: hdr.mode,
                output: hdr.output,
                exposure,
                tonemapper,
                scene_depth: scene_depth.filter(|_| reads.contains(SceneReads::DEPTH)),
                view_bind_group,
                scene_bind_group,
                layouts,
            };
            let mut o = lock_overlay(overlay);
            o.prepare_scene(&scene);
            o.prepare(device, queue, encoder, &target);
        }
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("space3d overlays (hdr scene)"),
                timestamp_writes: timer.render("overlays.hdr", None),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr.target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        let (w, h) = size;
        for (overlay, stage, _) in overlays.iter() {
            if stage != OverlayStage::HdrScene {
                continue;
            }
            pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, w, h);
            lock_overlay(overlay).render(&mut pass, &target);
        }
        drop(pass);
        stats.render_passes += 1;
    }

    /// Whether primary frames present to a window (whose surface texture may be unavailable)
    /// rather than to the offscreen target.
    pub(crate) fn presents_to_window(&self) -> bool {
        #[cfg(test)]
        if self.frame.simulate_occluded {
            return true;
        }
        self.surface.is_some()
    }

    /// Whether a capture re-records the last frame into the offscreen target: windowed
    /// renderers (surface textures are not readable), or a test forcing the windowed path.
    pub(crate) fn capture_replays(&self) -> bool {
        #[cfg(test)]
        if self.frame.force_capture_replay {
            return true;
        }
        self.surface.is_some()
    }

    /// Windowed capture: re-record the last frame's primary views into the offscreen target,
    /// then the overlays ([`crate::OverlayTargetKind::Capture`]) on top of them — in that
    /// order, as in a frame (overlays recorded first would be painted over by the views).
    pub(crate) fn encode_capture_replay(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_primary_to_offscreen(encoder);
        self.encode_capture_overlays(encoder);
    }

    /// Re-record the last frame's primary views into the offscreen target (for capture). The
    /// target gets the size those views were prepared for (their projection and viewports),
    /// even if the window was resized since. An intermediate primary group re-renders its scene
    /// views into `S` only if the frame skipped them (see `GroupEncode::render_scene`).
    pub(crate) fn encode_primary_to_offscreen(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.size_offscreen(self.frame.last_primary_size);
        // Re-recorded passes are not new draws: keep the frame's pass counters.
        let stats = self.stats.clone();
        self.encode_views(encoder, None, true);
        self.stats = stats;
    }
}

/// The frame facts node calls see: a capture replay's never advance time.
fn replay_info(mut info: FrameInfo, replay: bool) -> FrameInfo {
    if replay {
        info.kind = FrameKind::CaptureReplay;
        info.delta = 0.0;
    }
    info
}
