//! Post-processing on wgpu's GL backend — the WebGL2 / GLES 3.0 path (graphics spec §4.1 class
//! B: float color buffers through `EXT_color_buffer_float`): the `tests/post.rs` and
//! `tests/post_distortion.rs` cases on a GL adapter (auto exposure meters with the render-pass
//! chain there: GLES 3.0 has no compute; distortion meshes draw without `base_vertex`), plus
//! bloom parity with the default backend.
//!
//! Skips when there is no GL adapter. On macOS, ANGLE (GLES 3.0, WebGL2's feature level) provides
//! one; ANGLE crashes when several GL contexts render from different threads, so run it with one
//! test thread:
//!
//! ```text
//! DYLD_FALLBACK_LIBRARY_PATH=<dir with libEGL.dylib> \
//!     cargo test -p space3d-render --features angle --test gl_post -- --test-threads=1
//! ```

mod common;
#[path = "common/post.rs"]
mod post_common;

use post_common::{cases, distortion, hardening};
use space3d_core::prelude::*;
use space3d_render::*;
use space3d_test::{CompareOptions, compare_images};

fn gl_app(width: u32, height: u32) -> Option<App> {
    gl_app_with(width, height, 1)
}

fn gl_msaa_app(width: u32, height: u32) -> Option<App> {
    gl_app_with(width, height, 4)
}

fn gl_app_with(width: u32, height: u32, msaa_samples: u32) -> Option<App> {
    let renderer = pollster::block_on(Renderer::new_headless_with(RendererConfig {
        width,
        height,
        backends: Some(wgpu::Backends::GL),
        msaa_samples,
        ..Default::default()
    }));
    match renderer {
        Ok(r) => {
            assert_eq!(r.capabilities().adapter.backend, wgpu::Backend::Gl);
            eprintln!(
                "GL adapter: {} ({}), bloom chain {:?}",
                r.capabilities().adapter.name,
                r.capabilities().adapter.driver_info,
                r.graphics_caps().bloom_format,
            );
            if r.graphics_caps().bloom_format.is_none() {
                eprintln!("skipping GL post test: no float render targets (class C: no bloom)");
                return None;
            }
            let mut app = App::new();
            app.add_plugin(RenderPlugin::default());
            app.insert_resource(r);
            Some(app)
        }
        Err(e) => {
            eprintln!("skipping GL test: no GL adapter ({e})");
            None
        }
    }
}

#[test]
fn gl_bloom_only_above_threshold() {
    cases::bloom_only_above_threshold(gl_app);
}

#[test]
fn gl_bloom_is_in_exposed_units() {
    cases::bloom_is_in_exposed_units(gl_app);
}

#[test]
fn gl_energy_conserving_preserves_mean() {
    cases::energy_conserving_preserves_mean(gl_app);
}

#[test]
fn gl_bloom_does_not_bleed_across_viewports() {
    cases::bloom_does_not_bleed_across_viewports(gl_app);
}

#[test]
fn gl_hdr_scene_overlays_feed_bloom() {
    cases::hdr_scene_overlays_feed_bloom(gl_app);
}

#[test]
fn gl_bloom_off_in_ldr_fallback() {
    cases::bloom_off_in_ldr_fallback(gl_app);
}

#[test]
fn gl_bloom_skips_small_rects() {
    cases::bloom_skips_small_rects(gl_app);
}

#[test]
fn gl_transparent_rt_bloom_weighted_by_alpha() {
    cases::transparent_rt_bloom_weighted_by_alpha(gl_app);
}

#[test]
fn gl_bloom_steady_state() {
    cases::bloom_steady_state(gl_app);
}

#[test]
fn gl_bloom_halo_is_independent_of_position() {
    cases::bloom_halo_is_independent_of_position(gl_app);
}

#[test]
fn gl_bloom_downlevel_parity() {
    cases::bloom_downlevel_parity(gl_app);
}

#[test]
fn gl_fxaa_softens_edges() {
    cases::fxaa_softens_edges(gl_app);
}

#[test]
fn gl_fxaa_intermediate_is_dithered() {
    cases::fxaa_intermediate_is_dithered(gl_app);
}

#[test]
fn gl_film_grain_is_seeded() {
    cases::film_grain_is_seeded(gl_app);
}

#[test]
fn gl_grain_after_fxaa() {
    cases::grain_after_fxaa(gl_app);
}

#[test]
fn gl_after_tonemap_overlays_untouched_by_grain_ca_vignette() {
    cases::after_tonemap_overlays_untouched_by_grain_ca_vignette(gl_app);
}

#[test]
fn gl_vignette_darkens_corners() {
    cases::vignette_darkens_corners(gl_app);
}

#[test]
fn gl_chromatic_aberration_splits_toward_corners() {
    cases::chromatic_aberration_splits_toward_corners(gl_app);
}

#[test]
fn gl_lens_effects_do_not_bleed_across_viewports() {
    cases::lens_effects_do_not_bleed_across_viewports(gl_app);
}

#[test]
fn gl_fxaa_ldr_fallback_and_steady_state() {
    cases::fxaa_ldr_fallback_and_steady_state(gl_app);
}

#[test]
fn gl_transparent_rt_lens_effects_keep_alpha() {
    cases::transparent_rt_lens_effects_keep_alpha(gl_app);
}

/// Emissive quads of 2 … 40 over a dark clear with the default bloom: GL (level views bound
/// with `TEXTURE_BASE_LEVEL`/`MAX_LEVEL` while rendering the next level) looks like the
/// default backend.
#[test]
fn gl_bloom_matches_default_backend() {
    let scene = |mut app: App| {
        for (i, v) in [2.0f32, 5.0, 12.0, 40.0].into_iter().enumerate() {
            let color = [Vec3::new(1.0, 0.3, 0.1), Vec3::new(0.2, 0.6, 1.0)][i % 2] * v;
            post_common::quad(
                &mut app,
                color,
                Vec2::new(-1.5 + i as f32, 0.3 - 0.2 * i as f32),
                Vec2::splat(0.15),
            );
        }
        post_common::camera(
            &mut app,
            0.0,
            Vec4::new(0.01, 0.01, 0.02, 1.0),
            Some(PostProcess::default()),
        );
        post_common::render(&mut app)
    };
    let Some(gl) = gl_app(128, 64) else { return };
    let Some(default) = common::headless_app(128, 64) else {
        return;
    };
    let (gl, default) = (scene(gl), scene(default));
    let cmp = compare_images(&default, &gl, &CompareOptions::perceptual());
    eprintln!("bloom GL vs default: {}", cmp.summary());
    assert!(cmp.passed, "{}", cmp.summary());
}

#[test]
fn gl_auto_exposure_converges_and_matches_reference() {
    cases::auto_exposure_converges_and_matches_reference(gl_app, false);
}

#[test]
fn gl_instant_converges_frame_1() {
    cases::instant_converges_frame_1(gl_app);
}

#[test]
fn gl_render_to_image_is_idempotent_with_ae_and_grain() {
    cases::render_to_image_is_idempotent_with_ae_and_grain(gl_app);
}

#[test]
fn gl_auto_exposure_feeds_bloom() {
    cases::auto_exposure_feeds_bloom(gl_app);
}

#[test]
fn gl_auto_exposure_per_post_group() {
    cases::auto_exposure_per_post_group(gl_app);
}

#[test]
fn gl_auto_exposure_history_and_steady_state() {
    cases::auto_exposure_history_and_steady_state(gl_app, false);
}

#[test]
fn gl_grading_matches_formulas() {
    cases::grading_matches_formulas(gl_app);
}

#[test]
fn gl_display_gamma_matches_formula() {
    cases::display_gamma_matches_formula(gl_app);
}

#[test]
fn gl_lut_identity_is_neutral() {
    cases::lut_identity_is_neutral(gl_app);
}

#[test]
fn gl_lut_changes_colors() {
    cases::lut_changes_colors(gl_app);
}

#[test]
fn gl_invalid_grading_hook_falls_back() {
    cases::invalid_grading_hook_falls_back(gl_app);
}

#[test]
fn gl_distortion_shifts_the_background() {
    distortion::distortion_shifts_the_background(gl_app, |_| {});
}

#[test]
fn gl_distortion_shifts_the_background_msaa() {
    distortion::distortion_shifts_the_background(gl_msaa_app, |_| {});
}

#[test]
fn gl_distortion_chromatic_split() {
    distortion::distortion_chromatic_split(gl_app);
}

/// GL samples single-sample depth without an attachment: writers are occluded.
#[test]
fn gl_distortion_occluded_by_closer_geometry() {
    distortion::distortion_occluded_by_closer_geometry(gl_app, |_| {}, true);
}

/// GLES 3.0 cannot sample multisampled depth: with MSAA, distortion is not occluded (spec §1.6,
/// documented).
#[test]
fn gl_distortion_not_occluded_with_msaa() {
    distortion::distortion_occluded_by_closer_geometry(gl_msaa_app, |_| {}, false);
}

#[test]
fn gl_foreground_is_not_dragged_over_the_background() {
    distortion::foreground_is_not_dragged_over_the_background(gl_app);
}

#[test]
fn gl_hdr_scene_overlays_are_distorted() {
    distortion::hdr_scene_overlays_are_distorted(gl_app);
}

#[test]
fn gl_no_distortion_pass_without_writers() {
    distortion::no_distortion_pass_without_writers(gl_app);
}

#[test]
fn gl_radial_distortion_pushes_outward() {
    distortion::radial_distortion_pushes_outward(gl_app);
}

#[test]
fn gl_normal_writers_are_capped_by_their_size() {
    distortion::normal_writers_are_capped_by_their_size(gl_app);
}

#[test]
fn gl_distortion_does_not_bleed_across_viewports() {
    distortion::distortion_does_not_bleed_across_viewports(gl_app);
}

#[test]
fn gl_distortion_steady_state() {
    distortion::distortion_steady_state(gl_app);
}

#[test]
fn gl_post_pass_and_memory_counts() {
    hardening::post_pass_and_memory_counts(gl_app);
}

#[test]
fn gl_resize_reallocates_once_and_stays_bounded() {
    hardening::resize_reallocates_once_and_stays_bounded(gl_app);
}

#[test]
fn gl_a_lost_device_degrades_with_every_effect() {
    hardening::a_lost_device_degrades_with_every_effect(gl_app);
}

#[test]
fn gl_wide_split_is_a_spectrum() {
    cases::wide_split_is_a_spectrum(gl_app);
}
