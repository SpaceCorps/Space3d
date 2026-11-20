//! Post-processing effects (graphics spec §5.2 P-POST) on the default backend: bloom (PO1), auto
//! exposure (PO2), FXAA, vignette, chromatic aberration and film grain (PO3), color grading, 3D
//! LUTs and grading hooks (PO4), hardening (PO6: GPU timings, pass and memory counts, resizes,
//! device loss). Distortion (PO5) is in `tests/post_distortion.rs`.
//! The bodies live in `common/post.rs` (`cases`), shared with the GL run (`tests/gl_post.rs`);
//! `WGPU_BACKEND=gl` runs this file on GL too.

mod common;
#[path = "common/post.rs"]
mod post_common;

use common::headless_app;
use post_common::{cases, hardening};

#[test]
fn bloom_only_above_threshold() {
    cases::bloom_only_above_threshold(headless_app);
}

#[test]
fn bloom_is_in_exposed_units() {
    cases::bloom_is_in_exposed_units(headless_app);
}

#[test]
fn energy_conserving_preserves_mean() {
    cases::energy_conserving_preserves_mean(headless_app);
}

#[test]
fn bloom_does_not_bleed_across_viewports() {
    cases::bloom_does_not_bleed_across_viewports(headless_app);
}

#[test]
fn hdr_scene_overlays_feed_bloom() {
    cases::hdr_scene_overlays_feed_bloom(headless_app);
}

#[test]
fn bloom_off_in_ldr_fallback() {
    cases::bloom_off_in_ldr_fallback(headless_app);
}

#[test]
fn bloom_skips_small_rects() {
    cases::bloom_skips_small_rects(headless_app);
}

#[test]
fn transparent_rt_bloom_weighted_by_alpha() {
    cases::transparent_rt_bloom_weighted_by_alpha(headless_app);
}

#[test]
fn bloom_steady_state() {
    cases::bloom_steady_state(headless_app);
}

#[test]
fn bloom_halo_is_independent_of_position() {
    cases::bloom_halo_is_independent_of_position(headless_app);
}

#[test]
fn bloom_downlevel_parity() {
    cases::bloom_downlevel_parity(headless_app);
}

#[test]
fn fxaa_softens_edges() {
    cases::fxaa_softens_edges(headless_app);
}

#[test]
fn fxaa_intermediate_is_dithered() {
    cases::fxaa_intermediate_is_dithered(headless_app);
}

#[test]
fn film_grain_is_seeded() {
    cases::film_grain_is_seeded(headless_app);
}

#[test]
fn grain_after_fxaa() {
    cases::grain_after_fxaa(headless_app);
}

#[test]
fn after_tonemap_overlays_untouched_by_grain_ca_vignette() {
    cases::after_tonemap_overlays_untouched_by_grain_ca_vignette(headless_app);
}

#[test]
fn vignette_darkens_corners() {
    cases::vignette_darkens_corners(headless_app);
}

#[test]
fn chromatic_aberration_splits_toward_corners() {
    cases::chromatic_aberration_splits_toward_corners(headless_app);
}

#[test]
fn lens_effects_do_not_bleed_across_viewports() {
    cases::lens_effects_do_not_bleed_across_viewports(headless_app);
}

#[test]
fn fxaa_ldr_fallback_and_steady_state() {
    cases::fxaa_ldr_fallback_and_steady_state(headless_app);
}

#[test]
fn transparent_rt_lens_effects_keep_alpha() {
    cases::transparent_rt_lens_effects_keep_alpha(headless_app);
}

#[test]
fn auto_exposure_converges_and_matches_reference() {
    cases::auto_exposure_converges_and_matches_reference(headless_app, false);
}

/// The render-pass metering path (WebGL2 class) on the default backend: ±0.5 EV.
#[test]
fn auto_exposure_downlevel_matches_reference() {
    cases::auto_exposure_converges_and_matches_reference(headless_app, true);
}

#[test]
fn instant_converges_frame_1() {
    cases::instant_converges_frame_1(headless_app);
}

#[test]
fn render_to_image_is_idempotent_with_ae_and_grain() {
    cases::render_to_image_is_idempotent_with_ae_and_grain(headless_app);
}

#[test]
fn auto_exposure_feeds_bloom() {
    cases::auto_exposure_feeds_bloom(headless_app);
}

#[test]
fn auto_exposure_per_post_group() {
    cases::auto_exposure_per_post_group(headless_app);
}

#[test]
fn auto_exposure_history_and_steady_state() {
    cases::auto_exposure_history_and_steady_state(headless_app, false);
}

#[test]
fn auto_exposure_downlevel_history_and_steady_state() {
    cases::auto_exposure_history_and_steady_state(headless_app, true);
}

#[test]
fn grading_matches_formulas() {
    cases::grading_matches_formulas(headless_app);
}

#[test]
fn display_gamma_matches_formula() {
    cases::display_gamma_matches_formula(headless_app);
}

#[test]
fn lut_identity_is_neutral() {
    cases::lut_identity_is_neutral(headless_app);
}

#[test]
fn lut_changes_colors() {
    cases::lut_changes_colors(headless_app);
}

#[test]
fn invalid_grading_hook_falls_back() {
    cases::invalid_grading_hook_falls_back(headless_app);
}

#[test]
fn post_gpu_timings_are_labelled() {
    hardening::post_gpu_timings_are_labelled(headless_app);
}

#[test]
fn post_pass_and_memory_counts() {
    hardening::post_pass_and_memory_counts(headless_app);
}

#[test]
fn resize_reallocates_once_and_stays_bounded() {
    hardening::resize_reallocates_once_and_stays_bounded(headless_app);
}

#[test]
fn a_lost_device_degrades_with_every_effect() {
    hardening::a_lost_device_degrades_with_every_effect(headless_app);
}

#[test]
fn wide_split_is_a_spectrum() {
    cases::wide_split_is_a_spectrum(headless_app);
}
