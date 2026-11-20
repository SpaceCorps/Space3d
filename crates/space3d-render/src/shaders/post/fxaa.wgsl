// Space3d FXAA (graphics spec §5.2 PO3, design-post §5.5): FXAA 3.11 (Timothy Lottes), PC quality
// path, ported to WGSL. The renderer appends this to the composite source (PostParams, the
// finish) after the quality constants:
//
//   const S3D_FXAA_STEPS: i32;              // search steps (3, 5 or 12)
//   const S3D_FXAA_STEP: array<f32, 12>;    // step sizes P0..P11
//   const S3D_FXAA_EDGE: f32;               // edge threshold (1/4, 1/6, 1/8)
//   const S3D_FXAA_EDGE_MIN: f32;           // darkness threshold
//   const S3D_FXAA_SUBPIX: f32;             // sub-pixel aliasing removal
//
// It reads the post group's LDR intermediate L (binding 5: an sRGB texture, so FXAA sees linear
// values; luma is sqrt(Rec. 601 luma) ≈ perceptual; the composite dithered it) and writes D, then
// applies the last-pass finish (grain, dither where it blended, encoding). Every tap is clamped to the group's rect: split screens
// never bleed into each other. Explicit-LOD sampling only (WebGL2-safe).

@group(1) @binding(5) var fxaa_tex: texture_2d<f32>;

// L at pixel position `p` (D pixels, texel centers at +0.5), clamped to the group's rect in D.
fn fxaa_tap(p: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(fxaa_tex));
    let lo = max(post.out_rect.xy, vec2<f32>(0.0)) + 0.5;
    let hi = max(lo, min(post.out_rect.xy + post.out_rect.zw, size) - 0.5);
    return textureSampleLevel(fxaa_tex, post_sampler, clamp(p, lo, hi) / size, 0.0);
}

// Whether `fxaa` returned the pixel of L unchanged (no edge there).
var<private> fxaa_copied: bool = false;

fn fxaa_luma(c: vec4<f32>) -> f32 {
    return sqrt(dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114)));
}

fn fxaa_luma_at(p: vec2<f32>) -> f32 {
    return fxaa_luma(fxaa_tap(p));
}

// FXAA 3.11 quality at D pixel center `pos` (the original works in uv with rcpFrame; here one
// pixel is 1.0).
fn fxaa(pos: vec2<f32>) -> vec4<f32> {
    let rgby_m = fxaa_tap(pos);
    let luma_m = fxaa_luma(rgby_m);
    var luma_s = fxaa_luma_at(pos + vec2<f32>(0.0, 1.0));
    let luma_e = fxaa_luma_at(pos + vec2<f32>(1.0, 0.0));
    var luma_n = fxaa_luma_at(pos + vec2<f32>(0.0, -1.0));
    let luma_w = fxaa_luma_at(pos + vec2<f32>(-1.0, 0.0));
    let max_sm = max(luma_s, luma_m);
    let min_sm = min(luma_s, luma_m);
    let max_esm = max(luma_e, max_sm);
    let min_esm = min(luma_e, min_sm);
    let max_wn = max(luma_n, luma_w);
    let min_wn = min(luma_n, luma_w);
    let range_max = max(max_wn, max_esm);
    let range_min = min(min_wn, min_esm);
    let range = range_max - range_min;
    if (range < max(S3D_FXAA_EDGE_MIN, range_max * S3D_FXAA_EDGE)) {
        fxaa_copied = true;
        return rgby_m;
    }
    let luma_nw = fxaa_luma_at(pos + vec2<f32>(-1.0, -1.0));
    let luma_se = fxaa_luma_at(pos + vec2<f32>(1.0, 1.0));
    let luma_ne = fxaa_luma_at(pos + vec2<f32>(1.0, -1.0));
    let luma_sw = fxaa_luma_at(pos + vec2<f32>(-1.0, 1.0));
    let luma_ns = luma_n + luma_s;
    let luma_we = luma_w + luma_e;
    let subpix_rcp_range = 1.0 / range;
    let subpix_nswe = luma_ns + luma_we;
    let edge_horz1 = -2.0 * luma_m + luma_ns;
    let edge_vert1 = -2.0 * luma_m + luma_we;
    let luma_nese = luma_ne + luma_se;
    let luma_nwne = luma_nw + luma_ne;
    let edge_horz2 = -2.0 * luma_e + luma_nese;
    let edge_vert2 = -2.0 * luma_n + luma_nwne;
    let luma_nwsw = luma_nw + luma_sw;
    let luma_swse = luma_sw + luma_se;
    let edge_horz4 = abs(edge_horz1) * 2.0 + abs(edge_horz2);
    let edge_vert4 = abs(edge_vert1) * 2.0 + abs(edge_vert2);
    let edge_horz3 = -2.0 * luma_w + luma_nwsw;
    let edge_vert3 = -2.0 * luma_s + luma_swse;
    let edge_horz = abs(edge_horz3) + edge_horz4;
    let edge_vert = abs(edge_vert3) + edge_vert4;
    let subpix_nwswnese = luma_nwsw + luma_nese;
    // A horizontal edge (the span runs along x): step along x, shift the sample along y.
    let horz_span = edge_horz >= edge_vert;
    let subpix_a = subpix_nswe * 2.0 + subpix_nwswnese;
    if (!horz_span) {
        luma_n = luma_w;
        luma_s = luma_e;
    }
    let subpix_b = subpix_a * (1.0 / 12.0) - luma_m;
    let gradient_n = luma_n - luma_m;
    let gradient_s = luma_s - luma_m;
    var luma_nn = luma_n + luma_m;
    let luma_ss = luma_s + luma_m;
    let pair_n = abs(gradient_n) >= abs(gradient_s);
    let gradient = max(abs(gradient_n), abs(gradient_s));
    var length_sign = 1.0;
    if (pair_n) {
        length_sign = -1.0;
    }
    let subpix_c = clamp(abs(subpix_b) * subpix_rcp_range, 0.0, 1.0);
    var pos_b = pos;
    var off_np = vec2<f32>(0.0, 1.0);
    if (horz_span) {
        off_np = vec2<f32>(1.0, 0.0);
        pos_b.y += length_sign * 0.5;
    } else {
        pos_b.x += length_sign * 0.5;
    }
    var pos_n = pos_b - off_np * S3D_FXAA_STEP[0];
    var pos_p = pos_b + off_np * S3D_FXAA_STEP[0];
    let subpix_d = -2.0 * subpix_c + 3.0;
    var luma_end_n = fxaa_luma_at(pos_n);
    let subpix_e = subpix_c * subpix_c;
    var luma_end_p = fxaa_luma_at(pos_p);
    if (!pair_n) {
        luma_nn = luma_ss;
    }
    let gradient_scaled = gradient * 0.25;
    let luma_mm = luma_m - luma_nn * 0.5;
    let subpix_f = subpix_d * subpix_e;
    let luma_mlt_zero = luma_mm < 0.0;
    luma_end_n -= luma_nn * 0.5;
    luma_end_p -= luma_nn * 0.5;
    var done_n = abs(luma_end_n) >= gradient_scaled;
    var done_p = abs(luma_end_p) >= gradient_scaled;
    if (!done_n) {
        pos_n -= off_np * S3D_FXAA_STEP[1];
    }
    if (!done_p) {
        pos_p += off_np * S3D_FXAA_STEP[1];
    }
    var steps = S3D_FXAA_STEP;
    for (var i = 2; i < S3D_FXAA_STEPS; i++) {
        if (done_n && done_p) {
            break;
        }
        if (!done_n) {
            luma_end_n = fxaa_luma_at(pos_n) - luma_nn * 0.5;
        }
        if (!done_p) {
            luma_end_p = fxaa_luma_at(pos_p) - luma_nn * 0.5;
        }
        done_n = abs(luma_end_n) >= gradient_scaled;
        done_p = abs(luma_end_p) >= gradient_scaled;
        if (!done_n) {
            pos_n -= off_np * steps[i];
        }
        if (!done_p) {
            pos_p += off_np * steps[i];
        }
    }
    var dst_n = pos.x - pos_n.x;
    var dst_p = pos_p.x - pos.x;
    if (!horz_span) {
        dst_n = pos.y - pos_n.y;
        dst_p = pos_p.y - pos.y;
    }
    let good_span_n = (luma_end_n < 0.0) != luma_mlt_zero;
    let span_length = dst_p + dst_n;
    let good_span_p = (luma_end_p < 0.0) != luma_mlt_zero;
    let span_length_rcp = 1.0 / span_length;
    let direction_n = dst_n < dst_p;
    let dst = min(dst_n, dst_p);
    let good_span = select(good_span_p, good_span_n, direction_n);
    let subpix_g = subpix_f * subpix_f;
    let pixel_offset = dst * (-span_length_rcp) + 0.5;
    let subpix_h = subpix_g * S3D_FXAA_SUBPIX;
    let pixel_offset_good = select(0.0, pixel_offset, good_span);
    let pixel_offset_subpix = max(pixel_offset_good, subpix_h);
    var pos_m = pos;
    if (horz_span) {
        pos_m.y += pixel_offset_subpix * length_sign;
    } else {
        pos_m.x += pixel_offset_subpix * length_sign;
    }
    return fxaa_tap(pos_m);
}

@fragment
fn fs_fxaa(in: FullscreenOut) -> @location(0) vec4<f32> {
    let c = fxaa(in.pos.xy);
    let transparent = post_has(S3D_POST_TRANSPARENT);
    var a = 1.0;
    var rgb = c.rgb;
    if (transparent) {
        a = c.a;
        if (a > 1.0 / 255.0) {
            rgb = rgb / a;
        }
    }
    // A pixel FXAA copied is an 8-bit value the composite dithered already: a second dither would
    // only add noise (σ 0.5 → 0.7 LSB). Blended pixels are fractional again and get it; so does
    // everything under grain or in a transparent group (premultiplied, so not on a code).
    let dithered = fxaa_copied && !transparent && !post_has(S3D_POST_GRAIN);
    rgb = post_finish_with(rgb, in.pos.xy, !dithered);
    if (transparent) {
        rgb = rgb * a;
    }
    return vec4<f32>(rgb, a);
}
