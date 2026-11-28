//! The CC0 assets (Poly Haven, see `ASSETS.md`): the dusk HDRI and the PBR sets of the road,
//! pavement and roofs, embedded in the binary. Every loader returns `None` on a decode error
//! and the caller falls back to the procedural stand-ins.

use crate::procgen::{Facade, TextureSet};
use space3d::assets::edit::texture::decode_image;
use space3d::assets::{AssetLoader, HdrImageLoader, LoadContext};
use space3d::prelude::*;
use space3d::render::CubemapSource;

/// The sky and where its sun is.
pub struct Sky {
    pub source: CubemapSource,
    /// Azimuth (`atan2(x, -z)`) and elevation of the brightest spot, when it is a sun.
    pub sun: Option<(f32, f32)>,
}

pub fn sky(low: bool, face: u32) -> Option<Sky> {
    let bytes = if low {
        asset!("hdri/sunset_jhbcentral_1k.hdr")
    } else {
        asset!("hdri/sunset_jhbcentral_2k.hdr")
    }?;
    let hdr = HdrImageLoader
        .load(bytes, &LoadContext::new("sunset_jhbcentral.hdr"))
        .ok()?;
    // the brightest pixel, summed over a small neighbourhood so a single hot texel does not count
    let (w, h) = (hdr.width as usize, hdr.height as usize);
    let lum = |x: usize, y: usize| {
        let p = hdr.pixels[y * w + x];
        0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
    };
    let (mut best, mut bx, mut by) = (0.0f32, 0, 0);
    for y in (2..h - 2).step_by(2) {
        for x in (2..w - 2).step_by(2) {
            let l = lum(x, y) + lum(x + 1, y) + lum(x, y + 1) + lum(x - 1, y) + lum(x, y - 1);
            if l > best {
                (best, bx, by) = (l, x, y);
            }
        }
    }
    let u = bx as f32 / w as f32;
    let v = by as f32 / h as f32;
    let az = (u - 0.5) * std::f32::consts::TAU;
    let el = (0.5 - v) * std::f32::consts::PI;
    println!(
        "city_demo: HDRI sun peak {:.0} at azimuth {:.0}°, elevation {:.1}°",
        best / 5.0,
        az.to_degrees(),
        el.to_degrees()
    );
    Some(Sky {
        source: CubemapSource::from_equirect_hdr(&hdr).with_face_size(face),
        sun: (best / 5.0 > 8.0).then_some((az, el)),
    })
}

fn halve(img: ImageData) -> ImageData {
    let (w, h) = (img.width / 2, img.height / 2);
    let mut out = ImageData::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0u32; 4];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = img.get(x * 2 + dx, y * 2 + dy);
                for k in 0..4 {
                    acc[k] += u32::from(p[k]);
                }
            }
            out.set(
                x,
                y,
                [
                    (acc[0] / 4) as u8,
                    (acc[1] / 4) as u8,
                    (acc[2] / 4) as u8,
                    (acc[3] / 4) as u8,
                ],
            );
        }
    }
    out
}

/// A Poly Haven PBR set (diffuse, `arm` = AO/roughness/metal, GL normal), made wetter: the
/// albedo scaled by `base_mul` and the roughness by `rough_mul`. Halved in size on `low`.
fn surface(diff: &[u8], arm: &[u8], nor: &[u8], low: bool, base_mul: f32, rough_mul: f32) -> Option<TextureSet> {
    let (mut base, mut orm, mut normal) = (
        decode_image(diff).ok()?,
        decode_image(arm).ok()?,
        decode_image(nor).ok()?,
    );
    if low {
        (base, orm, normal) = (halve(base), halve(orm), halve(normal));
    }
    for p in base.pixels.chunks_mut(4) {
        for c in &mut p[..3] {
            *c = (f32::from(*c) * base_mul).min(255.0) as u8;
        }
    }
    for p in orm.pixels.chunks_mut(4) {
        p[1] = (f32::from(p[1]) * rough_mul).clamp(4.0, 255.0) as u8;
    }
    Some(TextureSet {
        base,
        normal,
        orm,
        emissive: ImageData::filled(1, 1, [0, 0, 0, 255]),
    })
}

/// The three maps of a set, `None` while a file is missing.
macro_rules! set {
    ($name:literal) => {
        match (
            asset!(concat!("tex/", $name, "_Diffuse.jpg")),
            asset!(concat!("tex/", $name, "_arm.jpg")),
            asset!(concat!("tex/", $name, "_nor_gl.jpg")),
        ) {
            (Some(d), Some(a), Some(n)) => (d, a, n),
            _ => return None,
        }
    };
}

/// Wet asphalt (`asphalt_02`).
pub fn asphalt(low: bool) -> Option<TextureSet> {
    let (d, a, n) = set!("asphalt_02");
    surface(
        d,
        a,
        n,
        low,
        crate::knob("CITY_ROAD_BASE", 0.12),
        crate::knob("CITY_ROAD_ROUGH", 0.18),
    )
}

/// Pavement (`concrete_pavement`), damp.
pub fn paving(low: bool) -> Option<TextureSet> {
    let (d, a, n) = set!("concrete_pavement");
    surface(
        d,
        a,
        n,
        low,
        crate::knob("CITY_PAVE_BASE", 0.3),
        crate::knob("CITY_PAVE_ROUGH", 0.6),
    )
}

/// Roofs (`bitumen`).
pub fn roof(low: bool) -> Option<TextureSet> {
    let (d, a, n) = set!("bitumen");
    surface(d, a, n, low, 0.7, 0.85)
}

// ----------------------------------------------------------------------------------------------
// Models (GLB)
// ----------------------------------------------------------------------------------------------

use space3d::assets::loaders::load_gltf;
use space3d::assets::render::{ModelUploadOptions, PreparedModel, prepare_model};

/// One vehicle model: size in model units (before `scale`) and how common it is.
pub struct CarModel {
    pub model: PreparedModel,
    pub half_len: f32,
    pub half_w: f32,
    pub scale: f32,
    pub weight: u32,
}

pub struct Models {
    pub cars: Vec<CarModel>,
    pub hydrant: Option<PreparedModel>,
    pub trash: Option<PreparedModel>,
    pub bench: Option<PreparedModel>,
}

fn prepared(bytes: &[u8], name: &str) -> Option<PreparedModel> {
    match load_gltf(bytes, &LoadContext::new(name), true) {
        Ok(m) => Some(prepare_model(&m, &ModelUploadOptions::default())),
        Err(e) => {
            eprintln!("city_demo: {name}: {e}");
            None
        }
    }
}

macro_rules! car {
    ($name:literal, $half_len:expr, $half_w:expr, $scale:expr, $weight:expr) => {
        (
            $name,
            asset!(concat!("models/kenney_", $name, ".glb")),
            $half_len,
            $half_w,
            $scale,
            $weight,
        )
    };
}

pub fn models() -> Models {
    let list = [
        car!("sedan", 1.3, 0.75, 1.3, 4),
        car!("sedan-sports", 1.3, 0.65, 1.3, 1),
        car!("hatchback-sports", 1.45, 0.65, 1.3, 2),
        car!("taxi", 1.4, 0.75, 1.3, 3),
        car!("suv", 1.35, 0.75, 1.32, 3),
        car!("suv-luxury", 1.45, 0.75, 1.32, 1),
        car!("van", 1.4, 0.75, 1.38, 1),
        car!("delivery", 1.65, 0.75, 1.5, 1),
        car!("truck", 1.5, 0.75, 1.5, 1),
        car!("police", 1.57, 0.75, 1.3, 1),
    ];
    let cars = list
        .into_iter()
        .filter_map(|(name, bytes, half_len, half_w, scale, weight)| {
            Some(CarModel {
                model: prepared(bytes?, name)?,
                half_len,
                half_w,
                scale,
                weight,
            })
        })
        .collect();
    Models {
        cars,
        hydrant: asset!("models/polyhaven_fire_hydrant.glb").and_then(|b| prepared(b, "hydrant.glb")),
        trash: asset!("models/polyhaven_metal_trash_can.glb").and_then(|b| prepared(b, "trash.glb")),
        bench: asset!("models/polyhaven_painted_wooden_bench.glb").and_then(|b| prepared(b, "bench.glb")),
    }
}

/// A real wall for a facade kind and the meters one repeat of it covers (`None` for glass:
/// curtain walls are smooth dark metal, drawn from the facade atlas alone).
pub fn wall(kind: Facade, low: bool) -> Option<(TextureSet, f32)> {
    let (set, tile) = match kind {
        Facade::Concrete => (set!("brushed_concrete_04"), 3.0),
        Facade::Brick => (set!("brick_wall_10"), 1.6),
        Facade::Stone => (set!("beige_wall_002"), 2.5),
        Facade::Glass => return None,
    };
    surface(
        set.0,
        set.1,
        set.2,
        low,
        if kind == Facade::Brick { 1.0 } else { 0.95 },
        0.95,
    )
    .map(|s| (s, tile))
}
