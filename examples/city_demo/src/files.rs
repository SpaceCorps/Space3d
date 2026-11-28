//! Where the CC0 asset files (`assets/`, see `ASSETS.md`) come from. Native builds embed them
//! (`include_bytes!`), so the binary is self-contained. The web build embeds nothing: it
//! fetches what the chosen tier needs from `assets/` next to the page (non-blocking, in
//! parallel), keeps the bytes for the generators, and the demo starts once they are in. A file
//! that did not arrive makes its user fall back to the procedural stand-in.

/// The bytes of `assets/<path>` (`Option<&'static [u8]>`).
macro_rules! asset {
    ($path:expr) => {{
        #[cfg(not(target_arch = "wasm32"))]
        {
            Some(include_bytes!(concat!("../assets/", $path)).as_slice())
        }
        #[cfg(target_arch = "wasm32")]
        {
            $crate::files::get($path)
        }
    }};
}

/// The three maps of a Poly Haven set under `tex/`.
pub const TEXTURE_SETS: [&str; 6] = [
    "asphalt_02",
    "concrete_pavement",
    "bitumen",
    "brushed_concrete_04",
    "brick_wall_10",
    "beige_wall_002",
];
pub const CARS: [&str; 10] = [
    "sedan",
    "sedan-sports",
    "hatchback-sports",
    "taxi",
    "suv",
    "suv-luxury",
    "van",
    "delivery",
    "truck",
    "police",
];

/// Every file the tier needs (the low tier takes the 1k HDRI and skips the street props, the
/// others the 2k HDRI and everything).
#[allow(dead_code)]
pub fn needed(low: bool) -> Vec<String> {
    let mut v = vec![format!("hdri/sunset_jhbcentral_{}.hdr", if low { "1k" } else { "2k" })];
    for set in TEXTURE_SETS {
        for map in ["Diffuse", "arm", "nor_gl"] {
            v.push(format!("tex/{set}_{map}.jpg"));
        }
    }
    for car in CARS {
        v.push(format!("models/kenney_{car}.glb"));
    }
    // the street props are 12.5 MB of 1k-textured scans for a few hydrants and bins: not on the
    // low tier (they fall away; the streets look the same from the air)
    if !low {
        for m in ["fire_hydrant", "metal_trash_can", "painted_wooden_bench"] {
            v.push(format!("models/polyhaven_{m}.glb"));
        }
    }
    v.push("models/quaternius_human.bin".into());
    v
}

#[cfg(target_arch = "wasm32")]
pub use web::{get, start};

/// Native builds embed everything: nothing to fetch.
#[cfg(not(target_arch = "wasm32"))]
pub fn start(_low: bool) -> Option<std::sync::Arc<Progress>> {
    None
}

/// How far the fetch is.
pub struct Progress {
    pub total: usize,
    pub done: std::sync::atomic::AtomicUsize,
    pub failed: std::sync::atomic::AtomicUsize,
    pub bytes: std::sync::atomic::AtomicUsize,
}

impl Progress {
    pub fn finished(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        self.done.load(Relaxed) + self.failed.load(Relaxed) >= self.total
    }

    pub fn text(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;
        format!(
            "{}/{} files, {:.1} MB",
            self.done.load(Relaxed) + self.failed.load(Relaxed),
            self.total,
            self.bytes.load(Relaxed) as f64 / 1e6
        )
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::Progress;
    use space3d::assets::{AssetPath, AssetSource, HttpSource};
    use std::collections::HashMap;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::{Arc, LazyLock, Mutex};

    static STORE: LazyLock<Mutex<HashMap<String, &'static [u8]>>> = LazyLock::new(Default::default);

    /// A fetched file (kept for the life of the page).
    pub fn get(path: &str) -> Option<&'static [u8]> {
        STORE.lock().ok()?.get(path).copied()
    }

    /// Start fetching every file the tier needs from `assets/` next to the page.
    pub fn start(low: bool) -> Option<Arc<Progress>> {
        let names = super::needed(low);
        let progress = Arc::new(Progress {
            total: names.len(),
            done: Default::default(),
            failed: Default::default(),
            bytes: Default::default(),
        });
        let source = Arc::new(HttpSource::new("assets"));
        for name in names {
            let (progress, source) = (progress.clone(), source.clone());
            let _ = space3d::tasks::spawn(async move {
                match source.read(&AssetPath::new(&name)).await {
                    Ok(bytes) => {
                        progress.bytes.fetch_add(bytes.len(), Relaxed);
                        if let Ok(mut s) = STORE.lock() {
                            s.insert(name, Box::leak(bytes.into_boxed_slice()));
                        }
                        progress.done.fetch_add(1, Relaxed);
                    }
                    Err(e) => {
                        eprintln!("city_demo: {name}: {e}");
                        progress.failed.fetch_add(1, Relaxed);
                    }
                }
            });
        }
        Some(progress)
    }
}
