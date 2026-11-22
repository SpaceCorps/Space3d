//! The demo's rules and simulation, ported from the browser prototype: volleys from the guns of
//! the shooter, the shield / hull split (clamp(absorbance − penetration)), piercing, siphon,
//! blasts, shield wear, break and regeneration, and the Aegis.

use crate::catalog::*;
use crate::envelope;
use crate::node::*;
use crate::sbe;
use space3d::assets::render::{ModelUploadOptions, PreparedModel, RenderModel, prepare_model, spawn_model};
use space3d::assets::{AssetPath, AssetSource, LoadContext, default_source};
use space3d::glam::Mat4;
use space3d::particles::prelude::*;
use space3d::prelude::*;
use space3d::render::{GraphicsQuality, Projection, Renderer};
use std::collections::{HashMap, VecDeque};
use std::f32::consts::{PI, TAU};
use std::sync::Arc;

/// World units per model unit of your ship (the client draws its models × 110).
pub const SCALE: f32 = 110.0;
/// Margin used to place the ships and fit the Aegis around them.
pub const MARGIN: f32 = 1.08;
const MAX_IMPACTS: usize = 12;
const BREAK_MAX: f32 = 1.2;
const AEGIS_DUR: f32 = 8.0;
const AEGIS_DEPLOY: f32 = 0.4;
const AEGIS_RETRACT: f32 = 0.4;
const AEGIS_FATIGUE: f32 = 20.0;
const SIPHON_COL: [f32; 3] = [0.12, 1.5, 1.25];
/// A drain's packets leave one after the other over this time (s)…
const SIPHON_SPREAD: f32 = 0.2;
/// …and each flies for this long (s).
const SIPHON_FLIGHT: f32 = 0.3;
/// What your Siphon gives your shield (%): x1, what a drain takes from yours with wear.
const SIPHON_GAIN: f32 = 7.0;
const BLAST_COL: [f32; 3] = [3.2, 1.6, 0.5];
const HULL_SPARK_COL: [f32; 3] = [3.0, 1.1, 0.25];
const FOV_Y: f32 = 35.0;

/// A test option: an environment variable natively, a URL parameter on the web.
pub fn config(key: &str) -> Option<String> {
    // the tests set their state themselves, whatever the shell exports
    if cfg!(test) {
        return None;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let search = web_sys::window()?.location().search().ok()?;
        web_sys::UrlSearchParams::new_with_str(&search).ok()?.get(key)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var(key).ok()
    }
}

/// A player command, applied at the next fixed tick (rule 4 of AGENTS.md).
#[derive(Clone, Copy, Debug)]
pub enum Cmd {
    /// A volley at a unit (`None`: the chosen target) from an angle (`None`: a side that
    /// doesn't cross the neighbours).
    Fire(Option<f32>, Option<usize>),
    Break,
    Restore,
    Aegis,
    /// The shield slider moved.
    Pct,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lang {
    Fr,
    En,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    All,
    You,
    Player,
    Mob,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Main,
    Ally,
    Foe,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Up,
    Breaking,
    Down,
    Restoring,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AegisState {
    Off,
    Deploy,
    On,
    Retract,
    Break,
}

/// Everything the panel can change.
pub struct Settings {
    pub lang: Lang,
    pub main: usize,
    pub ally: usize,
    pub foe: usize,
    pub multi: bool,
    pub skin: [f32; 3],
    pub ally_skin: usize,
    pub mob_colors: HashMap<usize, [f32; 3]>,
    pub absorb: f32,
    pub ally_absorb: f32,
    pub weapon: usize,
    pub pen: f32,
    pub miss: bool,
    pub shooter: usize,
    pub view_self: bool,
    /// Bolts per volley: 0 = as seen in game.
    pub shots: usize,
    pub auto: bool,
    pub rate: f32,
    pub speed: f32,
    pub paused: bool,
    pub target: Target,
    pub sound: bool,
    pub volume: f32,
    pub spin: bool,
    pub orbit: bool,
    pub piercing_fx: bool,
    pub wear: bool,
    pub regen: f32,
    pub aegis_cooldown: f32,
    pub pct: f32,
    pub intensity: f32,
    pub rest: f32,
    pub fresnel: f32,
    pub back: f32,
    pub hex: bool,
    pub hex_scale: f32,
    pub wave_speed: f32,
    pub wave_width: f32,
    pub wave_life: f32,
    pub flash: f32,
    pub bloom: f32,
    pub exposure: f32,
    pub quality: GraphicsQuality,
}

impl Default for Settings {
    fn default() -> Self {
        let env = config;
        let hull = |k: &str, d: &str| env(k).and_then(|v| hull_index(&v)).or(hull_index(d)).unwrap_or(0);
        let weapon = env("SHIELD_WEAPON")
            .and_then(|k| WEAPONS.iter().position(|w| w.key == k))
            .unwrap_or(0);
        let target = match env("SHIELD_TARGET").as_deref() {
            Some("you") => Target::You,
            Some("player") => Target::Player,
            Some("mob") => Target::Mob,
            _ => Target::All,
        };
        let quality = match env("SHIELD_QUALITY").as_deref() {
            Some("low") => GraphicsQuality::Low,
            Some("medium") => GraphicsQuality::Medium,
            Some("ultra") => GraphicsQuality::Ultra,
            _ => GraphicsQuality::High,
        };
        Self {
            lang: if env("SHIELD_LANG").as_deref() == Some("en") {
                Lang::En
            } else {
                Lang::Fr
            },
            main: hull("SHIELD_MAIN", "Storm"),
            ally: hull("SHIELD_ALLY", "Nomad"),
            foe: hull("SHIELD_FOE", "seeker"),
            multi: env("SHIELD_SOLO").is_none(),
            skin: PALETTES[0].2,
            ally_skin: 1,
            mob_colors: HashMap::new(),
            absorb: 62.0,
            ally_absorb: 91.0,
            weapon,
            pen: 0.0,
            miss: false,
            shooter: 3,
            view_self: true,
            shots: 0,
            auto: true,
            rate: env("SHIELD_RATE").and_then(|v| v.parse().ok()).unwrap_or(1.0),
            speed: 1.0,
            paused: false,
            target,
            sound: true,
            volume: 0.6,
            spin: true,
            orbit: false,
            piercing_fx: true,
            wear: env("SHIELD_WEAR").is_some(),
            regen: 15.0,
            aegis_cooldown: 5.0,
            pct: 100.0,
            intensity: 1.0,
            rest: 0.35,
            fresnel: 3.0,
            back: 0.45,
            hex: true,
            hex_scale: 14.0,
            wave_speed: 1.6,
            wave_width: 0.07,
            wave_life: 0.9,
            flash: 1.0,
            bloom: 0.35,
            exposure: 0.0,
            quality,
        }
    }
}

// ---------------------------------------------------------------- hull assets

/// CPU part of a hull, built off the frame.
pub struct HullContent {
    model: PreparedModel,
    center: Vec3,
    size: Vec3,
    radii: Vec3,
    shell: MeshData,
    shell_tris: Vec<[Vec3; 3]>,
    ship_tris: Vec<[Vec3; 3]>,
}

/// A hull ready to spawn. Everything is in the shield's local space: model units, centred on the
/// ship's bounding box.
pub struct HullAsset {
    pub model: RenderModel,
    pub shell_mesh: MeshHandle,
    pub center: Vec3,
    pub size: Vec3,
    pub len: f32,
    pub radii: Vec3,
    pub shell_tris: Vec<[Vec3; 3]>,
    pub shell_pts: Vec<(Vec3, Vec3)>,
    pub shell_z: (f32, f32),
    pub ship_tris: Vec<[Vec3; 3]>,
}

/// A model and, when it was baked, its envelope.
type FetchResult = Result<(Vec<u8>, Option<Vec<u8>>), String>;
/// Sound files read from the assets folder: (name, MP3 bytes).
type SoundFiles = Vec<(&'static str, Vec<u8>)>;

pub enum Slot {
    Idle,
    Fetch(Task<FetchResult>),
    Build(Task<Result<Box<HullContent>, String>>),
    Ready(Arc<HullAsset>),
    Failed(String),
}

/// The engine's small test ship: what a hull falls back to when its model can't be read or built.
const FALLBACK_GLB: &[u8] = include_bytes!("../../../crates/space3d-stories/assets/ship.glb");

/// The game sounds looked up in `<assets>/sfx/` (optional: without them the demo is silent).
const SFX_NAMES: [&str; 31] = [
    "dodge",
    "explosion_small_1",
    "explosion_small_2",
    "explosion_small_3",
    "hit_hull_1",
    "hit_hull_2",
    "hit_hull_3",
    "hit_shield_1",
    "hit_shield_2",
    "hit_shield_3",
    "immune_hit",
    "laser_siphon_1",
    "laser_siphon_2",
    "laser_siphon_3",
    "laser_x1_1",
    "laser_x1_2",
    "laser_x1_3",
    "laser_x2_1",
    "laser_x2_2",
    "laser_x2_3",
    "laser_x3_1",
    "laser_x3_2",
    "laser_x3_3",
    "laser_x4_1",
    "laser_x4_2",
    "laser_x4_3",
    "rocket_launch_1",
    "rocket_launch_2",
    "rocket_launch_3",
    "shield_buff_off",
    "shield_buff_on",
];

/// The models committed with the example (`examples/shield_demo/assets`).
#[cfg(not(target_arch = "wasm32"))]
const COMMITTED_ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

/// Where the SpaceCorps models (`<name>.glb`, `<name>.sbe`) and sounds (`sfx/*.mp3`) live:
/// `SPACE3D_SHIELD_DEMO_ASSETS` (a URL parameter on the web), else an `assets/` folder next to
/// the executable, else the example's own `assets/` folder (`../assets` next to the page on the
/// web).
fn asset_root() -> String {
    if let Some(dir) = config("SPACE3D_SHIELD_DEMO_ASSETS") {
        return dir;
    }
    #[cfg(target_arch = "wasm32")]
    {
        // the page is in web/, the shared assets next to it
        "../assets".to_string()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|p| p.join("assets")))
            .filter(|dir| dir.is_dir())
            .unwrap_or_else(|| COMMITTED_ASSETS.into())
            .to_string_lossy()
            .into_owned()
    }
}

/// Ellipsoid around the hull (the prototype's fit): half extents (at least 35 % of the largest),
/// grown until every vertex is inside.
fn fit_radii(points: &[[f32; 3]], center: Vec3, half: Vec3) -> Vec3 {
    let h = half.max(Vec3::splat(0.35 * half.max_element()));
    let mut s_max = 0.0f32;
    for p in points {
        s_max = s_max.max(((Vec3::from_array(*p) - center) / h).length_squared());
    }
    h * if s_max > 0.0 { s_max.sqrt() } else { 1.0 }
}

/// The CPU part of a hull. `sbe_bytes`: its baked envelope; `None` computes one here (48 cells,
/// a few milliseconds for the test ship).
fn build_hull(name: &str, glb: Vec<u8>, sbe_bytes: Option<Vec<u8>>) -> Result<HullContent, String> {
    let ctx = LoadContext::new(format!("{name}.glb"));
    let model = space3d::assets::loaders::load_gltf(&glb, &ctx, true).map_err(|e| e.to_string())?;
    let merged = model.merged_mesh();
    if merged.positions.is_empty() {
        return Err(format!("{name}: empty model"));
    }
    let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in &merged.positions {
        let p = Vec3::from_array(*p);
        min = min.min(p);
        max = max.max(p);
    }
    let center = (min + max) * 0.5;
    let size = max - min;
    let radii = fit_radii(&merged.positions, center, size * 0.5);
    let computed = || {
        let e = envelope::envelope(
            &merged.positions,
            &merged.indices,
            envelope::Settings {
                offset: 0.035,
                cells: 48,
            },
        );
        sbe::Sbe {
            positions: e.positions,
            normals: e.normals,
            indices: e.indices,
        }
    };
    let shell = match sbe_bytes.map(|b| sbe::decode(&b)) {
        Some(Ok(shell)) => shell,
        Some(Err(e)) => {
            space3d::tracing::warn!("{name}: {e}; envelope computed instead");
            computed()
        }
        None => computed(),
    };
    if shell.indices.is_empty() {
        return Err(format!("{name}: empty envelope"));
    }
    let local: Vec<Vec3> = shell.positions.iter().map(|p| Vec3::from_array(*p) - center).collect();
    let tris = |pts: &[Vec3], idx: &[u32]| -> Vec<[Vec3; 3]> {
        idx.as_chunks::<3>()
            .0
            .iter()
            .map(|t| [pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]])
            .collect()
    };
    let shell_tris = tris(&local, &shell.indices);
    let ship_pts: Vec<Vec3> = merged.positions.iter().map(|p| Vec3::from_array(*p) - center).collect();
    let ship_tris = tris(&ship_pts, &merged.indices);
    Ok(HullContent {
        model: prepare_model(&model, &ModelUploadOptions::default()),
        center,
        size,
        radii,
        shell: MeshData {
            positions: local.iter().map(|p| p.to_array()).collect(),
            normals: shell.normals,
            uvs: vec![[0.0, 0.0]; local.len()],
            indices: shell.indices,
            ..MeshData::default()
        },
        shell_tris,
        ship_tris,
    })
}

// ---------------------------------------------------------------- simulation state

/// A wave on a shield, sent to the shader as is. `kind`: 0 a shot, 1 a siphon drain, 2 a blast,
/// 3 your siphon's gain (`leak` is then the number of packets and `w` the time between them).
pub struct Impact {
    pub(crate) p: Vec3,
    t: f32,
    s: f32,
    leak: f32,
    pub(crate) kind: f32,
    w: f32,
    vid: u64,
}

pub struct Unit {
    pub role: Role,
    pub hull: usize,
    pub root: Entity,
    pub asset: Arc<HullAsset>,
    pub pos: Vec3,
    pub s: f32,
    pub side: f32,
    pub model: Mat4,
    pub impacts: Vec<Impact>,
}

impl Unit {
    pub fn bound(&self) -> f32 {
        self.asset.radii.max_element() * self.s * MARGIN
    }
    fn flat_radius(&self) -> f32 {
        self.asset.radii.x.max(self.asset.radii.z) * self.s * MARGIN
    }
}

#[derive(Default, Clone, Copy)]
struct Cast {
    shield: Option<(f32, Vec3)>,
    hull: Option<Option<f32>>,
}

struct Bolt {
    p: Vec3,
    o: Vec3,
    v: Vec3,
    speed: f32,
    life: f32,
    unit: usize,
    weapon: usize,
    miss: bool,
    centre: Vec3,
    best: f32,
    shown: bool,
    vid: u64,
    n: usize,
    quiet: bool,
    t0: f32,
    started: bool,
    passed: bool,
    leak: f32,
    rem: Option<f32>,
    aegis_done: bool,
    pre_aegis: bool,
    /// Fired by your ship (your Siphon at a neighbour), not by a shooter out of view.
    from_you: bool,
    travel: f32,
    cast: Option<Cast>,
    color: Vec3,
    half_width: f32,
    length: f32,
    trail: Option<Entity>,
}

struct Packet {
    loc: Vec3,
    unit: usize,
    from: Option<Vec3>,
    to: Vec3,
    /// Your own Siphon: where the packet enters your shield (it follows your ship).
    home: Option<Home>,
    /// The first packet of your drain carries the gain and the number of packets.
    gain: Option<usize>,
    t0: f32,
    dur: f32,
}

/// A point on your shield, in its own space: the hull shield, or the Aegis bubble while it is up.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Home {
    Hull(Vec3),
    Aegis(Vec3),
}

struct BreakFx {
    pt: Vec3,
    speed: f32,
    width: f32,
    dur: f32,
}

pub struct Aegis {
    pub state: AegisState,
    t0: f32,
    t_deploy: f32,
    pct_before: f32,
    pub reserve: f32,
    pub full: f32,
    pub absorb: f32,
    pub radii: Vec3,
    pub(crate) impacts: Vec<Impact>,
    cd_end: f32,
    fatigue_end: f32,
    pub message: String,
    message_t: f32,
    brk: Option<BreakFx>,
    worn: VecDeque<u64>,
}

pub struct FloatText {
    pub text: &'static str,
    pub pos: Vec3,
    pub t: f32,
}

pub struct Game {
    pub s: Settings,
    slots: Vec<Slot>,
    source: Option<Arc<dyn AssetSource>>,
    scene_key: Option<(usize, usize, usize, bool)>,
    wanted: Option<(usize, usize, usize, bool)>,
    pub units: Vec<Unit>,
    pub world_len: f32,
    pub sim: f32,
    pub state: State,
    state_t: f32,
    last_hit: f32,
    brk: Option<BreakFx>,
    pub aegis: Aegis,
    bolts: Vec<Bolt>,
    packets: Vec<Packet>,
    worn: VecDeque<u64>,
    volley_seq: u64,
    next_shot: f32,
    base_angle: f32,
    spin: f32,
    rng: u32,
    pub hits: u64,
    pub floats: Vec<FloatText>,
    // particles
    next_particle: u32,
    particle_count: u32,
    write_seq: u64,
    writes: VecDeque<(u64, u32, Arc<Vec<f32>>, u32)>,
    // sound
    sounds: HashMap<String, Vec<StaticSoundData>>,
    sound_task: Option<Task<HashMap<String, Vec<StaticSoundData>>>>,
    sound_fetch: Option<Task<SoundFiles>>,
    sounds_tried: bool,
    /// The SpaceCorps models were not found: every hull is the engine's test ship.
    pub fallback: bool,
    /// A model failed to build once: the next failure is reported instead of retried.
    fallback_failed: bool,
    /// Commands from the keyboard, the mouse and the panel, applied at the next fixed tick.
    pending: Vec<Cmd>,
    /// Random numbers of the visual and sound effects, apart from those of the rules: the
    /// simulation must not depend on the quality tier or on the sounds being loaded.
    fx_rng: u32,
    /// Simulation time to draw (the last fixed step plus the frame's share of the next one).
    render_t: f32,
    sound_last: HashMap<&'static str, f32>,
    // scene objects
    sphere: Option<MeshHandle>,
    ring: Option<MeshHandle>,
    light: Option<Entity>,
    light_t: f32,
    light_shield: bool,
    light_k: f32,
    light_col: [f32; 3],
    light_pos: Vec3,
    // view
    pub yaw: f32,
    pub pitch: f32,
    pub orbit_dist: f32,
    pub ui_busy: bool,
    press: Option<Vec2>,
    pressed_on_ui: bool,
    // quality
    quality_auto: bool,
    quality_applied: Option<GraphicsQuality>,
    // readout
    pub fps: f32,
    fps_t: f64,
    fps_n: u32,
    pub(crate) break_at: Option<u64>,
    pub(crate) aegis_at: Option<u64>,
    pub(crate) fire_at: Option<u64>,
    log_at: Option<u64>,
    switch_at: Option<u64>,
    /// Frames since the ships are on screen (the test options count from there).
    pub scene_frames: u64,
    sfx_played: u64,
    audio_state: (bool, u64),
}

impl Default for Game {
    fn default() -> Self {
        let frame_env = |k: &str| config(k).and_then(|v| v.parse().ok());
        Self {
            quality_auto: config("SHIELD_QUALITY").is_none(),
            s: Settings::default(),
            slots: (0..HULLS.len()).map(|_| Slot::Idle).collect(),
            source: None,
            scene_key: None,
            wanted: None,
            units: Vec::new(),
            world_len: 100.0,
            sim: 0.0,
            state: State::Up,
            state_t: 0.0,
            last_hit: -100.0,
            brk: None,
            aegis: Aegis {
                state: AegisState::Off,
                t0: 0.0,
                t_deploy: 0.0,
                pct_before: 100.0,
                reserve: 0.0,
                full: 0.0,
                absorb: 0.0,
                radii: Vec3::ONE,
                impacts: Vec::new(),
                cd_end: -99.0,
                fatigue_end: -99.0,
                message: String::new(),
                message_t: -99.0,
                brk: None,
                worn: VecDeque::new(),
            },
            bolts: Vec::new(),
            packets: Vec::new(),
            worn: VecDeque::new(),
            volley_seq: 0,
            next_shot: 0.4,
            base_angle: 0.0,
            spin: 0.0,
            rng: 0x9e37_79b9,
            hits: 0,
            floats: Vec::new(),
            next_particle: 0,
            particle_count: 0,
            write_seq: 0,
            writes: VecDeque::new(),
            sounds: HashMap::new(),
            sound_task: None,
            sound_fetch: None,
            sounds_tried: false,
            fallback: false,
            fallback_failed: false,
            pending: Vec::new(),
            fx_rng: 0x85eb_ca6b,
            render_t: 0.0,
            sound_last: HashMap::new(),
            sphere: None,
            ring: None,
            light: None,
            light_t: -9.0,
            light_shield: true,
            light_k: 1.0,
            light_col: [1.0; 3],
            light_pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 62f32.to_radians(),
            orbit_dist: 0.0,
            ui_busy: false,
            press: None,
            pressed_on_ui: false,
            quality_applied: None,
            fps: 0.0,
            fps_t: 0.0,
            fps_n: 0,
            break_at: frame_env("SHIELD_BREAK_AT"),
            aegis_at: frame_env("SHIELD_AEGIS_AT"),
            fire_at: frame_env("SHIELD_FIRE_AT"),
            log_at: frame_env("SHIELD_LOG_FRAME"),
            switch_at: frame_env("SHIELD_SWITCH_AT"),
            scene_frames: 0,
            sfx_played: 0,
            audio_state: (false, 0),
        }
    }
}

fn norm_or(v: Vec3, d: Vec3) -> Vec3 {
    v.try_normalize().unwrap_or(d)
}

/// Ray against triangles: nearest distance and the (unnormalised) face normal.
fn ray_tris(o: Vec3, d: Vec3, tris: &[[Vec3; 3]]) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    for [a, b, c] in tris {
        let e1 = *b - *a;
        let e2 = *c - *a;
        let p = d.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-14 {
            continue;
        }
        let inv = 1.0 / det;
        let s = o - *a;
        let u = s.dot(p) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = d.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = e2.dot(q) * inv;
        if t > 1e-6 && best.is_none_or(|b| t < b.0) {
            best = Some((t, e1.cross(e2)));
        }
    }
    best
}

impl Game {
    pub fn tr(&self, fr: &'static str, en: &'static str) -> &'static str {
        if self.s.lang == Lang::Fr { fr } else { en }
    }

    /// Random numbers of the effects (particles, sounds), apart from the rules' `rnd`.
    fn fx_rnd(&mut self, a: f32, b: f32) -> f32 {
        self.fx_rng ^= self.fx_rng << 13;
        self.fx_rng ^= self.fx_rng >> 17;
        self.fx_rng ^= self.fx_rng << 5;
        a + (b - a) * (self.fx_rng as f32 / u32::MAX as f32)
    }

    fn rnd(&mut self, a: f32, b: f32) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        a + (b - a) * (self.rng as f32 / u32::MAX as f32)
    }

    pub fn weapon(&self) -> &'static Weapon {
        &WEAPONS[self.s.weapon.min(WEAPONS.len() - 1)]
    }

    pub fn pen_of(&self, w: &Weapon) -> f32 {
        match w.kind {
            WeaponKind::Blast | WeaponKind::Siphon => 0.0,
            _ => w.pen.unwrap_or(self.s.pen),
        }
    }

    pub fn share_of(&self, absorb: f32, w: &Weapon) -> f32 {
        if w.kind == WeaponKind::Siphon {
            1.0
        } else {
            ((absorb - self.pen_of(w)) / 100.0).clamp(0.0, 1.0)
        }
    }

    pub fn is_ship(&self, hull: usize) -> bool {
        HULLS[hull].kind == HullKind::Ship
    }

    pub fn color_of_hull(&self, hull: usize) -> [f32; 3] {
        self.s.mob_colors.get(&hull).copied().unwrap_or(HULLS[hull].color)
    }

    /// The hull on screen (the selection may still be loading).
    fn main_hull(&self) -> usize {
        self.units
            .iter()
            .find(|u| u.role == Role::Main)
            .map_or(self.s.main, |u| u.hull)
    }

    pub fn main_color(&self) -> [f32; 3] {
        if self.is_ship(self.main_hull()) {
            self.s.skin
        } else {
            self.color_of_hull(self.main_hull())
        }
    }

    pub fn unit_color(&self, i: usize) -> [f32; 3] {
        let u = &self.units[i];
        match u.role {
            Role::Main => self.main_color(),
            Role::Ally => PALETTES[self.s.ally_skin].2,
            Role::Foe => self.color_of_hull(u.hull),
        }
    }

    pub fn absorb_main(&self) -> f32 {
        if self.is_ship(self.main_hull()) {
            self.s.absorb
        } else {
            ALIEN_ABSORB
        }
    }

    pub fn absorb_of(&self, i: usize) -> f32 {
        match self.units[i].role {
            Role::Main => self.absorb_main(),
            Role::Ally => self.s.ally_absorb,
            Role::Foe => ALIEN_ABSORB,
        }
    }

    pub fn immune(&self, i: usize) -> bool {
        HULLS[self.units[i].hull].immune
    }

    pub fn aegis_up(&self) -> bool {
        matches!(self.aegis.state, AegisState::Deploy | AegisState::On)
    }

    pub fn ally_index(&self) -> Option<usize> {
        self.units.iter().position(|u| u.role == Role::Ally)
    }

    fn foe_index(&self) -> Option<usize> {
        self.units.iter().position(|u| u.role == Role::Foe)
    }

    pub fn guns(&self) -> &'static Guns {
        &GUNS[self.s.shooter.min(GUNS.len() - 1)]
    }

    pub fn volley_count(&self) -> usize {
        if !self.s.view_self {
            2
        } else if self.s.shots == 0 {
            observed_shots(self.guns())
        } else {
            self.s.shots
        }
    }

    fn part_n(&self, n: usize) -> usize {
        if self.s.quality == GraphicsQuality::Low {
            ((n as f32 * 0.25).round() as usize).max(1)
        } else {
            n
        }
    }

    // ------------------------------------------------------------ loading

    fn request_hull(&mut self, i: usize) {
        if matches!(self.slots[i], Slot::Idle) {
            let source = self.source.get_or_insert_with(|| default_source(&asset_root())).clone();
            let file = HULLS[i].file;
            self.slots[i] = Slot::Fetch(spawn(async move {
                let glb = source
                    .read(&AssetPath::new(&format!("{file}.glb")))
                    .await
                    .map_err(|e| e.to_string())?;
                // optional: without a baked envelope, one is computed at load time
                let sbe = source.read(&AssetPath::new(&format!("{file}.sbe"))).await.ok();
                Ok((glb, sbe))
            }));
        }
    }

    fn poll_hulls(&mut self, ctx: &mut SystemCtx) {
        for (i, hull) in HULLS.iter().enumerate() {
            match &mut self.slots[i] {
                Slot::Fetch(t) => {
                    if let Some(r) = t.poll() {
                        self.slots[i] = match r {
                            Ok((glb, sbe)) => {
                                let file = hull.file;
                                Slot::Build(spawn_compute(move || build_hull(file, glb, sbe).map(Box::new)))
                            }
                            // no SpaceCorps assets: the engine's test ship stands in for every hull
                            Err(_) => {
                                self.fallback = true;
                                Slot::Build(spawn_compute(|| {
                                    build_hull("ship", FALLBACK_GLB.to_vec(), None).map(Box::new)
                                }))
                            }
                        };
                    }
                }
                Slot::Build(t) => {
                    if let Some(r) = t.poll() {
                        self.slots[i] = match r {
                            Ok(c) => Slot::Ready(Arc::new(upload_hull(*c, ctx.resources))),
                            // an unreadable model: the test ship stands in (the test ship failing
                            // too would be a bug, reported in the panel)
                            Err(e) if !self.fallback_failed => {
                                space3d::tracing::warn!("{}: {e}; using the test ship", hull.file);
                                self.fallback = true;
                                self.fallback_failed = true;
                                Slot::Build(spawn_compute(|| {
                                    build_hull("ship", FALLBACK_GLB.to_vec(), None).map(Box::new)
                                }))
                            }
                            Err(e) => Slot::Failed(e),
                        };
                    }
                }
                _ => {}
            }
        }
    }

    /// Text for the panel while something is missing.
    pub fn loading_text(&self) -> Option<String> {
        let wanted = [self.s.main, self.s.ally, self.s.foe];
        for (k, i) in wanted.iter().enumerate() {
            if k > 0 && !self.s.multi {
                continue;
            }
            match &self.slots[*i] {
                Slot::Ready(_) => {}
                Slot::Failed(e) => return Some(format!("{} : {e}", HULLS[*i].en)),
                _ => return Some(self.tr("Chargement des vaisseaux…", "Loading ships…").to_string()),
            }
        }
        None
    }

    fn asset(&self, i: usize) -> Option<Arc<HullAsset>> {
        match &self.slots[i] {
            Slot::Ready(a) => Some(a.clone()),
            _ => None,
        }
    }

    fn sync_scene(&mut self, ctx: &mut SystemCtx) {
        let key = (self.s.main, self.s.ally, self.s.foe, self.s.multi);
        if self.wanted != Some(key) {
            self.wanted = Some(key);
            for i in [key.0, key.1, key.2] {
                if matches!(self.slots[i], Slot::Failed(_)) {
                    self.slots[i] = Slot::Idle;
                }
            }
        }
        self.request_hull(key.0);
        if key.3 {
            self.request_hull(key.1);
            self.request_hull(key.2);
        }
        self.poll_hulls(ctx);
        if self.scene_key == Some(key) {
            return;
        }
        let Some(main) = self.asset(key.0) else { return };
        let (ally, foe) = if key.3 {
            let (a, f) = (self.asset(key.1), self.asset(key.2));
            let failed = |g: &Game, i: usize| matches!(g.slots[i], Slot::Failed(_));
            if (a.is_none() && !failed(self, key.1)) || (f.is_none() && !failed(self, key.2)) {
                return;
            }
            (a, f)
        } else {
            (None, None)
        };
        // a changed scene: the Aegis gives the shield back, shots and waves are cleared
        self.aegis_cancel();
        for b in std::mem::take(&mut self.bolts) {
            if let Some(e) = b.trail {
                let _ = ctx.world.despawn(e);
            }
        }
        self.packets.clear();
        self.brk = None;
        if self.state == State::Breaking {
            self.state = State::Down;
        }
        for u in std::mem::take(&mut self.units) {
            let children: Vec<Entity> = ctx
                .world
                .query::<(Entity, &Parent)>()
                .iter()
                .filter(|(_, p)| p.0 == u.root)
                .map(|(e, _)| e)
                .collect();
            for e in children {
                let _ = ctx.world.despawn(e);
            }
            let _ = ctx.world.despawn(u.root);
        }
        let w = main.len * SCALE;
        self.world_len = w;
        let main_s = SCALE;
        let me = main.radii.x.max(main.radii.z) * main_s * MARGIN;
        let mut list = vec![(Role::Main, key.0, main, Vec3::ZERO, main_s, 0.0)];
        for (role, hull, asset, side) in [(Role::Ally, key.1, ally, -1.0f32), (Role::Foe, key.2, foe, 1.0)] {
            if let Some(a) = asset {
                let s = w / a.len;
                let ext = a.radii.x.max(a.radii.z) * s * MARGIN;
                list.push((role, hull, a, Vec3::new(side * (me + ext + w * 0.3), 0.0, 0.0), s, side));
            }
        }
        for (role, hull, asset, pos, s, side) in list {
            let root = spawn_model(ctx.world, &asset.model, Transform::IDENTITY);
            self.units.push(Unit {
                role,
                hull,
                root,
                asset,
                pos,
                s,
                side,
                model: Mat4::IDENTITY,
                impacts: Vec::new(),
            });
        }
        self.scene_key = Some(key);
        self.orbit_dist = 0.0;
        self.update_transforms(ctx);
    }

    fn update_transforms(&mut self, ctx: &mut SystemCtx) {
        let rot = Quat::from_rotation_y(self.spin);
        for u in &mut self.units {
            u.model = Mat4::from_scale_rotation_translation(Vec3::splat(u.s), rot, u.pos);
            if let Ok(mut t) = ctx.world.get::<&mut Transform>(u.root) {
                t.translation = u.pos - rot * (u.asset.center * u.s);
                t.rotation = rot;
                t.scale = Vec3::splat(u.s);
            }
        }
    }

    // ------------------------------------------------------------ effects

    fn emit(&mut self, data: Vec<f32>) {
        let total = (data.len() / PARTICLE_FLOATS) as u32;
        if total == 0 {
            return;
        }
        let mut offset = 0u32;
        let mut left = total.min(PARTICLE_CAP);
        while left > 0 {
            let room = PARTICLE_CAP - self.next_particle;
            let n = left.min(room);
            let a = offset as usize * PARTICLE_FLOATS;
            let b = (offset + n) as usize * PARTICLE_FLOATS;
            self.write_seq += 1;
            self.writes
                .push_back((self.write_seq, self.next_particle, Arc::new(data[a..b].to_vec()), 0));
            self.next_particle = (self.next_particle + n) % PARTICLE_CAP;
            self.particle_count = if self.next_particle == 0 {
                PARTICLE_CAP
            } else {
                self.particle_count.max(self.next_particle)
            };
            offset += n;
            left -= n;
        }
    }

    fn sparks(&mut self, point: Vec3, dir: Option<Vec3>, n: usize, col: [f32; 3], speed: f32, life_k: f32) {
        let n = self.part_n(n);
        let w = self.world_len;
        let mut data = Vec::with_capacity(n * PARTICLE_FLOATS);
        for _ in 0..n {
            let r = norm_or(
                Vec3::new(self.fx_rnd(-1.0, 1.0), self.fx_rnd(-1.0, 1.0), self.fx_rnd(-1.0, 1.0)),
                Vec3::Y,
            );
            let d = match dir {
                Some(d) => norm_or(-d * 0.6 + r, r),
                None => r,
            };
            let v = d * w * self.fx_rnd(0.4, 1.4) * speed;
            let birth = self.sim + self.fx_rnd(0.0, 0.03);
            let life = self.fx_rnd(0.15, 0.45) * life_k;
            let size = w * self.fx_rnd(0.004, 0.011) * if dir.is_some() { 1.0 } else { 1.4 };
            data.extend_from_slice(&[
                point.x, point.y, point.z, birth, v.x, v.y, v.z, life, col[0], col[1], col[2], size,
            ]);
        }
        self.emit(data);
    }

    fn hull_sparks(&mut self, point: Vec3, dir: Vec3, leak: f32, k: f32) {
        let n = (((30.0 + 170.0 * (leak / 0.7).min(1.0)) * k).round() as usize).max(8);
        self.sparks(point, Some(dir), n, HULL_SPARK_COL, 1.0, 1.0);
    }

    fn cone(&mut self, axis: Vec3, spread: f32) -> Vec3 {
        let a = self.fx_rnd(0.0, TAU);
        let c = (spread * self.fx_rnd(0.0, 1.0).sqrt()).cos();
        let s = (1.0 - c * c).max(0.0).sqrt();
        let up = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let b1 = norm_or(axis.cross(up), Vec3::X);
        let b2 = axis.cross(b1);
        axis * c + b1 * (s * a.cos()) + b2 * (s * a.sin())
    }

    /// Shield fragments where a shot pierces: the torn rim of the hole, and a splash back toward
    /// the shooter. The envelope hugs the hull: nothing goes inward.
    fn piercing(&mut self, point: Vec3, dir: Vec3, normal: Option<Vec3>, leak: f32, col: [f32; 3], k: f32) {
        if !self.s.piercing_fx {
            return;
        }
        let l = self.world_len;
        let count = (((24.0 + 96.0 * leak.min(1.0)) * k).round() as usize).max(8);
        let n_part = self.part_n(count);
        let d = norm_or(dir, Vec3::Z);
        let mut n = normal.and_then(|v| v.try_normalize()).unwrap_or(-d);
        if n.dot(d) > 0.0 {
            n = -n;
        }
        let up = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let t1 = norm_or(n.cross(up), Vec3::X);
        let t2 = n.cross(t1);
        let r0 = l * 0.08 * leak.min(1.0).sqrt() * 0.7;
        let refl = d - n * (2.0 * d.dot(n));
        let intensity = self.s.intensity;
        let mut data = Vec::with_capacity(n_part * PARTICLE_FLOATS);
        for i in 0..n_part {
            let (p, v, life, size);
            if (i as f32) < n_part as f32 * 0.55 {
                let a = self.fx_rnd(0.0, TAU);
                let radial = t1 * a.cos() + t2 * a.sin();
                p = point + radial * (r0 * self.fx_rnd(0.6, 1.0)) + n * (l * 0.004);
                v = radial * (l * self.fx_rnd(0.3, 0.8)) + n * (l * self.fx_rnd(0.05, 0.25));
                life = self.fx_rnd(0.3, 0.6);
                size = l * self.fx_rnd(0.006, 0.012);
            } else {
                p = point + n * (l * 0.004);
                let mut c = self.cone(norm_or(refl + n, n), 0.8);
                if c.dot(n) < 0.1 {
                    c = norm_or(c + n * (0.1 - c.dot(n)), n);
                }
                v = c * (l * self.fx_rnd(0.3, 0.8));
                life = self.fx_rnd(0.2, 0.45);
                size = l * self.fx_rnd(0.004, 0.009);
            }
            let birth = self.sim + self.fx_rnd(0.0, 0.02);
            data.extend_from_slice(&[
                p.x,
                p.y,
                p.z,
                birth,
                v.x,
                v.y,
                v.z,
                life,
                col[0] * intensity,
                col[1] * intensity,
                col[2] * intensity,
                size,
            ]);
        }
        self.emit(data);
    }

    /// The surface scatters as the dissolve front passes.
    fn surface_debris(&mut self, pts: &[(Vec3, Vec3)], model: Mat4, pt: Vec3, speed: f32, bubble: Option<(Vec3, f32)>) {
        if pts.is_empty() {
            return;
        }
        let n = self.part_n(pts.len().min(2600));
        let w = self.world_len;
        let col = self.main_color();
        let intensity = self.s.intensity;
        let normal_m = model.inverse().transpose();
        let mut data = Vec::with_capacity(n * PARTICLE_FLOATS);
        for _ in 0..n {
            let j = (self.fx_rnd(0.0, pts.len() as f32) as usize).min(pts.len() - 1);
            let (p, nl) = pts[j];
            let d = match bubble {
                Some((r, mean)) => {
                    let (a, b) = (norm_or(p / r, Vec3::Z), norm_or(pt / r, Vec3::Z));
                    a.dot(b).clamp(-1.0, 1.0).acos() * mean
                }
                None => p.distance(pt),
            };
            let birth = self.sim + d / speed.max(1e-4) + self.fx_rnd(-0.02, 0.04);
            let life = self.fx_rnd(0.7, 1.5);
            let size = w * self.fx_rnd(0.006, 0.016);
            let wp = model.transform_point3(p);
            let nw = norm_or(normal_m.transform_vector3(nl), Vec3::Y);
            let away = norm_or(model.transform_vector3(p - pt), nw);
            let rv = norm_or(
                Vec3::new(self.fx_rnd(-1.0, 1.0), self.fx_rnd(-1.0, 1.0), self.fx_rnd(-1.0, 1.0)),
                Vec3::Y,
            );
            let v = norm_or(nw * self.fx_rnd(0.35, 0.9) + rv * 0.35 + away * 0.35, nw) * (w * self.fx_rnd(0.5, 1.3));
            data.extend_from_slice(&[
                wp.x,
                wp.y,
                wp.z,
                birth,
                v.x,
                v.y,
                v.z,
                life,
                col[0] * intensity,
                col[1] * intensity,
                col[2] * intensity,
                size,
            ]);
        }
        self.emit(data);
    }

    fn hit_light(&mut self, point: Vec3, shield: bool, col: [f32; 3], k: f32) {
        self.light_pos = point;
        self.light_t = self.sim;
        self.light_shield = shield;
        self.light_col = col;
        self.light_k = k;
    }

    fn float_text(&mut self, text: &'static str, unit: usize) {
        let u = &self.units[unit];
        let pos = u.pos + Vec3::Y * u.bound() * 0.6;
        let t = self.sim;
        self.floats.push(FloatText { text, pos, t });
    }

    fn sfx(&mut self, ctx: &mut SystemCtx, name: &'static str, pos: Vec3, volume: f32) {
        if !self.s.sound || self.s.paused {
            return;
        }
        let now = self.sim;
        if self.sound_last.get(name).is_some_and(|t| (now - *t).abs() < 0.03) {
            return;
        }
        let count = self.sounds.get(name).map_or(0, Vec::len);
        if count == 0 {
            return;
        }
        self.sound_last.insert(name, now);
        let pick = (self.fx_rnd(0.0, count as f32) as usize).min(count - 1);
        let pan = (pos.x / (self.world_len * 4.0)).clamp(-0.6, 0.6);
        let params = PlayParams::sfx().volume(volume * self.s.volume).pan(pan);
        if let (Some(audio), Some(sound)) = (
            ctx.resources.get_mut::<Audio>(),
            self.sounds.get(name).and_then(|v| v.get(pick)),
        ) {
            audio.play(sound, params);
            self.sfx_played += 1;
        }
    }

    // ------------------------------------------------------------ shots

    pub fn target_unit(&self) -> usize {
        match self.s.target {
            Target::Player => self.ally_index().unwrap_or(0),
            Target::Mob => self.foe_index().unwrap_or(0),
            _ => 0,
        }
    }

    fn target_ok(&self) -> bool {
        match self.s.target {
            Target::You => true,
            Target::Player => self.ally_index().is_some(),
            Target::Mob => self.foe_index().is_some(),
            Target::All => false,
        }
    }

    /// One volley at `unit`, from `angle` (world XZ) or from a side that doesn't cross the
    /// neighbours. Lasers leave from the guns of the shooter's ship, one after the other.
    pub fn fire(&mut self, angle: Option<f32>, unit: Option<usize>) {
        if self.units.is_empty() {
            return;
        }
        let unit = unit.unwrap_or_else(|| self.target_unit()).min(self.units.len() - 1);
        let w = self.world_len;
        let side = self.units[unit].side;
        let many = self.units.len() > 1;
        let a = match angle {
            Some(a) => a,
            None if unit != 0 => (if side < 0.0 { PI } else { 0.0 }) + self.rnd(-1.1, 1.1),
            None if many => (if self.rnd(0.0, 1.0) < 0.5 { 1.0 } else { -1.0 }) * PI / 2.0 + self.rnd(-0.8, 0.8),
            None => {
                self.base_angle += self.rnd(-0.8, 0.8);
                self.base_angle
            }
        };
        let aegis_on = self.aegis.state != AegisState::Off;
        let r = (w * 3.2).max(if aegis_on { self.aegis.radii.x * 2.0 } else { 0.0 });
        let mut dir = Vec3::new(a.cos(), 0.0, a.sin());
        let c0 = self.units[unit].pos;
        let mut origin = dir * r + c0;
        origin.y = self.rnd(-0.04, 0.08) * w;
        let weapon = self.s.weapon.min(WEAPONS.len() - 1);
        let wk = WEAPONS[weapon].kind;
        // your Siphon at a neighbour leaves from your ship, so the shield it takes comes back to
        // a shield you can see; every other shot comes from a shooter out of view
        let from_you = wk == WeaponKind::Siphon && self.s.view_self && unit != 0;
        if from_you {
            origin = self.units[0].pos;
            dir = norm_or(Vec3::new(origin.x - c0.x, 0.0, origin.z - c0.z), Vec3::X);
        } else if aegis_on {
            // the shooter is always outside, far from the wall
            for _ in 0..20 {
                if Vec2::new(origin.x, origin.z).length() >= self.aegis.radii.x * 1.8 {
                    break;
                }
                origin += dir * (self.aegis.radii.x * 0.2);
            }
        }
        let size = self.units[unit].asset.size * self.units[unit].s;
        let mut target = c0
            + Vec3::new(
                self.rnd(-0.16, 0.16) * size.x,
                self.rnd(-0.08, 0.08) * size.y,
                self.rnd(-0.16, 0.16) * size.z,
            );
        // a laser volley hits 80 % of the time at Laser tech level 1
        let miss = self.s.miss && matches!(wk, WeaponKind::Laser | WeaponKind::Siphon) && self.rnd(0.0, 1.0) < 0.2;
        if miss {
            let big = self.units[unit].flat_radius();
            let sign = if self.rnd(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 };
            target += Vec3::new(-dir.z, 0.0, dir.x) * (sign * big * self.rnd(1.25, 1.45));
        }
        let (radius, length, speed, base): (f32, f32, f32, [f32; 3]) = match wk {
            WeaponKind::Rocket | WeaponKind::Blast => (0.016, 0.07, 4.0, [3.4, 1.6, 0.5]),
            WeaponKind::Siphon => (0.004, 0.16, 7.0, [0.08, 0.9, 0.8]),
            WeaponKind::Laser => (0.008, 0.12, 7.0, [3.2, 0.35, 0.12]),
        };
        let n = if wk == WeaponKind::Laser {
            self.volley_count()
        } else {
            1
        };
        self.volley_seq += 1;
        let vid = self.volley_seq;
        let g = self.guns();
        let sc = w / g.l * 100.0 / 110.0;
        let fwd = norm_or(Vec3::new(target.x - origin.x, 0.0, target.z - origin.z), -dir);
        let right = norm_or(Vec3::Y.cross(fwd), Vec3::X);
        let guns: Vec<Option<([f32; 3], f32)>> = match n {
            2 => vec![Some((g.a, 1.0)), Some((g.a, -1.0))],
            3 => vec![Some((g.a, 1.0)), Some((g.a, -1.0)), Some((g.c, 0.0))],
            4 => vec![Some((g.a, 1.0)), Some((g.a, -1.0)), Some((g.b, 1.0)), Some((g.b, -1.0))],
            _ => vec![None],
        };
        // spread over the volley period (0.6–1.6 s), squeezed into 0.9 s minus the flight time
        let period = (1.0 / self.s.rate.max(0.01)).clamp(0.6, 1.6);
        let gap = if n > 1 {
            (period / n as f32).min((0.9f32 - 0.16) / (n as f32 - 1.0))
        } else {
            0.0
        };
        let dim = if wk == WeaponKind::Laser && n > 1 {
            1.0 / (n as f32).sqrt()
        } else {
            1.0
        };
        let centre = c0;
        for (k, gun) in guns.iter().enumerate() {
            let mut o = origin;
            if let Some((p, s)) = gun {
                o += right * (p[0] * s * sc) + Vec3::Y * (p[1] * sc) + fwd * (p[2] * sc);
            }
            let mut t = target;
            if n > 1 {
                t += Vec3::new(self.rnd(-1.0, 1.0), self.rnd(-0.5, 0.5), self.rnd(-1.0, 1.0)) * (w * 0.04);
            }
            let v = norm_or(t - o, fwd);
            self.bolts.push(Bolt {
                p: o,
                o,
                v,
                speed: w * speed,
                life: 3.0,
                unit,
                weapon,
                miss,
                centre,
                best: f32::MAX,
                shown: false,
                vid,
                n,
                quiet: k > 0,
                t0: self.sim + gap * k as f32,
                started: false,
                passed: false,
                leak: 0.0,
                rem: None,
                // fired from inside the Aegis: the bubble lets it out
                aegis_done: from_you,
                pre_aegis: false,
                from_you,
                travel: 0.0,
                cast: None,
                color: Vec3::from_array(base) * dim,
                half_width: w * radius * 2.0,
                length: w * length,
                trail: None,
            });
        }
    }

    fn cast_shield(&self, b: &Bolt) -> Option<(f32, Vec3)> {
        let u = &self.units[b.unit];
        let inv = u.model.inverse();
        let (o, d) = (inv.transform_point3(b.p), inv.transform_vector3(b.v));
        ray_tris(o, d, &u.asset.shell_tris).map(|(t, n)| (b.travel + t, norm_or(u.model.transform_vector3(n), -b.v)))
    }

    fn cast_ship(&self, b: &Bolt) -> Option<f32> {
        let u = &self.units[b.unit];
        let inv = u.model.inverse();
        let (o, d) = (inv.transform_point3(b.p), inv.transform_vector3(b.v));
        ray_tris(o, d, &u.asset.ship_tris).map(|(t, _)| b.travel + t)
    }

    fn shield_active(&self, b: &Bolt) -> bool {
        if b.passed {
            return false;
        }
        if b.unit == 0 {
            !(self.aegis_up() || b.aegis_done || matches!(self.state, State::Down | State::Breaking))
        } else {
            true
        }
    }

    fn step_bolts(&mut self, ctx: &mut SystemCtx, dt: f32) {
        let mut bolts = std::mem::take(&mut self.bolts);
        let mut keep = Vec::with_capacity(bolts.len());
        for mut b in bolts.drain(..) {
            if self.sim < b.t0 {
                keep.push(b);
                continue;
            }
            if !b.started {
                if dt <= 0.0 {
                    keep.push(b);
                    continue;
                }
                b.started = true;
                self.launch(ctx, &mut b);
            }
            if self.step_bolt(ctx, &mut b, dt) {
                keep.push(b);
            } else if let Some(e) = b.trail.take() {
                let _ = ctx.world.despawn(e);
            }
        }
        // shots fired while stepping (none today) stay queued
        keep.append(&mut self.bolts);
        self.bolts = keep;
    }

    fn launch(&mut self, ctx: &mut SystemCtx, b: &mut Bolt) {
        let w = &WEAPONS[b.weapon];
        let volume = match w.kind {
            WeaponKind::Siphon => 0.4,
            WeaponKind::Laser => {
                if self.s.view_self {
                    0.4
                } else {
                    0.3
                }
            }
            _ => 0.45,
        };
        self.sfx(ctx, w.sound, b.o, volume);
        if matches!(w.kind, WeaponKind::Rocket | WeaponKind::Blast) {
            let e = ctx.world.spawn((Transform::from_xyz(b.p.x, b.p.y, b.p.z),));
            ctx.spawn_effect(builtin::MISSILE_TRAIL, EffectSpawn::at(Vec3::ZERO).follow(e));
            b.trail = Some(e);
        }
    }

    /// Advance one bolt; false when it is finished.
    fn step_bolt(&mut self, ctx: &mut SystemCtx, b: &mut Bolt, dt: f32) -> bool {
        let step = b.speed * dt;
        let p1 = b.p + b.v * step;
        let unit = b.unit;
        if b.miss {
            if self.aegis_up()
                && !b.aegis_done
                && let Some(mp) = self.aegis_hit(b, b.p, p1)
            {
                if !b.quiet {
                    self.sfx(ctx, "dodge", mp, 0.5);
                    self.float_text("Miss", unit);
                }
                return false;
            }
            let d = p1.distance(b.centre);
            if !b.shown && d > b.best {
                b.shown = true;
                b.life = b.life.min(0.1);
                if !b.quiet {
                    self.sfx(ctx, "dodge", b.p, 0.5);
                    self.float_text("Miss", unit);
                }
            }
            b.best = b.best.min(d);
            b.p = p1;
            b.travel += step;
            b.life -= dt;
            return b.life > 0.0;
        }
        if let Some(ap) = self.aegis_hit(b, b.p, p1) {
            match self.aegis_on_hit(ctx, b, ap) {
                AegisResult::Stop => return false,
                AegisResult::Relay => {
                    b.p = ap;
                    b.travel = 0.0;
                    b.v = norm_or(self.units[unit].pos - ap, b.v);
                    let shield = self.cast_shield(b).filter(|_| self.shield_active(b));
                    match shield {
                        Some((t, n)) => {
                            let point = ap + b.v * t;
                            self.on_hit(ctx, b, point, true, Some(n));
                        }
                        None => {
                            if let Some(t) = self.cast_ship(b) {
                                let point = ap + b.v * t;
                                self.on_hit(ctx, b, point, false, None);
                            }
                        }
                    }
                    return false;
                }
                AegisResult::Go => {
                    b.travel += ap.distance(b.p) + self.world_len * 0.003;
                    b.p = ap + b.v * (self.world_len * 0.003);
                    b.cast = None;
                    return true;
                }
            }
        }
        // near the target: one ray against its envelope, then against the hull when needed
        let u = &self.units[unit];
        let near = p1.distance(u.pos) < u.bound() * 1.3 + step;
        if near && b.cast.is_none() {
            b.cast = Some(Cast {
                shield: self.cast_shield(b),
                hull: None,
            });
        }
        let end = b.travel + step;
        if let Some(mut cast) = b.cast {
            let active = self.shield_active(b);
            if active
                && let Some((t, n)) = cast.shield
                && t >= b.travel - 1e-3
                && t <= end
            {
                let point = b.o_at(t);
                let go_on = self.on_hit(ctx, b, point, true, Some(n));
                if !go_on {
                    return false;
                }
                let skip = self.world_len * 0.003;
                b.p = point + b.v * skip;
                b.travel = t + skip;
                b.cast = Some(Cast {
                    shield: None,
                    hull: None,
                });
                self.move_trail(ctx, b);
                return true;
            }
            if !active {
                if cast.hull.is_none() {
                    cast.hull = Some(self.cast_ship(b));
                    b.cast = Some(cast);
                }
                if let Some(Some(t)) = cast.hull
                    && t <= end
                {
                    let point = b.o_at(t);
                    self.on_hit(ctx, b, point, false, None);
                    return false;
                }
            }
        }
        b.p = p1;
        b.travel = end;
        b.life -= dt;
        self.move_trail(ctx, b);
        b.life > 0.0
    }

    fn move_trail(&self, ctx: &mut SystemCtx, b: &Bolt) {
        if let Some(e) = b.trail
            && let Ok(mut t) = ctx.world.get::<&mut Transform>(e)
        {
            t.translation = b.p;
        }
    }

    fn add_impact(&mut self, unit: usize, it: Impact) {
        let list = &mut self.units[unit].impacts;
        list.push(it);
        while list.len() > MAX_IMPACTS {
            list.remove(0);
        }
    }

    /// A shield impact on `unit`; the share of the hit that reaches the hull.
    #[allow(clippy::too_many_arguments)]
    fn impact(
        &mut self,
        ctx: &mut SystemCtx,
        unit: usize,
        point: Vec3,
        s: f32,
        leak: f32,
        kind: f32,
        w: f32,
        share: f32,
        vid: u64,
    ) -> f32 {
        let local = self.units[unit].model.inverse().transform_point3(point);
        let t = self.sim;
        self.add_impact(
            unit,
            Impact {
                p: local,
                t,
                s,
                leak,
                kind,
                w,
                vid,
            },
        );
        if unit != 0 {
            return leak;
        }
        // a volley is one hit: it wears the shield once
        let first = vid == 0 || !self.worn.contains(&vid);
        if vid != 0 && first {
            self.worn.push_back(vid);
            if self.worn.len() > 64 {
                self.worn.pop_front();
            }
        }
        let mut through = leak;
        if self.s.wear && self.state == State::Up && share > 0.0 && first {
            let want = 7.0 * share;
            let took = self.s.pct.min(want);
            self.s.pct = (self.s.pct - want).max(0.0);
            if kind != 1.0 {
                through = (through + share * (want - took) / want).min(1.0);
                if let Some(last) = self.units[0].impacts.last_mut() {
                    last.leak = last.leak.max(through);
                }
            }
            if self.s.pct <= 0.0 {
                for m in &mut self.units[0].impacts {
                    if m.vid == vid || m.t == t {
                        m.s = -m.s.abs();
                    }
                }
                self.sfx(ctx, "shield_buff_off", point, 0.9);
                self.break_shield(Some(local));
            }
        }
        through
    }

    /// The drained shield streams back to the shooter: to `to` (a shooter out of view), or to
    /// `home` on your shield when the drain is yours.
    fn siphon_packets(&mut self, from: Vec3, to: Vec3, unit: usize, delay: f32, home: Option<Home>) {
        let n = 3 + (self.fx_rnd(0.0, 8.0) as usize).min(7);
        let loc = self.units[unit].model.inverse().transform_point3(from);
        for i in 0..n {
            self.packets.push(Packet {
                loc,
                unit,
                from: None,
                to,
                home,
                gain: (home.is_some() && i == 0).then_some(n),
                t0: self.sim + delay + i as f32 * SIPHON_SPREAD / (n as f32 - 1.0),
                dur: SIPHON_FLIGHT,
            });
        }
    }

    /// Where your Siphon's packets enter your shield: on the line from the drained point to your
    /// ship, so on the side that faces the target. While the Aegis is up, your shield is the
    /// bubble: on the bubble, towards the target.
    fn siphon_home(&self, from: Vec3) -> Home {
        let u = &self.units[0];
        if self.aegis_up() {
            let r = self.aegis.radii;
            let l = self.aegis_matrix().inverse().transform_point3(from);
            let k = (l / r).length();
            return Home::Aegis(if k > 1e-6 { l / k } else { Vec3::new(r.x, 0.0, 0.0) });
        }
        // the shield's own space is centred on the ship: towards the origin is towards the ship
        let o = u.model.inverse().transform_point3(from);
        match ray_tris(o, -o, &u.asset.shell_tris) {
            Some((t, _)) => Home::Hull(o - o * t),
            None => {
                let r = u.asset.radii * MARGIN;
                let k = (o / r).length();
                Home::Hull(if k > 1e-6 { o / k } else { Vec3::new(r.x, 0.0, 0.0) })
            }
        }
    }

    /// A point of your shield in the world at `now`.
    fn home_world(&self, home: Home, now: f32) -> Vec3 {
        match home {
            Home::Hull(p) => self.units.first().map_or(p, |u| u.model.transform_point3(p)),
            Home::Aegis(p) => {
                if self.units.is_empty() {
                    p
                } else {
                    self.aegis_matrix_at(now).transform_point3(p)
                }
            }
        }
    }

    /// Your Siphon's first packet reaches your shield. It takes in what the drain took (x1: the
    /// 7 % a drain takes from your shield with wear), up to its maximum, and the regeneration is
    /// not delayed. Each packet then sends a teal wave from where it enters over the whole
    /// shield (one impact from `t`, its landing time: the shader draws one wave per packet).
    fn siphon_gain(&mut self, ctx: &mut SystemCtx, home: Home, n: usize, t: f32) {
        if self.units.is_empty() {
            return;
        }
        let wave = |p: Vec3| Impact {
            p,
            t,
            s: 0.9,
            leak: n as f32,
            kind: 3.0,
            w: SIPHON_SPREAD / (n.max(2) - 1) as f32,
            vid: 0,
        };
        match home {
            Home::Hull(p) => self.add_impact(0, wave(p)),
            Home::Aegis(p) => {
                if self.aegis_up() {
                    self.aegis.impacts.push(wave(p));
                    while self.aegis.impacts.len() > MAX_IMPACTS {
                        self.aegis.impacts.remove(0);
                    }
                }
            }
        }
        let point = self.home_world(home, self.sim);
        self.hit_light(point, true, SIPHON_COL, 0.6);
        if self.aegis_up() {
            self.aegis.reserve = (self.aegis.reserve + SIPHON_GAIN).min(100.0);
            self.aegis.full = self.aegis.full.max(self.aegis.reserve);
            self.s.pct = self.aegis.reserve;
        } else if self.s.wear {
            if matches!(self.state, State::Down | State::Breaking) {
                // a shield at 0 comes back with what was taken in
                self.s.pct = SIPHON_GAIN;
                self.restore_shield(false);
                self.sfx(ctx, "shield_buff_on", point, 0.8);
            } else {
                self.s.pct = (self.s.pct + SIPHON_GAIN).min(100.0);
            }
        }
    }

    /// A bolt hits: true when it goes on (a pierced shield).
    fn on_hit(&mut self, ctx: &mut SystemCtx, b: &mut Bolt, point: Vec3, shield: bool, normal: Option<Vec3>) -> bool {
        let unit = b.unit;
        let w = &WEAPONS[b.weapon];
        let absorb = self.absorb_of(unit);
        let col = self.unit_color(unit);
        self.hits += 1;
        // any damage, to the shield or the hull, delays the regeneration (as in game)
        if !self.immune(unit) && (shield || w.kind != WeaponKind::Siphon) {
            self.last_hit_update(unit);
        }
        if self.immune(unit) {
            self.sfx(ctx, "immune_hit", point, 0.7);
            self.hit_light(point, shield, col, 0.25);
            if shield {
                self.impact(ctx, unit, point, -0.5, 0.0, 0.0, 1.0, 0.0, b.vid);
            }
            if !b.quiet && HULLS[self.units[unit].hull].file == "unwakened" {
                self.float_text("Immune", unit);
            }
            return false;
        }
        let k_n = if b.n > 1 {
            (1.0 / (b.n as f32).sqrt()).max(0.35)
        } else {
            1.0
        };
        if shield {
            let (name, vol) = match w.kind {
                WeaponKind::Blast => ("explosion_small", 0.85),
                WeaponKind::Siphon => ("hit_shield", 0.55),
                _ => ("hit_shield", 0.75),
            };
            self.sfx(ctx, name, point, vol);
            if w.kind == WeaponKind::Siphon {
                let t = 0.5 * self.s.wave_life;
                self.hit_light(point, true, SIPHON_COL, 1.0);
                self.impact(ctx, unit, point, 0.9, 0.0, 1.0, t, 1.0, b.vid);
                if unit != 0 || self.state != State::Breaking {
                    let home = b.from_you.then(|| self.siphon_home(point));
                    self.siphon_packets(point, b.o, unit, t, home);
                }
                return false;
            }
            let share = self.share_of(absorb, w);
            let leak = 1.0 - share;
            let rem = b.rem.unwrap_or(1.0);
            self.hit_light(point, true, col, if w.kind == WeaponKind::Blast { 2.0 } else { 1.0 });
            if w.kind == WeaponKind::Blast {
                let through = self.impact(
                    ctx,
                    unit,
                    point,
                    (1.5 * rem).max(0.3),
                    leak * rem,
                    2.0,
                    3.0,
                    share * rem,
                    b.vid,
                );
                self.sparks(point, None, 160, BLAST_COL, 1.6, 1.4);
                ctx.spawn_effect(builtin::EXPLOSION, EffectSpawn::at(point).scale(0.35));
                if through > 0.02 {
                    let dir = norm_or(self.units[unit].pos - point, b.v);
                    let mut probe = Bolt {
                        p: point,
                        v: dir,
                        travel: 0.0,
                        ..b.clone_shallow()
                    };
                    probe.unit = unit;
                    if let Some(t) = self.cast_ship(&probe) {
                        self.hull_sparks(point + dir * t, dir, through, 1.0);
                    }
                }
                return false;
            }
            let rocket = w.kind == WeaponKind::Rocket;
            let strength = ((0.2 + 0.8 * share) * if rocket { 1.4 } else { 1.0 } * rem).max(0.15);
            let through = self.impact(
                ctx,
                unit,
                point,
                strength,
                leak * rem,
                0.0,
                if rocket { 1.3 } else { 1.0 },
                share * rem,
                b.vid,
            );
            if through > 0.02 {
                if unit != 0 || self.state != State::Breaking {
                    self.piercing(point, b.v, normal, through, col, k_n);
                }
                b.passed = true;
                b.leak = through;
                let k = through.sqrt();
                b.color *= 0.3 + 0.7 * k;
                b.half_width *= 0.45 + 0.55 * k;
                return true;
            }
            return false;
        }
        // the hull: shield down, or the share that went through
        if w.kind == WeaponKind::Siphon {
            self.hit_light(point, false, col, 0.25);
            return false;
        }
        let blast = w.kind == WeaponKind::Blast;
        let vol = if b.passed { 0.5 + 0.4 * b.leak.min(1.0) } else { 0.8 };
        self.sfx(
            ctx,
            if blast && !b.passed {
                "explosion_small"
            } else {
                "hit_hull"
            },
            point,
            vol,
        );
        self.hit_light(point, false, col, 1.0);
        if blast {
            self.sparks(point, None, 160, BLAST_COL, 1.6, 1.4);
            ctx.spawn_effect(builtin::EXPLOSION, EffectSpawn::at(point).scale(0.35));
        }
        let amount = if b.passed { b.leak } else { b.rem.unwrap_or(1.0) };
        self.hull_sparks(point, b.v, amount, k_n);
        false
    }

    fn last_hit_update(&mut self, unit: usize) {
        if unit == 0 {
            self.last_hit = self.sim;
        }
    }

    // ------------------------------------------------------------ your shield

    pub fn break_shield(&mut self, pt: Option<Vec3>) {
        if matches!(self.state, State::Down | State::Breaking) || self.units.is_empty() {
            return;
        }
        let asset = self.units[0].asset.clone();
        let model = self.units[0].model;
        let pt = pt.unwrap_or_else(|| {
            // no shot: from the side the camera sees
            let (yaw, pitch) = if self.s.orbit {
                (self.yaw, self.pitch)
            } else {
                (0.0, 62f32.to_radians())
            };
            let eye =
                Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos()) * self.orbit_dist.max(1.0);
            let dir = norm_or(model.inverse().transform_point3(eye), Vec3::Z);
            let t = ray_tris(dir * asset.len * 3.0, -dir, &asset.shell_tris).map_or(asset.len * 2.5, |h| h.0);
            let at = dir * (asset.len * 3.0 - t);
            let now = self.sim;
            self.add_impact(
                0,
                Impact {
                    p: at,
                    t: now,
                    s: -1.4,
                    leak: 0.0,
                    kind: 0.0,
                    w: 1.0,
                    vid: 0,
                },
            );
            at
        });
        self.state = State::Breaking;
        self.state_t = self.sim;
        self.s.pct = 0.0;
        let max_d = asset
            .shell_pts
            .iter()
            .step_by(7)
            .map(|p| p.0.distance(pt))
            .fold(0.0, f32::max);
        let speed = (self.s.wave_speed * asset.len).max(max_d / BREAK_MAX);
        self.brk = Some(BreakFx {
            pt,
            speed,
            width: asset.len * 0.05,
            dur: max_d / speed,
        });
        self.surface_debris(&asset.shell_pts, model, pt, speed, None);
    }

    pub fn restore_shield(&mut self, full: bool) {
        if full {
            self.s.pct = 100.0;
        } else if self.s.pct <= 0.0 {
            self.s.pct = 2.0;
        }
        if matches!(self.state, State::Down | State::Breaking) {
            self.state = State::Restoring;
            self.state_t = self.sim;
            self.brk = None;
        }
    }

    pub fn break_button(&mut self, ctx: &mut SystemCtx) {
        if self.aegis_up() {
            let p = Vec3::new(self.aegis.radii.x, 0.0, 0.0);
            self.aegis_break(ctx, p);
        } else {
            if !matches!(self.state, State::Down | State::Breaking) && !self.units.is_empty() {
                let p = self.units[0].pos;
                self.sfx(ctx, "shield_buff_off", p, 0.9);
            }
            self.break_shield(None);
        }
    }

    pub fn restore_button(&mut self, ctx: &mut SystemCtx) {
        if self.aegis_up() {
            self.aegis.reserve = 100.0;
            self.aegis.full = self.aegis.full.max(100.0);
            self.aegis.pct_before = 100.0;
            self.s.pct = 100.0;
        } else {
            if matches!(self.state, State::Down | State::Breaking) && !self.units.is_empty() {
                let p = self.units[0].pos;
                self.sfx(ctx, "shield_buff_on", p, 0.8);
            }
            self.restore_shield(true);
        }
    }

    pub fn pct_changed(&mut self, ctx: &mut SystemCtx) {
        if self.aegis_up() {
            self.aegis.reserve = self.s.pct;
            self.aegis.full = self.aegis.full.max(self.s.pct);
            self.aegis.pct_before = self.s.pct;
            if self.s.pct <= 0.0 {
                let p = Vec3::new(self.aegis.radii.x, 0.0, 0.0);
                self.aegis_break(ctx, p);
            }
        } else if self.s.pct <= 0.0 {
            self.break_button(ctx);
        } else if matches!(self.state, State::Down | State::Breaking) {
            self.restore_shield(false);
            if let Some(p) = self.units.first().map(|u| u.pos) {
                self.sfx(ctx, "shield_buff_on", p, 0.8);
            }
        }
    }

    // ------------------------------------------------------------ Aegis

    fn aegis_fatigue(&self) -> f32 {
        if self.s.aegis_cooldown == 5.0 {
            3.0
        } else {
            AEGIS_FATIGUE
        }
    }

    fn aegis_give_back(&mut self) {
        self.s.pct = if self.s.wear {
            self.aegis.reserve.min(100.0)
        } else {
            self.aegis.pct_before
        };
        self.aegis.reserve = 0.0;
        if self.s.pct > 0.0 {
            self.state = State::Restoring;
            self.state_t = self.sim;
        } else {
            self.state = State::Down;
            self.state_t = self.sim;
            self.last_hit = self.sim;
        }
    }

    fn aegis_reset(&mut self) {
        self.aegis.state = AegisState::Off;
        self.aegis.impacts.clear();
        self.aegis.brk = None;
    }

    fn aegis_cancel(&mut self) {
        if self.aegis_up() {
            self.aegis_give_back();
        }
        self.aegis_reset();
    }

    /// Why the Aegis can't start now, if so.
    pub fn aegis_refusal(&self) -> Option<String> {
        let t = |fr: &str, en: &str| {
            Some(if self.s.lang == Lang::Fr {
                fr.to_string()
            } else {
                en.to_string()
            })
        };
        match self.aegis.state {
            AegisState::Deploy | AegisState::On => return t("L'Égide est déjà déployée.", "The Aegis is already up."),
            AegisState::Retract => return t("L'Égide se termine.", "The Aegis is ending."),
            AegisState::Break => return t("L'Égide vient d'éclater.", "The Aegis just burst."),
            AegisState::Off => {}
        }
        if !self.is_ship(self.main_hull()) {
            return t(
                "Un mob ne porte pas l'Égide : choisissez un vaisseau de joueur.",
                "A mob can't carry the Aegis: pick a player ship.",
            );
        }
        if !self.s.multi || self.ally_index().is_none() {
            return t(
                "Aucun allié à couvrir : passez la scène « À plusieurs ».",
                "No ally to cover: switch the scene to Multiplayer.",
            );
        }
        if self.sim < self.aegis.cd_end {
            let s = (self.aegis.cd_end - self.sim).ceil();
            return Some(match self.s.lang {
                Lang::Fr => format!("Recharge : encore {s:.0} s."),
                Lang::En => format!("Cooldown: {s:.0} s left."),
            });
        }
        if self.sim < self.aegis.fatigue_end {
            let s = (self.aegis.fatigue_end - self.sim).ceil();
            return Some(match self.s.lang {
                Lang::Fr => format!("Fatigue : encore {s:.0} s."),
                Lang::En => format!("Fatigue: {s:.0} s left."),
            });
        }
        if self.state != State::Up {
            return t(
                "Votre bouclier est tombé ou en train de revenir.",
                "Your shield is down or coming back.",
            );
        }
        if self.s.pct <= 0.0 {
            return t("Votre bouclier est vide.", "Your shield is empty.");
        }
        None
    }

    fn aegis_fit(&mut self) {
        let Some(ally) = self.ally_index() else { return };
        let (main, a) = (&self.units[0], &self.units[ally]);
        let ext = a.flat_radius();
        let rxz = (a.pos.length() + ext * 1.25).max(main.flat_radius() * 1.6);
        let ry = (rxz * 0.35)
            .max(a.asset.radii.y * a.s * MARGIN * 1.3)
            .max(main.asset.radii.y * main.s * MARGIN * 1.3);
        self.aegis.radii = Vec3::new(rxz, ry, rxz);
    }

    pub fn aegis_deploy(&mut self, ctx: &mut SystemCtx) {
        if self.units.is_empty() {
            return;
        }
        if let Some(why) = self.aegis_refusal() {
            self.aegis.message = format!("{} {why}", self.tr("Refusé :", "Refused:"));
            self.aegis.message_t = self.sim;
            return;
        }
        self.aegis_fit();
        let a = &mut self.aegis;
        a.pct_before = self.s.pct;
        a.full = self.s.pct;
        a.reserve = self.s.pct;
        a.state = AegisState::Deploy;
        a.t0 = self.sim;
        a.t_deploy = self.sim;
        a.cd_end = self.sim + self.s.aegis_cooldown;
        a.impacts.clear();
        a.brk = None;
        a.worn.clear();
        a.message.clear();
        let ally = self.ally_index();
        for b in &mut self.bolts {
            if !b.miss && !b.passed && !b.aegis_done && (b.unit == 0 || Some(b.unit) == ally) {
                b.pre_aegis = true;
            }
        }
        self.absorb_for_aegis();
        let p = self.units[0].pos;
        self.sfx(ctx, "shield_buff_on", p, 0.9);
    }

    fn absorb_for_aegis(&mut self) {
        self.aegis.absorb = self.absorb_main();
    }

    fn aegis_scale(&self) -> f32 {
        self.aegis_scale_at(self.sim)
    }

    fn aegis_scale_at(&self, now: f32) -> f32 {
        let f0 = self.units[0].flat_radius() / self.aegis.radii.x.max(1e-3);
        let t = now - self.aegis.t0;
        match self.aegis.state {
            AegisState::Deploy => {
                let k = (t / AEGIS_DEPLOY).min(1.0);
                f0 + (1.0 - f0) * (1.0 - (1.0 - k) * (1.0 - k))
            }
            AegisState::Retract => {
                let k = (t / AEGIS_RETRACT).min(1.0);
                1.0 - (1.0 - f0) * k * k
            }
            _ => 1.0,
        }
    }

    fn aegis_matrix(&self) -> Mat4 {
        self.aegis_matrix_at(self.sim)
    }

    fn aegis_matrix_at(&self, now: f32) -> Mat4 {
        let f = self.aegis_scale_at(now);
        Mat4::from_scale_rotation_translation(Vec3::splat(f), Quat::from_rotation_y(self.spin), self.units[0].pos)
    }

    /// Does this bolt cross the Aegis wall from outside between `p0` and `p1`?
    fn aegis_hit(&self, b: &Bolt, p0: Vec3, p1: Vec3) -> Option<Vec3> {
        if !self.aegis_up() || b.aegis_done {
            return None;
        }
        if b.unit != 0 && Some(b.unit) != self.ally_index() {
            return None;
        }
        let r = self.aegis.radii * self.aegis_scale();
        let c = self.units[0].pos;
        let delta = p1 - p0;
        let len = delta.length();
        if len < 1e-6 {
            return None;
        }
        let dir = delta / len;
        let (o, d) = ((p0 - c) / r, dir / r);
        let (qa, qb, qc) = (d.dot(d), 2.0 * o.dot(d), o.dot(o) - 1.0);
        if qc <= 0.0 {
            // started inside: not stopped; already in flight at deployment: the wall sweeps it
            return b.pre_aegis.then_some(p0);
        }
        let disc = qb * qb - 4.0 * qa * qc;
        if disc < 0.0 {
            return None;
        }
        let t = (-qb - disc.sqrt()) / (2.0 * qa);
        (t >= 0.0 && t <= len).then(|| p0 + dir * t)
    }

    fn aegis_on_hit(&mut self, ctx: &mut SystemCtx, b: &mut Bolt, point: Vec3) -> AegisResult {
        let w = &WEAPONS[b.weapon];
        let local = self.aegis_matrix().inverse().transform_point3(point);
        b.aegis_done = true;
        if b.unit == 0 {
            self.last_hit = self.sim;
        }
        let color = self.main_color();
        self.hit_light(point, true, color, 1.0);
        let t = self.sim;
        let add = |g: &mut Game, s: f32, leak: f32, kind: f32, w: f32| {
            g.aegis.impacts.push(Impact {
                p: local,
                t,
                s,
                leak,
                kind,
                w,
                vid: 0,
            });
            while g.aegis.impacts.len() > MAX_IMPACTS {
                g.aegis.impacts.remove(0);
            }
        };
        if w.kind == WeaponKind::Siphon {
            let delay = 0.5 * self.s.wave_life;
            add(self, 0.9, 0.0, 1.0, delay);
            self.aegis_wear(b.vid, 1.0);
            self.sfx(ctx, "hit_shield", point, 0.55);
            self.siphon_packets(point, b.o, 0, delay, None);
            if self.aegis.reserve <= 0.0 {
                self.aegis_break(ctx, local);
            }
            return AegisResult::Stop;
        }
        let absorb = self.aegis.absorb;
        let share = if w.kind == WeaponKind::Blast {
            (absorb / 100.0).min(1.0)
        } else {
            self.share_of(absorb, w)
        };
        let over = self.aegis_wear(b.vid, share);
        let rem = (1.0 - share + over).min(1.0);
        if w.kind == WeaponKind::Blast {
            add(self, 1.5, 0.0, 2.0, 3.0);
            self.sparks(point, None, 160, BLAST_COL, 1.6, 1.4);
            ctx.spawn_effect(builtin::EXPLOSION, EffectSpawn::at(point).scale(0.35));
            self.sfx(ctx, "explosion_small", point, 0.85);
        } else {
            let rocket = w.kind == WeaponKind::Rocket;
            add(
                self,
                (0.2 + 0.8 * share) * if rocket { 1.4 } else { 1.0 },
                1.0 - share,
                0.0,
                if rocket { 1.3 } else { 1.0 },
            );
            self.sfx(ctx, "hit_shield", point, 0.7);
            if 1.0 - share > 0.02 {
                let k = if b.n > 1 { 1.0 / (b.n as f32).sqrt() } else { 1.0 };
                let normal = (point - self.units[0].pos) / (self.aegis.radii * self.aegis.radii);
                self.piercing(point, b.v, Some(normal), 1.0 - share, color, k);
            }
        }
        if self.aegis.reserve <= 0.0 {
            self.aegis_break(ctx, local);
        }
        if rem <= 0.02 {
            return AegisResult::Stop;
        }
        b.rem = Some(rem);
        let k = rem.sqrt();
        b.color *= 0.3 + 0.7 * k;
        b.half_width *= 0.45 + 0.55 * k;
        if w.kind == WeaponKind::Blast {
            AegisResult::Relay
        } else {
            AegisResult::Go
        }
    }

    /// A volley wears the reserve once; what the reserve couldn't take goes on to the target.
    fn aegis_wear(&mut self, vid: u64, share: f32) -> f32 {
        if vid != 0 && self.aegis.worn.contains(&vid) {
            return 0.0;
        }
        if vid != 0 {
            self.aegis.worn.push_back(vid);
        }
        let want = 7.0 * share;
        let took = self.aegis.reserve.min(want);
        self.aegis.reserve = (self.aegis.reserve - want).max(0.0);
        self.s.pct = self.aegis.reserve;
        if want > 0.0 {
            self.last_hit = self.sim;
            share * (want - took) / want
        } else {
            0.0
        }
    }

    pub fn aegis_break(&mut self, ctx: &mut SystemCtx, local: Vec3) {
        if matches!(self.aegis.state, AegisState::Off | AegisState::Break) {
            return;
        }
        let matrix = self.aegis_matrix();
        self.aegis.state = AegisState::Break;
        self.aegis.t0 = self.sim;
        self.aegis.fatigue_end = self.sim + self.aegis_fatigue();
        self.aegis.reserve = 0.0;
        if self.s.wear || self.aegis.pct_before <= 0.0 {
            self.s.pct = 0.0;
            self.state = State::Down;
            self.last_hit = self.sim;
        } else {
            self.s.pct = self.aegis.pct_before;
            self.state = State::Restoring;
        }
        self.state_t = self.sim;
        let r = self.aegis.radii;
        let w = self.world_len;
        let speed = (self.s.wave_speed * w * 1.5).max(PI * r.x / BREAK_MAX);
        self.aegis.brk = Some(BreakFx {
            pt: local,
            speed,
            width: w * 0.05,
            dur: PI * r.x / speed,
        });
        let pts: Vec<(Vec3, Vec3)> = (0..2600)
            .map(|k| {
                let u = ((k as f32 * 0.618_034) % 1.0) * 2.0 - 1.0;
                let a = k as f32 * 2.399_963;
                let s = (1.0 - u * u).max(0.0).sqrt();
                let n = Vec3::new(s * a.cos(), u * 0.999, s * a.sin());
                (n * r, norm_or(n / r, n))
            })
            .collect();
        let mean = (r.x + r.y + r.z) / 3.0;
        self.surface_debris(&pts, matrix, local, speed, Some((r, mean)));
        let at = matrix.transform_point3(local);
        self.sfx(ctx, "shield_buff_off", at, 0.9);
    }

    fn aegis_step(&mut self, ctx: &mut SystemCtx) {
        let t = self.sim - self.aegis.t0;
        match self.aegis.state {
            AegisState::Deploy if t >= AEGIS_DEPLOY => self.aegis.state = AegisState::On,
            AegisState::On if self.sim - self.aegis.t_deploy >= AEGIS_DUR => {
                self.aegis_give_back();
                self.aegis.state = AegisState::Retract;
                self.aegis.t0 = self.sim;
                self.aegis.fatigue_end = self.sim + self.aegis_fatigue();
                let p = self.units[0].pos;
                self.sfx(ctx, "shield_buff_off", p, 0.4);
            }
            AegisState::Retract if t >= AEGIS_RETRACT => self.aegis_reset(),
            AegisState::Break => {
                let done = self.aegis.brk.as_ref().is_none_or(|b| t >= b.dur + 0.1);
                if done {
                    self.aegis_reset();
                }
            }
            _ => {}
        }
    }

    /// One line of numbers for the automated checks.
    pub fn debug_line(&self) -> String {
        format!(
            "sim={:.2} units={} hits={} state={:?} pct={:.1} aegis={:?} reserve={:.1} R={:.0} bolts={} packets={} particles={} impacts={:?} quality={:?} sounds={} played={} audio={:?} loading={:?}",
            self.sim,
            self.units.len(),
            self.hits,
            self.state,
            self.s.pct,
            self.aegis.state,
            self.aegis.reserve,
            self.aegis.radii.x,
            self.bolts.len(),
            self.packets.len(),
            self.particle_count,
            self.units.iter().map(|u| u.impacts.len()).collect::<Vec<_>>(),
            self.s.quality,
            self.sounds.len(),
            self.sfx_played,
            self.audio_state,
            self.loading_text(),
        )
    }

    /// The Aegis status line of the panel.
    pub fn aegis_text(&self) -> String {
        let fr = self.s.lang == Lang::Fr;
        match self.aegis.state {
            AegisState::Deploy | AegisState::On => {
                let left = (AEGIS_DUR - (self.sim - self.aegis.t_deploy)).max(0.0);
                let share = self.share_of(self.aegis.absorb, self.weapon());
                if fr {
                    format!(
                        "Égide active : votre bouclier étendu tient encore {:.1} % (sur {:.0} %), encore {left:.1} s.\nAbsorption {:.0} % : contre cette arme, la bulle prend {:.0} %.",
                        self.aegis.reserve,
                        self.aegis.full,
                        self.aegis.absorb,
                        share * 100.0
                    )
                } else {
                    format!(
                        "Aegis up: your stretched shield holds {:.1}% (of {:.0}%), {left:.1} s left.\nAbsorbance {:.0}%: against this weapon the bubble takes {:.0}%.",
                        self.aegis.reserve,
                        self.aegis.full,
                        self.aegis.absorb,
                        share * 100.0
                    )
                }
            }
            AegisState::Retract => self
                .tr(
                    "Égide terminée : votre bouclier se resserre autour de vous.",
                    "Aegis over: your shield tightens back around you.",
                )
                .into(),
            AegisState::Break => {
                if self.s.wear {
                    self.tr(
                        "Égide brisée : votre bouclier est tombé.",
                        "Aegis broken: your shield is down.",
                    )
                    .into()
                } else {
                    self.tr(
                        "Égide brisée : sans usure, votre bouclier revient à sa valeur d'avant.",
                        "Aegis broken: without wear, your shield returns to its previous value.",
                    )
                    .into()
                }
            }
            AegisState::Off => {
                if !self.aegis.message.is_empty() && self.sim - self.aegis.message_t < 4.0 {
                    self.aegis.message.clone()
                } else if let Some(why) = self.aegis_refusal() {
                    format!("{} {why}", self.tr("Pas encore :", "Not yet:"))
                } else {
                    self.tr("Prête (bouton ou touche F).", "Ready (button or F key).")
                        .into()
                }
            }
        }
    }

    // ------------------------------------------------------------ frame

    fn step(&mut self, ctx: &mut SystemCtx, dt: f32) {
        self.sim += dt;
        if self.units.is_empty() {
            return;
        }
        if self.s.spin {
            self.spin += dt * 0.18;
        }
        self.update_transforms(ctx);
        if self.s.auto && dt > 0.0 {
            self.next_shot -= dt;
            if self.next_shot <= 0.0 {
                let count = self.units.len();
                let kind = self.weapon().kind;
                let per = if matches!(kind, WeaponKind::Rocket | WeaponKind::Blast) {
                    3.0
                } else {
                    1.0
                };
                let all = self.s.target == Target::All || !self.target_ok();
                let unit = if all {
                    (self.rnd(0.0, count as f32) as usize).min(count - 1)
                } else {
                    self.target_unit()
                };
                self.fire(None, Some(unit));
                let n = if all { count as f32 } else { 1.0 };
                self.next_shot = per / (self.s.rate.max(0.05) * n) * self.rnd(0.7, 1.3);
            }
        }
        self.step_bolts(ctx, dt);
        let sim = self.sim;
        let mut packets = std::mem::take(&mut self.packets);
        // the Aegis went up or down since the drain: your packets go to the shield you have now
        let aegis_up = self.aegis_up();
        for p in &mut packets {
            if let Some(home) = p.home
                && matches!(home, Home::Aegis(_)) != aegis_up
                && let Some(u) = self.units.get(p.unit)
            {
                p.home = Some(self.siphon_home(u.model.transform_point3(p.loc)));
            }
        }
        let mut gains = Vec::new();
        packets.retain(|p| {
            let flying = sim < p.t0 + p.dur;
            if !flying && let (Some(home), Some(n)) = (p.home, p.gain) {
                gains.push((home, n, p.t0 + p.dur));
            }
            flying
        });
        self.packets = packets;
        for (home, n, t) in gains {
            self.siphon_gain(ctx, home, n, t);
        }
        self.aegis_step(ctx);
        if self.state == State::Breaking && self.brk.as_ref().is_none_or(|b| sim - self.state_t > b.dur + 0.1) {
            self.state = State::Down;
            self.state_t = sim;
        }
        if self.s.wear {
            if self.state == State::Down && sim - self.state_t.max(self.last_hit) > self.s.regen {
                self.restore_shield(false);
                let p = self.units[0].pos;
                self.sfx(ctx, "shield_buff_on", p, 0.8);
            }
            if self.state == State::Up && !self.aegis_up() && sim - self.last_hit > self.s.regen && self.s.pct < 100.0 {
                self.s.pct = (self.s.pct + dt * 25.0).min(100.0);
            }
        }
        if self.state == State::Restoring && sim - self.state_t > 0.8 {
            self.state = State::Up;
        }
        self.floats.retain(|f| sim - f.t < 0.95);
    }

    fn update_light(&mut self, ctx: &mut SystemCtx) {
        let light = *self.light.get_or_insert_with(|| {
            ctx.world.spawn((
                Transform::IDENTITY,
                PointLight {
                    color: Vec3::ONE,
                    intensity: 0.0,
                    range: 1.0,
                },
            ))
        });
        let age = self.render_t - self.light_t;
        let base = if self.light_shield { 2.5 } else { 4.0 };
        let intensity = if age < 0.4 {
            base * self.light_k * (-age * 10.0).exp()
        } else {
            0.0
        };
        let c = self.light_col;
        let m = c[0].max(c[1]).max(c[2]);
        let color = if !self.light_shield {
            Vec3::new(1.0, 0.5, 0.2)
        } else if m > 1e-6 {
            Vec3::from_array(c) / m
        } else {
            Vec3::ONE
        };
        let range = self.world_len * if self.light_k == 2.0 { 1.8 } else { 1.1 };
        let pos = self.light_pos;
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(light) {
            t.translation = pos;
        }
        if let Ok(mut l) = ctx.world.get::<&mut PointLight>(light) {
            l.color = color;
            l.intensity = intensity * 40.0;
            l.range = range;
        }
    }

    fn update_camera(&mut self, ctx: &mut SystemCtx) {
        let window = ctx
            .resources
            .get::<Input>()
            .map_or(Vec2::new(16.0, 9.0), Input::window_size)
            .max(Vec2::ONE);
        let aspect = window.x / window.y;
        let fov = FOV_Y.to_radians();
        let half_v = fov * 0.5;
        let half_h = (half_v.tan() * aspect).atan();
        let w = self.world_len;
        let mut d = w * 3.4;
        if self.units.len() > 1 {
            let span = self.units[1..]
                .iter()
                .map(|u| u.pos.x.abs() + u.flat_radius())
                .fold(0.0, f32::max);
            d = d.max(span * 1.12 / half_h.tan());
        }
        if self.aegis.state != AegisState::Off {
            d = d
                .max(self.aegis.radii.x * 1.1 / half_h.tan())
                .max(self.aegis.radii.x * 1.35 / half_v.tan());
        }
        if self.orbit_dist <= 0.0 {
            self.orbit_dist = d;
        }
        let (yaw, pitch, dist) = if self.s.orbit {
            (self.yaw, self.pitch, self.orbit_dist)
        } else {
            self.orbit_dist = d;
            (0.0, 62f32.to_radians(), d)
        };
        let eye = Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos()) * dist;
        for (t, cam, post) in ctx
            .world
            .query_mut::<(&mut Transform, &mut Camera3d, &mut PostProcess)>()
        {
            *t = Transform::from_xyz(eye.x, eye.y, eye.z).looking_at(Vec3::ZERO, Vec3::Y);
            cam.projection = Projection::Perspective {
                fov_y: fov,
                near: w * 0.05,
                far: w * 200.0,
            };
            post.bloom.intensity = self.s.bloom;
            post.exposure = self.s.exposure;
        }
    }

    /// World-to-clip of the current camera, for the labels.
    pub fn view_proj(&self, ctx: &SystemCtx, aspect: f32) -> Option<Mat4> {
        let mut q = ctx.world.query::<(&Transform, &Camera3d)>();
        let (t, c) = q.iter().next()?;
        Some(c.projection.matrix(aspect) * t.matrix().inverse())
    }

    /// A click in the view fires from that side. Called by the panel, once this frame's
    /// `ui_busy` is known (the pointer can jump from the panel to the view between two frames).
    pub fn pointer(&mut self, ctx: &mut SystemCtx) {
        let Some(input) = ctx.resources.get::<Input>() else {
            return;
        };
        let mouse = input.mouse_position();
        let down = input.mouse_just_pressed(MouseButton::Left);
        let up = input.mouse_just_released(MouseButton::Left);
        if down {
            self.press = mouse;
            self.pressed_on_ui = self.ui_busy;
        }
        if up {
            if let (Some(a), Some(b), false, false) = (self.press, mouse, self.pressed_on_ui, self.ui_busy)
                && a.distance(b) < 6.0
                && !self.units.is_empty()
            {
                let unit = self.target_unit();
                if let Some(angle) = self.click_angle(ctx, b, unit) {
                    self.pending.push(Cmd::Fire(Some(angle), Some(unit)));
                }
            }
            self.press = None;
        }
    }

    fn click_angle(&self, ctx: &SystemCtx, at: Vec2, unit: usize) -> Option<f32> {
        let window = ctx.resources.get::<Input>()?.window_size().max(Vec2::ONE);
        let mut q = ctx.world.query::<(&Transform, &Camera3d)>();
        let (t, c) = q.iter().next()?;
        let ndc = Vec2::new(2.0 * at.x / window.x - 1.0, 1.0 - 2.0 * at.y / window.y);
        let (o, d) =
            space3d::render::unproject_ray(t.matrix().inverse(), c.projection.matrix(window.x / window.y), ndc);
        if d.y.abs() < 1e-5 {
            return None;
        }
        let k = -o.y / d.y;
        if k <= 0.0 {
            return None;
        }
        let p = o + d * k - self.units[unit].pos;
        (p.x * p.x + p.z * p.z > 1e-3).then(|| p.z.atan2(p.x))
    }

    fn controls(&mut self, ctx: &mut SystemCtx) {
        let Some(input) = ctx.resources.get::<Input>() else {
            return;
        };
        let (typing, popup_open) = ctx
            .resources
            .get::<space3d::ui::EguiContext>()
            .map_or((false, false), |e| {
                (e.wants_keyboard_input(), egui::Popup::is_any_open(e.ctx()))
            });
        let keys = [
            KeyCode::Space,
            KeyCode::KeyB,
            KeyCode::KeyR,
            KeyCode::KeyF,
            KeyCode::KeyP,
            KeyCode::Digit1,
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::Digit4,
            KeyCode::Escape,
        ]
        .map(|k| !typing && input.just_pressed(k));
        let held = input.mouse_pressed(MouseButton::Left);
        let delta = input.mouse_delta();
        let wheel = input.wheel_delta().y;
        if self.s.orbit && held && !self.pressed_on_ui {
            self.yaw -= delta.x * 0.006;
            self.pitch = (self.pitch + delta.y * 0.006).clamp(0.1, 1.5);
        }
        if self.s.orbit && !self.ui_busy && wheel != 0.0 {
            self.orbit_dist =
                (self.orbit_dist * (1.0 - wheel * 0.08).clamp(0.5, 1.5)).clamp(self.world_len, self.world_len * 40.0);
        }
        if self.units.is_empty() {
            return;
        }
        for (pressed, cmd) in [
            (keys[0], Cmd::Fire(None, None)),
            (keys[1], Cmd::Break),
            (keys[2], Cmd::Restore),
            (keys[3], Cmd::Aegis),
        ] {
            if pressed {
                self.pending.push(cmd);
            }
        }
        if keys[4] {
            self.s.paused = !self.s.paused;
        }
        for (k, q) in [
            GraphicsQuality::Low,
            GraphicsQuality::Medium,
            GraphicsQuality::High,
            GraphicsQuality::Ultra,
        ]
        .into_iter()
        .enumerate()
        {
            if keys[5 + k] {
                self.s.quality = q;
                self.quality_auto = false;
            }
        }
        // Esc closes an open list first; on the web there is nothing to quit to
        if keys[9] && !popup_open && cfg!(not(target_arch = "wasm32")) {
            ctx.send(AppExit);
        }
    }

    /// Queue a command for the next fixed tick.
    pub fn queue(&mut self, cmd: Cmd) {
        self.pending.push(cmd);
    }

    /// At the start of a fixed tick: the scripted options due at this tick, then the commands.
    fn apply_commands(&mut self, ctx: &mut SystemCtx) {
        let tick = self.scene_frames;
        let due = |at: &mut Option<u64>| {
            let hit = at.is_some_and(|f| tick >= f);
            if hit {
                *at = None;
            }
            hit
        };
        if due(&mut self.switch_at)
            && let Some(hull) = config("SHIELD_SWITCH_TO").and_then(|n| hull_index(&n))
        {
            self.s.main = hull;
        }
        if due(&mut self.fire_at) {
            self.pending.push(Cmd::Fire(None, None));
        }
        if due(&mut self.break_at) {
            self.pending.push(Cmd::Break);
        }
        if due(&mut self.aegis_at) {
            self.pending.push(Cmd::Aegis);
        }
        for cmd in std::mem::take(&mut self.pending) {
            match cmd {
                Cmd::Fire(angle, unit) => self.fire(angle, unit),
                Cmd::Break => self.break_button(ctx),
                Cmd::Restore => self.restore_button(ctx),
                Cmd::Aegis => self.aegis_deploy(ctx),
                Cmd::Pct => self.pct_changed(ctx),
            }
        }
    }

    pub fn set_quality(&mut self, q: GraphicsQuality) {
        self.s.quality = q;
        self.quality_auto = false;
    }

    fn apply_quality(&mut self, ctx: &mut SystemCtx) {
        let Some(recommended) = ctx.resources.get::<Renderer>().map(GraphicsQuality::recommended_for) else {
            return;
        };
        if self.quality_auto {
            self.quality_auto = false;
            self.s.quality = recommended;
        }
        let q = self.s.quality;
        if self.quality_applied != Some(q) {
            q.apply(ctx.world, ctx.resources);
            ctx.resources.insert(ParticleSettings::for_quality(q));
            self.quality_applied = Some(q);
        }
    }

    fn poll_sounds(&mut self) {
        if self.sound_fetch.is_none() && self.sound_task.is_none() && self.sounds.is_empty() && !self.sounds_tried {
            self.sounds_tried = true;
            let source = self.source.get_or_insert_with(|| default_source(&asset_root())).clone();
            self.sound_fetch = Some(spawn(async move {
                let mut files = Vec::new();
                for name in SFX_NAMES {
                    if let Ok(bytes) = source.read(&AssetPath::new(&format!("sfx/{name}.mp3"))).await {
                        files.push((name, bytes));
                    }
                }
                files
            }));
        }
        if let Some(t) = &mut self.sound_fetch
            && let Some(files) = t.poll()
        {
            self.sound_fetch = None;
            self.sound_task = Some(spawn_compute(move || {
                let mut map: HashMap<String, Vec<StaticSoundData>> = HashMap::new();
                for (name, bytes) in files {
                    // "hit_shield_2" → variant of "hit_shield"
                    let base = match name.rsplit_once('_') {
                        Some((b, n)) if n.chars().all(|c| c.is_ascii_digit()) => b,
                        _ => name,
                    };
                    if let Ok(s) = space3d::audio::decode(bytes) {
                        map.entry(base.to_string()).or_default().push(s);
                    }
                }
                map
            }));
        }
        if let Some(t) = &mut self.sound_task
            && let Some(map) = t.poll()
        {
            self.sounds = map;
            self.sound_task = None;
        }
    }

    fn base_uniform(&self, model: Mat4, color: [f32; 3], len: f32, look: [f32; 4]) -> [f32; UNIFORM_FLOATS] {
        let s = &self.s;
        let [intensity, rest, fresnel, back] = look;
        let mut u = [0.0f32; UNIFORM_FLOATS];
        u[..16].copy_from_slice(&model.to_cols_array());
        u[16..20].copy_from_slice(&[color[0], color[1], color[2], intensity]);
        u[20..24].copy_from_slice(&[rest, fresnel, back, self.render_t]);
        u[24..28].copy_from_slice(&[
            1.0,
            if s.hex { 1.0 } else { 0.0 },
            s.hex_scale / len,
            s.wave_speed * len,
        ]);
        u[28..32].copy_from_slice(&[s.wave_width * len, s.wave_life, s.flash, 1.0]);
        u[44..48].copy_from_slice(&[len * 0.08, 2.0, -1.0, 1.0]);
        u
    }

    fn write_impacts(u: &mut [f32; UNIFORM_FLOATS], list: &[Impact]) {
        for k in 0..MAX_IMPACTS {
            let (a, b) = (52 + k * 4, 116 + k * 4);
            match list.get(k) {
                Some(m) => {
                    u[a..a + 4].copy_from_slice(&[m.p.x, m.p.y, m.p.z, m.t]);
                    u[b..b + 4].copy_from_slice(&[m.s, m.leak, m.kind, m.w]);
                }
                None => u[a + 3] = -1.0,
            }
        }
        for k in MAX_IMPACTS..16 {
            u[52 + k * 4 + 3] = -1.0;
        }
        u[49] = list.len().min(MAX_IMPACTS) as f32;
    }

    fn publish(&mut self, ctx: &mut SystemCtx) {
        let s = &self.s;
        let net = |a: f32| ((100.0 - a) / 55.0).clamp(0.0, 1.0);
        let mut draws = Vec::new();
        let aegis_up = self.aegis_up();
        for (i, unit) in self.units.iter().enumerate() {
            let main = i == 0;
            if main && (self.state == State::Down || aegis_up) {
                continue;
            }
            let a = &unit.asset;
            let mut u = self.base_uniform(
                unit.model,
                self.unit_color(i),
                a.len,
                [s.intensity, s.rest, s.fresnel, s.back],
            );
            let absorb = if self.immune(i) { 100.0 } else { self.absorb_of(i) };
            u[40..44].copy_from_slice(&[0.0, 0.0, 0.0, net(absorb)]);
            u[24] = if main { s.pct / 100.0 } else { 1.0 };
            u[36..40].copy_from_slice(&[0.0, 0.0, a.len * 0.05, 9.0 / a.len]);
            u[32..36].copy_from_slice(&[a.radii.x, a.radii.y, a.radii.z, a.len * 0.5]);
            let mut reveal = 2.0;
            if main {
                if self.state == State::Breaking
                    && let Some(b) = &self.brk
                {
                    let t = self.render_t - self.state_t;
                    u[36..40].copy_from_slice(&[1.0, t * b.speed - b.width, b.width, 9.0 / a.len]);
                    u[40..43].copy_from_slice(&b.pt.to_array());
                }
                if self.state == State::Restoring {
                    reveal = (self.render_t - self.state_t) / 0.8 * 1.05;
                }
            }
            u[44..48].copy_from_slice(&[a.len * 0.08, reveal, a.shell_z.0, a.shell_z.1]);
            Self::write_impacts(&mut u, &unit.impacts);
            draws.push(ShieldDraw {
                kind: DrawKind::Hull,
                mesh: a.shell_mesh,
                uniform: u,
            });
        }
        // the Aegis: the carrier's shield stretched into a bubble, its ground ring and gauge
        if self.aegis.state != AegisState::Off
            && !self.units.is_empty()
            && let (Some(sphere), Some(ring)) = (self.sphere, self.ring)
        {
            let w = self.world_len;
            let r = self.aegis.radii;
            let col = self.main_color();
            let model = self.aegis_matrix_at(self.render_t);
            let mut u = self.base_uniform(
                model,
                col,
                w,
                [s.intensity * 0.85, s.rest * 0.7, s.fresnel * 1.3, s.back * 0.6],
            );
            let on_t = self.render_t - self.aegis.t_deploy;
            let ending = self.aegis.state == AegisState::On && on_t > AEGIS_DUR - 2.0;
            u[24] = if ending { 0.2 } else { 1.0 };
            u[31] = 0.0;
            u[32..36].copy_from_slice(&[r.x, r.y, r.z, (2.0 * r.x + r.y) / 3.0]);
            u[36..40].copy_from_slice(&[0.0, 0.0, w * 0.05, 9.0 / w]);
            u[40..44].copy_from_slice(&[0.0, 0.0, 0.0, net(self.aegis.absorb)]);
            u[44..48].copy_from_slice(&[w * 0.08, 2.0, -r.x, r.x]);
            if self.aegis.state == AegisState::Break
                && let Some(b) = &self.aegis.brk
            {
                let t = self.render_t - self.aegis.t0;
                u[36..40].copy_from_slice(&[1.0, t * b.speed - b.width, b.width, 9.0 / w]);
                u[40..43].copy_from_slice(&b.pt.to_array());
            }
            Self::write_impacts(&mut u, &self.aegis.impacts);
            draws.push(ShieldDraw {
                kind: DrawKind::Aegis,
                mesh: sphere,
                uniform: u,
            });
            if self.aegis.state != AegisState::Break {
                let ratio = if self.aegis.full > 0.0 {
                    self.aegis.reserve / self.aegis.full
                } else {
                    0.0
                };
                let pulse = if ratio < 0.15 {
                    0.55 + 0.45 * (self.render_t * 14.0).sin()
                } else {
                    1.0
                };
                let m = col[0].max(col[1]).max(col[2]).max(1e-6);
                let gauge = if ratio < 0.15 {
                    [0.95, 0.12, 0.1]
                } else if ratio < 0.5 {
                    [0.95, 0.55, 0.1]
                } else {
                    col.map(|v| 0.25 + 0.65 * v / m)
                }
                .map(|v| v * pulse);
                let f = self.aegis_scale_at(self.render_t) * r.x;
                let pos = self.units[0].pos - Vec3::Y * (r.y * 0.15);
                let ring_model =
                    Mat4::from_scale_rotation_translation(Vec3::splat(f), Quat::from_rotation_y(self.spin), pos);
                let mut u = self.base_uniform(ring_model, gauge, 1.0, [1.0, 0.0, 1.0, 1.0]);
                u[50] = ratio.clamp(0.0, 1.0);
                draws.push(ShieldDraw {
                    kind: DrawKind::Ring,
                    mesh: ring,
                    uniform: u,
                });
            }
        }
        // bolts and siphon packets
        let mut streaks: Vec<f32> = Vec::with_capacity(self.bolts.len() * STREAK_FLOATS);
        for b in &self.bolts {
            if !b.started {
                continue;
            }
            // drawn where it is at the displayed time, between two fixed steps
            let ahead = if b.life > 0.0 {
                b.speed * (self.render_t - self.sim).max(0.0)
            } else {
                0.0
            };
            let tail = b.p + b.v * (ahead - b.length * 0.5);
            streaks.extend_from_slice(&[
                tail.x,
                tail.y,
                tail.z,
                0.0,
                b.v.x,
                b.v.y,
                b.v.z,
                b.length,
                b.color.x,
                b.color.y,
                b.color.z,
                b.half_width,
            ]);
        }
        let w = self.world_len;
        let sim = self.render_t;
        let mut packets = std::mem::take(&mut self.packets);
        for p in &mut packets {
            let t = (sim - p.t0) / p.dur;
            if !(0.0..1.0).contains(&t) {
                continue;
            }
            let Some(unit) = self.units.get(p.unit) else { continue };
            let from = *p.from.get_or_insert_with(|| unit.model.transform_point3(p.loc));
            let to = p.home.map_or(p.to, |h| self.home_world(h, sim));
            let k = t * t * (3.0 - 2.0 * t);
            let at = from.lerp(to, k);
            let d = norm_or(to - from, Vec3::X);
            let tail = at - d * (w * 0.012);
            streaks.extend_from_slice(&[
                tail.x,
                tail.y,
                tail.z,
                0.0,
                d.x,
                d.y,
                d.z,
                w * 0.024,
                0.25,
                2.6,
                2.2,
                w * 0.018,
            ]);
        }
        self.packets = packets;
        for e in &mut self.writes {
            e.3 += 1;
        }
        while self.writes.front().is_some_and(|e| e.3 > 8) {
            self.writes.pop_front();
        }
        let mut global = [0.0; UNIFORM_FLOATS];
        global[19] = 1.0; // the fragments carry the intensity in their colour
        global[23] = self.render_t;
        let frame = ctx.resources.init::<ShieldFrame>();
        frame.global = global;
        frame.draws = draws;
        frame.particle_writes = self.writes.iter().map(|e| (e.0, e.1, e.2.clone())).collect();
        frame.particle_count = self.particle_count;
        frame.streaks = Arc::new(streaks);
    }
}

enum AegisResult {
    Stop,
    Go,
    Relay,
}

impl Bolt {
    fn o_at(&self, travel: f32) -> Vec3 {
        self.p + self.v * (travel - self.travel)
    }

    /// A copy for a one-off ray test.
    fn clone_shallow(&self) -> Bolt {
        Bolt {
            p: self.p,
            o: self.o,
            v: self.v,
            speed: self.speed,
            life: self.life,
            unit: self.unit,
            weapon: self.weapon,
            miss: self.miss,
            centre: self.centre,
            best: self.best,
            shown: self.shown,
            vid: self.vid,
            n: self.n,
            quiet: self.quiet,
            t0: self.t0,
            started: self.started,
            passed: self.passed,
            leak: self.leak,
            rem: self.rem,
            aegis_done: self.aegis_done,
            pre_aegis: self.pre_aegis,
            from_you: self.from_you,
            travel: self.travel,
            cast: None,
            color: self.color,
            half_width: self.half_width,
            length: self.length,
            trail: None,
        }
    }
}

fn upload_hull(c: HullContent, resources: &mut Resources) -> HullAsset {
    let assets = resources.expect_mut::<RenderAssets>();
    let shell_pts: Vec<(Vec3, Vec3)> = c
        .shell
        .positions
        .iter()
        .zip(&c.shell.normals)
        .map(|(p, n)| (Vec3::from_array(*p), Vec3::from_array(*n)))
        .collect();
    let shell_z = shell_pts
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.0.z), b.max(p.0.z)));
    HullAsset {
        model: c.model.upload(assets),
        shell_mesh: assets.add_mesh(c.shell),
        center: c.center,
        size: c.size,
        len: c.size.max_element(),
        radii: c.radii,
        shell_tris: c.shell_tris,
        shell_pts,
        shell_z,
        ship_tris: c.ship_tris,
    }
}

fn ring_mesh() -> MeshData {
    let seg = 240u32;
    let (r0, r1) = (0.97f32, 1.04f32);
    let mut m = MeshData::default();
    for i in 0..=seg {
        let a = i as f32 / seg as f32 * TAU;
        let (s, c) = a.sin_cos();
        for r in [r0, r1] {
            m.positions.push([r * s, 0.0, r * c]);
            m.normals.push([0.0, 1.0, 0.0]);
            m.uvs.push([0.0, 0.0]);
        }
    }
    for i in 0..seg {
        let k = i * 2;
        m.indices.extend_from_slice(&[k, k + 1, k + 2, k + 2, k + 1, k + 3]);
    }
    m
}

// ---------------------------------------------------------------- systems

pub fn setup(ctx: &mut SystemCtx) {
    ctx.world.spawn((
        Transform::from_xyz(0.0, 520.0, 260.0).looking_at(Vec3::ZERO, Vec3::Y),
        Camera3d::default().with_clear_color(Vec4::new(0.004, 0.005, 0.012, 1.0)),
        PostProcess::default()
            .with_tonemapper(Tonemapper::AgX)
            .with_bloom(Bloom::glow()),
    ));
    ctx.world.spawn((
        Transform::IDENTITY.looking_at(Vec3::new(-0.4, -1.0, -0.5), Vec3::Y),
        DirectionalLight {
            color: Vec3::new(1.0, 0.95, 0.88),
            intensity: 2.2,
        },
    ));
    ctx.world.spawn((
        Transform::IDENTITY.looking_at(Vec3::new(0.8, -0.4, 0.6), Vec3::Y),
        DirectionalLight {
            color: Vec3::new(0.5, 0.66, 1.0),
            intensity: 0.6,
        },
    ));
    ctx.resources.insert(AmbientLight {
        color: Vec3::new(0.6, 0.7, 1.0),
        intensity: 0.25,
    });
    let assets = ctx.resources.expect_mut::<RenderAssets>();
    let sphere = assets.add_mesh(MeshData::sphere(1.0, 128, 64));
    let ring = assets.add_mesh(ring_mesh());
    let game = ctx.resources.expect_mut::<Game>();
    game.sphere = Some(sphere);
    game.ring = Some(ring);
}

/// Test report: the console, and `window.__shield` on the web for the test scripts.
fn report_state(line: &str) {
    space3d::tracing::warn!("SHIELD_STATE {line}");
    #[cfg(target_arch = "wasm32")]
    if let Some(w) = web_sys::window() {
        let _ = js_sys::Reflect::set(&w, &"__shield".into(), &line.into());
    }
}

/// The simulation, one fixed step (`TickRate`, 60 Hz by default): shots, rules, Aegis.
pub fn tick(ctx: &mut SystemCtx) {
    let Some(mut g) = ctx.resources.remove::<Game>() else {
        return;
    };
    g.sync_scene(ctx);
    if !g.units.is_empty() {
        g.scene_frames += 1;
        g.apply_commands(ctx);
    }
    let dt = if g.s.paused {
        0.0
    } else {
        ctx.time().fixed_delta_secs() * g.s.speed
    };
    g.step(ctx, dt);
    if g.log_at.is_some_and(|f| g.scene_frames >= f) {
        g.log_at = None;
        report_state(&g.debug_line());
    }
    ctx.resources.insert(g);
}

/// Tests: install the test ship for every hull now, without the asynchronous loading.
#[cfg(test)]
pub(crate) fn preload_fallback(resources: &mut Resources) {
    let content = build_hull("ship", FALLBACK_GLB.to_vec(), None).expect("test ship");
    let asset = Arc::new(upload_hull(content, resources));
    let g = resources.expect_mut::<Game>();
    for slot in &mut g.slots {
        *slot = Slot::Ready(asset.clone());
    }
    g.fallback = true;
    g.sounds_tried = true;
}

/// Everything else, once per frame: loading, input, camera and the node's data.
pub fn update(ctx: &mut SystemCtx) {
    let Some(mut g) = ctx.resources.remove::<Game>() else {
        return;
    };
    let elapsed = ctx.time().elapsed_secs();
    g.fps_n += 1;
    if elapsed - g.fps_t >= 1.0 {
        g.fps = g.fps_n as f32 / (elapsed - g.fps_t) as f32;
        g.fps_t = elapsed;
        g.fps_n = 0;
    }
    g.apply_quality(ctx);
    g.poll_sounds();
    if let Some(a) = ctx.resources.get::<Audio>() {
        g.audio_state = (a.is_enabled(), a.ignored_calls());
    }
    g.controls(ctx);
    if let Some(p) = ctx.resources.get_mut::<ParticleSettings>() {
        p.time_scale = if g.s.paused { 0.0 } else { g.s.speed };
    }
    let ahead = if g.s.paused {
        0.0
    } else {
        ctx.time().alpha * ctx.time().fixed_delta_secs() * g.s.speed
    };
    g.render_t = g.sim + ahead;
    g.update_light(ctx);
    g.update_camera(ctx);
    g.publish(ctx);
    ctx.resources.insert(g);
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Every hull of the catalog is committed, loads, and comes with a readable baked envelope.
    #[test]
    fn the_committed_models_load_with_their_envelopes() {
        let dir = std::path::Path::new(COMMITTED_ASSETS);
        for h in &HULLS {
            let glb =
                std::fs::read(dir.join(format!("{}.glb", h.file))).unwrap_or_else(|e| panic!("{}.glb: {e}", h.file));
            let sbe_bytes =
                std::fs::read(dir.join(format!("{}.sbe", h.file))).unwrap_or_else(|e| panic!("{}.sbe: {e}", h.file));
            if let Err(e) = sbe::decode(&sbe_bytes) {
                panic!("{}.sbe: {e}", h.file);
            }
            let hull = build_hull(h.file, glb, Some(sbe_bytes)).unwrap_or_else(|e| panic!("{e}"));
            assert!(!hull.shell_tris.is_empty() && !hull.ship_tris.is_empty(), "{}", h.file);
        }
    }
}
