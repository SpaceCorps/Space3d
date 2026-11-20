//! Space3d audio: [kira](https://docs.rs/kira)-based playback with master/music/sfx buses,
//! fades, and spatial emitters synced from the ECS.
//!
//! * [`AudioPlugin`] inserts the [`Audio`] resource and a `PostUpdate` system that syncs
//!   [`AudioListener`]/[`AudioEmitter`] poses from `GlobalTransform` (after transform
//!   propagation) and starts queued emitter sounds.
//! * [`Audio`] plays one-shots and looping music from [`StaticSoundData`] or encoded bytes,
//!   controls per-sound volume/pan/pitch, bus volumes and fades.
//! * No output device (headless CI, servers)? [`Audio`] falls back to
//!   [`AudioMode::Disabled`]: every call is a cheap no-op, a single warning is logged, and
//!   the backend-independent state (bus levels, listener/emitter sync) keeps working.
//!   Set `SPACE3D_AUDIO=off` to force this, or `SPACE3D_AUDIO=mock` for a deviceless kira
//!   backend.
//!
//! ```
//! use space3d_core::prelude::*;
//! use space3d_audio::*;
//!
//! let mut app = App::new();
//! app.add_plugin(AudioPlugin::disabled());
//! let beep = sound_from_samples(48_000, &[0.0; 480]);
//! let audio = app.resources.expect_mut::<Audio>();
//! audio.set_bus_volume(Bus::Music, 0.5, Fade::secs(1.0));
//! let _id = audio.play(&beep, PlayParams::sfx().volume(0.8));
//! app.world.spawn((Transform::IDENTITY, AudioListener::default()));
//! let src = app.world.spawn((Transform::from_xyz(3.0, 0.0, 0.0), AudioEmitter::new()));
//! app.step_ticks(1);
//! assert!(app.world.get::<&AudioEmitter>(src).unwrap().mix().unwrap().balance() > 0.0);
//! ```
#![warn(missing_docs)]

mod backend;
mod spatial;
mod volume;

pub use kira;
pub use kira::sound::PlaybackState;
pub use kira::sound::static_sound::StaticSoundData;
pub use kira::{Easing, Frame};
pub use spatial::{
    AudioEmitter, AudioListener, EmitterSettings, ListenerPose, SpatialMix, ease, find_listener, spatial_mix,
};
pub use volume::{Bus, Fade, MixerLevels, PlayParams, SILENCE_DB, amplitude_to_db, db_to_amplitude};

use backend::Backend;
use space3d_core::hecs::Entity;
use space3d_core::{App, Plugin, Stage, SystemCtx};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Audio errors (decoding; playback failures are logged, not returned).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AudioError {
    /// The bytes are not a supported/valid audio file.
    #[error("could not decode audio: {0}")]
    Decode(String),
}

/// Decode an encoded audio file (wav, ogg/vorbis, mp3, flac) into memory.
///
/// This is CPU work; for large files call it inside `space3d_tasks::spawn_compute`.
pub fn decode(bytes: impl Into<Vec<u8>>) -> Result<StaticSoundData, AudioError> {
    StaticSoundData::from_cursor(std::io::Cursor::new(bytes.into())).map_err(|e| AudioError::Decode(e.to_string()))
}

/// Build a sound from mono samples in -1..=1 (procedural audio, tests).
pub fn sound_from_samples(sample_rate: u32, mono: &[f32]) -> StaticSoundData {
    StaticSoundData {
        sample_rate,
        frames: mono.iter().map(|&s| Frame::from_mono(s)).collect::<Arc<[Frame]>>(),
        settings: Default::default(),
        slice: None,
    }
}

/// Identifies a sound started through [`Audio`]. Valid even in disabled mode (it just
/// never refers to anything audible).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SoundId(pub u64);

/// Which backend is in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AudioMode {
    /// Real output device.
    Device,
    /// kira mock backend: full mixer graph, no output (tests/tools).
    Mock,
    /// No backend; all calls are no-ops.
    Disabled,
}

/// Backend selection for [`AudioSettings`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AudioBackend {
    /// Open the default device, falling back to [`AudioMode::Disabled`] if that fails.
    /// The `SPACE3D_AUDIO` env var (`off`/`disabled`/`0`, or `mock`) overrides this.
    #[default]
    Auto,
    /// Use the kira mock backend.
    Mock,
    /// Never open a device.
    Disabled,
}

/// Initial configuration for [`AudioPlugin`] / [`Audio::new`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AudioSettings {
    /// Backend selection.
    pub backend: AudioBackend,
    /// Initial bus levels.
    pub levels: MixerLevels,
}

impl AudioSettings {
    /// Apply the `SPACE3D_AUDIO` environment override (only affects [`AudioBackend::Auto`]).
    pub fn with_env_override(mut self) -> Self {
        if self.backend == AudioBackend::Auto
            && let Ok(v) = std::env::var("SPACE3D_AUDIO")
        {
            match v.trim().to_ascii_lowercase().as_str() {
                "off" | "disabled" | "0" | "none" | "false" => self.backend = AudioBackend::Disabled,
                "mock" => self.backend = AudioBackend::Mock,
                _ => {}
            }
        }
        self
    }
}

/// The audio resource. See the crate docs.
pub struct Audio {
    // Mutex only to make kira's handles `Sync` for `Resources`; accessed via `&mut` (no locking).
    backend: Option<Mutex<Backend>>,
    mode: AudioMode,
    disabled_reason: Option<String>,
    levels: MixerLevels,
    next_id: u64,
    music: Option<SoundId>,
    ignored_calls: u64,
    listener: Option<ListenerPose>,
    emitters: BTreeSet<Entity>,
    /// In the background ([`Audio::set_background`]), and the sounds it paused for that.
    background: bool,
    background_paused: Vec<SoundId>,
}

impl Audio {
    /// Create according to `settings` (does not read the env override; see
    /// [`AudioSettings::with_env_override`]).
    pub fn new(settings: &AudioSettings) -> Self {
        let mut audio = match settings.backend {
            AudioBackend::Disabled => Self::disabled("disabled by configuration"),
            AudioBackend::Mock => Self::from_backend(Backend::mock(), AudioMode::Mock),
            AudioBackend::Auto => Self::from_backend(Backend::device(), AudioMode::Device),
        };
        audio.set_levels(settings.levels, Fade::IMMEDIATE);
        audio
    }

    /// Audio with the kira mock backend (falls back to disabled if even that fails).
    pub fn mock() -> Self {
        Self::from_backend(Backend::mock(), AudioMode::Mock)
    }

    /// Audio that never touches a device. Logs `reason` once.
    pub fn disabled(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        tracing::warn!(%reason, "audio disabled; audio calls will be no-ops");
        Self {
            backend: None,
            mode: AudioMode::Disabled,
            disabled_reason: Some(reason),
            levels: MixerLevels::default(),
            next_id: 0,
            music: None,
            ignored_calls: 0,
            listener: None,
            emitters: BTreeSet::new(),
            background: false,
            background_paused: Vec::new(),
        }
    }

    fn from_backend(backend: Result<Backend, String>, mode: AudioMode) -> Self {
        match backend {
            Ok(b) => {
                tracing::debug!(?mode, "audio backend ready");
                Self {
                    backend: Some(Mutex::new(b)),
                    mode,
                    disabled_reason: None,
                    levels: MixerLevels::default(),
                    next_id: 0,
                    music: None,
                    ignored_calls: 0,
                    listener: None,
                    emitters: BTreeSet::new(),
                    background: false,
                    background_paused: Vec::new(),
                }
            }
            Err(err) => Self::disabled(format!("no audio output available: {err}")),
        }
    }

    fn backend(&mut self) -> Option<&mut Backend> {
        match self.backend.as_mut() {
            Some(m) => Some(m.get_mut().unwrap_or_else(|p| p.into_inner())),
            None => {
                self.ignored_calls += 1;
                None
            }
        }
    }

    fn backend_ref(&self) -> Option<std::sync::MutexGuard<'_, Backend>> {
        self.backend
            .as_ref()
            .map(|m| m.lock().unwrap_or_else(|p| p.into_inner()))
    }

    fn alloc_id(&mut self) -> SoundId {
        let id = SoundId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Current backend mode.
    pub fn mode(&self) -> AudioMode {
        self.mode
    }

    /// `true` unless in [`AudioMode::Disabled`].
    pub fn is_enabled(&self) -> bool {
        self.mode != AudioMode::Disabled
    }

    /// Why audio is disabled, if it is.
    pub fn disabled_reason(&self) -> Option<&str> {
        self.disabled_reason.as_deref()
    }

    /// Number of calls ignored because audio is disabled (diagnostics/tests).
    pub fn ignored_calls(&self) -> u64 {
        self.ignored_calls
    }

    /// Play a sound. Returns an id usable with the per-sound controls.
    pub fn play(&mut self, sound: &StaticSoundData, params: PlayParams) -> SoundId {
        let id = self.alloc_id();
        if let Some(b) = self.backend()
            && let Err(err) = b.play(id, sound, &params)
        {
            tracing::warn!(%err, "failed to play sound");
        }
        id
    }

    /// Fire-and-forget one-shot on the sfx bus.
    pub fn play_sfx(&mut self, sound: &StaticSoundData) -> SoundId {
        self.play(sound, PlayParams::sfx())
    }

    /// Decode encoded bytes (on this thread) and play them.
    pub fn play_bytes(&mut self, bytes: impl Into<Vec<u8>>, params: PlayParams) -> Result<SoundId, AudioError> {
        let data = decode(bytes)?;
        Ok(self.play(&data, params))
    }

    /// Switch background music: the current track fades out while `sound` loops in on the
    /// music bus, both over `crossfade`.
    pub fn play_music(&mut self, sound: &StaticSoundData, crossfade: impl Into<Fade>) -> SoundId {
        let fade = crossfade.into();
        self.stop_music(fade);
        let id = self.play(sound, PlayParams::music().fade_in(fade.duration));
        self.music = Some(id);
        id
    }

    /// Fade out and stop the current music, if any.
    pub fn stop_music(&mut self, fade: impl Into<Fade>) {
        if let Some(old) = self.music.take() {
            self.stop(old, fade);
        }
    }

    /// Id of the music started by [`Audio::play_music`].
    pub fn current_music(&self) -> Option<SoundId> {
        self.music
    }

    fn with_sound(&mut self, id: SoundId, f: impl FnOnce(&mut kira::sound::static_sound::StaticSoundHandle)) {
        if let Some(h) = self.backend().and_then(|b| b.sound(id)) {
            f(h);
        }
    }

    /// Stop a sound (fading out).
    pub fn stop(&mut self, id: SoundId, fade: impl Into<Fade>) {
        let t = fade.into().tween();
        self.with_sound(id, |h| h.stop(t));
    }

    /// Pause a sound (fading out).
    pub fn pause(&mut self, id: SoundId, fade: impl Into<Fade>) {
        let t = fade.into().tween();
        self.with_sound(id, |h| h.pause(t));
    }

    /// Resume a paused sound (fading in).
    pub fn resume(&mut self, id: SoundId, fade: impl Into<Fade>) {
        let t = fade.into().tween();
        self.with_sound(id, |h| h.resume(t));
    }

    /// The app goes to the background (`true`: everything that plays is paused) or comes back
    /// (`false`: what that paused plays on). [`AudioPlugin`] does this on a phone through
    /// `space3d_core::BackgroundHooks`: the sound has its own thread and would play on while
    /// the game stands still. A sound the game paused itself stays paused.
    pub fn set_background(&mut self, background: bool) {
        if self.background == background {
            return;
        }
        self.background = background;
        let mut paused = std::mem::take(&mut self.background_paused);
        if let Some(m) = self.backend.as_mut() {
            m.get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .set_background(background, &mut paused);
        }
        self.background_paused = paused;
    }

    /// Whether [`Audio::set_background`] has everything paused.
    pub fn in_background(&self) -> bool {
        self.background
    }

    /// Tween a sound's linear volume.
    pub fn set_sound_volume(&mut self, id: SoundId, volume: f32, fade: impl Into<Fade>) {
        let t = fade.into().tween();
        self.with_sound(id, |h| h.set_volume(amplitude_to_db(volume), t));
    }

    /// Tween a sound's pan (-1..=1).
    pub fn set_sound_pan(&mut self, id: SoundId, pan: f32, fade: impl Into<Fade>) {
        let t = fade.into().tween();
        let pan = if pan.is_nan() { 0.0 } else { pan.clamp(-1.0, 1.0) };
        self.with_sound(id, |h| h.set_panning(pan, t));
    }

    /// Tween a sound's playback rate.
    pub fn set_sound_pitch(&mut self, id: SoundId, pitch: f64, fade: impl Into<Fade>) {
        let t = fade.into().tween();
        self.with_sound(id, |h| h.set_playback_rate(pitch, t));
    }

    /// Playback state (`None` if unknown, finished and collected, or audio is disabled).
    pub fn sound_state(&self, id: SoundId) -> Option<PlaybackState> {
        self.backend_ref()?.state(id)
    }

    /// Number of tracked (not yet finished) sounds.
    pub fn active_sounds(&self) -> usize {
        self.backend_ref().map_or(0, |b| b.sound_count())
    }

    /// Bus levels.
    pub fn levels(&self) -> MixerLevels {
        self.levels
    }

    /// Replace all bus levels at once.
    pub fn set_levels(&mut self, levels: MixerLevels, fade: impl Into<Fade>) {
        self.levels = levels;
        let fade = fade.into();
        if let Some(m) = self.backend.as_mut() {
            m.get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .apply_levels(&levels, fade);
        }
    }

    /// Linear volume of one bus (not multiplied by master).
    pub fn bus_volume(&self, bus: Bus) -> f32 {
        self.levels.get(bus)
    }

    /// Tween one bus volume (linear).
    pub fn set_bus_volume(&mut self, bus: Bus, volume: f32, fade: impl Into<Fade>) {
        let mut l = self.levels;
        l.set(bus, volume);
        self.set_levels(l, fade);
    }

    /// Audible gain of a bus (`bus × master`, 0 when muted).
    pub fn effective_volume(&self, bus: Bus) -> f32 {
        self.levels.effective(bus)
    }

    /// Mute/unmute everything.
    pub fn set_muted(&mut self, muted: bool) {
        let mut l = self.levels;
        l.muted = muted;
        self.set_levels(l, Fade::IMMEDIATE);
    }

    /// Whether output is muted.
    pub fn is_muted(&self) -> bool {
        self.levels.muted
    }

    /// Listener pose used at the last sync.
    pub fn listener(&self) -> Option<ListenerPose> {
        self.listener
    }

    /// Emitter entities seen at the last sync.
    pub fn emitters(&self) -> impl Iterator<Item = Entity> + '_ {
        self.emitters.iter().copied()
    }

    /// Number of emitters seen at the last sync.
    pub fn emitter_count(&self) -> usize {
        self.emitters.len()
    }

    /// Number of live kira spatial tracks (0 when disabled).
    pub fn backend_emitter_tracks(&self) -> usize {
        self.backend_ref().map_or(0, |b| b.emitter_track_count())
    }

    /// [`AudioMode::Mock`] only: run one kira render cycle so queued commands, fades and
    /// resource removals are processed. No-op otherwise.
    pub fn pump_mock(&mut self) {
        if let Some(m) = self.backend.as_mut() {
            let b = m.get_mut().unwrap_or_else(|p| p.into_inner());
            if b.is_mock() {
                b.pump_mock();
            }
        }
    }

    fn sync(&mut self, listener: Option<ListenerPose>, frame: Vec<spatial::EmitterFrame>) {
        self.listener = listener;
        self.emitters = frame.iter().map(|f| f.entity).collect();
        match self.backend.as_mut() {
            Some(m) => {
                let b = m.get_mut().unwrap_or_else(|p| p.into_inner());
                b.sync_spatial(listener, frame);
                b.collect_finished();
            }
            None => {
                if frame.iter().any(|f| !f.plays.is_empty()) {
                    self.ignored_calls += 1;
                }
            }
        }
    }
}

/// `PostUpdate` system installed by [`AudioPlugin`]: syncs listener/emitters from
/// `GlobalTransform`, starts queued emitter sounds, drops finished sound handles.
pub fn sync_audio(ctx: &mut SystemCtx) {
    let listener = find_listener(ctx.world);
    let frame = spatial::collect_emitters(ctx.world, listener);
    if let Some(audio) = ctx.resources.get_mut::<Audio>() {
        audio.sync(listener, frame);
    }
}

/// Inserts [`Audio`] and the `PostUpdate` sync system.
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioPlugin {
    /// Initial configuration.
    pub settings: AudioSettings,
}

impl AudioPlugin {
    /// Plugin that never opens a device (tests, servers).
    pub fn disabled() -> Self {
        Self {
            settings: AudioSettings {
                backend: AudioBackend::Disabled,
                ..Default::default()
            },
        }
    }

    /// Plugin using the kira mock backend.
    pub fn mock() -> Self {
        Self {
            settings: AudioSettings {
                backend: AudioBackend::Mock,
                ..Default::default()
            },
        }
    }
}

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        let settings = self.settings.with_env_override();
        app.insert_resource(Audio::new(&settings));
        // A phone app in the background stands still, the sound's thread does not.
        app.resources
            .init::<space3d_core::BackgroundHooks>()
            .add(|resources, background| {
                if let Some(audio) = resources.get_mut::<Audio>() {
                    audio.set_background(background);
                }
            });
        app.add_system_named(Stage::PostUpdate, "audio::sync", sync_audio);
    }
}

/// Short fade helper: `fade_ms(250)`.
pub fn fade_ms(ms: u64) -> Fade {
    Fade::new(Duration::from_millis(ms))
}
