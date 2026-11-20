//! kira-backed implementation. Everything here only runs when audio is enabled.
use crate::SoundId;
use crate::spatial::{EmitterFrame, ListenerPose};
use crate::volume::{Bus, Fade, MixerLevels, PlayParams, amplitude_to_db};
use kira::backend::mock::MockBackend;
use kira::listener::ListenerHandle;
use kira::sound::PlaybackState;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::track::{SpatialTrackBuilder, SpatialTrackDistances, SpatialTrackHandle, TrackBuilder, TrackHandle};
use kira::{AudioManager, AudioManagerSettings, DefaultBackend, Tween};
use space3d_core::glam::{Quat, Vec3};
use space3d_core::hecs::Entity;
use std::collections::{HashMap, HashSet};

pub(crate) enum Manager {
    Device(Box<AudioManager<DefaultBackend>>),
    Mock(Box<AudioManager<MockBackend>>),
}

macro_rules! with_manager {
    ($m:expr, $x:ident => $body:expr) => {
        match $m {
            Manager::Device($x) => $body,
            Manager::Mock($x) => $body,
        }
    };
}

struct EmitterTrack {
    handle: SpatialTrackHandle,
    volume: f32,
    strength: f32,
}

pub(crate) struct Backend {
    manager: Manager,
    music: TrackHandle,
    sfx: TrackHandle,
    listener: ListenerHandle,
    sounds: HashMap<SoundId, StaticSoundHandle>,
    emitters: HashMap<Entity, EmitterTrack>,
}

impl Backend {
    /// Open the default output device.
    pub fn device() -> Result<Self, String> {
        let m = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).map_err(|e| e.to_string())?;
        Self::with_manager(Manager::Device(Box::new(m)))
    }

    /// Deviceless kira backend (renders nothing unless pumped) — for tests.
    pub fn mock() -> Result<Self, String> {
        let m = AudioManager::<MockBackend>::new(AudioManagerSettings::default())
            .map_err(|()| "mock backend failed".to_string())?;
        Self::with_manager(Manager::Mock(Box::new(m)))
    }

    fn with_manager(mut manager: Manager) -> Result<Self, String> {
        let (music, sfx, listener) = with_manager!(&mut manager, m => {
            let music = m.add_sub_track(TrackBuilder::new()).map_err(|e| e.to_string())?;
            let sfx = m.add_sub_track(TrackBuilder::new()).map_err(|e| e.to_string())?;
            let listener = m.add_listener(Vec3::ZERO, Quat::IDENTITY).map_err(|e| e.to_string())?;
            (music, sfx, listener)
        });
        Ok(Self {
            manager,
            music,
            sfx,
            listener,
            sounds: HashMap::new(),
            emitters: HashMap::new(),
        })
    }

    pub fn is_mock(&self) -> bool {
        matches!(self.manager, Manager::Mock(_))
    }

    pub fn apply_levels(&mut self, levels: &MixerLevels, fade: Fade) {
        let t = fade.tween();
        with_manager!(&mut self.manager, m => m.main_track().set_volume(levels.master_db(), t));
        self.music.set_volume(amplitude_to_db(levels.music), t);
        self.sfx.set_volume(amplitude_to_db(levels.sfx), t);
    }

    pub fn play(&mut self, id: SoundId, data: &StaticSoundData, params: &PlayParams) -> Result<(), String> {
        let d = params.apply(data);
        let handle = match params.bus {
            Bus::Master => with_manager!(&mut self.manager, m => m.play(d)),
            Bus::Music => self.music.play(d),
            Bus::Sfx => self.sfx.play(d),
        }
        .map_err(|e| e.to_string())?;
        self.sounds.insert(id, handle);
        Ok(())
    }

    pub fn sound(&mut self, id: SoundId) -> Option<&mut StaticSoundHandle> {
        self.sounds.get_mut(&id)
    }

    pub fn state(&self, id: SoundId) -> Option<PlaybackState> {
        self.sounds.get(&id).map(|h| h.state())
    }

    pub fn sound_count(&self) -> usize {
        self.sounds.len()
    }

    pub fn emitter_track_count(&self) -> usize {
        self.emitters.len()
    }

    /// Pause everything that plays (`true`) or resume what this paused (`false`): the two
    /// buses, every emitter's track and the sounds on the master bus. `paused` is the list
    /// of sounds this paused itself; a sound the game had paused stays paused.
    pub fn set_background(&mut self, background: bool, paused: &mut Vec<SoundId>) {
        let t = Tween {
            duration: std::time::Duration::from_millis(30),
            ..Default::default()
        };
        if background {
            self.music.pause(t);
            self.sfx.pause(t);
            for e in self.emitters.values_mut() {
                e.handle.pause(t);
            }
            for (id, h) in &mut self.sounds {
                if h.state() == PlaybackState::Playing {
                    h.pause(t);
                    paused.push(*id);
                }
            }
        } else {
            self.music.resume(t);
            self.sfx.resume(t);
            for e in self.emitters.values_mut() {
                e.handle.resume(t);
            }
            for id in paused.drain(..) {
                if let Some(h) = self.sounds.get_mut(&id) {
                    h.resume(t);
                }
            }
        }
    }

    /// Forget handles of sounds that finished.
    pub fn collect_finished(&mut self) {
        self.sounds.retain(|_, h| h.state() != PlaybackState::Stopped);
    }

    pub fn sync_spatial(&mut self, listener: Option<ListenerPose>, frame: Vec<EmitterFrame>) {
        let t = Tween::default();
        if let Some(l) = listener {
            self.listener.set_position(l.position, t);
            self.listener.set_orientation(l.orientation, t);
        }
        // Dropping a spatial track handle unloads the track and its sounds.
        let alive: HashSet<Entity> = frame.iter().map(|f| f.entity).collect();
        self.emitters.retain(|e, _| alive.contains(e));

        let listener_id = self.listener.id();
        for f in frame {
            let track = match self.emitters.entry(f.entity) {
                std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
                std::collections::hash_map::Entry::Vacant(v) => {
                    let s = f.settings;
                    let builder = SpatialTrackBuilder::new()
                        .volume(amplitude_to_db(s.volume))
                        .distances(SpatialTrackDistances {
                            min_distance: s.min_distance,
                            max_distance: s.max_distance.max(s.min_distance + f32::EPSILON),
                        })
                        .attenuation_function(s.attenuation)
                        .spatialization_strength(s.spatialization_strength.clamp(0.0, 1.0));
                    match self.sfx.add_spatial_sub_track(listener_id, f.position, builder) {
                        Ok(handle) => v.insert(EmitterTrack {
                            handle,
                            volume: s.volume,
                            strength: s.spatialization_strength,
                        }),
                        Err(err) => {
                            tracing::warn!(entity = ?f.entity, %err, "could not create spatial audio track");
                            continue;
                        }
                    }
                }
            };
            track.handle.set_position(f.position, t);
            if track.volume != f.settings.volume {
                track.volume = f.settings.volume;
                track.handle.set_volume(amplitude_to_db(f.settings.volume), t);
            }
            if track.strength != f.settings.spatialization_strength {
                track.strength = f.settings.spatialization_strength;
                track
                    .handle
                    .set_spatialization_strength(track.strength.clamp(0.0, 1.0), t);
            }
            for (data, params) in f.plays {
                if let Err(err) = track.handle.play(params.apply(&data)) {
                    tracing::warn!(entity = ?f.entity, %err, "could not play emitter sound");
                }
            }
        }
    }

    /// Mock only: run one renderer cycle so queued commands and removals are applied.
    pub fn pump_mock(&mut self) {
        if let Manager::Mock(m) = &mut self.manager {
            let b = m.backend_mut();
            b.on_start_processing();
            b.process();
        }
    }
}
