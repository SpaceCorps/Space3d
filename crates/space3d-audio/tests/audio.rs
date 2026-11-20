use space3d_audio::*;
use space3d_core::prelude::*;
use std::time::Duration;

fn beep() -> StaticSoundData {
    let samples: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
    sound_from_samples(48_000, &samples)
}

/// Minimal mono 16-bit PCM WAV file.
fn wav(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut v = Vec::new();
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&1u16.to_le_bytes()); // mono
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

#[test]
fn disabled_mode_is_a_quiet_noop() {
    let mut audio = Audio::disabled("no device in CI");
    assert_eq!(audio.mode(), AudioMode::Disabled);
    assert!(!audio.is_enabled());
    assert_eq!(audio.disabled_reason(), Some("no device in CI"));

    let s = beep();
    let a = audio.play_sfx(&s);
    let b = audio.play(&s, PlayParams::sfx().volume(0.3).pan(1.0).pitch(2.0).looped());
    assert_ne!(a, b, "ids stay unique so game code needs no special casing");
    let m = audio.play_music(&s, Duration::from_millis(500));
    assert_eq!(audio.current_music(), Some(m));
    audio.set_sound_volume(a, 0.1, Fade::secs(1.0));
    audio.set_sound_pan(a, -1.0, Fade::IMMEDIATE);
    audio.set_sound_pitch(a, 0.5, Fade::IMMEDIATE);
    audio.pause(a, fade_ms(10));
    audio.resume(a, fade_ms(10));
    audio.stop(a, fade_ms(10));
    audio.stop_music(Duration::from_secs(1));
    assert_eq!(audio.current_music(), None);
    assert_eq!(audio.sound_state(a), None);
    assert_eq!(audio.active_sounds(), 0);
    assert!(audio.ignored_calls() >= 8);

    // Mixer state is still tracked for settings menus.
    audio.set_bus_volume(Bus::Master, 0.5, Fade::IMMEDIATE);
    audio.set_bus_volume(Bus::Music, 0.4, Fade::IMMEDIATE);
    assert_eq!(audio.bus_volume(Bus::Music), 0.4);
    assert!((audio.effective_volume(Bus::Music) - 0.2).abs() < 1e-6);
    audio.set_muted(true);
    assert!(audio.is_muted());
    assert_eq!(audio.effective_volume(Bus::Sfx), 0.0);
}

#[test]
fn decode_and_play_bytes() {
    let samples: Vec<i16> = (0..800).map(|i| ((i % 40) as i16 - 20) * 500).collect();
    let bytes = wav(8000, &samples);
    let data = decode(bytes.clone()).expect("valid wav");
    assert_eq!(data.sample_rate, 8000);
    assert_eq!(data.num_frames(), 800);
    assert!((data.duration().as_secs_f64() - 0.1).abs() < 1e-6);
    assert!(matches!(decode(vec![1, 2, 3, 4]), Err(AudioError::Decode(_))));

    let mut audio = Audio::disabled("test");
    assert!(audio.play_bytes(bytes, PlayParams::sfx()).is_ok());
    assert!(audio.play_bytes(b"not audio".to_vec(), PlayParams::sfx()).is_err());
}

#[test]
fn emitters_and_listener_sync_from_global_transform_headless() {
    let mut app = App::new();
    app.add_plugin(AudioPlugin::disabled());
    let listener = app
        .world
        .spawn((Transform::from_xyz(0.0, 0.0, 0.0), AudioListener::default()));
    // Inactive listener with a lower priority must be ignored.
    app.world
        .spawn((Transform::from_xyz(100.0, 0.0, 0.0), AudioListener { active: false }));
    let right = app.world.spawn((
        Transform::from_xyz(5.0, 0.0, 0.0),
        AudioEmitter::new().with_sound(beep(), PlayParams::sfx()),
    ));
    let ship = app.world.spawn((Transform::from_xyz(-10.0, 0.0, 0.0),));
    let child = app.world.spawn((
        Transform::from_xyz(0.0, 0.0, -2.0),
        Parent(ship),
        AudioEmitter::new().with_distances(1.0, 20.0),
    ));
    app.step_ticks(1);

    let audio = app.resources.expect::<Audio>();
    assert_eq!(audio.listener().map(|l| l.position), Some(Vec3::ZERO));
    assert_eq!(audio.emitter_count(), 2);
    assert!(
        audio.ignored_calls() >= 1,
        "queued emitter sound is dropped quietly when disabled"
    );

    {
        let em = app.world.get::<&AudioEmitter>(right).unwrap();
        assert_eq!(em.position(), Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(em.queued(), 0, "queue drained by the sync system");
        let mix = em.mix().unwrap();
        assert!(mix.balance() > 0.0, "source on +X is heard on the right: {mix:?}");
        assert!((mix.distance - 5.0).abs() < 1e-5);
    }
    {
        let em = app.world.get::<&AudioEmitter>(child).unwrap();
        assert_eq!(
            em.position(),
            Vec3::new(-10.0, 0.0, -2.0),
            "child emitter uses world position"
        );
        let mix = em.mix().unwrap();
        assert!(mix.balance() < 0.0);
        assert!(mix.attenuation > 0.0 && mix.attenuation < 1.0);
    }

    // Turn the listener around: left/right swap.
    app.world.get::<&mut Transform>(listener).unwrap().rotation = Quat::from_rotation_y(std::f32::consts::PI);
    app.step_ticks(1);
    assert!(app.world.get::<&AudioEmitter>(right).unwrap().mix().unwrap().balance() < 0.0);

    // Despawn / component removal is picked up.
    app.world.despawn(right).unwrap();
    app.world.remove_one::<AudioEmitter>(child).unwrap();
    app.step_ticks(1);
    assert_eq!(app.resources.expect::<Audio>().emitter_count(), 0);

    // No listener -> no mix.
    app.world.despawn(listener).unwrap();
    let e = app.world.spawn((Transform::IDENTITY, AudioEmitter::new()));
    app.step_ticks(1);
    assert!(app.resources.expect::<Audio>().listener().is_none());
    assert!(app.world.get::<&AudioEmitter>(e).unwrap().mix().is_none());
}

#[test]
fn mock_backend_plays_stops_and_cleans_up() {
    let mut app = App::new();
    app.add_plugin(AudioPlugin::mock());
    if app.resources.expect::<Audio>().mode() != AudioMode::Mock {
        eprintln!("kira mock backend unavailable; skipping");
        return;
    }
    let s = beep();
    let audio = app.resources.expect_mut::<Audio>();
    audio.set_bus_volume(Bus::Sfx, 0.5, Fade::secs(0.2));
    let one_shot = audio.play_sfx(&s);
    let music = audio.play_music(&s, Duration::from_millis(100));
    assert_eq!(audio.sound_state(one_shot), Some(PlaybackState::Playing));
    assert_eq!(audio.active_sounds(), 2);

    audio.stop(one_shot, fade_ms(10));
    for _ in 0..4 {
        audio.pump_mock();
    }
    assert_eq!(audio.sound_state(one_shot), Some(PlaybackState::Stopped));
    assert_ne!(audio.sound_state(music), Some(PlaybackState::Stopped), "music loops");

    // Finished handles are collected by the sync system.
    app.step_ticks(1);
    let audio = app.resources.expect::<Audio>();
    assert_eq!(audio.sound_state(one_shot), None);
    assert_eq!(audio.active_sounds(), 1);

    // Spatial tracks follow emitter lifetimes.
    app.world.spawn((Transform::IDENTITY, AudioListener::default()));
    let em = app.world.spawn((
        Transform::from_xyz(2.0, 0.0, 0.0),
        AudioEmitter::new().with_sound(s.clone(), PlayParams::sfx()),
    ));
    app.step_ticks(1);
    assert_eq!(app.resources.expect::<Audio>().backend_emitter_tracks(), 1);
    app.world.get::<&mut AudioEmitter>(em).unwrap().play(s);
    app.resources.expect_mut::<Audio>().pump_mock();
    app.step_ticks(1);
    app.world.despawn(em).unwrap();
    app.step_ticks(1);
    let audio = app.resources.expect_mut::<Audio>();
    audio.pump_mock();
    assert_eq!(audio.backend_emitter_tracks(), 0);
    assert_eq!(audio.emitter_count(), 0);
}

/// In the background everything that plays is paused, and comes back; a sound the game had
/// paused itself stays paused. The plugin's hook does it (`space3d_core::BackgroundHooks`).
#[test]
fn the_background_pauses_what_plays_and_the_return_resumes_it() {
    let mut app = App::new();
    app.add_plugin(AudioPlugin::mock());
    if app.resources.expect::<Audio>().mode() != AudioMode::Mock {
        eprintln!("kira mock backend unavailable; skipping");
        return;
    }
    // Looping: the mock renders minutes with every pump.
    let s = beep();
    let looping = |bus| PlayParams {
        bus,
        looping: true,
        ..Default::default()
    };
    let audio = app.resources.expect_mut::<Audio>();
    let master = audio.play(&s, looping(Bus::Master));
    let own = audio.play(&s, looping(Bus::Sfx));
    audio.pause(own, Fade::IMMEDIATE);
    for _ in 0..4 {
        audio.pump_mock();
    }
    assert_eq!(audio.sound_state(master), Some(PlaybackState::Playing));
    assert_eq!(audio.sound_state(own), Some(PlaybackState::Paused));

    space3d_core::BackgroundHooks::run(&mut app.resources, true);
    let audio = app.resources.expect_mut::<Audio>();
    assert!(audio.in_background());
    for _ in 0..8 {
        audio.pump_mock();
    }
    assert_eq!(audio.sound_state(master), Some(PlaybackState::Paused));

    space3d_core::BackgroundHooks::run(&mut app.resources, false);
    let audio = app.resources.expect_mut::<Audio>();
    assert!(!audio.in_background());
    for _ in 0..8 {
        audio.pump_mock();
    }
    assert_eq!(audio.sound_state(master), Some(PlaybackState::Playing));
    assert_eq!(
        audio.sound_state(own),
        Some(PlaybackState::Paused),
        "the game's own pause is kept"
    );
}
