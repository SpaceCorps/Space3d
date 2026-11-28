//! The city's particle effects, authored as RON (units are meters).

/// Steam rising from a manhole or a roof vent: warm-lit soft puffs that billow and thin out.
pub const STEAM: &str = r#"
ParticleEffect(
    name: "city/steam",
    looping: true,
    layers: [(
        name: "plume",
        spawn: (rate: 9.0),
        shape: Sphere(radius: 0.3, inner: 0.0),
        direction: Fixed((0.0, 1.0, 0.0)),
        spread: 0.3,
        speed: (1.0, 2.0),
        drag: 0.5,
        gravity: (0.0, 0.35, 0.0),
        forces: [Turbulence(strength: 0.6, frequency: 0.5, speed: 0.3)],
        lifetime: (4.0, 6.5),
        size: (0.7, 1.2),
        size_over_life: Linear(0.5, 2.6),
        rotation: (0.0, 6.28),
        angular_velocity: (-0.25, 0.25),
        color_over_life: (keys: [
            (0.0, (1.5, 1.05, 0.72, 1.0)),
            (0.5, (1.05, 0.78, 0.62, 1.0)),
            (1.0, (0.6, 0.5, 0.5, 1.0)),
        ]),
        alpha_over_life: Keys([(0.0, 0.0), (0.08, 0.16), (0.5, 0.1), (1.0, 0.0)]),
        render: (shape: Smoke(breakup: 0.65, hot_core: 0.0), blend: Alpha),
    )],
)
"#;

/// A drizzle of fine streaks around the camera (the emitter is parented to it).
pub const RAIN: &str = r#"
ParticleEffect(
    name: "city/rain",
    looping: true,
    layers: [(
        name: "drops",
        spawn: (rate: 4200.0),
        shape: Box(half_extents: (22.0, 0.5, 22.0)),
        offset: (0.0, 15.0, 0.0),
        direction: Fixed((0.05, -1.0, 0.02)),
        speed: (14.0, 18.0),
        lifetime: (1.0, 1.2),
        size: 0.025,
        color_over_life: (keys: [(0.0, (0.8, 0.85, 1.0, 1.0))]),
        alpha_over_life: Keys([(0.0, 0.0), (0.1, 0.22), (0.9, 0.22), (1.0, 0.0)]),
        render: (
            shape: Streak(core: 0.4, taper: 0.5),
            facing: VelocityStretched(scale: 0.02, min: 1.0, max: 80.0),
        ),
    )],
)
"#;

/// A veil of low mist drifting at street level around the camera.
pub const MIST: &str = r#"
ParticleEffect(
    name: "city/mist",
    looping: true,
    layers: [(
        name: "veil",
        spawn: (rate: 7.0),
        shape: Box(half_extents: (60.0, 0.5, 60.0)),
        offset: (0.0, 2.0, 0.0),
        direction: Fixed((1.0, 0.0, 0.2)),
        speed: (0.4, 1.0),
        lifetime: (9.0, 13.0),
        size: (14.0, 24.0),
        size_over_life: Linear(0.8, 1.3),
        rotation: (0.0, 6.28),
        angular_velocity: (-0.05, 0.05),
        color_over_life: (keys: [(0.0, (0.95, 0.62, 0.42, 1.0)), (1.0, (0.7, 0.55, 0.55, 1.0))]),
        alpha_over_life: Keys([(0.0, 0.0), (0.2, 0.055), (0.8, 0.055), (1.0, 0.0)]),
        render: (shape: Smoke(breakup: 0.5, hot_core: 0.0), blend: Alpha),
    )],
)
"#;
