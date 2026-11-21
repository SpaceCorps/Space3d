//! Runtime + CPU simulation tests without a GPU (design-particles §13 unit tests).

use super::*;
use crate::components::{EffectSpawn, SpawnEffect};
use crate::cpu::{CpuSim, Particle};
use crate::curve::FloatRange;
use crate::effect::{Burst, EmitShape, ParticleLayer, Spawn};
use proptest::prelude::*;
use space3d_core::glam::Vec3;

const DT: f32 = 1.0 / 60.0;

/// A bolt trail of sprite glows laid per unit of distance (`LASER_TRAIL` before it became a
/// ribbon): the render-time bridges' heaviest client.
const SPRITE_TRAIL: &str = r#"ParticleEffect(
    name: "test/sprite_trail",
    looping: true,
    layers: [
        (
            name: "trail",
            spawn: (rate_per_distance: 1.0),
            lifetime: 0.15,
            size: 4.0,
            color_over_life: (keys: [(0.0, (5.0, 5.0, 5.0, 1.0)), (1.0, (0.6, 0.6, 0.6, 1.0))]),
            alpha_over_life: Preset(Out),
            capacity: Some(576),
            render: (shape: Glow(core: 0.35, hardness: 0.0)),
        ),
        (
            name: "head",
            space: Local,
            spawn: (rate: 60.0),
            lifetime: 0.05,
            size: 6.0,
            color_over_life: (keys: [(0.0, (8.0, 8.0, 8.0, 1.0))]),
            alpha_over_life: Preset(InOut),
            render: (shape: Glow(core: 0.6, hardness: 0.0)),
        ),
    ],
)"#;

fn sprite_trail() -> ParticleEffect {
    ParticleEffect::from_ron(SPRITE_TRAIL).expect("sprite trail")
}

struct Harness {
    world: World,
    effects: ParticleEffects,
    settings: ParticleSettings,
    rt: Runtime,
    sim: CpuSim,
}

impl Harness {
    fn new() -> Self {
        Self {
            world: World::new(),
            effects: ParticleEffects::default(),
            settings: ParticleSettings::default(),
            rt: Runtime::default(),
            sim: CpuSim::default(),
        }
    }
    fn add(&mut self, e: ParticleEffect) -> EffectHandle {
        self.effects.add(e).unwrap()
    }
    fn tick(&mut self) {
        self.rt.tick(&self.world, &self.settings, &self.effects, DT);
    }
    /// Tick and replay (a renderer drawing every tick).
    fn step(&mut self) {
        self.tick();
        self.sim.replay(&mut self.rt, self.settings.max_catchup_ticks);
    }
    fn alive(&self) -> Vec<(usize, Particle)> {
        self.sim
            .pool
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, p)| p.alive() && p.pos_age[3] < p.vel_life[3])
            .collect()
    }
}

fn burst(count: u32, life: f32) -> ParticleEffect {
    ParticleEffect {
        duration: 0.0,
        ..ParticleEffect::new("burst").layer(ParticleLayer::burst(count, life).speed((1.0, 2.0)))
    }
}

/// A burst of life 0.5 s emitted on tick 1 finishes on tick 1 + ceil(0.5 · 60) + 1 = 32,
/// exactly, and no particle is alive then.
#[test]
fn finish_on_the_exact_tick() {
    let mut h = Harness::new();
    let fx = h.add(burst(10, 0.5));
    let e = h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO));
    for t in 1..=40u64 {
        h.step();
        let done = h.rt.finished.iter().any(|f| f.entity == e);
        assert_eq!(done, t == 32, "tick {t}");
        if t == 1 {
            assert_eq!(h.alive().len(), 10);
        }
        if t == 29 {
            assert_eq!(h.alive().len(), 10, "alive until their life ends");
        }
        if t >= 32 {
            assert!(h.alive().is_empty(), "tick {t}");
            let s = h.rt.state(e).unwrap();
            assert!(s.finished && !s.emitting && s.reserved == 0);
        }
    }
    assert_eq!(h.rt.pool.reserved(), 0, "the ring was returned");
}

/// Particle `k` of a layer instance lands in slot `base + k mod capacity`, with a 64-bit spawn
/// index.
#[test]
fn ring_slots_follow_spawn_indices() {
    let mut h = Harness::new();
    let mut layer = ParticleLayer::continuous(600.0, 1.0);
    layer.capacity = Some(16);
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("ring").layer(layer)
    });
    h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
    h.step();
    let e = h.rt.entries.iter().position(|e| e.live).unwrap();
    // Jump the spawn index across 2³²: slots keep following index mod capacity.
    h.rt.entries[e].spawned = (1u64 << 32) - 3;
    h.step();
    let entry = h.rt.entries[e];
    assert_eq!(entry.spawned, (1u64 << 32) - 3 + 10, "600/s at 60 Hz: 10 per tick");
    for index in entry.spawned - 10..entry.spawned {
        let slot = entry.base as usize + (index % 16) as usize;
        let p = h.sim.pool[slot];
        assert!(p.alive());
        assert_eq!(p.misc[1], crate::rng::particle_seed(entry.seed, index), "index {index}");
    }
    assert!(h.sim.overwrites > 0, "a 16-slot ring at 600/s × 1 s overwrites");
}

/// An emitter whose parent disappears stops emitting where it was, drains and finishes.
#[test]
fn orphaned_emitters_stop_drain_and_finish() {
    let mut h = Harness::new();
    let mut layer = ParticleLayer::continuous(60.0, 0.25);
    layer.speed = FloatRange::constant(1.0);
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("trail").layer(layer)
    });
    let ship = h.world.spawn((Transform::from_xyz(5.0, 0.0, 0.0),));
    let e = h
        .world
        .spawn_effect(fx, EffectSpawn::at(Vec3::new(0.0, 0.0, 1.0)).follow(ship));
    for _ in 0..10 {
        h.step();
    }
    assert_eq!(
        h.rt.instances[h.rt.by_entity[&e] as usize].pos,
        Vec3::new(5.0, 0.0, 1.0)
    );
    h.world.despawn(ship).unwrap();
    let spawned = h.rt.spawned;
    let mut finished_at = None;
    for t in 0..40 {
        h.step();
        assert_eq!(h.rt.spawned, spawned, "orphans do not emit");
        if h.rt.finished.iter().any(|f| f.entity == e) {
            finished_at = Some(t);
            break;
        }
    }
    let t = finished_at.expect("the orphan finished");
    assert_eq!(
        t + 1,
        window_ticks(0.25, DT) as usize,
        "one window after its last emission"
    );
    assert!(h.alive().is_empty());
}

/// Removing the emitter drains (`OnRemove::Drain`, no event) or clears (`OnRemove::Clear`, the
/// ring is zeroed at that tick).
#[test]
fn removal_drains_or_clears() {
    for clear in [false, true] {
        let mut h = Harness::new();
        let fx = h.add(ParticleEffect {
            looping: true,
            ..ParticleEffect::new("loop").layer(ParticleLayer::continuous(120.0, 1.0))
        });
        let e = h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
        if clear {
            h.world.get::<&mut ParticleEmitter>(e).unwrap().on_remove = OnRemove::Clear;
        }
        for _ in 0..30 {
            h.step();
        }
        let alive = h.alive().len();
        assert!(alive >= 59, "{alive}");
        h.world.despawn(e).unwrap();
        h.step();
        assert!(h.rt.state(e).is_none(), "detached from the entity");
        if clear {
            assert!(h.alive().is_empty(), "cleared at once");
            assert_eq!(h.rt.pool.reserved(), 0);
        } else {
            assert_eq!(h.alive().len(), alive, "draining: nothing emitted, nothing died yet");
            for _ in 0..70 {
                h.step();
                assert!(h.rt.finished.is_empty(), "no event without an entity");
            }
            assert!(h.alive().is_empty());
            assert_eq!(h.rt.pool.reserved(), 0, "freed after draining");
        }
    }
}

/// The emitter frame comes from `Transform`s through the `Parent` chain (exact per tick).
#[test]
fn world_matrix_through_parent_chains() {
    let mut h = Harness::new();
    let fx = h.add(burst(1, 1.0));
    let root = h.world.spawn((Transform::from_xyz(10.0, 0.0, 0.0)
        .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
        .with_scale(Vec3::splat(2.0)),));
    let mid = h.world.spawn((Transform::from_xyz(0.0, 1.0, 0.0), Parent(root)));
    let e = h.world.spawn_effect(
        fx,
        EffectSpawn::from_transform(Transform::from_xyz(0.0, 0.0, -5.0)).follow(mid),
    );
    h.tick();
    let inst = h.rt.instances[h.rt.by_entity[&e] as usize];
    // (0, 1, −5) scaled by 2 and turned 90° about Y: (−10, 2, 0), then + (10, 0, 0).
    assert!((inst.pos - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4, "{}", inst.pos);
    assert!((inst.rot * Vec3::NEG_Z - Vec3::NEG_X).length() < 1e-5);
    let row = h.rt.log.rows.iter().find(|r| r.instance == h.rt.by_entity[&e]).unwrap();
    assert_eq!(row.pos, inst.pos);
}

/// Without a consumer the log keeps only the catch-up window; replaying it from an empty pool
/// gives exactly the pool of a replay of every tick.
#[test]
fn catch_up_reset_equals_full_replay() {
    let setup = || {
        let mut h = Harness::new();
        let mut plume = ParticleLayer::continuous(90.0, (0.2, 0.4));
        plume.shape = EmitShape::Cone {
            angle: 0.3,
            radius: 0.5,
        };
        plume.speed = FloatRange::new(5.0, 9.0);
        plume.drag = 2.0;
        plume.gravity = Vec3::new(0.0, -3.0, 0.0);
        plume.spawn.rate_per_distance = 2.0;
        let looping = h.add(ParticleEffect {
            looping: true,
            ..ParticleEffect::new("plume").layer(plume)
        });
        let pops = h.add(ParticleEffect {
            duration: 0.0,
            ..ParticleEffect::new("pop").layer(ParticleLayer {
                shape: EmitShape::Sphere {
                    radius: 1.0,
                    inner: 0.0,
                },
                speed: FloatRange::new(2.0, 6.0),
                particle_delay: FloatRange::new(0.0, 0.1),
                ..ParticleLayer::burst((5, 9), (0.3, 0.6))
            })
        });
        let e = h.world.spawn_effect(looping, EffectSpawn::at(Vec3::ZERO).keep());
        (h, e, pops)
    };
    let drive = |h: &mut Harness, e: Entity, pops: EffectHandle, t: u32| {
        h.world.get::<&mut Transform>(e).unwrap().translation =
            Vec3::new((t as f32 * 0.05).sin() * 4.0, 0.0, t as f32 * 0.02);
        if t.is_multiple_of(17) {
            h.world
                .spawn_effect(pops, EffectSpawn::at(Vec3::new(t as f32 * 0.1, 1.0, 0.0)));
        }
    };
    let (mut every, e1, p1) = setup();
    let (mut once, e2, p2) = setup();
    for t in 0..300 {
        drive(&mut every, e1, p1, t);
        every.step();
        drive(&mut once, e2, p2, t);
        once.tick();
        let window = once
            .rt
            .entries
            .iter()
            .filter(|e| e.live)
            .map(|e| window_ticks(e.sim.window, DT))
            .max()
            .unwrap();
        assert!(once.rt.log.frames.len() <= window as usize + 1, "bounded log");
    }
    assert!(once.rt.log.reset, "the log was trimmed");
    once.sim.replay(&mut once.rt, 1200);
    assert_eq!(once.sim.resets, 1);
    // Entry indices (`misc[2]`) are internal handles whose reuse follows the replays; the
    // particles themselves are identical.
    let strip = |v: Vec<(usize, Particle)>| {
        v.into_iter()
            .map(|(slot, mut p)| {
                p.misc[2] = 0;
                (slot, p)
            })
            .collect::<Vec<_>>()
    };
    let (a, b) = (strip(every.alive()), strip(once.alive()));
    assert!(a.len() > 20, "{}", a.len());
    assert_eq!(a, b, "identical alive sets, slots and state");
}

/// Prewarm synthesizes the past emission: a looping 60/s effect prewarmed by 1 s starts full.
#[test]
fn prewarm_fills_the_pool_at_once() {
    let mut h = Harness::new();
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("ambient").layer(ParticleLayer::continuous(60.0, 1.0).speed(1.0))
    });
    h.world
        .spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).prewarm(1.0).keep());
    h.step();
    let alive = h.alive();
    assert!((58..=61).contains(&alive.len()), "{}", alive.len());
    let (lo, hi) = alive
        .iter()
        .map(|(_, p)| p.pos_age[3])
        .fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
    assert!(lo < 0.05 && hi > 0.9, "ages spread over the prewarm: {lo}..{hi}");
}

/// Over the budget, rings shrink (to 1/8) before they are refused; low priorities stop at
/// 75 % of the pool.
#[test]
fn budget_shrinks_then_refuses() {
    let mut h = Harness::new();
    h.settings.max_particles_cpu = 100;
    let mut layer = ParticleLayer::burst(1, 1.0);
    layer.capacity = Some(80);
    let fx = h.add(ParticleEffect::new("big").layer(layer.clone()));
    let low = h.add(ParticleEffect {
        priority: crate::effect::EffectPriority::Low,
        ..ParticleEffect::new("low").layer(layer)
    });
    h.world.spawn_effect(fx, Vec3::ZERO);
    h.tick();
    assert_eq!((h.rt.pool.reserved(), h.rt.shrunk, h.rt.refused), (80, 0, 0));
    h.world.spawn_effect(fx, Vec3::ZERO);
    h.tick();
    assert_eq!((h.rt.pool.reserved(), h.rt.shrunk), (90, 1), "shrunk to 80/8 = 10");
    h.world.spawn_effect(low, Vec3::ZERO);
    h.tick();
    assert_eq!(h.rt.refused, 1, "past 75 % for low priority");
    assert_eq!(h.rt.pool.reserved(), 90);
}

/// A layer's `min_quality` leaves it out of instances whose quality × LOD level is below it
/// (neither reserved nor refused: a detail the tier drops), and its `priority` overrides the
/// effect's: a `Normal` layer of a `Low` effect is still reserved past 75 % of the pool, where
/// the effect's other layer is refused (M2 final review: engine plumes outlast the details).
#[test]
fn layer_min_quality_and_priority() {
    use crate::effect::EffectPriority;
    let ring = |capacity: u32| {
        let mut l = ParticleLayer::burst(1, 1.0);
        l.capacity = Some(capacity);
        l
    };
    let mut h = Harness::new();
    let detail = ParticleLayer {
        min_quality: 0.3,
        ..ring(10)
    };
    let fx = h.add(ParticleEffect::new("engine").layer(ring(10)).layer(detail));
    h.settings.quality = 0.25;
    let low = h.world.spawn_effect(fx, Vec3::ZERO);
    h.tick();
    assert_eq!(h.rt.state(low).unwrap().reserved, 3, "10 × 0.25, the detail left out");
    assert_eq!((h.rt.refused, h.rt.shrunk), (0, 0));
    h.settings.quality = 1.0;
    let full = h.world.spawn_effect(fx, Vec3::ZERO);
    h.tick();
    assert_eq!(h.rt.state(full).unwrap().reserved, 20, "both layers at quality 1");

    let mut h = Harness::new();
    h.settings.max_particles_cpu = 100;
    let filler = h.add(ParticleEffect {
        priority: EffectPriority::High,
        ..ParticleEffect::new("filler").layer(ring(80))
    });
    let plume = ParticleLayer {
        priority: Some(EffectPriority::Normal),
        ..ring(5)
    };
    let engine = h.add(ParticleEffect {
        priority: EffectPriority::Low,
        ..ParticleEffect::new("engine").layer(plume).layer(ring(5))
    });
    h.world.spawn_effect(filler, Vec3::ZERO);
    h.tick();
    let e = h.world.spawn_effect(engine, Vec3::ZERO);
    h.tick();
    assert_eq!(h.rt.state(e).unwrap().reserved, 5, "the Normal layer, to 85 %");
    assert_eq!(h.rt.refused, 1, "the Low layer, past 75 %");
    assert!(
        ParticleEffect::new("bad")
            .layer(ParticleLayer {
                min_quality: 1.5,
                ..ring(1)
            })
            .validate()
            .is_err()
    );
}

/// Restarting (bumping `restart`) or stopping and playing again start the effect over; the old
/// particles drain.
#[test]
fn restart_and_play_again() {
    let mut h = Harness::new();
    let fx = h.add(burst(4, 0.2));
    let e = h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
    h.step();
    assert_eq!(h.rt.spawned, 4);
    h.world.get::<&mut ParticleEmitter>(e).unwrap().restart();
    h.step();
    assert_eq!(h.rt.spawned, 8, "restarted");
    assert_eq!(h.alive().len(), 8, "the old burst drains");
    for _ in 0..20 {
        h.step();
    }
    assert!(h.rt.state(e).unwrap().finished);
    h.world.get::<&mut ParticleEmitter>(e).unwrap().stop();
    h.step();
    h.world.get::<&mut ParticleEmitter>(e).unwrap().playing = true;
    h.step();
    assert_eq!(h.rt.spawned, 12, "played again after it finished");
}

/// Delayed particles age hidden in place; gravity and drag integrate like the closed form.
#[test]
fn delays_gravity_and_drag() {
    let mut h = Harness::new();
    let fx = h.add(ParticleEffect {
        duration: 0.0,
        ..ParticleEffect::new("fall").layer(ParticleLayer {
            particle_delay: FloatRange::constant(0.1),
            gravity: Vec3::new(0.0, -10.0, 0.0),
            drag: 0.5,
            ..ParticleLayer::burst(1, 2.0)
        })
    });
    h.world.spawn_effect(fx, Vec3::ZERO);
    for _ in 0..5 {
        h.step();
        let (_, p) = h.alive()[0];
        assert!(p.pos_age[3] < 0.0 && p.pos() == Vec3::ZERO, "hidden while delayed");
    }
    for _ in 0..66 {
        h.step();
    }
    let (_, p) = h.alive()[0];
    let t = p.pos_age[3];
    // v(t) = −g(1 − e^{−kt})/k, falling from rest.
    let v = -10.0 * (1.0 - (-0.5 * t).exp()) / 0.5;
    assert!((p.vel().y - v).abs() < 0.05 * v.abs(), "{} vs {v}", p.vel().y);
    assert!(p.pos().y < 0.0 && p.pos().x == 0.0);
}

fn capacity_case(
    rate: f32,
    count: u32,
    cycles: u32,
    interval: f32,
    life: (f32, f32),
    delay: (f32, f32),
    looping: bool,
) -> Result<(), proptest::test_runner::TestCaseError> {
    let mut h = Harness::new();
    let layer = ParticleLayer {
        spawn: Spawn {
            rate,
            bursts: vec![Burst {
                time: 0.05,
                count: (count / 2, count).into(),
                cycles,
                interval,
            }],
            ..Spawn::default()
        },
        lifetime: FloatRange::new(life.0, life.0 + life.1),
        particle_delay: FloatRange::new(delay.0, delay.0 + delay.1),
        ..ParticleLayer::default()
    };
    // The ring is clamped to `MAX_LAYER_CAPACITY` by design: a layer that needs more alive at
    // once than that overwrites its oldest particles (see `capacity_clamp_overwrites_by_design`).
    prop_assume!(super::layer::capacity(&layer) < super::layer::MAX_LAYER_CAPACITY);
    let fx = h.add(ParticleEffect {
        looping,
        duration: 1.0,
        ..ParticleEffect::new("p").layer(layer)
    });
    h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
    for _ in 0..150 {
        h.step();
    }
    prop_assert_eq!(h.sim.overwrites, 0);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]
    /// The computed ring capacity holds every particle alive at once (no early overwrite) for
    /// random rates, bursts, lifetimes and delays.
    #[test]
    fn capacity_bound_holds(
        rate in 0.0f32..400.0,
        count in 0u32..40,
        cycles in 0u32..4,
        interval in 0.0f32..0.4,
        life in (0.02f32..1.5, 0.0f32..1.0),
        delay in (0.0f32..0.3, 0.0f32..0.3),
        looping in any::<bool>(),
    ) {
        capacity_case(rate, count, cycles, interval, life, delay, looping)?;
    }
}

/// Input the property found on CI (ubuntu, Windows): bursts of up to 24 every 0.28 ms for a
/// window of 1.04 s need about 90 000 live particles, more than the largest ring: the layer's
/// capacity is clamped, and when the clamp applies the ring overwrites its oldest particles
/// rather than growing without bound. (The property only claims "no overwrite" below the clamp.)
#[test]
fn capacity_clamp_overwrites_by_design() {
    let mut h = Harness::new();
    let layer = ParticleLayer {
        spawn: Spawn {
            rate: 0.0,
            bursts: vec![Burst {
                time: 0.05,
                count: (12, 24).into(),
                cycles: 0,
                interval: 0.00027557957,
            }],
            ..Spawn::default()
        },
        lifetime: FloatRange::new(0.02, 0.02 + 0.83847463),
        particle_delay: FloatRange::new(0.17784142, 0.17784142),
        ..ParticleLayer::default()
    };
    assert_eq!(
        super::layer::capacity(&layer),
        super::layer::MAX_LAYER_CAPACITY,
        "clamped"
    );
    let fx = h.add(ParticleEffect {
        looping: false,
        duration: 1.0,
        ..ParticleEffect::new("p").layer(layer)
    });
    h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
    for _ in 0..150 {
        h.step();
    }
    assert!(h.sim.overwrites > 0, "the clamp is what overwrites");
    assert!(h.alive().len() <= super::layer::MAX_LAYER_CAPACITY as usize);
}

#[allow(clippy::too_many_arguments)]
fn entry_bounds_case(
    shape: usize,
    speed: (f32, f32),
    drag: f32,
    gravity: f32,
    force: usize,
    strength: f32,
    local: bool,
    inherit: f32,
    delay: f32,
    scale: f32,
    spin: f32,
) -> Result<(), proptest::test_runner::TestCaseError> {
    let mut h = Harness::new();
    let shape = [
        EmitShape::Point,
        EmitShape::Sphere {
            radius: 5.0,
            inner: 0.2,
        },
        EmitShape::Cone {
            angle: 0.5,
            radius: 2.0,
        },
        EmitShape::Ring {
            radius: 6.0,
            width: 2.0,
            arc: 0.0,
        },
        EmitShape::Box {
            half_extents: Vec3::new(3.0, 1.0, 4.0),
        },
        EmitShape::Line { length: 10.0 },
    ][shape];
    let forces = match force {
        0 => vec![],
        1 => vec![crate::effect::Force::Attractor {
            position: Vec3::new(2.0, 0.0, 0.0),
            strength,
            radius: 0.0,
            falloff: crate::effect::Falloff::InverseSquare,
        }],
        2 => vec![crate::effect::Force::Vortex {
            axis: Vec3::Y,
            strength,
            pull: strength * 0.3,
        }],
        _ => vec![crate::effect::Force::Turbulence {
            strength,
            frequency: 0.4,
            speed: 1.0,
        }],
    };
    let layer = ParticleLayer {
        shape,
        speed: FloatRange::new(speed.0, speed.0 + speed.1),
        drag,
        gravity: Vec3::new(0.0, gravity, 0.0),
        forces,
        inherit_velocity: inherit,
        particle_delay: FloatRange::new(0.0, delay),
        space: if local {
            crate::effect::SimulationSpace::Local
        } else {
            crate::effect::SimulationSpace::World
        },
        ..ParticleLayer::continuous(120.0, (0.3, 1.2))
    };
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("b").layer(layer)
    });
    let e = h
        .world
        .spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).scale(scale).keep());
    for t in 0..150 {
        let a = t as f32 * spin;
        if let Ok(mut tr) = h.world.get::<&mut Transform>(e) {
            tr.translation = Vec3::new(a.sin() * 40.0, 0.0, t as f32 * 0.8);
            tr.rotation = Quat::from_rotation_y(a * 3.0);
        }
        h.step();
        let poses: Vec<(Vec3, Quat)> = h.rt.instances.iter().map(|i| (i.pos, i.rot)).collect();
        for (i, entry) in h.rt.entries.iter().enumerate().filter(|(_, e)| e.live) {
            let (lo, hi) = h.rt.entry_bounds(entry, &poses, 0.0);
            let (pp, pr) = poses[entry.instance as usize];
            for (slot, p) in h.alive() {
                if p.misc[2] != i as u32 || p.pos_age[3] < 0.0 {
                    continue;
                }
                let x = if local { pp + pr * p.pos() } else { p.pos() };
                prop_assert!(
                    x.cmpge(lo - Vec3::splat(1e-3)).all() && x.cmple(hi + Vec3::splat(1e-3)).all(),
                    "tick {t} slot {slot}: {x} outside {lo}..{hi}"
                );
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    /// The conservative bounds of a layer instance (the GPU backend's culling) contain every
    /// particle it has alive, for random shapes, speeds, drag, gravity, forces, delays, spaces
    /// and a moving, turning emitter.
    #[test]
    fn entry_bounds_contain_every_particle(
        shape in 0usize..6,
        speed in (0.0f32..30.0, 0.0f32..20.0),
        drag in 0.0f32..3.0,
        gravity in -20.0f32..20.0,
        force in 0usize..4,
        strength in 0.0f32..30.0,
        local in any::<bool>(),
        inherit in 0.0f32..1.0,
        delay in 0.0f32..0.4,
        scale in 0.2f32..3.0,
        spin in 0.0f32..0.2,
    ) {
        entry_bounds_case(shape, speed, drag, gravity, force, strength, local, inherit, delay, scale, spin)?;
    }
}

/// Failing input found by the property above on CI (ubuntu): a point emitter at rest with a
/// tiny drag was born 47 m away (`pre_age` lost its position to `f32` cancellation).
#[test]
fn entry_bounds_regression_point_emitter_at_rest() {
    entry_bounds_case(
        0,
        (0.0, 0.0),
        5.5877004e-5,
        -8.845601,
        0,
        0.0,
        false,
        0.0,
        0.0,
        0.2,
        0.0,
    )
    .unwrap();
}

/// Median of `n` timed runs of `f`, in milliseconds.
fn median_ms(n: usize, mut f: impl FnMut()) -> f64 {
    let mut samples: Vec<f64> = (0..n)
        .map(|_| {
            let t = std::time::Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[n / 2]
}

/// The budget multiplier: `SPACE3D_PERF_BUDGET_SCALE`, ×4 for the unoptimized-crate test
/// profile (the gates hold for release builds: `cargo test --release`).
fn budget_scale() -> f64 {
    let env = std::env::var("SPACE3D_PERF_BUDGET_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    env * if cfg!(debug_assertions) { 4.0 } else { 1.0 }
}

/// Graphics spec §4.4 gate: the CPU backend simulates and packs 16k particles in ≤ 0.35 ms per
/// tick on one thread (Dev machine).
#[test]
fn cpu_backend_16k_within_budget() {
    let mut h = Harness::new();
    h.settings.max_particles_cpu = 32_768;
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("stress").layer(ParticleLayer {
            shape: EmitShape::Sphere {
                radius: 20.0,
                inner: 0.0,
            },
            speed: FloatRange::new(5.0, 20.0),
            drag: 1.0,
            gravity: Vec3::new(0.0, -2.0, 0.0),
            capacity: Some(17_000),
            ..ParticleLayer::continuous(16_384.0, 1.0)
        })
    });
    h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
    for _ in 0..70 {
        h.step();
    }
    let poses = vec![(Vec3::ZERO, Quat::IDENTITY); h.rt.instances.len()];
    let (mut keys, mut out, mut items) = (Vec::new(), Vec::new(), Vec::new());
    let layer = crate::cpu::PackLayer {
        id: 0,
        batched: true,
        extent: 0.5,
    };
    let mut alive = 0;
    let ms = median_ms(60, || {
        h.tick();
        h.sim.replay(&mut h.rt, 1200);
        crate::cpu::pack(
            &crate::cpu::PackInput {
                rt: &h.rt,
                pool: &h.sim.pool,
                poses: &poses,
                extrapolate: 0.0,
            },
            &|_| Some(layer),
            &mut keys,
            &mut out,
            &mut items,
        );
        alive = out.len();
    });
    let budget = 0.35 * budget_scale();
    eprintln!("CPU backend: {alive} particles, tick + simulate + pack {ms:.3} ms (budget {budget:.2} ms)");
    assert!(alive > 16_000, "{alive}");
    assert!(ms <= budget, "{ms:.3} ms > {budget:.2} ms");
}

/// Graphics spec §4.4: `tick_system` ≤ 60 µs per tick at 500 layer instances (100 effects × 5
/// continuous layers, moving emitters).
#[test]
fn tick_500_layer_instances_within_budget() {
    let mut h = Harness::new();
    let mut effect = ParticleEffect {
        looping: true,
        ..ParticleEffect::new("five")
    };
    for _ in 0..5 {
        effect = effect.layer(ParticleLayer::continuous(20.0, 0.5));
    }
    let fx = h.add(effect);
    let emitters: Vec<Entity> = (0..100)
        .map(|i| {
            h.world
                .spawn_effect(fx, EffectSpawn::at(Vec3::new(i as f32, 0.0, 0.0)).keep())
        })
        .collect();
    let mut t = 0.0f32;
    let mut run = |h: &mut Harness| {
        t += 0.01;
        for (i, &e) in emitters.iter().enumerate() {
            h.world.get::<&mut Transform>(e).unwrap().translation.z = (t + i as f32).sin();
        }
        h.tick();
        h.rt.log.clear(); // consumed (the backend's replay is timed separately)
    };
    for _ in 0..60 {
        run(&mut h);
    }
    assert_eq!(h.rt.counts(), (100, 500));
    let ms = median_ms(100, || run(&mut h));
    let budget = 0.06 * budget_scale();
    eprintln!(
        "tick system: 500 layer instances in {:.1} µs (budget {:.0} µs)",
        ms * 1e3,
        budget * 1e3
    );
    assert!(ms <= budget, "{ms:.4} ms > {budget:.3} ms");
}

/// Per-layer forces (attractor, vortex, turbulence) and world attractors act on the particles.
#[test]
fn forces_pull_swirl_and_stir() {
    use crate::components::ParticleAttractor;
    use crate::effect::{Falloff, Force};
    let run = |forces: Vec<Force>, world_forces: u32, attractor_mask: u32| {
        let mut h = Harness::new();
        let fx = h.add(ParticleEffect {
            duration: 0.0,
            ..ParticleEffect::new("forces").layer(ParticleLayer {
                shape: EmitShape::Sphere {
                    radius: 1.0,
                    inner: 0.0,
                },
                forces,
                world_forces,
                ..ParticleLayer::burst(20, 5.0)
            })
        });
        h.world.spawn((
            Transform::from_xyz(0.0, 10.0, 0.0),
            ParticleAttractor {
                strength: 20.0,
                radius: 0.0,
                falloff: Falloff::Constant,
                mask: attractor_mask,
            },
        ));
        h.world.spawn_effect(fx, EffectSpawn::at(Vec3::new(5.0, 0.0, 0.0)));
        for _ in 0..30 {
            h.step();
        }
        h.alive().into_iter().map(|(_, p)| p).collect::<Vec<_>>()
    };
    let mean = |ps: &[Particle]| ps.iter().map(|p| p.pos()).sum::<Vec3>() / ps.len() as f32;
    let still = run(vec![], 0, 1);
    assert!(
        (mean(&still) - Vec3::new(5.0, 0.0, 0.0)).length() < 0.5,
        "no force: stays"
    );
    // An attractor at emitter-local (10, 0, 0): world (15, 0, 0).
    let pulled = run(
        vec![Force::Attractor {
            position: Vec3::new(10.0, 0.0, 0.0),
            strength: 20.0,
            radius: 0.0,
            falloff: Falloff::Constant,
        }],
        0,
        1,
    );
    assert!(mean(&pulled).x > 7.0, "pulled toward +X: {}", mean(&pulled));
    // A vortex about Y: angular momentum about the emitter's axis.
    let swirl = run(
        vec![Force::Vortex {
            axis: Vec3::Y,
            strength: 10.0,
            pull: 0.0,
        }],
        0,
        1,
    );
    let spin: f32 = swirl
        .iter()
        .map(|p| (p.pos() - Vec3::new(5.0, 0.0, 0.0)).cross(p.vel()).y)
        .sum();
    assert!(spin > 1.0, "{spin}");
    // Turbulence stirs them, deterministically.
    let turb = || {
        run(
            vec![Force::Turbulence {
                strength: 30.0,
                frequency: 0.5,
                speed: 1.0,
            }],
            0,
            1,
        )
    };
    let (a, b) = (turb(), turb());
    assert_eq!(a, b);
    let moved: f32 = a.iter().zip(&still).map(|(p, q)| (p.pos() - q.pos()).length()).sum();
    assert!(moved > 1.0, "{moved}");
    // Curl noise stirs them too, deterministically: two runs agree and the particles have moved.
    let curl = || {
        run(
            vec![Force::Curl {
                strength: 30.0,
                frequency: 0.5,
                speed: 1.0,
            }],
            0,
            1,
        )
    };
    let (a, b) = (curl(), curl());
    assert_eq!(a, b);
    let moved: f32 = a.iter().zip(&still).map(|(p, q)| (p.pos() - q.pos()).length()).sum();
    assert!(moved > 1.0, "{moved}");
    // A world attractor (entity at y = 10) acts on layers whose mask matches.
    assert!(mean(&run(vec![], 1, 1)).y > 1.0, "attracted up");
    assert!(mean(&run(vec![], 1, 2)).y.abs() < 0.5, "mask mismatch: no effect");
}

/// Distance LOD: full rate up to `near`, `min_scale` from `far`.
#[test]
fn lod_scales_with_camera_distance() {
    let lod = crate::ParticleLod {
        near: 100.0,
        far: 300.0,
        min_scale: 0.25,
    };
    assert_eq!(lod_scale(&lod, 50.0), 1.0);
    assert_eq!(lod_scale(&lod, 200.0), 0.625);
    assert_eq!(lod_scale(&lod, 1000.0), 0.25);
    assert_eq!(lod_scale(&crate::ParticleLod::default(), 1e6), 1.0, "off by default");
}

/// Ring levels: `q / 2^k`, the smallest of q … q/16 holding quality × LOD, 0 for nothing.
#[test]
fn ring_levels_quantize_quality_times_lod() {
    assert_eq!(ring_level(1.0, 1.0), 1.0);
    assert_eq!(ring_level(1.0, 0.6), 1.0);
    assert_eq!(ring_level(1.0, 0.5), 0.5);
    assert_eq!(ring_level(1.0, 0.49), 0.5);
    assert_eq!(ring_level(0.25, 0.25), 0.25);
    assert_eq!(ring_level(0.25, 0.1), 0.125);
    assert_eq!(ring_level(1.0, 0.001), 1.0 / 16.0, "at most four levels down");
    assert_eq!(ring_level(1.0, 0.0), 0.0);
    assert_eq!(ring_level(0.0, 1.0), 0.0);
    // Quality scales every `lod` ring (bursts keep at least one particle).
    let exhaust = crate::registry::builtin::effect(crate::registry::builtin::ENGINE_EXHAUST).unwrap();
    let full: u32 = exhaust.layers.iter().map(capacity).sum();
    let quarter: u32 = exhaust.layers.iter().map(|l| layer::capacity_at(l, 0.25)).sum();
    assert_eq!(full, 171);
    assert_eq!(quarter, 49);
    let explosion = crate::registry::builtin::effect(crate::registry::builtin::EXPLOSION).unwrap();
    for l in &explosion.layers {
        assert!(layer::capacity_at(l, 0.01) >= 1);
        assert_eq!(layer::capacity_at(l, 1.0), capacity(l));
    }
}

/// A camera 400 above the origin looking down (fov 60°, 16:9), and a looping one-layer effect
/// (60/s for 0.5 s: 33 slots at full quality).
fn lod_scene(h: &mut Harness) -> EffectHandle {
    use space3d_render::Camera3d;
    h.world.spawn((
        Transform::from_xyz(0.0, 400.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
        Camera3d::default(),
    ));
    h.rt.aspect = 16.0 / 9.0;
    h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("engine").layer(ParticleLayer::continuous(60.0, 0.5))
    })
}

/// Off screen (`offscreen_scale`), a looping emitter gets rings for its LOD level — none at
/// all at scale 0 — and restarts with full rings when it comes into view (the old instance
/// drains), then back down (after a fall of two levels) when it leaves; a raised quality
/// regrows it the same way. Its fewer particles draw bigger and brighter.
#[test]
fn offscreen_emitters_get_small_rings_and_regrow_in_view() {
    let mut h = Harness::new();
    let fx = lod_scene(&mut h);
    h.settings.offscreen_scale = 0.25;
    // 400 below the camera the view is ±231 tall and ±410 wide (×1.25 margin: ±289, ±513).
    let e = h
        .world
        .spawn_effect(fx, EffectSpawn::at(Vec3::new(800.0, 0.0, 0.0)).keep());
    h.step();
    let reserved = |h: &Harness| h.rt.state(e).unwrap().reserved;
    let full = capacity(&ParticleLayer::continuous(60.0, 0.5));
    assert_eq!(
        reserved(&h),
        layer::capacity_at(&ParticleLayer::continuous(60.0, 0.5), 0.25)
    );
    let i = h.rt.by_entity[&e];
    let entry = h.rt.entries[h.rt.instances[i as usize].entries[0] as usize];
    let (_, size) = h.rt.draw_tint_scale(&entry);
    assert!((size - 2f32.sqrt()).abs() < 1e-6, "thinned 4×: particles 1.41× as big");
    for _ in 0..30 {
        h.step();
    }
    // Into view: a full ring (the off-screen instance drains).
    h.world.get::<&mut Transform>(e).unwrap().translation.x = 300.0;
    h.step();
    assert_eq!(reserved(&h), full);
    assert_eq!(h.rt.counts().0, 2, "the old instance drains");
    let (_, size) =
        h.rt.draw_tint_scale(&h.rt.entries[h.rt.instances[h.rt.by_entity[&e] as usize].entries[0] as usize]);
    assert_eq!(size, 1.0);
    // Out again: a quarter is two levels down, so it shrinks back.
    h.world.get::<&mut Transform>(e).unwrap().translation.x = -900.0;
    h.step();
    assert!(reserved(&h) < full);
    // Culled entirely: no ring for its `lod` layers, and full ones once in view.
    h.settings.offscreen_scale = 0.0;
    let hidden = h
        .world
        .spawn_effect(fx, EffectSpawn::at(Vec3::new(0.0, 0.0, 700.0)).keep());
    h.step();
    assert_eq!(h.rt.state(hidden).unwrap().reserved, 0);
    h.world.get::<&mut Transform>(hidden).unwrap().translation.z = 0.0;
    h.step();
    assert_eq!(h.rt.state(hidden).unwrap().reserved, full);
    // Quality: a quarter, then raised back to 1 (the looping emitter regrows).
    h.settings.quality = 0.25;
    let q = h
        .world
        .spawn_effect(fx, EffectSpawn::at(Vec3::new(10.0, 0.0, 0.0)).keep());
    h.step();
    assert_eq!(
        h.rt.state(q).unwrap().reserved,
        layer::capacity_at(&ParticleLayer::continuous(60.0, 0.5), 0.25)
    );
    h.settings.quality = 1.0;
    h.step();
    assert_eq!(h.rt.state(q).unwrap().reserved, full);
    assert_eq!(h.rt.refused, 0);
}

/// Under a LOD, the emitters new in a tick start the most visible first: when the budget holds
/// only some, the ones in view get it.
#[test]
fn new_emitters_in_view_are_served_first() {
    let mut h = Harness::new();
    let fx = lod_scene(&mut h);
    h.settings.offscreen_scale = 0.5;
    let full = capacity(&ParticleLayer::continuous(60.0, 0.5));
    // Room for 4 full rings (Normal priority stops at 90 %).
    h.settings.max_particles_cpu = (full as f32 * 4.0 / 0.9).ceil() as u32 + 1;
    // Off-screen emitters spawned first (entity order), then 4 in view.
    let off: Vec<Entity> = (0..6)
        .map(|k| {
            h.world
                .spawn_effect(fx, EffectSpawn::at(Vec3::new(900.0 + 10.0 * k as f32, 0.0, 0.0)).keep())
        })
        .collect();
    let on: Vec<Entity> = (0..4)
        .map(|k| {
            h.world
                .spawn_effect(fx, EffectSpawn::at(Vec3::new(20.0 * k as f32, 0.0, 0.0)).keep())
        })
        .collect();
    h.step();
    for e in &on {
        assert_eq!(h.rt.state(*e).unwrap().reserved, full, "in view: a full ring");
    }
    assert!(off.iter().all(|e| h.rt.state(*e).unwrap().reserved < full));
}

/// A looping instance whose layer the budget refused retries it every `RETRY_TICKS` ticks and
/// gets it once there is room (the failed retries are not counted as refusals again).
#[test]
fn refused_looping_layers_retry() {
    let mut h = Harness::new();
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("engine").layer(ParticleLayer::continuous(60.0, 0.5))
    });
    let full = capacity(&ParticleLayer::continuous(60.0, 0.5));
    // The burst leaves too little for even an eighth of the ring under Normal's 90 %.
    h.settings.max_particles_cpu = full + 3;
    let hog = h.add(ParticleEffect {
        priority: EffectPriority::Critical,
        ..burst(full, 0.5)
    });
    h.world.spawn_effect(hog, EffectSpawn::at(Vec3::ZERO));
    h.step();
    let e = h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
    h.step();
    assert_eq!(
        h.rt.state(e).unwrap().reserved,
        0,
        "refused while the burst holds the pool"
    );
    let refused = h.rt.refused;
    // The burst ends after 0.5 s; the retry comes within RETRY_TICKS after that.
    for _ in 0..(32 + RETRY_TICKS) {
        h.step();
    }
    let got = h.rt.state(e).unwrap().reserved;
    assert!(
        got > 0 && got < full,
        "retried: a ring (shrunk to fit 90 % of the pool): {got}"
    );
    assert_eq!(h.rt.refused, refused, "failed retries are not counted");
    assert!(!h.alive().is_empty());
}

/// The render-time bridge of an emitter that gameplay moved after the tick (`Stage::Update`)
/// is the next tick's emission laid from the last tick's pose to the render pose: the same
/// spawn indices, count and span of the tick, and the same positions along the same motion,
/// so the bridge turns into the simulated trail without a jump. An emitter where the tick saw
/// it has none.
#[test]
fn bridge_previews_the_next_tick() {
    use crate::cpu::{BridgeInput, PackLayer, pack_bridges};
    let mut h = Harness::new();
    let mut layer = ParticleLayer::continuous(30.0, 1.0).speed(0.0);
    layer.spawn.rate_per_distance = 0.5;
    layer.capacity = Some(4096);
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("trail").layer(layer)
    });
    let bolt = h.world.spawn((Transform::IDENTITY,));
    let e = h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).follow(bolt));
    let set = |h: &mut Harness, x: f32| h.world.get::<&mut Transform>(bolt).unwrap().translation.x = x;
    for t in 1..=4 {
        set(&mut h, 10.0 * t as f32);
        h.step();
    }
    let i = h.rt.by_entity[&e] as usize;
    let ei = h.rt.instances[i].entries()[0];
    let entry = h.rt.entries[ei as usize];
    let mut poses: Vec<(Vec3, Quat)> = h.rt.instances.iter().map(|inst| (inst.pos, inst.rot)).collect();
    assert!(
        h.rt.bridge(&entry, &poses, DT).is_none(),
        "still since the tick: no bridge"
    );

    // Gameplay moves the bolt 10 units after the tick; the render pose follows.
    set(&mut h, 50.0);
    poses[i].0 = Vec3::new(50.0, 0.0, 0.0);
    let b = h.rt.bridge(&entry, &poses, DT).expect("bridge");
    let moved = vec![true; poses.len()];
    let (mut keys, mut out, mut items) = (Vec::new(), Vec::new(), Vec::new());
    let layer_of = |_: &Entry| {
        Some(PackLayer {
            id: 0,
            batched: true,
            extent: 0.5,
        })
    };
    let input = BridgeInput {
        rt: &h.rt,
        poses: &poses,
        moved: &moved,
        dt: DT,
        age: 0.0,
    };
    let n = pack_bridges(&input, &layer_of, &mut keys, &mut out, &mut items);
    // 30/s × 1/60 s (the gap is the whole motion of a tick) + 0.5 per unit × 10 units.
    assert!((5..=6).contains(&b.count), "{b:?}");
    assert_eq!((n, out.len(), items.len()), (b.count, b.count as usize, 1));
    assert_eq!((items[0].first, items[0].count), (0, b.count));
    assert_eq!(b.first, entry.spawned);
    // Laid from the tick's pose to the render pose (a particle may land exactly on the render
    // pose: the one whose accumulator crossing ends the span).
    for p in &out {
        assert!(p.pos_size[0] > 40.0 && p.pos_size[0] <= 50.0, "{:?}", p.pos_size);
    }

    // The next tick reads the moved pose and emits exactly the bridge.
    h.tick();
    let rec = *h.rt.log.records.iter().find(|r| r.entry == ei).expect("record");
    assert_eq!((rec.first, rec.count, rec.t0, rec.t1), (b.first, b.count, b.t0, b.t1));
    h.sim.replay(&mut h.rt, h.settings.max_catchup_ticks);
    let cap = entry.capacity as u64;
    for (j, p) in out.iter().enumerate() {
        let slot = entry.base as usize + ((b.first + j as u64) % cap) as usize;
        let sim = h.sim.pool[slot];
        let d = sim.pos() - Vec3::new(p.pos_size[0], p.pos_size[1], p.pos_size[2]);
        assert!(
            d.length() < 1e-4,
            "particle {j}: simulated {} bridged {:?}",
            sim.pos(),
            p.pos_size
        );
    }
    // A time rate alone bridges the share of a tick's motion the gap stands for.
    set(&mut h, 55.0);
    poses[i].0 = Vec3::new(55.0, 0.0, 0.0);
    let entry = h.rt.entries[ei as usize];
    let half = h.rt.bridge(&entry, &poses, DT).map_or(0, |b| b.count);
    assert!(
        (2..=3).contains(&half),
        "half a tick of the rate (0.25) + 0.5 × 5 units: {half}"
    );
}

/// The render-time bridges' CPU cost: 20 bolts (a sprite trail: glows per unit of distance) at
/// 3200 units/s moved
/// one 60 Hz frame (53 units) after the tick, as gameplay in `Stage::Update` does — about 1100
/// bridged glows per frame, built and packed within 0.1 ms (the graphics spec's budget for the
/// whole particles `prepare` of a Battle frame on the GPU backend; Dev machine, release).
#[test]
fn bridges_of_20_fast_bolts_within_budget() {
    use crate::cpu::{BridgeInput, PackLayer, pack_bridges};
    let mut h = Harness::new();
    h.settings.max_particles_cpu = 65_536;
    let fx = h.add(sprite_trail());
    let bolts: Vec<Entity> = (0..20)
        .map(|i| {
            let b = h.world.spawn((Transform::from_xyz(0.0, 0.0, i as f32 * 10.0),));
            h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).follow(b));
            b
        })
        .collect();
    let per_tick = 3200.0 / 60.0;
    let mut x = 0.0;
    for _ in 0..30 {
        x += per_tick;
        for &b in &bolts {
            h.world.get::<&mut Transform>(b).unwrap().translation.x = x;
        }
        h.step();
    }
    let mut poses: Vec<(Vec3, Quat)> = h.rt.instances.iter().map(|i| (i.pos, i.rot)).collect();
    for p in &mut poses {
        p.0.x += per_tick;
    }
    let moved = vec![true; poses.len()];
    let (mut keys, mut out, mut items) = (Vec::new(), Vec::new(), Vec::new());
    let layer = PackLayer {
        id: 0,
        batched: true,
        extent: 0.5,
    };
    let mut n = 0;
    let ms = median_ms(100, || {
        out.clear();
        items.clear();
        let input = BridgeInput {
            rt: &h.rt,
            poses: &poses,
            moved: &moved,
            dt: DT,
            age: 0.5 * DT,
        };
        n = pack_bridges(&input, &|_| Some(layer), &mut keys, &mut out, &mut items);
    });
    let budget = 0.1 * budget_scale();
    eprintln!(
        "bridges: {n} glows of 20 bolts in {:.1} µs (budget {:.0} µs)",
        ms * 1e3,
        budget * 1e3
    );
    assert!((20 * 50..=20 * 56).contains(&n), "{n}");
    assert!(ms <= budget, "{ms:.4} ms > {budget:.3} ms");
}

/// The render-time bridges of a Battle-sized fleet moved in `Stage::Update` (the game's
/// pattern: every moving world-space emitter is bridged every frame): 300 ships' engine
/// exhausts (`builtin::ENGINE_EXHAUST`) at 150 units/s and 20 bolts' laser trails at 3200
/// units/s, each moved one 60 Hz frame after the tick — about 1800 exhaust particles and 1060
/// trail glows, built and packed within 0.15 ms (Dev machine, release: about 0.10 ms, 0.15 ms
/// before the spawn math skipped directions at rest, libm's `sin`/`cos` and per-particle
/// constants). The render node adds the render poses, the upload and a draw: about 0.14 ms in
/// all for this fleet (docs/performance.md, "Particles").
#[test]
fn bridges_of_a_battle_fleet_within_budget() {
    use crate::cpu::{BridgeInput, PackLayer, pack_bridges};
    use crate::registry::builtin;
    let mut h = Harness::new();
    h.settings.max_particles_cpu = 262_144;
    let exhaust = h.add(builtin::effect(builtin::ENGINE_EXHAUST).expect("exhaust"));
    let trail = h.add(sprite_trail());
    // Ships fly −Z (their exhaust points +Z), bolts +X.
    let mut movers = Vec::new();
    let mut steps = std::collections::HashMap::new();
    for i in 0..300 {
        let ship = h
            .world
            .spawn((Transform::from_xyz((i % 20) as f32 * 60.0, 0.0, (i / 20) as f32 * 80.0),));
        let e = h
            .world
            .spawn_effect(exhaust, EffectSpawn::at(Vec3::new(0.0, 0.0, 3.2)).follow(ship).seed(i));
        movers.push((ship, Vec3::new(0.0, 0.0, -150.0 / 60.0)));
        steps.insert(e, Vec3::new(0.0, 0.0, -150.0 / 60.0));
    }
    for i in 0..20 {
        let bolt = h.world.spawn((Transform::from_xyz(0.0, 5.0, i as f32 * 10.0),));
        let e = h
            .world
            .spawn_effect(trail, EffectSpawn::at(Vec3::ZERO).follow(bolt).seed(1000 + i));
        movers.push((bolt, Vec3::new(3200.0 / 60.0, 0.0, 0.0)));
        steps.insert(e, Vec3::new(3200.0 / 60.0, 0.0, 0.0));
    }
    for _ in 0..40 {
        for &(e, step) in &movers {
            h.world.get::<&mut Transform>(e).unwrap().translation += step;
        }
        h.step();
    }
    // Render poses: one frame of motion past the tick (the exhausts restarted with rings for
    // their speed once: their first instances drain, standing still).
    let poses: Vec<(Vec3, Quat)> =
        h.rt.instances
            .iter()
            .map(|inst| {
                let step = if inst.attached {
                    steps.get(&inst.entity).copied()
                } else {
                    None
                };
                (inst.pos + step.unwrap_or(Vec3::ZERO), inst.rot)
            })
            .collect();
    let moved = vec![true; poses.len()];
    let (mut keys, mut out, mut items) = (Vec::new(), Vec::new(), Vec::new());
    let layer = PackLayer {
        id: 0,
        batched: true,
        extent: 0.5,
    };
    let mut n = 0;
    let ms = median_ms(100, || {
        out.clear();
        items.clear();
        let input = BridgeInput {
            rt: &h.rt,
            poses: &poses,
            moved: &moved,
            dt: DT,
            age: 0.5 * DT,
        };
        n = pack_bridges(&input, &|_| Some(layer), &mut keys, &mut out, &mut items);
    });
    let budget = 0.15 * budget_scale();
    eprintln!(
        "bridges: {n} particles of 300 exhausts and 20 bolts in {:.1} µs (budget {:.0} µs)",
        ms * 1e3,
        budget * 1e3
    );
    assert!((2500..=3100).contains(&n), "{n}");
    assert!(ms <= budget, "{ms:.4} ms > {budget:.3} ms");
}

/// Pool slots (and ribbon points) each built-in reserves per instance (the sum of its layers'
/// rings: design-particles §5.1), as documented in docs/graphics/particles.md "Budgets": on the
/// CPU backend's web budget (8192, `Normal` priority 90 %) that is, e.g., 30 engine exhausts and
/// 40 bolts' laser trails at once. A sprite trail's ring holds it at the game's 3200 units/s
/// without overwriting.
#[test]
fn builtin_slots() {
    use crate::registry::builtin;
    let table = [
        (builtin::EXPLOSION, 82, 0),
        (builtin::HIT_SPARK, 16, 0),
        (builtin::SHIELD_RIPPLE, 8, 0),
        (builtin::WARP_IN, 29, 0),
        (builtin::WARP_OUT, 29, 0),
        (builtin::ENGINE_EXHAUST, 171, 0),
        (builtin::LASER_TRAIL, 5, 64),
        (builtin::LASER_BEAM, 7, 0),
        (builtin::MISSILE_TRAIL, 56, 192),
        (builtin::DEBRIS_SPARKS, 78, 0),
        (builtin::ENGINE_AFTERBURNER, 215, 96),
        (builtin::EMP, 64, 0),
        (builtin::REPAIR_BEAM, 8, 0),
        (builtin::LASER_SHOT, 1, 0),
    ];
    assert_eq!(table.len(), builtin::ALL.len());
    for (h, want, want_points) in table {
        let fx = builtin::effect(h).unwrap();
        let slots: u32 = fx.layers.iter().filter(|l| l.enabled).map(capacity).sum();
        let points: u32 = fx.ribbons.iter().map(|r| r.max_points).sum();
        assert_eq!((slots, points), (want, want_points), "{}", fx.name);
    }
    let web = (8192.0 * priority_limit(EffectPriority::Normal)) as u32;
    assert!(
        30 * 171 + 40 * 5 <= web,
        "30 exhausts and 40 laser trails fit the web budget ({web})"
    );

    // A trail at 3200 units/s (and 3800) fills at most its ring.
    for speed in [3200.0f32, 3800.0] {
        let mut h = Harness::new();
        let fx = h.add(sprite_trail());
        let bolt = h.world.spawn((Transform::IDENTITY,));
        h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).follow(bolt));
        for t in 1..=60 {
            h.world.get::<&mut Transform>(bolt).unwrap().translation.x = speed * DT * t as f32;
            h.step();
        }
        assert_eq!(h.sim.overwrites, 0, "{speed} units/s: the ring overwrote live glows");
    }
}

/// A ribbon records a point whenever its emitter moved `min_distance` (× scale) from the newest
/// one, stamped with the end-of-tick time, the tint and the distance along the trail; the ring
/// keeps the newest `max_points`; the effect finishes exactly when the newest point has lived
/// its `lifetime` after the emitter stopped.
#[test]
fn ribbon_tracks_record_and_expire() {
    use crate::effect::RibbonLayer;
    let mut h = Harness::new();
    let fx = h.add(ParticleEffect {
        duration: 0.5,
        ..ParticleEffect::new("trail").ribbon(RibbonLayer {
            min_distance: 1.0,
            max_points: 8,
            lifetime: 0.25,
            ..RibbonLayer::default()
        })
    });
    let e = h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).scale(2.0));
    let mut last_point = 0;
    for t in 1..=60u64 {
        // 1.5 units per tick: a point every 2 ticks (min_distance 1 × scale 2).
        if t <= 30 {
            h.world.get::<&mut Transform>(e).unwrap().translation.x = 1.5 * t as f32;
        }
        h.step();
        let i = h.rt.by_entity.get(&e).copied();
        let Some(i) = i else { break };
        let inst = h.rt.instances[i as usize];
        if inst.finished {
            // The newest point (tick `last_point`) lived its 0.25 s (15 ticks) + 1.
            assert_eq!(
                t,
                last_point + window_ticks(0.25, DT) as u64,
                "finished on the exact tick"
            );
            assert_eq!(inst.n_tracks, 0, "the track is freed");
            assert_eq!(h.rt.ribbon_pool.reserved(), 0);
            return;
        }
        let tr = h.rt.tracks[inst.tracks[0] as usize];
        if tr.last_tick == t {
            last_point = t;
        }
        if t == 20 {
            // Points at ticks 1 (first sight), 3, 5, ..., 19 → 10 recorded, the ring holds 8.
            assert_eq!(tr.len, 8);
            let newest = h.rt.points[tr.newest().unwrap()];
            assert_eq!(newest.pos, Vec3::new(28.5, 0.0, 0.0));
            assert!((newest.time - 19.0 * DT as f64).abs() < 1e-9, "{}", newest.time);
            assert!(
                (newest.dist - 27.0).abs() < 1e-4,
                "distance along the trail {}",
                newest.dist
            );
            let oldest = h.rt.points[tr.slot(0)];
            assert_eq!(oldest.pos.x, 7.5, "the ring kept the newest 8");
        }
    }
    panic!("the effect never finished");
}

/// A light flash is a `PointLight` child entity of the emitter while it plays: spawned on its
/// first tick with the flash curve's intensity, updated every tick, despawned when it ends; the
/// effect finishes only after it (then despawns).
#[test]
fn light_flash_plays_a_point_light() {
    use crate::effect::LightFlash;
    use crate::{ParticleEffectFinished, ParticlesPlugin, builtin};
    use space3d_core::prelude::{App, EventReader, Events};
    use space3d_render::PointLight;
    let mut app = App::new();
    app.add_plugin(ParticlesPlugin::default());
    let fx = app
        .resources
        .expect_mut::<ParticleEffects>()
        .add(ParticleEffect {
            duration: 0.0,
            light: Some(LightFlash {
                start: 0.05,
                duration: 0.2,
                color: Vec3::new(1.0, 0.5, 0.2),
                peak: 30.0,
                range: 40.0,
                offset: Vec3::new(0.0, 5.0, 0.0),
                tinted: true,
                ..LightFlash::default()
            }),
            ..ParticleEffect::new("flash")
        })
        .unwrap();
    let e = app.spawn_effect(
        fx,
        EffectSpawn::at(Vec3::new(10.0, 0.0, 0.0))
            .scale(2.0)
            .tint(space3d_core::glam::Vec4::new(1.0, 1.0, 0.5, 1.0)),
    );
    let lights = |app: &App| {
        app.world
            .query::<(&PointLight, &Parent, &Transform)>()
            .iter()
            .map(|(l, p, t)| (*l, p.0, t.translation))
            .collect::<Vec<_>>()
    };
    let mut reader = EventReader::default();
    let (mut first, mut last) = (None, None);
    // Disabled particles light nothing: a flash under way goes out.
    {
        let mut probe = App::new();
        probe.add_plugin(ParticlesPlugin::default());
        let fx = probe
            .resources
            .expect_mut::<ParticleEffects>()
            .add(ParticleEffect {
                duration: 0.0,
                light: Some(LightFlash::default()),
                ..ParticleEffect::new("flash")
            })
            .unwrap();
        probe.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
        probe.step_ticks(2);
        assert_eq!(probe.world.query::<&PointLight>().iter().count(), 1);
        probe.resources.expect_mut::<crate::ParticleSettings>().enabled = false;
        probe.step_ticks(1);
        assert_eq!(probe.world.query::<&PointLight>().iter().count(), 0);
    }
    for t in 1..=40u32 {
        app.step_ticks(1);
        let l = lights(&app);
        let finished = reader
            .read(app.resources.expect::<Events<ParticleEffectFinished>>())
            .count();
        if let Some((light, parent, at)) = l.first() {
            first.get_or_insert(t);
            last = Some(t);
            assert_eq!(parent, &e);
            assert_eq!(at, &Vec3::new(0.0, 10.0, 0.0), "offset × scale, local to the emitter");
            assert_eq!(light.range, 80.0);
            assert_eq!(light.color, Vec3::new(1.0, 0.5, 0.1), "tinted");
            assert!(light.intensity > 0.0 && light.intensity <= 30.0 * 4.0, "peak × scale²");
            assert!(app.world.contains(e));
        }
        if finished > 0 {
            assert!(l.is_empty(), "the light went before the effect finished");
            assert!(!app.world.contains(e), "then the one-shot despawned");
            // Flash from 0.05 s (tick 3) for 0.2 s (12 ticks).
            assert_eq!((first, last), (Some(3), Some(14)), "on for the flash only");
            let _ = builtin::EXPLOSION;
            return;
        }
    }
    panic!("never finished (lights {first:?}..{last:?})");
}

/// Hot reload (`ParticleEffects::set`): a change that keeps the layers' structure updates the
/// running instance in place (same layer instances, new parameters); a structural change (a
/// layer added, a ring resized, a ribbon added) restarts the emitter as a new instance while the
/// old one's particles drain and it is freed once they are dead.
#[test]
fn hot_reload_structure_change_drains() {
    use crate::effect::RibbonLayer;
    let mut h = Harness::new();
    let layer = ParticleLayer::continuous(60.0, 0.5).speed((1.0, 2.0));
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("reload").layer(layer.clone())
    });
    let e = h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep());
    for _ in 0..30 {
        h.step();
    }
    let i = h.rt.by_entity[&e];
    let entry = h.rt.instances[i as usize].entries()[0];
    // Same structure, new color and speed: in place.
    let mut tweaked = layer.clone();
    tweaked.speed = FloatRange::new(3.0, 4.0);
    h.effects
        .set(
            fx,
            ParticleEffect {
                looping: true,
                ..ParticleEffect::new("reload").layer(tweaked.clone())
            },
        )
        .unwrap();
    h.step();
    assert_eq!(h.rt.by_entity[&e], i, "same instance");
    assert_eq!(h.rt.instances[i as usize].entries()[0], entry, "same layer instance");
    assert_eq!(h.rt.entries[entry as usize].sim.speed, (3.0, 4.0), "new parameters");
    // A second layer and a ribbon: a new instance; the old one drains.
    h.effects
        .set(
            fx,
            ParticleEffect {
                looping: true,
                ..ParticleEffect::new("reload")
                    .layer(tweaked.clone())
                    .layer(tweaked)
                    .ribbon(RibbonLayer::default())
            },
        )
        .unwrap();
    h.step();
    let j = h.rt.by_entity[&e];
    assert_ne!(j, i, "a new instance for the entity");
    assert_eq!(
        (
            h.rt.instances[j as usize].n_entries,
            h.rt.instances[j as usize].n_tracks
        ),
        (2, 1)
    );
    assert!(
        h.rt.instances[i as usize].live && !h.rt.instances[i as usize].attached,
        "draining"
    );
    assert!(
        h.alive().iter().any(|(_, p)| p.misc[2] == entry),
        "the old particles still live"
    );
    for _ in 0..window_ticks(0.5, DT) + 1 {
        h.step();
    }
    assert!(
        !h.rt.instances[i as usize].live || h.rt.instances[i as usize].entity != e,
        "drained and freed"
    );
    assert!(
        !h.alive()
            .iter()
            .any(|(_, p)| p.misc[2] == entry && h.rt.entries[entry as usize].live)
    );
    assert_eq!(h.rt.entries.iter().filter(|x| x.live).count(), 2, "only the new layers");
}

/// Emission positions of a stream on an emitter moved at 150 units/s, as the CPU backend
/// simulates them, with the app run at `fps` (App::update, 60 Hz ticks: 2 ticks per frame at
/// 30 fps, 1 or 2 at 45), the parent moved in `stage`. Returns the gaps between neighbours
/// (sorted along the motion) after the first half second.
fn stream_gaps(fps: f64, stage: space3d_core::Stage) -> Vec<f32> {
    use space3d_core::{App, Stage, SystemCtx};
    let mut app = App::new();
    app.add_plugin(crate::ParticlesPlugin::default());
    let mut layer = ParticleLayer::continuous(180.0, 5.0).speed(0.0);
    layer.lod = false;
    layer.capacity = Some(4096);
    let fx = app
        .resources
        .expect_mut::<ParticleEffects>()
        .add(ParticleEffect {
            looping: true,
            ..ParticleEffect::new("stream").layer(layer)
        })
        .unwrap();
    let ship = app.world.spawn((Transform::IDENTITY,));
    app.spawn_effect(fx, EffectSpawn::at(Vec3::new(0.0, 0.0, 3.2)).follow(ship));
    let fixed = stage == Stage::FixedUpdate;
    app.add_system(stage, move |ctx: &mut SystemCtx| {
        let dt = if fixed {
            ctx.time().fixed_delta_secs()
        } else {
            ctx.time().delta_secs()
        };
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(ship) {
            t.translation.x += 150.0 * dt;
        }
    });
    let mut sim = CpuSim::default();
    let frame = std::time::Duration::from_secs_f64(1.0 / fps);
    for _ in 0..(2.0 * fps) as usize {
        app.update(frame);
        let rt = &mut app.resources.expect_mut::<ParticleRuntime>().rt;
        sim.replay(rt, 1200);
    }
    let x_now = app.world.get::<&Transform>(ship).unwrap().translation.x;
    let mut xs: Vec<f32> = sim
        .pool
        .iter()
        .filter(|p| p.alive())
        .map(|p| p.pos().x)
        .filter(|&x| x > 75.0 && x < x_now - 10.0)
        .collect();
    xs.sort_by(f32::total_cmp);
    xs.windows(2).map(|w| w[1] - w[0]).collect()
}

/// game#2: an emitter moved in `Stage::Update` whose frames run two fixed ticks (30 fps, and
/// 45 fps: one or two) emits as evenly as one moved on the tick. Every tick used to read the
/// same `Transform`: the first laid the whole frame's motion, the second stacked its particles
/// on one point (beads one frame of motion apart). Each tick now extrapolates the pose by the
/// velocity measured between frames to the tick's time (`frame_motion_system`).
#[test]
fn update_moved_emitters_do_not_bead_at_low_frame_rates() {
    use space3d_core::Stage;
    let ideal = 150.0 / 180.0;
    for (fps, stage) in [
        (30.0, Stage::FixedUpdate),
        (30.0, Stage::Update),
        (45.0, Stage::Update),
        (60.0, Stage::Update),
        (144.0, Stage::Update),
    ] {
        let gaps = stream_gaps(fps, stage);
        let n = gaps.len() as f32;
        let mean = gaps.iter().sum::<f32>() / n;
        let sd = (gaps.iter().map(|g| (g - mean) * (g - mean)).sum::<f32>() / n).sqrt();
        let (lo, hi) = gaps.iter().fold((f32::MAX, 0.0f32), |(a, b), &g| (a.min(g), b.max(g)));
        eprintln!(
            "{fps} fps {stage:?}: {} gaps, mean {mean:.3} sd {sd:.3} min {lo:.3} max {hi:.3}",
            gaps.len()
        );
        assert!(gaps.len() > 100, "{fps} fps {stage:?}: {} gaps", gaps.len());
        assert!(
            (mean - ideal).abs() < 0.02,
            "{fps} fps {stage:?}: mean gap {mean} (ideal {ideal})"
        );
        assert!(
            sd < 0.05 * ideal && lo > 0.8 * ideal && hi < 1.2 * ideal,
            "{fps} fps {stage:?}: gaps sd {sd:.3}, {lo:.3}..{hi:.3} (ideal {ideal:.3})"
        );
    }
}

/// game#2, turning: a ship turned in `Stage::Update` at 30 fps (two ticks per frame) lays the
/// stream of an emitter 20 units off its axis along an even arc.
#[test]
fn update_turned_emitters_do_not_bead() {
    use space3d_core::{App, Stage, SystemCtx};
    let mut app = App::new();
    app.add_plugin(crate::ParticlesPlugin::default());
    let mut layer = ParticleLayer::continuous(180.0, 5.0).speed(0.0);
    layer.lod = false;
    layer.capacity = Some(4096);
    let fx = app
        .resources
        .expect_mut::<ParticleEffects>()
        .add(ParticleEffect {
            looping: true,
            ..ParticleEffect::new("stream").layer(layer)
        })
        .unwrap();
    let ship = app.world.spawn((Transform::IDENTITY,));
    app.spawn_effect(fx, EffectSpawn::at(Vec3::new(0.0, 0.0, 20.0)).follow(ship));
    app.add_system(Stage::Update, move |ctx: &mut SystemCtx| {
        let dt = ctx.time().delta_secs();
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(ship) {
            t.rotation = Quat::from_rotation_y(1.5 * dt) * t.rotation;
        }
    });
    let mut sim = CpuSim::default();
    for _ in 0..90 {
        app.update(std::time::Duration::from_secs_f64(1.0 / 30.0));
        sim.replay(&mut app.resources.expect_mut::<ParticleRuntime>().rt, 1200);
    }
    let mut angles: Vec<f32> = sim
        .pool
        .iter()
        .filter(|p| p.alive())
        .map(|p| p.pos().x.atan2(p.pos().z))
        .filter(|a| (0.5..2.5).contains(a))
        .collect();
    angles.sort_by(f32::total_cmp);
    let ideal = 1.5 / 180.0;
    for w in angles.windows(2) {
        let g = w[1] - w[0];
        assert!((g - ideal).abs() < 0.1 * ideal, "arc gap {g} (ideal {ideal})");
    }
    assert!(angles.len() > 200, "{}", angles.len());
}

/// Particles alive per `bin`-unit bin along +X (skipping the first `skip` and taking `take`).
fn bins(h: &Harness, bin: f32, skip: usize, take: usize) -> Vec<u32> {
    let mut hist = std::collections::BTreeMap::new();
    for (_, p) in h.alive() {
        *hist.entry((p.pos().x / bin).floor() as i32).or_insert(0u32) += 1;
    }
    hist.values().copied().skip(skip).take(take).collect()
}

fn sd(v: &[u32]) -> f32 {
    let n = v.len() as f32;
    let mean = v.iter().sum::<u32>() as f32 / n;
    (v.iter().map(|&c| (c as f32 - mean).powi(2)).sum::<f32>() / n).sqrt()
}

/// game#8: an emitter stepping toward 20 Hz server updates (7.5 units every third tick, 150
/// units/s on average) laid ENGINE_EXHAUST's plume rates (180/s + 0.4 per unit) in beads: 6
/// particles spread over a step and 6 stacked where it paused. With
/// `ParticleEmitter::velocity_smoothing` the time rate follows the distance of each tick
/// against the smoothed speed: an even trail, the same count.
#[test]
fn velocity_smoothing_evens_out_stepped_motion() {
    let run = |smoothing: f32| {
        let mut h = Harness::new();
        let mut layer = ParticleLayer::continuous(180.0, 3.0).speed(0.0);
        layer.spawn.rate_per_distance = 0.4;
        layer.lod = false;
        layer.capacity = Some(4096);
        let fx = h.add(ParticleEffect {
            looping: true,
            ..ParticleEffect::new("plume").layer(layer)
        });
        let e = h.world.spawn((
            Transform::IDENTITY,
            ParticleEmitter::new(fx).with_velocity_smoothing(smoothing),
        ));
        for t in 1..=150u32 {
            h.world.get::<&mut Transform>(e).unwrap().translation.x = 7.5 * (t / 3) as f32;
            h.step();
        }
        // After the smoothing settled (the first half second).
        let b = bins(&h, 2.5, 40, 48);
        (b.iter().sum::<u32>(), sd(&b), b)
    };
    let (n0, sd0, b0) = run(0.0);
    let (n1, sd1, b1) = run(0.15);
    eprintln!("steps without smoothing: {b0:?} sd {sd0:.2}\nwith 0.15 s: {b1:?} sd {sd1:.2}");
    assert!(sd0 > 1.2, "the repro beads: {b0:?}");
    assert!(sd1 < 0.6, "paced: {b1:?}");
    assert!(
        (n1 as f32 - n0 as f32).abs() < 0.05 * n0 as f32,
        "same count: {n0} vs {n1}"
    );
    // A still emitter emits its plain time rate.
    let mut h = Harness::new();
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("still").layer(ParticleLayer::continuous(60.0, 10.0))
    });
    h.world.spawn((
        Transform::IDENTITY,
        ParticleEmitter::new(fx).with_velocity_smoothing(0.15),
    ));
    for _ in 0..60 {
        h.step();
    }
    assert_eq!(h.alive().len(), 60);
}

/// game#10: a `rate_per_distance` ring is sized for `Spawn::max_speed` (100 units/s by
/// default); a looping emitter flying at 600 units/s kept only 26 of its 144 particles (a
/// 10-unit trail instead of 60). It now restarts with rings for its speed after three fast
/// ticks, and the trail is whole.
#[test]
fn distance_rings_grow_with_the_emitter_speed() {
    let mut h = Harness::new();
    let mut layer = ParticleLayer::continuous(0.0, 0.1).speed(0.0);
    layer.spawn.rate_per_distance = 1.2;
    layer.spawn.max_rate_scale = 2.0;
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("fast").layer(layer)
    });
    let e = h
        .world
        .spawn((Transform::IDENTITY, ParticleEmitter::new(fx).with_rate_scale(2.0)));
    for t in 1..=60u32 {
        h.world.get::<&mut Transform>(e).unwrap().translation.x = 10.0 * t as f32;
        h.step();
    }
    let alive = h.alive();
    let xs: Vec<f32> = alive.iter().map(|(_, p)| p.pos().x).collect();
    let span = xs.iter().copied().fold(f32::MIN, f32::max) - xs.iter().copied().fold(f32::MAX, f32::min);
    let reserved = h.rt.state(e).unwrap().reserved;
    eprintln!(
        "alive {} (expected 144), trail {span:.1} units, reserved {reserved}",
        alive.len()
    );
    assert!(alive.len() >= 138, "alive {}", alive.len());
    assert!(span > 56.0, "trail {span}");
    assert!((144..400).contains(&reserved), "rings for 800 units/s: {reserved}");
    // A single jump (a teleport) does not grow them.
    let mut h = Harness::new();
    let mut layer = ParticleLayer::continuous(0.0, 0.1).speed(0.0);
    layer.spawn.rate_per_distance = 1.2;
    let fx = h.add(ParticleEffect {
        looping: true,
        ..ParticleEffect::new("jump").layer(layer)
    });
    let e = h.world.spawn((Transform::IDENTITY, ParticleEmitter::new(fx)));
    h.step();
    let before = h.rt.state(e).unwrap().reserved;
    h.world.get::<&mut Transform>(e).unwrap().translation.x = 5000.0;
    for _ in 0..10 {
        h.step();
    }
    assert_eq!(h.rt.state(e).unwrap().reserved, before);
}

/// game#14: lowering the budget after the pool grew limits new rings at once (refused or
/// shrunk), gives the free tail back, and the stats report the budget in force.
#[test]
fn a_lowered_budget_is_enforced() {
    let mut h = Harness::new();
    h.settings.max_particles_cpu = 262_144;
    let fx = h.add(burst(400, 0.5));
    h.step();
    assert_eq!(h.rt.pool_capacity(), 262_144);
    h.settings.max_particles_cpu = 32_768;
    for _ in 0..120 {
        h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO));
    }
    h.step();
    assert_eq!(h.rt.pool_capacity(), 32_768);
    assert!(h.rt.pool.reserved() <= 32_768, "reserved {}", h.rt.pool.reserved());
    assert!(h.rt.refused + h.rt.shrunk > 0, "the budget binds");
    assert!(h.rt.pool.total() <= 32_768, "the free tail went: {}", h.rt.pool.total());
    assert!(h.rt.pool.high_water() <= 32_768);
    // Raised again: the pool grows back.
    h.settings.max_particles_cpu = 65_536;
    h.step();
    assert_eq!((h.rt.pool_capacity(), h.rt.pool.total()), (65_536, 65_536));
}

/// A map whose `capacity()` moves with its tombstones (entities spawned and despawned every
/// tick) is not a reallocation (`ParticleStats::cpu_reallocations` stays 0 in the steady state).
#[test]
fn emitter_churn_counts_no_reallocations() {
    let mut h = Harness::new();
    let fx = h.add(burst(4, 0.1));
    let mut live = std::collections::VecDeque::new();
    for t in 0..600u32 {
        for _ in 0..3 {
            live.push_back(h.world.spawn_effect(fx, EffectSpawn::at(Vec3::ZERO).keep()));
        }
        while live.len() > 30 {
            let _ = h.world.despawn(live.pop_front().unwrap());
        }
        h.step();
        if t == 300 {
            h.rt.reallocations = 0;
        }
    }
    assert_eq!(h.rt.reallocations, 0);
}

/// The built-in engine plume `fx` on a ship flying +X at 150 units/s for two seconds, `mode`
/// 0 steady, 1 in 20 Hz steps (7.5 units every third tick), 2 easing toward 20 Hz server
/// positions (half the gap per tick, as the game does), with `smoothing` (`None`: the
/// effect's default). Particles alive per 2.5-unit bin from 5 to 50 units behind the ship
/// summed over 8 seeds, and the plume length (95th percentile of the world-space particles'
/// distance behind it).
fn plume_bins(fx: EffectHandle, smoothing: Option<f32>, mode: u8) -> (Vec<u32>, f32) {
    let mut bins = vec![0u32; 18];
    let mut behind = Vec::new();
    for seed in 0..8 {
        let mut h = Harness::new();
        let fx = h.add(crate::registry::builtin::effect(fx).unwrap());
        let mut em = ParticleEmitter::new(fx).with_seed(seed);
        em.velocity_smoothing = smoothing;
        // The flame (the emitter's +Z) points to −X.
        let rot = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
        let e = h.world.spawn((Transform::IDENTITY.with_rotation(rot), em));
        let (mut eased, mut x) = (0.0f32, 0.0);
        for tick in 1..=120u32 {
            x = match mode {
                0 => tick as f32 * 2.5,
                1 => (tick / 3 * 3) as f32 * 2.5,
                _ => {
                    eased += ((tick / 3 * 3 + 3) as f32 * 2.5 - eased) * 0.5;
                    eased
                }
            };
            h.world.get::<&mut Transform>(e).unwrap().translation.x = x;
            h.step();
        }
        for (_, p) in h.alive() {
            let d = x - p.pos().x;
            // (Local-space layers keep emitter-relative positions: far out of this range.)
            if (-10.0..200.0).contains(&d) {
                behind.push(d);
            }
            let k = ((d - 5.0) / 2.5).floor();
            if (0.0..18.0).contains(&k) {
                bins[k as usize] += 1;
            }
        }
    }
    behind.sort_by(f32::total_cmp);
    (bins, behind[behind.len() * 95 / 100])
}

fn cv(v: &[u32]) -> f32 {
    sd(v) / (v.iter().sum::<u32>() as f32 / v.len() as f32).max(1e-3)
}

/// game#8 with the real plumes: a ship easing toward 20 Hz server positions swings between
/// about 65 and 260 units/s within every step. Pacing only the time rate (round 0) left
/// ENGINE_EXHAUST beaded (cv 0.54 with 0.15 s, 0.61 without): the particles of a fast tick
/// inherited 4× the velocity of a slow tick's and bunched up as they drifted. Velocity
/// smoothing now low-passes the inherited velocity too, and the engine plumes default to 0.1 s:
/// the eased plume is as even as a steady one (cv within 1.5×) and as long, and the stepped
/// one too; with smoothing off the repro still beads.
#[test]
fn engine_plumes_stay_even_when_easing_toward_20hz_updates() {
    use crate::registry::builtin;
    for fx in [builtin::ENGINE_EXHAUST, builtin::ENGINE_AFTERBURNER] {
        let name = &builtin::effect(fx).unwrap().name;
        let (steady, steady_len) = plume_bins(fx, None, 0);
        let (off, _) = plume_bins(fx, Some(0.0), 2);
        let (eased, eased_len) = plume_bins(fx, None, 2);
        let (stepped, stepped_len) = plume_bins(fx, None, 1);
        eprintln!(
            "{name}: steady {steady:?} cv {:.3} length {steady_len:.1}\n  eased, off {off:?} cv {:.3}\n  \
             eased {eased:?} cv {:.3} length {eased_len:.1}\n  stepped {stepped:?} cv {:.3} length {stepped_len:.1}",
            cv(&steady),
            cv(&off),
            cv(&eased),
            cv(&stepped),
        );
        assert!(
            cv(&off) > 2.5 * cv(&steady),
            "{name}: the repro beads without smoothing"
        );
        assert!(
            cv(&eased) < 1.5 * cv(&steady),
            "{name}: eased {eased:?} vs steady {steady:?}"
        );
        assert!(
            cv(&stepped) < 1.7 * cv(&steady),
            "{name}: stepped {stepped:?} vs steady {steady:?}"
        );
        for len in [eased_len, stepped_len] {
            assert!(
                (len / steady_len - 1.0).abs() < 0.08,
                "{name}: length {len} vs {steady_len}"
            );
        }
    }
}

/// A jump of more than 8× the smoothed speed (a teleport) restarts the velocity smoothing
/// instead of feeding it: the particles after it inherit the flight velocity, not the jump,
/// and the time rate carries on (it is not paced against the jump).
#[test]
fn velocity_smoothing_restarts_on_a_teleport() {
    let mut h = Harness::new();
    let mut layer = ParticleLayer::continuous(60.0, 1.0).speed(0.0);
    layer.inherit_velocity = 1.0;
    layer.lod = false;
    let fx = h.add(ParticleEffect {
        looping: true,
        velocity_smoothing: 0.1,
        ..ParticleEffect::new("teleport").layer(layer)
    });
    let e = h.world.spawn((Transform::IDENTITY, ParticleEmitter::new(fx)));
    let mut x = 0.0;
    for t in 1..=60u32 {
        x += if t == 30 { 1000.0 } else { 1.0 };
        h.world.get::<&mut Transform>(e).unwrap().translation.x = x;
        h.step();
    }
    let alive = h.alive();
    let fastest = alive.iter().map(|(_, p)| p.vel().length()).fold(0.0, f32::max);
    let after = alive.iter().filter(|(_, p)| p.pos().x > 1000.0).count();
    eprintln!("fastest {fastest:.1} units/s, {after} particles after the jump");
    assert!(fastest < 61.0, "a particle inherited the jump: {fastest}");
    assert!((29..=32).contains(&after), "time rate after the jump: {after}");
}
