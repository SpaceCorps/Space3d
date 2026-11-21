//! A sprite layer compiled for the simulation and the instance pack: plain `Copy` data per layer
//! instance (no heap), and the capacity / life window bounds of design-particles §5.1.

use crate::curve::FloatRange;
use crate::effect::{
    EmitDirection, EmitShape, Facing, Flicker, Force, MAX_FORCES_PER_LAYER, ParticleBlend, ParticleLayer,
    SimulationSpace, ThrottleResponse,
};
use space3d_core::glam::{Vec3, Vec4};

/// Largest ring per layer instance.
pub(crate) const MAX_LAYER_CAPACITY: u32 = 65_536;
/// Default emitter speed `rate_per_distance` rings are sized for (world units per second):
/// `Spawn::max_speed`'s default. A looping instance whose emitter keeps flying faster restarts
/// with rings for its speed ([`ring_speed_for`]).
pub(crate) const V_REF: f32 = 100.0;
/// Largest speed a looping instance's rings grow to (`V_REF × 2^6`).
const MAX_RING_SPEED_LEVEL: u32 = 6;

/// The speed rings are sized for to hold a trail laid at `speed` (units/s): the layers'
/// `max_speed` `base` doubled until it covers 1.25 × `speed` (at most `base × 2^6`).
pub(crate) fn ring_speed_for(base: f32, speed: f32) -> f32 {
    let base = if base > 0.0 && base.is_finite() { base } else { V_REF };
    let mut v = base;
    for _ in 0..MAX_RING_SPEED_LEVEL {
        if v >= 1.25 * speed {
            break;
        }
        v *= 2.0;
    }
    v
}

/// Simulation and per-particle attribute parameters of one sprite layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayerSim {
    pub shape: EmitShape,
    pub direction: EmitDirection,
    pub spread: f32,
    /// `cos(spread)` (the spread cone's half-angle, at most π).
    pub spread_cos: f32,
    /// `cos(angle)` of an `EmitShape::Cone` (at most π; 1 for other shapes).
    pub cone_cos: f32,
    /// An `EmitDirection::Fixed` direction, normalized, and a basis of its normal plane for the
    /// spread (unused otherwise).
    pub fixed_dir: (Vec3, Vec3, Vec3),
    pub offset: Vec3,
    pub flatten: f32,
    pub flatten_axis: Vec3,
    pub local: bool,
    pub lifetime: (f32, f32),
    pub delay: (f32, f32),
    pub speed: (f32, f32),
    pub speed_exponent: f32,
    pub inherit: f32,
    pub drag: f32,
    pub gravity: Vec3,
    pub forces: [Option<Force>; MAX_FORCES_PER_LAYER],
    pub world_forces: u32,
    pub size: (f32, f32),
    pub rotation: (f32, f32),
    pub angular: (f32, f32),
    pub mirror: bool,
    pub color_a: Vec4,
    pub color_b: Vec4,
    pub coverage: (f32, f32),
    pub flicker: Option<Flicker>,
    pub facing: Facing,
    pub blend: ParticleBlend,
    pub tinted: bool,
    /// Longest life after emission, delay included (seconds).
    pub window: f32,
    /// Bound of the particles' distance from the emitter ([`Self::reach`]).
    pub reach: Reach,
    /// Continuous emission (the render-time bridge, `Runtime::bridge`).
    pub spawn: super::spawn::Continuous,
    /// Response to the emitter's throttle.
    pub throttle: ThrottleResponse,
    /// The render instances carry the particle's age in seconds with its rotation (two
    /// halves): flipbooks played at frames per second ([`packs_age`]).
    pub pack_age: bool,
}

/// Whether a sprite layer's instances carry their age in seconds (a textured flipbook played at
/// frames per second: `FlipbookMode::{Fps, RandomStartFps}`); the rotation is then a half
/// float.
pub(crate) fn packs_age(l: &ParticleLayer) -> bool {
    use crate::effect::{FlipbookMode, SpriteShape};
    matches!(l.render.shape, SpriteShape::Texture(_))
        && matches!(
            l.render.flipbook.map(|f| f.mode),
            Some(FlipbookMode::Fps(_) | FlipbookMode::RandomStartFps(_))
        )
}

fn pair(r: FloatRange) -> (f32, f32) {
    (r.min, r.max)
}

impl LayerSim {
    pub fn new(l: &ParticleLayer) -> Self {
        let mut forces = [None; MAX_FORCES_PER_LAYER];
        for (slot, f) in forces.iter_mut().zip(&l.forces) {
            *slot = Some(*f);
        }
        // NaN spreads count as 0.
        let spread = l.spread.max(0.0);
        let facing = match l.render.facing {
            Facing::Plane { normal } => Facing::Plane {
                normal: normal.normalize_or(Vec3::Y),
            },
            f => f,
        };
        Self {
            shape: l.shape,
            direction: l.direction,
            spread,
            spread_cos: spread.min(std::f32::consts::PI).cos(),
            cone_cos: match l.shape {
                EmitShape::Cone { angle, .. } => angle.min(std::f32::consts::PI).cos(),
                _ => 1.0,
            },
            fixed_dir: {
                let d = match l.direction {
                    EmitDirection::Fixed(d) => d.normalize_or(Vec3::NEG_Z),
                    _ => Vec3::NEG_Z,
                };
                let (bx, by) = d.any_orthonormal_pair();
                (d, bx, by)
            },
            offset: l.offset,
            flatten: l.flatten.clamp(0.0, 1.0),
            flatten_axis: l.flatten_axis.normalize_or(Vec3::Y),
            local: l.space == SimulationSpace::Local,
            lifetime: pair(l.lifetime),
            delay: (l.particle_delay.min.max(0.0), l.particle_delay.max.max(0.0)),
            speed: pair(l.speed),
            speed_exponent: l.speed_scale_exponent,
            inherit: l.inherit_velocity,
            drag: l.drag.max(0.0),
            gravity: l.gravity,
            forces,
            world_forces: l.world_forces,
            size: pair(l.size),
            rotation: pair(l.rotation),
            angular: pair(l.angular_velocity),
            mirror: l.random_mirror,
            color_a: Vec4::from_array(l.start_color.a),
            color_b: Vec4::from_array(l.start_color.b),
            coverage: pair(l.coverage),
            flicker: l.flicker,
            facing,
            blend: l.render.blend,
            tinted: l.tinted,
            window: window_secs(l),
            reach: Reach::new(l, window_secs(l)),
            spawn: super::spawn::Continuous::new(l),
            throttle: l.throttle,
            pack_age: packs_age(l),
        }
    }

    /// The emitter throttle `rate_scale` clamped to the layer's `Spawn::max_rate_scale`.
    pub fn throttle_of(&self, rate_scale: f32) -> f32 {
        rate_scale.clamp(0.0, self.spawn.max_rate_scale.max(0.0))
    }

    /// Initial speed multiplier of the throttle `rate_scale` (`ThrottleResponse::speed`).
    pub fn speed_throttle(&self, rate_scale: f32) -> f32 {
        ThrottleResponse::factor(self.throttle_of(rate_scale), self.throttle.speed)
    }

    /// `(size, brightness)` multipliers of the throttle `rate_scale`.
    pub fn draw_throttle(&self, rate_scale: f32) -> (f32, f32) {
        let t = self.throttle_of(rate_scale);
        (
            ThrottleResponse::factor(t, self.throttle.size),
            ThrottleResponse::factor(t, self.throttle.intensity),
        )
    }
}

/// The parts of [`LayerSim::reach`] that do not depend on the instance (computed once).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Reach {
    /// Spawn shape extent plus the offset (× effect scale).
    shape: f32,
    /// Largest initial speed (× `scale^speed_exponent`).
    speed: f32,
    /// Distance covered per unit of initial speed over the life window (with drag).
    time: f32,
    /// Distance covered under gravity and the layer's forces over the life window (infinite
    /// with world attractors: they can be anywhere).
    accel: f32,
}

impl Reach {
    fn new(l: &ParticleLayer, window: f32) -> Self {
        let shape = match l.shape {
            EmitShape::Point => 0.0,
            EmitShape::Sphere { radius, .. } => radius.abs(),
            EmitShape::Cone { radius, .. } => radius.abs(),
            EmitShape::Ring { radius, width, .. } => radius.abs() + 0.5 * width.abs(),
            EmitShape::Box { half_extents } => half_extents.abs().length(),
            EmitShape::Line { length } => 0.5 * length.abs(),
        } + l.offset.length();
        // One tick more than the life window: extrapolation and the sub-tick pre-aging.
        let t = window + 0.1;
        let drag = l.drag.max(0.0);
        let time = if drag > 1e-6 {
            (1.0 - (-drag * t).exp()) / drag
        } else {
            t
        };
        let mut accel = l.gravity.length();
        for f in &l.forces {
            accel += match *f {
                // `attract` caps the inverse square at 1 / 0.01.
                Force::Attractor { strength, falloff, .. } => {
                    strength.abs()
                        * if matches!(falloff, crate::effect::Falloff::InverseSquare) {
                            100.0
                        } else {
                            1.0
                        }
                }
                Force::Vortex { strength, pull, .. } => strength.abs() + pull.abs(),
                Force::Turbulence { strength, .. } => strength.abs() * 3f32.sqrt(),
                // (each component of the curl is within 3: `cpu::curl_noise`)
                Force::Curl { strength, .. } => strength.abs() * 3.0 * 3f32.sqrt(),
            };
        }
        let accel = if l.world_forces != 0 {
            f32::INFINITY
        } else {
            0.5 * accel * t * t
        };
        // The throttle may speed particles up (`ThrottleResponse::speed` above a throttle of 1).
        let throttle = ThrottleResponse::factor(l.spawn.max_rate_scale.max(1.0), l.throttle.speed).max(1.0);
        Self {
            shape,
            speed: l.speed.min.abs().max(l.speed.max.abs()) * throttle,
            time,
            accel,
        }
    }
}

impl LayerSim {
    /// Upper bound of how far from its emitter a particle of this layer gets (world units, the
    /// sprite not included), for an effect `scale` and an emitter moving at most at
    /// `emitter_speed` (world-space layers inherit part of it): the spawn shape's extent plus
    /// the displacement over the whole life window under the initial speed (with drag), gravity
    /// and the layer's forces. Infinite with world attractors (they can be anywhere).
    /// Design-particles §5.1 "extent R".
    pub fn reach(&self, scale: f32, emitter_speed: f32) -> f32 {
        let r = &self.reach;
        let scale = scale.abs();
        let mut speed = r.speed;
        if self.speed_exponent != 0.0 {
            speed *= scale.powf(self.speed_exponent);
        }
        if !self.local {
            speed += self.inherit.abs() * emitter_speed;
        }
        let out = r.shape * scale + speed * r.time + r.accel;
        if out.is_finite() { out } else { f32::INFINITY }
    }
}

/// Longest particle life after emission, delay included.
pub(crate) fn window_secs(l: &ParticleLayer) -> f32 {
    l.lifetime.max.max(l.lifetime.min).max(0.0) + l.particle_delay.max.max(0.0)
}

/// Ticks after its last emission by which every particle of a layer is dead:
/// `ceil(window / dt) + 1` (the particle emitted latest in a tick lives at most
/// `ceil(window / dt)` more ticks; one more absorbs the rounding of the ages' f32 sums).
pub(crate) fn window_ticks(window: f32, dt: f32) -> u32 {
    if dt <= 0.0 {
        return u32::MAX;
    }
    ((window.max(0.0) as f64 / dt as f64).ceil().min(u32::MAX as f64 / 2.0) as u32) + 1
}

/// Ring size of a layer instance (design-particles §5.1): continuous rate × the rate at the
/// largest throttle (`max_rate_scale^ThrottleResponse::rate`, at least 1) ×
/// window × 1.05, the bursts that can overlap within the window, the distance rate at 100 u/s,
/// clamped to 1..=65 536; `ParticleLayer::capacity` overrides it.
///
/// Burst cycles are counted per window, which assumes a burst `interval` of at least one tick:
/// a shorter interval fires several cycles in every tick (all of them spawn), so the ring is
/// undersized and overwrites its oldest particles (the ring's fallback); set
/// `ParticleLayer::capacity` for such a layer.
pub(crate) fn capacity(l: &ParticleLayer) -> u32 {
    capacity_at(l, 1.0)
}

/// [`capacity`] at quality `q` (0..=1): the rates × `q` and every burst's count scaled the way
/// the spawn scales it (`spawn::scaled_count`: at least 1). `capacity_at(l, 1.0) ==
/// capacity(l)`; an explicit `ParticleLayer::capacity` scales too.
pub(crate) fn capacity_at(l: &ParticleLayer, q: f32) -> u32 {
    capacity_for(l, q, 0.0)
}

/// [`capacity_at`] with the `rate_per_distance` term sized for an emitter flying at `speed`
/// (units/s) when that exceeds the layer's `Spawn::max_speed` (the default 100: [`V_REF`]).
pub(crate) fn capacity_for(l: &ParticleLayer, q: f32, speed: f32) -> u32 {
    let q = q.clamp(0.0, 1.0);
    if let Some(c) = l.capacity {
        let c = if q < 1.0 { (c as f32 * q).ceil() as u32 } else { c };
        return c.clamp(1, MAX_LAYER_CAPACITY);
    }
    let win = window_secs(l);
    let s = &l.spawn;
    // The rates at the largest throttle (`ThrottleResponse::rate`), never below 1×.
    let scale = crate::effect::ThrottleResponse::factor(s.max_rate_scale.max(1.0), l.throttle.rate).max(1.0) * q;
    let mut n = 0.0f64;
    if s.rate > 0.0 && scale > 0.0 {
        n += (s.rate as f64 * scale as f64 * win as f64 * 1.05).ceil() + 1.0;
    }
    if s.rate_per_distance > 0.0 && scale > 0.0 {
        let v = if s.max_speed > 0.0 && s.max_speed.is_finite() {
            s.max_speed
        } else {
            V_REF
        }
        .max(speed);
        n += (s.rate_per_distance as f64 * scale as f64 * v as f64 * win as f64).ceil() + 1.0;
    }
    for b in &s.bursts {
        let overlapping = if b.interval > 0.0 {
            let within = (win / b.interval).floor() as u64 + 1;
            if b.cycles == 0 {
                within
            } else {
                within.min(b.cycles as u64)
            }
        } else {
            b.cycles.max(1) as u64
        };
        let count = crate::runtime::spawn::scaled_count(b.count.max.max(b.count.min), q);
        n += (count as u64 * overlapping) as f64;
    }
    (n.min(MAX_LAYER_CAPACITY as f64) as u32).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::Burst;

    #[test]
    fn capacities_cover_rates_and_bursts() {
        let l = ParticleLayer::continuous(60.0, (0.5, 1.0));
        assert_eq!(capacity(&l), (60.0f64 * 1.0 * 1.05).ceil() as u32 + 1);
        let mut l = ParticleLayer::burst((10, 20), 0.5);
        assert_eq!(capacity(&l), 20);
        l.spawn.bursts = vec![Burst {
            count: 5.into(),
            cycles: 0,
            interval: 0.2,
            ..Burst::default()
        }];
        assert_eq!(capacity(&l), 5 * 3, "cycles every 0.2 s overlapping a 0.5 s life");
        l.capacity = Some(7);
        assert_eq!(capacity(&l), 7);
        l.capacity = Some(1 << 20);
        assert_eq!(capacity(&l), MAX_LAYER_CAPACITY);
        assert_eq!(capacity(&ParticleLayer::default()), 1, "never zero");
        assert_eq!(window_ticks(0.5, 1.0 / 60.0), 31);
        assert_eq!(window_ticks(0.51, 1.0 / 60.0), 32);
        assert_eq!(window_ticks(0.0, 1.0 / 60.0), 1);
        assert_eq!(window_ticks(1.0, 0.0), u32::MAX);
    }

    /// game#10: distance rings hold `rate_per_distance × max_speed × life` (default 100 units/s),
    /// a faster `max_speed` or measured speed sizes them bigger.
    #[test]
    fn distance_rings_follow_the_speed() {
        let mut l = ParticleLayer::continuous(0.0, 0.1);
        l.spawn.rate_per_distance = 1.2;
        l.spawn.max_rate_scale = 2.0;
        let win = window_secs(&l) as f64;
        let at = |v: f64| (1.2 * 2.0 * v * win).ceil() as u32 + 1;
        assert_eq!(capacity(&l), at(100.0));
        assert_eq!(capacity_for(&l, 1.0, 800.0), at(800.0));
        assert_eq!(capacity_for(&l, 1.0, 50.0), at(100.0), "never below max_speed");
        l.spawn.max_speed = 600.0;
        assert_eq!(capacity(&l), at(600.0));
        assert!(capacity(&l) >= 144, "the 600 u/s repro's 144 particles");
        assert_eq!(ring_speed_for(100.0, 600.0), 800.0);
        assert_eq!(ring_speed_for(100.0, 70.0), 100.0);
        assert_eq!(ring_speed_for(0.0, 1e9), 6400.0);
    }
}
