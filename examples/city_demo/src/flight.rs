//! The cinematic flythrough: a closed Catmull-Rom spline through camera positions; the camera
//! looks a couple of seconds ahead along the path, lifted by a per-waypoint amount.

use space3d::prelude::*;

/// A waypoint: position, how far above the path ahead the camera looks (meters), and the
/// seconds of the leg to the next waypoint. The loop closes from the last to the first.
pub struct Waypoint {
    pub pos: [f32; 3],
    pub lift: f32,
    pub secs: f32,
}

const fn wp(x: f32, y: f32, z: f32, lift: f32, secs: f32) -> Waypoint {
    Waypoint {
        pos: [x, y, z],
        lift,
        secs,
    }
}

// Streets run along multiples of 75 m on both axes (x = 0 is the central avenue, heading
// north (-Z) is into the sun); blocks stand between them.
pub const WAYPOINTS: &[Waypoint] = &[
    // an establishing glide in from the south-west, high over the low-rise ring
    wp(-430.0, 200.0, 600.0, -12.0, 8.0),
    wp(-300.0, 150.0, 560.0, -10.0, 8.0),
    wp(-120.0, 95.0, 490.0, -6.0, 8.0),
    // down into the central avenue, then street level, heading north into the low sun
    wp(3.4, 28.0, 440.0, 0.0, 5.0),
    wp(3.4, 5.5, 380.0, 3.0, 6.0),
    wp(3.4, 2.6, 300.0, 5.0, 7.0),
    wp(3.4, 2.7, 200.0, 6.0, 7.0),
    wp(3.6, 3.0, 100.0, 8.0, 8.0),
    wp(3.4, 3.4, -10.0, 12.0, 7.5),
    wp(3.0, 4.0, -100.0, 14.0, 5.0),
    // west along a cross street with the sun at our back
    wp(-4.0, 4.5, -150.0, 6.0, 5.0),
    wp(-60.0, 5.0, -150.5, 5.0, 6.0),
    wp(-140.0, 6.0, -150.0, 8.0, 6.0),
    // up a cross avenue between the towers, then out over the roofs
    wp(-225.0, 25.0, -150.0, 25.0, 6.0),
    wp(-228.0, 60.0, -250.0, 20.0, 6.0),
    wp(-228.0, 150.0, -330.0, -10.0, 9.0),
    wp(-380.0, 200.0, -100.0, -15.0, 11.0),
    wp(-450.0, 200.0, 300.0, -15.0, 7.0),
];

/// Seconds of one loop.
pub fn duration() -> f32 {
    WAYPOINTS.iter().map(|w| w.secs).sum()
}

fn catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let (t2, t3) = (t * t, t * t * t);
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// The path position and the lift at `time` (seconds; the path loops).
fn point(time: f32) -> (Vec3, f32) {
    let n = WAYPOINTS.len() as i32;
    let mut t = time.rem_euclid(duration());
    let mut i = 0;
    while t > WAYPOINTS[i as usize].secs {
        t -= WAYPOINTS[i as usize].secs;
        i += 1;
    }
    let f = t / WAYPOINTS[i as usize].secs;
    // ease within the leg so the camera never lurches at a waypoint
    let f = (f * f * (3.0 - 2.0 * f)) * 0.4 + f * 0.6;
    let at = |k: i32| &WAYPOINTS[k.rem_euclid(n) as usize];
    let pos = |k: i32| Vec3::from(at(k).pos);
    let p = catmull(pos(i - 1), pos(i), pos(i + 1), pos(i + 2), f);
    let lift = at(i).lift + (at(i + 1).lift - at(i).lift) * f;
    (p, lift)
}

/// The camera position and look-at target at `time` seconds.
pub fn sample(time: f32) -> (Vec3, Vec3) {
    let (p, lift) = point(time);
    let (ahead, _) = point(time + 2.4);
    let mut target = ahead + Vec3::Y * lift;
    // never look straight along the vertical when the path climbs steeply
    if Vec2::new(target.x - p.x, target.z - p.z).length() < 5.0 {
        target += Vec3::new(0.0, 0.0, -10.0);
    }
    (p, target)
}
