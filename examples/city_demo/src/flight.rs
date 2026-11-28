//! The cinematic flythrough: a closed Catmull-Rom spline through camera positions with a second
//! spline for the point it looks at.

use space3d::prelude::*;

/// (camera position, look-at target) per waypoint; the loop closes from the last to the first.
pub const WAYPOINTS: &[([f32; 3], [f32; 3])] = &[
    // 0: an establishing shot over the south-west, looking toward the setting sun
    ([-330.0, 150.0, 520.0], [20.0, 70.0, 0.0]),
    // 1: gliding in over the low-rise ring
    ([-120.0, 85.0, 400.0], [10.0, 45.0, 60.0]),
    // 2: dropping onto the central avenue, heading north into the sun
    ([3.4, 14.0, 330.0], [0.0, 18.0, 90.0]),
    // 3: street level, wet road blazing ahead
    ([3.4, 2.4, 240.0], [-0.5, 9.0, 40.0]),
    ([3.4, 2.7, 130.0], [-1.0, 16.0, -60.0]),
    // 4: through the first intersections
    ([3.6, 3.3, 20.0], [3.0, 26.0, -140.0]),
    // 5: rising between the downtown towers, looking up
    ([5.0, 24.0, -75.0], [-12.0, 95.0, -140.0]),
    // 6: turning west along a cross street, sun at our back
    ([-20.0, 6.0, -150.0], [-200.0, 12.0, -150.0]),
    ([-130.0, 4.0, -148.0], [-300.0, 20.0, -150.0]),
    // 7: climbing over the rooftops
    ([-250.0, 55.0, -80.0], [0.0, 60.0, 0.0]),
    // 8: back out toward the start
    ([-340.0, 120.0, 150.0], [0.0, 70.0, 0.0]),
];

/// Seconds per waypoint.
pub const SECONDS_PER_LEG: f32 = 9.0;

pub fn duration() -> f32 {
    WAYPOINTS.len() as f32 * SECONDS_PER_LEG
}

fn catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let (t2, t3) = (t * t, t * t * t);
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// The camera position and look-at target at `time` seconds (the path loops).
pub fn sample(time: f32) -> (Vec3, Vec3) {
    let n = WAYPOINTS.len() as i32;
    let u = (time / SECONDS_PER_LEG).rem_euclid(n as f32);
    let i = u.floor() as i32;
    let f = u - i as f32;
    // ease within the leg so the camera never lurches at a waypoint
    let f = f * f * (3.0 - 2.0 * f) * 0.35 + f * 0.65;
    let at = |k: i32| WAYPOINTS[k.rem_euclid(n) as usize];
    let pos = |k: i32| Vec3::from(at(k).0);
    let tgt = |k: i32| Vec3::from(at(k).1);
    (
        catmull(pos(i - 1), pos(i), pos(i + 1), pos(i + 2), f),
        catmull(tgt(i - 1), tgt(i), tgt(i + 1), tgt(i + 2), f),
    )
}
