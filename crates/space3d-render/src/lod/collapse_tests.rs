//! Tests of [`simplify_edges`].

use super::*;
use glam::Vec3Swizzles;

/// A sheet of `n` by `n` quads over the unit square in xz, facing up, with normals and UVs; `height` lifts each vertex.
fn sheet(n: u32, height: impl Fn(f32, f32) -> f32) -> MeshData {
    let mut m = MeshData::default();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            m.positions.push([u, height(u, v), v]);
            m.normals.push([0.0, 1.0, 0.0]);
            m.uvs.push([u, v]);
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            let b = a + n + 1;
            m.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    m
}

/// `m` and a reversed copy of it with its own vertices and the opposite normals: a sheet seen from both sides.
fn two_sided(m: &MeshData) -> MeshData {
    let mut out = m.clone();
    let n = m.positions.len() as u32;
    out.positions.extend(&m.positions);
    out.uvs.extend(&m.uvs);
    out.normals.extend(m.normals.iter().map(|v| v.map(|c| -c)));
    for t in m.indices.as_chunks::<3>().0 {
        out.indices.extend([t[0] + n, t[2] + n, t[1] + n]);
    }
    out
}

/// A fin: a triangle in xz (root on x = 0, tip at x = 1) with an upper and a lower skin of their own vertices that meet in a
/// knife edge along both swept sides, and stand `thick` apart and open at the root.
fn fin(n: u32, thick: f32) -> MeshData {
    let mut m = MeshData::default();
    for side in [1.0f32, -1.0] {
        let base = m.positions.len() as u32;
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                // (the square squeezed into the triangle: its far side is the tip)
                let (x, z) = (u, (v - 0.5) * (1.0 - u));
                let h = thick * (1.0 - u) * (1.0 - (2.0 * v - 1.0).powi(2));
                m.positions.push([x, side * h, z]);
                m.normals.push([0.0, side, 0.0]);
                m.uvs.push([u * 0.5 + if side > 0.0 { 0.0 } else { 0.5 }, v]);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let a = base + j * (n + 1) + i;
                let b = a + n + 1;
                if side > 0.0 {
                    m.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
                } else {
                    m.indices.extend([a, a + 1, b, a + 1, b + 1, b]);
                }
            }
        }
    }
    m
}

/// A sphere of `rings` by `2 * rings` quads, radius 1, with normals and UVs: a seam down one side, a fan of vertices at each pole.
fn sphere(rings: u32) -> MeshData {
    let mut m = MeshData::default();
    let cols = rings * 2;
    for r in 0..=rings {
        let phi = std::f32::consts::PI * r as f32 / rings as f32;
        for c in 0..=cols {
            let theta = std::f32::consts::TAU * (c % cols) as f32 / cols as f32;
            let p = if r == 0 || r == rings {
                Vec3::new(0.0, phi.cos().signum(), 0.0)
            } else {
                Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin())
            };
            m.positions.push(p.to_array());
            m.normals.push(p.to_array());
            m.uvs.push([c as f32 / cols as f32, r as f32 / rings as f32]);
        }
    }
    for r in 0..rings {
        for c in 0..cols {
            let a = r * (cols + 1) + c;
            let b = a + cols + 1;
            // (at a pole the quad is one triangle)
            if r > 0 {
                m.indices.extend([a, a + 1, b]);
            }
            if r + 1 < rings {
                m.indices.extend([a + 1, b + 1, b]);
            }
        }
    }
    m
}

/// The surface seen along x, y and z: the triangles' areas, each times how squarely it faces the axis.
fn seen(m: &MeshData) -> [f32; 3] {
    let mut s = Vec3::ZERO;
    for t in m.indices.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| Vec3::from(m.positions[i as usize]));
        s += (b - a).cross(c - a).abs() * 0.5;
    }
    s.to_array()
}

type Edge = ([u32; 3], [u32; 3]);

/// How many triangles each edge has, by position.
fn edge_counts(m: &MeshData) -> HashMap<Edge, u32> {
    let mut edges = HashMap::new();
    let key = |i: u32| m.positions[i as usize].map(|c| (c + 0.0).to_bits());
    for t in m.indices.as_chunks::<3>().0 {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (a, b) = (key(a), key(b));
            if a != b {
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
    }
    edges
}

/// The mesh is one a renderer can take: indices in range, streams whole or absent, no vertex unused.
fn sound(m: &MeshData, e: &MeshExtras) {
    let n = m.positions.len();
    assert!(m.indices.len().is_multiple_of(3) && m.indices.iter().all(|i| (*i as usize) < n));
    for (name, len) in [
        ("normals", m.normals.len()),
        ("uvs", m.uvs.len()),
        ("colors", m.colors.len()),
        ("tangents", e.tangents.len()),
        ("uvs1", e.uvs1.len()),
    ] {
        assert!(len == 0 || len == n, "{name}: {len} for {n} vertices");
    }
    let mut used = vec![false; n];
    m.indices.iter().for_each(|i| used[*i as usize] = true);
    assert!(used.iter().all(|u| *u), "a vertex no triangle uses");
}

fn close(a: [f32; 3], b: [f32; 3], share: f32) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() <= share * b[k].max(1e-6),
            "axis {k}: {} for {}",
            a[k],
            b[k]
        );
    }
}

fn to_ratio(mesh: &MeshData, ratio: f32) -> MeshData {
    let (made, extras) = simplify_edges(mesh, &MeshExtras::new(), &EdgeCollapse::to_ratio(mesh, ratio));
    sound(&made, &extras);
    made
}

#[test]
fn a_thin_plate_keeps_its_outline_at_three_tenths_of_its_triangles() {
    for plate in [sheet(20, |_, _| 0.0), two_sided(&sheet(20, |_, _| 0.0))] {
        let full = plate.triangle_count();
        let made = to_ratio(&plate, 0.3);
        assert!(
            made.triangle_count() <= full * 3 / 10,
            "{} of {full}",
            made.triangle_count()
        );
        assert!(made.triangle_count() > full / 10);
        // a flat plate loses nothing at all: its surface and its bounds are what they were
        close(seen(&made), seen(&plate), 1e-4);
        assert_eq!(made.aabb(), plate.aabb());
        // every vertex still has the UV it had at its place
        for (p, uv) in made.positions.iter().zip(&made.uvs) {
            assert_eq!(*uv, [p[0], p[2]]);
        }
        // and the two sides of the two-sided plate are still two sheets: each face has the normals of its own side
        for t in made.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| Vec3::from(made.positions[i as usize]));
            let face = (b - a).cross(c - a).y.signum();
            assert!(
                t.iter().all(|i| made.normals[*i as usize][1] == face),
                "a face with the other side's normal"
            );
        }
    }
}

#[test]
fn a_wavy_two_sided_plate_keeps_its_surface_where_clustering_loses_it() {
    let plate = two_sided(&sheet(20, |u, v| 0.02 * (u * 9.0).sin() * (v * 7.0).cos()));
    let made = to_ratio(&plate, 0.3);
    assert!(made.triangle_count() <= plate.triangle_count() * 3 / 10);
    close(seen(&made), seen(&plate), 0.03);
    assert_eq!(made.aabb().0.xz(), plate.aabb().0.xz());
    assert_eq!(made.aabb().1.xz(), plate.aabb().1.xz());
}

#[test]
fn a_fin_keeps_its_silhouette_at_three_tenths_of_its_triangles() {
    let fin = fin(16, 0.04);
    let full = fin.triangle_count();
    let made = to_ratio(&fin, 0.3);
    assert!(
        made.triangle_count() <= full * 3 / 10,
        "{} of {full}",
        made.triangle_count()
    );
    // seen from above (the fin's outline) within 3 %
    let (a, b) = (seen(&made), seen(&fin));
    assert!((a[1] - b[1]).abs() < 0.03 * b[1], "{a:?} for {b:?}");
    // the tip and the root corners are where they were
    let (min, max) = made.aabb();
    assert_eq!((min.x, max.x, min.z, max.z), (0.0, 1.0, -0.5, 0.5));
    // no skin has a vertex of the other: the upper faces have the upper normals and the upper half of the texture
    for t in made.indices.as_chunks::<3>().0 {
        let side = made.normals[t[0] as usize][1];
        assert!(t.iter().all(|i| made.normals[*i as usize][1] == side));
        assert!(t.iter().all(|i| {
            let u = made.uvs[*i as usize][0];
            u == 0.5 || (u < 0.5) == (side > 0.0)
        }));
    }
}

#[test]
fn a_closed_sphere_stays_closed_and_keeps_its_seam() {
    let ball = sphere(24);
    assert!(
        edge_counts(&ball).values().all(|c| *c == 2),
        "the test's sphere is closed"
    );
    let made = to_ratio(&ball, 0.3);
    assert!(made.triangle_count() <= ball.triangle_count() * 3 / 10);
    assert!(made.triangle_count() > ball.triangle_count() / 5);
    assert!(
        edge_counts(&made).values().all(|c| *c == 2),
        "an open or a pinched edge"
    );
    // still the sphere: every face looks outwards, and the volume is within a few percent
    let mut volume = 0.0;
    for t in made.indices.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| Vec3::from(made.positions[i as usize]));
        assert!((b - a).cross(c - a).dot(a + b + c) > 0.0);
        volume += a.dot(b.cross(c)) / 6.0;
    }
    let sphere_volume = 4.0 / 3.0 * std::f32::consts::PI;
    assert!((volume - sphere_volume).abs() < 0.06 * sphere_volume, "{volume}");
    // the seam: no triangle reaches across the texture, and both sides of it are still there
    for t in made.indices.as_chunks::<3>().0 {
        let u = t.map(|i| made.uvs[i as usize][0]);
        let span = u.iter().fold(0.0f32, |m, v| m.max(*v)) - u.iter().fold(1.0f32, |m, v| m.min(*v));
        assert!(span < 0.5, "a triangle across the seam: {u:?}");
    }
    assert!(made.uvs.iter().any(|uv| uv[0] == 0.0) && made.uvs.iter().any(|uv| uv[0] == 1.0));
    // each vertex kept its own normal
    assert!(made.positions.iter().zip(&made.normals).all(|(p, normal)| p == normal));
}

#[test]
fn the_same_mesh_every_time() {
    for mesh in [sphere(16), fin(12, 0.05), two_sided(&sheet(12, |u, v| u * v * 0.1))] {
        let (a, b) = (to_ratio(&mesh, 0.25), to_ratio(&mesh, 0.25));
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.positions, b.positions);
    }
}

#[test]
fn the_error_limit_stops_it() {
    let ball = sphere(24);
    let free = simplify_edges(&ball, &MeshExtras::new(), &EdgeCollapse::to(40)).0;
    let held = simplify_edges(&ball, &MeshExtras::new(), &EdgeCollapse::to(40).with_max_error(0.002)).0;
    assert!(free.triangle_count() <= 40, "{}", free.triangle_count());
    assert!(
        held.triangle_count() > 200 && held.triangle_count() < ball.triangle_count(),
        "{}",
        held.triangle_count()
    );
    // a flat plate costs nothing: a limit of nothing still lets it go down
    let plate = sheet(10, |_, _| 0.0);
    let flat = simplify_edges(&plate, &MeshExtras::new(), &EdgeCollapse::to(20).with_max_error(0.0)).0;
    assert!(flat.triangle_count() <= 20, "{}", flat.triangle_count());
    close(seen(&flat), seen(&plate), 1e-4);
}

#[test]
fn a_mesh_with_a_vertex_for_every_corner_is_joined_by_position() {
    // flat shading: no two triangles share a vertex; a sheet with a fold in it
    let plate = sheet(12, |u, _| if u > 0.5 { (u - 0.5) * 0.6 } else { 0.0 });
    let mut loose = MeshData::default();
    for t in plate.indices.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| Vec3::from(plate.positions[i as usize]));
        let normal = (b - a).cross(c - a).normalize();
        for i in t {
            loose.indices.push(loose.positions.len() as u32);
            loose.positions.push(plate.positions[*i as usize]);
            loose.uvs.push(plate.uvs[*i as usize]);
            loose.normals.push(normal.to_array());
        }
    }
    let made = to_ratio(&loose, 0.3);
    assert!(
        made.triangle_count() <= loose.triangle_count() * 3 / 10,
        "{}",
        made.triangle_count()
    );
    close(seen(&made), seen(&loose), 0.02);
    // the fold is still a hard edge: every face has the normals of its own slope
    for t in made.indices.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| Vec3::from(made.positions[i as usize]));
        let normal = (b - a).cross(c - a).normalize();
        assert!(
            t.iter()
                .all(|i| Vec3::from(made.normals[*i as usize]).dot(normal) > 0.99)
        );
    }
}

#[test]
fn every_stream_follows_its_vertex() {
    let mut mesh = sheet(8, |u, v| (u * 3.0).sin() * (v * 2.0).cos() * 0.2);
    let n = mesh.positions.len();
    mesh.colors = (0..n).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect();
    let extras = MeshExtras::new()
        .with_tangents((0..n).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect())
        .with_uvs1(mesh.uvs.clone());
    let source = mesh.clone();
    let (made, made_extras) = simplify_edges(&mesh, &extras, &EdgeCollapse::to(40));
    sound(&made, &made_extras);
    assert!(made.triangle_count() <= 40, "{}", made.triangle_count());
    assert_eq!(made.colors.len(), made.positions.len());
    for (k, color) in made.colors.iter().enumerate() {
        let i = color[0] as usize;
        assert_eq!(made.positions[k], source.positions[i]);
        assert_eq!(made.uvs[k], source.uvs[i]);
        assert_eq!(made_extras.tangents[k][0], i as f32);
        assert_eq!(made_extras.uvs1[k], source.uvs[i]);
    }
    // a stream of another length is left out, as `simplify` leaves it
    mesh.colors.pop();
    let short = MeshExtras::new().with_tangents(vec![[0.0; 4]; 3]);
    let (made, made_extras) = simplify_edges(&mesh, &short, &EdgeCollapse::to(40));
    assert!(made.colors.is_empty() && made_extras.tangents.is_empty() && made.uvs.len() == made.positions.len());
}

#[test]
fn meshes_that_are_nothing_much_come_back_sound() {
    let none = MeshExtras::new();
    // nothing
    let (made, _) = simplify_edges(&MeshData::default(), &none, &EdgeCollapse::to(10));
    assert_eq!((made.positions.len(), made.indices.len()), (0, 0));
    // under the target already: as it is
    let plate = sheet(2, |_, _| 0.0);
    let (made, _) = simplify_edges(&plate, &none, &EdgeCollapse::to(100));
    assert_eq!((&made.indices, &made.positions), (&plate.indices, &plate.positions));
    // a target of none: the last triangle of a piece is never taken, and with a limit on the error the square stays a square
    let (made, extras) = simplify_edges(&plate, &none, &EdgeCollapse::to(0));
    sound(&made, &extras);
    assert_eq!(made.triangle_count(), 1);
    let (made, extras) = simplify_edges(&plate, &none, &EdgeCollapse::to(0).with_max_error(0.01));
    sound(&made, &extras);
    close(seen(&made), seen(&plate), 1e-4);
    assert_eq!(made.triangle_count(), 2);
    // one triangle, a triangle that is a line, one that is a point, an index of no vertex, a stray index at the end
    let odd = MeshData {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        indices: vec![0, 1, 2, 0, 3, 1, 2, 2, 2, 0, 1, 9, 1],
        ..MeshData::default()
    };
    let (made, extras) = simplify_edges(&odd, &none, &EdgeCollapse::to(0));
    sound(&made, &extras);
    assert_eq!(made.indices, vec![0, 1, 2]);
    // positions that are not numbers do not stop it
    let mut bad = sheet(4, |_, _| 0.0);
    bad.positions[7] = [f32::NAN, 0.0, f32::INFINITY];
    let (made, extras) = simplify_edges(&bad, &none, &EdgeCollapse::to(8));
    sound(&made, &extras);
}

/// How many places have triangles that hang together at the place only (two fans that share no edge there).
fn pinched(m: &MeshData) -> usize {
    let key = |i: u32| m.positions[i as usize].map(|c| (c + 0.0).to_bits());
    let mut at: HashMap<[u32; 3], Vec<[[u32; 3]; 2]>> = HashMap::new();
    for t in m.indices.as_chunks::<3>().0 {
        let k = t.map(key);
        for i in 0..3 {
            at.entry(k[i]).or_default().push([k[(i + 1) % 3], k[(i + 2) % 3]]);
        }
    }
    let mut bad = 0;
    for fans in at.values() {
        // (grow one piece from the first triangle over shared neighbours)
        let mut reached = vec![false; fans.len()];
        reached[0] = true;
        let mut grown = true;
        while grown {
            grown = false;
            for i in 0..fans.len() {
                if !reached[i] && (0..fans.len()).any(|j| reached[j] && fans[i].iter().any(|p| fans[j].contains(p))) {
                    reached[i] = true;
                    grown = true;
                }
            }
        }
        bad += usize::from(reached.contains(&false));
    }
    bad
}

/// A ribbon of `n` quads, `width` wide, wound `turns` times round a radius of 10 and drifting along the axis; positions only.
fn ribbon(n: u32, width: f32, turns: f32) -> MeshData {
    let mut m = MeshData::default();
    for y in 0..2 {
        for x in 0..=n {
            let a = x as f32 / n as f32 * turns * std::f32::consts::TAU;
            m.positions
                .push([10.0 * a.cos(), y as f32 * width, 10.0 * a.sin() + x as f32 * 0.05]);
        }
    }
    for x in 0..n {
        let (a, w) = (x, n + 1);
        m.indices.extend([a, a + 1, a + w + 1, a, a + w + 1, a + w]);
    }
    m
}

#[test]
fn a_ribbon_is_not_cut_through_across_a_rung() {
    // (width 1, one turn, down to 4 was two pieces joined at a point; so were two turns at 12, where the bends are knife edges)
    for (width, turns) in [(1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (0.3, 2.0), (0.1, 1.0)] {
        let mesh = ribbon(40, width, turns);
        for target in [40, 30, 20, 12, 8, 4, 1] {
            let (made, extras) = simplify_edges(&mesh, &MeshExtras::new(), &EdgeCollapse::to(target));
            sound(&made, &extras);
            assert_eq!(pinched(&made), 0, "width {width}, {turns} turns, to {target}");
            // one piece with one outline: a ribbon of t triangles has t + 2 border edges
            let borders = edge_counts(&made).values().filter(|c| **c == 1).count();
            assert_eq!(
                borders,
                made.triangle_count() + 2,
                "width {width}, {turns} turns, to {target}"
            );
        }
        // and with a reversed copy for its back
        let both = two_sided(&MeshData {
            normals: vec![[0.0, 1.0, 0.0]; mesh.positions.len()],
            uvs: vec![[0.0, 0.0]; mesh.positions.len()],
            ..mesh.clone()
        });
        for target in [40, 16, 4] {
            let (made, extras) = simplify_edges(&both, &MeshExtras::new(), &EdgeCollapse::to(target));
            sound(&made, &extras);
            assert_eq!(
                pinched(&made),
                0,
                "two-sided, width {width}, {turns} turns, to {target}"
            );
        }
    }
    // a flat strip still goes down to its two triangles, along its borders
    let strip = ribbon(40, 1.0, 0.0);
    let made = simplify_edges(&strip, &MeshExtras::new(), &EdgeCollapse::to(2)).0;
    assert_eq!(made.triangle_count(), 2);
}

#[test]
fn a_small_closed_piece_is_not_flattened() {
    let back_to_back = |m: &MeshData| {
        let mut seen = std::collections::HashSet::new();
        m.indices.as_chunks::<3>().0.iter().any(|t| {
            let mut k = *t;
            k.sort_unstable();
            !seen.insert(k)
        })
    };
    let tetrahedron = MeshData {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.5, 0.3, 0.8]],
        indices: vec![0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3],
        ..MeshData::default()
    };
    let bipyramid = MeshData {
        positions: vec![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.8, 0.0],
            [-0.5, -0.8, 0.0],
            [0.0, 0.0, -1.0],
        ],
        indices: vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 4, 2, 1, 4, 3, 2, 4, 1, 3],
        ..MeshData::default()
    };
    // (asked for 3, the tetrahedron came back as 2 triangles on the same three vertices; the bipyramid asked for 2 likewise)
    for (mesh, least) in [(&tetrahedron, 4), (&bipyramid, 4)] {
        for target in [3, 2, 1, 0] {
            let made = simplify_edges(mesh, &MeshExtras::new(), &EdgeCollapse::to(target)).0;
            assert!(!back_to_back(&made), "to {target}: {:?}", made.indices);
            assert_eq!(made.triangle_count(), least, "to {target}");
            assert!(edge_counts(&made).values().all(|c| *c == 2), "to {target}: opened");
        }
    }
}

#[test]
fn a_fan_round_one_vertex_and_a_flat_sheet_do_not_take_long() {
    // The whole neighbourhood of a collapse used to be queued again: a fan of 4,000 took 27 s and a flat sheet of 20,000 a
    // second (optimized). Generous limits (a loaded PC, a debug build): they catch the old growth, not a slow day.
    let mut fan = MeshData::default();
    let n = 4000u32;
    fan.positions.push([0.0, 0.0, 0.0]);
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        fan.positions.push([a.cos(), a.sin(), 0.0]);
        fan.indices.extend([0, 1 + i, 1 + (i + 1) % n]);
    }
    let start = std::time::Instant::now();
    let made = simplify_edges(&fan, &MeshExtras::new(), &EdgeCollapse::to(8)).0;
    assert!((5..=8).contains(&made.triangle_count()), "{}", made.triangle_count());
    assert!(
        start.elapsed().as_secs_f32() < 5.0,
        "a fan of {n}: {:?}",
        start.elapsed()
    );
    let plate = sheet(100, |_, _| 0.0);
    let start = std::time::Instant::now();
    let made = simplify_edges(&plate, &MeshExtras::new(), &EdgeCollapse::to_ratio(&plate, 0.3)).0;
    assert!(made.triangle_count() <= 6000, "{}", made.triangle_count());
    assert!(
        start.elapsed().as_secs_f32() < 10.0,
        "a sheet of 20,000: {:?}",
        start.elapsed()
    );
}

#[test]
fn several_counts_in_one_go_are_the_meshes_of_each_alone() {
    let same = |a: &(MeshData, MeshExtras), b: &(MeshData, MeshExtras)| {
        a.0.positions == b.0.positions
            && a.0.normals == b.0.normals
            && a.0.uvs == b.0.uvs
            && a.0.indices == b.0.indices
            && a.1.tangents == b.1.tangents
    };
    let mut extras = MeshExtras::new();
    let ball = sphere(12);
    extras.tangents = vec![[1.0, 0.0, 0.0, 1.0]; ball.positions.len()];
    for (mesh, extras) in [
        (ball, extras),
        (
            two_sided(&sheet(8, |u, v| (u * 9.0).sin() * (v * 7.0).cos() * 0.1)),
            MeshExtras::new(),
        ),
        (fin(8, 0.02), MeshExtras::new()),
        (ribbon(40, 1.0, 2.0), MeshExtras::new()),
    ] {
        let full = mesh.triangle_count();
        // (not in order, one twice, one over the mesh, some that no allowed collapse reaches)
        let counts = [full / 2, full + 5, 3, full / 5, full / 2, full - 1, 0, full / 3];
        let together = simplify_edges_steps(&mesh, &extras, &EdgeCollapse::to(0), &counts);
        assert_eq!(together.len(), counts.len());
        for (count, made) in counts.iter().zip(&together) {
            sound(&made.0, &made.1);
            let alone = simplify_edges(&mesh, &extras, &EdgeCollapse::to(*count));
            assert!(
                same(made, &alone),
                "to {count} of {full}: {} against {}",
                made.0.triangle_count(),
                alone.0.triangle_count()
            );
        }
    }
    assert!(simplify_edges_steps(&sheet(4, |_, _| 0.0), &MeshExtras::new(), &EdgeCollapse::to(8), &[]).is_empty());
}
