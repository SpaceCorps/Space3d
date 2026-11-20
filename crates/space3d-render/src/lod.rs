//! Levels of detail: simpler meshes for objects that are small on screen.
//!
//! A crowd of models a few dozen pixels tall costs the GPU their full triangle count, in the
//! scene pass and again in every shadow map. [`simplify`] makes a coarser mesh from a detailed
//! one, [`MeshLod`] lists an entity's meshes from the most detailed down, and the renderer
//! picks one each frame from the size of the entity's bounding sphere on screen
//! ([`LodSettings`]): the largest it has in any active camera, so a model shown close up in a
//! preview keeps its detail while the same model far away in the world does not.
//!
//! ```ignore
//! let (coarse, extras) = lod::simplify_to(&data, &extras, 0.25);
//! let low = assets.add_mesh_with(coarse, extras);
//! world.spawn((Transform::default(), Mesh3d(full), material, MeshLod::new(radius, &[full, low])));
//! ```
//!
//! A level can also be no mesh at all ([`MeshLod::from_levels`]): a model of many parts then
//! draws one merged mesh far away, on one entity, while its parts draw nothing.

use crate::assets::RenderAssets;
use crate::components::{Camera3d, Mesh3d, Projection, RenderTarget};
use crate::culling::max_axis_scale;
use crate::handles::MeshHandle;
use crate::material::MeshExtras;
use crate::renderer::Renderer;
use glam::Vec3;
use space3d_core::{GlobalTransform, MeshData, Resources, World};
use std::collections::{HashMap, HashSet};

mod collapse;
pub use collapse::{EdgeCollapse, simplify_edges, simplify_edges_steps};

/// The most levels a [`MeshLod`] holds, the full mesh included.
pub const MAX_LOD_LEVELS: usize = 4;

/// The levels of detail of an entity's mesh. The renderer sets the entity's [`Mesh3d`] to one
/// of `meshes` each frame; an entity without this component keeps the mesh it has.
///
/// Two of them are equal when they list the same levels over the same sphere: the level the
/// renderer last picked is not compared.
#[derive(Clone, Copy, Debug)]
pub struct MeshLod {
    meshes: [MeshHandle; MAX_LOD_LEVELS],
    count: u8,
    /// The level the size on screen last asked for (`UNKNOWN` before the first frame). Kept
    /// apart from the mesh that is drawn: while the wanted mesh is not uploaded another level
    /// is drawn in its place, and the hysteresis must not start from that one.
    wanted: u8,
    /// Radius of the sphere that is measured on screen, in the entity's own units (it is
    /// scaled by the entity's `GlobalTransform`). For a model of several parts give every part
    /// the whole model's radius, so the parts change level together.
    pub radius: f32,
    /// Where the sphere's centre is, in the entity's own units.
    pub center: Vec3,
}

/// `MeshLod::wanted` of a component the renderer has not seen yet: its level is then the one
/// of the mesh the entity has.
const UNKNOWN: u8 = u8::MAX;

impl PartialEq for MeshLod {
    fn eq(&self, other: &Self) -> bool {
        self.meshes() == other.meshes() && self.radius == other.radius && self.center == other.center
    }
}

impl MeshLod {
    /// `meshes` from the most detailed (the full mesh) down, at most [`MAX_LOD_LEVELS`] (the
    /// rest is left out); `radius` as [`Self::radius`].
    ///
    /// # Panics
    /// With no mesh at all.
    pub fn new(radius: f32, meshes: &[MeshHandle]) -> Self {
        assert!(!meshes.is_empty(), "a MeshLod needs its full mesh");
        let count = meshes.len().min(MAX_LOD_LEVELS);
        let mut all = [meshes[0]; MAX_LOD_LEVELS];
        all[..count].copy_from_slice(&meshes[..count]);
        Self {
            meshes: all,
            count: count as u8,
            wanted: UNKNOWN,
            radius,
            center: Vec3::ZERO,
        }
    }

    /// As [`Self::new`], with levels that may be no mesh: at a `None` level the entity is not
    /// drawn (its [`Mesh3d`] is set to [`MeshHandle::NONE`], which no view draws; its
    /// `Visibility` is not touched). At most [`MAX_LOD_LEVELS`] levels (the rest is left out).
    ///
    /// For a model of many parts that should cost one draw far away: give every part
    /// `[Some(full), Some(simpler), None]` and one more entity, with the model's merged mesh and
    /// its own material, `[None, None, Some(merged)]`, all with the same `radius` and centre
    /// and the same number of levels, so they change level in the same frame. A part with no
    /// simpler mesh repeats its full one (`[Some(full), Some(full), None]`). With levels of
    /// detail off every entity draws its first level: the parts, not the merged mesh.
    ///
    /// The renderer cannot tell that such entities belong together, so two rules are the app's
    /// to keep:
    ///
    /// * Have every mesh of the model uploaded ([`RenderAssets::is_mesh_uploaded`], the merged
    ///   one included) before the entities are spawned. When the level asked for has a mesh that
    ///   is not uploaded yet the renderer draws another level of the same entity, never a `None`
    ///   one, and an entity with no uploaded mesh at all is not drawn: with the merged mesh
    ///   still on its way the parts draw nothing at the far level and the model is not drawn.
    /// * Spawn all of them in the same frame. Each keeps its own hysteresis state, which starts
    ///   at the level of the entity's [`Mesh3d`]; one added later, or given a new `MeshLod`,
    ///   starts again and can sit one level away from the others until the model leaves the
    ///   band around a threshold. Give such a one [`Self::with_level_of`] a sibling's `MeshLod`.
    ///
    /// # Panics
    /// With no level at all.
    pub fn from_levels(radius: f32, levels: &[Option<MeshHandle>]) -> Self {
        assert!(!levels.is_empty(), "a MeshLod needs a level");
        let count = levels.len().min(MAX_LOD_LEVELS);
        let mut all = [levels[0].unwrap_or(MeshHandle::NONE); MAX_LOD_LEVELS];
        for (to, from) in all.iter_mut().zip(&levels[..count]) {
            *to = from.unwrap_or(MeshHandle::NONE);
        }
        Self {
            meshes: all,
            count: count as u8,
            wanted: UNKNOWN,
            radius,
            center: Vec3::ZERO,
        }
    }

    /// With the measured sphere's centre moved (a part whose own origin is not the model's).
    pub fn with_center(mut self, center: Vec3) -> Self {
        self.center = center;
        self
    }

    /// Starting from the level `other` was last asked for, instead of the level of the entity's
    /// [`Mesh3d`]: for an entity added to a model whose other entities are already drawn (or a
    /// `MeshLod` that replaces one), so that it is at their level from its first frame, also
    /// inside the hysteresis band where its own start would pick the neighbouring level. Read
    /// `other` from a sibling in the world; both need the same sphere and number of levels.
    pub fn with_level_of(mut self, other: &MeshLod) -> Self {
        self.wanted = other.wanted;
        self
    }

    /// The meshes, most detailed first ([`MeshHandle::NONE`] for a level that draws nothing).
    pub fn meshes(&self) -> &[MeshHandle] {
        &self.meshes[..self.count as usize]
    }

    /// Which level `mesh` is (0: the full mesh); `None` for a mesh that is not one of them.
    /// The first of them when several levels hold the same mesh.
    pub fn level_of(&self, mesh: MeshHandle) -> Option<usize> {
        self.meshes().iter().position(|m| *m == mesh)
    }

    /// The mesh to draw while the one of `level` is not uploaded: of the levels that have an
    /// uploaded mesh the nearest to `level`, the simpler one of two equally near (it is the
    /// cheaper to draw until the wanted one arrives). `None` when no level has one.
    fn nearest_uploaded(&self, level: usize, assets: &RenderAssets) -> Option<MeshHandle> {
        let count = self.count as usize;
        (1..count)
            .flat_map(|d| [level.checked_add(d), level.checked_sub(d)])
            .flatten()
            .filter(|l| *l < count)
            .map(|l| self.meshes[l])
            .find(|m| assets.is_mesh_drawable(*m))
    }
}

/// How levels of detail are picked (resource; the defaults without it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodSettings {
    /// Off: every [`MeshLod`] entity draws its full mesh.
    pub enabled: bool,
    /// Level `i + 1` is drawn while the bounding sphere's radius on screen is under
    /// `pixels[i]` physical pixels (descending). The default keeps the full mesh above 56 px,
    /// then a second level down to 20 px and a third below.
    pub pixels: [f32; MAX_LOD_LEVELS - 1],
    /// Multiplies `pixels`: above 1 simpler meshes are used sooner (a low quality tier),
    /// below 1 later.
    pub bias: f32,
}

impl Default for LodSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            pixels: [56.0, 20.0, 0.0],
            bias: 1.0,
        }
    }
}

/// A level changes only once the size is this far past its threshold, so an object whose size
/// sits on a threshold does not flicker between two meshes.
const HYSTERESIS: f32 = 0.1;

impl LodSettings {
    /// The level for a sphere `px` pixels in radius, ignoring the level it has now.
    fn level_for(&self, px: f32) -> usize {
        self.pixels.iter().filter(|t| px < **t * self.bias).count()
    }

    /// The level for a sphere `px` pixels in radius that is drawn at `current` now.
    pub fn pick(&self, px: f32, current: usize) -> usize {
        if !self.enabled {
            return 0;
        }
        let finest = self.level_for(px * (1.0 + HYSTERESIS));
        let coarsest = self.level_for(px * (1.0 - HYSTERESIS));
        current.clamp(finest, coarsest)
    }
}

/// How a camera sizes a sphere on screen.
#[derive(Clone, Copy)]
enum Scale {
    /// Pixels = radius × this ÷ distance from the eye.
    Perspective(Vec3, f32),
    /// Pixels = radius × this.
    Orthographic(f32),
}

/// More active cameras than this and every entity keeps its full mesh (it does not happen in
/// practice; the scratch stays on the stack).
const MAX_CAMERAS: usize = 16;

/// Set every [`MeshLod`] entity's [`Mesh3d`] to the level its size on screen asks for. Runs in
/// the Render stage before the frame is extracted; allocates nothing.
///
/// With a renderer, a level whose mesh is not on the GPU yet is not drawn while another level
/// of the entity is: the nearest uploaded level is drawn in its place, and the wanted one from
/// the frame after it arrives (this runs before the frame's uploads). The level asked for is
/// remembered in the [`MeshLod`], so the hysteresis goes on from it and not from the stand-in.
/// Without a renderer or [`RenderAssets`] nothing is ever uploaded, and the level asked for is
/// set as it is.
pub(crate) fn select_lods(world: &mut World, resources: &Resources) {
    let settings = resources.get::<LodSettings>().copied().unwrap_or_default();
    let renderer = resources.get::<Renderer>();
    let primary = renderer.map_or((1280, 720), Renderer::size);
    let assets = resources.get::<RenderAssets>();
    let uploads = renderer.and(assets);
    let mut cameras = [Scale::Orthographic(0.0); MAX_CAMERAS];
    let mut count = 0;
    let mut overflow = false;
    for (camera, at) in world.query_mut::<(&Camera3d, &GlobalTransform)>() {
        if !camera.active {
            continue;
        }
        if count == MAX_CAMERAS {
            overflow = true;
            break;
        }
        let target_height = match camera.target {
            RenderTarget::Primary => primary.1,
            RenderTarget::Texture(t) => assets.and_then(|a| a.texture_size(t)).map_or(primary.1, |s| s.1),
        } as f32;
        let height = target_height * camera.viewport.map_or(1.0, |v| v[3]);
        cameras[count] = match camera.projection {
            Projection::Perspective { fov_y, .. } => {
                Scale::Perspective(at.0.w_axis.truncate(), height * 0.5 / (fov_y * 0.5).tan().max(1e-6))
            }
            Projection::Orthographic { height: world, .. } => Scale::Orthographic(height / world.max(1e-6)),
        };
        count += 1;
    }
    let cameras = &cameras[..count];
    let full = !settings.enabled || overflow || cameras.is_empty();
    for (lod, mesh, at) in world.query_mut::<(&mut MeshLod, &mut Mesh3d, &GlobalTransform)>() {
        let level = if full {
            0
        } else {
            let current = match lod.wanted {
                UNKNOWN => lod.level_of(mesh.0).unwrap_or(0),
                level => level as usize,
            };
            let center = at.0.transform_point3(lod.center);
            let radius = lod.radius * max_axis_scale(&at.0);
            let px = cameras
                .iter()
                .map(|c| match *c {
                    Scale::Perspective(eye, k) => radius * k / eye.distance(center).max(radius.max(1e-6)),
                    Scale::Orthographic(k) => radius * k,
                })
                .fold(0.0, f32::max);
            settings.pick(px, current)
        };
        let level = level.min(lod.count as usize - 1);
        if lod.wanted != level as u8 {
            lod.wanted = level as u8;
        }
        let mut drawn = lod.meshes[level];
        if let Some(assets) = uploads
            && !drawn.is_none()
            && !assets.is_mesh_drawable(drawn)
            && let Some(uploaded) = lod.nearest_uploaded(level, assets)
        {
            drawn = uploaded;
        }
        if mesh.0 != drawn {
            mesh.0 = drawn;
        }
    }
}

/// A coarser copy of a mesh by vertex clustering: the vertices inside each cube of side `cell`
/// (in the mesh's units) become one, the one nearest their middle, with the normals of all of
/// them averaged; triangles that collapse are dropped, and so are repeats of a triangle (the
/// same three vertices in the same winding). Cheap and robust (it needs no manifold mesh), and
/// good for meshes drawn small.
///
/// Not everything inside a cell is joined:
///
/// - Vertices are only joined where their texture coordinates are close too (within
///   [`UV_REACH`] cells' worth of texture, by how much texture the mesh lays over a unit of
///   space): the two sides of a seam lie at one place in space and far apart in the texture,
///   and a triangle between vertices of two islands would be painted with whatever lies between
///   them in the atlas.
/// - Vertices are only joined where their normals face the same way: along the same axis, in
///   the same direction (the largest component of the normal and its sign, six ways in all).
///   The two sides of a thin sheet (a wing, a fin: a face and a reversed copy of it at the same
///   place) so stay two sheets, each with its own normal and winding, and a hard edge between
///   faces that look along different axes keeps its two normals. A reversed copy of a triangle
///   is a triangle of its own, not a repeat: with back faces culled it is the only one seen
///   from its side.
///
/// So seams, the two sides of a sheet and such hard edges are not welded, however large the
/// cell: a mesh that is mostly these gets less simple than its cell alone would make it, and a
/// closed mesh with normals never gets simpler than one vertex for each way it faces.
///
/// A stream (normals, UVs, colours, tangents, second UVs) counts only when it has a value for
/// every position; one of any other length is left out of the result, as an empty one is.
pub fn simplify(data: &MeshData, extras: &MeshExtras, cell: f32) -> (MeshData, MeshExtras) {
    let n = data.positions.len();
    // A stream shorter than the positions cannot be read by a position's index: it is not there.
    let (has_normals, has_uvs) = (data.normals.len() == n, data.uvs.len() == n);
    let (min, _) = data.aabb();
    let inv = 1.0 / cell.max(1e-9);
    let mut ids: HashMap<[i32; 6], u32> = HashMap::with_capacity(n / 4 + 1);
    // How much texture a unit of space is given, over the whole mesh: a cell of space is that much texture across.
    let uv_inv = {
        let (mut space, mut texture) = (0.0f64, 0.0f64);
        if has_uvs {
            for t in data.indices.as_chunks::<3>().0 {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let (a, b) = (a as usize, b as usize);
                    space += f64::from(Vec3::from(data.positions[a]).distance(Vec3::from(data.positions[b])));
                    texture += f64::from(glam::Vec2::from(data.uvs[a]).distance(glam::Vec2::from(data.uvs[b])));
                }
            }
        }
        let density = if space > 0.0 { (texture / space) as f32 } else { 0.0 };
        if density > 0.0 {
            1.0 / (cell.max(1e-9) * density * UV_REACH)
        } else {
            0.0
        }
    };
    let mut cluster = Vec::with_capacity(n);
    // Per cluster: the sum of its positions and normals, and how many they are.
    let mut sums: Vec<(Vec3, Vec3, u32)> = Vec::new();
    for (i, p) in data.positions.iter().enumerate() {
        let p = Vec3::from(*p);
        let [x, y, z] = ((p - min) * inv).floor().as_ivec3().to_array();
        let [u, v] = if has_uvs {
            data.uvs[i].map(|c| (c * uv_inv).floor() as i32)
        } else {
            [0, 0]
        };
        let normal = if has_normals {
            Vec3::from(data.normals[i])
        } else {
            Vec3::ZERO
        };
        let key = [x, y, z, u, v, if has_normals { facing(normal) } else { 0 }];
        let id = *ids.entry(key).or_insert_with(|| {
            sums.push((Vec3::ZERO, Vec3::ZERO, 0));
            sums.len() as u32 - 1
        });
        let s = &mut sums[id as usize];
        s.0 += p;
        s.1 += normal;
        s.2 += 1;
        cluster.push(id);
    }
    // The member nearest each cluster's middle stands for it.
    let mut nearest: Vec<(f32, u32)> = vec![(f32::INFINITY, 0); sums.len()];
    for (i, p) in data.positions.iter().enumerate() {
        let c = cluster[i] as usize;
        let d = Vec3::from(*p).distance_squared(sums[c].0 / sums[c].2 as f32);
        if d < nearest[c].0 {
            nearest[c] = (d, i as u32);
        }
    }
    // Copy a stream that has a value for each of the `n` positions; any other is left out.
    fn pick<T: Copy>(nearest: &[(f32, u32)], n: usize, from: &[T], to: &mut Vec<T>) {
        if from.len() == n {
            to.extend(nearest.iter().map(|(_, i)| from[*i as usize]));
        }
    }
    let mut out = MeshData::default();
    let mut out_extras = MeshExtras::new();
    pick(&nearest, n, &data.positions, &mut out.positions);
    pick(&nearest, n, &data.uvs, &mut out.uvs);
    pick(&nearest, n, &data.colors, &mut out.colors);
    pick(&nearest, n, &extras.tangents, &mut out_extras.tangents);
    pick(&nearest, n, &extras.uvs1, &mut out_extras.uvs1);
    if has_normals {
        out.normals.extend(nearest.iter().zip(&sums).map(|((_, i), s)| {
            let mean = s.1.normalize_or_zero();
            if mean == Vec3::ZERO {
                data.normals[*i as usize]
            } else {
                mean.to_array()
            }
        }));
    }
    let mut seen: HashSet<[u32; 3]> = HashSet::with_capacity(data.indices.len() / 6 + 1);
    for t in data.indices.as_chunks::<3>().0 {
        let (a, b, c) = (cluster[t[0] as usize], cluster[t[1] as usize], cluster[t[2] as usize]);
        if a == b || b == c || a == c {
            continue;
        }
        // The same triangle whichever corner it starts at, and not its reverse: the key keeps the winding.
        let key = if a < b && a < c {
            [a, b, c]
        } else if b < c {
            [b, c, a]
        } else {
            [c, a, b]
        };
        if seen.insert(key) {
            out.indices.extend([a, b, c]);
        }
    }
    (out, out_extras)
}

/// Which way a normal faces, for [`simplify`]'s clusters: its largest component and that
/// component's sign (1..=6 for +x, -x, +y, -y, +z, -z). Opposite normals never share a value.
fn facing(normal: Vec3) -> i32 {
    let a = normal.abs();
    let axis = if a.x >= a.y && a.x >= a.z {
        0
    } else if a.y >= a.z {
        1
    } else {
        2
    };
    1 + axis * 2 + i32::from(normal[axis as usize] < 0.0)
}

/// How far apart in the texture [`simplify`] still joins two vertices of one cell of space:
/// this many times the texture a cell of space is across, on average over the mesh.
pub const UV_REACH: f32 = 3.0;

/// [`simplify`] with the smallest cell that leaves at most `ratio` of the mesh's triangles
/// (0.25: a quarter), found by halving. A mesh that cannot be brought that low without
/// vanishing comes back as coarse as it can be; one with no triangles comes back as it is.
pub fn simplify_to(data: &MeshData, extras: &MeshExtras, ratio: f32) -> (MeshData, MeshExtras) {
    let triangles = data.triangle_count();
    let target = ((triangles as f32 * ratio.clamp(0.0, 1.0)) as usize).max(1);
    if triangles <= target {
        return (data.clone(), extras.clone());
    }
    let (min, max) = data.aabb();
    let extent = (max - min).max_element().max(1e-6);
    // Cells from 1/1024 of the mesh (as good as the mesh itself) to half of it, on a log scale
    let cell_at = |t: f32| extent * 2f32.powf(-10.0 + 9.0 * t);
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    let mut best: Option<(MeshData, MeshExtras)> = None;
    for _ in 0..10 {
        let mid = (lo + hi) * 0.5;
        let made = simplify(data, extras, cell_at(mid));
        let count = made.0.triangle_count();
        if count == 0 {
            // too coarse: nothing is left
            hi = mid;
        } else if count <= target {
            best = Some(made);
            hi = mid;
        } else {
            // the coarsest so far that still has triangles, in case the target is never met
            if best.as_ref().is_none_or(|b| b.0.triangle_count() > target) {
                best = Some(made);
            }
            lo = mid;
        }
    }
    best.unwrap_or_else(|| (data.clone(), extras.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handles::MeshHandle;
    use glam::Mat4;
    use space3d_core::Transform;

    /// A sphere of `rings` by `2 * rings` quads, radius 1, with normals and UVs.
    fn sphere(rings: u32) -> MeshData {
        let mut m = MeshData::default();
        let cols = rings * 2;
        for r in 0..=rings {
            let phi = std::f32::consts::PI * r as f32 / rings as f32;
            for c in 0..=cols {
                let theta = std::f32::consts::TAU * c as f32 / cols as f32;
                let p = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
                m.positions.push(p.to_array());
                m.normals.push(p.to_array());
                m.uvs.push([c as f32 / cols as f32, r as f32 / rings as f32]);
            }
        }
        for r in 0..rings {
            for c in 0..cols {
                let a = r * (cols + 1) + c;
                let b = a + cols + 1;
                m.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        m
    }

    #[test]
    fn a_simplified_mesh_is_smaller_and_whole() {
        let full = sphere(48);
        let extras = MeshExtras::new().with_tangents(vec![[1.0, 0.0, 0.0, 1.0]; full.positions.len()]);
        let (coarse, coarse_extras) = simplify(&full, &extras, 0.25);
        assert!(coarse.triangle_count() > 50, "{}", coarse.triangle_count());
        assert!(coarse.triangle_count() * 4 < full.triangle_count());
        // every stream has a value for every vertex, and every index a vertex
        let n = coarse.positions.len();
        assert_eq!(
            (coarse.normals.len(), coarse.uvs.len(), coarse_extras.tangents.len()),
            (n, n, n)
        );
        assert!(coarse.colors.is_empty() && coarse_extras.uvs1.is_empty());
        assert!(coarse.indices.iter().all(|i| (*i as usize) < n));
        // it is still the sphere: vertices on it, normals pointing out, the same bounds within a cell
        for (p, normal) in coarse.positions.iter().zip(&coarse.normals) {
            let (p, normal) = (Vec3::from(*p), Vec3::from(*normal));
            assert!((p.length() - 1.0).abs() < 1e-4);
            assert!(normal.dot(p) > 0.8 && (normal.length() - 1.0).abs() < 1e-3);
        }
        let ((min, max), (full_min, full_max)) = (coarse.aabb(), full.aabb());
        assert!((min - full_min).abs().max_element() < 0.25 && (max - full_max).abs().max_element() < 0.25);
        // no triangle twice in the same winding (a triangle and its reverse are two), none collapsed
        let mut seen = HashSet::new();
        for t in coarse.indices.as_chunks::<3>().0 {
            assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
            let first = (0..3).min_by_key(|i| t[*i]).unwrap();
            assert!(seen.insert([t[first], t[(first + 1) % 3], t[(first + 2) % 3]]));
        }
    }

    /// Two sheets at one place with their own islands of the texture (a hull's two sides, a
    /// seam) are not welded into each other, however large the cell: no triangle of the simpler
    /// mesh reaches from one island to the other.
    #[test]
    fn a_seam_is_not_welded() {
        let mut m = MeshData::default();
        let n = 24u32;
        for (sheet, u0) in [(0u32, 0.0f32), (1, 0.6)] {
            for y in 0..=n {
                for x in 0..=n {
                    let (fx, fy) = (x as f32 / n as f32, y as f32 / n as f32);
                    m.positions.push([fx, fy, 0.0]);
                    m.normals.push([0.0, 0.0, if sheet == 0 { 1.0 } else { -1.0 }]);
                    m.uvs.push([u0 + 0.3 * fx, 0.3 * fy]);
                }
            }
            let base = sheet * (n + 1) * (n + 1);
            for y in 0..n {
                for x in 0..n {
                    let a = base + y * (n + 1) + x;
                    let b = a + n + 1;
                    m.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
                }
            }
        }
        let (coarse, _) = simplify(&m, &MeshExtras::new(), 0.2);
        assert!(coarse.triangle_count() > 20 && coarse.triangle_count() * 4 < m.triangle_count());
        for t in coarse.indices.as_chunks::<3>().0 {
            let u: Vec<f32> = t.iter().map(|i| coarse.uvs[*i as usize][0]).collect();
            let island = |u: f32| u >= 0.45;
            assert!(island(u[0]) == island(u[1]) && island(u[1]) == island(u[2]), "{u:?}");
            // and the two sides keep their own normals
            let z: Vec<f32> = t.iter().map(|i| coarse.normals[*i as usize][2]).collect();
            assert!(
                z.iter().all(|z| z.abs() > 0.99) && z[0] == z[1] && z[1] == z[2],
                "{z:?}"
            );
        }
    }

    /// A thin two-sided part as the game makes it: a face and a reversed copy of it a hair
    /// behind, with the same UVs. Both sides survive, each with its own winding and normal.
    #[test]
    fn both_sides_of_a_thin_sheet_are_kept() {
        let mut m = MeshData::default();
        let n = 24u32;
        for back in [false, true] {
            for y in 0..=n {
                for x in 0..=n {
                    let (fx, fy) = (x as f32 / n as f32, y as f32 / n as f32);
                    m.positions.push([fx, fy, if back { -0.001 } else { 0.0 }]);
                    m.normals.push([0.0, 0.0, if back { -1.0 } else { 1.0 }]);
                    m.uvs.push([fx, fy]);
                }
            }
            let base = u32::from(back) * (n + 1) * (n + 1);
            for y in 0..n {
                for x in 0..n {
                    let a = base + y * (n + 1) + x;
                    let b = a + n + 1;
                    if back {
                        m.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
                    } else {
                        m.indices.extend([a, a + 1, b, a + 1, b + 1, b]);
                    }
                }
            }
        }
        let (coarse, _) = simplify(&m, &MeshExtras::new(), 0.2);
        assert!(coarse.triangle_count() > 20 && coarse.triangle_count() * 4 < m.triangle_count());
        let (mut front, mut back) = (0, 0);
        for t in coarse.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| Vec3::from(coarse.positions[i as usize]));
            // the way the triangle faces by its winding is the way its vertices' normals point
            let faces = (b - a).cross(c - a).z;
            assert!(faces != 0.0);
            for i in t {
                let z = coarse.normals[*i as usize][2];
                assert!(
                    z.abs() > 0.99 && (z > 0.0) == (faces > 0.0),
                    "normal {z}, facing {faces}"
                );
            }
            if faces > 0.0 {
                front += 1;
            } else {
                back += 1;
            }
        }
        assert!(front > 10 && back > 10, "{front} front, {back} back");
        assert!(
            front * 4 >= back * 3 && back * 4 >= front * 3,
            "{front} front, {back} back"
        );
    }

    /// A stream with fewer values than there are positions is not read (it would be read past
    /// its end): it is left out, as if it were empty.
    #[test]
    fn a_short_stream_is_left_out() {
        let full = sphere(16);
        let n = full.positions.len();
        let whole = MeshExtras::new()
            .with_tangents(vec![[1.0, 0.0, 0.0, 1.0]; n])
            .with_uvs1(vec![[0.5, 0.5]; n]);
        let short = |len: usize| {
            let mut m = full.clone();
            m.colors = vec![[1.0; 4]; n];
            let mut e = whole.clone();
            m.normals.truncate(len);
            m.uvs.truncate(len);
            m.colors.truncate(len);
            e.tangents.truncate(len);
            e.uvs1.truncate(len);
            simplify(&m, &e, 0.3)
        };
        let (coarse, extras) = short(n);
        let v = coarse.positions.len();
        assert!(coarse.triangle_count() > 20);
        assert_eq!([coarse.normals.len(), coarse.uvs.len(), coarse.colors.len()], [v, v, v]);
        assert_eq!([extras.tangents.len(), extras.uvs1.len()], [v, v]);
        for len in [n - 1, n / 2, 1] {
            let (coarse, extras) = short(len);
            assert!(coarse.triangle_count() > 20 && coarse.positions.len() > 10);
            assert!(coarse.indices.iter().all(|i| (*i as usize) < coarse.positions.len()));
            assert!(coarse.normals.is_empty() && coarse.uvs.is_empty() && coarse.colors.is_empty());
            assert!(extras.tangents.is_empty() && extras.uvs1.is_empty());
        }
        // one short stream alone goes; the others stay
        let mut m = full.clone();
        m.uvs.truncate(3);
        let (coarse, extras) = simplify(&m, &whole, 0.3);
        assert!(coarse.uvs.is_empty());
        assert_eq!(
            [coarse.normals.len(), extras.tangents.len()],
            [coarse.positions.len(); 2]
        );
    }

    #[test]
    fn a_tiny_cell_keeps_the_mesh_and_a_huge_one_next_to_nothing() {
        let full = sphere(12);
        let (same, _) = simplify(&full, &MeshExtras::new(), 1e-5);
        // (nothing welds: the doubled vertices of the seam and the poles lie at one place in
        // space and apart in the texture, so even the slivers at the poles stay)
        assert_eq!(same.triangle_count(), full.triangle_count());
        // One cell for the whole sphere: a vertex for each of the six ways its normals face is
        // all that is not joined, an octahedron
        let (least, _) = simplify(&full, &MeshExtras::new(), 100.0);
        assert_eq!((least.positions.len(), least.triangle_count()), (6, 8));
        // and without normals nothing at all
        let bare = MeshData {
            positions: full.positions.clone(),
            indices: full.indices.clone(),
            ..MeshData::default()
        };
        let (none, _) = simplify(&bare, &MeshExtras::new(), 100.0);
        assert_eq!((none.positions.len(), none.triangle_count()), (1, 0));
    }

    #[test]
    fn simplify_to_meets_its_ratio() {
        let full = sphere(64);
        for ratio in [0.5, 0.25, 0.06] {
            let (coarse, _) = simplify_to(&full, &MeshExtras::new(), ratio);
            let (got, most) = (coarse.triangle_count(), (full.triangle_count() as f32 * ratio) as usize);
            assert!(got <= most && got * 3 > most, "ratio {ratio}: {got} of at most {most}");
        }
        // nothing to do, and nothing to work with
        assert_eq!(simplify_to(&full, &MeshExtras::new(), 1.0).0.indices, full.indices);
        assert_eq!(
            simplify_to(&MeshData::default(), &MeshExtras::new(), 0.1)
                .0
                .triangle_count(),
            0
        );
        // a single triangle cannot get smaller: it stays
        let one = MeshData {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
            ..MeshData::default()
        };
        assert_eq!(simplify_to(&one, &MeshExtras::new(), 0.1).0.triangle_count(), 1);
    }

    #[test]
    fn a_level_changes_past_its_threshold_only() {
        let s = LodSettings::default();
        assert_eq!((s.pick(200.0, 0), s.pick(40.0, 0), s.pick(5.0, 0)), (0, 1, 2));
        // just under 56 px the full mesh stays, and just over it the second level does
        assert_eq!((s.pick(54.0, 0), s.pick(58.0, 1)), (0, 1));
        assert_eq!((s.pick(49.0, 0), s.pick(63.0, 1)), (1, 0));
        // a bias moves the thresholds; off, it is always the full mesh
        let low = LodSettings { bias: 2.0, ..s };
        assert_eq!((low.pick(100.0, 0), low.pick(30.0, 0)), (1, 2));
        assert_eq!(LodSettings { enabled: false, ..s }.pick(1.0, 2), 0);
    }

    fn handles() -> [MeshHandle; 3] {
        let mut assets = RenderAssets::default();
        [
            assets.add_mesh(MeshData::cube(1.0)),
            assets.add_mesh(MeshData::cube(1.0)),
            assets.add_mesh(MeshData::cube(1.0)),
        ]
    }

    #[test]
    fn the_mesh_follows_the_size_on_screen() {
        let [full, mid, low] = handles();
        let mut world = World::new();
        let resources = Resources::default();
        // a camera at the origin looking down -Z, 720 px tall, 60 degrees: a radius-1 sphere at distance d is 624 / d px
        world.spawn((Camera3d::default(), GlobalTransform(Mat4::IDENTITY)));
        let at = |z: f32| GlobalTransform(Transform::from_xyz(0.0, 0.0, z).matrix());
        let lod = MeshLod::new(1.0, &[full, mid, low]);
        let near = world.spawn((lod, Mesh3d(full), at(-5.0)));
        let middle = world.spawn((lod, Mesh3d(full), at(-20.0)));
        let far = world.spawn((lod, Mesh3d(full), at(-100.0)));
        let plain = world.spawn((Mesh3d(full), at(-100.0)));
        // twice the size: the scale counts
        let big = world.spawn((
            lod,
            Mesh3d(full),
            GlobalTransform(Mat4::from_scale_rotation_translation(
                Vec3::splat(4.0),
                glam::Quat::IDENTITY,
                Vec3::new(0.0, 0.0, -20.0),
            )),
        ));
        let mesh = |world: &World, e| world.get::<&Mesh3d>(e).unwrap().0;
        select_lods(&mut world, &resources);
        assert_eq!(
            [near, middle, far, plain, big].map(|e| mesh(&world, e)),
            [full, mid, low, full, full]
        );
        // it comes nearer: finer again
        *world.get::<&mut GlobalTransform>(far).unwrap() = at(-8.0);
        select_lods(&mut world, &resources);
        assert_eq!(mesh(&world, far), full);
        // a second camera right next to the far one keeps it whole
        *world.get::<&mut GlobalTransform>(far).unwrap() = at(-100.0);
        select_lods(&mut world, &resources);
        assert_eq!(mesh(&world, far), low);
        world.spawn((Camera3d::default(), at(-97.0)));
        select_lods(&mut world, &resources);
        assert_eq!(mesh(&world, far), full);
        // switched off: everything whole
        let mut resources = Resources::default();
        resources.insert(LodSettings {
            enabled: false,
            ..LodSettings::default()
        });
        select_lods(&mut world, &resources);
        assert_eq!(mesh(&world, middle), full);
    }

    #[test]
    fn fewer_meshes_than_levels_stop_at_the_last() {
        let [full, mid, _] = handles();
        let mut world = World::new();
        world.spawn((Camera3d::default(), GlobalTransform(Mat4::IDENTITY)));
        let e = world.spawn((
            MeshLod::new(1.0, &[full, mid]),
            Mesh3d(full),
            GlobalTransform(Transform::from_xyz(0.0, 0.0, -500.0).matrix()),
        ));
        select_lods(&mut world, &Resources::default());
        assert_eq!(world.get::<&Mesh3d>(e).unwrap().0, mid);
        assert_eq!(MeshLod::new(1.0, &[full, mid]).meshes(), [full, mid]);
    }

    #[test]
    fn a_level_can_be_no_mesh() {
        let [full, mid, merged] = handles();
        let part = MeshLod::from_levels(2.0, &[Some(full), Some(mid), None]);
        assert_eq!(part.meshes(), [full, mid, MeshHandle::NONE]);
        assert_eq!((part.radius, part.level_of(MeshHandle::NONE)), (2.0, Some(2)));
        let carrier = MeshLod::from_levels(2.0, &[None, None, Some(merged)]);
        assert_eq!(carrier.meshes(), [MeshHandle::NONE, MeshHandle::NONE, merged]);
        // the same levels are the same component, whatever level each was last drawn at
        assert_eq!(
            MeshLod::from_levels(1.0, &[Some(full), Some(mid)]),
            MeshLod::new(1.0, &[full, mid])
        );
        assert_ne!(part, carrier);
        // more levels than there can be: the rest is left out
        let many = MeshLod::from_levels(1.0, &[None, Some(full), None, Some(mid), Some(merged)]);
        assert_eq!(many.meshes(), [MeshHandle::NONE, full, MeshHandle::NONE, mid]);
    }

    /// A model of two parts and a carrier of its merged mesh: far away the carrier alone is
    /// drawn, nearer the parts alone, and at no distance both or neither, going out and coming
    /// back (the two directions change at different distances: the hysteresis).
    #[test]
    fn siblings_swap_parts_for_the_merged_mesh_in_the_same_frame() {
        let [full, mid, merged] = handles();
        let mut world = World::new();
        let resources = Resources::default();
        // 720 px tall, 60 degrees: a radius-1 sphere at distance d is 624 / d px, so the far
        // level (under 20 px) starts near d = 31; with the hysteresis at 34.3 going out and at
        // 28.4 coming back
        world.spawn((Camera3d::default(), GlobalTransform(Mat4::IDENTITY)));
        let at = |z: f32| GlobalTransform(Transform::from_xyz(0.0, 0.0, -z).matrix());
        let part = MeshLod::from_levels(1.0, &[Some(full), Some(mid), None]);
        // a part too simple to have a simpler mesh repeats its full one
        let plain_part = MeshLod::from_levels(1.0, &[Some(full), Some(full), None]);
        let carrier = MeshLod::from_levels(1.0, &[None, None, Some(merged)]);
        let entities = [
            world.spawn((part, Mesh3d(full), at(5.0))),
            world.spawn((plain_part, Mesh3d(full), at(5.0))),
            world.spawn((carrier, Mesh3d(MeshHandle::NONE), at(5.0))),
        ];
        let step = |world: &mut World, d: f32| {
            for e in entities {
                *world.get::<&mut GlobalTransform>(e).unwrap() = at(d);
            }
            select_lods(world, &resources);
            let [a, b, c] = entities.map(|e| world.get::<&Mesh3d>(e).unwrap().0);
            let far = c == merged;
            assert_eq!(
                c.is_none(),
                !far,
                "the carrier draws the merged mesh or nothing (at {d})"
            );
            assert_eq!(
                (a.is_none(), b.is_none()),
                (far, far),
                "parts and carrier disagree at {d}"
            );
            assert!(far || (b == full && (a == full || a == mid)), "at {d}");
            far
        };
        assert!(!step(&mut world, 5.0));
        let distances: Vec<f32> = (0..=400).map(|i| 5.0 + i as f32 * 0.1).collect();
        let out = distances.iter().copied().find(|d| step(&mut world, *d)).unwrap();
        assert!((34.0..35.0).contains(&out), "merged from {out}");
        assert!(step(&mut world, 45.0));
        let back = distances.iter().rev().copied().find(|d| !step(&mut world, *d)).unwrap();
        assert!((27.5..28.5).contains(&back), "parts again from {back}");
        assert!(!step(&mut world, 5.0));
        // levels of detail off: the parts, never the merged mesh
        let mut off = Resources::default();
        off.insert(LodSettings {
            enabled: false,
            ..LodSettings::default()
        });
        for e in entities {
            *world.get::<&mut GlobalTransform>(e).unwrap() = at(100.0);
        }
        select_lods(&mut world, &resources);
        assert_eq!(world.get::<&Mesh3d>(entities[2]).unwrap().0, merged);
        select_lods(&mut world, &off);
        let [a, b, c] = entities.map(|e| world.get::<&Mesh3d>(e).unwrap().0);
        assert_eq!((a, b, c), (full, full, MeshHandle::NONE));
    }

    /// An entity added to a model that sits inside the hysteresis band starts from its own mesh
    /// and lands one level away from its siblings; started from a sibling's level it is with
    /// them from its first frame.
    #[test]
    fn a_late_sibling_starts_at_the_level_of_the_others() {
        let [full, mid, merged] = handles();
        let mut world = World::new();
        let resources = Resources::default();
        world.spawn((Camera3d::default(), GlobalTransform(Mat4::IDENTITY)));
        let at = |z: f32| GlobalTransform(Transform::from_xyz(0.0, 0.0, -z).matrix());
        let part = MeshLod::from_levels(1.0, &[Some(full), Some(mid), None]);
        let carrier_lod = MeshLod::from_levels(1.0, &[None, None, Some(merged)]);
        let first = world.spawn((part, Mesh3d(full), at(45.0)));
        let carrier = world.spawn((carrier_lod, Mesh3d(MeshHandle::NONE), at(45.0)));
        let mesh = |world: &World, e| world.get::<&Mesh3d>(e).unwrap().0;
        select_lods(&mut world, &resources);
        // back to 31: 20.1 px, inside the band of the 20 px threshold, so the far level stays
        for e in [first, carrier] {
            *world.get::<&mut GlobalTransform>(e).unwrap() = at(31.0);
        }
        select_lods(&mut world, &resources);
        assert_eq!((mesh(&world, first), mesh(&world, carrier)), (MeshHandle::NONE, merged));
        // a part added now picks the middle level by itself: drawn on top of the merged mesh
        let alone = world.spawn((part, Mesh3d(full), at(31.0)));
        // started from a sibling's level it draws nothing, as they do; and a carrier made
        // again goes on drawing the merged mesh
        let sibling = *world.get::<&MeshLod>(first).unwrap();
        let late = world.spawn((part.with_level_of(&sibling), Mesh3d(full), at(31.0)));
        *world.get::<&mut MeshLod>(carrier).unwrap() = carrier_lod.with_level_of(&sibling);
        select_lods(&mut world, &resources);
        assert_eq!(mesh(&world, alone), mid);
        assert_eq!((mesh(&world, late), mesh(&world, carrier)), (MeshHandle::NONE, merged));
        // from there it changes with the others
        for e in [first, carrier, late] {
            *world.get::<&mut GlobalTransform>(e).unwrap() = at(27.0);
        }
        select_lods(&mut world, &resources);
        assert_eq!(
            [first, late, carrier].map(|e| mesh(&world, e)),
            [mid, mid, MeshHandle::NONE]
        );
        // a component the renderer has not seen has no level to give: the entity starts from
        // its own mesh, as without it
        let unseen = world.spawn((part.with_level_of(&part), Mesh3d(full), at(31.0)));
        select_lods(&mut world, &resources);
        assert_eq!(mesh(&world, unseen), mid);
    }

    /// With a renderer, a level whose mesh is not uploaded is not drawn while another level
    /// is: the nearest uploaded one stands in, and the wanted one takes over when it arrives,
    /// also inside the hysteresis band, where a level read back from the stand-in would stick.
    #[test]
    fn a_level_that_is_not_uploaded_is_stood_in_for_by_the_nearest_that_is() {
        let renderer = match pollster::block_on(Renderer::new_headless(1280, 720)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping GPU test: no adapter available ({e})");
                return;
            }
        };
        let mut resources = Resources::default();
        resources.insert(renderer);
        resources.insert(RenderAssets::default());
        let upload = |resources: &mut Resources| {
            let mut assets = std::mem::take(resources.get_mut::<RenderAssets>().unwrap());
            resources.get_mut::<Renderer>().unwrap().upload_assets(&mut assets);
            *resources.get_mut::<RenderAssets>().unwrap() = assets;
        };
        let add = |resources: &mut Resources| {
            resources
                .get_mut::<RenderAssets>()
                .unwrap()
                .add_mesh(MeshData::cube(1.0))
        };
        // only the simplest level is on the GPU
        let low = add(&mut resources);
        upload(&mut resources);
        let (full, mid) = (add(&mut resources), add(&mut resources));
        let merged = add(&mut resources);

        let mut world = World::new();
        world.spawn((Camera3d::default(), GlobalTransform(Mat4::IDENTITY)));
        let at = |z: f32| GlobalTransform(Transform::from_xyz(0.0, 0.0, -z).matrix());
        let lod = MeshLod::new(1.0, &[full, mid, low]);
        // 624 / d px: 5 is well inside the full level, 11.5 (54 px) inside the band under its
        // threshold, where the full mesh is kept and the second level is not left for the
        // first, and 20 inside the second level
        let near = world.spawn((lod, Mesh3d(full), at(5.0)));
        let band = world.spawn((lod, Mesh3d(full), at(11.5)));
        let middle = world.spawn((lod, Mesh3d(full), at(20.0)));
        // a carrier whose only mesh is not uploaded has nothing to stand in: a level that
        // draws nothing is never chosen in place of a mesh
        let carrier = world.spawn((
            MeshLod::from_levels(1.0, &[None, None, Some(merged)]),
            Mesh3d(MeshHandle::NONE),
            at(100.0),
        ));
        // and a part far away draws nothing because that is its level, uploaded or not
        let part = world.spawn((
            MeshLod::from_levels(1.0, &[Some(full), Some(mid), None]),
            Mesh3d(full),
            at(100.0),
        ));
        let mesh = |world: &World, e| world.get::<&Mesh3d>(e).unwrap().0;
        for _ in 0..2 {
            select_lods(&mut world, &resources);
            assert_eq!([near, band, middle].map(|e| mesh(&world, e)), [low, low, low]);
            assert_eq!((mesh(&world, carrier), mesh(&world, part)), (merged, MeshHandle::NONE));
        }
        // everything arrives: each draws the level its size asks for, the one in the band its
        // full mesh (from the stand-in, the third level, it would have stopped at the second)
        upload(&mut resources);
        select_lods(&mut world, &resources);
        assert_eq!([near, band, middle].map(|e| mesh(&world, e)), [full, full, mid]);
        assert_eq!(mesh(&world, carrier), merged);

        // of two levels equally near the simpler one stands in; a nearer one wins over both
        let wanted = add(&mut resources);
        let lod = MeshLod::new(1.0, &[full, wanted, low]);
        let e = world.spawn((lod, Mesh3d(full), at(20.0)));
        select_lods(&mut world, &resources);
        assert_eq!(mesh(&world, e), low);
        let far = world.spawn((MeshLod::new(1.0, &[full, mid, wanted]), Mesh3d(full), at(100.0)));
        select_lods(&mut world, &resources);
        assert_eq!(mesh(&world, far), mid);
        // a mesh whose update waits still has its previous copy: it is drawn, not stood in for
        let assets = resources.get_mut::<RenderAssets>().unwrap();
        assert!(assets.update_mesh(mid, MeshData::cube(2.0)) && !assets.is_mesh_uploaded(mid));
        select_lods(&mut world, &resources);
        assert_eq!((mesh(&world, middle), mesh(&world, far)), (mid, mid));
        upload(&mut resources);
        select_lods(&mut world, &resources);
        assert_eq!((mesh(&world, e), mesh(&world, far)), (wanted, wanted));
        assert_eq!(resources.get::<Renderer>().unwrap().gpu_error_count(), 0);
    }
}
