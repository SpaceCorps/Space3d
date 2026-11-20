use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self {
            translation: Vec3::new(x, y, z),
            ..Self::IDENTITY
        }
    }
    pub fn with_rotation(mut self, r: Quat) -> Self {
        self.rotation = r;
        self
    }
    pub fn with_scale(mut self, s: Vec3) -> Self {
        self.scale = s;
        self
    }
    pub fn looking_at(mut self, target: Vec3, up: Vec3) -> Self {
        let m = glam::camera::rh::view::look_at_mat4(self.translation, target, up).inverse();
        self.rotation = Quat::from_mat4(&m);
        self
    }
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }
}

/// World-space matrix, written by the core `PostUpdate` transform system.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlobalTransform(pub Mat4);

/// Optional parent link; child `GlobalTransform = parent.Global * child.Transform`.
#[derive(Clone, Copy, Debug)]
pub struct Parent(pub hecs::Entity);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Name(pub String);

/// A child of the hierarchy: entity, parent, local matrix, and whether the parent is itself a
/// child (only those can need more than one pass).
type ChildEntry = (hecs::Entity, hecs::Entity, Mat4, bool);

thread_local! {
    /// Scratch of [`propagate_transforms`], kept between calls so a steady-state frame with
    /// parented entities allocates nothing.
    static CHILDREN: std::cell::Cell<Vec<ChildEntry>> = const { std::cell::Cell::new(Vec::new()) };
    /// The query behind [`propagate_transforms`]' view of every `GlobalTransform`, kept between calls for the same reason: it is
    /// made again only when the world's set of archetypes changes.
    static GLOBALS: std::cell::Cell<Option<hecs::PreparedQuery<&'static mut GlobalTransform>>> =
        const { std::cell::Cell::new(None) };
}

/// Propagates `Transform` → `GlobalTransform` (supports one-level-deep and
/// deeper hierarchies via iterative resolution). Allocates only while entities gain their
/// `GlobalTransform` or the hierarchy outgrows its largest size so far.
pub fn propagate_transforms(world: &mut hecs::World) {
    // Roots first.
    let mut missing = Vec::new();
    for (e, t, g) in world
        .query_mut::<(hecs::Entity, &Transform, Option<&mut GlobalTransform>)>()
        .without::<&Parent>()
    {
        match g {
            Some(g) => g.0 = t.matrix(),
            None => missing.push((e, GlobalTransform(t.matrix()))),
        }
    }
    for (e, g) in missing.drain(..) {
        let _ = world.insert_one(e, g);
    }
    // Children. One pass settles every child of a root (the common case: the parts of a
    // model under its root), the parent's matrix kept from one sibling to the next. Only children of children are resolved again, until nothing
    // changes (depth-bounded).
    let mut children = CHILDREN.take();
    children.clear();
    {
        let mut last: Option<(hecs::Entity, bool)> = None;
        children.extend(
            world
                .query::<(hecs::Entity, &Parent, &Transform)>()
                .iter()
                .map(|(e, p, t)| {
                    let nested = match last {
                        Some((parent, nested)) if parent == p.0 => nested,
                        _ => {
                            let nested = world.get::<&Parent>(p.0).is_ok();
                            last = Some((p.0, nested));
                            nested
                        }
                    };
                    (e, p.0, t.matrix(), nested)
                }),
        );
    }
    let any_nested = children.iter().any(|c| c.3);
    let mut prepared = GLOBALS.take().unwrap_or_default();
    for depth in 0..32 {
        let mut changed = false;
        {
            // Every `GlobalTransform` of the world through one view: a child's and its parent's are found by the entity alone, without
            // the archetype's column and its borrow being looked up again for each (with the 3,600 parts of a wave of 200 aliens that
            // lookup was most of this system).
            let mut globals = prepared.view_mut(world);
            let mut last: Option<(hecs::Entity, Option<Mat4>)> = None;
            for &(e, parent, local, nested) in &children {
                if depth > 0 && !nested {
                    continue;
                }
                let pg = match last {
                    Some((p, pg)) if p == parent => pg,
                    _ => {
                        let pg = globals.get_mut(parent).map(|g| g.0);
                        last = Some((parent, pg));
                        pg
                    }
                };
                let Some(pg) = pg else {
                    continue;
                };
                let new = pg * local;
                match globals.get_mut(e) {
                    Some(g) => {
                        if g.0 != new {
                            g.0 = new;
                            changed = true;
                        }
                    }
                    None => missing.push((e, GlobalTransform(new))),
                }
            }
        }
        for (e, g) in missing.drain(..) {
            let _ = world.insert_one(e, g);
            changed = true;
        }
        if !changed || !any_nested {
            break;
        }
    }
    CHILDREN.set(children);
    GLOBALS.set(Some(prepared));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parts of a model follow its root, a part of a part follows both, a child whose parent is gone stays where it was, and a
    /// second world's hierarchy is settled as well as the first's (the view's query is kept between calls, and made again for a world
    /// whose archetypes are others).
    #[test]
    fn children_follow_their_parents_in_every_world() {
        let mut world = hecs::World::new();
        let root =
            world
                .spawn((Transform::from_xyz(10.0, 0.0, 0.0)
                    .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),));
        let part = world.spawn((Transform::from_xyz(0.0, 0.0, -2.0), Parent(root)));
        let tip = world.spawn((Transform::from_xyz(0.0, 1.0, 0.0), Parent(part), Name("tip".into())));
        let orphan_parent = world.spawn((Transform::from_xyz(0.0, 5.0, 0.0),));
        let orphan = world.spawn((Transform::from_xyz(1.0, 0.0, 0.0), Parent(orphan_parent)));
        let at = |world: &hecs::World, e| world.get::<&GlobalTransform>(e).unwrap().0.w_axis.truncate();
        propagate_transforms(&mut world);
        // (a quarter turn about Y takes -Z to -X)
        assert!(
            at(&world, part).distance(Vec3::new(8.0, 0.0, 0.0)) < 1e-5,
            "{}",
            at(&world, part)
        );
        assert!(
            at(&world, tip).distance(Vec3::new(8.0, 1.0, 0.0)) < 1e-5,
            "{}",
            at(&world, tip)
        );
        assert!(at(&world, orphan).distance(Vec3::new(1.0, 5.0, 0.0)) < 1e-5);
        // The root moves: both follow in one call; the orphan's parent is despawned: it keeps its last place
        world.get::<&mut Transform>(root).unwrap().translation.y = 3.0;
        world.despawn(orphan_parent).unwrap();
        propagate_transforms(&mut world);
        assert!(
            at(&world, tip).distance(Vec3::new(8.0, 4.0, 0.0)) < 1e-5,
            "{}",
            at(&world, tip)
        );
        assert!(at(&world, orphan).distance(Vec3::new(1.0, 5.0, 0.0)) < 1e-5);
        // Another world, with other archetypes, between two calls for the first
        let mut other = hecs::World::new();
        let a = other.spawn((Name("a".into()), Transform::from_xyz(1.0, 2.0, 3.0)));
        let b = other.spawn((Transform::from_xyz(1.0, 0.0, 0.0), Parent(a), Name("b".into())));
        propagate_transforms(&mut other);
        assert!(at(&other, b).distance(Vec3::new(2.0, 2.0, 3.0)) < 1e-5);
        world.get::<&mut Transform>(part).unwrap().translation.z = -4.0;
        propagate_transforms(&mut world);
        assert!(
            at(&world, tip).distance(Vec3::new(6.0, 4.0, 0.0)) < 1e-5,
            "{}",
            at(&world, tip)
        );
    }
}
