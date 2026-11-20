//! What must happen at once when the app goes to the background or comes back.
//!
//! On a phone no frame runs between the two (the platform runner stops them with the
//! surface), so a system that reads an event would hear of the suspension only after the
//! resume. A hook runs inside the platform's own callback instead.

use crate::Resources;

type Hook = Box<dyn Fn(&mut Resources, bool) + Send + Sync>;

/// The hooks (resource). A plugin adds one with [`BackgroundHooks::add`]; the platform runner
/// calls [`BackgroundHooks::run`] with `true` when the app is suspended and `false` when it
/// is resumed (also at the first start).
#[derive(Default)]
pub struct BackgroundHooks(Vec<Hook>);

impl BackgroundHooks {
    /// Add a hook: `hook(resources, in_background)`.
    pub fn add(&mut self, hook: impl Fn(&mut Resources, bool) + Send + Sync + 'static) {
        self.0.push(Box::new(hook));
    }

    /// Run every hook, in the order they were added. A hook added by a hook is kept for the
    /// next time.
    pub fn run(resources: &mut Resources, in_background: bool) {
        let Some(hooks) = resources.remove::<Self>() else {
            return;
        };
        for hook in &hooks.0 {
            hook(resources, in_background);
        }
        let added = resources.insert(hooks);
        if let Some(mut added) = added {
            resources.init::<Self>().0.append(&mut added.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hooks_run_in_order_with_the_resources() {
        let mut resources = Resources::default();
        resources.insert(Vec::<(u8, bool)>::new());
        let hooks = resources.init::<BackgroundHooks>();
        hooks.add(|r, bg| r.expect_mut::<Vec<(u8, bool)>>().push((1, bg)));
        hooks.add(|r, bg| {
            r.expect_mut::<Vec<(u8, bool)>>().push((2, bg));
            // A hook may add a hook; it runs from the next time on.
            if bg {
                r.init::<BackgroundHooks>()
                    .add(|r, bg| r.expect_mut::<Vec<(u8, bool)>>().push((3, bg)));
            }
        });
        BackgroundHooks::run(&mut resources, true);
        BackgroundHooks::run(&mut resources, false);
        assert_eq!(
            resources.expect::<Vec<(u8, bool)>>(),
            &vec![(1, true), (2, true), (1, false), (2, false), (3, false)]
        );
        // No hooks at all: nothing happens.
        BackgroundHooks::run(&mut Resources::default(), true);
    }
}
