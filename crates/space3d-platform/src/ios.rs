//! iOS: the scene life cycle winit 0.30 does not have.
//!
//! From the iOS 27 SDK on, UIKit stops an app at launch when it has not adopted scenes
//! (`UIApplicationEvaluateRuntimeIssueForNoSceneLifecycleAdoption`, a trap; seen with Xcode 27.2).
//! winit 0.30 makes its `UIWindow` the old way, with no scene. So the engine brings the least
//! a scene app needs:
//!
//! * the app's `Info.plist` names a scene configuration whose delegate class is
//!   [`SCENE_DELEGATE_CLASS`] (`UIApplicationSceneManifest`; `scripts/ios-bundle.sh` writes it);
//! * that class is registered here before UIKit starts ([`register`]);
//! * when the scene connects, winit's window is given to it ([`attach`] from the runner once the
//!   window exists, the delegate's `scene:willConnectToSession:options:` once the scene does;
//!   whichever comes second joins the two).
//!
//! Everything else stays winit's: it listens to the application's notifications
//! (did-become-active, did-enter-background...), which UIKit still posts for a scene app.

use std::cell::RefCell;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2::{ClassType, DeclaredClass, declare_class, msg_send, msg_send_id, mutability};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

/// The Objective-C name of the scene delegate, for `UISceneDelegateClassName` in `Info.plist`.
pub const SCENE_DELEGATE_CLASS: &str = "Space3dSceneDelegate";

thread_local! {
    /// The connected `UIWindowScene` and winit's `UIWindow` (main thread only, as all of UIKit).
    static SCENE: RefCell<Option<Retained<AnyObject>>> = const { RefCell::new(None) };
    static WINDOW: RefCell<Option<Retained<AnyObject>>> = const { RefCell::new(None) };
}

declare_class!(
    struct SceneDelegate;

    unsafe impl ClassType for SceneDelegate {
        type Super = NSObject;
        type Mutability = mutability::InteriorMutable;
        const NAME: &'static str = "Space3dSceneDelegate";
    }

    impl DeclaredClass for SceneDelegate {}

    unsafe impl SceneDelegate {
        #[method(scene:willConnectToSession:options:)]
        fn scene_will_connect(&self, scene: &AnyObject, _session: &AnyObject, _options: &AnyObject) {
            // SAFETY: `scene` is a live object UIKit passes; retaining it is always allowed.
            let scene = unsafe { Retained::retain(scene as *const AnyObject as *mut AnyObject) };
            SCENE.with(|s| *s.borrow_mut() = scene);
            join();
        }

        #[method(sceneDidDisconnect:)]
        fn scene_did_disconnect(&self, _scene: &AnyObject) {
            SCENE.with(|s| *s.borrow_mut() = None);
        }
    }
);

/// Registers the scene delegate's class with the Objective-C runtime. Before the event loop
/// runs (`UIApplicationMain` looks the class up by its name when the scene connects).
pub(crate) fn register() {
    let class = SceneDelegate::class();
    // UIKit checks that the class conforms to the protocols (it throws otherwise). They are
    // added by name: objc2 0.5 has no UIKit protocol types without the whole UIKit crate.
    for name in ["UISceneDelegate", "UIWindowSceneDelegate"] {
        match objc2::runtime::AnyProtocol::get(name) {
            // SAFETY: a registered class and a registered protocol; adding one a class has
            // already is a no-op that returns false.
            Some(protocol) => unsafe {
                objc2::ffi::class_addProtocol(
                    (class as *const objc2::runtime::AnyClass).cast_mut().cast(),
                    (protocol as *const objc2::runtime::AnyProtocol).cast(),
                );
            },
            None => tracing::error!("iOS: the Objective-C protocol {name} is not known"),
        }
    }
}

/// The runner's window exists: remember its `UIWindow` and give it to the scene if one is
/// connected already.
pub(crate) fn attach(window: &winit::window::Window) {
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::UiKit(handle) = handle.as_raw() else {
        return;
    };
    let view = handle.ui_view.as_ptr().cast::<AnyObject>();
    // SAFETY: `ui_view` is winit's live `UIView`; `window` is its `UIWindow` (or nil).
    let ui_window: Option<Retained<AnyObject>> = unsafe { msg_send_id![view, window] };
    WINDOW.with(|w| *w.borrow_mut() = ui_window);
    join();
}

/// `UIEdgeInsets`, as `-[UIView safeAreaInsets]` returns it (points).
#[repr(C)]
#[derive(Clone, Copy)]
struct EdgeInsets {
    top: f64,
    left: f64,
    bottom: f64,
    right: f64,
}

// SAFETY: four `CGFloat`s (64-bit on every iOS target Rust has) in UIKit's order.
unsafe impl objc2::Encode for EdgeInsets {
    const ENCODING: objc2::Encoding = objc2::Encoding::Struct(
        "UIEdgeInsets",
        &[
            objc2::Encoding::Double,
            objc2::Encoding::Double,
            objc2::Encoding::Double,
            objc2::Encoding::Double,
        ],
    );
}

/// The parts of the screen the system keeps ([`crate::window::SafeArea`]), in physical pixels:
/// the view's own insets, as UIKit gives them.
///
/// Not right yet in landscape, in the Simulator at least (iPhone 18 Pro, iOS 27.2): UIKit
/// answers 62 points at the top and 34 at the bottom, the portrait values, for a landscape
/// scene whose island is at the side; the window's insets say the same and a forced layout
/// changes nothing. Unknown on a device.
pub(crate) fn safe_area(window: &winit::window::Window) -> crate::window::SafeArea {
    let Ok(handle) = window.window_handle() else {
        return crate::window::SafeArea::default();
    };
    let RawWindowHandle::UiKit(handle) = handle.as_raw() else {
        return crate::window::SafeArea::default();
    };
    let view = handle.ui_view.as_ptr().cast::<AnyObject>();
    // SAFETY: `ui_view` is winit's live `UIView`, on the main thread.
    let insets: EdgeInsets = unsafe { msg_send![view, safeAreaInsets] };
    let scale = window.scale_factor();
    let px = |points: f64| (points.max(0.0) * scale).round() as u32;
    crate::window::SafeArea {
        left: px(insets.left),
        top: px(insets.top),
        right: px(insets.right),
        bottom: px(insets.bottom),
    }
}

/// Puts the window in the scene once both exist. Without it a scene app's window never shows.
fn join() {
    let scene = SCENE.with(|s| s.borrow().clone());
    let window = WINDOW.with(|w| w.borrow().clone());
    let (Some(scene), Some(window)) = (scene, window) else {
        return;
    };
    // SAFETY: both are live UIKit objects on the main thread; `setWindowScene:` takes a
    // `UIWindowScene`, which a scene of the application role is.
    unsafe {
        let _: () = msg_send![&*window, setWindowScene: &*scene];
        let _: () = msg_send![&*window, makeKeyAndVisible];
    }
    tracing::info!("iOS: the window joined its scene");
}
