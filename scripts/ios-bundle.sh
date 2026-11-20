#!/usr/bin/env bash
# An iOS .app bundle from a Rust executable, without an Xcode project (docs/platforms.md, "iOS").
#
#   scripts/ios-bundle.sh <executable> <name> <bundle id> <out dir> [assets dir] [version]
#
# <executable>: what cargo linked for aarch64-apple-ios-sim (the Simulator) or aarch64-apple-ios
#   (a device); a Rust `main` runs the app, cargo links the frameworks itself.
# [assets dir]: copied into the bundle as `assets/` (the engine's default asset root on iOS is
#   the bundle's `assets` folder).
# The bundle is <out dir>/<name>.app, unsigned: enough for `xcrun simctl install`. A device
# needs a signing identity and a provisioning profile (codesign), which this script does not do.
#
# Example (the engine's own proof; SIM is the device's name or UDID):
#   cargo rustc -p hello_cube_mobile --lib --crate-type bin --target aarch64-apple-ios-sim
#   scripts/ios-bundle.sh target/aarch64-apple-ios-sim/debug/hello_cube_mobile HelloCube \
#     org.spacecorps.space3d.hellocube target/ios examples/hello_cube_mobile/assets
#   xcrun simctl boot "$SIM"; xcrun simctl install "$SIM" target/ios/HelloCube.app
#   xcrun simctl launch --console-pty "$SIM" org.spacecorps.space3d.hellocube
set -euo pipefail

exe="${1:?the executable}"
name="${2:?the app name}"
id="${3:?the bundle id}"
out="${4:?the output folder}"
assets="${5:-}"
version="${6:-0.1.0}"

# The platform the executable was linked for (LC_BUILD_VERSION: 2 = iOS, 7 = the Simulator).
platform="$(xcrun vtool -show-build "$exe" | awk '$1 == "platform" { print $2; exit }')"
case "$platform" in
  IOSSIMULATOR | 7) sdk=iPhoneSimulator ;;
  IOS | 2) sdk=iPhoneOS ;;
  *) echo "$exe is not an iOS executable (platform: ${platform:-unknown})" >&2; exit 2 ;;
esac
minos="$(xcrun vtool -show-build "$exe" | awk '$1 == "minos" { print $2; exit }')"

app="$out/$name.app"
rm -rf "$app"
mkdir -p "$app"
cp "$exe" "$app/$name"
if [ -n "$assets" ]; then
  # -c: clones on APFS (no second copy of the bytes on disk).
  cp -cR "$assets" "$app/assets" 2>/dev/null || cp -R "$assets" "$app/assets"
fi

cat > "$app/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>$name</string>
  <key>CFBundleIdentifier</key><string>$id</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>$name</string>
  <key>CFBundleDisplayName</key><string>$name</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleSupportedPlatforms</key><array><string>$sdk</string></array>
  <key>LSRequiresIPhoneOS</key><true/>
  <key>MinimumOSVersion</key><string>${minos:-13.0}</string>
  <key>UIDeviceFamily</key><array><integer>1</integer><integer>2</integer></array>
  <!-- A launch screen (an empty one): without it iOS runs the app letterboxed at an old size. -->
  <key>UILaunchScreen</key><dict/>
  <!-- The scene life cycle (required from the iOS 27 SDK on); the delegate is the engine's
       class (crates/space3d-platform/src/ios.rs). -->
  <key>UIApplicationSceneManifest</key>
  <dict>
    <key>UIApplicationSupportsMultipleScenes</key><false/>
    <key>UISceneConfigurations</key>
    <dict>
      <key>UIWindowSceneSessionRoleApplication</key>
      <array>
        <dict>
          <key>UISceneConfigurationName</key><string>Default</string>
          <key>UISceneDelegateClassName</key><string>Space3dSceneDelegate</string>
        </dict>
      </array>
    </dict>
  </dict>
  <key>UIRequiresFullScreen</key><true/>
  <key>UIStatusBarHidden</key><true/>
  <key>UIViewControllerBasedStatusBarAppearance</key><false/>
  <key>UISupportedInterfaceOrientations</key>
  <array>
    <string>UIInterfaceOrientationLandscapeLeft</string>
    <string>UIInterfaceOrientationLandscapeRight</string>
  </array>
  <key>UISupportedInterfaceOrientations~ipad</key>
  <array>
    <string>UIInterfaceOrientationLandscapeLeft</string>
    <string>UIInterfaceOrientationLandscapeRight</string>
  </array>
</dict>
</plist>
EOF
plutil -lint "$app/Info.plist" >/dev/null
echo "$app ($sdk, iOS $minos or later, $(du -sh "$app" | cut -f1))"
