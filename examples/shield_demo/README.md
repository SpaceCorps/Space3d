# shield_demo

A 3D shield for SpaceCorps 2027, as a Space3d example. Every ship gets a shield that hugs its hull. Each bolt makes a wave on it. The share of a hit that the shield doesn't absorb pierces it, and the shield breaks into fragments when it runs out. A group shield, the Aegis, stretches the carrier's shield into a bubble over its allies. Issue #57 records the design decisions.

![The demo: hull shields on Nomad, Storm and a Seeker, the panel, bolts in flight](screenshot.jpg)

## What it shows

- **Shield rendering:** a `RenderNode` (`src/node.rs`, Transparent phase, order 50) with its WGSL (`src/shield.wgsl`) draws:
  - the hull shields, one draw per ship with a 720-byte uniform at a dynamic offset;
  - the Aegis bubble with its ground ring and gauge;
  - the shield fragments, from one instanced ring buffer;
  - the bolts.

  It never writes depth, and it runs on WebGPU and WebGL2.
- **Hull envelopes:** a smooth shell at a constant distance from the hull, from a distance field. The envelope is read from a baked `.sbe` file when there is one (`src/sbe.rs`). Otherwise it is computed at load time (`src/envelope.rs`, 48 cells).
- **The game's rules** (`src/game.rs`, simulated in `Stage::FixedUpdate`):
  - Shield split: the shield takes `clamp(absorbance − penetration)` of each hit.
  - Weapons: lasers x1 to x4, Antimatter, Gemini, Lancet, blast rocket, Siphon.
  - Volleys: 2 or 4 bolts from the shooter's real gun positions.
  - Wear: one wear per volley, then break and regeneration.
  - Targets: missed shots and immune targets.
  - Siphon: the target's shield is sucked in towards the hit point, and the drained shield streams back to the shooter as packets. When the drain is yours (*Your shots*, at the other player or the mob), the probe leaves from your ship. Each packet then sends a teal wave over your shield, from where it enters (the side facing the target), and your shield takes in 7% once per drain (with wear; under the Aegis, into its reserve). If the Aegis goes up or down while the packets fly, they go to the shield you have when they land.
- **The Aegis:** a proposal, not a game rule yet. For 8 s, the carrier's whole shield becomes a bubble around its group, and its hull has no shield. The bubble takes its share of each hit from outside, and the rest goes on to the target. It has a reserve, a break, an end, a cooldown and fatigue.
- **The engine's quality tiers:** `GraphicsQuality::apply` and `ParticleSettings::for_quality`. The starting tier is `GraphicsQuality::recommended_for(renderer)`.
- **A French / English panel** (egui): ships, scene, target, weapon, shooter, wear, colours, shield look, sound, speed, orbit camera.

## Running it

Natively:

```bash
cargo run -p shield_demo --release
```

In the browser:

```bash
cargo build -p shield_demo --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-name shield_demo --out-dir examples/shield_demo/dist/web target/wasm32-unknown-unknown/release/shield_demo.wasm
cp examples/shield_demo/web/index.html examples/shield_demo/dist/web/
cp -r examples/shield_demo/assets examples/shield_demo/dist/
```

Serve `examples/shield_demo/dist/` with any static server and open `/web/`. Add `?webgl2` to the URL to take the WebGL2 path. The page looks for the models in `../assets`.

| Input | Action |
|---|---|
| Click in the view | Fire a volley from that side |
| `Space` | Fire a volley |
| `F` | Deploy the Aegis (needs the group scene) |
| `B` / `R` | Break / restore your shield |
| `P` | Pause |
| `1`–`4` | Low, Medium, High, Ultra |
| `Esc` | Quit (desktop) |

## The SpaceCorps assets

The game's ship models are committed in `assets/`: one `.glb` per hull and its baked envelope (`.sbe`). The game's sounds are not in this repository, so the demo is silent unless you add them.

```text
assets/
  Storm.glb   Storm.sbe   …               one .glb per hull, .sbe optional
  sfx/hit_shield_1.mp3  laser_x1_1.mp3 …  optional, names in SFX_NAMES (src/game.rs)
```

- **Where they are read from:**
  - `SPACE3D_SHIELD_DEMO_ASSETS`, a folder (a URL parameter on the web);
  - otherwise an `assets/` folder next to the executable;
  - otherwise this example's `assets/` folder, natively, and `../assets` next to the page on the web.
- **Hull names:** they are listed in `src/catalog.rs`:
  - Protos, Kitefin, Ostirion, Nomad, Paragon, Miyuki, Storm, Wraith, Ironclad;
  - seeker, goombah, facet, void, phantasm, bulwark, crystalys, tesseract, inert-mass, unwakened;
  - turret, excavator.
- **Envelopes:** a `.sbe` file is a baked envelope (88 cells, smoother). Without one, the envelope is computed when the model loads (48 cells).
- **Fallback:** if a model can't be read, its hull is the engine's test ship (`crates/space3d-stories/assets/ship.glb`).

## Scripted runs

Environment variables, or URL parameters on the web, script a run:

- **Weapon:** `SHIELD_WEAPON=x4`
- **Target:** `SHIELD_TARGET=player`. `SHIELD_WEAPON=siphon SHIELD_TARGET=mob` shows your Siphon's packets coming back into your shield (lower your shield and turn wear on in the panel to see the 7% gain).
- **Aegis:** `SHIELD_AEGIS_AT=<tick>`
- **State report:** `SHIELD_LOG_FRAME=<tick>` prints one line of state in the console, and in `window.__shield` on the web.

The full list is at the top of `src/main.rs`.

A headless capture, for example:

```bash
SPACE3D_HEADLESS=1 SHIELD_SHOT=out.png SHIELD_SHOT_FRAME=300 cargo run -p shield_demo --release
```

## Tests

`cargo test -p shield_demo`

The simulation tests load the test ship synchronously, then drive the simulation with `App::step_ticks`. They need no GPU. They cover:

- a volley making waves;
- wear down to a break, and restoring;
- the Aegis taking hits and ending;
- your Siphon refilling your shield, bringing it back from 0, and refilling the Aegis;
- determinism.

Another test checks that every committed model loads with its baked envelope.

## Known limits

- **Overlapping layers:** where the hull envelope folds over itself, two layers can add up. Nodes outside the Opaque phase must not write depth, so there is no depth pre-pass.
- **Where your Siphon's packets enter:** in the game your ship faces its target, so they come in at the nose. Here the ships turn slowly and don't face anything: they come in on the side facing the target. Under the Aegis they go to the bubble, towards the target, even when the target is inside it.
- **Bolts at 60 Hz:** bolts move on the fixed tick. Their drawn position is extrapolated to the displayed time.
