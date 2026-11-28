# Assets of `city_demo`

Everything below is **CC0 1.0** (public domain), from [Poly Haven](https://polyhaven.com). The
licence was checked on https://polyhaven.com/license ("Our assets are all licensed as CC0")
and each asset's entry at `https://api.polyhaven.com/info/<id>` (authors below). Downloaded
with the public API (`https://api.polyhaven.com/files/<id>`), no account. The files were
copied unchanged into `assets/`; the demo embeds them with `include_bytes!` and decodes them
at start-up (`src/real.rs`). Anything that fails to decode falls back to the procedural
version in `src/procgen.rs`.

| asset | URL | author | files used | used for |
|---|---|---|---|---|
| Joburg Central Sunset (`sunset_jhbcentral`) | https://polyhaven.com/a/sunset_jhbcentral | Greg Zaal (HDRI), Dimitrios Savva (photography) | `assets/hdri/sunset_jhbcentral_2k.hdr` (Medium and up), `..._1k.hdr` (Low / web) | skybox and image-based light; its sun sets the sun's elevation |
| Asphalt 02 (`asphalt_02`) | https://polyhaven.com/a/asphalt_02 | Rob Tuytel | `assets/tex/asphalt_02_{Diffuse,arm,nor_gl}.jpg` (1k) | roads (darkened and made smoother in code: wet) |
| Concrete Pavement (`concrete_pavement`) | https://polyhaven.com/a/concrete_pavement | Charlotte Baglioni | `assets/tex/concrete_pavement_{Diffuse,arm,nor_gl}.jpg` (1k) | pavements |
| Bitumen (`bitumen`) | https://polyhaven.com/a/bitumen | Rob Tuytel | `assets/tex/bitumen_{Diffuse,arm,nor_gl}.jpg` (1k) | roofs |

| Car Kit 3.1 (Kenney) | https://kenney.nl/assets/car-kit, downloaded from https://opengameart.org/content/car-kit (page states CC0, uploader Kenney) | Kenney | `assets/models/kenney_{sedan,sedan-sports,hatchback-sports,taxi,suv,suv-luxury,van,delivery,truck,police}.glb` (GLB repacked with its `colormap.png` embedded) | moving and parked cars; head and tail lights are my emissive overlay |
| Fire Hydrant (`fire_hydrant`) | https://polyhaven.com/a/fire_hydrant | Gonçalo Felicio | `assets/models/polyhaven_fire_hydrant.glb` (1k textures, packed from the glTF) | hydrants near the flythrough |
| Metal Trash Can (`metal_trash_can`) | https://polyhaven.com/a/metal_trash_can | GurJas Studios | `assets/models/polyhaven_metal_trash_can.glb` | bins near the flythrough |
| Painted Wooden Bench (`painted_wooden_bench`) | https://polyhaven.com/a/painted_wooden_bench | Kirill Sannikov | `assets/models/polyhaven_painted_wooden_bench.glb` | benches near the flythrough |

The Kenney kit's `License.txt` says "Creative Commons Zero, CC0". The Poly Haven glTFs were
packed into binary GLB (external `.bin` and 1k JPEG textures embedded) with a small script; the
geometry and textures are unchanged. The Low tier halves the textures in code and uses the 1k HDRI.
Procedural (no suitable CC0 source found or reachable): facades and windows, shopfronts, the
bus, lamps, signals, trees, pedestrians, steam and rain. The procedural cars are also the
fallback when a model fails to load.
