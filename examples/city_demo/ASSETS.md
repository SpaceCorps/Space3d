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

The Low tier halves the textures in code and uses the 1k HDRI.
Not from assets (procedural, no suitable CC0 source used): facades and windows, shopfronts,
cars, lamps, signals, trees, pedestrians, steam and rain.
