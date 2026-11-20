# UI

`space3d-ui` runs egui inside the engine: `UiPlugin` translates input in `Stage::PreUpdate`, UI
systems (`app.add_ui_system`, `app.add_root_ui_system`) run in `Stage::Update`, and the frame is
tessellated in `Stage::PostUpdate` and drawn by the renderer's overlay hook (so it shows up in
headless screenshots). `UiPlugin::deterministic()` gives reproducible pixels for golden tests.
Pointer, focus and egui plugin details: [platforms.md, "Mouse buttons, focus and the UI"](platforms.md#mouse-buttons-focus-and-the-ui).

## Theming: the compact density

egui's default sizes are generous for tool panels and HUDs. The engine ships a **compact
preset** — dense editor panels in the spirit of Blender or Unity that stay readable at 1 pixel
per point — and it is strictly **opt-in**:

```rust
app.insert_resource(space3d_ui::UiTheme::compact()); // space3d::ui::UiTheme with the facade
```

* **Without the resource the app's UI does not change.** The engine never writes egui's `Style`
  unless a `UiTheme` resource exists: the UI of an app that does not opt in renders exactly as
  before, pixel for pixel.
* **The engine's dev tools are compact in every app.** The F3 overlay, F4 profiler and console
  are the engine's tools, not the app's UI: with or without a `UiTheme` they have their own
  compact look, and they get it by restyling only their own windows (a scoped copy of the style
  on their own `Ui`s) — never the global style, never the app's windows. See
  [In-game tools](#in-game-tools).
* **Applied once, between passes.** `UiPlugin` applies the theme when the resource is inserted or
  changed, right before the next egui pass begins, to both the dark and the light style. In every
  other frame it costs one comparison and no allocation (`UiThemeSync`; tokens compare bit for
  bit, so even a NaN token counts as unchanged); there is no per-frame style churn.
* **Sizes only.** The theme writes the tokens below. Colors, fonts (families), shadows, animation
  time and every other `Style` field keep the values the context has, so the preset also works
  on top of a game's own look.
* **Removing the resource** leaves the style as it is (the engine does not restore anything).
  Inserting a theme again applies it again.

`UiTheme::Compact(UiDensity)` is the engine preset (`UiTheme::compact()`, possibly tweaked);
`UiTheme::Custom(UiDensity)` names a density of your own. Both are applied the same way:

```rust
use space3d_ui::{UiDensity, UiTheme};

app.insert_resource(UiTheme::Custom(UiDensity {
    body_size: 13.0,
    button_size: 13.0,
    ..UiDensity::compact()
}));
```

### Tokens

| token | egui `Style` field | compact | egui default | why |
|---|---|---|---|---|
| `item_spacing` | `spacing.item_spacing` | 6 × 2 | 8 × 3 | rows 1 pt closer, columns 2 pt |
| `button_padding` | `spacing.button_padding` | 4 × 1 | 4 × 1 | already tight |
| `window_margin` | `spacing.window_margin` | 4 | 6 | also the engine tools' panel margin |
| `menu_margin` | `spacing.menu_margin` | 3 | 6 | menus, popups, tooltips |
| `control_height` | `spacing.interact_size.y` | 18 | 18 | the smallest hit target stays 18 pt |
| `control_min_width` | `spacing.interact_size.x` | 32 | 40 | narrower drag values / color buttons |
| `indent` | `spacing.indent` | 14 | 18 | tree levels, collapsing bodies |
| `icon_size` | `spacing.icon_width` (inner: 4/7) | 12 | 14 | checkbox / radio / collapse icons |
| `icon_spacing` | `spacing.icon_spacing` | 4 | 4 | |
| `slider_width` | `spacing.slider_width` | 90 | 100 | |
| `combo_width` | `spacing.combo_width` | 84 | 100 | |
| `text_edit_width` | `spacing.text_edit_width` | 160 | 280 | egui's 280 pt fields waste the most room |
| `scroll_bar_width` | `spacing.scroll.bar_width` | 8 | 10 | |
| `heading_size` | `TextStyle::Heading` | 14 | 18 | window titles too |
| `body_size` | `TextStyle::Body` | 12.5 | 13 | readable at 1 ppp |
| `button_size` | `TextStyle::Button` | 12.5 | 13 | also drag values |
| `small_size` | `TextStyle::Small` | 10 | 9 | 9 pt is hard to read; 10 still small |
| `monospace_size` | `TextStyle::Monospace` | 12 | 13 | |
| `widget_radius` | `visuals.widgets.*.corner_radius` | 2 | 2 (3 hovered) | one radius for every state |
| `window_radius` | `visuals.window_corner_radius` | 4 | 6 | |
| `menu_radius` | `visuals.menu_corner_radius` | 3 | 6 | |
| `table_row_height` | — (read by tables) | 16 | — (18) | text rows; rows with controls stay 18 |
| `tooltip_delay` | `interaction.tooltip_delay` | 0.3 s | 0.5 s | snappier tool tips |

`UiDensity::egui_default()` holds the right-hand column. egui's `Style` has no table row height:
tables read it with `UiDensity::row_height(ui)` (the applied theme's `table_row_height`, or the
style's control height without a theme) or `UiDensity::of(ctx)`. Both only report a theme that
still holds: after the `UiTheme` resource is removed, or once the game changed the style's sizes
after the theme was applied (colors do not count), `of` is `None` and `row_height` follows the
style again:

```rust
egui::Grid::new("assets")
    .striped(true)
    .min_row_height(UiDensity::row_height(ui))
    .show(ui, |ui| { /* ... */ });
```

The pure functions are public too: `compact_style(&base, &density) -> egui::Style`,
`apply_density(&mut style, &density)` and `apply_theme(&ctx, &theme)`. For tool panels,
`visible_scroll_bars(width)` is a `ScrollStyle` whose bars stay visible while content is hidden,
keep a column of their own and draw the handle in the text color (so it shows on any palette):
`ui.spacing_mut().scroll = visible_scroll_bars(8.0)`.

### How much denser

`ThemePreview` is a representative tool panel (menu bar, toolbar, buttons, checkbox, slider,
combo box, text field, a collapsing section and a 10-row striped table). Its width is that of its
"Space [combo] Name [text field]" row at the style's default widths (`ThemePreview::panel_width`),
and every other row fills it: the slider rail, the transform fields and the table's columns
(numbers and their headers right-aligned). With the compact preset it takes **338 × 354 pt
instead of 487 × 390 pt — 63.1 % of the area** — with every widget whole, no text wrapped or
clipped and every control at least 18 pt tall (`crates/space3d-ui/tests/theme.rs`). The stories
`Engine/UI/Theme/Default` and `Engine/UI/Theme/Compact` show the same panel side by side, each in
an `egui::Window`.

### Window width

The theme holds sizes of egui's `Style`; the width of a window is not one of them. An
`egui::Window` is **340 pt wide by default whatever it holds** (its contents fill that width), so
a small game window ("Cargo hold: 12 t [Jettison]") keeps a lot of empty space under any theme,
`UiTheme::compact()` included. That width is the game's choice, not the theme's: the engine does
not resize an app's windows (its own tools size themselves). To size game windows compactly:

* `.auto_sized()`: the window takes the width of its contents. Give the contents a width
  (`ui.set_width(..)`, fixed-width widgets): widgets that fill the available width — separators,
  right-to-left rows, `TextEdit::desired_width(f32::INFINITY)` — stretch an auto-sized window to
  the screen edge (its first frame spans the screen, so pin it with `fixed_pos` or keep it clear
  of the edges);
* or `.default_width(w)` (with `.resizable(false)` for a fixed width).

```rust
// A small HUD window as wide as "Salvage: 12 t [Jettison]", not 340 pt.
egui::Window::new("Cargo hold")
    .fixed_pos([520.0, 300.0])
    .auto_sized()
    .show(ctx, |ui| {
        ui.label("Salvage: 12 t");
        let _ = ui.button("Jettison");
    });
```

### Games that style egui themselves

A game that sets egui's `Style` itself (say, once at startup) needs to do nothing: without a
`UiTheme` resource the engine never touches the style. Should it want the compact sizes under
its own colors and fonts, it has two options:

* insert `UiTheme::compact()` (or a `Custom` density): the tokens are written over the game's
  style at the start of the next frame, once. A game that restyles later is not overwritten —
  the theme is applied again only when the resource changes (or call `apply_theme` yourself);
* or build its style from the preset: `compact_style(&my_style, &UiDensity::compact())`.

## In-game tools

The debug overlay (**F3**), profiler (**F4**) and console (**`**) show up inside games, so they
never read their sizes from the global style and never change it. They are the engine's tools,
not the app's UI: they are compact **in every app**, with or without a `UiTheme`. Each applies
the compact density to its own `Ui` only (cached copies of the style, none per frame), keeps the
context's colors, and draws its own compact header (egui's title bar height follows the global
heading size). Their size and layout are the same under egui's default, the compact theme or a
game's roomy style; only the colors follow the game's (story `Engine/UI/Tools/Game Style`,
`crates/space3d-ui/tests/tools.rs`). Text people read — the header summaries, counts, log
targets — is in the normal text color. Weak text (trace / debug log lines, placeholders) takes an
opacity derived from the palette: the least, from the palette's own `weak_text_alpha` up to 0.85,
at which it reaches 65 % of the normal text's contrast on the window fill (at least 3:1, at most
4.5:1) — about 0.78 on egui's greys (3.4:1 where egui's 0.6 gives 2.4:1), and egui's 0.6 under a
bright game palette, where that already reads at 5.9:1 and stays clearly dimmer than the 16:1 of
normal text. Scroll bars stay visible while content is hidden, with the handle in the text color
(`visible_scroll_bars`).

### Docking

The tools dock to the corners of the **dock area**, 6 pt in from its edges (and from each other):
the overlay top-left, the profiler top-right, the console bottom-left (in an area too short for
that, side by side: [Short dock areas](#short-dock-areas)). The dock area is the whole
screen, or the `ToolDockArea` resource when a host sets it: hosts that cover part of the screen
with their own panels set it every frame to what is left, and the tools then never cover their
menu bars or side panels — the editor sets it to its viewport (between the menu bar, hierarchy
and inspector; the whole screen while F1 hides the editor), the storybook viewer to the story
viewport. An area partly off screen is clipped to the screen; one too small to hold a tool window
is ignored. A game with panels of its own can do the same:

```rust
app.add_root_ui_system(|ui: &mut egui::Ui, sys: &mut SystemCtx| {
    egui::Panel::right("inventory").show(ui, |ui| { /* ... */ });
    // The tools draw early in the frame: they use the area set in the previous one.
    space3d_ui::set_tool_dock_area(sys.resources, ui.available_rect_before_wrap());
});
```

* **Docked tools follow the dock area.** When the window or a panel is resized — larger or
  smaller — the docked corner goes to the new corner, and a tool the user has not resized takes
  its default size for the new area: the same layout as an app started at that size (the console
  40 % of the new height; a profiler above it, the room left over it). A tool the user resized
  — larger or smaller — keeps that size, capped to the area and to the room the other tools
  leave it, and gets it back when the area grows again. A resize changes only the edges it drags:
  dragging the console's right edge leaves its height (a whole number of log lines) as it was.
* **Dragging a tool undocks it**: it stays where it was put (inside the screen) until its title is
  double-clicked, which docks it again. Clicks inside a docked tool, or resizing it by a free edge
  (the console's top or right edge, the profiler's left edge), leave it docked; resizing it by a
  docked edge undocks it.
* **Above the game's windows**: the tool windows sit on egui's foreground layer (with menus and
  popups, under tooltips), so a game window (egui's middle layer) never draws over them. Windows
  that must be able to come in front of the tools share that layer: the editor's texture and model
  editors do. Within the layer the window opened or clicked last is in front — clicking an editor
  window brings it over F3 / F4 / the console, clicking a tool brings the tool back. The debug
  overlay never takes the pointer: a window opened or clicked after it covers it (toggle F3 twice
  to bring it back on top).

The profiler is exactly as tall as its contents, up to 45 % of the dock area's height (at least
300 pt, but never more than the area) and never down into the open console. With the debug
overlay open beside it (small screens, e.g. 640 × 360 pt at 2x) it ends 6 pt right of the
overlay, down to 240 pt wide (where names would get narrower than 12 digits, under about 370 pt,
its table keeps the avg and calls columns only); when not even that fits (a tall, narrow screen
such as a phone held upright) it docks under the overlay if 160 pt fit there. Its table scrolls
by whole rows (no row is ever cut), under a header whose titles sit right above their
right-aligned columns whether or not the scroll bar takes a column of its own. In a window too
short for its two sections and 3 table rows (all its rows, with fewer systems; under about
190 pt) it leaves out the *Timeline*
and *Systems* section headers — the flame view and the table stay, always open — so 3 rows still
fit in about 150 pt. The console is 640 × 40 % of the dock area (140–360 pt), a whole number of
log lines tall (stuck to the newest line, the oldest shown is never a sliver; at least 3 lines,
about 125 pt), and wraps long lines (paths, backtraces) under their message instead of widening.
It lays out only the lines in view (the others keep the height measured when they arrived, or
when the log's width last changed), so its cost per frame does not grow with the 1000 lines it
holds.

#### Short dock areas

A game whose HUD leaves only a band of the screen free sets a dock area too short to stack the
tools (SpaceCorps2027: 1024 × 233 pt of a 1024 × 640 screen, where the overlay, the profiler's
300 pt and the console's 40 % used to overlap). The console then docks beside the tools instead
of under them, so all three stay inside the area and apart:

* **Right of the overlay** when the overlay reaches down into it: the overlay's height, 6 pt and
  the console's height as drawn (a whole number of log lines: about 125 pt of its default
  140 pt) are more than the area's height. It stays docked at the bottom.
* **Left of the profiler** when the profiler's smallest layout with 3 table rows (about 150 pt,
  measured as drawn; before its first frames only its message, about 76 pt) does not fit above
  the console as drawn; the profiler then takes the area's height. The profiler gives up width
  (440 down to 240 pt) before the console gets narrower than 320 pt; a profiler the user
  narrowed leaves the console the room.
* With neither tool above it the console is **as tall as the area** (up to 360 pt) instead of
  40 % of it. In the band above: overlay | console (347 pt wide) | profiler (440 pt), each as tall
  as the area's 221 pt allow.
* When the area gets tall again the tools **dock back** to their corners, at the sizes a fresh
  start at that area gives them. A console or profiler the user resized docks by the same rules
  at its own size, squeezed only as far as the area and the other tools need, and gets that size
  back when the area is tall again (the console widened to 890 pt is 347 pt in the band above,
  890 pt again after it); a dragged one floats and is left out.
* **Resizing one toward the other stops a gap before it**, on every frame of the drag: the
  console's right edge at the profiler (as wide as the user left it), the profiler's left edge at
  the console. In a wide, short area where the profiler is clear of the console (1920 × 260 pt)
  both keep their places, the console its height.
* **Overlap only where nothing else fits.** Side by side needs an area about 640 pt wide
  (overlay 213, console 160, profiler 240 and the 6 pt gaps). In a narrower short area the
  console still docks right of the overlay, under the profiler, which shows what rows fit (at
  least one, about 115 pt; without frames yet its message, which wraps there, about 90 pt); when
  not even that fits the two overlap, the one opened or clicked last in front (the console, when
  both open together, covers the profiler's last row). With the overlay alone the console needs
  160 pt right of it or keeps its corner under the overlay. The overlay (about 145 pt tall with
  render stats, 200 with its details) and the smallest profiler and console (3 log lines, about
  125 pt) do not shrink further, so each reaches past the edges of an area shorter than it and
  its gaps — the overlay under about 157 pt (212 pt with its details), the console under about
  137 pt: the overlay and the profiler past the bottom edge (the profiler is cut off there), the
  console, docked at the bottom, past the top one — its title row is cut off (its key, backquote
  by default, still closes it), its newest lines and command line stay in the area.
* Normal and tall areas, and short ones where the stacked layout fits as drawn (a 1100 × 300 pt
  band), keep the layout above: the console moves only when the stacked layout would overlap.
  The first tool system to run in a frame computes the plan and the others take it
  (`space3d-ui/src/dock.rs`); the edges the console docks against beside the others (the area's
  own left edge too) snap to the pixel grid, so the 6 pt gaps stay whole on a dock area off the
  grid.

* **Debug overlay** (`DebugOverlay`): a small panel at the top-left that never takes the pointer,
  opaque (the renderer blends in linear space, where a translucent black lets a bright scene or
  window through far more than its alpha suggests: at 98 % the text of a window under it still
  showed between the values; its labels keep 8:1) —
  FPS, average and worst frame time, a sparkline of the last 240 frames (60 / 30 Hz guides), and a
  monospace grid of label/value pairs with right-aligned numbers: frame, tick, rate, dropped
  ticks, entities, alpha, draws, triangles, drawn instances, uploaded bytes, egui primitives and
  vertices; the last line names the other tools' keys. `DebugOverlay::details` (console:
  `debug details`) adds the draw path, indirect / multi draws, batches, views, culled instances
  and instance bytes. Each grid row is one text the values are written into (no string per label
  or value).
* **Profiler** (`Profiler`): header with the last frame's time, span count and a close button;
  Freeze / Timing toggles; collapsible, full-width *Timeline* (flame view; names truncated with
  "…", full name and time on hover) and *Systems* (striped table sorted by average time: a bar of
  the average, the name — truncated, full name on hover — and right-aligned avg / last / max /
  calls columns under a fixed header) sections. Names are shortened to their last two path
  segments, generic arguments to their last (`events::Events<AppExit>`).
  `Profiler::record(FrameTimeline)` aggregates frames from other sources (recordings, replays,
  tests).
* **Console** (`Console`): level / filter / clear toolbar (the filter fills the row), one dense
  monospace line per entry — the level tag in a column of its own, then target and message; a
  wrapped entry continues under its message — and the command line. `Console::capture_logs = false` keeps only the console's own lines (command
  output and `Console::print`).

## Engine tools

`run_editor` / `editor_app` and `run_storybook` / `storybook_app` insert `UiTheme::compact()`
themselves. Their panels use property rows (a label column in the normal text color and the value
on one line, shared by every section so values align), collapsible component sections with header
actions (remove, edit JSON), and panel margins from the style. Rows never ask for more than the
panel's width, under any style or density: sliders get a fixed-width value box (the same in every
row, so the boxes line up) and the rail takes the rest; x / y / z and paired fields split the row
into equal cells (axis color bars instead of "x " prefixes) and drop decimals before they would
widen it; groups of checkboxes or buttons wrap within the value column; actions sit in rows of
equal cells as wide as the property rows (the texture editor's "Apply …", flip and rotate rows),
so every slider's value box stays in one column. Rows reuse the styles they derive (slider
widths, header spacing, scroll bars) frame after frame instead of copying the `Style` per row.
The side panels keep
their width frame after frame and whatever is selected (hierarchy 160–400 pt, inspector 220–480
pt), their scroll bars stay visible while content is hidden, and buttons (the section "×"
included) are at least 18 pt tall. A slider's rail stays at least two control heights long: in a
narrow column with big fonts the value box gives up room first (down to three digits), and the
texture and model editors' tool columns grow with the body text size. The menu bar is as tall as
egui's own top panel (one control plus 2 pt above and below), and the texture and model editors
open over the viewport, 6 pt clear of the menu bar and the side panels (`TextureEditor::show_in`,
`ModelEditor::show_in` with the viewport rect; `show` keeps egui's placement). Both sit on egui's
foreground layer with F3 / F4 / the console, so the one opened or clicked last is in front (see
[Docking](#docking)). The editor and the storybook viewer set the `ToolDockArea` to their
viewport, so F3 / F4 / the console dock there too. `EditorPlugin` added to a game does **not**
insert the theme: the game's style stays as it is (insert the theme yourself to get the same
density), and its panels still keep their width.

## Fonts for Chinese, Japanese and Korean

egui's built-in fonts have no Han characters, kana or Hangul: add a font that has them, e.g.
right after the Latin font so that it also draws the full-width punctuation (，。！（）), which
egui's emoji fonts would otherwise draw:

```rust
use space3d::ui::egui::{self, FontData, FontDefinitions, FontFamily};
use std::sync::Arc;

let mut fonts = FontDefinitions::default();
fonts.font_data.insert("cjk".into(), Arc::new(FontData::from_static(CJK_FONT_BYTES)));
for family in [FontFamily::Proportional, FontFamily::Monospace] {
    let faces = fonts.families.entry(family).or_default();
    faces.insert(faces.len().min(1), "cjk".into());
}
ctx.set_fonts(fonts); // `EguiContext::ctx()`, or in a UI system
```

## Chinese and Japanese line breaking (kinsoku)

Space3d's egui breaks Chinese and Japanese text by the kinsoku (禁則処理) rules of JIS X 4051
and the W3C JLREQ/CLREQ: no line starts with a closing bracket or quote, a comma or full stop
(、。，．), a colon, ?, !, a middle dot, a small kana (ッ ャ ェ), the prolonged sound mark ー, an
iteration mark (々), a postfixed sign (％ ℃), an ellipsis or a Bopomofo tone mark (ㄏㄠˇ); no
line ends with an opening bracket or quote (「（【“) or ￥, also not across spaces (「　重要　」);
"——" and "……" stay whole; Latin words and numbers inside CJK text stay whole and get breaks
around them. A run the rules keep together that is wider than the row breaks where the row is
full, never inside "——" "……" that fit a row, and the rules never make a row wider than the
wrap width, also in a label after another widget in `horizontal_wrapped` (egui indents its
first row; only that row can be left empty, when the label's first run doesn't fit after the
widget). Upstream egui starts lines with "，" "？" "）" "、" and ends them with "「". Text
without Chinese or Japanese characters lays out as with upstream egui (German „…“ quotes
included), except for three rules of Unicode line breaking (UAX #14) that hold in every script and
ellipses and doubled dashes: no line ends with an ASCII opening bracket `(` `[` `{` (also not
with the spaces after it), none starts with a no-break space U+00A0 unless a space comes
before it, and none breaks next to a word joiner U+2060 (or U+FEFF) unless a space is on the other
side — a chat line's company tag joined to the name with a no-break space
("[T]\u{a0}Name:") moves to the next row whole and a name wider than a row breaks inside the
name, where upstream left "[T]" alone ("[T]" | "\u{a0}Name:") or split it ("[" | "T] Name:"), and
a clan tag whose marks are glued to what follows with word joiners ("[A-\u{2060}B]") is not split
("[A-" | "B]");
no line starts with "…" after a word and "——" "……" are not split, so at narrow widths such
text can wrap differently ("ove" | "r……" where upstream has "over" | "……"). Laying text out takes about 5 % longer for
Latin text and 8–11 % for Chinese or Japanese than with upstream egui; egui caches laid-out
text, so this is paid only when a text or its width changes. The rules and their limits are in
`third_party/README.md`; games that lay text out themselves can use them directly:
`egui::epaint::text::kinsoku::{line_break, Break, is_line_start_prohibited,
is_line_end_prohibited}` (`kinsoku::VERSION` is 5 with the word-joiner rule and 4 with the
other two UAX #14 rules; a game whose chat joins a tag to a name with a no-break space can assert
`VERSION >= 4`, one that glues a clan tag with word joiners `VERSION >= 5`).

The rules live in a patched copy of egui's text layout crate, `epaint`, in
`third_party/epaint`. Cargo applies a `[patch]` only from the **root** `Cargo.toml` of the
workspace being built, so Space3d's own patch covers Space3d's apps, examples and tests, and a
game needs the same entry in its root `Cargo.toml`, pointing at the **same Space3d revision** as
its `space3d` dependency:

```toml
[dependencies]
space3d = { git = "https://github.com/SpaceCorps/Space3d", rev = "<rev>" }

# Chinese and Japanese line breaking (kinsoku) for egui: Space3d's patched epaint.
[patch.crates-io]
epaint = { git = "https://github.com/SpaceCorps/Space3d", rev = "<rev>" }  # same rev as space3d
```

Cargo finds the `epaint` package inside the Space3d repository by its name
(`third_party/epaint`); with the same `rev` both come from one checkout. With a local engine
checkout, use `epaint = { path = "../Space3d/third_party/epaint" }` next to
`space3d = { path = "../Space3d/crates/space3d" }`. `space3d new` writes the entry for you.

When the entry is missing, or points at a Space3d revision whose egui differs from the one
`space3d` uses, cargo only warns (`Patch 'epaint v… (…)' was not used in the crate graph`) and
builds crates.io's epaint, without kinsoku. Check with:

* `space3d doctor` — the `ui` line warns and prints the entry to add when epaint comes from
  crates.io, from another Space3d revision than `space3d`, or from a path without Space3d's
  kinsoku module (a game's own older copy), and says when `space3d`'s revision predates
  `third_party/epaint` (update `space3d` first);
* `cargo tree -i epaint` — shows `epaint v0.36.2 (https://github.com/SpaceCorps/Space3d?rev=…)`
  (or the local path), not a bare version;
* a test in the game that only compiles with the patched epaint:
  `const _: u32 = space3d::ui::egui::epaint::text::kinsoku::VERSION;` (with the UAX #14 rules,
  `const _: () = assert!(space3d::ui::egui::epaint::text::kinsoku::VERSION >= 5);`).

When updating Space3d, bump the `rev` of both lines together. A game that carried its own
`third_party/epaint` (as SpaceCorps did) replaces its `[patch.crates-io]` path with the git
entry above and deletes the copy; its kinsoku tests keep passing.

Space3d itself: `third_party/README.md` explains the patch, its tests and the egui upgrade
procedure (`scripts/update-epaint.py`); `crates/space3d-ui/tests/kinsoku.rs` lays out Japanese
and Chinese paragraphs at many widths and holds the golden story `UI/Text/Kinsoku`;
`crates/space3d-ui/tests/line_breaks.rs` wraps chat lines with company tags and brackets at
every width from 140 to 600 pt.

## Stories and tests

The engine storybook (`space3d_stories::engine_storybook`, feature `ui`: not a default feature, since
it pulls in egui; the editor and the facade's `stories` feature turn it on) holds
`Engine/UI/Theme/Default`, `Engine/UI/Theme/Compact`, `Engine/UI/Tools/Debug Overlay`,
`Engine/UI/Tools/Profiler`, `Engine/UI/Tools/Console` and `Engine/UI/Tools/Game Style`, with
goldens in `crates/space3d-stories/tests/__screenshots__/Engine/UI/`. The tool stories have a
`style` knob (egui default, compact theme, roomy game style).

Tests: `crates/space3d-ui/tests/theme.rs` (density, default path untouched, applied once, never
fights a game's style; unit tests: `UiDensity::of` forgets a removed or overridden theme),
`crates/space3d-ui/tests/tools.rs` (docked tools stay inside the screen and apart at several sizes
and scale factors, never grow, same geometry under a game's style; the profiler keeps clear of the
debug overlay and above the console on 640 × 330–400 pt screens and shares 480 × 320 and
390 × 844 pt screens without overlaps; docked tools follow the screen as it grows and shrinks,
or the dock area, at the size a fresh start gives them (also when a host sets the area after the
first frame, as the editor does), while a tool the user resized keeps its size and stays docked;
a dragged tool floats until its title is double-clicked; tools dock inside a `ToolDockArea`; a
game window never covers them; in short dock areas — the reported 1024 × 233 band and 800 × 200,
1280 × 180, 1920 × 260, 640 × 240, 1100 × 280 pt — they dock side by side, inside and apart, the
full 6 pt apart on an area off the pixel grid, and dock back when the area gets tall again, while
short areas where the stacked layout fits as drawn — 1100 × 300 pt at 1–2x, 640 × 330 pt with
nested profiler spans — keep it; a profiler without frames stays above the console; a tool made
smaller by a free edge keeps that size, and a narrowed profiler leaves the console its room;
resized tools get their sizes back after a short area, at 1–2x; an edge dragged toward the
other tool stops a gap from it on every frame, without losing a log line, also in a wide band),
the dock plan's unit tests (`dock.rs`: roomy areas stay stacked, the plan goes by the console's
drawn height, the band goes side by side, the console moves right of a tall overlay, gaps stay
whole on the pixel grid with or without the overlay, the console's limit holds as it reaches the
profiler, a resized profiler and the console stay apart, a tiny area keeps the corners), the
console's
`drawn_height_is_the_height_the_console_is_drawn_at` (at 1–2x; the estimate before its first
frame is exact at 1x), the profiler's
`table_headers_line_up_with_their_columns` (header titles over their columns with and without a
scroll bar, and in a narrow table), `a_short_profiler_leaves_out_its_section_headers` (at
least 3 rows from about 150 pt; the measured minimum is exact) and
`a_profiler_with_few_systems_keeps_its_sections_where_they_fit`,
`crates/space3d-ui/tests/alloc_free.rs` (no allocation while
the theme is unchanged or absent; the console allocates as much per frame with 1000 lines as with
40) and the editor's `standalone_tools_are_compact_and_the_plugin_keeps_a_games_style`,
`editor_panels_keep_their_width_under_any_style`, `tool_windows_stay_over_the_viewport`,
`editor_windows_and_engine_tools_come_to_the_front_when_clicked`,
`engine_tools_dock_inside_the_viewport` (at 1280 × 720 and 1920 × 1080, after resizes, and the
whole screen with the editor hidden) and `rows_reuse_their_styles_frame_after_frame`.
