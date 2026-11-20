//! Three rules of Unicode line breaking (UAX #14) that egui lacks and Space3d's patched epaint
//! (`third_party/epaint`, `text/kinsoku.rs`; `third_party/README.md`, `docs/ui.md`) applies to
//! text in every script: no break just before a no-break space U+00A0 (class GL), no break
//! after an ASCII opening bracket `(` `[` `{` (class OP), and no break on either side of a word
//! joiner U+2060 (class WJ).
//!
//! The scenario is a chat line, which is where they showed: "12:04  [T]\u{a0}Name: text", the
//! time, the sender's company tag joined to the name with a no-break space, the name, the
//! message, wrapped at every width a chat window can have. Upstream egui left the tag alone
//! after the time ("[T]" | " Name:"), and for a name longer than a row split it: "[G]" |
//! " Wwww…", then "[" | "G] Wwww…"; a clan tag whose marks are glued to what follows with word
//! joiners wrapped as "[A-" | "B]". The unit tests in the vendored epaint
//! (`text::kinsoku::tests`) hold the rules character by character; this file lays out whole
//! lines through egui.
//!
//! Like `tests/kinsoku.rs`, the checks use this file's own lists, so they fail whenever the
//! patch is lost (upstream egui fails them at most widths).

use space3d_ui::egui::text::LayoutJob;
use space3d_ui::egui::{self, Color32, FontId, TextFormat};

/// What joins a company tag to the name: a no-break space, as the SpaceCorps chat does.
const TAG_GAP: char = '\u{a0}';

/// Chat widths, in points: the narrowest window up to a wide one.
const WIDTHS: std::ops::RangeInclusive<usize> = 140..=600;

/// A chat line's parts.
struct Line<'a> {
    time: &'a str,
    /// The company tag without its brackets.
    tag: &'a str,
    name: &'a str,
    text: &'a str,
}

impl Line<'_> {
    /// `12:04  [T]\u{a0}name: text` as sections like the game's chat: the time in a small
    /// monospace font, the tag in its colour, the name, the message after a gap.
    fn job(&self, gap: char, width: f32) -> LayoutJob {
        let font = FontId::proportional(12.5);
        let mut job = LayoutJob::default();
        job.append(
            &format!("{}  ", self.time),
            0.0,
            TextFormat::simple(FontId::monospace(10.5), Color32::GRAY),
        );
        job.append(
            &format!("[{}]{gap}", self.tag),
            0.0,
            TextFormat::simple(font.clone(), Color32::GREEN),
        );
        job.append(
            &format!("{}:", self.name),
            0.0,
            TextFormat::simple(font.clone(), Color32::LIGHT_BLUE),
        );
        job.append(self.text, 6.0, TextFormat::simple(font, Color32::WHITE));
        job.wrap.max_width = width;
        job
    }
}

fn context() -> egui::Context {
    let ctx = egui::Context::default();
    // Fonts are installed at the start of a pass; nothing uploads its font atlas here.
    ctx.run_ui(egui::RawInput::default(), |_| {}).textures_delta.clear();
    ctx
}

fn rows(ctx: &egui::Context, job: LayoutJob) -> Vec<String> {
    let text = job.text.clone();
    let rows: Vec<String> = ctx
        .fonts_mut(|fonts| fonts.layout_job(job))
        .rows
        .iter()
        .map(|row| row.text())
        .collect();
    assert_eq!(rows.concat(), text, "characters lost");
    rows
}

/// Rows of the line at every chat width.
fn rows_at_every_width(ctx: &egui::Context, line: &Line, gap: char) -> Vec<(usize, Vec<String>)> {
    WIDTHS
        .map(|width| (width, rows(ctx, line.job(gap, width as f32))))
        .collect()
}

/// A row (not the last) that ends with an opening bracket, before its trailing spaces.
fn ends_with_opening_bracket(row: &str) -> bool {
    row.trim_end().ends_with(['(', '[', '{'])
}

/// The rows show `tag` with something of the name after it on the same row.
fn tag_row_holds_the_name(rows: &[String], tag: &str) -> bool {
    rows.iter()
        .find_map(|row| row.split_once(tag))
        .is_some_and(|(_, rest)| !rest.trim().is_empty())
}

/// A long sender wraps with its company tag: at every chat width the tag's row holds the start
/// of the name, and no row starts with the gap between them. The longest name a pilot can have
/// (32 letters, wider than a row) breaks inside the name, and so does a long one with
/// underscores. Upstream egui: "[T]" alone after the time at most widths.
#[test]
fn a_wrapped_sender_keeps_its_company_tag() {
    let ctx = context();
    let longest = "W".repeat(32);
    let senders = [
        ("T", "Commander_Starfighter_Alpha27"),
        ("G", longest.as_str()),
        ("M", "nova"),
        ("M", "Rory (the second)"),
    ];
    let mut bad = Vec::new();
    for (tag, name) in senders {
        let line = Line {
            time: "19:42",
            tag,
            name,
            text: "anyone for the gate run?",
        };
        let tag = format!("[{tag}]");
        for (width, rows) in rows_at_every_width(&ctx, &line, TAG_GAP) {
            if !tag_row_holds_the_name(&rows, &tag) || rows.iter().any(|r| r.starts_with(TAG_GAP)) {
                bad.push((width, rows));
            }
        }
    }
    assert!(bad.is_empty(), "{} widths: {:#?}", bad.len(), &bad[..bad.len().min(4)]);
}

/// "[T]\u{a0}nova:" moves to the next row whole: at every width the row with the tag has the
/// name and its colon, and the time may be left alone before it (as before the tag existed).
#[test]
fn a_tag_and_its_name_move_together() {
    let ctx = context();
    for (tag, name) in [("T", "nova"), ("G", "Vesper"), ("M", "rorychatt")] {
        let line = Line {
            time: "12:04",
            tag,
            name,
            text: "free credits at the station, one jump from the gate",
        };
        let whole = format!("[{tag}]{TAG_GAP}{name}:");
        for (width, rows) in rows_at_every_width(&ctx, &line, TAG_GAP) {
            assert!(rows.iter().any(|row| row.contains(&whole)), "{width}: {rows:?}");
        }
    }
}

/// A chat message can have brackets of every kind, spaces inside them, nested and adjacent
/// ones, in words and in tokens without a space (a link) that are wider than a row. At every
/// chat width no row ends with an opening bracket (before its trailing spaces): there is
/// always a legal break before it, since every unit the rules keep together is far narrower
/// than a row. Upstream broke after the bracket in a token wider than a row (where egui
/// breaks at a mark) at most widths.
#[test]
fn a_bracket_never_ends_a_row() {
    let ctx = context();
    let messages = [
        "the gate (sector 4-2) [north] {see map} is open ( now ) [ ping me ] (( [[ {{ ))",
        "meet at ([{ the station }]) or at [(a)] later (maybe) - f(x)+g[y]*h{z}=k(l(m[n{o}]))",
        "(a)(b)[c][d]{e}{f} )( ][ }{ (a) [b] {c} and (last",
        "( ( ( ( ( x ) ) ) ) ) [ [ [ y ] ] ] { { z } } and more words to wrap around a bit",
        // Tokens without a space, wider than a row at most widths (a link, a formula, a run of
        // brackets), where the row breaks at a mark or anywhere: upstream after the bracket.
        "see https://spacecorps.example/wiki/route_(gate)/[north]?sector={4-2}&ship=(nova)#part[2] and more",
        "f(x)+g[y]*h{z}=k(l(m[n{o}]))-p(q)+(((((((((((((((((r)))))))))))))))))=[[[[[[[[[[[[[s]]]]]]]]]]]]]",
    ];
    let mut bad = Vec::new();
    let mut layouts = 0;
    for text in messages {
        for (tag, gap) in [("T", TAG_GAP), ("G", ' ')] {
            let line = Line {
                time: "12:04",
                tag,
                name: "nova",
                text,
            };
            for (width, rows) in rows_at_every_width(&ctx, &line, gap) {
                layouts += 1;
                if rows[..rows.len() - 1].iter().any(|row| ends_with_opening_bracket(row)) {
                    bad.push((width, rows));
                }
            }
        }
    }
    assert!(layouts > 3000, "{layouts} layouts");
    assert!(bad.is_empty(), "{} widths: {:#?}", bad.len(), &bad[..bad.len().min(4)]);
}

/// With a plain space between tag and name (no rule against it: a space breaks as egui
/// decides) the tag may still end a row, but is never split: no row ends with its opening
/// bracket, and no row starts with the space's blank (a no-break space stays with its tag).
#[test]
fn a_tag_with_a_plain_space_is_never_split() {
    let ctx = context();
    let line = Line {
        time: "19:42",
        tag: "T",
        name: "Commander_Starfighter_Alpha27",
        text: "anyone for the gate run?",
    };
    let mut alone = 0;
    for (width, rows) in rows_at_every_width(&ctx, &line, ' ') {
        assert!(
            rows[..rows.len() - 1].iter().all(|row| !ends_with_opening_bracket(row)),
            "{width}: {rows:?}"
        );
        assert!(rows.iter().all(|row| !row.starts_with(TAG_GAP)), "{width}: {rows:?}");
        alone += rows.iter().any(|row| row.trim_end().ends_with("[T]")) as usize;
    }
    // The space is a legal break: the rule is the no-break space's, not a ban on tags at row ends.
    assert!(alone > 0, "a plain space never let the tag end a row");
}

/// A clan tag the way the SpaceCorps chat lays it out, "[VNGD]" with a word joiner (U+2060,
/// invisible, no break on either side of it) after each ASCII mark, so no wrapped line splits
/// the tag: "19:42  [T]\u{a0}[A-\u{2060}B]\u{a0}name: text" at every chat width that holds the
/// tag. Upstream egui and the rules above without the joiner's: "[A-" | "B]" at most widths.
#[test]
fn a_clan_tag_glued_with_word_joiners_is_never_split() {
    const JOINER: char = '\u{2060}';
    let ctx = context();
    let longest = "W".repeat(32);
    let mut layouts = 0;
    let mut bad = Vec::new();
    for clan in ["A-B", "A.B", "-_-", "!?", "A/B", "A:B", "A-B-C"] {
        let mut tag = String::from("[");
        for c in clan.chars() {
            tag.push(c);
            if c.is_ascii_punctuation() {
                tag.push(JOINER);
            }
        }
        tag.push(']');
        for name in ["nova", "Commander_Starfighter_Alpha27", longest.as_str()] {
            for width in WIDTHS {
                let font = FontId::proportional(12.5);
                let mut job = LayoutJob::default();
                job.append(
                    "19:42  ",
                    0.0,
                    TextFormat::simple(FontId::monospace(10.5), Color32::GRAY),
                );
                for (text, color) in [
                    (format!("[T]{TAG_GAP}"), Color32::GREEN),
                    (format!("{tag}{TAG_GAP}"), Color32::GOLD),
                    (format!("{name}:"), Color32::LIGHT_BLUE),
                ] {
                    job.append(&text, 0.0, TextFormat::simple(font.clone(), color));
                }
                job.append(
                    "anyone for the gate run?",
                    6.0,
                    TextFormat::simple(font, Color32::WHITE),
                );
                job.wrap.max_width = width as f32;
                let rows = rows(&ctx, job);
                layouts += 1;
                // Character offsets of the tag in the line, and of every row break.
                let line = rows.concat();
                let start = line[..line.find(&tag).expect("the tag is in the line")].chars().count();
                let end = start + tag.chars().count();
                let mut offset = 0;
                let split = rows[..rows.len() - 1].iter().any(|row| {
                    offset += row.chars().count();
                    start < offset && offset < end
                });
                if split {
                    bad.push((width, tag.clone(), rows));
                }
            }
        }
    }
    assert!(layouts > 3000, "{layouts} layouts");
    assert!(bad.is_empty(), "{} widths: {:#?}", bad.len(), &bad[..bad.len().min(4)]);
}
