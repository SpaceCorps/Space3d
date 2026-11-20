//! Kinsoku shori (禁則処理): where a line of Chinese or Japanese text may break.
//!
//! **Space3d patch**, not part of upstream epaint: see `third_party/README.md` in the Space3d
//! repository. Derived from the kinsoku patch the `SpaceCorps` game carried in its own copy
//! of epaint (game commits b8af874 and e531f10).
//!
//! egui breaks a row at a space, after a Han character or kana, before one, and as a fallback
//! at a dash, after ASCII punctuation or anywhere. Its only CJK rule was a short list of marks
//! that may not follow a Han character or kana at the start of a line, so a Chinese or
//! Japanese line could start with "，" "？" "）" "…" or, through the fallbacks, with "、" after
//! a Latin word or a small kana ("ロ" | "ッ"), and end with "（" or "「".
//!
//! [`line_break`] rules on every position between two characters before egui records it as a
//! break candidate:
//!
//! * [`Break::Prohibited`]: no break here. A run of characters kinsoku keeps together moves to
//!   the next row whole when it fits there; one wider than a row (a run of marks, a word and
//!   its "、" in a very narrow column) still breaks where the row is full, as upstream egui
//!   breaks a word wider than the row — but not inside "——" or "……" when the pair fits a row,
//!   nor inside a character and its combining mark or variation selector.
//! * [`Break::Allowed`]: a word boundary, also where egui had none: after CJK punctuation
//!   before a Latin word or a number ("、" | "Thulium", "。" | "4"), before an opening bracket
//!   that follows a Latin word or a mark ("Skylab" | "「", "！" | "（").
//! * [`Break::Upstream`]: egui's own rules decide. This is the answer for every pair of
//!   characters without a Chinese or Japanese character in it (apart from the ellipses and
//!   dashes below and the three Unicode rules after this list), so Latin, Cyrillic, Greek,
//!   Korean… text lays out as upstream, and “ ‘ « — which open a quote in Chinese and English
//!   but close one in German („Tor“) — only count as opening brackets next to Chinese or
//!   Japanese characters.
//!
//! Three rules of Unicode line breaking (UAX #14) hold in every script, because egui lacks them
//! and they show in ordinary Latin text (a chat line's "\[T\] name:" wrapped as "\[T\]" | " name"
//! and "[" | "T] name", a clan tag glued with word joiners as "\[A-" | "B\]"): no break just
//! before a no-break space U+00A0 (class GL, LB12a) unless a space comes before it, no break
//! after an ASCII opening bracket `(` `[` `{` (class OP, LB14), also across the spaces after it
//! ("[ x"), and no break next to a word joiner U+2060 or U+FEFF (class WJ, LB11) unless a space
//! is on the other side. See [`line_break`].
//!
//! Row breaking uses [`line_break`] through a paragraph-level scan (crate-private), which also
//! looks past spaces: a space between a bracket and its text ("「　重要　」") doesn't let the
//! bracket end or start a line.
//!
//! The character classes follow JIS X 4051 (Japanese) and the W3C requirements for Japanese
//! and Chinese layout (JLREQ, CLREQ), strict level: no line starts with a closing bracket or
//! quote, a comma, full stop, colon, semicolon, ?, !, middle dot, hyphen or wave dash, small
//! kana, the prolonged sound mark, an iteration mark, a postfixed sign (％ ℃) or an ellipsis;
//! no line ends with an opening bracket or quote or a prefixed sign (￥ ＄); "——" "……" "‥‥"
//! stay together; no line breaks on either side of a straight quote `"` `'` (it may open or
//! close). Hanging punctuation (ぶら下げ) and justification are out of scope.

use crate::text::Glyph;

/// Revision of these rules, bumped whenever they change. Its presence tells a program or a
/// test that epaint is Space3d's patched copy (`egui::epaint::text::kinsoku::VERSION`).
pub const VERSION: u32 = 5;

/// Kinsoku's ruling on a line break between two characters, see [`line_break`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Break {
    /// No break here, unless the row has no other place to break at all.
    Prohibited,

    /// A good place to break (a word boundary), even where egui's own rules have none.
    Allowed,

    /// Kinsoku has no say: egui's own rules decide (spaces, Latin text, …).
    Upstream,
}

/// May a line break between `before` and `after`?
///
/// Text without Chinese or Japanese characters gets [`Break::Upstream`], so it lays out as
/// with upstream egui, except that in every script
///
/// * no line starts with an ellipsis after a non-space, and "——" "……" stay together;
/// * no line ends with an ASCII opening bracket `(` `[` `{` (UAX #14 class OP, rule LB14;
///   upstream broke after it as after any ASCII mark, "[" | "T] name");
/// * no line starts with a no-break space U+00A0 (class GL, LB12a) unless a space comes
///   before it (upstream broke before it after a mark or a letter, "\[T\]" | "\u{a0}name"). Only
///   the break before it is ruled on: after it egui decides as before;
/// * no line breaks next to a word joiner U+2060 or U+FEFF (class WJ, LB11) unless a space is
///   on the other side (upstream broke after a dash or a mark, a clan tag glued with joiners as
///   "\[A-" | "B\]").
///
/// A space decides for itself — a line breaks after it as egui does — except that no line ends
/// with an opening bracket before it and next to Chinese or Japanese no line starts with a
/// closing mark after it or ends with an opening one before it ("字　」", "（\u{3000}").
pub fn line_break(before: char, after: char) -> Break {
    if is_inseparable(before, after) || is_unicode_glue(before, after) {
        return Break::Prohibited;
    }
    if before.is_whitespace() || after.is_whitespace() {
        // A space breaks as egui decides, but next to Chinese or Japanese it takes no closing
        // mark to the start of a line and leaves no opening one at the end ("字　」", "「 字").
        let cjk = is_cjk(before) || is_cjk(after);
        return if cjk && (is_line_start_prohibited(after) || is_line_end_prohibited(before)) {
            Break::Prohibited
        } else {
            Break::Upstream
        };
    }
    if !is_cjk(before) && !is_cjk(after) {
        // Unicode line breaking (UAX #14, LB22) keeps an ellipsis with what precedes it in
        // every script; Chinese text puts "……" right after Latin words ("Thulium……").
        return if is_ellipsis(after) {
            Break::Prohibited
        } else {
            Break::Upstream
        };
    }
    if is_line_end_prohibited(before)
        || is_line_start_prohibited(after)
        || is_straight_quote(before)
        || is_glue(before)
        || is_glue(after)
        || is_mark(after)
    {
        Break::Prohibited
    } else {
        Break::Allowed
    }
}

/// Characters of Chinese and Japanese text, which switch the kinsoku rules on next to them.
///
/// Han characters, kana, Bopomofo, CJK symbols and punctuation, full-width forms and
/// half-width katakana. Not Hangul: Korean breaks at spaces, as upstream egui does.
pub fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{2E80}'..='\u{2FDF}' // CJK Radicals Supplement, Kangxi Radicals
        | '\u{2FF0}'..='\u{2FFF}' // Ideographic Description Characters
        | '\u{3000}'..='\u{303F}' // CJK Symbols and Punctuation (、。「」〜々)
        | '\u{3040}'..='\u{30FF}' // Hiragana, Katakana
        | '\u{3100}'..='\u{312F}' // Bopomofo
        | '\u{3190}'..='\u{31FF}' // Kanbun, Bopomofo Extended, CJK Strokes, Katakana Phonetic Extensions
        | '\u{3200}'..='\u{33FF}' // Enclosed CJK Letters and Months, CJK Compatibility
        | '\u{3400}'..='\u{4DBF}' // CJK Unified Ideographs Extension A
        | '\u{4E00}'..='\u{9FFF}' // CJK Unified Ideographs
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{FE10}'..='\u{FE1F}' // Vertical Forms
        | '\u{FE30}'..='\u{FE4F}' // CJK Compatibility Forms
        | '\u{FE50}'..='\u{FE6F}' // Small Form Variants (﹐﹑)
        | '\u{FF00}'..='\u{FF9F}' // Full-width ASCII (，！（）：；？％), half-width katakana
        | '\u{FFE0}'..='\u{FFEF}' // Full-width signs (￥￠), half-width symbols
        | '\u{1AFF0}'..='\u{1B16F}' // Kana Extended-B, Kana Supplement, Kana Extended-A, Small Kana Extension
        | '\u{20000}'..='\u{3FFFF}' // Supplementary and Tertiary Ideographic Planes
    )
}

/// Characters that may not start a line of Chinese or Japanese text (行頭禁則).
///
/// A superset of the list egui checks after a Han character or kana (the ASCII marks among
/// them only count next to Chinese or Japanese characters, see [`line_break`]).
pub fn is_line_start_prohibited(c: char) -> bool {
    matches!(c,
        // Closing brackets and quotes (JIS X 4051 cl-02).
        ')' | ']' | '}' | '\'' | '"' | '’' | '”' | '»'
        | '）' | '］' | '｝' | '〕' | '〉' | '》' | '」' | '』' | '】' | '〙' | '〗' | '〛'
        | '〞' | '〟' | '｠' | '｣'
        | '︘' | '︶' | '︸' | '︺' | '︼' | '︾' | '﹀' | '﹂' | '﹄' | '﹈' | '﹚' | '﹜' | '﹞'
        // Hyphens and wave dashes (cl-03).
        | '‐' | '–' | '゠' | '〜' | '～'
        // Question and exclamation marks (cl-04).
        | '?' | '!' | '？' | '！' | '‼' | '⁇' | '⁈' | '⁉' | '︕' | '︖' | '﹖' | '﹗'
        // Middle dots, colons and semicolons (cl-05), the Chinese interpunct (CLREQ).
        | ':' | ';' | '：' | '；' | '・' | '･' | '·' | '‧' | '︓' | '︔' | '﹔' | '﹕'
        // Full stops (cl-06).
        | '.' | '。' | '．' | '｡' | '︒' | '﹒'
        // Commas (cl-07).
        | ',' | '、' | '，' | '､' | '︐' | '︑' | '﹐' | '﹑'
        // Iteration marks (cl-09).
        | 'ヽ' | 'ヾ' | 'ゝ' | 'ゞ' | '々' | '〻'
        // Prolonged sound mark (cl-10).
        | 'ー' | 'ｰ'
        // Small kana (cl-11).
        | 'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' | 'っ' | 'ゃ' | 'ゅ' | 'ょ' | 'ゎ' | 'ゕ' | 'ゖ'
        | 'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ' | 'ッ' | 'ャ' | 'ュ' | 'ョ' | 'ヮ' | 'ヵ' | 'ヶ'
        | '\u{31F0}'..='\u{31FF}' // ㇰ-ㇿ (Ainu)
        | '\u{FF67}'..='\u{FF6F}' // ｧ-ｯ (half-width)
        | '\u{1B132}' | '\u{1B150}'..='\u{1B152}' | '\u{1B155}' | '\u{1B164}'..='\u{1B167}'
        // Voiced sound marks, spacing forms.
        | '゛' | '゜' | 'ﾞ' | 'ﾟ'
        // Postfixed signs.
        | '%' | '％' | '‰' | '°' | '′' | '″' | '℃' | '℉' | '¢' | '￠'
        // Bopomofo tone marks, written after their syllable (ㄏㄠˇ), and the light-tone dot.
        | 'ˉ' | 'ˊ' | 'ˇ' | 'ˋ' | '˪' | '˫' | '˙'
    ) || is_ellipsis(c)
}

/// Characters that may not end a line of Chinese or Japanese text (行末禁則).
///
/// Opening brackets and quotes (JIS X 4051 cl-01), the full-width prefixed signs, which stay
/// with the number after them, and the Bopomofo light-tone dot, which Taiwan's standard writes
/// before its syllable (˙ㄇㄚ) and much text after it (so it may neither start nor end a line).
pub fn is_line_end_prohibited(c: char) -> bool {
    matches!(
        c,
        // Opening brackets and quotes (JIS X 4051 cl-01).
        '(' | '[' | '{' | '‘' | '“' | '«'
        | '（' | '［' | '｛' | '〔' | '〈' | '《' | '「' | '『' | '【' | '〘' | '〖' | '〚'
        | '〝' | '｟' | '｢'
        | '︗' | '︵' | '︷' | '︹' | '︻' | '︽' | '︿' | '﹁' | '﹃' | '﹇' | '﹙' | '﹛' | '﹝'
        // Prefixed signs.
        | '￥' | '＄' | '￡' | '＃'
        // The Bopomofo light-tone dot.
        | '˙'
    )
}

/// Pairs that stay together (分離禁止): the two halves of "——" "――" "……" "‥‥" "⋯⋯", and
/// the vertical kana repeat marks 〳〵 〴〵.
pub fn is_inseparable(before: char, after: char) -> bool {
    (before == after && matches!(before, '—' | '―' | '…' | '‥' | '⋯' | '︙'))
        || (matches!(before, '〳' | '〴') && after == '〵')
}

/// The three rules of Unicode line breaking (UAX #14) that [`line_break`] applies in every
/// script: no break after an ASCII opening bracket (LB14), none before a no-break space
/// (LB12a) that doesn't follow a space, and none on either side of a word joiner (LB11). A
/// hyphen or another break-after character before a no-break space doesn't lift the rule, as
/// it does in UAX #14: no row should start with the blank. Next to a space a word joiner
/// leaves the break to egui (the space breaks after itself), as the no-break space does.
fn is_unicode_glue(before: char, after: char) -> bool {
    let space = |c: char| c.is_whitespace() && c != NO_BREAK_SPACE;
    is_ascii_opening_bracket(before)
        || (after == NO_BREAK_SPACE && (!before.is_whitespace() || before == NO_BREAK_SPACE))
        || ((is_word_joiner(before) || is_word_joiner(after)) && !space(before) && !space(after))
}

/// The ASCII opening brackets `(` `[` `{` (Unicode line breaking class OP), which stay with
/// what follows them in every script.
fn is_ascii_opening_bracket(c: char) -> bool {
    matches!(c, '(' | '[' | '{')
}

/// Class GL: glues the characters on both sides (which egui doesn't treat as a space).
const NO_BREAK_SPACE: char = '\u{A0}';

/// Class WJ, the word joiner U+2060 and the zero-width no-break space U+FEFF: no line breaks on
/// either side of them. [`is_glue`] also counts the zero-width joiner, which only matters next
/// to Chinese and Japanese text.
fn is_word_joiner(c: char) -> bool {
    matches!(c, '\u{2060}' | '\u{FEFF}')
}

/// `"` and `'` open and close quotes (Unicode line breaking class QU): next to Chinese or
/// Japanese no line breaks on either side of them.
fn is_straight_quote(c: char) -> bool {
    matches!(c, '"' | '\'')
}

fn is_ellipsis(c: char) -> bool {
    matches!(c, '…' | '‥' | '⋯' | '︙')
}

/// Word joiners (and zero-width joiners), across which no line breaks.
fn is_glue(c: char) -> bool {
    matches!(c, '\u{2060}' | '\u{FEFF}' | '\u{200D}')
}

/// Combining marks and variation selectors continue the character before them (か + ゙,
/// 葛 + an ideographic variation selector).
fn is_mark(c: char) -> bool {
    use unicode_general_category::{GeneralCategory, get_general_category};
    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

// ---- Glyph level (used by `RowBreakCandidates` in `text_layout.rs`) -------------------------

/// Kinsoku along a paragraph: [`line_break`] at each position between two glyphs, plus what
/// takes more than the two characters around a position. Spaces between a bracket and its text
/// keep the bracket on the line of its text ("「　字", "字　」": no line ends with "「　" or
/// starts with "　」"), and a cluster stays whole even when the shaper gave its variation
/// selector the base's character (see [`continues`]).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Scan {
    /// In a run of spaces after an opening bracket of Chinese or Japanese text, or an ASCII one.
    after_opener: bool,

    /// In (or right before) a run of spaces that a closing mark of Chinese or Japanese text
    /// follows.
    before_closer: bool,
}

impl Scan {
    /// Kinsoku's ruling on the position after `glyphs[0]` (`glyphs`: the rest of the
    /// paragraph). Call it for every position of a paragraph, in order.
    pub(crate) fn next(&mut self, glyphs: &[Glyph]) -> Break {
        let glyph = &glyphs[0];
        let Some(next) = glyphs.get(1) else {
            return Break::Upstream;
        };
        if let Some(ruling) = common_pair(glyph, next) {
            // Neither is a space: no bracket to keep across spaces.
            (self.after_opener, self.before_closer) = (false, false);
            return ruling;
        }
        self.rules(glyphs)
    }

    /// [`Scan::next`] without the shortcut for common pairs.
    fn rules(&mut self, glyphs: &[Glyph]) -> Break {
        let (glyph, next) = (&glyphs[0], &glyphs[1]);
        if !glyph.chr.is_whitespace() {
            // Before a run of spaces, look past it once.
            let spaces = next.chr.is_whitespace();
            self.after_opener = spaces
                && (is_ascii_opening_bracket(glyph.chr)
                    || (is_cjk(glyph.chr) && is_line_end_prohibited(glyph.chr)));
            self.before_closer = spaces
                && glyphs[1..]
                    .iter()
                    .find(|g| !g.chr.is_whitespace())
                    .is_some_and(|g| is_cjk(g.chr) && is_line_start_prohibited(g.chr));
        }
        if (glyph.chr.is_whitespace() && self.after_opener)
            || (next.chr.is_whitespace() && self.before_closer)
            || continues(glyph, next)
        {
            Break::Prohibited
        } else {
            line_break(glyph.chr, next.chr)
        }
    }
}

/// The ruling on the most common pairs of characters, found quickly (`None` for the others):
/// two characters before U+2000 that aren't spaces (Latin, Cyrillic, Greek, Arabic… text; the
/// rules have no characters there but the ASCII opening brackets, so egui decides), and a Han
/// character or kana after one, unless it may not start a line or is a combining mark.
/// [`Scan::rules`] rules the same.
fn common_pair(glyph: &Glyph, next: &Glyph) -> Option<Break> {
    let (before, after) = (glyph.chr, next.chr);
    if (before as u32) < 0x2000 && (after as u32) < 0x2000 {
        return (!before.is_whitespace()
            && !after.is_whitespace()
            && !is_ascii_opening_bracket(before))
        .then_some(Break::Upstream);
    }
    let han = |c| matches!(c, '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}');
    let kana = |c| matches!(c, '\u{3040}'..='\u{30FF}');
    let letter_after = han(after)
        || (kana(after)
            && !matches!(after, '\u{3099}' | '\u{309A}')
            && !is_line_start_prohibited(after));
    ((han(before) || kana(before)) && letter_after && next.advance_width > 0.0)
        .then_some(Break::Allowed)
}

/// `next` continues the cluster of `glyph`, a Chinese or Japanese character: it has no width
/// (a combining mark, or a variation selector or joiner, which the shaper gives the base's
/// character when the font lacks it). No row breaks between them, not even as a last resort:
/// that would only start the next row with an empty glyph.
pub(crate) fn continues(glyph: &Glyph, next: &Glyph) -> bool {
    next.advance_width <= 0.0 && is_cjk(glyph.chr)
}

/// A last-resort break between `glyph` and `next` is better than one elsewhere in the row:
/// not inside "——" "……" (unless the pair alone is wider than the row) or a cluster.
pub(crate) fn separable(glyph: &Glyph, next: &Glyph) -> bool {
    !is_inseparable(glyph.chr, next.chr) && !continues(glyph, next)
}

/// Whether `glyphs` (from the first after a break to the one that overflowed the row) fit on
/// a row `width` wide.
pub(crate) fn fits(glyphs: &[Glyph], width: f32) -> bool {
    match (glyphs.first(), glyphs.last()) {
        (Some(first), Some(last)) => last.max_x() - first.pos.x <= width,
        _ => true,
    }
}

/// Whether a row may break after `glyphs[0]` and leave the rest (up to the glyph that
/// overflowed the row) for the next row: they fit there, or there is no Chinese or Japanese
/// in `glyphs` — then kinsoku had no say about the breaks in them, and the rest of a Latin
/// word may overflow a little as with upstream egui.
pub(crate) fn may_carry(glyphs: &[Glyph], width: f32) -> bool {
    fits(&glyphs[1..], width) || !glyphs.iter().any(|glyph| is_cjk(glyph.chr))
}

/// Whether the run kinsoku keeps together after a break after `glyphs[index]` (up to the next
/// position it allows a break at) fits on a row `width` wide. `RowBreakCandidates` moves such
/// a run to the next row only when it fits there whole; a wider one breaks where the row is
/// full, as upstream egui does, instead of overflowing the next row.
pub(crate) fn run_fits(glyphs: &[Glyph], index: usize, width: f32) -> bool {
    let Some(first) = glyphs.get(index + 1) else {
        return true;
    };
    // A break is allowed after `index`, so a scan from there sees what the paragraph's saw.
    let mut scan = Scan::default();
    scan.next(&glyphs[index..]);
    for end in index + 1..glyphs.len() {
        if glyphs[end].max_x() - first.pos.x > width {
            return false;
        }
        if scan.next(&glyphs[end..]) != Break::Prohibited {
            return true;
        }
    }
    true
}

/// Whether a row of `paragraph` (from `row_start` to the glyph that overflowed it) may be left
/// empty, egui's way to start a paragraph after a first-row indentation (its `leading_space`,
/// as for a label after another widget in a `horizontal_wrapped` layout) when the row has no
/// good place to break: the text moves to the next row, which has the full width.
///
/// egui asks at every row of such a paragraph, although only an empty first row helps: further
/// down it adds a blank line and lets the next row overflow the width by the indentation. Its
/// own rules rarely leave a row without a word boundary (a Latin word wider than the row);
/// kinsoku often does (a run it keeps together). So in a paragraph with Chinese or Japanese
/// (`has_cjk`, found out the first time it matters) only the first row may be empty, and only
/// if its text fits on the next row (or is a single glyph); otherwise the row breaks as it would
/// without the indentation. Other paragraphs keep egui's behaviour.
pub(crate) fn may_leave_row_empty(
    paragraph: &[Glyph],
    row_start: usize,
    overflowing: usize,
    width: f32,
    has_cjk: &mut Option<bool>,
) -> bool {
    (row_start == 0 && (overflowing == 0 || fits(&paragraph[..=overflowing], width)))
        || !*has_cjk.get_or_insert_with(|| paragraph.iter().any(|glyph| is_cjk(glyph.chr)))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        Break, Scan, common_pair, is_cjk, is_inseparable, is_line_end_prohibited,
        is_line_start_prohibited, line_break,
    };
    use crate::text::{
        FontDefinitions, FontId, FontsImpl, LayoutJob, TextFormat, TextOptions,
        font::is_cjk_break_allowed, layout,
    };

    /// Asserts `line_break` on each two-character string.
    fn expect(expected: Break, pairs: &[&str]) {
        for pair in pairs {
            let chars: Vec<char> = pair.chars().collect();
            let [before, after] = chars[..] else {
                panic!("{pair:?} is not two characters");
            };
            assert_eq!(line_break(before, after), expected, "{pair:?}");
        }
    }

    fn fonts() -> FontsImpl {
        FontsImpl::new(TextOptions::default(), FontDefinitions::default())
    }

    /// Rows of `text` wrapped at `width` points. egui's default fonts have no Han characters or
    /// kana (they are replacement glyphs), which is fine: the rules only look at characters.
    fn rows(fonts: &mut FontsImpl, text: &str, width: f32) -> Vec<String> {
        let mut job = LayoutJob::single_section(
            text.to_owned(),
            TextFormat::simple(FontId::proportional(14.0), crate::Color32::WHITE),
        );
        job.wrap.max_width = width;
        let galley = layout(fonts, 1.0, Arc::new(job));
        let rows: Vec<String> = galley.rows.iter().map(|row| row.text()).collect();
        assert_eq!(rows.concat(), text, "characters lost at {width}");
        rows
    }

    /// Kinsoku violations in `rows`: a break that [`line_break`] prohibits, a row (after the
    /// first) starting with one of the closing marks egui used to put there or a Bopomofo tone
    /// mark (the light-tone dot may start a line after a space: "˙ㄇㄚ"), or one (before the
    /// last) ending with an opening bracket.
    fn violations(rows: &[String]) -> Vec<String> {
        const NOT_FIRST: &str =
            "，．：；？！）］｝〕〉》」』】、。・ーっゃゅょァィゥェォッャュョ々～％…”’ˉˊˇˋ";
        const NOT_LAST: &str = "（［｛〔〈《「『【“‘˙";
        let mut bad = Vec::new();
        for (i, pair) in rows.windows(2).enumerate() {
            let (Some(last), Some(first)) = (pair[0].chars().last(), pair[1].chars().next()) else {
                continue;
            };
            if line_break(last, first) == Break::Prohibited
                || NOT_FIRST.contains(first)
                || NOT_LAST.contains(last)
            {
                bad.push(format!("{last:?} | {first:?} after row {i}: {rows:?}"));
            }
        }
        bad
    }

    /// Japanese and Chinese only.
    const CJK: &[&str] = &[
        "まっさらなスタート：育成のみで、パイロット同士は攻撃できません。「重要」（注意）次へ！本当に？はい、そうです。",
        "ロッカーのキャッシュ：人々は時々、こゝろの中でいすゞの名を呼んだ。コーヒー――それとも……紅茶？ェェッ！",
        "『星の航路』（第三版）――出発はまだ先だ。どこへ行くのか、誰も知らない。",
        "星门还差三个部件：星门完成后扫描即停止，只按实际进行的扫描收费。你确定吗？是的！（请注意）继续吧，好的；完成。",
        "按「确认」锁定（冻结）传送缓存：锁定后无法放入或取出物品，【注意】解锁后可再次修改。",
        "连接中断……正在重新连接，请稍候。服务器没有响应……请检查网络。他说：“你好！”——然后走了。",
        "注音符號：ㄓㄨˋ ㄧㄣ ㄈㄨˊ ㄏㄠˋ。ㄋㄧˇㄏㄠˇㄇㄚ˙？˙ㄇㄚ ˙ㄇㄚ，ㄨㄛˇㄏㄣˇㄏㄠˇ！",
    ];

    /// Japanese and Chinese with Latin words and numbers (from the `SpaceCorps` help texts).
    const MIXED: &[&str] = &[
        "シールドボーナス、Shield Wall ブースター（+25%）、Shield Boost バフで合計が増えます。",
        "账号的其余部分都会保留：信用点、Thulium、等级、荣誉、重置点数、永久加成、Skylab 和战队。",
        "齐射随机造成该数值的 80–100%，可能暴击，并乘以弹药倍率。机库显示的是未计增益时 x1 弹药齐射的伤害。",
        "欢迎来到 SpaceCorps。本周末双倍经验值活动进行中。4-2 星区目前海盗活动频繁，请谨慎前往。",
        "価格は￥1,200（税込）、送料は無料です。詳しくはSupport「FAQ」をご覧ください。",
    ];

    const GERMAN: &[&str] = &[
        "Wie Piloten beitreten: von den Offizieren unter „Verwaltung“ festgelegt",
        "Aktive Konfiguration (wechseln mit der Taste für „Konfig wechseln“)",
        "Kaufe „Slave Drones“ im Shop, um Slots freizuschalten, und ‚Mehr‘ dazu.",
        "»Die Flotte«, sagte sie, »wartet nicht.« – Ende…",
    ];

    // ---- Character classes -----------------------------------------------------------------

    fn assert_class(chars: &str, class: fn(char) -> bool, name: &str) {
        for c in chars.chars() {
            assert!(class(c), "{c:?} (U+{:04X}) should be {name}", c as u32);
        }
    }

    #[test]
    fn closing_brackets_and_quotes_may_not_start_a_line() {
        let closing = "）］｝〕〉》」』】〙〗〛〞〟｠｣’”»)]}";
        assert_class(closing, is_line_start_prohibited, "line-start prohibited");
        for c in closing.chars() {
            expect(Break::Prohibited, &[&format!("字{c}"), &format!("。{c}")]);
        }
    }

    #[test]
    fn punctuation_may_not_start_a_line() {
        // Commas, full stops, colons, semicolons, middle dots, ? and ! in full and half width.
        let marks = "、，､。．｡：；・･·？！‼⁇⁈⁉,.:;?!";
        assert_class(marks, is_line_start_prohibited, "line-start prohibited");
        for c in marks.chars() {
            expect(Break::Prohibited, &[&format!("字{c}"), &format!("か{c}")]);
            // After a Latin letter only the CJK marks count: "A," is Latin text.
            let after_latin = if is_cjk(c) {
                Break::Prohibited
            } else {
                Break::Upstream
            };
            expect(after_latin, &[&format!("A{c}")]);
        }
    }

    #[test]
    fn small_kana_prolonged_sound_and_iteration_marks_may_not_start_a_line() {
        let marks = "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿｧｨｩｪｫｬｭｮｯーｰ々ゝゞヽヾ〻゛゜ﾞﾟ";
        assert_class(marks, is_line_start_prohibited, "line-start prohibited");
        for c in marks.chars() {
            expect(Break::Prohibited, &[&format!("ロ{c}"), &format!("字{c}")]);
        }
        // Full-size kana may start a line (upstream broke "ロ" | "ッ" through its fallback).
        for c in "あいうえおつやゆよわかけアイウエオツヤユヨワカケｱｶ".chars()
        {
            assert!(!is_line_start_prohibited(c), "{c:?}");
            expect(Break::Allowed, &[&format!("ロ{c}")]);
        }
    }

    #[test]
    fn hyphens_postfixed_signs_and_ellipses_may_not_start_a_line() {
        let marks = "‐–゠〜～％‰℃°′″￠…‥⋯";
        assert_class(marks, is_line_start_prohibited, "line-start prohibited");
        for c in marks.chars() {
            expect(Break::Prohibited, &[&format!("字{c}")]);
        }
    }

    #[test]
    fn opening_brackets_and_prefixed_signs_may_not_end_a_line() {
        let opening = "（［｛〔〈《「『【〘〖〚〝｟｢‘“«([{￥＄￡＃";
        assert_class(opening, is_line_end_prohibited, "line-end prohibited");
        for c in opening.chars() {
            assert!(!is_line_start_prohibited(c), "{c:?}");
            expect(Break::Prohibited, &[&format!("{c}字"), &format!("{c}か")]);
        }
        // A line may break before one, also after a Latin word, a number or a mark.
        for c in "（［「『【《〈".chars() {
            let pairs = ["字", "。", "！", "b", "5", ")"].map(|b| format!("{b}{c}"));
            expect(Break::Allowed, &pairs.each_ref().map(String::as_str));
        }
    }

    #[test]
    fn inseparable_pairs_stay_together() {
        let pairs = ["——", "――", "……", "‥‥", "⋯⋯", "〳〵", "〴〵"];
        for pair in pairs {
            let chars: Vec<char> = pair.chars().collect();
            assert!(is_inseparable(chars[0], chars[1]), "{pair}");
        }
        expect(Break::Prohibited, &pairs);
        // A single dash may start and end a line; a dash and an ellipsis are not a pair.
        expect(Break::Allowed, &["字—", "—字"]);
        assert!(!is_inseparable('—', '…'), "—…");
    }

    #[test]
    fn marks_and_joiners_stay_with_their_character() {
        // か + a combining voiced sound mark (a decomposed が), 葛 + an ideographic variation
        // selector, word joiners.
        expect(
            Break::Prohibited,
            &["か\u{3099}", "葛\u{E0100}", "字\u{2060}", "\u{2060}字"],
        );
        expect(Break::Allowed, &["\u{3099}き"]);
    }

    /// Bopomofo (zhuyin) tone marks are spacing letters, not combining marks: they stay with
    /// the syllable before them, and the light-tone dot, which comes before or after its
    /// syllable, with both neighbours. A line may break after a tone mark, between syllables.
    #[test]
    fn bopomofo_tone_marks_stay_with_their_syllable() {
        assert_class("ˉˊˇˋ˪˫˙", is_line_start_prohibited, "line-start prohibited");
        expect(
            Break::Prohibited,
            &[
                "ㄣˉ", "ㄨˊ", "ㄠˇ", "ㄨˋ", "ㄠ˪", "ㄠ˫", "ㄚ˙", "˙ㄇ", "字ˋ", "ˋ，", "ˇ。",
            ],
        );
        expect(Break::Allowed, &["ˋㄧ", "ˇㄏ", "ˉ字", "ㄓㄨ"]);
        // Next to other characters they are egui's.
        expect(Break::Upstream, &["aˇ", "ˇb", "a˙", "˙b", "ˇ˙"]);
    }

    /// Every mark in egui's own list still may not start a line after a Han character or kana:
    /// the patch only adds to it.
    #[test]
    fn egui_list_is_included() {
        let upstream = ")]｝〕〉》」』】〙〗〟'\"｠»ヽヾーァィゥェォッャュョヮヵヶぁぃぅぇぉっゃゅょゎゕゖㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ々〻‐゠–〜?!‼⁇⁈⁉・、:;,。.";
        for c in upstream.chars() {
            assert!(!is_cjk_break_allowed(c), "{c:?} left egui's list");
            expect(Break::Prohibited, &[&format!("日{c}"), &format!("か{c}")]);
        }
        assert_class(upstream, is_line_start_prohibited, "line-start prohibited");
    }

    #[test]
    fn cjk_characters() {
        assert_class(
            "日本語かなカナ㐀𠀋〇、。「」〜々，！（）：；？％ｱﾞ￥ㄅ豈︵",
            is_cjk,
            "CJK",
        );
        for c in "aZ09 -()…—“”‘’«»„ÄßПλ한글ㄱ😀".chars() {
            assert!(!is_cjk(c), "{c:?} (U+{:04X}) is not CJK", c as u32);
        }
    }

    // ---- Break rules -----------------------------------------------------------------------

    #[test]
    fn between_han_characters_and_kana_a_line_may_break() {
        expect(
            Break::Allowed,
            &[
                "日本", "本語", "語か", "かな", "なカ", "カナ", "ナ日", "〇日", "𠀋日",
            ],
        );
    }

    #[test]
    fn latin_words_and_numbers_inside_cjk_text() {
        // A Latin word or a number is one unit: breaks before and after it, none inside.
        expect(Break::Allowed, &["用T", "m等", "的8", "0年"]);
        expect(Break::Upstream, &["Th", "25", "+2"]);
        // After CJK punctuation before a Latin word or a number, where egui had no break and
        // fell back to breaking anywhere ("Thulium" | "、").
        expect(Break::Allowed, &["、T", "。4", "）x", "」S", "，8"]);
        // Never before CJK closing punctuation, also not after an ASCII mark ("+25%" | "）")
        // or before an ellipsis after a Latin word.
        expect(
            Break::Prohibited,
            &["m、", "%）", "5％", ")、", "-」", "s…", "m…"],
        );
        // Never after an opening bracket or a prefixed sign.
        expect(Break::Prohibited, &["（+", "「F", "(中", "“你", "￥1"]);
        // A space breaks as in upstream egui.
        expect(Break::Upstream, &["字 ", " S", "　字", "、 "]);
    }

    #[test]
    fn chinese_quotes() {
        // “ ” and ‘ ’ open and close quotes in Chinese.
        expect(Break::Allowed, &["说“", "：“", "”然", "’字"]);
        expect(Break::Prohibited, &["“你", "‘你", "好”", "！”", "好’"]);
        // " and ' open and close: no break on either side of them next to CJK.
        expect(
            Break::Prohibited,
            &["说\"", "\"你", "好'", "'你", "\"（", "）\""],
        );
        expect(Break::Upstream, &["s'", "'s", "\"T", "T\""]);
    }

    /// A space next to a bracket of Chinese or Japanese text: no line starts with the closing
    /// mark after it or ends with the opening one before it (as upstream, a line still breaks
    /// after a space elsewhere).
    #[test]
    fn spaces_do_not_separate_brackets_from_their_text() {
        expect(
            Break::Prohibited,
            &[
                "\u{3000}」",
                " 」",
                "\u{A0}）",
                "\u{3000}。",
                "\u{3000}…",
                "\u{3000})",
                "（\u{3000}",
                "（\u{A0}",
                "「 ",
                "【 ",
                "“\u{3000}",
                "‘\u{3000}",
            ],
        );
        expect(
            Break::Upstream,
            &[
                "字 ",
                " 字",
                "\u{3000}字",
                "」\u{3000}",
                "\u{3000}「",
                "。 ",
                "“ ",
                " )",
                "a ",
            ],
        );
    }

    /// No Chinese or Japanese character in the pair: egui's own rules, so text without them
    /// lays out as upstream (`RowBreakCandidates` records every other break as egui does). The
    /// only exceptions are the rules of Unicode line breaking that egui lacks, in every script:
    /// an ellipsis after a non-space, "——" "……", an ASCII opening bracket before anything, a
    /// no-break space after anything but a space, and a word joiner next to anything but a
    /// space.
    #[test]
    fn text_without_cjk_is_left_to_egui() {
        let mut sample: Vec<char> = (' '..='~').collect();
        sample.extend('\u{A0}'..='\u{24F}'); // Latin-1, Latin Extended-A and -B
        sample.extend('\u{370}'..='\u{3FF}'); // Greek
        sample.extend('\u{400}'..='\u{4FF}'); // Cyrillic
        sample.extend('\u{600}'..='\u{6FF}'); // Arabic
        sample.extend('\u{2000}'..='\u{206F}'); // General Punctuation („“‚‘ – — …)
        sample.extend('\u{20A0}'..='\u{20CF}'); // Currency Symbols
        sample.extend('\u{2100}'..='\u{214F}'); // Letterlike Symbols (℃)
        sample.extend('\u{AC00}'..='\u{AC80}'); // Hangul syllables
        sample.extend('\u{1F600}'..='\u{1F64F}'); // Emoji
        sample.push('\u{FEFF}'); // the zero-width no-break space is a word joiner too
        let mut checked = 0_usize;
        for &a in &sample {
            assert!(!is_cjk(a), "{a:?}");
            for &b in &sample {
                let ellipsis = matches!(b, '…' | '‥' | '⋯');
                let opening_bracket = matches!(a, '(' | '[' | '{');
                let before_nbsp = b == '\u{A0}' && !(a.is_whitespace() && a != '\u{A0}');
                let space = |c: char| c.is_whitespace() && c != '\u{A0}';
                let word_joiner = (matches!(a, '\u{2060}' | '\u{FEFF}')
                    || matches!(b, '\u{2060}' | '\u{FEFF}'))
                    && !space(a)
                    && !space(b);
                let expected = if is_inseparable(a, b)
                    || (ellipsis && !a.is_whitespace() && !b.is_whitespace())
                    || opening_bracket
                    || before_nbsp
                    || word_joiner
                {
                    Break::Prohibited
                } else {
                    Break::Upstream
                };
                assert_eq!(line_break(a, b), expected, "{a:?} {b:?}");
                checked += 1;
            }
        }
        assert!(checked > 1_000_000, "only {checked} pairs");
    }

    // ---- Unicode line breaking rules that hold in every script -----------------------------

    /// No break just before a no-break space (UAX #14 class GL): not after a mark, a letter, a
    /// digit, Chinese or Japanese text or another no-break space; after a space egui breaks as
    /// always. Only the break before it is ruled on: after it egui decides.
    #[test]
    fn no_break_before_a_no_break_space() {
        expect(
            Break::Prohibited,
            &[
                "]\u{a0}",
                ")\u{a0}",
                "a\u{a0}",
                "5\u{a0}",
                ".\u{a0}",
                ",\u{a0}",
                "-\u{a0}",
                ":\u{a0}",
                "!\u{a0}",
                "…\u{a0}",
                "П\u{a0}",
                "한\u{a0}",
                "”\u{a0}",
                "\u{a0}\u{a0}",
                "字\u{a0}",
                "。\u{a0}",
                "」\u{a0}",
                "（\u{a0}",
            ],
        );
        expect(
            Break::Upstream,
            &[
                " \u{a0}",
                "\t\u{a0}",
                "\u{2003}\u{a0}",
                "\u{3000}\u{a0}",
                "\u{a0} ",
                "\u{a0}a",
                "\u{a0}]",
                "\u{a0}-",
                "\u{a0}5",
                "\u{a0}字",
            ],
        );
        // Closing Chinese and Japanese marks stay off the start of a line after it (as before).
        expect(Break::Prohibited, &["\u{a0}）", "\u{a0}。"]);
    }

    /// No break after an ASCII opening bracket (UAX #14 class OP), before any character,
    /// spaces and the other brackets included: upstream broke after it as after any ASCII mark.
    #[test]
    fn no_break_after_an_ascii_opening_bracket() {
        for bracket in "([{".chars() {
            let pairs: Vec<String> = [
                "a", "G", "1", ".", "-", "П", "字", " ", "\t", "\u{a0}", "\u{3000}", "(", "[", "{",
                ")", "“", "…",
            ]
            .iter()
            .map(|after| format!("{bracket}{after}"))
            .collect();
            expect(
                Break::Prohibited,
                &pairs.iter().map(String::as_str).collect::<Vec<_>>(),
            );
        }
        // Nested and adjacent brackets: only an opening one holds the break back.
        expect(
            Break::Prohibited,
            &["([", "[(", "{[", "((", "[[", "{{", "[{"],
        );
        expect(
            Break::Upstream,
            &[
                ")(", "](", "][", "}{", ")[", "))", "]]", "}}", "])", "a(", "a[", "a{", "5(", "-(",
            ],
        );
        // Not class OP: closing brackets, and before an opening one after a space.
        expect(
            Break::Upstream,
            &[")a", "]G", "}1", " (", " [", " {", "\u{a0}(", "\u{a0}["],
        );
    }

    /// No break on either side of a word joiner (UAX #14 class WJ), after any character before
    /// it and before any after it: a joiner glued to a clan tag's marks keeps "[A-" and "B]"
    /// together. Next to a space egui decides, as for the no-break space.
    #[test]
    fn no_break_next_to_a_word_joiner() {
        for joiner in ['\u{2060}', '\u{FEFF}'] {
            let pairs: Vec<String> = [
                "a", "G", "1", ".", "-", "_", "/", "]", ")", "!", "П", "한", "字", "。", "」",
                "（", "“", "…", "\u{a0}", "\u{2060}", "\u{FEFF}",
            ]
            .iter()
            .flat_map(|other| [format!("{other}{joiner}"), format!("{joiner}{other}")])
            .collect();
            expect(
                Break::Prohibited,
                &pairs.iter().map(String::as_str).collect::<Vec<_>>(),
            );
            // Next to a space the break stays egui's (a no-break space is not one here).
            let spaced: Vec<String> = [" ", "\t", "\u{2003}"]
                .iter()
                .flat_map(|space| [format!("{space}{joiner}"), format!("{joiner}{space}")])
                .collect();
            expect(
                Break::Upstream,
                &spaced.iter().map(String::as_str).collect::<Vec<_>>(),
            );
        }
        // A zero-width joiner is not a word joiner: it glues only next to Chinese or Japanese.
        expect(Break::Upstream, &["a\u{200D}", "\u{200D}a", "-\u{200D}"]);
        expect(Break::Prohibited, &["字\u{200D}", "\u{200D}字"]);
    }

    /// German quotes and the other typographic quotes are not touched: „Tor“, ‚Mehr‘, »Flotte«
    /// (“ ‘ « only count as opening brackets next to Chinese or Japanese characters).
    #[test]
    fn typographic_quotes_are_left_to_egui() {
        expect(
            Break::Upstream,
            &[
                "„T", "‚M", "»F", "“ ", "‘ ", "« ", "r“", "e‘", "e«", "“T", "‘M", "«F", "“(",
            ],
        );
        // Except that "(" holds back the break after it as any opening bracket does.
        expect(Break::Prohibited, &["(“", "(„", "(»"]);
    }

    /// Every position of a paragraph (after each of its characters but the last) that its
    /// [`Scan`] prohibits a break at, as character indices.
    fn prohibited_positions(fonts: &mut FontsImpl, text: &str) -> Vec<usize> {
        let glyphs = glyphs(fonts, text);
        assert_eq!(glyphs.len(), text.chars().count(), "{text:?}");
        let mut scan = Scan::default();
        (0..glyphs.len())
            .filter(|&i| scan.next(&glyphs[i..]) == Break::Prohibited)
            .collect()
    }

    /// Spaces between an ASCII opening bracket and its text don't let a break in either: as
    /// with Unicode's OP SP* × (LB14), as with the brackets of Chinese and Japanese text.
    #[test]
    fn spaces_after_an_ascii_opening_bracket_keep_the_text_with_it() {
        let mut fonts = fonts();
        // Positions: a0 ␣1 (2 ␣3 b4 ␣5 )6 ␣7 c8
        assert_eq!(prohibited_positions(&mut fonts, "a ( b ) c"), [2, 3]);
        // Any number of spaces, of any kind, and no-break spaces.
        assert_eq!(
            prohibited_positions(&mut fonts, "x [  \u{a0} y"),
            [2, 3, 4, 5, 6]
        );
        // Brackets after brackets, the bracket alone, and none at the end (nothing to keep).
        assert_eq!(prohibited_positions(&mut fonts, "([ { x"), [0, 1, 2, 3, 4]);
        assert_eq!(prohibited_positions(&mut fonts, "b ( "), [2]);
        assert_eq!(prohibited_positions(&mut fonts, "b ("), Vec::<usize>::new());
        // Closing brackets and spaces elsewhere: no change.
        assert_eq!(
            prohibited_positions(&mut fonts, "a ) b ] c"),
            Vec::<usize>::new()
        );
        // The state doesn't leak past the text after the spaces.
        assert_eq!(prohibited_positions(&mut fonts, "( a b"), [0, 1]);
    }

    /// Rows of `text` at every whole width from the widest run the rules keep together
    /// (`min_width`, plus a point) to `max_width`.
    fn rows_at_every_width(
        fonts: &mut FontsImpl,
        text: &str,
        max_width: usize,
    ) -> Vec<(usize, Vec<String>)> {
        let min_width = widest_scanned_unit(fonts, text).ceil() as usize + 1;
        assert!(min_width < 120, "{min_width}: {text}");
        (min_width..=max_width)
            .map(|width| (width, rows(fonts, text, width as f32)))
            .collect()
    }

    /// Text with brackets of every kind, nested, adjacent, with and without spaces, and inside
    /// tokens without spaces (a URL, a formula), the way chat lines and help texts have them.
    const BRACKETED: &[&str] = &[
        "Open the hatch (deck 4, port side) and check [the seal] before {launch}: nothing else (really) matters.",
        "Nested (calls [with {braces}] inside (parentheses)) and [[double]] ((pairs)) {{x}} ([{mixed}]) end",
        "Adjacent: a(b)c[d]e{f}g (a)(b)[c][d]{e}{f} )( ][ }{ done (a)[b]{c}",
        "Spaces after them: ( inside ) [ inside ] { inside } and ( ( x ) ) [ [ y ] ] more (  wide  ) text",
        "No spaces at all: f(x)+g[y]*h{z}=k(l(m[n{o}]))-p(q) https://example.com/a(b)/c[d]?e={f}",
        "Long_token(with_brackets)[and_more]{and_even_more}(((deep(nesting(here))))) done",
        "(start of the text and the end of it)[",
    ];

    /// No row ends with an ASCII opening bracket (and its spaces) at any width that holds the
    /// widest run kept together: there is always a break before it instead. Upstream broke
    /// after it at most widths: "[" | "G] …".
    #[test]
    fn no_row_ends_with_an_ascii_opening_bracket() {
        let mut fonts = fonts();
        let mut layouts = 0;
        for text in BRACKETED {
            for (width, rows) in rows_at_every_width(&mut fonts, text, 300) {
                layouts += 1;
                for row in &rows[..rows.len() - 1] {
                    let last = row.trim_end_matches(char::is_whitespace).chars().last();
                    assert!(
                        !last.is_some_and(|c| matches!(c, '(' | '[' | '{')),
                        "{width}: {rows:?}"
                    );
                }
            }
        }
        assert!(layouts > 1000, "{layouts}");
    }

    /// The same at widths narrower than the widest run kept together, down to a single
    /// character, where a break inside the run can't be avoided: nothing is lost (`rows` checks
    /// that), and no row overflows by a whole glyph or more (as upstream, a row can overflow by
    /// less than a glyph in columns a few characters wide), also for a run of brackets wider
    /// than the row: no break is allowed in it, yet it wraps at the row's width.
    #[test]
    fn brackets_wrap_at_any_width() {
        let mut fonts = fonts();
        for text in BRACKETED.iter().copied().chain([
            "((((((((((((((((((((((((((((((((((((((((((((((((((((",
            "[ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ x",
            "([{([{([{([{([{([{([{([{([{([{([{([{([{([{([{([{x",
        ]) {
            for width in 1..=200 {
                let width = width as f32;
                let rows = row_widths(&mut fonts, text, width, 0.0);
                for (i, (row, row_width, _)) in rows.iter().enumerate() {
                    if *row_width <= width + 0.5 + 1e-3 {
                        continue;
                    }
                    // Wider than the row: the row without its last glyph fits.
                    let last = row.chars().last().expect("a row that overflows has glyphs");
                    let last_width = self::width(&mut fonts, &last.to_string());
                    assert!(
                        *row_width - last_width <= width + 0.5 + 1e-3,
                        "{width}: row {i} {row:?} overflows by a glyph or more: {rows:?}"
                    );
                }
                let text_again: String = rows.iter().map(|(row, ..)| row.as_str()).collect();
                assert_eq!(text_again, text, "characters lost at {width}");
            }
        }
    }

    /// A company tag joined to the name with a no-break space ("[T]\u{a0}nova:") stays with it:
    /// at every width that holds "[T]\u{a0}nova:" the row with the tag has the whole name, and
    /// whatever the width, no row starts with the no-break space and the row with the tag holds
    /// the start of the name (also for a name wider than the row). Upstream left "[T]" alone
    /// ("[T]" | " nova:") or split the tag ("[" | "T] nova:").
    #[test]
    fn a_tag_stays_with_its_name() {
        let mut fonts = fonts();
        let names = ["nova", "Commander_Starfighter_Alpha27", &"W".repeat(32)];
        for tag in ["[T]", "[G]", "(M)", "{X}"] {
            // Narrower than the tag and one letter, they can't share a row.
            let both = width(&mut fonts, &format!("{tag}\u{a0}W")).ceil() as usize + 1;
            for name in names {
                let text = format!("19:42  {tag}\u{a0}{name}: anyone for the gate run?");
                let whole = format!("{tag}\u{a0}{name}:");
                let whole_width = width(&mut fonts, &whole).ceil() as usize + 1;
                for max_width in both..=600 {
                    let rows = rows(&mut fonts, &text, max_width as f32);
                    assert!(
                        rows.iter().all(|r| !r.starts_with('\u{a0}')),
                        "{max_width}: starts with the gap: {rows:?}"
                    );
                    let with_name = rows
                        .iter()
                        .find_map(|r| r.split_once(tag))
                        .is_some_and(|(_, rest)| !rest.trim().is_empty());
                    assert!(with_name, "{max_width}: the tag is alone: {rows:?}");
                    if max_width >= whole_width {
                        assert!(
                            rows.iter().any(|r| r.contains(&whole)),
                            "{max_width}: {whole:?} was split: {rows:?}"
                        );
                    }
                }
            }
        }
    }

    /// A clan tag whose marks are glued to the next character with word joiners (the SpaceCorps
    /// chat does: "[A-\u{2060}B]") is never split, at any width that holds it: upstream broke
    /// after the dash or the mark, "[A-" | "B]". The company tag and the name stay glued with
    /// no-break spaces as before: "[T]\u{a0}[A-B]\u{a0}name:".
    #[test]
    fn a_clan_tag_glued_with_word_joiners_is_never_split() {
        let mut fonts = fonts();
        let mut layouts = 0;
        for clan in ["A-B", "A.B", "-_-", "!?", "A/B", "A:B", "A-B-C"] {
            // A word joiner after each ASCII mark but an opening bracket, as the chat puts them.
            let mut tag = String::from("[");
            for c in clan.chars() {
                tag.push(c);
                if c.is_ascii_punctuation() {
                    tag.push('\u{2060}');
                }
            }
            tag.push(']');
            // Narrower than both tags, their no-break spaces and one letter of the name, a row
            // can't hold the run the rules keep together and breaks where it is full, which can
            // be inside the clan tag ("[A-B" | "]\u{a0}nova:" when the space doesn't fit).
            let tag_width =
                width(&mut fonts, &format!("[T]\u{a0}{tag}\u{a0}W")).ceil() as usize + 1;
            for name in ["nova", "Commander_Starfighter_Alpha27", &"W".repeat(32)] {
                let text = format!("19:42  [T]\u{a0}{tag}\u{a0}{name}: anyone for the gate run?");
                let start = text[..text.find(&tag).expect("the tag is in the text")]
                    .chars()
                    .count();
                let end = start + tag.chars().count();
                for max_width in tag_width..=600 {
                    let rows = rows(&mut fonts, &text, max_width as f32);
                    layouts += 1;
                    let mut offset = 0;
                    for row in &rows[..rows.len() - 1] {
                        offset += row.chars().count();
                        assert!(
                            offset <= start || offset >= end,
                            "{max_width}: the tag {tag:?} is split: {rows:?}"
                        );
                    }
                }
            }
        }
        assert!(layouts > 3000, "{layouts}");
    }

    /// A plain space between tag and name may still break (a space is egui's): the tag's own
    /// bracket doesn't, and the closing one never starts a row here.
    #[test]
    fn a_tag_with_a_plain_space_may_leave_its_name_but_never_splits() {
        let mut fonts = fonts();
        let text = "19:42  [T] Commander_Starfighter_Alpha27: anyone for the gate run?";
        let mut alone = 0;
        for width in 60..=600 {
            let rows = rows(&mut fonts, text, width as f32);
            assert!(
                rows.iter().all(|r| !r.trim_end().ends_with('[')),
                "{width}: {rows:?}"
            );
            alone += rows.iter().any(|r| r.trim_end().ends_with("[T]")) as usize;
        }
        assert!(alone > 0, "a plain space never let the tag end a row");
    }

    /// Text without brackets, no-break spaces, ellipses or doubled dashes wraps as upstream:
    /// egui's rules prefer a space, then a dash, then a mark, then any character. The rows are
    /// those upstream epaint 0.36.2 gives (recorded from it).
    #[test]
    fn text_without_the_rules_characters_keeps_its_rows() {
        let mut fonts = fonts();
        let text = "Plain words, dash-separated-words, a URL www.example.com/a/b_c and some_snake_case \
                    names: 12,345.67 and “quoted” or „German“ text; the end.";
        assert_eq!(
            rows(&mut fonts, text, 60.0),
            [
                "Plain ",
                "words, ",
                "dash-",
                "separate",
                "d-words, ",
                "a URL ",
                "www.",
                "example.",
                "com/a/b_",
                "c and ",
                "some_",
                "snake_",
                "case ",
                "names: ",
                "12,345.",
                "67 and ",
                "“quoted” ",
                "or ",
                "„German“",
                " text; the ",
                "end.",
            ],
            "60 pt"
        );
        assert_eq!(
            rows(&mut fonts, text, 100.0),
            [
                "Plain words, ",
                "dash-separated-",
                "words, a URL ",
                "www.example.",
                "com/a/b_c and ",
                "some_snake_",
                "case names: ",
                "12,345.67 and ",
                "“quoted” or ",
                "„German“ text; ",
                "the end.",
            ],
            "100 pt"
        );
    }

    // ---- Layout ----------------------------------------------------------------------------

    fn width(fonts: &mut FontsImpl, text: &str) -> f32 {
        let job = LayoutJob::single_section(
            text.to_owned(),
            TextFormat::simple(FontId::proportional(14.0), crate::Color32::WHITE),
        );
        layout(fonts, 1.0, Arc::new(job)).size().x
    }

    /// The width of the widest run of `text` that kinsoku keeps on one line ("茶？ェェッ！",
    /// "m、", "%）、"): narrower rows must break inside such a run, as upstream would.
    fn widest_unit(fonts: &mut FontsImpl, text: &str) -> f32 {
        let chars: Vec<char> = text.chars().collect();
        let mut widest = 0.0_f32;
        let mut start = 0;
        for i in 0..chars.len() {
            let joined = chars
                .get(i + 1)
                .is_some_and(|&next| line_break(chars[i], next) == Break::Prohibited);
            if !joined {
                let unit: String = chars[start..=i].iter().collect();
                widest = widest.max(width(fonts, &unit));
                start = i + 1;
            }
        }
        widest
    }

    /// Japanese and Chinese paragraphs, alone and with Latin words and numbers, at every width
    /// from the widest run kinsoku keeps together to 420 points: no line starts with a closing
    /// mark or ends with an opening one, and no "——" or "……" is split. Upstream egui fails
    /// this at most widths.
    #[test]
    fn wrapped_cjk_text_follows_kinsoku() {
        let mut fonts = fonts();
        let mut bad = Vec::new();
        for text in CJK.iter().chain(MIXED) {
            let min_width = widest_unit(&mut fonts, text).ceil() as usize + 1;
            assert!(min_width < 100, "{min_width}: {text}");
            for width in (min_width..=420).step_by(2) {
                let rows = rows(&mut fonts, text, width as f32);
                bad.extend(
                    violations(&rows)
                        .into_iter()
                        .map(|v| format!("{width}: {v}")),
                );
            }
        }
        assert!(
            bad.is_empty(),
            "{} violations:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// German quotes „…“ and ‚…‘ close with the characters that open Chinese and English ones:
    /// no German line starts with “ or ‘ at any width that holds the longest word (an earlier
    /// version of the game's patch broke before them).
    #[test]
    fn german_closing_quotes_stay_on_their_word() {
        let mut fonts = fonts();
        for text in GERMAN {
            let min_width = text
                .split(' ')
                .map(|word| width(&mut fonts, word))
                .fold(0.0, f32::max)
                .ceil() as usize
                + 1;
            for width in (min_width..=460).step_by(2) {
                let rows = rows(&mut fonts, text, width as f32);
                assert!(
                    rows.iter().skip(1).all(|r| !r.starts_with(['“', '‘', '«'])),
                    "{width}: {rows:?}"
                );
            }
        }
    }

    /// A run of marks that may not start a line still wraps at the width, filling each row as
    /// upstream does, rather than overflowing or breaking after every character.
    #[test]
    fn a_run_of_prohibited_marks_still_wraps() {
        let mut fonts = fonts();
        for text in [
            "ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー",
            "……………………………………………………………………",
        ] {
            let rows = rows(&mut fonts, text, 100.0);
            assert!(rows.len() > 2, "{rows:?}");
            assert!(
                rows.iter().rev().skip(1).all(|r| r.chars().count() > 2),
                "{rows:?}"
            );
        }
    }

    /// A Latin word wider than the row still breaks inside, but not before the "、" after it
    /// (at any width that holds "m、" and "点、", which may not be split).
    #[test]
    fn a_word_wider_than_the_row_keeps_its_comma() {
        let mut fonts = fonts();
        let text = "信用点、Thulium、等级";
        let min_width = ["m、", "点、"]
            .map(|t| width(&mut fonts, t))
            .into_iter()
            .fold(0.0, f32::max);
        for width in (min_width.ceil() as usize..=120).step_by(1) {
            let rows = rows(&mut fonts, text, width as f32);
            assert!(
                rows.iter().skip(1).all(|r| !r.starts_with('、')),
                "{width}: {rows:?}"
            );
        }
    }

    /// Glyphs of `text` laid out on one row.
    fn glyphs(fonts: &mut FontsImpl, text: &str) -> Vec<crate::text::Glyph> {
        let job = LayoutJob::single_section(
            text.to_owned(),
            TextFormat::simple(FontId::proportional(14.0), crate::Color32::WHITE),
        );
        let galley = layout(fonts, 1.0, Arc::new(job));
        galley
            .rows
            .iter()
            .flat_map(|row| row.glyphs.clone())
            .collect()
    }

    /// Width of the widest run of `text` that a paragraph's [`Scan`] keeps together.
    fn widest_scanned_unit(fonts: &mut FontsImpl, text: &str) -> f32 {
        let glyphs = glyphs(fonts, text);
        let mut scan = Scan::default();
        let (mut widest, mut start) = (0.0_f32, 0);
        for i in 0..glyphs.len() {
            if scan.next(&glyphs[i..]) != Break::Prohibited {
                widest = widest.max(glyphs[i].max_x() - glyphs[start].pos.x);
                start = i + 1;
            }
        }
        widest
    }

    /// The shortcut for common pairs rules as the full rules do (and leaves no bracket to keep
    /// across spaces), for the characters below U+0300 and a sample of the rest below U+2000,
    /// the CJK punctuation, Hiragana and Katakana blocks, some Han characters and the rules'
    /// own characters, with and without width.
    #[test]
    fn common_pairs_are_ruled_as_by_the_full_rules() {
        let mut fonts = fonts();
        let template = glyphs(&mut fonts, "字")[0];
        let glyph = |chr, advance_width| crate::text::Glyph {
            chr,
            advance_width,
            ..template
        };
        let mut chars: Vec<char> = (0..0x300)
            .chain((0x300..0x2000).step_by(7))
            .chain(0x3000..0x3100)
            .chain([
                0x3400, 0x4DBF, 0x4E00, 0x5B57, 0x9FFF, 0xA0, 0x1680, 0xFE0F, 0xE0100,
            ])
            .filter_map(char::from_u32)
            .collect();
        chars.extend("）」。、，！？（「“‘”’«»…‥——〳〵￥％\u{2060}\u{200D}\u{3000}ˇˋ˙".chars());
        let mut shortcuts = 0;
        for &before in &chars {
            for &after in &chars {
                for width in [0.0, 7.0] {
                    let pair = [glyph(before, 7.0), glyph(after, width)];
                    if let Some(ruling) = common_pair(&pair[0], &pair[1]) {
                        shortcuts += 1;
                        let mut scan = Scan {
                            after_opener: true,
                            before_closer: true,
                        };
                        assert_eq!(scan.rules(&pair), ruling, "{before:?} {after:?}, {width}");
                        assert!(
                            !scan.after_opener && !scan.before_closer,
                            "{before:?} {after:?}"
                        );
                    }
                }
            }
        }
        assert!(shortcuts > 500_000, "{shortcuts}");
    }

    /// Spaces between brackets and their text, as Japanese sometimes has: at every width that
    /// holds the widest run kept together, no row starts with a closing mark after its leading
    /// spaces or ends with an opening one before its trailing spaces.
    #[test]
    fn brackets_stay_with_their_text_across_spaces() {
        let mut fonts = fonts();
        let text = "第一章　「　始まり　」　そして誰も（ 注意 ）いなくなった　。「重要」　です";
        let min_width = widest_scanned_unit(&mut fonts, text).ceil() as usize + 1;
        assert!(min_width < 120, "{min_width}");
        for width in (min_width..=420).step_by(1) {
            let rows = rows(&mut fonts, text, width as f32);
            for pair in rows.windows(2) {
                let last = pair[0].trim_end_matches(char::is_whitespace).chars().last();
                let first = pair[1]
                    .trim_start_matches(char::is_whitespace)
                    .chars()
                    .next();
                assert!(
                    !last.is_some_and(is_line_end_prohibited)
                        && !first.is_some_and(is_line_start_prohibited),
                    "{width}: {rows:?}"
                );
            }
        }
    }

    /// Row widths of `text` wrapped at `width` points after a first-row indentation of
    /// `leading_space` points (as egui gives a label after another widget in a
    /// `horizontal_wrapped` layout), with the number of glyphs with a width in each row.
    fn row_widths(
        fonts: &mut FontsImpl,
        text: &str,
        width: f32,
        leading_space: f32,
    ) -> Vec<(String, f32, usize)> {
        let mut job = LayoutJob::single_section(
            text.to_owned(),
            TextFormat::simple(FontId::proportional(14.0), crate::Color32::WHITE),
        );
        job.sections[0].leading_space = leading_space;
        job.wrap.max_width = width;
        let galley = layout(fonts, 1.0, Arc::new(job));
        galley
            .rows
            .iter()
            .map(|row| {
                let visible = row.glyphs.iter().filter(|g| g.advance_width > 0.0).count();
                (row.text(), row.size.x, visible)
            })
            .collect()
    }

    /// Chinese and Japanese rows never overflow the width (egui's rounding allows half a point)
    /// unless a single glyph is wider, also where a run kinsoku keeps together is wider than a
    /// row and must break: it breaks where the row is full, and a run moved to the next row
    /// whole fits there. The same after a first-row indentation (egui's `leading_space`), where
    /// only the first row may be empty (when what fits after the indentation ends inside a run
    /// kinsoku keeps together): an empty row further down would add a blank line and let the
    /// row after it overflow by the indentation.
    #[test]
    fn cjk_rows_never_overflow() {
        let mut fonts = fonts();
        let texts = CJK.iter().copied().chain([
            "字「『（【〔〈《“‘字’”》〉〕】）』」。字「『（【〔〈《“‘字’”》〉〕】）』」、字",
            "「えっ？」「ちょっと待って！」――彼は言った。「そんな……まさか」。ァィゥェォッャュョ",
            "。。。。。。、、、、、，，，，！！！？？？：：；；・・・ーーーーッッッ",
        ]);
        for text in texts {
            for leading_space in [0.0, 8.0, 37.0] {
                let mut width = 1.0;
                while width <= 240.0 {
                    let rows = row_widths(&mut fonts, text, width, leading_space);
                    for (i, (row, row_width, visible)) in rows.iter().enumerate() {
                        assert!(
                            *row_width <= width + 0.5 + 1e-3 || *visible <= 1,
                            "{width}, indented {leading_space}: {row:?} is {row_width} wide"
                        );
                        assert!(
                            i == 0 || !row.is_empty(),
                            "{width}, indented {leading_space}: empty row {i}: {rows:?}"
                        );
                    }
                    width += 0.5;
                }
            }
        }
    }

    /// "——" and "……" stay whole at every width that holds them, in any script — also where
    /// the rest of the row is a word wider than the row ("over……").
    #[test]
    fn dashes_and_ellipses_stay_whole() {
        let mut fonts = fonts();
        let pair_width = ["——", "……", "――"]
            .map(|t| width(&mut fonts, t))
            .into_iter()
            .fold(0.0, f32::max);
        for text in [
            "Wait… what? He said——no, he shouted——that it was over……",
            "Loading…… please wait… still loading… — done — really—really—done",
            "连接中断……正在重新连接——请稍候。服务器没有响应……请检查网络",
            "コーヒー――それとも……紅茶？",
        ] {
            let mut width = pair_width.ceil();
            while width <= 300.0 {
                let rows = rows(&mut fonts, text, width);
                for pair in rows.windows(2) {
                    let (Some(last), Some(first)) =
                        (pair[0].chars().last(), pair[1].chars().next())
                    else {
                        continue;
                    };
                    assert!(!is_inseparable(last, first), "{width}: {rows:?}");
                }
                width += 0.5;
            }
        }
    }

    /// A Chinese or Japanese character stays with its combining mark or variation selector at
    /// every width, even where the shaper gives the selector the base's character (egui's
    /// default fonts have no glyph for it) and even when the row is narrower than one glyph.
    #[test]
    fn clusters_stay_whole() {
        let mut fonts = fonts();
        let text = "葛\u{E0100}字\u{FE0F}か\u{3099}き\u{3099}葛\u{E0100}辻\u{E0101}";
        for width in (0..=120).map(|w| w as f32 * 0.5) {
            let mut job = LayoutJob::single_section(
                text.to_owned(),
                TextFormat::simple(FontId::proportional(14.0), crate::Color32::WHITE),
            );
            job.wrap.max_width = width;
            let galley = layout(&mut fonts, 1.0, Arc::new(job));
            for row in galley.rows.iter().skip(1) {
                assert!(
                    row.glyphs.first().is_some_and(|g| g.advance_width > 0.0),
                    "{width}: a row starts inside a cluster: {:?}",
                    galley.rows.iter().map(|r| r.text()).collect::<Vec<_>>()
                );
            }
        }
    }

    /// The upstream cases (`text_layout::tests::test_cjk` and `test_pre_cjk`) keep their rows.
    #[test]
    fn upstream_cjk_rows_are_unchanged() {
        let mut fonts = fonts();
        let text = "日本語とEnglishの混在した文章";
        let mut rows_at = |width: f32| {
            let mut job = LayoutJob::single_section(text.into(), TextFormat::default());
            job.wrap.max_width = width;
            let galley = layout(&mut fonts, 1.0, Arc::new(job));
            galley.rows.iter().map(|r| r.text()).collect::<Vec<_>>()
        };
        assert_eq!(
            rows_at(90.0),
            ["日本語と", "Englishの混在", "した文章"],
            "90 pt"
        );
        assert_eq!(
            rows_at(110.0),
            ["日本語とEnglish", "の混在した文章"],
            "110 pt"
        );
    }
}
