// Renders ../docs/**/*.md to site/docs/<slug>/index.html (vite inputs, see vite.config.ts) using the
// header/footer/head of site/docs.html, so the docs share the site's chrome whatever it looks like.
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, posix, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import ini from "highlight.js/lib/languages/ini";
import json from "highlight.js/lib/languages/json";
import rust from "highlight.js/lib/languages/rust";
import { Marked } from "marked";

hljs.registerLanguage("rust", rust);
hljs.registerLanguage("sh", bash);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("json", json);
hljs.registerLanguage("toml", ini);
hljs.registerLanguage("wgsl", rust);
hljs.registerLanguage("ron", rust);

const site = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(site, "..");
const docsDir = join(root, "docs");
const outDir = join(site, "docs");
const GH = "https://github.com/SpaceCorps/Space3d";
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const walk = (d) =>
  readdirSync(d).flatMap((f) => {
    const p = join(d, f);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith(".md") ? [p] : [];
  });
const files = walk(docsDir)
  .map((p) => relative(docsDir, p).replaceAll("\\", "/"))
  .sort();
const slugOf = (f) => f.replace(/\.md$/, "");

// Chrome (head, header, footer) from the docs index page.
const chromeSrc = readFileSync(join(site, "docs.html"), "utf8").replace(/\r\n/g, "\n");
const headEnd = chromeSrc.indexOf("<main");
const footStart = chromeSrc.indexOf("<footer");
if (headEnd < 0 || footStart < 0) throw new Error("docs.html needs <main> and <footer>");
const top = chromeSrc
  .slice(0, headEnd)
  .replace(/<title>.*?<\/title>/s, "<title>@TITLE@</title>")
  .replace(/<meta name="description"[^>]*>/, '<meta name="description" content="@DESC@">')
  .replace("</head>", '<link rel="stylesheet" href="/src/docs.css"><script type="module" src="/src/docs.js"></script></head>');
const bottom = chromeSrc.slice(footStart);

const titles = {};
const parsed = files.map((f) => {
  const md = readFileSync(join(docsDir, f), "utf8").replace(/\r\n/g, "\n");
  const title = (md.match(/^#\s+(.+)$/m)?.[1] ?? slugOf(f)).replace(/[`*]/g, "");
  titles[f] = title;
  return { f, md, title };
});

// The nine top-level guides are ONE page (/docs/), a section each; the reference pages stay separate.
const MAIN = ["getting-started.md", "platforms.md", "graphics.md", "networking.md", "server.md", "testing.md", "ui.md", "storybook.md", "performance.md"].filter((f) => files.includes(f));
const isMain = (f) => MAIN.includes(f);
const urlOf = (f) => (isMain(f) ? `/docs/#${slugOf(f)}` : `/docs/${slugOf(f)}/`);

function fixLink(href, from) {
  if (/^(https?:|mailto:)/.test(href)) return href;
  if (href.startsWith("#")) return isMain(from) ? `#${slugOf(from)}--${href.slice(1)}` : href;
  const [path, hash = ""] = href.split("#");
  const frag = hash ? "#" + hash : "";
  const target = posix.normalize(posix.join(posix.dirname(from), path)); // relative to docs/
  if (target.startsWith("..")) {
    return `${GH}/blob/main/${posix.normalize(posix.join("docs", posix.dirname(from), path))}${frag}`;
  }
  if (isMain(target)) return `/docs/#${slugOf(target)}${hash ? "--" + hash : ""}`;
  if (target.endsWith(".md") && files.includes(target)) return `/docs/${slugOf(target)}/${frag}`;
  return `${GH}/blob/main/docs/${target}${frag}`;
}

const groups = [
  ["Guides (one page)", MAIN],
  ["Graphics reference", files.filter((f) => f.startsWith("graphics/"))],
  ["Design documents", files.filter((f) => f.startsWith("design/"))],
];
const known = new Set(groups.flatMap(([, l]) => l));
const rest = files.filter((f) => !known.has(f));
if (rest.length) groups.push(["More", rest]);
const order = groups.flatMap(([, l]) => l).filter((f) => titles[f]);
const sidebar = (cur) =>
  `<button class="doc-drawer-btn" type="button" aria-expanded="false" aria-controls="doc-nav">Docs menu</button>
<nav class="doc-nav" id="doc-nav" aria-label="Docs"><label class="doc-search"><span class="sr">Search the docs</span><input type="search" placeholder="Search the docs" autocomplete="off" spellcheck="false"></label><div class="doc-results" role="listbox" hidden></div>${groups
    .filter(([, l]) => l.length)
    .map(
      ([g, l]) =>
        `<h4>${g}</h4><ul>${l
          .filter((f) => titles[f])
          .map((f) => `<li><a href="${isMain(f) && cur === "all" ? "#" + slugOf(f) : urlOf(f)}"${f === cur ? ' aria-current="page"' : ""}>${esc(titles[f])}</a></li>`)
          .join("")}</ul>`,
    )
    .join("")}</nav>`;

rmSync(outDir, { recursive: true, force: true });
const searchIndex = [];
const sections = [];
const write = (rel, html) => {
  const out = join(outDir, rel, "index.html");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, html);
};
const wrap = (title, desc, inner) =>
  `${top.replace("@TITLE@", () => esc(title)).replace("@DESC@", () => esc(desc))}<main id="main" class="wrap doc-layout">\n${inner}\n</main>\n${bottom}`;

for (const { f, md, title } of parsed) {
  const main = isMain(f);
  const slug = slugOf(f);
  const toc = [];
  const used = new Set();
  const marked = new Marked({
    gfm: true,
    renderer: {
      heading({ tokens, depth, text }) {
        const inner = this.parser.parseInline(tokens);
        const base =
          text
            .toLowerCase()
            .replace(/<[^>]+>/g, "")
            .replace(/[^\p{L}\p{N}\s-]/gu, "")
            .trim()
            .replace(/\s+/g, "-") || "section";
        let id = base,
          i = 1;
        while (used.has(id)) id = `${base}-${i++}`;
        used.add(id);
        if (main) {
          // One page: ids are namespaced by section, the section's # title becomes the section heading.
          const full = depth === 1 ? slug : `${slug}--${id}`;
          if (depth === 2) toc.push([full, text.replace(/[`*]/g, "")]);
          const d = Math.min(depth + 1, 6);
          return `<h${d} id="${full}"${depth === 1 ? ' class="sec-title"' : ""}><a class="anchor" href="#${full}" aria-label="Link to this section">#</a>${inner}</h${d}>\n`;
        }
        if (depth === 2) toc.push([id, text.replace(/[`*]/g, "")]);
        return `<h${depth} id="${id}"><a class="anchor" href="#${id}" aria-label="Link to this section">#</a>${inner}</h${depth}>\n`;
      },
      link({ href, title: t, tokens }) {
        const inner = this.parser.parseInline(tokens);
        const h = fixLink(href, f);
        return `<a href="${esc(h)}"${t ? ` title="${esc(t)}"` : ""}${h.startsWith("http") ? ' rel="noopener"' : ""}>${inner}</a>`;
      },
      image({ href, text }) {
        const p = posix.normalize(posix.join("docs", posix.dirname(f), href));
        const src = /^https?:/.test(href) ? href : `https://raw.githubusercontent.com/SpaceCorps/Space3d/main/${p}`;
        return `<img src="${esc(src)}" alt="${esc(text)}"${main ? "" : ' loading="lazy"'}>`;
      },
      code({ text, lang }) {
        const l = (lang ?? "").split(/\s/)[0];
        const body = l && hljs.getLanguage(l) ? hljs.highlight(text, { language: l }).value : esc(text);
        return `<pre><code class="hljs${l ? " language-" + l : ""}">${body}</code></pre>\n`;
      },
      table({ header, rows }) {
        const cell = (c, tag) =>
          `<${tag}${c.align ? ` style="text-align:${c.align}"` : ""}>${this.parser.parseInline(c.tokens)}</${tag}>`;
        const head = header.map((c) => cell(c, "th")).join("");
        const body = rows.map((r) => `<tr>${r.map((c) => cell(c, "td")).join("")}</tr>`).join("");
        return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>\n`;
      },
    },
  });
  const body = marked.parse(md);
  const desc = (md.match(/^(?!#|\s|[-*>|`])(.{40,})$/m)?.[1] ?? title).replace(/[`*[\]]/g, "").slice(0, 200);
  const plain = body.replace(/<pre>[\s\S]*?<\/pre>/g, " ").replace(/<[^>]+>/g, " ").replace(/&[a-z#0-9]+;/g, " ").replace(/\s+/g, " ");
  searchIndex.push({ s: slug, u: main ? "/docs/" : `/docs/${slug}/`, t: title, h: toc, x: plain.slice(0, 6000) });
  if (main) {
    sections.push({ f, slug, title, body, toc });
    // The old per-section URL keeps working: it forwards to the section on the one page.
    write(
      slug,
      `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${esc(title)} — Space3d docs</title><link rel="canonical" href="https://space3d.dev/docs/#${slug}"><meta http-equiv="refresh" content="0; url=/docs/#${slug}"><script>location.replace("/docs/#${slug}")</script></head><body><p>Moved to <a href="/docs/#${slug}">${esc(title)}</a>.</p></body></html>`,
    );
    continue;
  }
  const k = order.indexOf(f);
  const pn = (g, lab) => (g ? `<a class="${lab === "Previous" ? "prev" : "next"}" href="${urlOf(g)}" rel="${lab === "Previous" ? "prev" : "next"}"><small>${lab}</small><span>${esc(titles[g])}</span></a>` : "<span></span>");
  const pager = `<nav class="doc-pager" aria-label="Previous and next">${pn(order[k - 1], "Previous")}${pn(order[k + 1], "Next")}</nav>`;
  const tocHtml =
    toc.length > 2
      ? `<aside class="doc-toc" aria-label="On this page"><h4>On this page</h4><ul>${toc.map(([id, t]) => `<li><a href="#${id}">${esc(t)}</a></li>`).join("")}</ul></aside>`
      : "";
  write(
    slug,
    wrap(`${title} — Space3d docs`, desc, `${sidebar(f)}\n<article class="doc">${body}<p class="doc-edit"><a href="${GH}/edit/main/docs/${f}">Edit this page on GitHub</a></p>${pager}</article>\n${tocHtml}`),
  );
}

// The one page: all nine guides, sidebar of sections, right-hand outline of the section in view.
const art = sections
  .map((x) => `<section class="doc-sec" id="sec-${x.slug}" data-sec="${x.slug}">${x.body}<p class="doc-edit"><a href="${GH}/edit/main/docs/${x.f}">Edit this section on GitHub</a></p></section>`)
  .join("\n");
const outline = sections
  .map((x) => `<ul data-for="${x.slug}" hidden>${x.toc.map(([id, t]) => `<li><a href="#${id}">${esc(t)}</a></li>`).join("")}</ul>`)
  .join("");
write(
  "",
  wrap(
    "Space3d docs",
    "Everything about building with Space3d on one page: getting started, platforms, graphics, networking, server, testing, UI, storybook and performance.",
    `${sidebar("all")}\n<article class="doc doc-all">${art}</article>\n<aside class="doc-toc" aria-label="On this page"><h4>On this page</h4>${outline}</aside>`,
  ),
);
mkdirSync(join(site, "public"), { recursive: true });
writeFileSync(join(site, "public", "docs-search.json"), JSON.stringify(searchIndex));
writeFileSync(join(outDir, "index.json"), JSON.stringify(parsed.map(({ f, title }) => ({ slug: slugOf(f), title })), null, 1));
console.log(`docs: ${files.length} pages (${sections.length} guides on one page) -> site/docs/`);
