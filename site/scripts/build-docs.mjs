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
  .replace("</head>", '<link rel="stylesheet" href="/src/docs.css"></head>');
const bottom = chromeSrc.slice(footStart);

const titles = {};
const parsed = files.map((f) => {
  const md = readFileSync(join(docsDir, f), "utf8").replace(/\r\n/g, "\n");
  const title = (md.match(/^#\s+(.+)$/m)?.[1] ?? slugOf(f)).replace(/[`*]/g, "");
  titles[f] = title;
  return { f, md, title };
});

function fixLink(href, from) {
  if (/^(https?:|mailto:|#)/.test(href)) return href;
  const [path, hash = ""] = href.split("#");
  const frag = hash ? "#" + hash : "";
  const target = posix.normalize(posix.join(posix.dirname(from), path)); // relative to docs/
  if (target.startsWith("..")) {
    return `${GH}/blob/main/${posix.normalize(posix.join("docs", posix.dirname(from), path))}${frag}`;
  }
  if (target.endsWith(".md") && files.includes(target)) return `/docs/${slugOf(target)}/${frag}`;
  return `${GH}/blob/main/docs/${target}${frag}`;
}

const groups = [
  ["Start", ["getting-started.md", "platforms.md", "testing.md", "storybook.md"]],
  ["Engine", ["graphics.md", "ui.md", "networking.md", "server.md", "performance.md"]],
  ["Graphics reference", files.filter((f) => f.startsWith("graphics/"))],
  ["Design documents", files.filter((f) => f.startsWith("design/"))],
];
const known = new Set(groups.flatMap(([, l]) => l));
const rest = files.filter((f) => !known.has(f));
if (rest.length) groups.push(["More", rest]);
const sidebar = (cur) =>
  `<nav class="doc-nav" aria-label="Docs">${groups
    .filter(([, l]) => l.length)
    .map(
      ([g, l]) =>
        `<h4>${g}</h4><ul>${l
          .filter((f) => titles[f])
          .map((f) => `<li><a href="/docs/${slugOf(f)}/"${f === cur ? ' aria-current="page"' : ""}>${esc(titles[f])}</a></li>`)
          .join("")}</ul>`,
    )
    .join("")}</nav>`;

rmSync(outDir, { recursive: true, force: true });
for (const { f, md, title } of parsed) {
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
        return `<img src="${esc(src)}" alt="${esc(text)}" loading="lazy">`;
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
  const tocHtml =
    toc.length > 2
      ? `<aside class="doc-toc" aria-label="On this page"><h4>On this page</h4><ul>${toc.map(([id, t]) => `<li><a href="#${id}">${esc(t)}</a></li>`).join("")}</ul></aside>`
      : "";
  const page = `${top.replace("@TITLE@", () => esc(title) + " — Space3d docs").replace("@DESC@", () => esc(desc))}<main id="main" class="wrap doc-layout">
${sidebar(f)}
<article class="doc">${body}<p class="doc-edit"><a href="${GH}/edit/main/docs/${f}">Edit this page on GitHub</a></p></article>
${tocHtml}
</main>
${bottom}`;
  const out = join(outDir, slugOf(f), "index.html");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, page);
}
writeFileSync(join(outDir, "index.json"), JSON.stringify(parsed.map(({ f, title }) => ({ slug: slugOf(f), title })), null, 1));
console.log(`docs: ${files.length} pages -> site/docs/`);
