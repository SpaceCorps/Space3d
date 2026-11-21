// Copies repo-owned sources (Claude Code integration, docs) into public/ so the site serves
// exactly what the repo ships, and builds llms-full.txt from docs/. Run before every build.
import { cpSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const pub = join(root, "site", "public");
mkdirSync(join(pub, "claude"), { recursive: true });
cpSync(join(root, "integrations/claude-code/AGENTS.template.md"), join(pub, "claude/AGENTS.md"));
cpSync(join(root, "integrations/claude-code/skills/space3d-game/SKILL.md"), join(pub, "claude/SKILL.md"));
cpSync(join(root, "integrations/claude-code/SETUP.md"), join(pub, "claude/SETUP.md"));

const docs = readdirSync(join(root, "docs")).filter((f) => f.endsWith(".md")).sort();
// The setup text first: an agent that reads only the start of the file learns how to begin.
const body = [readFileSync(join(root, "integrations/claude-code/SETUP.md"), "utf8"), readFileSync(join(root, "README.md"), "utf8")]
  .concat(docs.map((f) => `\n\n<!-- docs/${f} -->\n` + readFileSync(join(root, "docs", f), "utf8")))
  .join("\n");
writeFileSync(join(pub, "llms-full.txt"), body);
console.log(`sync: ${docs.length} docs -> llms-full.txt (${body.length} bytes)`);

// /play/<id>/index.html for every example in examples.json (the wasm is built by build-play.sh).
const examples = JSON.parse(readFileSync(join(root, "site", "examples.json"), "utf8"));
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
for (const e of examples) {
  const dir = e.path ?? e.id;
  mkdirSync(join(pub, "play", dir), { recursive: true });
  writeFileSync(join(pub, "play", dir, "index.html"), `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
<title>${esc(e.title)} — Space3d</title><link rel="icon" href="/favicon.svg">
<style>
@font-face{font-family:"Geist Variable";font-display:swap;font-weight:100 900;src:url("/fonts/geist-latin.woff2") format("woff2-variations")}
@font-face{font-family:"Geist Mono Variable";font-display:swap;font-weight:100 900;src:url("/fonts/geist-mono-latin.woff2") format("woff2-variations")}
html,body{margin:0;height:100%;background:#04070a;color:#fafafa;font:14px "Geist Variable",system-ui,sans-serif;overflow:hidden;overscroll-behavior:none}
/* The engine attaches to #space3d if present, else appends its own canvas to <body>: fill the viewport either way. */
canvas{position:fixed;inset:0;display:block;width:100vw!important;height:100vh!important;height:100dvh!important;touch-action:none;outline:none}
/* Site chrome lives in the bottom corners: the engine's own UI (F3 stats, panels, console) is top-left. */
#bar{z-index:10;position:fixed;right:12px;bottom:max(12px,env(safe-area-inset-bottom));display:flex;padding:4px;border-radius:99px;background:rgb(4 7 10 / .72);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid rgb(255 255 255 / .16);pointer-events:none;font-weight:600}
#bar a{pointer-events:auto;display:inline-flex;align-items:center;min-height:32px;padding:0 12px;border-radius:99px;background:#00cc92;color:#000;text-decoration:none;font-weight:600}
#bar a:hover{box-shadow:0 0 0 3px rgb(0 204 146 / .3)}
#bar a:focus-visible{outline:2px solid #fff;outline-offset:2px}
#status{z-index:10;position:fixed;left:12px;bottom:max(12px,env(safe-area-inset-bottom));max-width:calc(100vw - 175px);transition:opacity .6s;padding:8px 14px;border-radius:12px;background:rgb(4 7 10 / .72);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid rgb(255 255 255 / .16);color:#cfcfd6;font:12px/1.5 "Geist Mono Variable",ui-monospace,monospace;pointer-events:none}
#status.gone{opacity:0}
</style></head><body${e.chrome === "editor" ? ' class="editor"' : ""}>
<div id="bar"><a href="/examples.html" aria-label="Back to examples">← Examples</a></div>
<div id="status">loading… (first load downloads the engine)</div>
<script type="module">
import init from "./${e.id}.js";
const status = document.getElementById("status");
init().then(() => { status.textContent = ${JSON.stringify(e.controls)}; setTimeout(() => status.classList.add("gone"), 10000); })
  .catch((err) => { status.textContent = "failed to start: " + err + " (needs WebGPU or WebGL2)"; console.error(err); });
</script></body></html>
`);
}
