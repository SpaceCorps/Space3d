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

const docs = readdirSync(join(root, "docs")).filter((f) => f.endsWith(".md")).sort();
const body = [readFileSync(join(root, "README.md"), "utf8")]
  .concat(docs.map((f) => `\n\n<!-- docs/${f} -->\n` + readFileSync(join(root, "docs", f), "utf8")))
  .join("\n");
writeFileSync(join(pub, "llms-full.txt"), body);
console.log(`sync: ${docs.length} docs -> llms-full.txt (${body.length} bytes)`);

// /play/<id>/index.html for every example in examples.json (the wasm is built by build-play.sh).
const examples = JSON.parse(readFileSync(join(root, "site", "examples.json"), "utf8"));
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
for (const e of examples) {
  mkdirSync(join(pub, "play", e.id), { recursive: true });
  writeFileSync(join(pub, "play", e.id, "index.html"), `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
<title>${esc(e.title)} — Space3d</title><link rel="icon" href="/favicon.svg">
<style>
html,body{margin:0;height:100%;background:#05050a;color:#ccd;font:14px system-ui,sans-serif;overflow:hidden;overscroll-behavior:none}
canvas#space3d{display:block;width:100vw!important;height:100vh!important;height:100dvh!important;touch-action:none;outline:none}
#bar{position:fixed;left:0;right:0;top:0;display:flex;gap:12px;align-items:center;padding:8px 12px;background:linear-gradient(#000a,#0000);pointer-events:none}
#bar a{pointer-events:auto;color:#00cc92;text-decoration:none;font-weight:600}
#status{position:fixed;left:12px;right:12px;bottom:10px;opacity:.75;pointer-events:none}
</style></head><body>
<canvas id="space3d" tabindex="0"></canvas>
<div id="bar"><a href="/examples.html">← Examples</a><span>${esc(e.title)}</span></div>
<div id="status">loading… (first load downloads the engine)</div>
<script type="module">
import init from "./${e.id}.js";
const status = document.getElementById("status");
init().then(() => { status.textContent = ${JSON.stringify(e.controls)}; })
  .catch((err) => { status.textContent = "failed to start: " + err + " (needs WebGPU or WebGL2)"; console.error(err); });
</script></body></html>
`);
}
