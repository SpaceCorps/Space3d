/* Docs: client-side search over /docs-search.json, mobile drawer, TOC scroll-spy. */
const nav = document.getElementById("doc-nav");
const btn = document.querySelector(".doc-drawer-btn");
if (nav && btn) {
  const set = (open) => {
    nav.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", String(open));
  };
  btn.addEventListener("click", () => set(!nav.classList.contains("open")));
  nav.addEventListener("click", (e) => { if (e.target.closest("a")) set(false); });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && nav.classList.contains("open")) { set(false); btn.focus(); } });
}

const input = document.querySelector(".doc-search input");
const box = document.querySelector(".doc-results");
if (input && box) {
  let index = null;
  const load = () => (index ??= fetch("/docs-search.json").then((r) => r.json()).catch(() => []));
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const run = async () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { box.hidden = true; box.innerHTML = ""; return; }
    const words = q.split(/\s+/);
    const hits = [];
    for (const d of await load()) {
      let score = 0, section = null, snippet = "";
      const title = d.t.toLowerCase();
      if (words.every((w) => title.includes(w))) score += 10;
      for (const [id, h] of d.h) if (words.every((w) => h.toLowerCase().includes(w))) { score += 5; section ??= [id, h]; }
      const x = d.x.toLowerCase();
      const at = x.indexOf(words[0]);
      if (words.every((w) => x.includes(w))) { score += 1; if (at >= 0) snippet = d.x.slice(Math.max(0, at - 40), at + 100); }
      if (score) hits.push({ d, score, section, snippet });
    }
    hits.sort((a, b) => b.score - a.score);
    box.hidden = false;
    box.innerHTML = hits.length
      ? hits.slice(0, 8).map(({ d, section, snippet }) =>
          `<a role="option" href="/docs/${d.s}/${section ? "#" + section[0] : ""}"><strong>${esc(d.t)}</strong>${section ? `<em>${esc(section[1])}</em>` : ""}${snippet ? `<small>…${esc(snippet)}…</small>` : ""}</a>`).join("")
      : `<p>No results for “${esc(input.value.trim())}”.</p>`;
  };
  input.addEventListener("input", run);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { input.value = ""; run(); }
    if (e.key === "Enter") box.querySelector("a")?.click();
  });
  addEventListener("keydown", (e) => {
    if (e.key === "/" && !/INPUT|TEXTAREA/.test(document.activeElement?.tagName ?? "")) { e.preventDefault(); nav?.classList.add("open"); input.focus(); }
  });
}

const links = [...document.querySelectorAll(".doc-toc a")];
if (links.length && "IntersectionObserver" in window) {
  const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) {
      links.forEach((a) => a.removeAttribute("aria-current"));
      byId.get(e.target.id)?.setAttribute("aria-current", "true");
    }
  }, { rootMargin: "-80px 0px -70% 0px" });
  for (const id of byId.keys()) { const h = document.getElementById(id); if (h) io.observe(h); }
}
