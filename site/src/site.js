const root = document.documentElement;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Theme: dark by default (the inline head script applies a remembered choice first). */
for (const b of document.querySelectorAll(".theme-btn")) {
  b.setAttribute("aria-pressed", String(root.classList.contains("dark")));
  b.addEventListener("click", () => {
    const dark = !root.classList.contains("dark");
    root.classList.toggle("dark", dark);
    try { localStorage.setItem("s3d-theme", dark ? "dark" : "light"); } catch {}
    b.setAttribute("aria-pressed", String(dark));
  });
}

/* Mobile menu: close on link click or Escape. */
const menu = document.querySelector("details.menu");
if (menu) {
  menu.addEventListener("click", (e) => { if (e.target.closest("a")) menu.open = false; });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && menu.open) { menu.open = false; menu.querySelector("summary").focus(); } });
}

/* Copy buttons on every code block (and [data-copy] elements). */
const copyIcon = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
function attach(btn, getText) {
  btn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(getText()); } catch { return; }
    btn.dataset.done = "1";
    btn.innerHTML = "Copied";
    setTimeout(() => { delete btn.dataset.done; btn.innerHTML = copyIcon + "Copy"; }, 1600);
  });
}
for (const pre of document.querySelectorAll("pre")) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "copy"; b.innerHTML = copyIcon + "Copy";
  b.setAttribute("aria-label", "Copy code to clipboard");
  pre.append(b);
  attach(b, () => (pre.querySelector("code") ?? pre).innerText.replace(/\s+#.*$/gm, "").trim());
}
for (const el of document.querySelectorAll("[data-copy]")) {
  const b = el.querySelector(".copy");
  if (b) b.innerHTML = copyIcon + "Copy";
  if (b) attach(b, () => el.dataset.copy);
}

/* Scroll reveal. Content is visible without JS; reduced motion shows everything immediately. */
const rv = [...document.querySelectorAll(".rv")];
if (reduce || !("IntersectionObserver" in window)) rv.forEach((e) => e.classList.add("in"));
else {
  const io = new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }, { rootMargin: "0px 0px -8% 0px" });
  rv.forEach((e, n) => { e.style.setProperty("--i", String(n % 4)); io.observe(e); });
}

/* Starfield: ~150 points on one canvas. Static under reduced motion, paused off-screen or hidden. */
const cv = document.getElementById("stars");
if (cv) {
  const ctx = cv.getContext("2d");
  let w = 0, h = 0, stars = [], raf = 0, visible = true, t0 = performance.now();
  const resize = () => {
    const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    w = r.width; h = r.height; cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    stars = Array.from({ length: Math.round(Math.min(160, (w * h) / 5500)) }, () => ({ x: Math.random(), y: Math.random(), z: Math.random(), p: Math.random() * 6.28 }));
    draw(0);
  };
  const draw = (t) => {
    ctx.clearRect(0, 0, w, h);
    for (const s of stars) {
      const tw = reduce ? 0.7 : 0.55 + 0.45 * Math.sin(t / 1400 + s.p);
      const x = (s.x * w + (reduce ? 0 : (t / 1000) * (2 + s.z * 8))) % w;
      ctx.globalAlpha = (0.25 + 0.6 * s.z) * tw;
      ctx.fillStyle = s.z > 0.86 ? "#9ad9ff" : s.z > 0.7 ? "#7bf0cf" : "#ffffff";
      ctx.fillRect(x, s.y * h, 0.6 + s.z * 1.5, 0.6 + s.z * 1.5);
    }
  };
  const loop = (now) => { draw(now - t0); raf = visible && !document.hidden ? requestAnimationFrame(loop) : 0; };
  const kick = () => { if (!reduce && !raf && visible && !document.hidden) raf = requestAnimationFrame(loop); };
  new ResizeObserver(resize).observe(cv);
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; kick(); }).observe(cv);
  document.addEventListener("visibilitychange", kick);
  kick();
}
