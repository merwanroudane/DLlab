"""Functional step-animation player (spec §19, §78).

Python builds a list of `Frame`s (a visual + a synchronized Arabic caption +
the current action + old/new values). The browser-side player provides
Play · Pause · Previous · Next · Restart · 0.5×/1×/2× · step counter without
Streamlit reruns; it reports the current step back (`setStateValue`) only on
pause / manual stepping / end of playback, so Python can show extra detail
(e.g. `st.latex`) for the step the learner is looking at.

Honours `prefers-reduced-motion` (autoplay disabled, stepping still works).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from core.rtl import esc


@dataclass
class Frame:
    html: str                                   # LTR visual (SVG/HTML), trusted content
    caption_ar: str                             # synchronized explanation (RTL, plain text or <b>/<code>)
    action: str = ""                            # current action label, e.g. "Forward pass"
    values: list[tuple[str, str, str]] = field(default_factory=list)   # (name, old, new)
    equation: str = ""                          # plain-text/unicode equation shown inside the player
    highlight: int | None = None                # optional pipeline stage index (if `stages` given)


_HTML = """
<div class="ap" dir="ltr">
  <div class="ap-head">
    <span class="ap-title"></span>
    <span class="ap-counter"></span>
  </div>
  <div class="ap-pipe"></div>
  <div class="ap-stage"></div>
  <div class="ap-action"></div>
  <div class="ap-caption" dir="rtl"></div>
  <div class="ap-eq"></div>
  <div class="ap-values"></div>
  <div class="ap-controls">
    <button class="ap-btn" data-act="restart" title="Restart">⟲</button>
    <button class="ap-btn" data-act="prev" title="Previous">⏮</button>
    <button class="ap-btn ap-play" data-act="play" title="Play / Pause">▶</button>
    <button class="ap-btn" data-act="next" title="Next">⏭</button>
    <span class="ap-speed">
      <button class="ap-sp" data-speed="0.5">0.5×</button>
      <button class="ap-sp ap-sp-on" data-speed="1">1×</button>
      <button class="ap-sp" data-speed="2">2×</button>
    </span>
    <input class="ap-range" type="range" min="0" max="0" value="0" aria-label="step">
  </div>
</div>
"""

_CSS = """
.ap { font-family: var(--st-font, 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif); color: var(--st-text-color, #2B2A28);
      border: 1px solid var(--st-border-color, #EADFCD); border-radius: 14px; background: #fff; padding: .8rem 1rem;
      box-shadow: 0 1px 2px rgba(60,40,10,.06), 0 4px 14px rgba(60,40,10,.05); }
.ap-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:.4rem; }
.ap-title { font-weight:700; direction:rtl; }
.ap-counter { font-family: 'JetBrains Mono', Consolas, monospace; font-size:.85rem; color:#6B675F; }
.ap-pipe { display:flex; flex-wrap:wrap; gap:.3rem; align-items:center; margin:.2rem 0 .5rem; }
.ap-pipe .s { padding:.2rem .6rem; border-radius:8px; border:1px solid #EADFCD; background:#fff; font-size:.82rem; font-weight:500; }
.ap-pipe .s.on { background:#1F7A78; color:#fff; border-color:#1F7A78; }
.ap-pipe .a { color:#B9B2A6; font-size:.8rem; }
.ap-stage { min-height: 60px; text-align:center; overflow-x:auto; }
.ap-stage svg { max-width:100%; height:auto; }
.ap-action { display:inline-block; margin:.5rem 0 .2rem; padding:.15rem .6rem; border-radius:999px; background:#E0F3F1; color:#1F7A78; font-weight:600; font-size:.85rem; }
.ap-action:empty { display:none; }
.ap-caption { text-align:right; line-height:1.8; margin:.3rem 0; }
.ap-caption code { direction:ltr; unicode-bidi:isolate; display:inline-block; background:#F6EEDF; border-radius:6px; padding:0 .35em; font-family:'JetBrains Mono', Consolas, monospace; font-size:.9em; }
.ap-eq { font-family:'JetBrains Mono', Consolas, monospace; background:#FBF4E8; border-radius:8px; padding:.35rem .6rem; margin:.3rem 0; font-size:.9rem; }
.ap-eq:empty { display:none; }
.ap-values table { border-collapse:collapse; font-size:.85rem; margin:.3rem 0; }
.ap-values td, .ap-values th { padding:.2rem .6rem; border-bottom:1px solid #EADFCD; text-align:left; font-family:'JetBrains Mono', Consolas, monospace; }
.ap-values th { background:#F6EEDF; font-family:inherit; }
.ap-values .new { color:#1F7A78; font-weight:700; }
.ap-controls { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin-top:.6rem; }
.ap-btn { border:1px solid #EADFCD; background:#FBF4E8; border-radius:8px; padding:.25rem .6rem; cursor:pointer; font-size:1rem; }
.ap-btn:hover { background:#F6EEDF; }
.ap-play { background:#1F7A78; color:#fff; border-color:#1F7A78; min-width:2.6rem; }
.ap-speed { display:inline-flex; gap:.2rem; margin-left:.4rem; }
.ap-sp { border:1px solid #EADFCD; background:#fff; border-radius:6px; padding:.15rem .45rem; cursor:pointer; font-size:.8rem; }
.ap-sp-on { background:#E0F3F1; border-color:#1F7A78; color:#1F7A78; font-weight:600; }
.ap-range { flex:1 1 140px; accent-color:#1F7A78; }
/* multicolour layer */
.ap { border:0; position:relative; overflow:hidden; padding-top:1.1rem;
      box-shadow: 0 2px 6px rgba(76,29,149,.08), 0 8px 26px rgba(37,99,235,.10); }
.ap::before { content:""; position:absolute; inset:0 0 auto 0; height:5px;
      background:linear-gradient(90deg,#7C3AED,#2563EB,#0891B2,#059669,#D97706,#EA580C,#DB2777); }
.ap-title { color:#5B21B6; }
.ap-counter { color:#DB2777; font-weight:700; }
.ap-pipe .s { font-weight:600; border-width:1.5px; transition: transform .25s, box-shadow .25s; }
.ap-pipe .s:nth-child(14n+1)  { color:#7C3AED; background:#F3EEFF; border-color:#C4B5FD; }
.ap-pipe .s:nth-child(14n+3)  { color:#2563EB; background:#EAF1FF; border-color:#BFD3FE; }
.ap-pipe .s:nth-child(14n+5)  { color:#0891B2; background:#E4F7FB; border-color:#A5E3F0; }
.ap-pipe .s:nth-child(14n+7)  { color:#059669; background:#E3F8EF; border-color:#A7E9CC; }
.ap-pipe .s:nth-child(14n+9)  { color:#D97706; background:#FFF4DE; border-color:#FCD9A0; }
.ap-pipe .s:nth-child(14n+11) { color:#EA580C; background:#FFEDE3; border-color:#FDC4A6; }
.ap-pipe .s:nth-child(14n+13) { color:#DB2777; background:#FFE8F3; border-color:#F9B4D5; }
.ap-pipe .s.on { color:#fff !important; border-color:transparent !important; transform: translateY(-2px) scale(1.06);
      background:linear-gradient(135deg,#7C3AED,#DB2777) !important; box-shadow:0 4px 12px rgba(219,39,119,.35); }
.ap-pipe .a { color:#A78BFA; font-weight:700; }
.ap-stage { border-radius:12px; background:linear-gradient(180deg,#FFFFFF,#FAF8FF); padding:.3rem; }
.ap-stage.ap-anim { animation: apFade .35s ease; }
@keyframes apFade { from { opacity:.35; transform: translateY(4px); } to { opacity:1; transform:none; } }
.ap-action { background:linear-gradient(135deg,#E0F2FE,#F3EEFF); color:#4C1D95; border:1px solid #C4B5FD; }
.ap-caption { background:#FFFBF2; border-right:4px solid #F59E0B; border-radius:8px; padding:.45rem .7rem; }
.ap-caption b { color:#1E1B4B; }
.ap-caption code { background:#F3EEFF; color:#5B21B6; }
.ap-eq { background:linear-gradient(90deg,#EAF1FF,#E4F7FB); color:#1E3A8A; border-right:4px solid #2563EB; font-weight:600; }
.ap-values th { background:linear-gradient(90deg,#F3EEFF,#EAF1FF); color:#3B1E8A; }
.ap-values td:first-child { color:#7C3AED; font-weight:700; }
.ap-values .new { color:#DB2777; }
.ap-btn { background:#F3EEFF; border-color:#D9C8FF; color:#5B21B6; }
.ap-btn:hover { background:#E9DEFF; }
.ap-play { background:linear-gradient(135deg,#7C3AED,#2563EB); color:#fff; border:0; box-shadow:0 3px 10px rgba(124,58,237,.35); }
.ap-sp-on { background:#FFE8F3; border-color:#DB2777; color:#DB2777; }
.ap-range { accent-color:#DB2777; }
@media (prefers-reduced-motion: reduce) { .ap * { transition:none !important; animation:none !important; } }
"""

_JS = """
const registry = new WeakMap();

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const root = parentElement.querySelector(".ap");
  if (!root) return;
  const frames = data.frames || [];
  const stages = data.stages || [];
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let st = registry.get(parentElement);
  if (!st) {
    st = { i: 0, playing: false, speed: 1, timer: null, sig: "" };
    registry.set(parentElement, st);
    wire();
  }
  // New frames from Python (different signature) -> reset position.
  const sig = data.sig || String(frames.length);
  if (st.sig !== sig) { st.sig = sig; st.i = Math.min(data.start || 0, Math.max(0, frames.length - 1)); stop(false); root.querySelector(".ap-stage").dataset.i = ""; }

  const q = (s) => root.querySelector(s);
  q(".ap-title").textContent = data.title || "";
  const range = q(".ap-range");
  range.max = Math.max(0, frames.length - 1);

  function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

  function render() {
    const f = frames[st.i] || { html: "", caption: "", action: "", values: [], equation: "", highlight: null };
    q(".ap-counter").textContent = (st.i + 1) + " / " + frames.length;
    const stg = q(".ap-stage");
    if (stg.dataset.i !== String(st.i)) {           // new frame -> replay the fade-in
      stg.dataset.i = String(st.i);
      stg.innerHTML = f.html || "";
      stg.classList.remove("ap-anim"); void stg.offsetWidth; stg.classList.add("ap-anim");
    }
    q(".ap-action").textContent = f.action || "";
    q(".ap-caption").innerHTML = f.caption || "";
    q(".ap-eq").textContent = f.equation || "";
    const v = f.values || [];
    q(".ap-values").innerHTML = v.length
      ? "<table><tr><th>value</th><th>before</th><th>after</th></tr>" +
        v.map(r => "<tr><td>" + esc(r[0]) + "</td><td>" + esc(r[1]) + "</td><td class='new'>" + esc(r[2]) + "</td></tr>").join("") + "</table>"
      : "";
    q(".ap-pipe").innerHTML = stages.length
      ? stages.map((s, k) => "<span class='s" + (k === f.highlight ? " on" : "") + "'>" + esc(s) + "</span>" + (k < stages.length - 1 ? "<span class='a'>→</span>" : "")).join("")
      : "";
    range.value = st.i;
    q(".ap-play").textContent = st.playing ? "⏸" : "▶";
    root.querySelectorAll(".ap-sp").forEach(b => b.classList.toggle("ap-sp-on", Number(b.dataset.speed) === st.speed));
  }

  function report() { setStateValue("step", st.i); }

  function stop(doReport) {
    st.playing = false;
    if (st.timer) { clearInterval(st.timer); st.timer = null; }
    if (doReport) report();
  }

  function tick() {
    if (st.i >= frames.length - 1) { stop(true); render(); return; }
    st.i += 1; render();
  }

  function play() {
    if (reduced || frames.length < 2) return;
    if (st.i >= frames.length - 1) st.i = 0;
    st.playing = true;
    const ms = Math.max(120, (data.interval_ms || 1200) / st.speed);
    st.timer = setInterval(tick, ms);
    render();
  }

  function wire() {
    root.querySelectorAll(".ap-btn").forEach(b => b.addEventListener("click", () => {
      const act = b.dataset.act;
      if (act === "play") { st.playing ? stop(true) : play(); }
      else if (act === "next") { stop(false); st.i = Math.min(frames.length - 1, st.i + 1); report(); }
      else if (act === "prev") { stop(false); st.i = Math.max(0, st.i - 1); report(); }
      else if (act === "restart") { stop(false); st.i = 0; report(); }
      render();
    }));
    root.querySelectorAll(".ap-sp").forEach(b => b.addEventListener("click", () => {
      st.speed = Number(b.dataset.speed);
      if (st.playing) { stop(false); play(); } else render();
    }));
    root.querySelector(".ap-range").addEventListener("input", (e) => {
      stop(false); st.i = Number(e.target.value); render();
    });
    root.querySelector(".ap-range").addEventListener("change", () => report());
  }

  render();
  return () => { if (st.timer) clearInterval(st.timer); };
}
"""

_NAME = "dlia_animation_player"


def _register():
    return st.components.v2.component(_NAME, html=_HTML, css=_CSS, js=_JS)


_PLAYER = _register()


def _mount(**kwargs):
    """Mount the player; re-register once if the current runtime's registry
    does not know the component (happens when a fresh runtime — e.g. a new
    `AppTest` — starts after this module was imported)."""
    global _PLAYER
    from streamlit.errors import StreamlitAPIException

    try:
        return _PLAYER(**kwargs)
    except StreamlitAPIException as exc:
        if "not registered" not in str(exc):
            raise
        _PLAYER = _register()
        return _PLAYER(**kwargs)


def animation_player(
    key: str,
    frames: list[Frame],
    *,
    title_ar: str = "",
    stages: list[str] | None = None,
    interval_ms: int = 1200,
    start: int = 0,
) -> int:
    """Mount the player and return the step index the learner last stopped on."""
    payload = [
        {
            "html": f.html,
            "caption": f.caption_ar,
            "action": f.action,
            "values": [list(v) for v in f.values],
            "equation": f.equation,
            "highlight": f.highlight,
        }
        for f in frames
    ]
    sig = f"{len(frames)}:{hash(tuple(f.caption_ar for f in frames)) & 0xFFFFFFFF}"
    result = _mount(
        key=key,
        data={
            "frames": payload,
            "stages": stages or [],
            "title": title_ar,
            "interval_ms": interval_ms,
            "start": start,
            "sig": sig,
        },
        on_step_change=lambda: None,
    )
    step = getattr(result, "step", None)
    if step is None:
        step = start
    return max(0, min(int(step), max(0, len(frames) - 1)))


def caption(text_ar: str) -> str:
    """Escape a caption and turn `code` spans into <code> (mirrors callouts)."""
    import re

    t = esc(text_ar)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    return t
