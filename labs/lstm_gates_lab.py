"""LSTM Gates Lab — gate-by-gate animation of one LSTM cell over a short
sequence: forget, input, candidate, cell update, output (spec §41)."""

import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note
from components.diagram import svg_arrow, svg_box, svg_defs, svg_text
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.rnn import lstm_forward, random_weights

LAB = Lab(
    id="labs.lstm_gates_lab",
    title_ar="معمل بوابات LSTM",
    title_en="LSTM Gates Lab",
    category="sequence",
    description_ar="خلية LSTM بوابةً بوابة على تسلسل قصير: بوابة النسيان، بوابة الإدخال والمرشَّح، تحديث حالة الخلية، بوابة الإخراج — بقيم حقيقية عند كل خطوة زمنية.",
    related_lessons=["course.w11.cell_gates_equations", "course.w11.train_eval_diagnose"],
)

STAGES = ["forget f", "input i · candidate g", "cell c", "output o · hidden h"]


def _cell_svg(stp: dict, stage: int) -> str:
    def v(arr):
        return "[" + ", ".join(f"{a:+.2f}" for a in arr[:3]) + ("…" if len(arr) > 3 else "") + "]"
    on = lambda k: "#C8473A" if k == stage else "#B9B2A6"
    fill = lambda k: "#FBE6E2" if k == stage else "#FFFDF9"
    s = ['<svg viewBox="0 0 760 300" width="100%" style="max-width:760px">', svg_defs()]
    s.append(svg_text(80, 30, f"x_t = {stp['x'][0]:+.2f}", size=12, mono=True, color="#2F6FB5"))
    s.append(svg_text(80, 55, f"h_prev = {v(stp['h_prev'])}", size=11, mono=True, color="#6B675F"))
    s.append(svg_text(600, 30, f"c_prev = {v(stp['c_prev'])}", size=11, mono=True, color="#7C5CBF"))
    s.append(svg_box(40, 90, 150, 46, "f = σ([x,h]·Wf + bf)", fill(0), stroke=on(0), font=11)); s.append(svg_text(115, 152, f"f = {v(stp['f'])}", size=11, mono=True, color=on(0)))
    s.append(svg_box(220, 90, 150, 46, "i = σ(…Wi)  g = tanh(…Wc)", fill(1), stroke=on(1), font=10)); s.append(svg_text(295, 152, f"i = {v(stp['i'])}", size=10, mono=True, color=on(1))); s.append(svg_text(295, 168, f"g = {v(stp['g'])}", size=10, mono=True, color=on(1)))
    s.append(svg_box(400, 90, 150, 46, "c = f ⊙ c_prev + i ⊙ g", fill(2), stroke=on(2), font=11)); s.append(svg_text(475, 152, f"c = {v(stp['c'])}", size=11, mono=True, color=on(2)))
    s.append(svg_box(580, 90, 160, 46, "o = σ(…Wo)  h = o ⊙ tanh(c)", fill(3), stroke=on(3), font=10)); s.append(svg_text(660, 152, f"o = {v(stp['o'])}", size=10, mono=True, color=on(3))); s.append(svg_text(660, 168, f"h = {v(stp['h'])}", size=10, mono=True, color=on(3)))
    for x1, x2 in ((192, 218), (372, 398), (552, 578)):
        s.append(svg_arrow(x1, 113, x2, 113))
    s.append(f'<line x1="40" y1="230" x2="740" y2="230" stroke="#7C5CBF" stroke-width="3"/>')
    s.append(svg_text(390, 250, "cell state highway: c flows forward through + and ⊙ (no repeated tanh) — gradients survive", size=11, color="#7C5CBF"))
    s.append(svg_text(390, 285, "f≈1 keep · f≈0 forget      i≈1 write · i≈0 ignore      o≈1 expose · o≈0 hide", size=11, color="#6B675F"))
    s.append("</svg>")
    return "".join(s)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3 = st.columns(3)
    with c1:
        seq_txt = st.text_input("التسلسل x", "1.0, 1.0, -2.0, 0.5", key="lg_seq")
    with c2:
        H = st.select_slider("H", options=[2, 3, 4], value=3, key="lg_h")
    with c3:
        seed = st.slider("بذرة الأوزان", 0, 5, 0, key="lg_seed")
    try:
        xs = np.array([float(v) for v in seq_txt.replace("،", ",").split(",") if v.strip()])
    except ValueError:
        xs = np.array([1.0, 1.0, -2.0, 0.5])
    W = random_weights("lstm", 1, int(H), seed=int(seed), scale=0.8)
    steps = lstm_forward(xs, W, int(H))
    st.code(f"per timestep: f, i, g, o each = act([x_t, h_prev] @ W + b)   W: ({1 + H}, {H}) ×4   -> params = {4 * ((1 + H) * H + H)}", language="text")
    frames = []
    for stp in steps:
        t = stp["t"]; r2 = lambda a: np.round(a, 2).tolist()
        frames.append(Frame(_cell_svg(stp, 0), caption(f"**t = {t} — بوابة النسيان** `f = σ([x_t, h_prev]·Wf + bf) = {r2(stp['f'])}`: لكل مكوّن من حالة الخلية، كم نُبقي (≈1) أو ننسى (≈0) من `c_prev = {r2(stp['c_prev'])}`."), action="forget", highlight=0, values=[("f[0]", "", f"{stp['f'][0]:.2f}")]))
        frames.append(Frame(_cell_svg(stp, 1), caption(f"t = {t} — **بوابة الإدخال والمرشَّح**: `i = {r2(stp['i'])}` كم نكتب، `g = tanh(…) = {r2(stp['g'])}` ماذا نكتب. الحاصل `i ⊙ g` هو المعلومة الجديدة المرشَّحة."), action="input", highlight=1, values=[("i[0]", "", f"{stp['i'][0]:.2f}"), ("g[0]", "", f"{stp['g'][0]:+.2f}")]))
        frames.append(Frame(_cell_svg(stp, 2), caption(f"t = {t} — **تحديث حالة الخلية**: `c = f ⊙ c_prev + i ⊙ g = {r2(stp['c'])}`. جمع لا ضرب متكرر: هذا «الطريق السريع» الذي يحفظ التدرج عبر الزمن."), action="cell", highlight=2, equation=f"c[0] = {stp['f'][0]:.2f}×{stp['c_prev'][0]:+.2f} + {stp['i'][0]:.2f}×{stp['g'][0]:+.2f} = {stp['c'][0]:+.2f}", values=[("c[0]", f"{stp['c_prev'][0]:+.2f}", f"{stp['c'][0]:+.2f}")]))
        frames.append(Frame(_cell_svg(stp, 3), caption(f"t = {t} — **بوابة الإخراج**: `o = {r2(stp['o'])}` كم نكشف من الحالة؛ `h = o ⊙ tanh(c) = {r2(stp['h'])}` هو ما يخرج للطبقة التالية ويعود في الخطوة القادمة."), action="output", highlight=3, values=[("h[0]", f"{stp['h_prev'][0]:+.2f}", f"{stp['h'][0]:+.2f}")]))
    animation_player("lg_anim", frames, title_ar="خلية LSTM بوابةً بوابة", stages=STAGES, interval_ms=1800)
    table(["t", "x", "f", "i", "g", "c", "o", "h"], [(str(s_["t"]), f"{s_['x'][0]:+.1f}") + tuple(str(np.round(s_[k], 2).tolist()) for k in ("f", "i", "g", "c", "o", "h")) for s_ in steps], ["num", "num", "code", "code", "code", "code", "code", "code"])
    intuition("جرّب تسلسلًا يبدأ بإشارة قوية ثم يهدأ (2, 0, 0, 0): راقب `c` — إن كانت f ≈ 1 تحتفظ الخلية بالإشارة لخطوات بلا تلاشٍ. في RNN البسيط كانت ستُضرب في Wh وtanh كل خطوة.")
    practical_note("الأوزان هنا عشوائية لتوضيح الآلية؛ في التدريب تتعلم الشبكة **متى** تنسى ومتى تكتب ومتى تكشف. في Keras: `layers.LSTM(units)` تحوي كل هذا؛ `return_sequences=True` يعيد h لكل خطوة بدل الأخيرة فقط.")
