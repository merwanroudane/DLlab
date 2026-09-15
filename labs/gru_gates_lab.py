"""GRU Gates Lab — update and reset gates of one GRU cell over a short
sequence, side by side with the LSTM/RNN parameter counts (spec §41)."""

import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note
from components.diagram import svg_arrow, svg_box, svg_defs, svg_text
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.rnn import count_params, gru_forward, random_weights

LAB = Lab(
    id="labs.gru_gates_lab",
    title_ar="معمل بوابات GRU",
    title_en="GRU Gates Lab",
    category="sequence",
    description_ar="خلية GRU على تسلسل قصير: بوابة التحديث z، بوابة إعادة الضبط r، الحالة المرشَّحة h̃، والمزج (1−z)·h + z·h̃ — مع مقارنة عدد المعلمات بـ RNN وLSTM.",
    related_lessons=["course.w12.update_reset_gates", "course.w12.compare_rnn_lstm_gru"],
)

STAGES = ["update z", "reset r", "candidate h̃", "mix h"]


def _cell_svg(stp: dict, stage: int) -> str:
    def v(arr):
        return "[" + ", ".join(f"{a:+.2f}" for a in arr[:3]) + ("…" if len(arr) > 3 else "") + "]"
    on = lambda k: "#C8473A" if k == stage else "#B9B2A6"
    fill = lambda k: "#FBE6E2" if k == stage else "#FFFDF9"
    s = ['<svg viewBox="0 0 760 260" width="100%" style="max-width:760px">', svg_defs()]
    s.append(svg_text(80, 30, f"x_t = {stp['x'][0]:+.2f}", size=12, mono=True, color="#2F6FB5"))
    s.append(svg_text(80, 55, f"h_prev = {v(stp['h_prev'])}", size=11, mono=True, color="#6B675F"))
    s.append(svg_box(40, 90, 150, 46, "z = σ([x,h]·Wz + bz)", fill(0), stroke=on(0), font=11)); s.append(svg_text(115, 152, f"z = {v(stp['z'])}", size=11, mono=True, color=on(0)))
    s.append(svg_box(220, 90, 150, 46, "r = σ([x,h]·Wr + br)", fill(1), stroke=on(1), font=11)); s.append(svg_text(295, 152, f"r = {v(stp['r'])}", size=11, mono=True, color=on(1)))
    s.append(svg_box(400, 90, 150, 46, "h̃ = tanh([x, r⊙h]·Wh + bh)", fill(2), stroke=on(2), font=10)); s.append(svg_text(475, 152, f"h̃ = {v(stp['h_tilde'])}", size=11, mono=True, color=on(2)))
    s.append(svg_box(580, 90, 160, 46, "h = (1−z)⊙h + z⊙h̃", fill(3), stroke=on(3), font=11)); s.append(svg_text(660, 152, f"h = {v(stp['h'])}", size=11, mono=True, color=on(3)))
    for x1, x2 in ((192, 218), (372, 398), (552, 578)):
        s.append(svg_arrow(x1, 113, x2, 113))
    s.append(svg_text(390, 200, "z≈0 keep old h (skip) · z≈1 replace with h̃      r≈0 ignore old h when proposing · r≈1 use it", size=11, color="#6B675F"))
    s.append(svg_text(390, 235, "one state h (no separate c); two gates instead of three; ~25% fewer parameters than LSTM", size=11, color="#7C5CBF"))
    s.append("</svg>")
    return "".join(s)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3 = st.columns(3)
    with c1:
        seq_txt = st.text_input("التسلسل x", "2.0, 0.0, 0.0, -1.0", key="gg_seq")
    with c2:
        H = st.select_slider("H", options=[2, 3, 4], value=3, key="gg_h")
    with c3:
        seed = st.slider("بذرة الأوزان", 0, 5, 0, key="gg_seed")
    try:
        xs = np.array([float(v) for v in seq_txt.replace("،", ",").split(",") if v.strip()])
    except ValueError:
        xs = np.array([2.0, 0.0, 0.0, -1.0])
    W = random_weights("gru", 1, int(H), seed=int(seed), scale=0.8)
    steps = gru_forward(xs, W, int(H))
    frames = []
    for stp in steps:
        t = stp["t"]; r2 = lambda a: np.round(a, 2).tolist()
        frames.append(Frame(_cell_svg(stp, 0), caption(f"**t = {t} — بوابة التحديث** `z = σ([x_t, h_prev]·Wz + bz) = {r2(stp['z'])}`: كم نستبدل من الحالة القديمة (≈1) وكم نبقي (≈0)."), action="update", highlight=0, values=[("z[0]", "", f"{stp['z'][0]:.2f}")]))
        frames.append(Frame(_cell_svg(stp, 1), caption(f"t = {t} — **بوابة إعادة الضبط** `r = {r2(stp['r'])}`: كم نستخدم من الحالة القديمة عند **اقتراح** الحالة الجديدة."), action="reset", highlight=1, values=[("r[0]", "", f"{stp['r'][0]:.2f}")]))
        frames.append(Frame(_cell_svg(stp, 2), caption(f"t = {t} — **الحالة المرشَّحة** `h̃ = tanh([x_t, r ⊙ h_prev]·Wh + bh) = {r2(stp['h_tilde'])}`."), action="candidate", highlight=2, values=[("h̃[0]", "", f"{stp['h_tilde'][0]:+.2f}")]))
        frames.append(Frame(_cell_svg(stp, 3), caption(f"t = {t} — **المزج** `h = (1−z) ⊙ h_prev + z ⊙ h̃ = {r2(stp['h'])}`: متوسط موزون بين القديم والجديد — مسار جمع يحفظ التدرج مثل LSTM لكن بحالة واحدة."), action="mix", highlight=3, equation=f"h[0] = (1−{stp['z'][0]:.2f})×{stp['h_prev'][0]:+.2f} + {stp['z'][0]:.2f}×{stp['h_tilde'][0]:+.2f} = {stp['h'][0]:+.2f}", values=[("h[0]", f"{stp['h_prev'][0]:+.2f}", f"{stp['h'][0]:+.2f}")]))
    animation_player("gg_anim", frames, title_ar="خلية GRU: تحديث، إعادة ضبط، مرشَّح، مزج", stages=STAGES, interval_ms=1800)
    table(["t", "x", "z", "r", "h̃", "h"], [(str(s_["t"]), f"{s_['x'][0]:+.1f}") + tuple(str(np.round(s_[k], 2).tolist()) for k in ("z", "r", "h_tilde", "h")) for s_ in steps], ["num", "num", "code", "code", "code", "code"])
    st.markdown("### عدد المعلمات لكل خلية (D = 1 مدخل)")
    table(["H", "SimpleRNN", "GRU", "LSTM"], [(str(h), str(count_params("rnn", 1, h)), str(count_params("gru", 1, h)), str(count_params("lstm", 1, h))) for h in (8, 16, 32, 64)], ["num"] * 4)
    intuition("جرّب (2, 0, 0, −1): مع z صغيرة في الخطوات الوسطى تحتفظ GRU بالإشارة الأولى (h لا يتغير كثيرًا) — نفس ما تفعله f ≈ 1 في LSTM، بحالة واحدة وبوابتين.")
    practical_note("GRU ≈ LSTM في الأداء على معظم المسائل الصغيرة والمتوسطة، بمعلمات أقل وتدريب أسرع قليلًا؛ LSTM قد تتفوق مع تسلسلات طويلة جدًا. القرار تجريبي على التحقق (الأسبوع 12: المقارنة).")
