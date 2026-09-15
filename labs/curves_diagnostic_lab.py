"""Training Curves Diagnostic Lab (spec §38): eight synthetic cases plus a
challenge mode that hides the label until "Reveal Diagnosis"."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.curves_diagnostic_lab",
    title_ar="معمل تشخيص منحنيات التدريب",
    title_en="Training Curves Diagnostic Lab",
    category="evaluation",
    description_ar="ثماني حالات لمنحنيات الخسارة (صحي، فرط تخصيص، قصور تعلم، معدل تعلم صغير/كبير، تباعد، ضوضاء، هضبة) وتحدٍّ: منحنى بلا اسم، شخّصه ثم اكشف الإجابة.",
    related_lessons=["foundations.generalization.curves", "foundations.optim.learning_rate"],
)

CASES = {
    "صحي": (lambda t, r: 1.0 * np.exp(-t / 8) + 0.15 + r.normal(0, 0.01, len(t)), lambda t, r: 1.0 * np.exp(-t / 8) + 0.2 + r.normal(0, 0.02, len(t)),
            "كلاهما يهبط ويستقر بفجوة صغيرة ثابتة.", "لا إجراء؛ ربما حقب إضافية قليلة."),
    "فرط تخصيص": (lambda t, r: 1.0 * np.exp(-t / 5) + 0.02 + r.normal(0, 0.005, len(t)), lambda t, r: 1.0 * np.exp(-t / 6) + 0.2 + 0.012 * np.maximum(t - 12, 0) + r.normal(0, 0.02, len(t)),
                   "التدريب يهبط باستمرار، التحقق يهبط ثم يصعد: الفجوة تتسع.", "إيقاف مبكر، تنظيم، بيانات أكثر، نموذج أصغر."),
    "قصور تعلم": (lambda t, r: 1.0 * np.exp(-t / 30) + 0.55 + r.normal(0, 0.01, len(t)), lambda t, r: 1.0 * np.exp(-t / 30) + 0.58 + r.normal(0, 0.015, len(t)),
                  "كلاهما مرتفع ومتقارب ويستقر مبكرًا.", "قدرة أكبر، حقب أكثر، خصائص أفضل، تقليل التنظيم."),
    "معدل تعلم صغير جدًا": (lambda t, r: 1.0 - 0.012 * t + r.normal(0, 0.005, len(t)), lambda t, r: 1.02 - 0.012 * t + r.normal(0, 0.01, len(t)),
                           "هبوط خطي بطيء جدًا بلا انحناء.", "ارفع η بمرتبة."),
    "معدل تعلم كبير": (lambda t, r: 0.5 + 0.25 * np.sin(t * 1.3) + r.normal(0, 0.03, len(t)), lambda t, r: 0.55 + 0.25 * np.sin(t * 1.3 + 0.3) + r.normal(0, 0.03, len(t)),
                       "تذبذب حاد لا يستقر.", "اخفض η، جدولة تنازلية."),
    "تباعد": (lambda t, r: 0.7 * np.exp(t / 9), lambda t, r: 0.75 * np.exp(t / 9), "صعود أسّي ثم NaN.", "اخفض η كثيرًا، قصّ التدرج، تحجيم، تهيئة."),
    "تدريب ضجيج": (lambda t, r: 0.6 * np.exp(-t / 8) + 0.2 + r.normal(0, 0.08, len(t)), lambda t, r: 0.6 * np.exp(-t / 8) + 0.28 + r.normal(0, 0.12, len(t)),
                    "اتجاه هابط بتذبذب كبير.", "دفعة أكبر، بيانات أكثر، متوسط عدة بذور، η أصغر قليلًا."),
    "هضبة": (lambda t, r: np.where(t < 15, 1.0 - 0.02 * t, 0.7 - 0.0005 * (t - 15)) + r.normal(0, 0.005, len(t)), lambda t, r: np.where(t < 15, 1.02 - 0.02 * t, 0.72 - 0.0005 * (t - 15)) + r.normal(0, 0.01, len(t)),
             "هبوط ثم توقف طويل على مستوى مرتفع.", "ReduceLROnPlateau، زخم/Adam، فحص التهيئة والتنشيط."),
}


def _plot(name: str, seed: int, show_title: bool):
    r = np.random.default_rng(seed); t = np.arange(0, 40)
    f_tr, f_va, _, _ = CASES[name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t + 1, y=f_tr(t, r), name="loss (train)", line=dict(color="#2F6FB5", width=3)))
    fig.add_trace(go.Scatter(x=t + 1, y=f_va(t, r), name="val_loss", line=dict(color="#C8473A", width=3)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title=name if show_title else "؟", xaxis_title="epoch", yaxis=dict(range=[0, 1.4]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    return fig


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    mode = st.segmented_control("الوضع", ["استعراض الحالات", "تحدٍّ تشخيصي"], default="استعراض الحالات", key="cdl_mode")
    if mode == "استعراض الحالات":
        name = st.selectbox("الحالة", list(CASES), key="cdl_case")
        st.plotly_chart(_plot(name, 0, True), width="stretch", key="cdl_fig")
        table(["ما تراه", "الإجراء"], [(CASES[name][2], CASES[name][3])], ["rtl", "rtl"])
        intuition("اقرأ بالترتيب: المستوى (مقارنة بالمرجع)، الفجوة، الاتجاه، النعومة.")
    else:
        st.session_state.setdefault("cdl_round", 0); st.session_state.setdefault("cdl_score", [0, 0])
        rnd = st.session_state["cdl_round"]
        rng = np.random.default_rng(1000 + rnd)
        answer = list(CASES)[int(rng.integers(0, len(CASES)))]
        st.plotly_chart(_plot(answer, 500 + rnd, False), width="stretch", key=f"cdl_chal_{rnd}")
        guess = st.radio("تشخيصك", list(CASES), index=None, key=f"cdl_guess_{rnd}", horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            reveal = st.button("اكشف التشخيص", type="primary", icon=":material/visibility:", key=f"cdl_reveal_{rnd}")
        with c2:
            if st.button("حالة جديدة", icon=":material/refresh:", key=f"cdl_next_{rnd}"):
                st.session_state["cdl_round"] += 1; st.rerun()
        if reveal:
            ok = guess == answer
            st.session_state["cdl_score"][1] += 1; st.session_state["cdl_score"][0] += int(ok)
            (st.success if ok else st.error)(f"التشخيص الصحيح: **{answer}** — {CASES[answer][2]} الإجراء: {CASES[answer][3]}", icon="✅" if ok else "❌")
        s = st.session_state["cdl_score"]
        st.caption(f"النتيجة: {s[0]} / {s[1]}")
