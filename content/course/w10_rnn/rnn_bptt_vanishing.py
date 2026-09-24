import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from components.animation_player import Frame, animation_player, caption
from content.course.w10_rnn._viz import bptt_norms, bptt_svg
from core.routing import go as goto
from labs.rnn import bptt_gradient_norms

LESSON = Lesson(
    id="course.w10.rnn_bptt_vanishing",
    title_ar="النشر عبر الزمن (BPTT) وتلاشي/انفجار التدرج — ولماذا LSTM",
    title_en="Backpropagation Through Time & Vanishing/Exploding Gradients — and Why LSTM",
    module="course.w10",
    order=3,
    prerequisites=["course.w10.sequences_hidden_state", "foundations.backprop.backpropagation.vanishing_exploding", "foundations.calculus.chain_rule"],
    objectives_ar=["BPTT: الشبكة المنشورة هي شبكة عميقة بعمق T بأوزان مشتركة؛ التدرج يمر عبر T يعقوبيًا — ونشاهده يعود خطوةً خطوة (تحريك).", "التلاشي والانفجار كدالة في نصف قطر Wh وطول التسلسل، بالأرقام والرسم.", "العلاجات: القصّ، النوافذ الأقصر، التهيئة — ولماذا LSTM/GRU الحل البنيوي (تمهيد)."],
    terms=["gradient", "chain_rule", "backpropagation"],
    labs=["labs.rnn_unrolling_lab"],
    difficulty="advanced",
    summary_ar="∂L/∂h_t = ∂L/∂h_T · Π (diag(1−h²) Whᵀ): حاصل ضرب T−t مصفوفة. نصف قطر < 1 → تلاشٍ (الماضي البعيد لا يتعلَّم)، > 1 → انفجار (nan). القصّ يعالج الانفجار؛ LSTM/GRU تعالج التلاشي بمسار جمع.",
)


def render() -> None:
    lesson_header(LESSON)
    definition("**النشر عبر الزمن** `BPTT`: انشر الشبكة التكرارية على T خطوة فتصبح شبكة أمامية بعمق T تتشارك طبقاتها الأوزان؛ طبّق الانتشار الخلفي العادي (الأسس 14) ثم **اجمع** تدرجات كل النسخ لأن الأوزان واحدة. الفرق عن الشبكة الأمامية: نفس المصفوفة Wh تُضرب T مرة.")
    equation(r"\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=t+1}^{T} \mathrm{diag}\big(1 - h_k^2\big)\, W_h^{\top}",
             [(r"\mathrm{diag}(1 - h_k^2)", "مشتقة tanh: ≤ 1، وتقترب من 0 عند التشبع."), ("W_h^\\top", "نفس المصفوفة T−t مرة."), (r"\prod", "حاصل ضرب مصفوفات: معياره ينمو أو يتلاشى أسيًا مع T−t بحسب أكبر قيمة ذاتية لـ Wh (نصف القطر الطيفي ρ).")],
             meaning_ar="التدرج الذي يصل من الخسارة عند T إلى الخطوة t يُضرب في ρ تقريبًا (T−t) مرة: ρ < 1 → 0 (تلاشٍ)، ρ > 1 → ∞ (انفجار). مشتقة tanh تدفع دائمًا نحو التلاشي.",
             example_ar="ρ = 0.5 وT−t = 20: 0.5²⁰ ≈ 10⁻⁶ — الخطوة الأولى لا تتعلم شيئًا من خطأ الخطوة العشرين.", dl_link_ar="هذا سبب `clipnorm` في المحسّنات وسبب وجود LSTM/GRU أصلًا.", title_ar="تدرج الحالة عبر الزمن")
    why("في الدرس السابق كان فارق الذاكرة 0.015: أثر الماضي ضعيف في التمرير الأمامي، والأسوأ في الخلفي — التدرج الذي يعلّم الشبكة «تذكّر ما حدث قبل 20 خطوة» يصل مضروبًا في ρ²⁰. النموذج **لا يستطيع** تعلّم الاعتماديات الطويلة حتى لو كانت في البيانات. هذه ليست مشكلة ضبط بل مشكلة بنية.")
    h2("شاهد التدرج يعود عبر الزمن", "Watch the gradient travel back in time")
    bn = bptt_norms()
    T = len(bn["0.5"])
    shown = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
    bcaps = []
    for u in shown:
        t = T - u + 1
        vals = {k: v[u - 1] for k, v in bn.items()}
        bcaps.append(f"**من الخطوة 30 إلى الخطوة {t}** ({u - 1} خطوة إلى الخلف): ρ = 0.5 → {vals['0.5']:.1e}؛ ρ = 1.0 → {vals['1.0']:.1e}؛ ρ = 1.5 → {vals['1.5']:.1e}."
                     + (" مع ρ = 1.5 التدرج **يكبر** أولًا (أكثر من 1)…" if u == 3 else "")
                     + (" …لكن تشبع tanh (1 − h² صغيرة) يكبحه ثم يهبط. مع ρ = 0.5 وصلنا إلى الملايين من الجزء — الخطوات الأولى لا تتعلم شيئًا." if u == 20 else ""))
    animation_player("w10_bptt_anim", [Frame(bptt_svg(bn, u), caption(c), action=f"{u - 1} steps back") for u, c in zip(shown, bcaps)],
                     title_ar="‖∂h₃₀/∂hₜ‖ لثلاث قيم لنصف القطر الطيفي (خلية tanh حقيقية)", interval_ms=1600)
    h2("التجربة", "The experiment")
    c1, c2 = st.columns(2)
    with c1:
        T = st.slider("طول التسلسل T", 5, 60, 30, 5, key="w10_T")
    with c2:
        rhos = st.multiselect("نصف القطر الطيفي ρ(Wh)", [0.3, 0.5, 0.9, 1.0, 1.2, 2.0, 4.0], default=[0.5, 0.9, 1.0, 2.0], key="w10_rhos")
    fig = go.Figure(); rows = []
    colors = ["#2563EB", "#059669", "#7C3AED", "#D97706", "#DB2777", "#6B675F", "#E8710A"]
    for rho, color in zip(rhos, colors):
        norms = bptt_gradient_norms(int(T), wh_scale=float(rho))
        fig.add_trace(go.Scatter(x=list(range(T, 0, -1)), y=norms, name=f"ρ = {rho}", line=dict(color=color, width=2.5)))
        rows.append((str(rho), f"{norms[min(9, len(norms) - 1)]:.2e}", f"{norms[-1]:.2e}", "تلاشٍ" if norms[-1] < 1e-3 else ("انفجار" if norms[-1] > 10 else "معتدل")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="timestep t (gradient flows from T back to t)", yaxis_title="‖∂h_T/∂h_t‖ (log)", yaxis_type="log", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="w10_bptt_fig")
    from core.rtl import table
    table(["ρ(Wh)", "‖∂h_T/∂h_(T−10)‖", "‖∂h_T/∂h_1‖", "الحكم"], rows, ["num", "num", "num", "rtl"])
    intuition("المحور الرأسي لوغاريتمي: الخطوط المستقيمة النازلة تعني تلاشيًا **أسيًا**. حتى ρ = 1 تتلاشى بسبب مشتقة tanh (< 1). فقط ρ كبير ينفجر — وحتى هو يُكبح جزئيًا بتشبع tanh، لكن مع أوزان متعلَّمة وتسلسلات طويلة يظهر nan.")
    h2("العلاجات", "Remedies")
    compare_table(["العلاج", "يعالج", "الكود/الطريقة", "الحد"],
                  [("قصّ التدرج", "الانفجار", "`Adam(clipnorm=1.0)` أو `clipvalue`", "لا يعالج التلاشي"), ("نوافذ أقصر / TBPTT", "الاثنين جزئيًا", "timesteps أصغر", "يفقد الاعتماديات الأطول من النافذة"), ("تهيئة متعامدة لـ Wh", "التلاشي جزئيًا", "`recurrent_initializer='orthogonal'` (افتراضي في Keras)", "tanh ما زالت تتلاشى"),
                   ("ReLU في RNN", "التلاشي", "`activation='relu'`", "انفجار أسهل؛ نادر"), ("**LSTM / GRU**", "التلاشي (بنيويًا)", "`layers.LSTM` / `layers.GRU`", "معلمات ×4 / ×3؛ الأسبوعان 11–12"), ("Transformers", "الاثنين (بلا تكرار)", "الانتباه (خارج المقرر)", "—")],
                  ["rtl", "rtl", "code", "rtl"])
    research_note("**تمهيد LSTM**: بدل أن تمر الحالة عبر tanh وWh في كل خطوة (ضرب متكرر)، تضيف LSTM **حالة خلية** c_t تتحدث بجمع: c_t = f ⊙ c_{t−1} + i ⊙ g. التدرج عبر مسار الجمع يُضرب في f (بوابة يمكن أن تكون ≈ 1) لا في Whᵀ·tanh′ — فيبقى. البوابات f وi وo تتعلم متى تنسى/تكتب/تكشف. الأسبوع 11 يفكّكها بوابة بوابة.")
    st.button("معمل نشر RNN: شاهد التلاشي بالأرقام", icon=":material/science:", type="primary", on_click=goto, args=("labs.rnn_unrolling_lab",), key="w10_lab_bptt")
    common_mistake("«سأزيد الحقب حتى تتعلم RNN الاعتمادية الطويلة». إن كان التدرج 10⁻⁶ فلن تتعلمها بأي عدد حقب: الإشارة غائبة لا ضعيفة. غيّر البنية (LSTM/GRU) أو قصّر المسألة.")
    quiz("w10.bptt", [
        Q("BPTT هو…", ["خوارزمية جديدة", "الانتشار الخلفي على الشبكة المنشورة مع جمع تدرجات الأوزان المشتركة", "تدريب بلا تدرج"], 1, ""),
        Q("ρ(Wh) = 0.5 وT = 20: التدرج إلى الخطوة الأولى ≈", ["0.5", "10⁻⁶", "1"], 1, "0.5²⁰."),
        Q("القصّ (clipnorm) يعالج…", ["التلاشي", "الانفجار", "الاثنين"], 1, ""),
        Q("LSTM تعالج التلاشي بـ…", ["أوزان أكبر", "مسار جمع لحالة الخلية مع بوابات", "حقب أكثر"], 1, ""),
    ])
    takeaway("BPTT = انتشار خلفي على شبكة بعمق T بأوزان مشتركة؛ التدرج يُضرب في ρ نحو T مرة: تلاشٍ أو انفجار. القصّ للانفجار؛ LSTM/GRU للتلاشي.")
    lesson_footer(LESSON, ["المعادلة والحدس.", "التجربة التفاعلية.", "العلاجات وتمهيد LSTM."])
