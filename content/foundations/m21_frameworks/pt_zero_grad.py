import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway, why
from components.comparison import compare_table, good_vs_bad
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table
from labs.fw import torch_zero_grad_experiment

LESSON = Lesson(
    id="foundations.frameworks.pytorch.zero_grad",
    title_ar="لماذا zero_grad()؟ تجربة التراكم",
    title_en="Why zero_grad()? The Accumulation Experiment",
    module="foundations.frameworks",
    parent="foundations.frameworks.pytorch",
    order=25,
    prerequisites=["foundations.frameworks.pytorch.training_loop", "foundations.frameworks.pytorch.autograd"],
    objectives_ar=["مشاهدة .grad يتراكم عبر الدفعات عند حذف zero_grad، وأثره على معيار التدرج واتجاه التحديث.", "الصياغة الرياضية: التحديث بمجموع تدرجات كل الدفعات السابقة بدل تدرج الدفعة الحالية.", "متى يكون التراكم مقصودًا (تجميع التدرجات) وكيف يُفعل بشكل صحيح."],
    terms=["gradient", "batch", "iteration"],
    difficulty="intermediate",
    summary_ar="backward يجمع (+=) في .grad. بلا zero_grad، الخطوة k تستخدم g₁+…+g_k: اتجاه قديم ومعيار متزايد. zero_grad قبل backward في كل دفعة — إلا عند تجميع مقصود لعدة دفعات ثم خطوة واحدة.",
)


def render() -> None:
    lesson_header(LESSON)
    definition("**`optimizer.zero_grad()`**: يصفّر (أو يجعل `None`) حقل `.grad` لكل معلمة سجّلها المحسّن. ضروري لأن `loss.backward()` **يجمع** التدرج الجديد على القديم (`p.grad += ∂L/∂p`) ولا يستبدله. من دونه، تدرج كل دفعة يحمل معه تدرجات كل الدفعات السابقة.")
    why("لا تحفظها كتعليمة غامضة. التجربة أدناه تشغّل **نفس النموذج بنفس الدفعات** مرتين — مع التصفير وبدونه — وتقارن تدرج وزن واحد ومعيار التدرج والخسارة خطوةً خطوة. ثم نصوغ ما رأيته رياضيًا.")
    h2("التجربة", "The experiment")
    c1, c2 = st.columns(2)
    with c1:
        steps = st.slider("عدد الخطوات (دفعات)", 3, 6, 6, key="zg_steps")
    with c2:
        lr = st.select_slider("lr", options=[0.05, 0.1, 0.2], value=0.1, key="zg_lr")
    res = torch_zero_grad_experiment(int(steps), float(lr), 0)
    w, wo = res["with"], res["without"]
    rows = []
    for a, b in zip(w, wo):
        fresh = b["fresh"]
        rows.append((str(a["step"]), f"{a['g00']:+.4f}", f"{b['g00']:+.4f}", "—" if fresh is None else f"{fresh:+.4f}", f"{a['gnorm']:.3f}", f"{b['gnorm']:.3f}", f"{a['loss']:.4f}", f"{b['loss']:.4f}"))
    table(["الخطوة", "grad[0,0] مع zero_grad", "grad[0,0] بدون", "إسهام هذه الدفعة وحدها (بدون)", "‖grad‖ مع", "‖grad‖ بدون", "loss مع", "loss بدون"], rows, ["num"] * 8)
    fig = go.Figure()
    s = [r["step"] for r in w]
    fig.add_trace(go.Scatter(x=s, y=[r["gnorm"] for r in w], name="‖grad‖ with zero_grad", line=dict(color="#1F7A78", width=3), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=s, y=[r["gnorm"] for r in wo], name="‖grad‖ WITHOUT zero_grad", line=dict(color="#C8473A", width=3), mode="lines+markers"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="step (batch)", yaxis_title="‖weight.grad‖", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="zg_fig")
    with st.expander("كيف أقرأ الجدول والرسم؟", icon=":material/visibility:", expanded=True):
        st.markdown(f"""
- **الخطوة 1 متطابقة** في العمودين: لا تراكم بعد (`.grad` كان `None`).
- من الخطوة 2: العمود «بدون» = العمود السابق «بدون» + «إسهام هذه الدفعة وحدها». مثلًا الخطوة 2: {wo[0]['g00']:+.4f} + ({wo[1]['fresh']:+.4f}) = {wo[1]['g00']:+.4f}.
- **‖grad‖ بدون** يكبر تقريبًا خطيًا مع عدد الخطوات (الخط الأحمر) بينما «مع» يبقى في نطاق ثابت (الأخضر).
- الخسارة «بدون» ليست بالضرورة أسوأ في الخطوات الأولى (التدرجات المتراكمة تشير لاتجاه مشابه) — لكن الخطوة الفعلية تكبر بلا حدود: مع مئات الدفعات ينفجر التدريب أو يتذبذب.
""")
    h2("رياضيًا", "Mathematically")
    equation(r"\text{with:}\quad \theta_{k+1} = \theta_k - \eta\, g_k \qquad\qquad \text{without:}\quad \theta_{k+1} = \theta_k - \eta \sum_{i=1}^{k} g_i",
             [(r"g_k", "تدرج خسارة الدفعة k عند المعلمات الحالية θ_k — ما نريده."), (r"\sum_{i\le k} g_i", "ما يحويه `.grad` فعليًا بلا تصفير: تدرجات حُسبت عند معلمات **قديمة** θ₁…θ_{k−1} لم تعد صحيحة.")],
             meaning_ar="الخطوة k بلا تصفير تساوي k خطوات بمتوسط تدرجات قديمة: اتجاه متأخر ومعيار ينمو ~k. حتى لو كان الاتجاه معقولًا، حجم الخطوة يخرج عن السيطرة.",
             example_ar=f"في الجدول: عند الخطوة {len(wo)} المعيار «بدون» = {wo[-1]['gnorm']:.3f} مقابل {w[-1]['gnorm']:.3f} «مع» — أي خطوة أكبر {wo[-1]['gnorm'] / max(w[-1]['gnorm'], 1e-9):.1f}× بلا سبب.",
             dl_link_ar="Keras لا تعرض هذا لأن `fit` تحسب التدرج وتطبقه وتنساه. في PyTorch التراكم ميزة تصميمية تخدم تجميع التدرجات.", title_ar="التحديث مع/بدون التصفير")
    intuition("تخيّل أنك تصحح اتجاهك كل 10 أمتار بحسب البوصلة. «بدون تصفير» = تجمع كل قراءات البوصلة السابقة وتتحرك بمجموعها: في الخطوة العاشرة تقفز عشرة أضعاف في اتجاه يعكس أماكن لم تعد فيها.")
    h2("متى يكون التراكم مقصودًا؟", "When accumulation is intended")
    research_note("**تجميع التدرجات (gradient accumulation)**: تريد دفعة فعلية 256 لكن الذاكرة تتسع لـ 64. مرّر 4 دفعات صغيرة مع `backward()` كل مرة **بلا تصفير**، ثم `step()` واحدة، ثم `zero_grad()`. التدرج المجمَّع ≈ تدرج الدفعة الكبيرة (اقسم الخسارة على 4 لتطابق المتوسط). هنا التراكم مقصود ومحسوب — والفرق عن الخطأ هو أن الخطوة **واحدة** بعد التجميع.")
    good_vs_bad("تجميع مقصود", "backward على 4 دفعات، step واحدة، ثم zero_grad.", "نسيان التصفير", "step بعد كل backward مع تراكم مستمر: اتجاه قديم ومعيار ينفجر.",
                good_code="optimizer.zero_grad()\nfor i, (xb, yb) in enumerate(loader):\n    loss = loss_fn(model(xb), yb) / 4\n    loss.backward()            # يتراكم عمدًا\n    if (i + 1) % 4 == 0:\n        optimizer.step()\n        optimizer.zero_grad()",
                bad_code="for xb, yb in loader:\n    loss = loss_fn(model(xb), yb)\n    loss.backward()            # يتراكم بلا قصد\n    optimizer.step()           # خطوة بمجموع كل الماضي",
                verdict_ar="التراكم أداة عندما تملكه، وعطب عندما ينساك.")
    compare_table(["الخيار", "الكود", "ملاحظة"],
                  [("تصفير عبر المحسّن", "`optimizer.zero_grad()`", "الافتراضي؛ يضبط `.grad = None` (أسرع من الصفر) في النسخ الحديثة"), ("تصفير عبر النموذج", "`model.zero_grad()`", "نفس الأثر على معلمات النموذج"), ("تصفير يدوي", "`p.grad.zero_()` لكل p", "ما رأيته في درس Autograd")],
                  ["rtl", "code", "rtl"])
    common_mistake("وضع `zero_grad()` **بعد** `step()` بدل قبله يعمل أيضًا — لكنه يُنسى عند إضافة `continue` أو استثناء. الموضع القياسي: أول سطر داخل حلقة الدفعات.")
    quiz("pt.zero", [
        Q("`backward()` مرتين بلا تصفير يعطي في `.grad`…", ["آخر تدرج", "مجموع التدرجين", "متوسطهما"], 1, "+=."),
        Q("بلا zero_grad، معيار التدرج عبر الخطوات…", ["ثابت", "ينمو تقريبًا خطيًا", "يتلاشى"], 1, "تراكم."),
        Q("تجميع التدرجات المقصود يعني…", ["step بعد كل backward", "عدة backward ثم step واحدة ثم zero_grad", "بلا backward"], 1, "دفعة فعلية أكبر."),
        Q("Keras لا تحتاج zero_grad لأن…", ["لا تحسب تدرجات", "fit يحسب ويطبق وينسى داخل الخطوة", "التدرجات لا تتراكم في TensorFlow"], 1, "لا تخزين في المعلمات."),
    ])
    takeaway("backward يجمع؛ zero_grad يمسح. بلا مسح: التحديث بمجموع تدرجات قديمة ومعيار ينفجر. المسح أول سطر في حلقة الدفعات — إلا عند تجميع مقصود بخطوة واحدة.")
    lesson_footer(LESSON, ["التجربة: مع/بدون جنبًا إلى جنب.", "الصياغة الرياضية.", "التراكم المقصود."])
