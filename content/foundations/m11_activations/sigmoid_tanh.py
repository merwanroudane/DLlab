import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.activations.sigmoid_tanh",
    title_ar="Sigmoid وtanh: الإشباع",
    title_en="Sigmoid & tanh: Saturation",
    module="foundations.activations",
    order=2,
    prerequisites=["foundations.activations.why_nonlinearity", "foundations.calculus.derivative"],
    objectives_ar=["صيغتا Sigmoid وtanh ومشتقتاهما ومداهما.", "فهم الإشباع: لماذا تموت المشتقة بعيدًا عن الصفر وما أثره عبر الطبقات.", "معرفة متى تبقى Sigmoid ضرورية (الإخراج الثنائي، بوابات LSTM)."],
    terms=["exponential", "derivative"],
    labs=["labs.activation_lab"],
    difficulty="intermediate",
    summary_ar="σ(z) ∈ (0,1) بمشتقة ≤ 0.25؛ tanh ∈ (−1,1) بمشتقة ≤ 1؛ كلاهما يُشبع بعيدًا عن الصفر فيتلاشى التدرج عبر الطبقات.",
)

CODE = '''import numpy as np
sigmoid = lambda z: 1/(1+np.exp(-z)); dsig = lambda z: sigmoid(z)*(1-sigmoid(z))
tanh = np.tanh;                        dtanh = lambda z: 1 - np.tanh(z)**2

for z in [0, 1, 2, 4, 6, 10]:
    print(f"z={z:>3}  sigmoid={sigmoid(z):.4f} d={dsig(z):.4f}   tanh={tanh(z):.4f} d={dtanh(z):.4f}")

# تلاشي التدرج: حاصل ضرب مشتقات L طبقة عند z=2
for L in [1, 3, 5, 10]:
    print(f"{L:>2} sigmoid layers at z=2: gradient factor = {dsig(2)**L:.2e}   tanh: {dtanh(2)**L:.2e}")'''


def render() -> None:
    lesson_header(LESSON)
    h2("Sigmoid", "Sigmoid")
    equation(r"\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,(1 - \sigma(z)) \le 0.25",
             [(r"\sigma(z)", "المدى (0, 1): احتمال."), (r"\sigma'(z)", "المشتقة بدلالة الدالة نفسها؛ قصوى عند $z = 0$ (0.25) وتتلاشى بعيدًا.")],
             meaning_ar="تضغط أي عدد إلى احتمال. بعيدًا عن الصفر تصبح مسطحة: **مشبعة**.",
             example_ar="$\\sigma(0) = 0.5$، $\\sigma(4) = 0.982$، $\\sigma'(4) = 0.018$.",
             dl_link_ar="ضرورية لطبقة الإخراج الثنائي ولبوابات LSTM/GRU (نريد قيمًا في (0,1)). غير محبذة للطبقات المخفية العميقة.", title_ar="Sigmoid")
    h2("tanh", "tanh")
    equation(r"\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = 2\sigma(2z) - 1, \qquad \tanh'(z) = 1 - \tanh^2(z) \le 1",
             [(r"\tanh(z)", "المدى (−1, 1)، مركزها صفر."), ("2\\sigma(2z) - 1", "Sigmoid ممدودة ومزاحة."), (r"\tanh'(z)", "قصوى 1 عند الصفر؛ تُشبع أيضًا.")],
             meaning_ar="مثل Sigmoid لكن متمركزة حول الصفر (مخرجات موجبة وسالبة) ومشتقتها أكبر بأربع مرات عند المركز.",
             example_ar="$\\tanh(1) = 0.76$، $\\tanh'(2) = 0.07$.",
             dl_link_ar="التنشيط القياسي داخل خلايا RNN/LSTM/GRU لحالة الخلية. أفضل من Sigmoid للمخفية لكنها تُشبع أيضًا.", title_ar="tanh")
    z = np.linspace(-6, 6, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=1 / (1 + np.exp(-z)), name="σ(z)", line=dict(color="#C8473A", width=3)))
    fig.add_trace(go.Scatter(x=z, y=(1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))), name="σ'(z)", line=dict(color="#C8473A", dash="dot")))
    fig.add_trace(go.Scatter(x=z, y=np.tanh(z), name="tanh(z)", line=dict(color="#C77A1A", width=3)))
    fig.add_trace(go.Scatter(x=z, y=1 - np.tanh(z) ** 2, name="tanh'(z)", line=dict(color="#C77A1A", dash="dot")))
    fig.add_vrect(x0=3, x1=6, fillcolor="#FBE6E2", opacity=0.4, line_width=0, annotation_text="saturation"); fig.add_vrect(x0=-6, x1=-3, fillcolor="#FBE6E2", opacity=0.4, line_width=0)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="sig_fig")
    intuition("المنطقة الوردية هي **الإشباع**: الدالة مسطحة، المشتقة ≈ 0، فلا تعرف الخلية في أي اتجاه تتحرك. تدرج يمر بخمس طبقات مشبعة يُضرب في خمسة أرقام صغيرة — يتلاشى.")
    code_lab(CodeLab(
        key="act_sig", title_ar="قيم ومشتقات، وتلاشي التدرج عبر الطبقات", code=CODE,
        before=Before(goal_ar="جدول قيم/مشتقات للدالتين، ثم حاصل ضرب المشتقات عبر L طبقات لرؤية التلاشي.", stage_ar="التنشيط ← التشخيص.",
                      inputs_ar="قيم z وأعداد طبقات.", expected_ar="مشتقات تتناقص مع |z|؛ عبر 10 طبقات Sigmoid عند z = 2 عامل ≈ 1e-9."),
        explain=[("2-3", "الدالتان ومشتقتاهما بصيغة مغلقة."), ("5-6", "لاحظ أن مشتقة Sigmoid عند 0 هي 0.25 فقط — حتى أفضل حالاتها تُضعف التدرج للربع."), ("9-10", "قاعدة السلسلة: العامل عبر L طبقات = حاصل ضرب L مشتقة. مع 0.105 (Sigmoid عند z=2) يهبط أسّيًا.")],
        run=run_printed(CODE),
        after_ar="- عامل 1e-9 يعني الطبقات الأولى لا تتعلم عمليًا. هذا هو سبب هجر Sigmoid في المخفية بعد 2010 واعتماد ReLU.\n- tanh أفضل (مشتقة قصوى 1) لكنها تُشبع أيضًا.",
    ))
    why("لماذا نبقي Sigmoid إذن؟ لأن بعض المخرجات **يجب** أن تكون احتمالات (الإخراج الثنائي) أو نسبًا في (0,1) (بوابات LSTM: «افتح 30%»). هناك الإشباع مقبول لأنه طبقة واحدة، لا عشر.")
    common_mistake("شبكة من 4 طبقات Sigmoid تتدرب ببطء شديد فيُزاد معدل التعلم حتى تنفجر. المشكلة ليست معدل التعلم بل تلاشي التدرج: استبدل التنشيط بـ ReLU.")
    quiz("act.sig", [
        Q("أقصى مشتقة لـ Sigmoid…", ["1", "0.25", "0.5"], 1, "عند z = 0."),
        Q("مدى tanh…", ["(0, 1)", "(−1, 1)", "[0, ∞)"], 1, "متمركز حول الصفر."),
        Q("10 طبقات Sigmoid عند z ≈ 2: التدرج في الطبقة الأولى…", ["يتضخم", "≈ 1e-9 (يتلاشى)", "لا يتغير"], 1, "0.105^10."),
        Q("أين تبقى Sigmoid ضرورية؟", ["المخفية العميقة", "الإخراج الثنائي وبوابات LSTM", "الانحدار"], 1, "احتمال / نسبة."),
    ])
    takeaway("Sigmoid وtanh تُشبعان؛ مشتقاتهما صغيرة بعيدًا عن الصفر فيتلاشى التدرج عبر الطبقات. أبقِهما حيث المدى مطلوب (إخراج ثنائي، بوابات).")
    lesson_footer(LESSON, ["σ' ≤ 0.25، tanh' ≤ 1.", "الإشباع = مشتقة صفر = لا تعلم.", "Sigmoid للإخراج الثنائي والبوابات فقط."])
