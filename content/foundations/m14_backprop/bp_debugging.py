import streamlit as st

from components.callouts import debugging_note, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.backprop.backpropagation.debugging",
    title_ar="تشخيص التدرجات: المعايير، التحقق العددي، القصّ",
    title_en="Debugging Gradients: Norms, Numerical Checks, Clipping",
    module="foundations.backprop",
    parent="foundations.backprop.backpropagation",
    order=7,
    prerequisites=["foundations.backprop.backpropagation.vanishing_exploding"],
    objectives_ar=["أدوات ثلاث: معيار التدرج لكل طبقة، التحقق العددي، قصّ التدرج بالمعيار.", "قراءة جدول معايير كأداة تشخيص روتينية."],
    terms=["norm", "gradient"],
    difficulty="intermediate",
    summary_ar="سجّل ‖g‖ لكل طبقة؛ تحقق عدديًا عند الشك؛ قصّ بالمعيار عند الانفجار.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)

# تدرجات لثلاث طبقات بمعايير مختلفة (محاكاة)
grads = [rng.normal(0, s, (16, 16)) for s in ({scales})]
norms = [np.linalg.norm(g) for g in grads]
total = np.sqrt(sum(n**2 for n in norms))
for i, n in enumerate(norms, 1):
    print(f"layer {{i}}: ||g|| = {{n:8.3f}}")
print(f"global norm = {{total:.3f}}   ratio first/last = {{norms[0]/norms[-1]:.2e}}")

# قصّ بالمعيار: إن كان المعيار الكلي > max_norm نضرب كل التدرجات بعامل واحد
max_norm = {max_norm}
factor = min(1.0, max_norm / (total + 1e-12))
clipped = [g * factor for g in grads]
print(f"clip factor = {{factor:.4f}} -> new global norm = {{np.sqrt(sum(np.linalg.norm(g)**2 for g in clipped)):.3f}}")
print("directions preserved:", np.allclose(clipped[0]/np.linalg.norm(clipped[0]), grads[0]/np.linalg.norm(grads[0])))'''


def _controls() -> dict:
    preset = st.selectbox("سيناريو", ["صحي (1, 1, 1)", "تلاشٍ (0.001, 0.03, 1)", "انفجار (50, 5, 1)"], key="ctrl_bpdbg_preset")
    scales = {"صحي (1, 1, 1)": "1.0, 1.0, 1.0", "تلاشٍ (0.001, 0.03, 1)": "0.001, 0.03, 1.0", "انفجار (50, 5, 1)": "50.0, 5.0, 1.0"}[preset]
    max_norm = st.select_slider("max_norm للقصّ", options=[0.5, 1.0, 5.0, 10.0, 100.0], value=5.0, key="ctrl_bpdbg_max")
    return {"scales": scales, "max_norm": max_norm}


def render() -> None:
    lesson_header(LESSON)
    h2("ثلاث أدوات", "Three tools")
    table(["الأداة", "ماذا تفعل", "متى", "في الأطر"],
          [("معيار التدرج لكل طبقة", "رقم واحد لكل طبقة كل حقبة: هل يتلاشى/ينفجر؟", "دائمًا (رخيص)", "callback / hooks؛ TensorBoard histograms"),
           ("التحقق العددي", "يقارن التدرج التحليلي بالفروق المنتهية", "عند تنفيذ طبقة/خسارة مخصصة", "tf.test.compute_gradient، torch.autograd.gradcheck"),
           ("قصّ التدرج", "يحدّ المعيار الكلي بعامل واحد يحفظ الاتجاه", "RNN، انفجار، بداية التدريب", "clipnorm= في Keras، clip_grad_norm_ في PyTorch")],
          ["rtl", "rtl", "rtl", "code"])
    code_lab(CodeLab(
        key="bp_dbg", title_ar="جدول معايير + قصّ بالمعيار", code=CODE, template=True, defaults={"scales": "1.0, 1.0, 1.0", "max_norm": 5.0},
        before=Before(goal_ar="محاكاة تدرجات ثلاث طبقات بثلاثة سيناريوهات، وقراءة الجدول، ثم تطبيق القصّ ورؤية أنه يحفظ الاتجاه.", stage_ar="الانتشار الخلفي ← التشخيص.",
                      inputs_ar="مقاييس التدرج لكل طبقة ومعيار أقصى.", expected_ar="نسبة أولى/أخيرة ≈ 1 (صحي)، 1e-3 (تلاشٍ)، 50 (انفجار)؛ القصّ يخفض المعيار الكلي إلى max_norm عند الحاجة."),
        explain=[("4-9", "معيار كل طبقة والمعيار الكلي (جذر مجموع المربعات). النسبة أولى/أخيرة هي المقياس التشخيصي."),
                 ("12-16", "القصّ بالمعيار: عامل واحد لكل التدرجات ⇒ الاتجاه لا يتغير، الحجم فقط. هذا هو `clipnorm` (وليس `clipvalue` الذي يقصّ كل عنصر على حدة ويشوّه الاتجاه).")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- في السيناريو الصحي القصّ لا يفعل شيئًا (factor = 1) إن كان max_norm كبيرًا.\n- في الانفجار يعمل؛ في التلاشي **لا يساعد** — القصّ للانفجار فقط؛ التلاشي يحتاج تنشيطًا/تهيئة/BN.",
    ))
    debugging_note("روتين مقترح لكل تدريب: (1) اطبع معيار التدرج الكلي كل N خطوة؛ (2) إن قفز فوق 10× متوسطه فعّل القصّ وخفّض معدل التعلم؛ (3) إن كانت نسبة الأولى/الأخيرة < 1e-3 فالمشكلة تلاشٍ.")
    practical_note("`clipnorm=1.0` في محسّنات Keras، و`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` قبل `optimizer.step()`. القيمة 1.0 شائعة للـ RNN؛ 5–10 للشبكات الكثيفة.")
    quiz("bp.dbg", [
        Q("clipnorm مقابل clipvalue…", ["متطابقان", "clipnorm يحفظ الاتجاه؛ clipvalue يقصّ كل عنصر ويشوّهه", "clipvalue أفضل"], 1, "عامل واحد."),
        Q("القصّ يعالج…", ["التلاشي", "الانفجار", "كليهما"], 1, "لا يكبّر الصغير."),
        Q("نسبة معيار الأولى/الأخيرة 1e-4…", ["صحي", "تلاشٍ", "انفجار"], 1, "تناقص أسّي."),
    ])
    takeaway("راقب المعايير، تحقق عدديًا عند الشك، قصّ عند الانفجار. القصّ لا يعالج التلاشي.")
    lesson_footer(LESSON, ["جدول المعايير روتين.", "clipnorm يحفظ الاتجاه.", "أدوات الأطر المقابلة."])
