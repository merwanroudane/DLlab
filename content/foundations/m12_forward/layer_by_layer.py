import streamlit as st

from components.callouts import definition, intuition, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.forward.layer_by_layer",
    title_ar="طبقة بعد طبقة: من المدخل إلى المخرج",
    title_en="Layer by Layer: From Input to Output",
    module="foundations.forward",
    order=1,
    prerequisites=["foundations.architecture.build_and_read", "foundations.activations.why_nonlinearity"],
    objectives_ar=["كتابة التمرير الأمامي كتكرار لعملية واحدة: z = aW + b ثم a = f(z).", "تتبع الأشكال عبر الطبقات لدفعة.", "معرفة ما يُحفظ أثناء التمرير للانتشار الخلفي."],
    terms=["batch_dimension", "shape"],
    difficulty="beginner",
    summary_ar="التمرير الأمامي = تكرار (ضرب مصفوفات + انحياز + تنشيط) طبقة بعد طبقة؛ الأشكال تتغير على المحور الأخير فقط.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("عملية واحدة تتكرر", "One operation, repeated")
    pipeline(["X (n, d)", "z₁ = XW₁ + b₁", "a₁ = f(z₁)", "z₂ = a₁W₂ + b₂", "a₂ = f(z₂)", "…", "ŷ = g(z_L)"], active=1)
    equation(r"\mathbf{a}^{(0)} = X, \qquad \mathbf{z}^{(\ell)} = \mathbf{a}^{(\ell-1)} W^{(\ell)} + \mathbf{b}^{(\ell)}, \qquad \mathbf{a}^{(\ell)} = f^{(\ell)}\big(\mathbf{z}^{(\ell)}\big), \qquad \hat{\mathbf{y}} = \mathbf{a}^{(L)}",
             [(r"\ell", "رقم الطبقة من 1 إلى $L$ (رمز علوي بين قوسين: ليس أسًا)."), (r"\mathbf{a}^{(\ell-1)}", "مخرج الطبقة السابقة = مدخل هذه الطبقة."), (r"\mathbf{z}^{(\ell)}", "ما قبل التنشيط `pre-activation`."), (r"f^{(\ell)}", "تنشيط الطبقة؛ الأخير $g$ حسب المهمة.")],
             meaning_ar="التمرير الأمامي حلقة: خذ مخرج الطبقة السابقة، اضربه في الأوزان، أضف الانحياز، طبّق التنشيط، مرّره للتالية.",
             example_ar="دفعة `(32, 4)` عبر 4 → 16 → 8 → 1: `(32,4) → (32,16) → (32,8) → (32,1)`.",
             dl_link_ar="`model.predict(X)` و`model(X)` في Keras وPyTorch ينفذان هذه الحلقة بالضبط. وكل `z` و`a` يُحفظان في الذاكرة لأن الانتشار الخلفي يحتاجهما — لذلك تستهلك الدفعات الكبيرة ذاكرة GPU.", title_ar="التمرير الأمامي")
    intuition("سلسلة مصانع: كل مصنع يستلم منتج السابق، يعالجه بآلته (W, b) ثم بفلتره (f)، ويسلّمه للتالي. المنتج النهائي هو التنبؤ. حجم الدفعة (عدد القطع) لا يتغير؛ شكل القطعة يتغير.")
    compare_table(["الطبقة", "المدخل", "W", "b", "المخرج", "ما يُحفظ للخلفي"],
                  [("1", "(n, 4)", "(4, 16)", "(16,)", "(n, 16)", "z₁, a₁"), ("2", "(n, 16)", "(16, 8)", "(8,)", "(n, 8)", "z₂, a₂"), ("3 (إخراج)", "(n, 8)", "(8, 1)", "(1,)", "(n, 1)", "z₃, ŷ")],
                  ["num", "code", "code", "code", "code", "code"])
    definition("**التمرير الأمامي** `Forward pass`: حساب المخرج من المدخل عبر كل الطبقات بالترتيب. لا تُعدَّل معلمات. وهو نفسه ما يحدث في **الاستدلال**، مع فرق واحد: أثناء التدريب نحفظ الوسائط (z, a) لأجل الانتشار الخلفي، وأثناء الاستدلال لا نحتاجها (`no_grad`).")
    quiz("fwd.layers", [
        Q("مدخل `(64, 10)` عبر 10 → 32 → 5: شكل a₁…", ["(64, 32)", "(32,)", "(10, 32)"], 0, "الدفعة تبقى.", kind="shape"),
        Q("الرمز $W^{(3)}$ يعني…", ["W تكعيب", "أوزان الطبقة الثالثة", "W ثلاث مرات"], 1, "رقم طبقة."),
        Q("لماذا يستهلك التدريب ذاكرة أكثر من الاستدلال؟", ["لأن الأوزان أكبر", "لأننا نحفظ z وa لكل طبقة", "لأن الدفعة أكبر"], 1, "الوسائط للخلفي."),
    ])
    takeaway("z = aW + b ثم a = f(z)، طبقة بعد طبقة. المحور 0 ثابت، الأخير يتغير. احفظ z وa للخلفي.")
    lesson_footer(LESSON, ["حلقة واحدة على الطبقات.", "جدول الأشكال قبل التشغيل.", "الاستدلال = أمامي بلا حفظ."])
