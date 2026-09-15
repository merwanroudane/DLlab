import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.backprop.backpropagation",
    title_ar="الانتشار الخلفي: لماذا، ومراجعة الأمامي والخسارة والمشتقات المحلية",
    title_en="Backpropagation: Why, and a Review of Forward, Loss & Local Derivatives",
    module="foundations.backprop",
    order=1,
    prerequisites=["foundations.calculus.chain_rule", "foundations.forward.forward_pass_code", "foundations.loss.cce_sparse"],
    objectives_ar=["تحديد المشكلة: نحتاج ∂L/∂θ لملايين المعلمات بكفاءة.", "مراجعة المكونات الثلاثة التي يبنى عليها: الأمامي، الخسارة، المشتقات المحلية.", "خريطة الصفحات الفرعية للوحدة."],
    terms=["backpropagation", "gradient", "chain_rule"],
    labs=["labs.chain_rule_lab"],
    difficulty="intermediate",
    summary_ar="المشكلة: تدرج الخسارة لكل معلمة. الحل: قاعدة السلسلة إلى الخلف مع إعادة استخدام الوسائط. هذه الصفحة الأم؛ التفاصيل في الصفحات الفرعية.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("لماذا الانتشار الخلفي؟", "Why backpropagation?")
    why("لدينا الخسارة $L(\\theta)$ ونريد $\\nabla_\\theta L$ لكل معلمة كي نحدّثها. الفروق المنتهية تحتاج تمريرتين أماميتين **لكل معلمة**: مليون معلمة = مليونا تمريرة لكل خطوة. الانتشار الخلفي يحسبها كلها بتمريرة أمامية واحدة وتمريرة خلفية واحدة.")
    definition("**الانتشار الخلفي** `Backpropagation`: خوارزمية تحسب $\\partial L / \\partial \\theta$ لكل معلمة بتطبيق قاعدة السلسلة على الرسم الحسابي من الخسارة إلى الخلف، **مع إعادة استخدام** المشتقات الوسيطة. لا يحدّث المعلمات — المحسّن يفعل ذلك بعده.")
    pipeline(["Forward: X → z₁ → a₁ → … → ŷ → L", "Backward: ∂L/∂ŷ → ∂L/∂z_L → ∂L/∂W_L … → ∂L/∂W₁", "Optimizer: θ ← θ − η∇L"], active=1)
    intuition("الأمامي يسأل «ما التنبؤ؟». الخلفي يسأل «من المسؤول عن الخطأ وبكم؟» ويوزّع اللوم إلى الخلف: من الخسارة إلى المخرج، إلى آخر طبقة، إلى ما قبلها… كل طبقة تأخذ نصيبها من اللوم وتمرر الباقي.")
    h2("المكونات الثلاثة (مراجعة)", "The three ingredients (review)")
    compare_table(["المكوّن", "من أي درس", "ما نحتاجه هنا"],
                  [("التمرير الأمامي", "الوحدة 12", "القيم الوسيطة z وa لكل طبقة (محفوظة في cache)"), ("الخسارة", "الوحدة 13", "مشتقتها بالنسبة للمخرج: MSE → 2(ŷ−y)/n؛ Sigmoid+BCE أو Softmax+CCE → (p−y)/n بالنسبة لـ z"),
                   ("المشتقات المحلية", "الوحدة 5", "مشتقة كل عملية: التنشيط f'(z)، الطبقة الخطية (∂z/∂W = a_prev, ∂z/∂a_prev = W)")],
                  ["rtl", "rtl", "rtl"])
    equation(r"\frac{\partial L}{\partial W^{(\ell)}} = \frac{\partial L}{\partial z^{(\ell)}}\cdot\frac{\partial z^{(\ell)}}{\partial W^{(\ell)}}, \qquad \frac{\partial L}{\partial z^{(\ell)}} = \frac{\partial L}{\partial a^{(\ell)}}\cdot f'\big(z^{(\ell)}\big), \qquad \frac{\partial L}{\partial a^{(\ell-1)}} = \frac{\partial L}{\partial z^{(\ell)}}\cdot\frac{\partial z^{(\ell)}}{\partial a^{(\ell-1)}}",
             [(r"\partial L/\partial z^{(\ell)}", "«الخطأ» عند الطبقة $\\ell$، يُرمز له $\\delta^{(\\ell)}$: الكمية المركزية التي تنتقل إلى الخلف."), (r"\partial z/\partial W", "المدخل $a^{(\\ell-1)}$ (مشتقة $aW + b$ بالنسبة لـ $W$)."), (r"\partial z/\partial a^{(\ell-1)}", "الأوزان $W^{(\\ell)}$: كيف يصل الخطأ إلى الطبقة السابقة.")],
             meaning_ar="ثلاث قواعد سلسلة تتكرر لكل طبقة. الصفحات الفرعية تطبقها بالمصفوفات والأرقام والكود.",
             example_ar="طبقة أخيرة بـ Softmax+CCE: $\\delta^{(L)} = (p - y)/n$، ثم $\\partial L/\\partial W^{(L)} = a^{(L-1)\\mathsf T}\\delta^{(L)}$.",
             dl_link_ar="هذا ما ينفذه الاشتقاق التلقائي في الأطر لأي رسم حسابي، لا للطبقات الكثيفة فقط.", title_ar="القواعد الثلاث")
    h2("خريطة الوحدة", "Module map")
    st.markdown("""
1. **تدفق التدرج** — δ والمصفوفات: كيف ينتقل الخطأ طبقة إلى الخلف.
2. **مثال خلية واحدة** — بالأرقام.
3. **مثال طبقتين** — بالمصفوفات والأرقام وتحريك أمامي/خلفي.
4. **التنفيذ في بايثون** — `backward` كامل + تحقق عددي.
5. **التلاشي والانفجار** — مشكلتان بإطار كامل.
6. **تشخيص التدرجات** — معايير التدرج، التحقق العددي، القصّ.
7. **تمارين وخلاصة**.
""")
    common_mistake("«الانتشار الخلفي يحدّث الأوزان». لا: يحسب التدرجات فقط. التحديث خطوة منفصلة للمحسّن — لذلك يمكن تجميع التدرجات على عدة دفعات قبل تحديث واحد، ولذلك يجب تصفير التدرجات في PyTorch قبل كل `backward`.")
    quiz("bp.overview", [
        Q("لماذا لا نستخدم الفروق المنتهية لتدريب الشبكات؟", ["غير دقيقة", "تمريرتان لكل معلمة: مكلفة جدًا", "لا تعمل مع ReLU"], 1, "الكلفة."),
        Q("الكمية التي تنتقل إلى الخلف عبر الطبقات هي…", ["W", "δ = ∂L/∂z", "a"], 1, "الخطأ عند الطبقة."),
        Q("الانتشار الخلفي…", ["يحدّث المعلمات", "يحسب التدرجات فقط", "يحسب الخسارة"], 1, "المحسّن يحدّث."),
    ])
    takeaway("الانتشار الخلفي = قاعدة السلسلة إلى الخلف مع إعادة استخدام δ. ثلاثة مكونات: cache الأمامي، مشتقة الخسارة، المشتقات المحلية. لا يحدّث شيئًا.")
    lesson_footer(LESSON, ["المشكلة: ∇L لكل معلمة بكفاءة.", "δ الكمية المركزية.", "الصفحات الفرعية تفصّل كل خطوة."])
