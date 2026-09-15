import streamlit as st

from components.callouts import takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.backprop.backpropagation.practice",
    title_ar="تمارين وخلاصة الانتشار الخلفي",
    title_en="Backpropagation: Practice & Summary",
    module="foundations.backprop",
    parent="foundations.backprop.backpropagation",
    order=8,
    prerequisites=["foundations.backprop.backpropagation.debugging"],
    objectives_ar=["حل تمارين أساسية ومتوسطة وتحدٍّ، مع تلميحات وحلول مخفية.", "خلاصة الوحدة في جدول واحد."],
    terms=["backpropagation"],
    difficulty="intermediate",
    summary_ar="تمارين على الأشكال والقواعد والتشخيص، وجدول خلاصة.",
)


def _exercise(title: str, body: str, hint: str, solution: str, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body)
        with st.expander("تلميح"):
            st.markdown(hint)
        with st.expander("الحل"):
            st.markdown(solution)


def render() -> None:
    lesson_header(LESSON)
    h2("تمارين", "Exercises")
    _exercise("أساسي — الأشكال", "شبكة 5 → 12 → 4 على دفعة 40. اكتب أشكال: δ₂، gW₂، gb₂، δ₁، gW₁، gb₁.",
              "δ لكل طبقة بالشكل (n, units). gW يطابق W. gb يطابق b.",
              "δ₂ (40, 4)، gW₂ (12, 4)، gb₂ (4,)، δ₁ (40, 12)، gW₁ (5, 12)، gb₁ (12,).", "ex1")
    _exercise("متوسط — الاختصار", "طبقة إخراج Softmax بثلاث فئات مع CCE، دفعة من ملاحظة واحدة: p = (0.2, 0.5, 0.3) والفئة الصحيحة 1. احسب δ.",
              "مع Softmax+CCE، ∂L/∂z = p − y حيث y one-hot.",
              "y = (0, 1, 0) ⇒ δ = (0.2, −0.5, 0.3). الفئة الصحيحة تحصل على δ سالب (يجب رفع z لها)، والأخريان موجب.", "ex2")
    _exercise("متوسط — البوابة", "طبقة مخفية ReLU بأربع خلايا، لملاحظة واحدة z₁ = (2, −1, 0.5, −3) وδ₂W₂ᵀ = (0.4, −0.8, 0.1, 0.9). احسب δ₁.",
              "اضرب عنصريًا في ReLU′(z) = 1 للموجب، 0 للسالب.",
              "ReLU′ = (1, 0, 1, 0) ⇒ δ₁ = (0.4, 0, 0.1, 0). خليتان لا تتعلمان من هذه الملاحظة.", "ex3")
    _exercise("تحدٍّ — التشخيص", "سجل معايير التدرج لخمس طبقات: [3e-7, 4e-5, 2e-3, 0.08, 0.6]. التنشيط tanh. ما التشخيص وما أول ثلاثة إصلاحات؟",
              "انظر إلى النسبة الأولى/الأخيرة وإلى التنشيط.",
              "نسبة 5e-7: تلاشٍ واضح. الإصلاحات: (1) ReLU/ELU بدل tanh في المخفية، (2) تهيئة He، (3) Batch Normalization؛ وللشبكات الأعمق وصلات متبقية. القصّ لا يساعد هنا.", "ex4")
    _exercise("تحدٍّ — الكود", "في `backward`، نسي أحدهم السطر `delta = delta * drelu(z_prev)`. أي التدرجات تبقى صحيحة وأيها يصبح خاطئًا؟ وكيف يكشفه التحقق العددي؟",
              "أي طبقة تُحسب بعد أول عودة عبر تنشيط.",
              "تدرجات الطبقة الأخيرة صحيحة (لا تمر عبر تنشيط مخفي). كل الطبقات السابقة خاطئة. التحقق العددي يعطي خطأ نسبيًا كبيرًا لكل الطبقات ما عدا الأخيرة — بصمة هذا الخطأ تحديدًا.", "ex5")
    h2("خلاصة الوحدة", "Module summary")
    table(["المفهوم", "الصيغة / القاعدة", "الخطأ الشائع"],
          [("δ للطبقة الأخيرة", "(p − y)/n مع Sigmoid+BCE أو Softmax+CCE؛ 2(ŷ−y)/n مع MSE", "نسيان /n"),
           ("انتقال δ", "(δ Wᵀ) ⊙ f′(z_prev)", "نسيان f′"), ("تدرج W", "a_prevᵀ δ", "استخدام a الحالية بدل السابقة"), ("تدرج b", "Σ_rows δ", "نسيان الجمع على الدفعة"),
           ("التحقق", "الفروق المنتهية، ε = 1e-5، خطأ نسبي < 1e-5", "ε غير مناسب"), ("التلاشي", "عوامل < 1 عبر العمق", "رفع معدل التعلم"), ("الانفجار", "عوامل > 1؛ قصّ بالمعيار", "clipvalue بدل clipnorm")],
          ["rtl", "code", "rtl"])
    quiz("bp.practice", [
        Q("شبكة 8 → 32 → 1 على دفعة 16: شكل gW₁…", ["(16, 32)", "(8, 32)", "(32, 8)"], 1, "يطابق W₁.", kind="shape"),
        Q("δ للفئة الصحيحة في Softmax+CCE…", ["موجب", "سالب (p − 1)", "صفر"], 1, "p < 1."),
        Q("نسيان f′ عند العودة يظهر في التحقق العددي كـ…", ["خطأ في الطبقة الأخيرة فقط", "خطأ في كل الطبقات ما عدا الأخيرة", "لا خطأ"], 1, "البصمة."),
    ])
    takeaway("أربع قواعد، اختبار واحد، مشكلتان. من يستطيع حل التمارين الخمسة يفهم loss.backward() فعلًا.")
    lesson_footer(LESSON, ["الأشكال أولًا.", "p − y اختصار الإخراج.", "التحقق العددي يكشف كل شيء."])
