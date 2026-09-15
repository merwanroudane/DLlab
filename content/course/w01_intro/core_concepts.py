import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="course.w01.core_concepts",
    title_ar="المفاهيم الأساسية: نموذج، بنية، تدريب، استدلال",
    title_en="Core Concepts: Model, Architecture, Training, Inference",
    module="course.w01",
    order=2,
    prerequisites=["course.w01.overview", "foundations.start.model_training_prediction"],
    objectives_ar=[
        "وصف الشبكة العصبية كسلسلة تحويلات من المدخل إلى المخرج.",
        "التمييز بين البنية (التصميم) والنموذج (البنية + المعلمات المدرَّبة).",
        "وصف دورة حياة النموذج: تصميم ← تدريب ← تقييم ← استدلال.",
    ],
    terms=["model", "parameter", "training", "inference", "deep_learning"],
    related=["foundations.start.ai_ml_dl"],
    difficulty="beginner",
    summary_ar="الشبكة سلسلة طبقات؛ البنية تصميم، النموذج بنية + معلمات مدرَّبة، والاستدلال استخدامه.",
)


def _layers_svg() -> str:
    s = '<svg viewBox="0 0 640 230" width="100%" style="max-width:640px">' + svg_defs()
    boxes = [("Input", "#E6F1FB", "#2F6FB5"), ("Layer 1", "#EFE9FA", "#7C5CBF"), ("Layer 2", "#EFE9FA", "#7C5CBF"),
             ("Output", "#DDF5EA", "#2E8B57")]
    x = 30
    for label, fill, stroke in boxes:
        s += svg_box(x, 80, 120, 60, label, fill, stroke=stroke, font=15, bold=True)
        x += 160
    for i in range(3):
        s += svg_arrow(30 + 120 + i * 160, 110, 30 + 160 * (i + 1) - 2, 110)
    s += svg_text(90, 165, "x  (features)", size=12, color="#2F6FB5")
    s += svg_text(250, 165, "z = W₁x + b₁ → f(z)", size=12, color="#5B4A88")
    s += svg_text(410, 165, "z = W₂a₁ + b₂ → f(z)", size=12, color="#5B4A88")
    s += svg_text(570, 165, "ŷ  (prediction)", size=12, color="#2E8B57")
    s += svg_text(320, 40, "Architecture = how many layers, how wide, which activations", size=13, bold=True)
    s += svg_text(320, 205, "Model = architecture + trained parameters (W₁, b₁, W₂, b₂)", size=13, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("الشبكة العصبية في جملة واحدة", "A neural network in one sentence")
    definition(
        "**الشبكة العصبية** سلسلة من **الطبقات**؛ كل طبقة تأخذ متجهًا، تضربه في مصفوفة أوزان، تضيف انحيازًا، "
        "وتمرر النتيجة عبر **دالة تنشيط** غير خطية. تراكب هذه الطبقات هو ما يجعلها «عميقة»."
    )
    intuition("كل طبقة تعيد وصف المدخل بلغة جديدة أكثر تجريدًا: من بكسلات إلى حواف، إلى أشكال، إلى «هذه فاتورة».")
    diagram(
        "من المدخل إلى التنبؤ عبر الطبقات",
        _layers_svg(),
        what_ar="أربعة صناديق: المدخل، طبقتان مخفيتان، المخرج. كل سهم = تحويل خطي متبوع بتنشيط.",
        how_ar="اقرأ من اليسار إلى اليمين. المعادلات أسفل الصناديق ستُشرح رمزًا رمزًا في وحدات الجبر الخطي والتمرير الأمامي.",
        takeaway_ar="البنية تصف الصناديق والأسهم؛ النموذج هو البنية بعد أن تعلمت قيم W و b.",
        title_en="Input → Layers → Output",
    )
    h2("بنية أم نموذج؟", "Architecture vs model")
    compare_table(
        ["", "البنية", "النموذج"],
        [
            ("English", "Architecture", "Model"),
            ("ما هي؟", "تصميم: عدد الطبقات، عرضها، دوال التنشيط، الروابط", "البنية + قيم المعلمات بعد التدريب"),
            ("متى تُحدد؟", "قبل التدريب (قرار الباحث)", "بعد التدريب"),
            ("مثال", "MLP بطبقتين مخفيتين من 64 وحدة", "نفس الـMLP بأوزان محددة تتنبأ بالتعثر بدقة 87%"),
        ],
        ["rtl", "rtl", "rtl"],
    )
    why("عندما تُبلّغ عن نتيجة بحثية يجب أن تذكر **البنية** (ليستطيع غيرك إعادة البناء) و**إجراء التدريب** (ليستطيع إعادة الوصول إلى نفس النموذج تقريبًا).")
    h2("دورة حياة النموذج", "Model lifecycle")
    pipeline(["Define problem", "Prepare data", "Design architecture", "Train", "Evaluate", "Diagnose & fix", "Inference"], active=3)
    st.markdown(
        """
1. **تعريف المسألة**: ما الهدف؟ انحدار أم تصنيف؟
2. **إعداد البيانات**: تنظيف، ترميز، تحجيم، تقسيم (تدريب/تحقق/اختبار).
3. **تصميم البنية**: عدد الطبقات والوحدات ودوال التنشيط.
4. **التدريب**: تقليل الخسارة بتحديث المعلمات دفعة بعد دفعة، حقبة بعد حقبة.
5. **التقييم**: قياس الأداء على بيانات لم يرها النموذج.
6. **التشخيص والإصلاح**: قراءة المنحنيات، اكتشاف فرط التخصيص أو قصور التعلم، تعديل ما يلزم.
7. **الاستدلال**: استخدام النموذج المجمّد على بيانات جديدة.
"""
    )
    common_mistake("تقييم النموذج على بيانات التدريب نفسها وإعلان دقة 99%. هذا يقيس الحفظ لا التعلم. التقييم يكون على بيانات محجوزة.")
    quiz(
        "w01.core",
        [
            Q("«MLP بثلاث طبقات مخفية من 128 وحدة مع ReLU» يصف…", ["نموذجًا مدرَّبًا", "بنية", "خوارزمية تحسين"], 1, "هذا تصميم بلا معلمات مدرَّبة."),
            Q("ما الذي يميز النموذج عن البنية؟", ["عدد الطبقات", "قيم المعلمات المدرَّبة", "نوع البيانات"], 1, "النموذج = بنية + معلمات."),
            Q("في أي مرحلة تُقاس القدرة على التعميم؟", ["التدريب", "التقييم على بيانات محجوزة", "تصميم البنية"], 1, "بيانات لم يرها النموذج."),
        ],
    )
    takeaway("شبكة = طبقات متراكبة. بنية = تصميم؛ نموذج = بنية + معلمات مدرَّبة. القيّم على بيانات محجوزة.")
    lesson_footer(LESSON, [
        "الطبقة: ضرب في أوزان + انحياز + تنشيط غير خطي.",
        "البنية قرار الباحث قبل التدريب؛ النموذج نتيجة التدريب.",
        "دورة حياة من سبع مراحل، والتشخيص جزء أصيل منها.",
    ])
