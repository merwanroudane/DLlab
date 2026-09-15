import streamlit as st

from components.callouts import intuition, research_note, takeaway, why
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.start.why_data_math_optimization",
    title_ar="لماذا يحتاج التعلم العميق إلى بيانات + رياضيات + تحسين؟",
    title_en="Why DL Needs Data + Math + Optimization",
    module="foundations.start",
    order=6,
    prerequisites=["foundations.start.model_training_prediction"],
    objectives_ar=[
        "فهم الأركان الثلاثة التي يقوم عليها أي نظام تعلم عميق.",
        "معرفة أي فرع رياضي يخدم أي جزء من الشبكة.",
        "ربط الأركان الثلاثة بأقسام أكاديمية الأسس.",
    ],
    terms=["gradient", "loss", "optimizer", "tensor"],
    difficulty="beginner",
    summary_ar="البيانات هي الخبرة، الرياضيات هي اللغة، والتحسين هو محرك التعلم.",
)


def _triad_svg() -> str:
    s = '<svg viewBox="0 0 640 300" width="100%" style="max-width:640px">' + svg_defs()
    s += svg_box(20, 110, 170, 80, "Data", "#E6F1FB", stroke="#2F6FB5", font=18, bold=True)
    s += svg_box(235, 110, 170, 80, "Mathematics", "#DDF5EA", stroke="#2E8B57", font=18, bold=True)
    s += svg_box(450, 110, 170, 80, "Optimization", "#EFE9FA", stroke="#7C5CBF", font=18, bold=True)
    s += svg_arrow(190, 150, 232, 150) + svg_arrow(405, 150, 447, 150)
    s += svg_text(105, 215, "examples (x, y)", size=12, color="#4A6B8A")
    s += svg_text(105, 232, "tensors, shapes", size=12, color="#4A6B8A")
    s += svg_text(320, 215, "z = Wx + b, loss L", size=12, color="#3F6E52")
    s += svg_text(320, 232, "derivatives, gradients", size=12, color="#3F6E52")
    s += svg_text(535, 215, "w ← w − η ∇L", size=12, color="#5B4A88")
    s += svg_text(535, 232, "SGD, Adam, epochs", size=12, color="#5B4A88")
    s += svg_text(320, 50, "Deep Learning = learn parameters from data by minimizing a loss", size=14, bold=True)
    s += svg_text(320, 275, "Data provides the signal · Math defines the model & loss · Optimization moves the parameters", size=12, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("الأركان الثلاثة", "The three pillars")
    diagram(
        "بيانات ← رياضيات ← تحسين",
        _triad_svg(),
        what_ar="ثلاثة صناديق: البيانات تدخل إلى صياغة رياضية (نموذج + خسارة)، ثم يحرك التحسين المعلمات.",
        how_ar="اقرأ من اليسار إلى اليمين كأنه خط إنتاج: بدون بيانات لا إشارة، بدون رياضيات لا نموذج ولا خسارة، بدون تحسين لا تعلّم.",
        takeaway_ar="لا يمكن حذف ركن. ضعف أحدها يظهر كمشكلة في الآخر (مثلًا بيانات سيئة تبدو كأنها فشل في التحسين).",
        title_en="Data → Math → Optimization",
    )

    h2("الركن الأول: البيانات", "Pillar 1 — Data")
    st.markdown(
        """
النموذج لا «يفهم» شيئًا؛ هو يستخرج أنماطًا إحصائية من الأمثلة. لذلك:

- **كمية** البيانات تحدد كم معلمة يمكن تعلّمها دون حفظ (`Memorization`).
- **جودة** البيانات (تسميات صحيحة، لا تسريب، لا تكرار) تحدد سقف الأداء.
- **تمثيل** البيانات كـ**موترات** `Tensors` بأشكال `Shape` صحيحة هو أول ما يُكسر عمليًا.
"""
    )
    intuition("شبكة عملاقة مدرَّبة على بيانات خاطئة تتعلم الخطأ بإتقان. «البيانات أولًا» ليست شعارًا، بل ترتيب التشخيص الفعلي.")

    h2("الركن الثاني: الرياضيات", "Pillar 2 — Mathematics")
    table(
        ["الفرع", "Branch", "ماذا يخدم داخل الشبكة؟", "وحدة الأسس"],
        [
            ("الجبر الخطي", "Linear Algebra", "المدخلات والأوزان مصفوفات؛ الطبقة = ضرب مصفوفات + جمع", "الوحدة 4"),
            ("التفاضل", "Calculus", "المشتقة تخبرنا كيف تتغير الخسارة عند تغيير وزن؛ التدرج والقاعدة السلسلية = الانتشار الخلفي", "الوحدة 5"),
            ("الاحتمال والإحصاء", "Probability & Statistics", "دوال الخسارة للتصنيف (الإنتروبيا المتقاطعة)، Softmax، التعميم، الضوضاء", "الوحدة 6"),
            ("الرياضيات الأساسية", "Basic math", "الدوال، الأسس، اللوغاريتم، رمز المجموع Σ", "الوحدة 3"),
        ],
        ["rtl", "ltr", "rtl", "rtl"],
    )
    why(
        "لماذا لا نكتفي بأطر العمل التي «تحسب كل شيء»؟ لأن `Keras` تحسب التدرجات نيابة عنك لكنها لا تخبرك "
        "لماذا أصبحت الخسارة `NaN` أو لماذا اختفى التدرج في الطبقات الأولى. التشخيص يحتاج الرياضيات."
    )

    h2("الركن الثالث: التحسين", "Pillar 3 — Optimization")
    st.markdown(
        """
التحسين هو **محرك التعلم**: خوارزمية تأخذ التدرج (اتجاه زيادة الخسارة) وتحرك المعلمات في الاتجاه المعاكس:

$$w_{t+1} = w_t - \\eta \\, \\frac{\\partial L}{\\partial w_t}$$

- $w_t$ الوزن الحالي، $w_{t+1}$ الوزن بعد التحديث.
- $\\eta$ معدل التعلم `Learning Rate`: معلمة فائقة يحددها الباحث — حجم الخطوة.
- $\\partial L / \\partial w_t$ التدرج: كيف تتغير الخسارة إذا غيّرنا هذا الوزن قليلًا.

كل ما ستراه لاحقًا (`SGD`, `Momentum`, `RMSprop`, `Adam`, جداول معدل التعلم) هو تنويعات على هذا السطر.
"""
    )
    research_note(
        "الشبكات العصبية دوال **غير محدبة**؛ لا ضمان للوصول إلى الحد الأدنى العام. عمليًا نصل إلى حدود محلية "
        "جيدة بما يكفي. لهذا تختلف النتائج بين تشغيلين بنفس الكود إذا اختلفت البذرة العشوائية."
    )

    h2("كيف تظهر المشكلة عندما يضعف ركن؟", "Failure signatures")
    table(
        ["الركن الضعيف", "ما يراه الباحث", "التشخيص الصحيح"],
        [
            ("بيانات", "الخسارة تنخفض على التدريب ولا تتحسن على التحقق", "تسريب/قلة بيانات/عدم تمثيل، وليس «الشبكة سيئة»"),
            ("رياضيات (تمثيل)", "خطأ `ValueError: Shapes ... are incompatible`", "فهم `shape` و`rank` وبُعد الدفعة"),
            ("تحسين", "الخسارة تتذبذب أو تنفجر إلى `NaN`", "معدل تعلم كبير، أو تدرج منفجر، أو مقياس مدخلات خاطئ"),
        ],
        ["rtl", "rtl", "rtl"],
    )
    quiz(
        "start.why_three",
        [
            Q("أي فرع رياضي يفسر «كيف تتغير الخسارة عند تغيير وزن واحد»؟",
              ["الجبر الخطي", "التفاضل", "الإحصاء الوصفي"], 1, "المشتقة الجزئية والتدرج."),
            Q("في $w \\leftarrow w - \\eta \\nabla L$، ما دور $\\eta$؟",
              ["اتجاه التحديث", "حجم الخطوة", "عدد الحقب"], 1, "معدل التعلم يحدد حجم الخطوة."),
            Q("الخسارة تنخفض على التدريب ولا تتحسن على التحقق. أول ما تفحصه؟",
              ["زيادة عدد الطبقات", "البيانات والتقسيم والتسريب", "تغيير المحسّن"], 1, "البيانات أولًا."),
        ],
    )
    takeaway("البيانات إشارة، الرياضيات لغة، التحسين محرك. التشخيص يبدأ من البيانات ثم التمثيل ثم التحسين.")
    lesson_footer(LESSON, [
        "ثلاثة أركان لا يُحذف أحدها: بيانات، رياضيات، تحسين.",
        "الجبر الخطي للطبقات، التفاضل للتدرجات، الاحتمال للخسائر والتعميم.",
        "قاعدة التحديث الواحدة w ← w − η∇L هي أصل كل المحسّنات.",
    ])
