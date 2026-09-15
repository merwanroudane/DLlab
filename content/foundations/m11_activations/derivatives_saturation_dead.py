import streamlit as st

from components.callouts import takeaway
from components.comparison import compare_table
from components.diagnostics import Problem, problem_card
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.activations.derivatives_saturation_dead",
    title_ar="المشتقات والمديات: تشخيص الإشباع وموت ReLU",
    title_en="Derivatives & Ranges: Diagnosing Saturation and Dead ReLU",
    module="foundations.activations",
    order=6,
    prerequisites=["foundations.activations.relu_family", "foundations.activations.softmax_output"],
    objectives_ar=["جمع مشتقات ومديات كل الدوال في جدول واحد.", "تشخيص تلاشي التدرج بسبب الإشباع وموت ReLU بإطار المشكلة الكامل.", "التدرب على المعمل التفاعلي."],
    terms=["derivative", "gradient", "norm"],
    labs=["labs.activation_lab"],
    difficulty="intermediate",
    summary_ar="جدول المشتقات + مشكلتان مُشخَّصتان بالكامل: تلاشي التدرج بالإشباع، وموت ReLU.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الجدول المرجعي", "Reference table")
    compare_table(["الدالة", "f(z)", "f'(z)", "المدى", "f'(z) القصوى", "المشكلة النموذجية"],
                  [("Sigmoid", "1/(1+e^{−z})", "σ(1−σ)", "(0,1)", "0.25", "إشباع → تلاشٍ"), ("tanh", "tanh z", "1−tanh²", "(−1,1)", "1", "إشباع → تلاشٍ (أخف)"),
                   ("ReLU", "max(0,z)", "1 أو 0", "[0,∞)", "1", "موت"), ("Leaky ReLU", "z أو αz", "1 أو α", "ℝ", "1", "—"), ("ELU", "z أو α(e^z−1)", "1 أو αe^z", "(−α,∞)", "1", "—"),
                   ("Softmax", "e^{z_k}/Σ", "p_k(δ_kj − p_j)", "احتمالات", "—", "تنشيط مزدوج"), ("خطي", "z", "1", "ℝ", "1", "انهيار الطبقات")],
                  ["ltr", "code", "code", "code", "num", "rtl"])
    problem_card(Problem(
        key="vanishing_saturation", name_ar="تلاشي التدرج بسبب الإشباع", name_en="Vanishing gradient (saturation)",
        description_ar="تنشيطات مشبعة (Sigmoid/tanh) في عدة طبقات تضرب التدرج في أرقام صغيرة متتالية فيصل إلى الطبقات الأولى شبه معدوم.",
        symptoms_ar=["خسارة التدريب تهبط ببطء شديد ثم تستقر مرتفعة.", "الطبقات الأولى لا تتغير أوزانها تقريبًا.", "زيادة معدل التعلم لا تفيد أو تنفجر."],
        sees_ar=["معيار التدرج للطبقة الأولى أصغر بمراتب من الأخيرة.", "`loss` مستوية من الحقبة الأولى.", "تنشيطات الطبقات المخفية قرب 0 أو 1 (Sigmoid)."],
        possible_causes_ar=["Sigmoid/tanh في طبقات مخفية متعددة.", "تهيئة أوزان كبيرة تدفع z إلى الإشباع.", "مدخلات غير محجّمة تعطي z ضخمة."],
        root_causes_ar=["حاصل ضرب مشتقات ≤ 0.25 عبر L طبقات (قاعدة السلسلة)."],
        diagnosis_ar=["اطبع معيار التدرج لكل طبقة؛ ابحث عن تناقص أسّي من الأخيرة إلى الأولى.", "افحص توزيع تنشيطات كل طبقة (هيستوغرام): تكدس عند الحدود = إشباع.", "جرّب نفس البنية بـ ReLU وقارن سرعة الهبوط."],
        evidence_ar=["استبدال التنشيط بـ ReLU يحل المشكلة فورًا.", "تحجيم المدخلات يحرك z نحو الوسط."],
        fixes_ar=["ReLU وعائلته في المخفية.", "تهيئة He/Glorot.", "تحجيم المدخلات.", "Batch Normalization (الوحدة 20).", "وصلات متبقية في الشبكات العميقة جدًا (تعمّق)."],
        misdiagnosis_ar=["«معدل التعلم صغير» فيُرفع وتنفجر الطبقة الأخيرة.", "«البيانات غير كافية»."],
        related_ar=["انفجار التدرج", "موت ReLU", "التهيئة"],
        checklist_ar=["أي تنشيط في المخفية؟", "كم طبقة؟", "المدخلات محجّمة؟", "معيار التدرج لكل طبقة مطبوع؟"],
        challenge=[Q("شبكة 6 طبقات Sigmoid، الخسارة ثابتة عند ln 2 منذ البداية. أول تغيير تجرّبه؟", ["زيادة الحقب", "استبدال Sigmoid بـ ReLU في المخفية", "تقليل حجم الدفعة"], 1, "الإشباع هو السبب المرجح.", kind="scenario")],
    ))
    problem_card(Problem(
        key="dead_relu", name_ar="موت ReLU", name_en="Dead ReLU",
        description_ar="وحدات ReLU يصبح مجموعها الموزون سالبًا لكل المدخلات فتخرج صفرًا وتتلقى تدرجًا صفريًا إلى الأبد.",
        symptoms_ar=["نسبة كبيرة من الوحدات بمخرج 0 دائمًا.", "قدرة الشبكة الفعلية أصغر من المصممة؛ قصور تعلم غير متوقع.", "تحدث غالبًا فجأة بعد خطوات أولى بمعدل تعلم كبير."],
        sees_ar=["هيستوغرام تنشيط طبقة: كتلة عند الصفر بالضبط.", "`(activations == 0).mean()` > 0.5 لطبقة ما.", "أوزان/انحيازات بعض الوحدات لا تتغير بعد الحقبة الثانية."],
        possible_causes_ar=["معدل تعلم كبير دفع الانحيازات بعيدًا في السالب.", "تهيئة سيئة.", "مدخلات بمقياس كبير."],
        root_causes_ar=["مشتقة ReLU صفر للسالب: لا آلية تعافٍ."],
        diagnosis_ar=["احسب نسبة الوحدات الصفرية لكل طبقة على دفعة تحقق.", "قارنها في بداية التدريب وبعده: القفزة تدل على معدل تعلم كبير.", "جرّب Leaky ReLU بنفس الإعدادات."],
        evidence_ar=["نسبة الصفرية تنخفض مع معدل تعلم أصغر أو Leaky ReLU."],
        fixes_ar=["خفض معدل التعلم (خاصة في البداية: warm-up).", "Leaky ReLU / ELU / GELU.", "تهيئة He.", "تحجيم المدخلات."],
        tradeoffs_ar=["Leaky ReLU أبطأ قليلًا ونادرًا ما يُحدث فرقًا كبيرًا إن لم يكن هناك موت فعلًا."],
        misdiagnosis_ar=["«الشبكة صغيرة جدًا» فتُكبَّر وتموت وحدات أكثر."],
        related_ar=["تلاشي التدرج", "معدل التعلم", "التهيئة"],
        checklist_ar=["نسبة الوحدات الصفرية لكل طبقة؟", "معدل التعلم في البداية؟", "التهيئة He؟"],
        challenge=[Q("بعد 3 حقب، 70% من وحدات الطبقة الأولى تعطي 0 دائمًا. التشخيص؟", ["إشباع Sigmoid", "موت ReLU بمعدل تعلم كبير", "تسريب"], 1, "كتلة عند الصفر.", kind="scenario")],
    ))
    st.button("افتح معمل دوال التنشيط", icon=":material/science:", type="primary", on_click=go, args=("labs.activation_lab",), key="act_lab_btn")
    quiz("act.diag", [
        Q("معيار التدرج: الطبقة الأخيرة 1e-1، الأولى 1e-8، Sigmoid في المخفية…", ["انفجار", "تلاشٍ بالإشباع", "طبيعي"], 1, "تناقص أسّي."),
        Q("أفضل مؤشر لموت ReLU…", ["الخسارة ترتفع", "نسبة التنشيطات الصفرية لكل طبقة", "الدقة على الاختبار"], 1, "كتلة عند الصفر."),
    ])
    takeaway("المشتقة تخبرك بكل شيء: صغيرة (إشباع) = تلاشٍ؛ صفر (ReLU سالب) = موت. راقب معايير التدرج ونِسب الصفرية لكل طبقة.")
    lesson_footer(LESSON, ["جدول المشتقات والمديات.", "مشكلتان بإطار كامل.", "المعمل لرؤية كل هذا بالأرقام."])
