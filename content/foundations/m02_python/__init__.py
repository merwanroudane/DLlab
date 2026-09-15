from core.models import Module

MODULE = Module(
    id="foundations.python",
    section="foundations",
    title_ar="الوحدة 2 — بايثون للتعلم العميق",
    title_en="Module 2 — Python Foundations for Deep Learning",
    order=2,
    icon=":material/code:",
    prerequisites=["foundations.data"],
    purpose_ar=(
        "الحد الأدنى من بايثون و`NumPy` و`pandas` الذي يحتاجه الباحث ليقرأ ويكتب كود التعلم العميق بفهم: "
        "المتغيرات والأنواع، القوائم والقواميس، الفهرسة والتقطيع، الدوال، ثم المصفوفات والبث والعشوائية والتكرارية."
    ),
    why_ar=(
        "كل ما ستراه في `Keras` و`PyTorch` هو بايثون: كائنات، دوال بوسائط، مصفوفات لها `shape` و`dtype`. "
        "من لا يفهم `X[:, 0]` أو البث لن يفهم لماذا فشل سطر في النموذج."
    ),
    objectives_ar=[
        "قراءة وكتابة كود بايثون أساسي: متغيرات، قوائم، قواميس، شروط، حلقات، دوال.",
        "التعامل مع `ndarray`: الإنشاء، الشكل، النوع، الفهرسة، التقطيع، التجميع على محور.",
        "فهم البث `Broadcasting` وتوقّع شكل النتيجة قبل التشغيل.",
        "تثبيت البذرة العشوائية وفهم حدود التكرارية.",
    ],
    challenges_ar=["الخلط بين القائمة والمصفوفة.", "توقّع أن البث «يصلح» أي شكلين.", "نسيان أن التقطيع لا ينسخ البيانات في NumPy."],
)

LESSON_MODULES = [
    "variables_types",
    "collections",
    "indexing_slicing",
    "control_flow",
    "functions_imports",
    "numpy_ndarray",
    "vectorization_broadcasting",
    "pandas_basics",
    "randomness_reproducibility",
    "list_vs_array_vs_tensor",
]
