import streamlit as st

from components.callouts import common_mistake, intuition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.python.list_vs_array_vs_tensor",
    title_ar="قائمة بايثون مقابل مصفوفة NumPy مقابل الموتر",
    title_en="Python List vs NumPy Array vs Tensor",
    module="foundations.python",
    order=10,
    prerequisites=["foundations.python.vectorization_broadcasting", "foundations.python.randomness_reproducibility"],
    objectives_ar=[
        "مقارنة الحاويات الثلاث من حيث النوع والسرعة والعمليات والجهاز والاشتقاق التلقائي.",
        "معرفة متى تحوّل بين القائمة والمصفوفة والموتر وبأي دالة.",
    ],
    terms=["tensor", "dtype", "shape"],
    difficulty="beginner",
    summary_ar="القائمة عامة وبطيئة، المصفوفة عددية وسريعة، الموتر مصفوفة + جهاز + اشتقاق تلقائي.",
)

CODE = '''import numpy as np

lst = [1, 2, 3]
arr = np.array([1, 2, 3])

print(lst * 2)          # القائمة: تكرار!
print(arr * 2)          # المصفوفة: ضرب عنصري
print(lst + [10, 20, 30])   # ضمّ قائمتين
print(arr + np.array([10, 20, 30]))   # جمع عنصري

mixed = [1, "a", 2.5]                  # قائمة تقبل أنواعًا مختلفة
print(type(mixed[1]).__name__)
print(np.array([1, 2.5]).dtype)        # المصفوفة ترفع الكل إلى نوع واحد

# التحويلات
back = arr.tolist()
print(type(back).__name__, back)
print(np.asarray([[1, 2], [3, 4]]).shape)

# ما يضيفه الموتر (شكل توضيحي بدون تشغيل الإطار):
#   torch.tensor(arr)  / tf.constant(arr)   ← نفس البيانات
#   tensor.to("cuda")  / with tf.device     ← الانتقال إلى GPU
#   requires_grad=True / tf.GradientTape    ← تسجيل العمليات للاشتقاق التلقائي'''


def render() -> None:
    lesson_header(LESSON)
    h2("ثلاث حاويات للأرقام", "Three containers for numbers")
    compare_table(
        ["الجانب", "Python list", "NumPy ndarray", "Tensor (TF / PyTorch)"],
        [("الأنواع", "مختلطة", "نوع واحد dtype", "نوع واحد dtype"),
         ("* 2", "تكرار القائمة", "ضرب عنصري", "ضرب عنصري"),
         ("السرعة", "بطيئة (حلقات بايثون)", "سريعة (C)", "سريعة، وعلى GPU متوازية"),
         ("shape / axis", "لا", "نعم", "نعم"),
         ("الجهاز", "CPU فقط", "CPU فقط", "CPU أو GPU"),
         ("الاشتقاق التلقائي", "لا", "لا", "نعم (Autograd / GradientTape)"),
         ("الاستخدام", "بيانات صغيرة، إعدادات", "تحضير البيانات، حسابات علمية", "داخل النموذج والتدريب")],
        ["rtl", "rtl", "rtl", "rtl"],
    )
    intuition("القائمة حقيبة يد، المصفوفة رف منظم، الموتر رف منظم موصول بمحرك (GPU) وبمسجّل يتذكر كيف صُنع كل رقم ليحسب التدرج لاحقًا.")
    code_lab(CodeLab(
        key="py_lat", title_ar="نفس البيانات، سلوك مختلف", code=CODE,
        before=Before(goal_ar="رؤية الفروق السلوكية بين القائمة والمصفوفة، والتحويل بينهما.", stage_ar="أساسيات بايثون ← التمثيل.",
                      inputs_ar="قائمة ومصفوفة من ثلاثة أرقام.", expected_ar="تكرار مقابل ضرب، ضمّ مقابل جمع، نوع مختلط مقابل نوع مرفوع."),
        explain=[("6-9", "أشهر فخ: `lst * 2` يكرر القائمة و`+` يضمّها؛ في المصفوفة كل شيء عنصري."),
                 ("11-13", "القائمة تقبل الخليط؛ المصفوفة ترفع `[1, 2.5]` إلى `float64` كي يصبح النوع واحدًا."),
                 ("16-18", "`tolist()` للعودة إلى بايثون، `asarray` للدخول إلى NumPy دون نسخ إن أمكن."),
                 ("20-23", "ما يضيفه الموتر — مذكور تعليقًا لأن الأطر تُقدَّم في وحدتها الخاصة بعد شرح ما يحدث داخلها.")],
        run=run_printed(CODE),
        after_ar="- `[1, 2, 3] * 2 = [1, 2, 3, 1, 2, 3]` ليس خطأ برمجيًا بل معنى مختلف؛ لذا لا نحسب على القوائم.\n- الرفع إلى `float64` هو سبب أن `NumPy` يعطي 64-bit افتراضيًا بينما الأطر تريد 32-bit.",
    ))
    practical_note("المسار العملي: `pandas` ← `to_numpy(float32)` ← `torch.tensor` / `tf.constant` أو مباشرة إلى `model.fit`. الأطر تقبل مصفوفات `NumPy` وتحوّلها داخليًا.")
    common_mistake("حساب متوسط قائمة بـ `sum(lst)/len(lst)` لبيانات كبيرة، أو تمرير قائمة من القوائم بأطوال مختلفة إلى `np.array` (يعطي `object` أو خطأ).")
    quiz("py.lat", [
        Q("`[1, 2] * 3` يعطي…", ["[3, 6]", "[1, 2, 1, 2, 1, 2]", "خطأ"], 1, "تكرار.", kind="output"),
        Q("ما الذي يميز الموتر عن مصفوفة NumPy؟", ["shape", "الجهاز والاشتقاق التلقائي", "dtype"], 1, "GPU + autograd."),
        Q("`np.array([1, 2.5]).dtype` هو…", ["int64", "float64", "object"], 1, "رفع إلى نوع واحد.", kind="output"),
    ])
    takeaway("قائمة للأشياء العامة، مصفوفة للحساب، موتر للنموذج (جهاز + تدرجات). الحساب لا يكون على القوائم.")
    lesson_footer(LESSON, ["* و+ لهما معنيان مختلفان في القائمة والمصفوفة.", "المصفوفة نوع واحد؛ الموتر مصفوفة + جهاز + autograd.", "pandas → NumPy(float32) → الإطار."])
