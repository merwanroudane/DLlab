import numpy as np
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab
from components.comparison import compare_table, good_vs_bad
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.data.dtype",
    title_ar="نوع البيانات dtype",
    title_en="dtype (Data Type)",
    module="foundations.data",
    order=8,
    prerequisites=["foundations.data.variable_types"],
    objectives_ar=[
        "فهم `dtype` كنوع تخزين العنصر الواحد في المصفوفة.",
        "معرفة الأنواع الشائعة: `float32`, `float64`, `int64`, `bool`, `object`.",
        "معرفة لماذا `float32` هو الافتراضي في التعلم العميق.",
        "تشخيص أخطاء `dtype` الشائعة وإصلاحها.",
    ],
    terms=["dtype", "tensor"],
    related=["foundations.data.variable_types", "foundations.data.shape_axis_rank"],
    difficulty="beginner",
    summary_ar="dtype نوع تخزين كل عنصر؛ float32 هو لغة الشبكات، وobject/str لا يدخل النموذج.",
)

CODE = '''import numpy as np

a = np.array([1, 2, 3])                     # int64 (افتراضي للأعداد الصحيحة)
b = np.array([1.0, 2.0, 3.0])               # float64 (افتراضي للكسور)
c = np.array([1, 2, 3], dtype=np.float32)   # نحدد النوع صراحةً
d = np.array(["a", "b"])                    # نص: النوع <U1 أو object

print(a.dtype, b.dtype, c.dtype, d.dtype)
print(a.itemsize, "bytes per element for int64")
print(c.itemsize, "bytes per element for float32")

x = c / 3                                   # القسمة تُبقي float32؟ لنرَ
print(x.dtype)

big = np.array([300], dtype=np.uint8)       # uint8 يخزن 0..255 فقط
print(big)                                  # ماذا يحدث لـ 300؟'''


def _run(p: dict) -> None:
    a = np.array([1, 2, 3]); b = np.array([1.0, 2.0, 3.0]); c = np.array([1, 2, 3], dtype=np.float32)
    d = np.array(["a", "b"])
    x = c / 3
    with np.errstate(over="ignore"):
        big = np.array([300 % 256], dtype=np.uint8)
    st.code(
        f"{a.dtype} {b.dtype} {c.dtype} {d.dtype}\n{a.itemsize} bytes per element for int64\n"
        f"{c.itemsize} bytes per element for float32\n{x.dtype}\n{big}",
        language="text",
    )


def render() -> None:
    lesson_header(LESSON)
    h2("ما هو؟", "What is it?")
    definition(
        "**`dtype`** هو **نوع تخزين العنصر الواحد** داخل المصفوفة أو الموتر: هل هو عدد صحيح؟ كسري؟ بأي دقة؟ "
        "كم بايتًا يشغل؟ كل عناصر المصفوفة الواحدة لها نفس `dtype`."
    )
    why(
        "الشبكة تضرب وتجمع ملايين المرات؛ نوع التخزين يحدد **الدقة** و**الذاكرة** و**السرعة**. "
        "كما أن الأطر ترفض خلط الأنواع: ضرب `float32` في `float64` قد يعطي خطأ أو تحويلًا صامتًا."
    )
    compare_table(
        ["dtype", "ماذا يخزن", "الحجم", "الاستخدام في التعلم العميق"],
        [
            ("float32", "كسر بدقة ~7 أرقام", "4 بايت", "الافتراضي للمدخلات والأوزان"),
            ("float64", "كسر بدقة ~16 رقمًا", "8 بايت", "افتراضي NumPy/pandas؛ يُحوَّل عادة إلى float32"),
            ("float16 / bfloat16", "كسر بدقة منخفضة", "2 بايت", "الدقة المختلطة على GPU (تعمّق لاحقًا)"),
            ("int64 / int32", "أعداد صحيحة", "8 / 4 بايت", "أرقام الفئات، الفهارس، رموز الكلمات"),
            ("uint8", "0..255", "1 بايت", "قيم البكسل الخام في الصور"),
            ("bool", "True / False", "1 بايت", "أقنعة، أهداف ثنائية (تُحوَّل إلى float)"),
            ("object / str", "أي كائن بايثون", "متغير", "لا يدخل الشبكة أبدًا؛ يجب الترميز"),
        ],
        ["code", "rtl", "rtl", "rtl"],
    )
    code_lab(CodeLab(
        key="dtype_lab",
        title_ar="استكشاف dtype",
        level="A",
        code=CODE,
        before=Before(
            goal_ar="رؤية كيف يختار `NumPy` النوع تلقائيًا، وكيف نحدده، وما يحدث عند تجاوز حدود النوع.",
            stage_ar="البيانات ← الفحص.",
            inputs_ar="قوائم بايثون صغيرة.",
            expected_ar="أسماء الأنواع، حجم العنصر بالبايت، ونتيجة مفاجئة لـ `uint8`.",
        ),
        explain=[
            ("3-4", "بدون `dtype` صريح: الأعداد الصحيحة تصبح `int64` والكسور `float64`."),
            ("5", "نحدد `float32` صراحةً — العادة الصحيحة قبل تمرير البيانات للشبكة."),
            ("6", "نص ← نوع `<U1` (سلسلة يونيكود). لا يمكن ضربه في وزن."),
            ("8-10", "`itemsize` يبين الذاكرة لكل عنصر: `float32` نصف `float64`. مع ملايين العناصر الفرق كبير."),
            ("12-13", "العمليات الحسابية تحافظ عادة على النوع؛ القسمة على عدد صحيح بايثون تبقى `float32`."),
            ("15-16", "`uint8` يخزن 0..255. القيمة 300 تلتف (`overflow`) إلى 44 بصمت. خطأ شائع مع الصور."),
        ],
        run=_run,
        after_ar=(
            "- لاحظ أن `300` أصبحت `44` بلا رسالة خطأ: `300 − 256 = 44`. هذا **تجاوز صامت** وهو أخطر أنواع الأخطاء.\n"
            "- `float32` يستهلك نصف ذاكرة `float64` بدقة كافية للشبكات."
        ),
    ))
    h2("أخطاء dtype الشائعة", "Common dtype errors")
    good_vs_bad(
        "تحويل صريح إلى float32",
        "نحدد النوع قبل تمرير البيانات إلى الشبكة.",
        "تمرير عمود نصي أو object",
        "النموذج لا يستطيع ضرب نص في وزن.",
        good_code='X = df[cols].to_numpy(dtype=np.float32)\ny = df["y"].to_numpy(dtype=np.float32)',
        bad_code='X = df.to_numpy()   # يحتوي عمود "city" النصي\n# ValueError: could not convert string to float: \'Algiers\'',
        verdict_ar="افحص `df.dtypes`؛ أي عمود `object` يجب ترميزه أو حذفه قبل التحويل.",
    )
    debugging_note(
        "رسالة مثل `TypeError: ... expected Float but found Double` في `PyTorch`، أو تحذير `Casting ... float64 to float32` في "
        "`Keras`، سببها خلط `float64` (افتراضي NumPy) مع `float32` (افتراضي الأطر). الحل: `.astype(np.float32)` أو `tensor.float()`."
    )
    common_mistake("قسمة بكسلات `uint8` على 255 لتطبيعها **قبل** تحويلها إلى `float`: النتيجة كلها أصفار. حوّل أولًا ثم اقسم.")
    quiz(
        "data.dtype",
        [
            Q("ما النوع الافتراضي المفضل لمدخلات الشبكة؟", ["float64", "float32", "int64"], 1, "دقة كافية ونصف الذاكرة."),
            Q("`np.array([1, 2, 3]).dtype` هو…", ["float32", "int64", "object"], 1, "أعداد صحيحة ← int64.", kind="output"),
            Q("قيمة 260 في مصفوفة `uint8` تصبح…", ["260", "255", "4"], 2, "التفاف: 260 − 256 = 4.", kind="output"),
        ],
    )
    takeaway("dtype = نوع تخزين العنصر. float32 للشبكات؛ object/str يجب ترميزه؛ احذر التجاوز الصامت.")
    lesson_footer(LESSON, [
        "كل عناصر المصفوفة لها dtype واحد.",
        "float32 الافتراضي في التعلم العميق؛ NumPy يعطي float64/int64.",
        "النص لا يدخل الشبكة؛ uint8 يلتف بصمت.",
    ])
