import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.linalg.tensors",
    title_ar="الموترات: الرتبة والشكل والمحاور وبُعد الدفعة والقناة",
    title_en="Tensors: Rank, Shape, Axes, Batch & Channel Dimensions",
    module="foundations.linalg",
    order=5,
    prerequisites=["foundations.linalg.matrix_multiplication", "foundations.data.data_modalities"],
    objectives_ar=[
        "تعميم العدد والمتجه والمصفوفة إلى الموتر برتبة عشوائية.",
        "قراءة أشكال موترات الجداول والسلاسل والصور والشبكات، وتحديد بُعد الدفعة والقناة.",
        "استخدام `reshape`, `transpose`, `expand_dims`, `squeeze` بأمان.",
    ],
    terms=["tensor", "rank", "batch_dimension", "axis"],
    labs=["labs.tensor_shape_explorer"],
    difficulty="intermediate",
    summary_ar="الموتر مصفوفة برتبة أي؛ المحور 0 الدفعة؛ الصور (n,H,W,C) في Keras و(n,C,H,W) في PyTorch.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)

scalar = np.array(3.5)                        # رتبة 0
vector = rng.normal(size=(4,))                # رتبة 1
matrix = rng.normal(size=(32, 4))             # رتبة 2: دفعة جدولية
series = rng.normal(size=(32, 30, 4))         # رتبة 3: دفعة سلاسل (n, T, f)
images = rng.integers(0, 256, size=(32, 28, 28, 3))   # رتبة 4: صور ملونة (n, H, W, C)
for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("series", series), ("images", images)]:
    print(f"{name:<7} ndim={t.ndim} shape={t.shape} size={t.size}")

one = matrix[0]                               # (4,)   ← فقدنا محور الدفعة
one_b = matrix[0:1]                           # (1, 4)
one_e = np.expand_dims(matrix[0], axis=0)     # (1, 4) بطريقة أخرى
print(one.shape, one_b.shape, one_e.shape, np.squeeze(one_b).shape)

pt_images = np.transpose(images, (0, 3, 1, 2))   # Keras (n,H,W,C) → PyTorch (n,C,H,W)
print(pt_images.shape)

flat = images.reshape(32, -1)                 # كل صورة متجه بطول 28*28*3
print(flat.shape)
print(images.mean(axis=(1, 2)).shape)         # متوسط كل قناة لكل صورة: (32, 3)'''


def render() -> None:
    lesson_header(LESSON)
    h2("التعميم", "The generalization")
    definition("**الموتر** `Tensor` مصفوفة متعددة المحاور: رتبة 0 عدد، 1 متجه، 2 مصفوفة، 3 فأكثر «موتر» ببساطة. **الرتبة** عدد المحاور، **الشكل** أطوالها. في أطر العمل كل شيء موتر — حتى العدد الواحد.")
    intuition("الموتر رتبة 3 هو رف من المصفوفات؛ رتبة 4 خزانة من الرفوف. الرقم الأول في الشكل يعدّ الرفوف (الدفعة)، وكل رقم بعده يعدّ داخل الرف.")
    compare_table(["الرتبة", "الاسم", "الشكل النموذجي", "المحور 0", "المحاور التالية", "مثال"],
                  [("0", "عدد", "()", "—", "—", "الخسارة"), ("1", "متجه", "(d,)", "الخصائص", "—", "ملاحظة واحدة"),
                   ("2", "مصفوفة", "(n, d)", "الدفعة", "الخصائص", "جدول"), ("3", "موتر", "(n, T, f)", "الدفعة", "الزمن، المتغيرات", "سلاسل زمنية / نص"),
                   ("4", "موتر", "(n, H, W, C)", "الدفعة", "ارتفاع، عرض، قنوات", "صور (Keras)"), ("5", "موتر", "(n, T, H, W, C)", "الدفعة", "الزمن ثم الصورة", "فيديو")],
                  ["num", "rtl", "code", "rtl", "rtl", "rtl"])
    h2("بُعد الدفعة وبُعد القناة", "Batch & channel dims")
    st.markdown("""
- **بُعد الدفعة** `Batch dimension`: المحور 0 دائمًا في مدخل النموذج. الأطر تتوقعه حتى لملاحظة واحدة `(1, ...)`.
- **بُعد القناة** `Channel dimension`: للصور، عدد «الطبقات اللونية» (1 رمادي، 3 RGB). موضعه اصطلاحي: **آخر محور** في `Keras` (`channels_last`)، **بعد الدفعة مباشرة** في `PyTorch` (`channels_first`).
- **بُعد الزمن**: للسلاسل، المحور 1 `(n, T, f)` في كلا الإطارين غالبًا.
""")
    research_note("ترتيب المحاور لا يغيّر البيانات بل تفسيرها. تمرير `(n, C, H, W)` إلى طبقة تتوقع `(n, H, W, C)` قد لا يعطي خطأ إن تطابقت الأطوال صدفة (مثلًا 3×3×3) — ويتدرب النموذج على هراء. `transpose` هو الأداة الصحيحة، لا `reshape`.")
    code_lab(CodeLab(
        key="la_tensors", title_ar="رتب من 0 إلى 4 وعمليات الشكل", code=CODE,
        before=Before(goal_ar="إنشاء موترات بخمس رتب، وإجراء العمليات الأربع على الشكل: إضافة محور، إزالته، تبديل ترتيب، وفرد.", stage_ar="جبر خطي ← تمثيل البيانات.",
                      inputs_ar="موترات عشوائية.", expected_ar="جدول رتبة/شكل، ثم `(4,) (1,4) (1,4) (4,)`، ثم `(32,3,28,28)`، `(32,2352)`، `(32,3)`."),
        explain=[("4-10", "خمس رتب. لاحظ أن `size` = حاصل ضرب الشكل: للصور 32×28×28×3 = 75264."),
                 ("12-15", "ثلاث طرق للتعامل مع محور الدفعة: الفهرسة تسقطه، التقطيع يبقيه، `expand_dims` يضيفه، `squeeze` يزيل المحاور ذات الطول 1."),
                 ("17-18", "`transpose` بترتيب محاور صريح: القناة من الموضع 3 إلى الموضع 1. البيانات تُعاد ترتيبها فعليًا."),
                 ("20-22", "`reshape(32, -1)` يفرد كل صورة لمتجه — ما تفعله طبقة `Flatten`. التجميع على محورين معًا `axis=(1, 2)` يعطي متوسط كل قناة.")],
        run=run_printed(CODE),
        after_ar="- `Flatten` على `(28, 28, 3)` يعطي 2352 — عدد مدخلات أول طبقة كثيفة بعده.\n- `expand_dims(axis=0)` هو الحل القياسي لخطأ «ملاحظة واحدة بلا دفعة».",
    ))
    common_mistake("استخدام `reshape` لتحويل `(n,H,W,C)` إلى `(n,C,H,W)`: الشكل يبدو صحيحًا لكن البكسلات تختلط. الصحيح `transpose`/`permute`.")
    st.button("افتح مستكشف أشكال الموترات", icon=":material/science:", type="primary", on_click=go, args=("labs.tensor_shape_explorer",), key="tensors_open_lab")
    quiz("la.tensors", [
        Q("`(16, 100, 6)` لسلاسل زمنية: ما المحور الزمني؟", ["0", "1", "2"], 1, "(n, T, f)."),
        Q("صورة رمادية واحدة 28×28 تُمرَّر إلى شبكة Keras؛ الشكل الصحيح؟", ["(28, 28)", "(1, 28, 28, 1)", "(28, 28, 1)"], 1, "دفعة + قناة."),
        Q("لتحويل (n,H,W,C) إلى (n,C,H,W) نستخدم…", ["reshape", "transpose(0, 3, 1, 2)", "squeeze"], 1, "إعادة ترتيب المحاور."),
        Q("`Flatten` على `(32, 32, 3)` يعطي متجهًا بطول…", ["96", "3072", "1024"], 1, "32×32×3.", kind="shape"),
    ])
    takeaway("الموتر = مصفوفة بأي رتبة. المحور 0 دفعة. القناة آخرًا في Keras وثانيًا في PyTorch. transpose لإعادة الترتيب، reshape للفرد فقط.")
    lesson_footer(LESSON, ["رتبة 0–4 وأشكالها النموذجية.", "expand_dims/squeeze/transpose/reshape.", "channels_last مقابل channels_first."])
