import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, warning_note
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.python.numpy_ndarray",
    title_ar="مصفوفات NumPy: ndarray والشكل والنوع",
    title_en="NumPy ndarray, Shape & dtype",
    module="foundations.python",
    order=6,
    prerequisites=["foundations.python.functions_imports", "foundations.data.shape_axis_rank"],
    objectives_ar=[
        "إنشاء مصفوفات بـ `array`, `zeros`, `ones`, `arange`, `linspace`, `random`.",
        "الفهرسة ثنائية البُعد `X[i, j]`, `X[:, j]`, `X[i]` والأقنعة المنطقية.",
        "التجميع على محور `sum/mean/max(axis=...)` وإعادة التشكيل `reshape`, `T`.",
    ],
    terms=["tensor", "shape", "dtype", "axis"],
    labs=["labs.dataset_anatomy"],
    difficulty="beginner",
    summary_ar="ndarray هو المصفوفة العددية الأساسية: إنشاء، فهرسة ثنائية، أقنعة، تجميع على محور.",
)

CODE = '''import numpy as np

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])                 # (2, 3)
print(X.shape, X.ndim, X.dtype, X.size)

print(np.zeros((2, 3)))                         # مصفوفة أصفار
print(np.ones(3))                               # متجه آحاد
print(np.arange(0, 10, 2))                      # 0,2,4,6,8
print(np.linspace(0, 1, 5))                     # 5 قيم متساوية البعد من 0 إلى 1

print(X[0, 2])                                  # صف 0، عمود 2
print(X[:, 1])                                  # العمود 1 لكل الصفوف → (2,)
print(X[1])                                     # الصف 1 → (3,)
mask = X > 2.5                                  # قناع منطقي
print(mask)
print(X[mask])                                  # العناصر التي تحقق الشرط → متجه

print(X.sum(axis=0), X.mean(axis=1), X.max())
print(X.T.shape, X.reshape(3, 2).shape, X.reshape(-1).shape)
print((X * 10).astype(np.int32))                # عملية على كل العناصر + تحويل النوع'''


def render() -> None:
    lesson_header(LESSON)
    h2("لماذا NumPy؟", "Why NumPy?")
    definition("**`ndarray`** مصفوفة متعددة المحاور من عناصر بنفس `dtype` مخزنة متجاورة في الذاكرة. العمليات عليها تُنفَّذ بكود مُجمَّع سريع على كل العناصر دفعة واحدة. كل الموترات في `TensorFlow` و`PyTorch` مبنية على نفس الفكرة.")
    intuition("القائمة صناديق منفصلة من أي نوع؛ المصفوفة شبكة منتظمة من نوع واحد. الانتظام هو ما يسمح بحساب ملايين العناصر في ميلي ثانية.")
    code_lab(CodeLab(
        key="py_ndarray", title_ar="إنشاء وفهرسة وتجميع", code=CODE,
        before=Before(goal_ar="تغطية العمليات الأكثر استخدامًا على المصفوفات في كود التعلم العميق.", stage_ar="أساسيات بايثون ← البيانات.",
                      inputs_ar="مصفوفة `(2, 3)`.", expected_ar="أشكال، مصفوفات منشأة، مقاطع، قناع منطقي، تجميعات على محاور."),
        explain=[("3-5", "`array` من قوائم متداخلة: كل قائمة داخلية صف. الخصائص الأربع الأساسية."),
                 ("7-10", "منشئات: أصفار/آحاد لتهيئة الأوزان أو الأقنعة، `arange` كـ `range`، `linspace` لعدد محدد من النقاط."),
                 ("12-14", "الفهرسة ثنائية البُعد `[صف, عمود]`. `:` تعني كل الصفوف. `X[:, 1]` عمود كخاصية واحدة لكل الملاحظات."),
                 ("15-17", "القناع المنطقي مصفوفة `bool` بنفس الشكل؛ `X[mask]` يختار العناصر الصحيحة — أساس التصفية."),
                 ("19", "`axis=0` يطوي الصفوف (نتيجة لكل عمود)، `axis=1` يطوي الأعمدة (نتيجة لكل صف)، بلا محور يعطي رقمًا واحدًا."),
                 ("20", "`T` المنقولة `(3, 2)`؛ `reshape` يعيد ترتيب نفس العناصر؛ `-1` يحسب البعد تلقائيًا."),
                 ("21", "العمليات الحسابية تطبَّق عنصرًا عنصرًا؛ `astype` يحوّل النوع (ينشئ نسخة).")],
        run=run_printed(CODE),
        after_ar="- `X[:, 1]` شكله `(2,)` لا `(2, 1)`: الفهرسة برقم واحد تُسقط ذلك المحور.\n- `X[mask]` يعطي متجهًا مهما كان شكل `X` لأن عدد العناصر المحققة غير منتظم.\n- `astype(np.int32)` يقطع الكسور.",
    ))
    table(
        ["العملية", "الكود", "الشكل الناتج من (n, d)"],
        [("عمود j", "X[:, j]", "(n,)"), ("صف i", "X[i]", "(d,)"), ("صف i كدفعة", "X[i:i+1]", "(1, d)"),
         ("متوسط كل خاصية", "X.mean(axis=0)", "(d,)"), ("متوسط كل ملاحظة", "X.mean(axis=1)", "(n,)"), ("المنقولة", "X.T", "(d, n)")],
        ["rtl", "code", "code"],
    )
    warning_note("`X[0:2]` عرض `view` على نفس الذاكرة؛ `X[mask]` و`astype` و`reshape` (غالبًا) نسخ. إن عدّلت عرضًا تعدّل الأصل.")
    common_mistake("`np.array([1, 2, 3]) + np.array([1, 2])` يعطي `ValueError: operands could not be broadcast` — الأطوال مختلفة. الدرس التالي يشرح متى تنجح العملية بين شكلين مختلفين.")
    quiz("py.ndarray", [
        Q("`X.shape == (50, 8)`؛ `X[:, 3].shape`؟", ["(50,)", "(8,)", "(50, 1)"], 0, "عمود واحد لكل الملاحظات.", kind="shape"),
        Q("`X.sum(axis=1)` على `(50, 8)` يعطي…", ["(8,)", "(50,)", "رقمًا"], 1, "المحور 1 يختفي.", kind="shape"),
        Q("`np.arange(0, 1, 0.25)` يعطي كم عنصرًا؟", ["4", "5", "3"], 0, "0, .25, .5, .75 — النهاية غير مشمولة.", kind="output"),
    ])
    takeaway("ndarray: نوع واحد، شبكة منتظمة، عمليات على الكل. الفهرسة [صف, عمود]، الأقنعة للتصفية، axis للتجميع.")
    lesson_footer(LESSON, ["array/zeros/ones/arange/linspace.", "X[:, j] عمود، X[i] صف، X[mask] تصفية.", "axis المذكور يختفي."])
