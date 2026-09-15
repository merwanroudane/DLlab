import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.python.collections",
    title_ar="القائمة والصف والقاموس",
    title_en="List, Tuple & Dictionary",
    module="foundations.python",
    order=2,
    prerequisites=["foundations.python.variables_types"],
    objectives_ar=[
        "استخدام القائمة `list` (قابلة للتعديل)، والصف `tuple` (ثابت)، والقاموس `dict` (مفتاح ← قيمة).",
        "التعرف على أين تظهر هذه البنى في أطر العمل: `shape` صف، `history.history` قاموس، طبقات قائمة.",
    ],
    terms=["shape"],
    difficulty="beginner",
    summary_ar="ثلاث حاويات: list للتسلسلات المتغيرة، tuple للثوابت مثل shape، dict للأسماء ← قيم.",
)

CODE = '''layers = [64, 32, 1]                  # list: عدد الوحدات في كل طبقة
layers.append(1)                      # القوائم قابلة للتعديل
layers[0] = 128
print(layers, len(layers))

shape = (120, 4)                      # tuple: ثابت لا يتغير
print(shape[0], shape[1], len(shape))
# shape[0] = 5   ← يعطي TypeError لأن الصف ثابت

history = {"loss": [0.9, 0.5, 0.3], "val_loss": [1.0, 0.6, 0.5]}   # dict
history["accuracy"] = [0.6, 0.8, 0.9]
print(history["loss"][-1])            # آخر قيمة خسارة
print(list(history.keys()))

config = {"lr": 0.01, "batch_size": 32, "optimizer": "adam"}
for key, value in config.items():
    print(key, "=", value)'''


def render() -> None:
    lesson_header(LESSON)
    h2("ثلاث حاويات", "Three containers")
    table(
        ["البنية", "Python", "قابلة للتعديل؟", "الوصول", "أين تراها في التعلم العميق"],
        [
            ("قائمة", "list  [ ]", "نعم", "بالموضع `a[0]`", "قائمة الطبقات، قيم الخسارة عبر الحقب"),
            ("صف", "tuple ( )", "لا", "بالموضع `t[0]`", "`shape = (120, 4)`، `input_shape=(4,)`"),
            ("قاموس", "dict { }", "نعم", "بالمفتاح `d[\"loss\"]`", "`history.history`، إعدادات النموذج"),
        ],
        ["rtl", "code", "rtl", "code", "rtl"],
    )
    definition("**القائمة** تسلسل مرتب قابل للتعديل. **الصف** تسلسل مرتب ثابت. **القاموس** يربط مفاتيح بقيم؛ الوصول بالاسم لا بالموضع.")
    code_lab(CodeLab(
        key="py_collections", title_ar="القائمة والصف والقاموس عمليًا", code=CODE,
        before=Before(goal_ar="إنشاء الحاويات الثلاث والتعامل معها كما في كود التدريب.", stage_ar="أساسيات بايثون.",
                      inputs_ar="قيم ثابتة.", expected_ar="قوائم معدّلة، عناصر صف، وقيم من قاموس."),
        explain=[("1-4", "القائمة تُعدَّل في مكانها: `append` يضيف، `layers[0] = 128` يستبدل."),
                 ("6-8", "الصف ثابت: مناسب لـ `shape` لأن الشكل لا يجب أن يتغير بالخطأ. `(4,)` صف من عنصر واحد — الفاصلة إلزامية."),
                 ("10-13", "القاموس يربط اسمًا بقيمة. `history[\"loss\"][-1]`: الفهرس `-1` يعني الأخير."),
                 ("15-17", "`items()` يعطي أزواج (مفتاح، قيمة) للتكرار عليها — نمط طباعة الإعدادات.")],
        run=run_printed(CODE),
        after_ar="- الفاصلة في `(4,)` هي ما يجعله صفًا؛ `(4)` مجرد رقم.\n- الفهرس السالب يعد من النهاية: `-1` الأخير، `-2` قبل الأخير.",
    ))
    common_mistake("كتابة `input_shape=(4)` بدل `(4,)`: الأول عدد صحيح وليس صفًا، فتفشل الطبقة.")
    practical_note("`History.history` في `Keras` قاموس مفاتيحه أسماء المقاييس وقيمه قوائم بطول عدد الحقب — بالضبط كمتغير `history` أعلاه.")
    quiz("py.collections", [
        Q("أي بنية مناسبة لـ `shape`؟", ["list", "tuple", "dict"], 1, "الشكل ثابت."),
        Q("`history[\"loss\"][-1]` يعطي…", ["أول خسارة", "آخر خسارة", "عدد الحقب"], 1, "-1 = الأخير.", kind="output"),
        Q("`(4)` هو…", ["صف من عنصر", "عدد صحيح", "قائمة"], 1, "بلا فاصلة ليس صفًا.", kind="code"),
    ])
    takeaway("list للمتغيّر، tuple للثابت (shape)، dict للأسماء ← قيم (history).")
    lesson_footer(LESSON, ["ثلاث حاويات بثلاثة استخدامات واضحة.", "(4,) صف؛ (4) رقم.", "الفهرس -1 = الأخير."])
