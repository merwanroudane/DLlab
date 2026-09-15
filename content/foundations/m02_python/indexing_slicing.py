import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, warning_note
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.python.indexing_slicing",
    title_ar="الفهرسة والتقطيع",
    title_en="Indexing & Slicing",
    module="foundations.python",
    order=3,
    prerequisites=["foundations.python.collections"],
    objectives_ar=[
        "قراءة `a[i]` و`a[start:stop:step]` وفهم أن `stop` غير مشمول.",
        "استخدام الفهارس السالبة والتقطيع لبناء الدفعات والنوافذ الزمنية.",
        "التمييز بين `X[0]` و`X[0:1]` و`X[:, 0]` في المصفوفات.",
    ],
    terms=["batch", "shape"],
    difficulty="beginner",
    summary_ar="الفهرسة تبدأ من 0، والتقطيع [start:stop) لا يشمل النهاية؛ هكذا تُبنى الدفعات والنوافذ.",
)

CODE = '''scores = [52, 55, 61, 64, 70, 74, 79, 85]

print(scores[0], scores[-1])        # الأول والأخير
print(scores[2:5])                  # الفهارس 2,3,4  ← 5 غير مشمول
print(scores[:3], scores[5:])       # الثلاثة الأولى / من الخامس إلى النهاية
print(scores[::2])                  # كل عنصر ثانٍ
print(scores[::-1])                 # معكوس

batch_size = 3
for b in range(0, len(scores), batch_size):      # بناء الدفعات
    print("batch", b // batch_size + 1, "=", scores[b:b + batch_size])

window = 3                                        # نوافذ زمنية للسلاسل
for t in range(len(scores) - window):
    print("X:", scores[t:t + window], "-> y:", scores[t + window])'''


def render() -> None:
    lesson_header(LESSON)
    h2("القاعدة", "The rule")
    definition("الفهرسة تبدأ من **0**. التقطيع `a[start:stop:step]` يعطي العناصر من `start` إلى `stop` **دون** `stop`. القيم المحذوفة تعني «من البداية» أو «إلى النهاية».")
    intuition("تخيّل الفهارس كخطوط بين العناصر لا كأرقام عليها: `a[2:5]` يقص بين الخط 2 والخط 5 فيحوي 3 عناصر (5 − 2).")
    table(
        ["التعبير", "المعنى", "الناتج على [52, 55, 61, 64, 70]"],
        [("a[0]", "الأول", "52"), ("a[-1]", "الأخير", "70"), ("a[1:3]", "الفهرسان 1 و2", "[55, 61]"),
         ("a[:2]", "أول اثنين", "[52, 55]"), ("a[3:]", "من 3 إلى النهاية", "[64, 70]"), ("a[::2]", "بخطوة 2", "[52, 61, 70]")],
        ["code", "rtl", "code"],
    )
    code_lab(CodeLab(
        key="py_slicing", title_ar="التقطيع وبناء الدفعات والنوافذ", code=CODE,
        before=Before(goal_ar="إتقان التقطيع ثم استخدامه في أهم نمطين: تقسيم البيانات إلى دفعات، وبناء نوافذ زمنية.",
                      stage_ar="أساسيات بايثون ← تحضير البيانات.", inputs_ar="قائمة من 8 درجات.",
                      expected_ar="مقاطع مختلفة، ثم 3 دفعات (الأخيرة جزئية)، ثم 5 نوافذ (X من 3 قيم، y القيمة التالية)."),
        explain=[("3-7", "أشكال التقطيع الأساسية. `[::-1]` يعكس القائمة."),
                 ("9-11", "`range(0, 8, 3)` يعطي 0, 3, 6؛ كل بداية دفعة. آخر دفعة `[6:9]` تحوي عنصرين فقط — الدفعة الجزئية التي رأيتها في المحاكي."),
                 ("13-15", "نافذة بطول 3: المدخل ثلاث قيم متتالية والهدف القيمة التالية. هذا بالضبط كيف تُبنى بيانات `RNN/LSTM`.")],
        run=run_printed(CODE),
        after_ar="- التقطيع خارج الحدود لا يعطي خطأ بل يقصّ ما هو متاح — لهذا الدفعة الأخيرة أقصر.\n- عدد النوافذ = الطول − حجم النافذة.",
    ))
    warning_note("في `NumPy` التقطيع يعيد **عرضًا** `view` على نفس الذاكرة لا نسخة: تعديل `X[0:2]` يعدّل `X`. استخدم `.copy()` عند الحاجة.")
    common_mistake("توقّع أن `a[2:5]` يشمل الفهرس 5. لا يشمله؛ عدد العناصر = 5 − 2 = 3.")
    quiz("py.slicing", [
        Q("`[10, 20, 30, 40, 50][1:4]` يعطي…", ["[20, 30, 40]", "[20, 30, 40, 50]", "[10, 20, 30]"], 0, "من 1 إلى 4 دون 4.", kind="output"),
        Q("قائمة بطول 10 ونافذة 4: كم نافذة (X, y)؟", ["10", "6", "4"], 1, "10 − 4 = 6.", kind="shape"),
        Q("`a[::-1]` يعطي…", ["الأول فقط", "القائمة معكوسة", "خطأ"], 1, "خطوة سالبة = عكس.", kind="output"),
    ])
    takeaway("[start:stop) بلا النهاية. الدفعات = تقطيع بخطوة batch_size؛ النوافذ = تقطيع منزلق.")
    lesson_footer(LESSON, ["الفهرسة من 0؛ -1 الأخير.", "stop غير مشمول.", "الدفعات والنوافذ الزمنية مجرد تقطيع."])
