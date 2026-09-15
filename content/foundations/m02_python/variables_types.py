import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.python.variables_types",
    title_ar="المتغيرات والأنواع الأساسية",
    title_en="Variables & Basic Types",
    module="foundations.python",
    order=1,
    prerequisites=["foundations.data.dtype"],
    objectives_ar=[
        "إنشاء متغير وفهم أنه اسم يشير إلى قيمة.",
        "التمييز بين `int` و`float` و`bool` و`str` ومعرفة أيها يدخل النموذج.",
        "استخدام `type()` والتحويل بين الأنواع.",
    ],
    terms=["dtype"],
    difficulty="beginner",
    summary_ar="المتغير اسم لقيمة؛ الأنواع الأساسية الأربعة وأيها تفهمه الشبكة.",
)

CODE = '''learning_rate = 0.01        # float: عدد كسري
epochs = 10                 # int: عدد صحيح
shuffle = True              # bool: صواب/خطأ
model_name = "mlp_v1"       # str: نص

print(type(learning_rate), type(epochs), type(shuffle), type(model_name))

steps = epochs * 5          # عملية حسابية على int
print("steps:", steps)

half = epochs / 4           # القسمة تعطي float دائمًا
print("half:", half, type(half))

print(int(3.9), float(7), str(0.5), bool(0), bool(2.5))   # تحويلات
print(True + True)          # bool يتصرف كـ 1/0 في الحساب'''


def render() -> None:
    lesson_header(LESSON)
    h2("ما هو المتغير؟", "What is a variable?")
    definition("**المتغير** اسم يشير إلى قيمة في الذاكرة. `learning_rate = 0.01` تعني: احفظ القيمة 0.01 وسمّها `learning_rate`. الاسم لا يحمل نوعًا؛ القيمة هي التي لها نوع.")
    intuition("فكّر في المتغير كملصق على صندوق. يمكن نقل الملصق إلى صندوق آخر (`epochs = 20`) لكن الصندوق نفسه له محتوى بنوع محدد.")
    table(
        ["النوع", "Python", "مثال", "في التعلم العميق"],
        [
            ("عدد صحيح", "int", "epochs = 10", "عدد الحقب، حجم الدفعة، عدد الوحدات"),
            ("عدد كسري", "float", "learning_rate = 0.01", "معدل التعلم، الأوزان، الخسارة"),
            ("منطقي", "bool", "shuffle = True", "خيارات: خلط، تدريب/تقييم"),
            ("نص", "str", "activation = \"relu\"", "أسماء الدوال والمقاييس؛ لا يدخل الحساب"),
        ],
        ["rtl", "code", "code", "rtl"],
    )
    code_lab(CodeLab(
        key="py_vars", title_ar="المتغيرات والأنواع", code=CODE,
        before=Before(goal_ar="رؤية الأنواع الأربعة وسلوك العمليات عليها.", stage_ar="أساسيات بايثون قبل أي بيانات.",
                      inputs_ar="قيم ثابتة.", expected_ar="أسماء الأنواع، نتائج حسابية، وتحويلات."),
        explain=[("1-4", "أربعة متغيرات بأربعة أنواع؛ لاحظ أن بايثون يستنتج النوع من القيمة."),
                 ("6", "`type()` يخبرك بنوع القيمة الحالية — أول أداة تشخيص."),
                 ("8-9", "ضرب صحيحين يعطي صحيحًا."),
                 ("11-12", "القسمة `/` تعطي `float` دائمًا حتى لو كانت النتيجة صحيحة؛ للقسمة الصحيحة استخدم `//`."),
                 ("14", "التحويل الصريح: `int(3.9)` يقطع لا يقرّب؛ `bool(0)` خطأ وأي رقم غير صفري صواب."),
                 ("15", "`True` يساوي 1 في الحساب؛ لهذا يمكن حساب الدقة بـ `mean(predictions == labels)`.")],
        run=run_printed(CODE),
        after_ar="- `<class 'float'>` هو نوع القيمة لا نوع الاسم.\n- `10 / 4 = 2.5` من نوع `float`؛ `int(3.9) = 3` قطع بلا تقريب.\n- `True + True = 2`: الأساس الذي تُبنى عليه دوال الدقة.",
    ))
    common_mistake("كتابة `batch_size = \"32\"` (نص) ثم تمريره إلى `fit`؛ النص لا يُستخدم في الحساب. الخطأ الناتج `TypeError` يصف المشكلة بدقة.")
    quiz("py.vars", [
        Q("ما نوع `7 / 7`؟", ["int", "float", "bool"], 1, "القسمة تعطي float دائمًا.", kind="output"),
        Q("`int(2.99)` يعطي…", ["3", "2", "2.99"], 1, "قطع لا تقريب.", kind="output"),
        Q("أي نوع لا يدخل حساب الشبكة مباشرة؟", ["float", "bool", "str"], 2, "النص يحتاج ترميزًا."),
    ])
    takeaway("المتغير ملصق على قيمة؛ القيمة لها نوع. الأعداد والمنطقيات تدخل الحساب، النص لا.")
    lesson_footer(LESSON, ["int, float, bool, str.", "type() للتشخيص.", "/ تعطي float؛ int() يقطع؛ True = 1."])
