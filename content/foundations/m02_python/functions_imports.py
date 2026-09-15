import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.python.functions_imports",
    title_ar="الدوال والوسائط والاستيراد",
    title_en="Functions, Arguments & Imports",
    module="foundations.python",
    order=5,
    prerequisites=["foundations.python.control_flow"],
    objectives_ar=[
        "تعريف دالة بوسائط إلزامية واختيارية وقيمة إرجاع.",
        "قراءة توقيع دالة مثل `fit(x, y, batch_size=32, epochs=1)` بثقة.",
        "فهم `import` و`from ... import` والأسماء المستعارة `np`، `pd`.",
    ],
    terms=["loss", "hyperparameter"],
    difficulty="beginner",
    summary_ar="الدالة تغلّف حسابًا بوسائط؛ الوسائط الاختيارية هي المعلمات الفائقة بقيمها الافتراضية.",
)

CODE = '''import math
import numpy as np                       # اسم مستعار متفق عليه
from statistics import mean              # استيراد اسم واحد

def mse(y_true, y_pred):
    """متوسط مربع الخطأ بين قائمتين بنفس الطول."""
    errors = [(p - t) ** 2 for p, t in zip(y_pred, y_true)]
    return mean(errors)

def train_summary(name, epochs=10, batch_size=32, lr=0.01, verbose=True):
    steps = math.ceil(100 / batch_size) * epochs
    if verbose:
        print(f"{name}: epochs={epochs}, batch_size={batch_size}, lr={lr} -> steps={steps}")
    return steps

y_true = [52, 55, 61]
y_pred = [50, 58, 60]
print("mse =", mse(y_true, y_pred))
print("np  =", np.mean((np.array(y_pred) - np.array(y_true)) ** 2))   # نفس النتيجة بـ NumPy

train_summary("mlp")                                  # القيم الافتراضية
train_summary("mlp", epochs=5)                        # تغيير وسيط واحد بالاسم
train_summary("mlp", 5, 16)                           # بالموضع (أقل وضوحًا)
s = train_summary("cnn", batch_size=10, verbose=False)
print("returned:", s)'''


def render() -> None:
    lesson_header(LESSON)
    h2("ما هي الدالة؟", "What is a function?")
    definition("**الدالة** كتلة كود مسماة تأخذ **وسائط** `arguments` وتعيد قيمة بـ `return`. الوسيط **الإلزامي** بلا قيمة افتراضية؛ **الاختياري** له قيمة افتراضية تُستخدم إن لم تمرر غيرها.")
    table(
        ["في التوقيع", "المعنى", "مثال من Keras"],
        [("x, y", "وسيطان إلزاميان بالموضع", "fit(X_train, y_train)"),
         ("batch_size=32", "اختياري بقيمة افتراضية", "fit(..., batch_size=64)"),
         ("**kwargs", "وسائط إضافية بالاسم", "Dense(64, activation=\"relu\")")],
        ["code", "rtl", "code"],
    )
    code_lab(CodeLab(
        key="py_functions", title_ar="دوال بوسائط افتراضية + استيراد", code=CODE,
        before=Before(goal_ar="تعريف دالة خسارة صغيرة ودالة بوسائط اختيارية، واستدعاؤهما بأساليب مختلفة.",
                      stage_ar="أساسيات بايثون.", inputs_ar="قائمتان قصيرتان.", expected_ar="قيمة `mse`، ونفس القيمة بـ `NumPy`، وأسطر ملخص التدريب."),
        explain=[("1-3", "ثلاثة أشكال للاستيراد: الوحدة كاملة، بالاسم المستعار `np` (عرف عالمي)، واسم واحد منها."),
                 ("5-8", "دالة بوسيطين إلزاميين. السطر بين علامات الاقتباس الثلاثية توثيق `docstring`. `zip` يقرن العنصرين المتقابلين."),
                 ("10-14", "وسيط إلزامي واحد `name` وأربعة اختيارية — هذا شكل توقيعات `compile`/`fit`. القيمة الافتراضية تُستخدم عند عدم التمرير."),
                 ("18-19", "نفس الحساب بحلقة بايثون ثم بـ `NumPy` في سطر واحد — مقدمة للتوجيه المتجهي."),
                 ("21-25", "أربعة أساليب استدعاء: افتراضي، بالاسم، بالموضع، والتقاط القيمة المعادة.")],
        run=run_printed(CODE),
        after_ar="- تمرير الوسائط **بالاسم** أوضح وأقل عرضة للخطأ من الموضع؛ `fit(X, y, 32, 10)` غامض بينما `fit(X, y, batch_size=32, epochs=10)` واضح.\n- الدالة التي لا تطبع لكن تُعيد قيمة أفضل لإعادة الاستخدام.",
    ))
    practical_note("قاعدة قراءة التوثيق: الوسائط قبل `=` إلزامية، وما بعده اختياري بقيمته الافتراضية. `epochs=1` في `fit` تعني حقبة واحدة إن لم تحدد — سبب شائع لنتائج ضعيفة.")
    common_mistake("تمرير الوسائط الاختيارية بالموضع بترتيب خاطئ: `train_summary(\"mlp\", 32, 10)` يجعل `epochs=32` و`batch_size=10` بصمت.")
    quiz("py.functions", [
        Q("في `def f(a, b=2)`، ما الإلزامي؟", ["a", "b", "كلاهما"], 0, "b له قيمة افتراضية."),
        Q("`import numpy as np` يعني…", ["استيراد دالة np", "استيراد numpy تحت اسم np", "تثبيت numpy"], 1, "اسم مستعار."),
        Q("`fit(X, y, 10, 32)` مقابل `fit(X, y, batch_size=32, epochs=10)`: أيهما أوضح ولماذا؟", ["الأول لأنه أقصر", "الثاني لأن الاسم يمنع تبديل القيم", "لا فرق"], 1, "بالاسم أأمن."),
    ])
    takeaway("الدالة = وسائط + عملية + إرجاع. الاختياري بقيمة افتراضية؛ مرّر بالاسم. np وpd أسماء مستعارة متفق عليها.")
    lesson_footer(LESSON, ["def, return, وسائط إلزامية/اختيارية.", "مرّر بالاسم لتجنب التبديل.", "import numpy as np."])
