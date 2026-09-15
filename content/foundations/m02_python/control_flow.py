import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.python.control_flow",
    title_ar="الشروط والحلقات",
    title_en="Conditions & Loops",
    module="foundations.python",
    order=4,
    prerequisites=["foundations.python.indexing_slicing"],
    objectives_ar=[
        "كتابة `if / elif / else` و`for` و`while` وقراءة المسافة البادئة كبنية.",
        "رؤية أن حلقة التدريب هي حلقتان متداخلتان: حقب ← دفعات.",
        "استخدام `break` لإيقاف مبكر و`enumerate` للعدّ.",
    ],
    terms=["epoch", "batch"],
    difficulty="beginner",
    summary_ar="الشروط تقرر، والحلقات تكرر؛ حلقة التدريب = for epoch: for batch.",
)

CODE = '''losses = [0.90, 0.61, 0.45, 0.44, 0.44, 0.43]
patience = 2                    # كم حقبة نصبر دون تحسّن ملموس؟

best = float("inf")             # ما لا نهاية: أي خسارة أصغر منه
bad_epochs = 0
for epoch, loss in enumerate(losses, start=1):
    if loss < best - 0.01:      # تحسّن ملموس
        best = loss
        bad_epochs = 0
        status = "improved"
    else:
        bad_epochs += 1
        status = "no improvement"
    print(f"epoch {{epoch}}: loss={{loss:.2f}} ({{status}})")
    if bad_epochs >= patience:
        print("early stopping at epoch", epoch)
        break

# حلقة التدريب في صورتها الهيكلية
n, batch_size, epochs = {n}, {bs}, 2
for epoch in range(1, epochs + 1):
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        print(f"epoch {{epoch}} batch [{{start}}:{{end}}]")'''


def _controls() -> dict:
    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("n (عدد الملاحظات)", 4, 12, 7, key="ctrl_py_flow_n")
    with c2:
        bs = st.slider("batch_size", 1, 6, 3, key="ctrl_py_flow_bs")
    return {"n": n, "bs": bs}


def render() -> None:
    lesson_header(LESSON)
    h2("الشرط والحلقة", "if / for / while")
    definition("**الشرط** `if` ينفّذ كتلة عندما يكون التعبير صوابًا. **الحلقة** `for` تكرر كتلة لكل عنصر في تسلسل؛ `while` تكرر ما دام الشرط صوابًا. الكتلة تُحدَّد بالمسافة البادئة (4 مسافات) لا بأقواس.")
    intuition("التدريب نفسه حلقة داخل حلقة: لكل حقبة، لكل دفعة، نفّذ خطوة. والإيقاف المبكر شرط داخل الحلقة الخارجية.")
    code_lab(CodeLab(
        key="py_flow", title_ar="إيقاف مبكر + هيكل حلقة التدريب", code=CODE, template=True, defaults={"n": 7, "bs": 3},
        before=Before(goal_ar="محاكاة منطق الإيقاف المبكر بشرط وحلقة، ثم كتابة هيكل حلقة التدريب المتداخلة.",
                      stage_ar="أساسيات بايثون ← التدريب (هيكلًا فقط).", inputs_ar="قائمة خسائر لست حقب، وأرقام `n` و`batch_size`.",
                      expected_ar="حالة كل حقبة، رسالة إيقاف مبكر، ثم أسطر `epoch/batch` بعدد `epochs × ceil(n / batch_size)`."),
        explain=[("1-2", "خسائر وهمية وصبر `patience` — نفس المعلمة في `EarlyStopping` لاحقًا."),
                 ("4-5", "`float(\"inf\")` قيمة أكبر من أي عدد؛ حيلة لتهيئة «الأفضل»."),
                 ("6", "`enumerate(..., start=1)` يعطي (رقم الحقبة، الخسارة) معًا."),
                 ("7-13", "إن تحسنت الخسارة بأكثر من 0.01 نصفّر العداد، وإلا نزيده. `f\"...\"` سلسلة منسّقة."),
                 ("15-17", "`break` يخرج من الحلقة فورًا عندما ينفد الصبر."),
                 ("20-24", "حلقتان متداخلتان: الحقبة خارجية، الدفعة داخلية؛ `min` يعالج الدفعة الجزئية.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- لاحظ الحقبتين 4 و5: تحسّن أقل من 0.01 لا يُحتسب، فيتوقف التدريب في الحقبة 5.\n- غيّر `n` و`batch_size` وتحقق أن عدد الدفعات في الحقبة = ceil(n / batch_size).",
    ))
    common_mistake("نسيان `:` في نهاية `if`/`for` أو خطأ في المسافة البادئة يعطي `IndentationError`/`SyntaxError` — اقرأ رقم السطر في الرسالة.")
    quiz("py.flow", [
        Q("`for start in range(0, 10, 4)` يعطي القيم…", ["0, 4, 8", "0, 4, 8, 12", "4, 8"], 0, "من 0 إلى 10 بخطوة 4.", kind="output"),
        Q("ماذا يفعل `break`؟", ["يتخطى العنصر الحالي", "يخرج من الحلقة", "يعيد الحلقة"], 1, "الخروج فورًا."),
        Q("حلقة التدريب هي…", ["حلقة واحدة على الدفعات", "حلقة حقب تحوي حلقة دفعات", "حلقة while لا نهائية"], 1, "متداخلة."),
    ])
    takeaway("الشروط تقرر والحلقات تكرر؛ حلقة التدريب حقب ← دفعات، والإيقاف المبكر شرط مع صبر.")
    lesson_footer(LESSON, ["if/elif/else, for, while, break, enumerate.", "المسافة البادئة هي البنية.", "التدريب = حلقتان متداخلتان."])
