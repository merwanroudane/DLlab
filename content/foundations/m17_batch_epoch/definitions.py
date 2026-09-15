import math

import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import term_table

LESSON = Lesson(
    id="foundations.batch_epoch.definitions",
    title_ar="الحقبة والدفعة وحجمها والتكرار والتحديث — بالأرقام",
    title_en="Epoch, Batch, Batch Size, Iteration & Update — by the Numbers",
    module="foundations.batch_epoch",
    order=1,
    prerequisites=["foundations.training_loop.the_loop", "foundations.python.indexing_slicing"],
    objectives_ar=["تعريف المفاهيم الخمسة بدقة وربطها بالكود.", "حساب steps_per_epoch = ceil(n / batch_size) والتحديثات الكلية.", "التعامل مع الدفعة الجزئية الأخيرة."],
    terms=["epoch", "batch", "batch_size", "iteration"],
    labs=["labs.epoch_batch_simulator"],
    difficulty="beginner",
    summary_ar="حقبة = مرور كامل؛ دفعة = مجموعة؛ batch_size = حجمها؛ تكرار = تحديث واحد على دفعة؛ steps_per_epoch = ceil(n/bs).",
)

CODE = '''import math
n, batch_size, epochs = {n}, {bs}, {epochs}
steps_per_epoch = math.ceil(n / batch_size)
last = n - (steps_per_epoch - 1) * batch_size
print(f"n={{n}} batch_size={{batch_size}} -> steps_per_epoch = ceil({{n}}/{{batch_size}}) = {{steps_per_epoch}}")
print(f"last batch size = {{last}}  ({{'partial' if last < batch_size else 'full'}})")
print(f"total updates over {{epochs}} epochs = {{steps_per_epoch * epochs}}")
print(f"each observation is used {{epochs}} times (once per epoch)")
print("with drop_remainder=True: steps_per_epoch =", n // batch_size, " observations dropped per epoch =", n % batch_size)

# البناء الفعلي للدفعات في حقبة واحدة
for i, s in enumerate(range(0, n, batch_size), start=1):
    e = min(s + batch_size, n)
    print(f"  batch {{i:>2}}: rows [{{s}}, {{e}})  size {{e - s}}")'''


def _controls() -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("n", 5, 200, 100, 5, key="ctrl_be_n")
    with c2:
        bs = st.slider("batch_size", 1, 64, 32, key="ctrl_be_bs")
    with c3:
        epochs = st.slider("epochs", 1, 20, 3, key="ctrl_be_epochs")
    return {"n": n, "bs": bs, "epochs": epochs}


def render() -> None:
    lesson_header(LESSON)
    h2("المفاهيم الخمسة", "The five concepts")
    term_table([
        ("الحقبة", "Epoch", "مرور كامل واحد على كل ملاحظات التدريب", "epochs=10"),
        ("الدفعة", "Batch", "مجموعة ملاحظات تُعالج معًا وتنتج تحديثًا واحدًا", "Batch 3 = rows [40, 60)"),
        ("حجم الدفعة", "Batch Size", "عدد الملاحظات في الدفعة؛ معلمة فائقة", "batch_size=32"),
        ("التكرار / الخطوة", "Iteration / Step", "دورة واحدة: أمامي ← خسارة ← خلفي ← تحديث على دفعة", "step 47"),
        ("التحديث", "Update", "تغيير المعلمات مرة واحدة (نهاية التكرار)", "θ ← θ − η∇L"),
    ])
    equation(r"\text{steps\_per\_epoch} = \left\lceil \frac{n}{\text{batch\_size}} \right\rceil, \qquad \text{total\_updates} = \text{steps\_per\_epoch} \times \text{epochs}",
             [(r"\lceil\cdot\rceil", "التقريب لأعلى: الدفعة الأخيرة قد تكون أصغر (جزئية)."), ("n", "عدد ملاحظات التدريب.")],
             meaning_ar="عدد التحديثات في الحقبة هو عدد الدفعات. كل ملاحظة تُستخدم مرة واحدة في كل حقبة.",
             example_ar="n = 100، batch_size = 32: ⌈3.125⌉ = 4 دفعات (32، 32، 32، 4). 10 حقب ⇒ 40 تحديثًا.",
             dl_link_ar="`25/25` في سجل Keras هو steps_per_epoch. في PyTorch `len(loader)`.", title_ar="الحساب الأساسي")
    intuition("«الحقبة» وحدة **عرض** البيانات، و«التكرار» وحدة **التعلم**. عشر حقب بدفعة 32 على 3200 ملاحظة = 1000 تكرار؛ نفس الحقب بدفعة 3200 = 10 تكرارات فقط.")
    code_lab(CodeLab(
        key="be_defs", title_ar="حاسبة الدفعات + بناء الدفعات فعليًا", code=CODE, template=True, defaults={"n": 100, "bs": 32, "epochs": 3},
        before=Before(goal_ar="حساب steps_per_epoch وحجم الدفعة الأخيرة والتحديثات الكلية، ثم طباعة حدود كل دفعة.", stage_ar="التدريب ← الحساب.",
                      inputs_ar="n وbatch_size وepochs.", expected_ar="عدد الدفعات وحجم الأخيرة وإجمالي التحديثات، وقائمة الدفعات بفهارسها."),
        explain=[("2-4", "الصيغة؛ `last` حجم الدفعة الأخيرة."), ("9", "مع `drop_remainder=True` (tf.data) أو `drop_last=True` (DataLoader): تُهمل البقية كل حقبة."), ("12-14", "نفس التقطيع الذي رأيته في درس الفهرسة: `range(0, n, batch_size)`.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- جرّب n = 100 و batch_size = 32: الدفعة الرابعة بـ 4 ملاحظات فقط. خسارتها أكثر ضوضاء (عينة صغيرة) — لهذا يُخلط الترتيب كل حقبة حتى لا تكون نفس الملاحظات دائمًا في الدفعة الجزئية.\n- اختيار batch_size يقسم n (100 ← 20 أو 25) يلغي الدفعة الجزئية.",
    ))
    common_mistake("«درّبت 10 حقب» بلا ذكر حجم الدفعة وحجم البيانات لا يخبر أحدًا كم تعلّم النموذج. أبلغ عن التحديثات أو عن الثلاثة معًا.")
    st.button("افتح محاكي الحقبة والدفعة", icon=":material/science:", type="primary", on_click=goto, args=("labs.epoch_batch_simulator",), key="be_lab")
    quiz("be.defs", [
        Q("n = 1000، batch_size = 64: steps_per_epoch…", ["15", "16", "64"], 1, "⌈15.6⌉.", kind="shape"),
        Q("حجم الدفعة الأخيرة في المثال السابق…", ["64", "40", "8"], 1, "1000 − 15×64.", kind="shape"),
        Q("في 5 حقب بدفعة 64 على 1000 ملاحظة، كم مرة تُستخدم كل ملاحظة؟", ["5", "16", "80"], 0, "مرة لكل حقبة."),
    ])
    takeaway("حقبة = عرض كامل؛ تكرار = تحديث واحد؛ steps_per_epoch = ⌈n/bs⌉؛ الأخيرة قد تكون جزئية؛ أبلغ عن التحديثات.")
    lesson_footer(LESSON, ["المصطلحات الخمسة بجدول.", "الصيغة والحساب.", "الدفعة الجزئية وdrop_remainder."])
