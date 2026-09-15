import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.training_loop.the_loop",
    title_ar="الحلقة كاملة: اثنتا عشرة خطوة",
    title_en="The Full Loop: Twelve Steps",
    module="foundations.training_loop",
    order=1,
    prerequisites=["foundations.backprop.backpropagation", "foundations.optim.gd_variants"],
    objectives_ar=["حفظ ترتيب الخطوات وفهم ما يدخل ويخرج من كل خطوة وأشكالها.", "التمييز بين ما يحدث لكل دفعة وما يحدث لكل حقبة."],
    terms=["epoch", "batch", "iteration", "optimizer"],
    labs=["labs.training_loop_simulator", "labs.epoch_batch_simulator"],
    difficulty="intermediate",
    summary_ar="لكل حقبة: اخلط، ولكل دفعة: أمامي ← خسارة ← خلفي ← تحديث؛ ثم تحقق في نهاية الحقبة؛ توقف بشرط.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الحلقة", "The loop")
    pipeline(["Init θ", "Shuffle", "Load batch", "Forward", "Loss", "Backward", "Optimizer step", "Next batch", "Epoch end", "Validate", "Next epoch", "Stop"], active=3)
    compare_table(["#", "الخطوة", "English", "المستوى", "المدخل ← المخرج", "من يفعلها في Keras / PyTorch"],
                  [("1", "تهيئة المعلمات", "Initialize", "مرة", "أشكال الطبقات ← W, b عشوائية صغيرة", "بناء النموذج / `nn.Module.__init__`"),
                   ("2", "خلط البيانات", "Shuffle", "كل حقبة", "ترتيب الملاحظات", "`shuffle=True` / `DataLoader(shuffle=True)`"),
                   ("3", "تحميل دفعة", "Load batch", "كل دفعة", "(X, y) كاملة ← (X_b, y_b) بحجم batch_size", "داخل fit / `for X_b, y_b in loader`"),
                   ("4", "التمرير الأمامي", "Forward", "كل دفعة", "X_b ← ŷ_b (+ cache)", "داخل fit / `model(X_b)`"),
                   ("5", "الخسارة", "Loss", "كل دفعة", "ŷ_b, y_b ← رقم", "داخل fit / `loss_fn(...)`"),
                   ("6", "الانتشار الخلفي", "Backward", "كل دفعة", "cache ← تدرج لكل معلمة", "داخل fit / `loss.backward()`"),
                   ("7", "خطوة المحسّن", "Optimizer step", "كل دفعة", "تدرجات ← تحديث θ", "داخل fit / `optimizer.step()`"),
                   ("8", "الدفعة التالية", "Next batch", "كل دفعة", "—", "الحلقة الداخلية"),
                   ("9", "نهاية الحقبة", "Epoch end", "كل حقبة", "متوسط خسارة الدفعات ← loss", "سجل التقدم"),
                   ("10", "التحقق", "Validate", "كل حقبة", "X_val ← val_loss, val_metric (بلا تدرج)", "`validation_data` / حلقة `no_grad`"),
                   ("11", "الحقبة التالية", "Next epoch", "كل حقبة", "—", "الحلقة الخارجية"),
                   ("12", "التوقف", "Stop", "مرة", "epochs أو إيقاف مبكر", "`epochs=` / `EarlyStopping`")],
                  ["num", "rtl", "ltr", "rtl", "rtl", "code"])
    definition("**التكرار** `Iteration` = الخطوات 3–8 على دفعة واحدة = تحديث واحد. **الحقبة** = كل التكرارات التي تغطي البيانات مرة واحدة + التحقق.")
    intuition("حلقتان متداخلتان فقط: الخارجية تعدّ الحقب وتخلط وتتحقق؛ الداخلية تعدّ الدفعات وتُعلّم. كل شيء آخر تفاصيل داخل إحداهما.")
    common_mistake("حساب التحقق داخل حلقة الدفعات (بطيء ومضلل)، أو تحديث المعلمات قبل الانتشار الخلفي (بلا معنى)، أو نسيان تصفير التدرجات المتراكمة في PyTorch قبل `backward` (تدرجات مضاعفة).")
    st.button("افتح محاكي حلقة التدريب", icon=":material/science:", type="primary", on_click=goto, args=("labs.training_loop_simulator",), key="loop_lab")
    quiz("loop.steps", [
        Q("الترتيب الصحيح داخل الدفعة…", ["خلفي ← أمامي ← خسارة ← تحديث", "أمامي ← خسارة ← خلفي ← تحديث", "تحديث ← أمامي ← خسارة"], 1, "لا يمكن الاشتقاق قبل الحساب."),
        Q("التحقق يُنفَّذ…", ["كل دفعة", "كل حقبة", "مرة في النهاية"], 1, "الحلقة الخارجية."),
        Q("`optimizer.step()` يقابل الخطوة…", ["4", "6", "7"], 2, "التحديث."),
    ])
    takeaway("اثنتا عشرة خطوة في حلقتين. الداخلية تُعلّم (دفعة)، الخارجية تخلط وتتحقق وتقرر التوقف (حقبة).")
    lesson_footer(LESSON, ["الجدول يربط كل خطوة بالإطارين.", "iteration = خطوات 3–8.", "التحقق كل حقبة بلا تدرج."])
