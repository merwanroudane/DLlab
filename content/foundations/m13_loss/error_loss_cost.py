import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.loss.error_loss_cost",
    title_ar="الخطأ، المتبقي، الخسارة، الكلفة، الهدف، المقياس",
    title_en="Error, Residual, Loss, Cost, Objective, Metric",
    module="foundations.loss",
    order=1,
    prerequisites=["foundations.ml.objective_loss_metric", "foundations.forward.logits_probability_threshold"],
    objectives_ar=["ضبط ستة مصطلحات متقاربة بتعريف واحد لكل منها.", "التمييز بين خسارة الملاحظة وخسارة الدفعة/المجموعة."],
    terms=["loss"],
    difficulty="beginner",
    summary_ar="خطأ لكل ملاحظة؛ خسارة = دالة الخطأ لملاحظة؛ كلفة = متوسطها على المجموعة؛ هدف = ما نحسّنه؛ مقياس = ما نبلّغ عنه.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ستة مصطلحات", "Six terms")
    compare_table(["المصطلح", "English", "التعريف", "الشكل", "مثال"],
                  [("الخطأ", "Error", "الفرق بين التنبؤ والحقيقة لملاحظة", "(n,)", "ŷ − y = −3"), ("المتبقي", "Residual", "نفس الخطأ بلغة الإحصاء (y − ŷ عادةً)", "(n,)", "+3"),
                   ("الخسارة", "Loss (per sample)", "دالة موجبة للخطأ لملاحظة واحدة", "(n,)", "e² = 9"), ("الكلفة", "Cost (batch/dataset)", "متوسط خسائر الملاحظات", "()", "MSE = 12.5"),
                   ("الهدف", "Objective", "الدالة الكلية التي نقلّلها (الكلفة + التنظيم)", "()", "MSE + λ‖w‖²"), ("المقياس", "Metric", "ما نبلّغ عنه، قد لا يُشتق", "()", "RMSE، الدقة")],
                  ["rtl", "ltr", "rtl", "code", "code"])
    definition("عمليًا في الأطر: كلمة **loss** تُستخدم لخسارة الملاحظة وللكلفة معًا؛ `model.compile(loss=...)` تعني «دالة خسارة الملاحظة، وسنأخذ متوسطها على الدفعة». الرقم المطبوع أثناء التدريب هو الكلفة (المتوسط).")
    intuition("الخطأ يقول «كم أخطأت وفي أي اتجاه». الخسارة تقول «كم يؤلم هذا الخطأ» (المربع يؤلم أكثر للكبير). الكلفة تلخص الألم على الدفعة. الهدف يضيف «ولا تجعل الأوزان ضخمة». المقياس يقول للبشر كيف الحال.")
    common_mistake("مقارنة خسارة دفعة واحدة بخسارة الحقبة: الأولى تتذبذب (عينة صغيرة) والثانية متوسط. Keras يطبع متوسطًا متحركًا خلال الحقبة؛ القيمة النهائية للحقبة هي المتوسط الكامل.")
    quiz("loss.terms", [
        Q("الرقم `loss: 0.34` في سجل Keras هو…", ["خسارة ملاحظة", "متوسط خسائر الدفعات في الحقبة", "المقياس"], 1, "كلفة."),
        Q("الهدف = الكلفة + …", ["المقياس", "حد التنظيم", "الخطأ"], 1, "ما نحسّنه فعلًا."),
        Q("المتبقي في الإحصاء يساوي…", ["ŷ − y", "y − ŷ", "|y|"], 1, "الاتفاق الإحصائي."),
    ])
    takeaway("خطأ → خسارة (ملاحظة) → كلفة (متوسط) → هدف (+ تنظيم). المقياس للتقرير. loss في الأطر = الاثنان.")
    lesson_footer(LESSON, ["ستة مصطلحات بجدول واحد.", "loss المطبوعة = متوسط.", "الهدف قد يشمل التنظيم."])
