import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.eval.classification_metrics",
    title_ar="مصفوفة الالتباس والدقة والصحة والاستدعاء وF1",
    title_en="Confusion Matrix, Accuracy, Precision, Recall & F1",
    module="foundations.eval",
    order=2,
    prerequisites=["foundations.eval.splits_metrics", "foundations.prep.imbalance_augmentation"],
    objectives_ar=["بناء مصفوفة الالتباس واشتقاق كل المقاييس منها.", "تفسير الصحة والاستدعاء كسؤالين مختلفين وF1 كتوازن.", "تعميم المقاييس على عدة فئات (macro/micro)."],
    terms=["probability", "target"],
    labs=["labs.confusion_matrix_lab"],
    difficulty="intermediate",
    summary_ar="TP/FP/FN/TN ← الدقة (كل الصواب)، الصحة (من الإيجابيات المتنبأة)، الاستدعاء (من الإيجابيات الحقيقية)، F1 توسط توافقي.",
)

CODE = '''import numpy as np
y_true = np.array([1,0,1,1,0,0,1,0,0,0, 1,0,0,0,0,1,0,0,0,0])
y_pred = np.array([1,0,0,1,0,1,1,0,0,0, 0,0,0,0,0,1,0,0,0,0])

TP = ((y_pred==1)&(y_true==1)).sum(); FP = ((y_pred==1)&(y_true==0)).sum()
FN = ((y_pred==0)&(y_true==1)).sum(); TN = ((y_pred==0)&(y_true==0)).sum()
print(f"confusion matrix:  TP={TP} FP={FP}\\n                   FN={FN} TN={TN}")
acc = (TP+TN)/len(y_true); prec = TP/(TP+FP); rec = TP/(TP+FN); f1 = 2*prec*rec/(prec+rec)
print(f"accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")
print("majority baseline accuracy =", max(y_true.mean(), 1-y_true.mean()).round(3))

# متعدد الفئات: مصفوفة K×K ومتوسطات macro/micro
yt = np.array([0,0,0,1,1,2,2,2,2,2]); yp = np.array([0,0,1,1,2,2,2,2,0,2])
K = 3; cm = np.zeros((K,K), int)
for t, p in zip(yt, yp): cm[t, p] += 1
print("multiclass confusion (rows=true, cols=pred):\\n", cm)
prec_k = np.diag(cm) / np.maximum(cm.sum(0), 1); rec_k = np.diag(cm) / np.maximum(cm.sum(1), 1)
print("per-class precision:", prec_k.round(2), " recall:", rec_k.round(2))
print("macro-F1 =", np.mean(2*prec_k*rec_k/np.maximum(prec_k+rec_k, 1e-9)).round(3), "  accuracy (=micro-F1) =", (np.diag(cm).sum()/cm.sum()).round(3))'''


def render() -> None:
    lesson_header(LESSON)
    h2("مصفوفة الالتباس", "Confusion matrix")
    definition("**مصفوفة الالتباس** جدول يعدّ التنبؤات حسب (الحقيقة، التنبؤ). للثنائي: **TP** إيجابي صحيح، **FP** إيجابي خاطئ (إنذار كاذب)، **FN** سلبي خاطئ (تفويت)، **TN** سلبي صحيح.")
    compare_table(["", "تنبؤ: إيجابي", "تنبؤ: سلبي"], [("حقيقة: إيجابي", "TP", "FN (تفويت)"), ("حقيقة: سلبي", "FP (إنذار كاذب)", "TN")], ["rtl", "code", "code"])
    equation(r"\text{Acc} = \frac{TP+TN}{n}, \quad \text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad F_1 = \frac{2\,PR}{P+R}",
             [(r"\text{Acc}", "نسبة الصواب الكلية — تضلل مع عدم التوازن."), (r"\text{Precision}", "الصحة: من كل ما قلت إنه إيجابي، كم كان صحيحًا؟ (كلفة الإنذار الكاذب)"), (r"\text{Recall}", "الاستدعاء (الحساسية): من كل الإيجابيات الحقيقية، كم التقطت؟ (كلفة التفويت)"), ("F_1", "التوسط التوافقي: يعاقب اختلال أحدهما.")],
             meaning_ar="أربعة أرقام، أربعة أسئلة. المهمة تحدد أيها يهم.",
             example_ar="TP=4, FP=1, FN=3, TN=12: Acc=0.80، P=0.80، R=0.57، F1=0.67. خط الأساس (كل سلبي) Acc=0.65.",
             dl_link_ar="`metrics=['accuracy', Precision(), Recall()]` في Keras؛ `sklearn.metrics.classification_report` يعطي الكل.", title_ar="المقاييس من المصفوفة")
    intuition("كشف الاحتيال: الاستدعاء يهم (لا تفوّت احتيالًا). تصفية الرسائل المزعجة: الصحة تهم (لا تحذف رسالة مهمة). القروض: يعتمد على كلفة التعثر مقابل كلفة رفض عميل جيد — قرار اقتصادي.")
    worked_steps([("عدّ", r"TP=4,\ FP=1,\ FN=3,\ TN=12"), ("الصحة", r"4/(4+1) = 0.80"), ("الاستدعاء", r"4/(4+3) = 0.57"), ("F1", r"2(0.80)(0.57)/(0.80+0.57) = 0.67")])
    code_lab(CodeLab(
        key="eval_cm", title_ar="المصفوفة والمقاييس يدويًا، ثنائي ومتعدد", code=CODE,
        before=Before(goal_ar="حساب المصفوفة والمقاييس الأربعة من متجهي حقيقة/تنبؤ، ثم مصفوفة K×K مع macro/micro.", stage_ar="التقييم.",
                      inputs_ar="20 تنبؤًا ثنائيًا و10 متعددة الفئات.", expected_ar="TP=4 FP=1 FN=3 TN=12 والمقاييس؛ مصفوفة 3×3 ومتوسطات."),
        explain=[("5-6", "الأربعة بعدّ الأقنعة."), ("8-10", "المقاييس ومقارنتها بخط الأساس: الدقة 0.80 مقابل 0.65 — أفضل من الأساس لكن الاستدعاء 0.57 يقول إن 43% من الإيجابيات فاتتنا."),
                 ("13-16", "متعدد الفئات: الصف الحقيقة والعمود التنبؤ؛ القطر صواب."), ("17-19", "لكل فئة صحتها واستدعاؤها؛ **macro** يتوسط الفئات بالتساوي (يهتم بالنادرة)، **micro** يعدّ كل التنبؤات (= الدقة).")],
        run=run_printed(CODE),
        after_ar="- macro-F1 أقل من الدقة عندما تكون الفئة النادرة ضعيفة الأداء: هذا ما تريده أن يظهر.\n- خارج القطر في المصفوفة يخبرك **أي** فئتين تلتبسان — أثمن من أي رقم مجمّع.",
    ))
    st.button("افتح معمل مصفوفة الالتباس", icon=":material/science:", type="primary", on_click=goto, args=("labs.confusion_matrix_lab",), key="cm_lab")
    common_mistake("الإبلاغ عن الدقة فقط في مسألة بفئة نادرة 5%: نموذج يتنبأ «سلبي» دائمًا يحقق 95%. أبلغ عن الصحة والاستدعاء وF1 والمصفوفة.")
    quiz("eval.cm", [
        Q("الصحة تجيب عن…", ["كم من الإيجابيات الحقيقية التقطت؟", "كم من تنبؤاتي الإيجابية صحيح؟", "نسبة الصواب الكلية"], 1, "TP/(TP+FP)."),
        Q("في كشف الاحتيال يهم أكثر…", ["الصحة", "الاستدعاء", "الدقة"], 1, "لا تفوّت."),
        Q("micro-F1 في متعدد الفئات يساوي…", ["macro-F1", "الدقة", "الاستدعاء"], 1, "عدّ كل التنبؤات."),
        Q("TP=10, FP=10, FN=0: الصحة والاستدعاء…", ["0.5 و 1.0", "1.0 و 0.5", "0.5 و 0.5"], 0, "إنذارات كاذبة كثيرة، لا تفويت."),
    ])
    takeaway("المصفوفة أصل كل شيء. الصحة = كلفة الإنذار الكاذب، الاستدعاء = كلفة التفويت، F1 توازنهما، macro للفئات النادرة.")
    lesson_footer(LESSON, ["TP/FP/FN/TN.", "أربعة أسئلة لأربعة مقاييس.", "خارج القطر يشخّص."])
