import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.eval.regression_metrics",
    title_ar="مقاييس الانحدار: MSE وRMSE وMAE وMAPE وR²",
    title_en="Regression Metrics: MSE, RMSE, MAE, MAPE & R²",
    module="foundations.eval",
    order=4,
    prerequisites=["foundations.eval.splits_metrics", "foundations.loss.mse_mae_huber"],
    objectives_ar=["حساب المقاييس الخمسة وتفسير وحداتها.", "قراءة R² كنسبة تباين مفسَّر مقارنة بخط الأساس (المتوسط)، وفهم متى يكون سالبًا.", "اختيار المقياس بحسب الشواذ ووحدة الهدف ومقياسه."],
    terms=["mean", "variance"],
    difficulty="beginner",
    summary_ar="RMSE وMAE بوحدة الهدف؛ MAPE نسبة مئوية (احذر الأصفار)؛ R² = 1 − SSE/SST: كم أفضل من التنبؤ بالمتوسط.",
)

CODE = '''import numpy as np
y    = np.array([200., 180., 250., 300., 220., 275.])         # آلاف
pred = np.array([210., 175., 240., 330., 215., 260.])
e = pred - y
mse = np.mean(e**2); rmse = np.sqrt(mse); mae = np.mean(np.abs(e)); mape = np.mean(np.abs(e/y))*100
sse = np.sum(e**2); sst = np.sum((y - y.mean())**2); r2 = 1 - sse/sst
print(f"MSE={mse:.1f}  RMSE={rmse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%  R2={r2:.3f}")
print("baseline (predict mean): RMSE =", np.sqrt(sst/len(y)).round(2), " R2 = 0 by definition")

# R² سالب: نموذج أسوأ من المتوسط
bad = np.full(6, 400.0)
print("constant 400: R2 =", (1 - np.sum((bad-y)**2)/sst).round(3), " <- worse than predicting the mean")

# MAPE مع قيم قرب الصفر
y2 = np.array([0.5, 100., 100.]); p2 = np.array([1.5, 101., 99.])
print("MAPE with a near-zero target:", (np.mean(np.abs((p2-y2)/y2))*100).round(1), "%  <- dominated by the 0.5 case")'''


def render() -> None:
    lesson_header(LESSON)
    h2("المقاييس", "The metrics")
    equation(r"\text{RMSE} = \sqrt{\tfrac{1}{n}\sum e_i^2}, \quad \text{MAE} = \tfrac{1}{n}\sum|e_i|, \quad \text{MAPE} = \tfrac{100}{n}\sum\left|\tfrac{e_i}{y_i}\right|, \quad R^2 = 1 - \frac{\sum e_i^2}{\sum (y_i - \bar y)^2}",
             [("e_i", "الخطأ $\\hat y_i - y_i$."), (r"\text{RMSE}", "جذر MSE: بوحدة الهدف، حساس للشواذ."), (r"\text{MAE}", "بوحدة الهدف، مقاوم."), (r"\text{MAPE}", "نسبة مئوية؛ غير معرّفة عند $y = 0$ ومتحيزة للقيم الصغيرة."), ("R^2", "1 − (خطأ النموذج / خطأ التنبؤ بالمتوسط): 1 مثالي، 0 = المتوسط، سالب = أسوأ من المتوسط.")],
             meaning_ar="RMSE/MAE تقولان «كم نخطئ بوحدة الهدف»؛ MAPE «بكم بالمئة»؛ R² «كم نحن أفضل من خط الأساس».",
             example_ar="أخطاء (10, −5, −10, 30, −5, −15): RMSE = 15.5، MAE = 12.5، R² ≈ 0.87.",
             dl_link_ar="`metrics=['mae', RootMeanSquaredError()]` في Keras. R² ليس مقياس خسارة؛ يُحسب بعد التنبؤ.", title_ar="مقاييس الانحدار")
    definition("**R²** ليس «نسبة الصواب» ولا احتمالًا. هو مقارنة بخط الأساس: 0.87 يعني أن النموذج يزيل 87% من تباين الهدف الذي يتركه التنبؤ بالمتوسط. يمكن أن يكون سالبًا على بيانات جديدة.")
    code_lab(CodeLab(
        key="eval_reg", title_ar="المقاييس الخمسة وحالتان مضللتان", code=CODE,
        before=Before(goal_ar="حساب المقاييس لتنبؤات أسعار، ثم R² سالب لنموذج ثابت سيئ، وMAPE منفجرة بقيمة قرب الصفر.", stage_ar="التقييم.",
                      inputs_ar="6 أسعار وتنبؤات، ومثالان صغيران.", expected_ar="RMSE ≈ 15.5، MAE = 12.5، R² ≈ 0.87؛ R² سالب للثابت 400؛ MAPE ضخمة مع y = 0.5."),
        explain=[("4-7", "الصيغ. لاحظ أن خط أساس RMSE = الانحراف المعياري للهدف."), ("10-11", "R² سالب: النموذج أسوأ من التنبؤ بالمتوسط — يحدث فعلًا على بيانات اختبار مع فرط تخصيص."), ("14-15", "MAPE: خطأ 1.0 على هدف 0.5 = 200%. لا تستخدمها مع أهداف قرب الصفر أو سالبة.")],
        run=run_printed(CODE),
        after_ar="- أبلغ عن RMSE أو MAE بوحدة الهدف (آلاف الدينار) ليفهمها القارئ؛ MSE بوحدة مربعة غير بديهية.\n- R² يعتمد على تباين مجموعة الاختبار: نفس النموذج يعطي R² مختلفًا على عينة أضيق.",
    ))
    compare_table(["المقياس", "الوحدة", "الشواذ", "مقارنة بين مجموعات بيانات مختلفة", "متى"],
                  [("RMSE", "وحدة y", "حساس", "لا", "الافتراضي مع MSE"), ("MAE", "وحدة y", "مقاوم", "لا", "شواذ أو تفسير مباشر"), ("MAPE", "%", "حساس للصغير", "نعم (بحذر)", "أهداف موجبة بعيدة عن الصفر"), ("R²", "بلا", "حساس", "نعم", "«كم أفضل من المتوسط»")],
                  ["ltr", "rtl", "rtl", "rtl", "rtl"])
    common_mistake("«R² = 0.95 على التدريب» في التقرير: R² التدريب يرتفع بمجرد زيادة القدرة. الرقم ذو المعنى على الاختبار — وقد يكون سالبًا.")
    quiz("eval.reg", [
        Q("R² = 0 يعني…", ["نموذج ممتاز", "يساوي التنبؤ بالمتوسط", "خطأ في الحساب"], 1, "خط الأساس."),
        Q("هدف قد يكون صفرًا أو سالبًا: تجنب…", ["MAE", "MAPE", "RMSE"], 1, "القسمة على y."),
        Q("أي مقياس بوحدة الهدف نفسها؟", ["MSE", "RMSE وMAE", "R²"], 1, "للتفسير."),
    ])
    takeaway("RMSE/MAE بوحدة الهدف، MAPE نسبة (احذر الصفر)، R² مقارنة بالمتوسط ويمكن أن يكون سالبًا. أبلغ عن الاختبار.")
    lesson_footer(LESSON, ["خمسة مقاييس وصيغها.", "R² ليس نسبة صواب.", "MAPE والأصفار."])
