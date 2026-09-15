import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.loss.mse_mae_huber",
    title_ar="خسائر الانحدار: MSE وMAE وHuber",
    title_en="Regression Losses: MSE, MAE & Huber",
    module="foundations.loss",
    order=2,
    prerequisites=["foundations.loss.error_loss_cost", "foundations.math.summation_average_abs"],
    objectives_ar=["صيغ الثلاث ومشتقاتها بالنسبة للتنبؤ.", "متى يفضَّل كل منها (شواذ، وحدة، استقرار التدرج).", "قراءة قيم الخسارة بوحدة الهدف."],
    terms=["loss", "mean", "derivative"],
    labs=["labs.loss_lab"],
    difficulty="intermediate",
    summary_ar="MSE يعاقب الكبير تربيعيًا (تدرج يتناسب مع الخطأ)، MAE خطيًا (تدرج ثابت)، Huber يجمعهما بعتبة δ.",
)

CODE = '''import numpy as np
y = np.array([100., 102., 98., 101., 99., 150.])       # الأخير شاذ
pred = np.full(6, 100.0)
e = pred - y
mse, mae = np.mean(e**2), np.mean(np.abs(e))
delta = 5.0
huber = np.where(np.abs(e) <= delta, 0.5*e**2, delta*(np.abs(e) - 0.5*delta)).mean()
print(f"MSE={mse:.2f}  RMSE={np.sqrt(mse):.2f}  MAE={mae:.2f}  Huber(δ=5)={huber:.2f}")

# تدرج كل خسارة بالنسبة للتنبؤ (لكل ملاحظة)
g_mse, g_mae = 2*e/len(e), np.sign(e)/len(e)
g_huber = np.where(np.abs(e) <= delta, e, delta*np.sign(e))/len(e)
print("dMSE/dpred  :", g_mse.round(3))
print("dMAE/dpred  :", g_mae.round(3))
print("dHuber/dpred:", g_huber.round(3))
print("MSE gradient is dominated by the outlier; MAE/Huber cap its influence.")

# أي ثابت يقلل كل خسارة؟ MSE → المتوسط، MAE → الوسيط
c = np.linspace(95, 115, 2001)
print("argmin MSE =", c[np.argmin([np.mean((k - y)**2) for k in c])].round(2), " (mean =", y.mean().round(2), ")")
print("argmin MAE =", c[np.argmin([np.mean(np.abs(k - y)) for k in c])].round(2), " (median =", np.median(y), ")")'''


def render() -> None:
    lesson_header(LESSON)
    h2("الثلاث", "The three")
    equation(r"\text{MSE} = \frac{1}{n}\sum e_i^2, \qquad \text{MAE} = \frac{1}{n}\sum |e_i|, \qquad \text{Huber}_\delta(e) = \begin{cases}\tfrac{1}{2}e^2 & |e| \le \delta\\ \delta(|e| - \tfrac{1}{2}\delta) & |e| > \delta\end{cases}",
             [("e_i", "الخطأ $\\hat{y}_i - y_i$."), (r"\delta", "عتبة Huber: تربيعية للأخطاء الصغيرة، خطية للكبيرة."), (r"\text{RMSE} = \sqrt{\text{MSE}}", "بوحدة الهدف، للتقرير.")],
             meaning_ar="MSE تضخّم الأخطاء الكبيرة (مربع)، MAE تعاملها خطيًا، Huber وسط: ناعمة قرب الصفر ومقاومة للشواذ.",
             example_ar="أخطاء (0, 2, 50): MSE = 834.7، MAE = 17.3، Huber(δ=5) ≈ 79.5.",
             dl_link_ar="التدرج بالنسبة للتنبؤ: MSE ∝ e (يكبر مع الخطأ: خطر انفجار مع الشواذ)، MAE = ±1 ثابت (لا يبطئ قرب الحل)، Huber يجمع الحسنيين. خيار الخسارة = افتراض عن ضوضاء الهدف (درس الاحتمال الأرجح).", title_ar="خسائر الانحدار")
    e = np.linspace(-6, 6, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e, y=e ** 2, name="MSE (e²)", line=dict(color="#C8473A", width=3)))
    fig.add_trace(go.Scatter(x=e, y=np.abs(e), name="MAE (|e|)", line=dict(color="#2F6FB5", width=3)))
    fig.add_trace(go.Scatter(x=e, y=np.where(np.abs(e) <= 2, 0.5 * e ** 2, 2 * (np.abs(e) - 1)), name="Huber (δ=2)", line=dict(color="#1F7A78", width=3)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="error e", yaxis=dict(range=[0, 20]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="reg_loss_fig")
    code_lab(CodeLab(
        key="loss_reg", title_ar="ثلاث خسائر وتدرجاتها مع قيمة شاذة، وما يقلل كلًّا منها", code=CODE,
        before=Before(goal_ar="حساب MSE/MAE/Huber على أخطاء فيها شاذ، مقارنة تدرجاتها، وإثبات أن MSE تُقلَّل بالمتوسط وMAE بالوسيط.", stage_ar="الخسارة.",
                      inputs_ar="6 قيم إحداها شاذة (150) وتنبؤ ثابت 100.", expected_ar="MSE ≈ 419 (يهيمن عليه الشاذ)، MAE ≈ 9.3، Huber ≈ 40؛ تدرج MSE للشاذ 16.7 مقابل 0.17 لـ MAE؛ argmin MSE = المتوسط (108.3)، argmin MAE = الوسيط (100.5)."),
        explain=[("2-8", "الثلاث بصيغها. لاحظ أن RMSE بوحدة الهدف."), ("11-16", "التدرج بالنسبة للتنبؤ لكل ملاحظة: في MSE الشاذ يساهم بـ 100 ضعف مساهمة خطأ 0.5؛ في MAE الجميع ±1."),
                 ("19-21", "النتيجة النظرية: أفضل تنبؤ ثابت تحت MSE هو المتوسط، وتحت MAE الوسيط. لذلك MAE «مقاومة».")],
        run=run_printed(CODE),
        after_ar="- إن كانت الشواذ أخطاء قياس اختر MAE/Huber؛ إن كانت حقيقية ومهمة (خسائر كبيرة نادرة) فقد تكون MSE مناسبة.\n- التقرير بـ RMSE أو MAE بوحدة الهدف؛ MSE بوحدة مربعة يصعب تفسيرها.",
    ))
    compare_table(["الخسارة", "التدرج", "الشواذ", "ما تقلّله", "متى"],
                  [("MSE", "∝ e", "حساسة جدًا", "المتوسط الشرطي", "ضوضاء طبيعية، شواذ قليلة"), ("MAE", "±1", "مقاومة", "الوسيط الشرطي", "شواذ كثيرة، وحدة مفهومة"), ("Huber", "e ثم ±δ", "مقاومة", "بين الاثنين", "افتراضي جيد للانحدار العملي")],
                  ["ltr", "code", "rtl", "rtl", "rtl"])
    practical_note("هدف بذيل طويل (أسعار، دخل): درّب على `log(1 + y)` بـ MSE ثم أعد التحويل — يقلل هيمنة القيم الكبيرة ويجعل الخطأ نسبيًا.")
    st.button("افتح معمل دوال الخسارة", icon=":material/science:", on_click=goto, args=("labs.loss_lab",), key="reg_loss_lab")
    common_mistake("تقييم انحدار بالدقة (accuracy): «التنبؤ 99.7 والحقيقة 100 = خطأ». الانحدار يُقاس بـ RMSE/MAE/R²، لا بالتطابق.")
    quiz("loss.reg", [
        Q("أخطاء (1, −1, 10): أي خسارة يهيمن عليها الخطأ 10 أكثر؟", ["MAE", "MSE", "متساويتان"], 1, "المربع."),
        Q("التنبؤ الثابت الذي يقلل MAE…", ["المتوسط", "الوسيط", "المنوال"], 1, "مقاومة."),
        Q("تدرج MAE بالنسبة للتنبؤ…", ["يكبر مع الخطأ", "ثابت الحجم ±1", "صفر"], 1, "إشارة الخطأ."),
        Q("Huber مع δ كبير جدًا تصبح…", ["MAE", "MSE", "صفرًا"], 1, "المنطقة التربيعية تغطي الكل."),
    ])
    takeaway("MSE مربع وحساس، MAE مطلق ومقاوم، Huber وسط بعتبة δ. الخسارة تفترض ضوضاء؛ التقرير بوحدة الهدف.")
    lesson_footer(LESSON, ["الصيغ والتدرجات.", "المتوسط مقابل الوسيط.", "log للذيول الطويلة."])
