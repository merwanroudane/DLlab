import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.start.model_training_prediction",
    title_ar="النموذج والتدريب والتنبؤ والاستدلال",
    title_en="Model, Training, Prediction & Inference",
    module="foundations.start",
    order=5,
    prerequisites=["foundations.start.ai_ml_dl"],
    objectives_ar=[
        "تعريف النموذج كدالة ذات معلمات قابلة للتعديل.",
        "فهم التدريب كعملية بحث عن أفضل قيم للمعلمات بتقليل الخطأ.",
        "التمييز بين التنبؤ والاستدلال والتدريب.",
        "تجربة نموذج من معلمتين يدويًا على مثال «ساعات المذاكرة ← درجة الامتحان».",
    ],
    terms=["model", "parameter", "training", "prediction", "inference", "loss", "weight", "bias"],
    related=["foundations.start.why_data_math_optimization"],
    labs=["labs.epoch_batch_simulator"],
    difficulty="beginner",
    summary_ar="النموذج دالة بمعلمات؛ التدريب يضبطها لتقليل الخطأ؛ التنبؤ يستخدمها على مدخل جديد.",
)

# The progressive example (spec §27): study hours -> exam score
HOURS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
SCORES = np.array([52.0, 55.0, 61.0, 64.0, 70.0, 74.0, 79.0, 85.0])

CODE = '''import numpy as np

hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)     # المدخل x
scores = np.array([52, 55, 61, 64, 70, 74, 79, 85], dtype=float)  # الهدف y

w = 4.0   # الوزن  (معلمة)
b = 48.0  # الانحياز (معلمة)

predictions = w * hours + b            # التنبؤ: ŷ = w·x + b
errors      = predictions - scores     # الخطأ لكل ملاحظة
mse         = np.mean(errors ** 2)     # متوسط مربع الخطأ (الخسارة)

print("predictions:", predictions.round(1))
print("errors     :", errors.round(1))
print("MSE        :", round(mse, 2))'''


def _controls() -> dict:
    c1, c2 = st.columns(2)
    with c1:
        w = st.slider("الوزن w (ميل الخط)", 0.0, 8.0, 4.0, 0.1, key="ctrl_mtp_w")
    with c2:
        b = st.slider("الانحياز b (نقطة التقاطع)", 30.0, 70.0, 48.0, 0.5, key="ctrl_mtp_b")
    return {"w": w, "b": b}


def _run(p: dict) -> None:
    w, b = p["w"], p["b"]
    pred = w * HOURS + b
    err = pred - SCORES
    mse = float(np.mean(err ** 2))
    st.code(
        f"predictions: {np.round(pred, 1).tolist()}\n"
        f"errors     : {np.round(err, 1).tolist()}\n"
        f"MSE        : {mse:.2f}",
        language="text",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=HOURS, y=SCORES, mode="markers", name="البيانات الحقيقية",
                             marker=dict(size=11, color="#2F6FB5")))
    xs = np.linspace(0, 9, 50)
    fig.add_trace(go.Scatter(x=xs, y=w * xs + b, mode="lines", name=f"النموذج: ŷ = {w:.1f}·x + {b:.1f}",
                             line=dict(color="#7C5CBF", width=3)))
    for h, s, pr in zip(HOURS, SCORES, pred):
        fig.add_trace(go.Scatter(x=[h, h], y=[s, pr], mode="lines", showlegend=False,
                                 line=dict(color="#C8473A", width=1.5, dash="dot")))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Study hours (x)", yaxis_title="Exam score (y)",
        legend=dict(orientation="h", y=1.12), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9",
        title=dict(text=f"MSE = {mse:.2f}  (الخطوط الحمراء المنقطة = الأخطاء)", x=0.5),
    )
    st.plotly_chart(fig, width="stretch", key="mtp_chart")
    df = pd.DataFrame({"hours (x)": HOURS, "score (y)": SCORES, "ŷ = w·x + b": np.round(pred, 1),
                       "error = ŷ − y": np.round(err, 1), "error²": np.round(err ** 2, 1)})
    st.dataframe(df, hide_index=True, width="stretch")


def render() -> None:
    lesson_header(LESSON)

    h2("لماذا نحتاج هذه الكلمات الأربع؟", "Why these four words?")
    why(
        "كل حديث عن التعلم العميق يدور حول: **نموذج** نبنيه، **تدريب** يضبطه، **تنبؤ** نستخرجه، "
        "و**استدلال** نشغّله في الواقع. إن لم تكن هذه الكلمات دقيقة في ذهنك فلن تفهم لاحقًا ما الذي "
        "يفعله `model.fit()` أو `model.predict()`."
    )

    h2("ما هو النموذج؟", "What is a model?")
    definition(
        "**النموذج** `Model` هو **دالة** تأخذ مدخلًا وتعيد مخرجًا، ولها **معلمات** `Parameters` "
        "قابلة للتعديل تحدد سلوكها. قبل التدريب المعلمات عشوائية والنموذج «جاهل»؛ بعد التدريب "
        "تُضبط المعلمات ليصبح المخرج قريبًا من الحقيقة."
    )
    intuition(
        "أبسط نموذج ممكن: خط مستقيم. **ساعات المذاكرة** مدخل، **درجة الامتحان** مخرج، والخط له "
        "معلمتان فقط: ميله `w` ونقطة تقاطعه `b`. تغيير هاتين المعلمتين يغيّر كل التنبؤات."
    )
    equation(
        r"\hat{y} = w \cdot x + b",
        [
            (r"\hat{y}", "التنبؤ (درجة الامتحان المتوقعة). القبعة تعني «تقدير» لا قيمة حقيقية."),
            ("x", "المدخل: ساعات المذاكرة."),
            ("w", "الوزن `Weight`: معلمة يتعلمها النموذج — كم درجة تضيف كل ساعة مذاكرة."),
            ("b", "الانحياز `Bias`: معلمة يتعلمها النموذج — الدرجة المتوقعة عند صفر ساعات."),
        ],
        meaning_ar="التنبؤ يساوي المدخل مضروبًا في الوزن مضافًا إليه الانحياز.",
        example_ar="إذا كان $w = 4$ و $b = 48$ و $x = 5$ ساعات، فإن $\\hat{y} = 4 \\times 5 + 48 = 68$ درجة.",
        dl_link_ar="الخلية العصبية الواحدة هي بالضبط هذه المعادلة (مع عدة مدخلات) متبوعة بدالة تنشيط. الشبكة العميقة ملايين من هذه المعادلات المتراكبة.",
        title_ar="نموذج خطي بمعلمتين",
    )

    h2("ما هو التدريب؟", "What is training?")
    definition(
        "**التدريب** `Training` هو عملية البحث عن قيم المعلمات (`w`, `b`) التي تجعل تنبؤات النموذج "
        "أقرب ما يمكن إلى القيم الحقيقية في **بيانات التدريب**. «الأقرب» يُقاس برقم واحد اسمه "
        "**الخسارة** `Loss`؛ التدريب = تقليل الخسارة."
    )
    pipeline(["Data", "Model (w, b)", "Prediction ŷ", "Error ŷ − y", "Loss", "Adjust w, b", "Repeat"], active=5)
    equation(
        r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{y}_i - y_i\right)^2",
        [
            (r"\text{MSE}", "متوسط مربع الخطأ `Mean Squared Error`: الخسارة التي نريد تقليلها."),
            ("n", "عدد الملاحظات في بيانات التدريب."),
            (r"\hat{y}_i", "تنبؤ النموذج للملاحظة رقم $i$."),
            ("y_i", "القيمة الحقيقية للملاحظة رقم $i$."),
            (r"(\hat{y}_i - y_i)^2", "مربع الخطأ: موجب دائمًا، ويعاقب الأخطاء الكبيرة أكثر."),
        ],
        meaning_ar="نحسب الخطأ لكل ملاحظة، نربّعه، ثم نأخذ المتوسط.",
        example_ar="ثلاث ملاحظات بأخطاء $2, -1, 3$: المربعات $4, 1, 9$، المجموع $14$، المتوسط $14/3 \\approx 4.67$.",
        dl_link_ar="كل شبكة عصبية تُدرَّب بتقليل خسارة ما؛ MSE للانحدار، والإنتروبيا المتقاطعة للتصنيف. سنشرح كل واحدة في وحدة الخسارة.",
        title_ar="الخسارة: متوسط مربع الخطأ",
    )

    h3("جرّب التدريب يدويًا", "Train by hand")
    st.markdown(
        "أنت الآن **خوارزمية التدريب**. حرّك `w` و`b` وراقب الخسارة. هدفك أصغر `MSE` ممكن. "
        "لاحظ أن كل تغيير في معلمة واحدة يغيّر **كل** التنبؤات."
    )
    code_lab(CodeLab(
        key="mtp_manual",
        title_ar="نموذج خطي وخسارته — بأيدينا",
        level="A",
        code=CODE,
        before=Before(
            goal_ar="حساب تنبؤات نموذج خطي وخسارته على 8 ملاحظات.",
            stage_ar="النموذج + التقييم (لا يوجد تدريب آلي بعد؛ أنت من يعدّل المعلمات).",
            inputs_ar="`hours` مصفوفة بالشكل `(8,)` من نوع `float64`، و`scores` بنفس الشكل.",
            expected_ar="مصفوفة تنبؤات بالشكل `(8,)`، مصفوفة أخطاء بالشكل `(8,)`، ورقم واحد `MSE`.",
            math_ar="$\\hat{y} = w x + b$ ثم $\\text{MSE} = \\frac{1}{n}\\sum(\\hat{y}_i - y_i)^2$.",
        ),
        explain=[
            ("1", "نستورد `NumPy` للحساب على مصفوفات كاملة دفعة واحدة (بدل حلقة `for`)."),
            ("3-4", "المدخل `hours` والهدف `scores`: مصفوفتان بالشكل `(8,)`. `dtype=float` مهم لأن المعلمات كسرية."),
            ("6-7", "المعلمتان. قيمهما هنا يدوية؛ التدريب الآلي سيبحث عنهما بنفسه."),
            ("9", "التنبؤ لكل الملاحظات في سطر واحد: `NumPy` يضرب كل عنصر في `w` ويضيف `b` (بث Broadcasting)."),
            ("10", "الخطأ = التنبؤ − الحقيقة. الإشارة تخبرنا هل النموذج يبالغ أم يقلّل."),
            ("11", "الخسارة: نربّع الأخطاء ثم نأخذ المتوسط. رقم واحد يلخص جودة المعلمتين."),
            ("13-15", "الطباعة. لاحظ الشكل: 8 تنبؤات، 8 أخطاء، خسارة واحدة."),
        ],
        controls=_controls,
        run=_run,
        after_ar=(
            "- **الخطوط الحمراء المنقطة** هي الأخطاء الفردية؛ الخسارة هي متوسط مربعاتها.\n"
            "- إذا كانت كل الأخطاء موجبة فالخط أعلى من البيانات: قلّل `b` أو `w`.\n"
            "- إذا كانت الأخطاء موجبة على اليمين وسالبة على اليسار فالميل كبير: قلّل `w`.\n"
            "- أفضل قيم تقريبًا: `w ≈ 4.7` و`b ≈ 46.4` بخسارة قريبة من `1.2`. التدريب الآلي (الانحدار التدريجي) يصل إليها بحساب اتجاه التعديل بدل التخمين — وهذا موضوع وحدة التحسين."
        ),
    ))

    h2("التنبؤ والاستدلال", "Prediction & inference")
    compare_table(
        ["المصطلح", "English", "متى؟", "هل تتغير المعلمات؟", "مثال"],
        [
            ("التدريب", "Training", "قبل الاستخدام", "نعم، هذا هدفه", "ضبط w و b على 8 طلاب"),
            ("التنبؤ", "Prediction", "عند إعطاء مدخل للنموذج", "لا", "x = 6.5 ساعة ← ŷ ≈ 77"),
            ("الاستدلال", "Inference", "تشغيل النموذج المدرَّب في الواقع/الإنتاج", "لا", "تطبيق يتنبأ لكل طالب جديد"),
        ],
        ["rtl", "ltr", "rtl", "rtl", "rtl"],
    )
    st.markdown(
        "**التنبؤ** هو الفعل الواحد (مدخل ← مخرج). **الاستدلال** هو مرحلة الاستخدام كلها بعد انتهاء التدريب؛ "
        "في هذه المرحلة المعلمات **مجمّدة**. في أطر العمل ستقابل `model.predict()` للتنبؤ، و`model.eval()` "
        "أو `inference_mode` لإعلان أننا في مرحلة الاستدلال."
    )
    common_mistake(
        "الخلط بين «النموذج يتنبأ جيدًا على بيانات التدريب» و«النموذج مفيد». المقياس الحقيقي هو الأداء على "
        "بيانات **لم يرها** أثناء التدريب. هذا موضوع التعميم والتحقق لاحقًا."
    )

    quiz(
        "start.model_training_prediction",
        [
            Q("ما النموذج بأبسط تعريف؟",
              ["مجموعة بيانات كبيرة", "دالة ذات معلمات قابلة للتعديل تحوّل مدخلًا إلى مخرج", "برنامج يكتب القواعد يدويًا"], 1,
              "النموذج دالة بمعلمات."),
            Q("في $\\hat{y} = w x + b$، كان $w = 3$ و$b = 50$ و$x = 4$. ما التنبؤ؟",
              ["62", "57", "200"], 0, "3 × 4 + 50 = 62.", kind="equation"),
            Q("ماذا يحدث للمعلمات أثناء الاستدلال؟",
              ["تتغير مع كل تنبؤ", "تبقى مجمّدة", "تُحذف"], 1,
              "الاستدلال يستخدم معلمات مدرَّبة ثابتة."),
            Q("ثلاث ملاحظات بأخطاء 1، −2، 2. ما MSE؟",
              ["1", "3", "9"], 1, "المربعات 1، 4، 4 ← المجموع 9 ← المتوسط 3.", kind="shape"),
        ],
    )
    takeaway("النموذج دالة بمعلمات. التدريب يقلّل الخسارة بتعديل المعلمات. التنبؤ يستخدمها، والاستدلال يشغّلها مجمّدة.")
    lesson_footer(LESSON, [
        "النموذج = دالة + معلمات (مثل w و b).",
        "التدريب = البحث عن معلمات تقلّل الخسارة على بيانات التدريب.",
        "الخسارة رقم واحد يلخص الخطأ (مثل MSE).",
        "التنبؤ فعل واحد؛ الاستدلال مرحلة الاستخدام بمعلمات مجمّدة.",
    ])
