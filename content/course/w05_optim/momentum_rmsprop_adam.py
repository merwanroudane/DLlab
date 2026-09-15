import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, research_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.fw import keras_mlp_run

LESSON = Lesson(
    id="course.w05.momentum_rmsprop_adam",
    title_ar="الزخم وRMSprop وAdam: ما يعالجه كل منها، وسباق المحسّنات",
    title_en="Momentum, RMSprop & Adam: What Each Fixes, and the Optimizer Race",
    module="course.w05",
    order=3,
    prerequisites=["course.w05.gd_lr_sgd", "foundations.optim.momentum_nesterov", "foundations.optim.rmsprop_adam"],
    objectives_ar=["المعادلات الثلاث وما تعالجه: الزخم (التذبذب والبطء)، RMSprop (المقاييس المختلفة)، Adam (كلاهما).", "سباق حقيقي على شبكة Keras: نفس البذرة والدفعات، أربعة محسّنات.", "قواعد الاختيار والافتراضات العملية."],
    terms=["optimizer", "learning_rate", "gradient"],
    labs=["labs.optimizer_race"],
    difficulty="intermediate",
    summary_ar="الزخم يراكم اتجاهًا؛ RMSprop يقسّم على جذر متوسط مربعات التدرج لكل معلمة؛ Adam يجمعهما. Adam(1e-3) افتراضي آمن؛ SGD+زخم مع جدول قد يعمّم أفضل. في Keras: optimizer=Adam(learning_rate=…).",
)


def render() -> None:
    lesson_header(LESSON)
    why("SGD الخام يتذبذب في الاتجاهات الحادة ويزحف في المسطحة، ويحتاج η واحدًا لكل المعلمات رغم اختلاف مقاييسها. ثلاثة تحسينات تعالج ذلك — وكلها سطر واحد في compile. المهم أن تعرف **ما يعالجه كل واحد** لتختار وتشخّص.")
    h2("المعادلات الثلاث", "The three update rules")
    equation(r"v_t = \beta v_{t-1} + \nabla L, \qquad \theta \leftarrow \theta - \eta\, v_t",
             [("v", "سرعة متراكمة: متوسط أسي للتدرجات السابقة."), (r"\beta", "0.9 عادةً: كم تتذكر من الماضي.")], meaning_ar="الاتجاهات المتسقة تتسارع، والمتذبذبة تتلاشى (تلغي بعضها).", example_ar="واد ضيق: SGD يقفز بين الجدارين؛ الزخم يمضي على طول الوادي.", dl_link_ar="`SGD(learning_rate, momentum=0.9)`", title_ar="الزخم")
    equation(r"s_t = \rho s_{t-1} + (1-\rho)(\nabla L)^2, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t} + \epsilon}\nabla L",
             [("s", "متوسط أسي لمربع التدرج **لكل معلمة**."), (r"\eta/\sqrt{s}", "خطوة فعلية أصغر للمعلمات ذات التدرجات الكبيرة والعكس.")], meaning_ar="معدل تعلم مكيَّف لكل معلمة: يوازن مقاييس مختلفة.", example_ar="معلمة تدرجها 100 وأخرى 0.01: بلا تكييف، η واحد يهدم الأولى أو يجمّد الثانية.", dl_link_ar="`RMSprop(learning_rate)`", title_ar="RMSprop")
    equation(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L, \quad s_t = \beta_2 s_{t-1} + (1-\beta_2)(\nabla L)^2, \quad \theta \leftarrow \theta - \eta\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}",
             [("m", "زخم (β₁ = 0.9)."), ("s", "تكييف RMSprop (β₂ = 0.999)."), (r"\hat m, \hat s", "تصحيح انحياز البداية (الخطوات الأولى).")], meaning_ar="زخم + تكييف لكل معلمة + تصحيح انحياز: افتراضي قوي بلا ضبط كثير.", example_ar="Adam(1e-3) يعمل «معقولًا» على معظم المسائل من أول محاولة.", dl_link_ar="`Adam(learning_rate=1e-3)`؛ `AdamW` مع اضمحلال أوزان مفصول.", title_ar="Adam")
    h2("سباق المحسّنات على شبكة Keras", "Optimizer race on a Keras network")
    c1, c2 = st.columns(2)
    with c1:
        lr = st.select_slider("η لكل المحسّنات", options=[0.001, 0.01, 0.1], value=0.01, key="w05_race_lr")
    with c2:
        epochs = st.slider("epochs", 5, 30, 15, 5, key="w05_race_ep")
    fig = go.Figure(); rows = []
    for name, color in (("sgd", "#B9B2A6"), ("rmsprop", "#7C5CBF"), ("adam", "#1F7A78")):
        r = keras_mlp_run((16,), "relu", name, float(lr), int(epochs), 32, 0)
        e = np.arange(1, len(r["history"]["loss"]) + 1)
        fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name=name, line=dict(color=color, width=2.5)))
        rows.append((name, f"{r['history']['loss'][-1]:.3f}", f"{r['history']['val_loss'][-1]:.3f}", f"{r['history']['val_accuracy'][-1]:.3f}", str(int(np.argmin(r["history"]["val_loss"])) + 1)))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="val_loss", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="w05_race_fig")
    table(["المحسّن", "loss", "val_loss", "val_accuracy", "أفضل حقبة"], rows, ["code", "num", "num", "num", "num"])
    intuition("عند η = 0.001: SGD يزحف بينما Adam يصل — التكييف يكبّر الخطوات الصغيرة. عند η = 0.1: Adam/RMSprop قد يتذبذبان لأن η كبير لهما (خطوتهما الفعلية ≈ η لكل معلمة) بينما SGD يستفيد. **لا يوجد فائز مطلق؛ يوجد زوج (محسّن، η)**.")
    st.button("معمل سباق المحسّنات (سطح خسارة + تحريك)", icon=":material/science:", type="primary", on_click=goto, args=("labs.optimizer_race",), key="w05_lab_race")
    h2("قواعد الاختيار", "Choosing")
    compare_table(["الموقف", "الاختيار", "η البداية", "ملاحظة"],
                  [("مسألة جديدة، لا وقت للضبط", "Adam", "1e-3 (3e-4 للنماذج الكبيرة)", "الافتراضي الآمن في المقرر"), ("رؤية/CNN مع وقت للضبط", "SGD + momentum 0.9 + جدول", "0.01–0.1", "قد يعمّم أفضل قليلًا"), ("تدرجات بمقاييس متباينة جدًا", "RMSprop / Adam", "1e-3", "التكييف ضروري"),
                   ("Adam يتذبذب أو val_loss يسوء", "قلّل η ×3–10، أو AdamW مع weight_decay", "3e-4", "η كبير لـ Adam شائع"), ("RNN/LSTM (الأسابيع 10–12)", "Adam + قصّ التدرج", "1e-3", "الانفجار شائع")],
                  ["rtl", "rtl", "code", "rtl"])
    research_note("Adam يتقارب أسرع في البداية؛ SGD بالزخم وجدول جيد قد يصل إلى حلول تعمّم أفضل في الرؤية الحاسوبية. في الجداول الصغيرة (مشاريع المقرر) الفرق ضمن الضوضاء غالبًا — أبلغ المحسّن وη والبذرة دائمًا (استنساخ).")
    common_mistake("`optimizer='adam'` بالنص ثم الشكوى من عدم التقارب: η الافتراضي 0.001 قد يكون صغيرًا لمسألة صغيرة محجّمة أو كبيرًا لأخرى. استخدم الكائن بـ η صريح، وجرّب ×3 و÷3.")
    quiz("w05.opt", [
        Q("الزخم يعالج…", ["المقاييس المختلفة", "التذبذب والبطء في الاتجاهات المتسقة", "فرط التخصيص"], 1, ""),
        Q("RMSprop يعالج…", ["الضوضاء", "معلمات بتدرجات بمقاييس مختلفة (η مكيَّف)", "الذاكرة"], 1, ""),
        Q("Adam =", ["SGD + L2", "زخم + تكييف لكل معلمة + تصحيح انحياز", "RMSprop بلا η"], 1, ""),
        Q("Adam يتذبذب وval_loss تسوء:", ["زد η", "قلّل η (3e-4)", "أزل الزخم"], 1, ""),
    ])
    takeaway("زخم = اتجاه متراكم؛ RMSprop = η لكل معلمة؛ Adam = الاثنان. الزوج (محسّن، η) هو الاختيار لا المحسّن وحده. Adam(1e-3) بداية؛ SGD+زخم+جدول عند الضبط.")
    lesson_footer(LESSON, ["ثلاث معادلات وما تعالجه.", "سباق حقيقي على Keras.", "قواعد الاختيار."])
