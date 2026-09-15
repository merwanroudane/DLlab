import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.fw import keras_mlp_run

LESSON = Lesson(
    id="course.w05.gd_lr_sgd",
    title_ar="الانحدار التدرجي ومعدل التعلم وSGD: من المعادلة إلى compile",
    title_en="Gradient Descent, Learning Rate & SGD: From the Equation to compile()",
    module="course.w05",
    order=2,
    prerequisites=["course.w05.overview", "foundations.optim.gd_variants", "foundations.optim.learning_rate"],
    objectives_ar=["خطوة الانحدار التدرجي على دالة بسيطة بثلاث قيم لـ η، ثم على شبكة Keras بـ SGD.", "قراءة أثر η من منحنى الخسارة: بطء، تقارب، تذبذب، انفجار.", "SGD كتقدير للتدرج على دفعة، وأثر حجم الدفعة على الضوضاء."],
    terms=["gradient", "learning_rate", "batch_size", "iteration"],
    labs=["labs.gradient_descent_lab", "labs.learning_rate_lab"],
    difficulty="intermediate",
    summary_ar="θ ← θ − η∇L. η صغير = بطيء، مناسب = تقارب، كبير = تذبذب/انفجار. SGD يقدّر ∇L على دفعة: أسرع وأكثر ضوضاء. في Keras: optimizer=SGD(learning_rate=η).",
)

CODE = '''import numpy as np
# دالة خسارة بسيطة: L(w) = (w - 3)² + 1 ; ∇L = 2(w - 3) ; الحد الأدنى عند w = 3
grad = lambda w: 2 * (w - 3)
for lr in (0.05, 0.3, 0.9, 1.05):
    w = -2.0; path = [w]
    for step in range(12):
        w = w - lr * grad(w); path.append(w)
    status = "diverges" if abs(path[-1] - 3) > abs(path[0] - 3) else ("oscillates" if lr > 0.5 else "converges")
    print(f"lr={lr:<5} w after 12 steps = {path[-1]:>10.4f}   path: {np.round(path[:6], 2).tolist()} ...  -> {status}")
print("\\nrule: for L = a(w-w*)², steps converge iff lr < 2/(2a) = 1/a = 1.0 here; lr < 0.5 monotone; lr in (0.5, 1) oscillates but converges")'''


def render() -> None:
    lesson_header(LESSON)
    equation(r"\theta_{t+1} = \theta_t - \eta \,\nabla_\theta L(\theta_t)",
             [(r"\nabla_\theta L", "اتجاه أشد صعود للخسارة (الأسس 5)؛ نسير عكسه."), (r"\eta", "معدل التعلم: طول الخطوة. المعلمة الفائقة الأولى."), (r"\theta_t", "كل المعلمات (الأوزان والانحيازات) في الخطوة t.")],
             meaning_ar="كل خطوة تحرّك كل معلمة عكس تدرجها بمقدار يتناسب مع η وحجم التدرج.", example_ar="w = −2، ∇L = 2(w−3) = −10، η = 0.3 → w = −2 + 3 = 1.", dl_link_ar="`keras.optimizers.SGD(learning_rate=0.3)` تطبّق هذا السطر لكل معلمة في كل دفعة.", title_ar="خطوة الانحدار التدرجي")
    definition("**الانحدار التدرجي** `Gradient descent`: خوارزمية تقليل الخسارة بخطوات عكس التدرج. **معدل التعلم** η: طول الخطوة. **SGD** (العشوائي): نفس الخطوة لكن التدرج يُحسب على **دفعة** لا على كل البيانات — تقدير أسرع وأكثر ضوضاء (الأسس 17).")
    code_lab(CodeLab(
        key="w05_gd", title_ar="أربع قيم لـ η على دالة بسيطة: بطء، تقارب، تذبذب، انفجار", code=CODE, level="A",
        before=Before(goal_ar="مشاهدة الأثر الثلاثي لمعدل التعلم بالأرقام على L(w) = (w−3)² + 1، واستنتاج شرط التقارب.", stage_ar="الأسبوع 05: GD وη.", inputs_ar="بداية w = −2، 12 خطوة.", expected_ar="0.05 بطيء (لم يصل)، 0.3 يصل بسلاسة، 0.9 يتذبذب حول 3 ويقترب، 1.05 ينفجر."),
        explain=[("3", "التدرج التحليلي (الأسس 5)."), ("4-9", "لكل η: 12 خطوة من نفس البداية؛ نطبع المسار الأول ونصنّف السلوك."), ("10", "لدالة تربيعية بانحناء a، الشرط η < 1/a. الشبكات ليست تربيعية لكن الحدس نفسه: η كبير مقابل انحناء السطح = انفجار.")],
        run=run_printed(CODE),
        after_ar="- المسار عند 0.9: 3 ± مبالغة تتناقص — هذا «التذبذب المتقارب» الذي تراه كتموّج في منحنى الخسارة.\n- عند 1.05 القيم تكبر بلا حدود: `loss: nan` في Keras بعد حقب قليلة.\n- لا يوجد η صحيح مطلق: يعتمد على انحناء السطح، الذي يعتمد على البيانات وتحجيمها والبنية.",
    ))
    h2("نفس الأثر على شبكة Keras", "The same effect on a Keras network")
    st.markdown("MLP (2 → 8 → 1) على moons بـ **SGD** وأربع قيم لـ η؛ نفس البذرة والدفعات:")
    bs = st.select_slider("batch_size", options=[8, 32, 150, 450], value=32, key="w05_bs")
    fig = go.Figure(); rows = []
    for lr, color in ((0.01, "#B9B2A6"), (0.1, "#2F6FB5"), (1.0, "#1F7A78"), (3.0, "#C8473A")):
        r = keras_mlp_run((8,), "relu", "sgd", float(lr), 20, int(bs), 0)
        e = np.arange(1, len(r["history"]["loss"]) + 1)
        fig.add_trace(go.Scatter(x=e, y=r["history"]["loss"], name=f"η = {lr}", line=dict(color=color, width=2.5)))
        h = r["history"]["loss"]; rows.append((str(lr), f"{h[0]:.3f}", f"{h[-1]:.3f}", f"{r['history']['val_accuracy'][-1]:.3f}", "انفجار/nan" if (np.isnan(h[-1]) or h[-1] > h[0]) else ("بطيء" if h[-1] > 0.5 else "تقارب")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="loss (train)", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="w05_lr_fig")
    table(["η", "loss الحقبة 1", "loss الحقبة 20", "val_accuracy", "السلوك"], rows, ["num", "num", "num", "num", "rtl"])
    intuition(f"مع batch_size = {bs}: كل حقبة = {int(np.ceil(450 / bs))} تحديثًا. صغّر الدفعة إلى 8 لترى منحنيات أكثر ضوضاء وتعلمًا أسرع لكل حقبة (تحديثات أكثر)؛ كبّرها إلى 450 (دفعة كاملة = GD) لترى منحنى أملس وبطيئًا. η المناسب يتغير مع حجم الدفعة.")
    with st.container(horizontal=True):
        st.button("معمل الانحدار التدرجي", icon=":material/science:", on_click=goto, args=("labs.gradient_descent_lab",), key="w05_lab_gd")
        st.button("معمل معدل التعلم", icon=":material/science:", on_click=goto, args=("labs.learning_rate_lab",), key="w05_lab_lr")
        st.button("محاكي الدفعة/الحقبة", icon=":material/science:", on_click=goto, args=("labs.epoch_batch_simulator",), key="w05_lab_eb")
    why("لماذا نبدأ بـ SGD لا Adam؟ لأن Adam يخفي أثر η جزئيًا (يكيّفه لكل معلمة). من يرى الأثر الخام في SGD يفهم ما يعالجه الزخم وRMSprop في الدرس التالي — ويعرف لماذا يبقى η مهمًا حتى مع Adam.")
    common_mistake("مدخل غير محجّم (دخل بالآلاف بجانب نسبة بين 0 و1) يجعل سطح الخسارة مستطيلًا جدًا: η المناسب لبُعد ينفجر في الآخر. التحجيم (الأسس 8) هو أول علاج لمشاكل η، قبل أي محسّن.")
    quiz("w05.gd", [
        Q("η = 0.9 على L = (w−3)²:", ["ينفجر", "يتذبذب ويتقارب", "لا يتحرك"], 1, "بين 0.5 و1."),
        Q("`loss: nan` بعد حقبتين مع SGD:", ["η صغير", "η كبير (انفجار)", "بيانات قليلة"], 1, ""),
        Q("batch_size = n (كل البيانات) يعطي…", ["SGD ضوضائيًا", "GD أملس وبطيئًا لكل حقبة", "خطأ"], 1, ""),
        Q("أول علاج لسطح خسارة مستطيل:", ["Adam", "تحجيم المدخلات", "حقب أكثر"], 1, ""),
    ])
    takeaway("θ ← θ − η∇L. η: بطيء/مناسب/تذبذب/انفجار بحسب انحناء السطح. SGD يقدّر التدرج على دفعة. التحجيم أولًا؛ η ثانيًا؛ المحسّن ثالثًا.")
    lesson_footer(LESSON, ["المعادلة والأثر الثلاثي.", "نفس الأثر على Keras بأربع قيم.", "حجم الدفعة وη."])
