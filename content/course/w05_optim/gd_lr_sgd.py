import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, interpretation_note, intuition, math_note, takeaway, warning_note, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from content.course.w05_optim._viz import noise_svg, parabola_path, parabola_svg, sgd_noise
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
    objectives_ar=[
        "مشاهدة أربع قيم لـ η على نفس القطع المكافئ خطوة بخطوة: بطء، تقارب، تذبذب، انفجار.",
        "اشتقاق شرط التقارب η < 1/a خطوة بخطوة وفهم علاقته بانحناء السطح.",
        "رؤية ضوضاء SGD: مسارات الدفعة الكاملة والدفعة 8 والدفعة 1 على نفس سطح الخسارة.",
        "نقل الأثر إلى شبكة Keras بـ SGD وقراءة منحنيات الخسارة.",
    ],
    terms=["gradient", "learning_rate", "batch_size", "iteration", "stochastic_gradient_descent", "batch", "epoch"],
    labs=["labs.gradient_descent_lab", "labs.learning_rate_lab", "labs.epoch_batch_simulator"],
    difficulty="intermediate",
    summary_ar="θ ← θ − η∇L. η صغير = بطيء، مناسب = تقارب، كبير = تذبذب/انفجار؛ الحد η < 1/a يحدده الانحناء. SGD يقدّر ∇L على دفعة: تحديثات أكثر لكل حقبة وضوضاء أكثر. في Keras: optimizer=SGD(learning_rate=η).",
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


def _conv_steps() -> list[tuple[str, str, str]]:
    return [
        ("**الدالة**: قطع مكافئ بانحناء `a` وقاع عند `w*`. كل سطح خسارة أملس يشبه هذا الشكل قرب قاعه.", "L(w) = a(w − w*)² + c", r"L(w) = a\,(w-w^*)^2 + c"),
        ("**التدرج**: خطي في المسافة إلى القاع.", "L'(w) = 2a(w − w*)", r"L'(w) = 2a\,(w-w^*)"),
        ("**خطوة واحدة**: نعوّض في قاعدة التحديث ونطرح w* من الطرفين لنتابع **الخطأ** `eₜ = wₜ − w*`.", "w_{t+1} − w* = (w_t − w*) − η·2a(w_t − w*)", r"w_{t+1}-w^* = (w_t - w^*) - 2a\eta\,(w_t-w^*)"),
        ("**الخطأ يُضرب في ثابت** كل خطوة: `r = 1 − 2aη`. بعد t خطوة: `eₜ = rᵗ·e₀`.", "e_{t+1} = (1 − 2aη)·e_t   ⇒   e_t = (1 − 2aη)^t · e_0", r"e_{t+1} = (1-2a\eta)\,e_t \;\Rightarrow\; e_t = (1-2a\eta)^t\, e_0"),
        ("**شرط التقارب**: `|r| < 1` ⇔ `0 < η < 1/a`. إن كان r سالبًا (η > 1/(2a)) تتبدل الإشارة كل خطوة = **تذبذب**. إن |r| > 1 = **انفجار**.", "|1 − 2aη| < 1  ⇔  0 < η < 1/a", r"|1-2a\eta| < 1 \iff 0 < \eta < \tfrac{1}{a}"),
        ("**بأرقام الدرس** (a = 1): r = 0.9 عند η = 0.05 (بطيء)، 0.4 عند 0.3 (سريع)، −0.8 عند 0.9 (يتذبذب ويتقارب)، −1.1 عند 1.05 (ينفجر).", "η: 0.05→r=0.9 · 0.3→0.4 · 0.9→−0.8 · 1.05→−1.1", r"\eta=0.05\Rightarrow r=0.9,\;\; \eta=0.3\Rightarrow r=0.4,\;\; \eta=0.9\Rightarrow r=-0.8,\;\; \eta=1.05\Rightarrow r=-1.1"),
    ]


def render() -> None:
    lesson_header(LESSON)
    equation(r"\theta_{t+1} = \theta_t - \eta \,\nabla_\theta L(\theta_t)",
             [(r"\nabla_\theta L", "اتجاه أشد صعود للخسارة (الأسس 5)؛ نسير عكسه."), (r"\eta", "معدل التعلم: طول الخطوة. المعلمة الفائقة الأولى."), (r"\theta_t", "كل المعلمات (الأوزان والانحيازات) في الخطوة t.")],
             meaning_ar="كل خطوة تحرّك كل معلمة عكس تدرجها بمقدار يتناسب مع η وحجم التدرج.", example_ar="w = −2، ∇L = 2(w−3) = −10، η = 0.3 → w = −2 + 3 = 1.", dl_link_ar="`keras.optimizers.SGD(learning_rate=0.3)` تطبّق هذا السطر لكل معلمة في كل دفعة.", title_ar="خطوة الانحدار التدرجي")
    definition("**الانحدار التدرجي** `Gradient descent`: خوارزمية تقليل الخسارة بخطوات عكس التدرج. **معدل التعلم** η: طول الخطوة. **SGD** (العشوائي): نفس الخطوة لكن التدرج يُحسب على **دفعة** لا على كل البيانات — تقدير أسرع وأكثر ضوضاء (الأسس 17).")

    # ------------------------------------------------------------------ parabola animation
    h2("أربعة معدلات تعلم على نفس الوادي", "Four learning rates on the same valley")
    lrs = (0.05, 0.3, 0.9, 1.05)
    paths = {lr: parabola_path(lr) for lr in lrs}
    pcaps = []
    for k in range(13):
        ws = {lr: paths[lr][min(k, len(paths[lr]) - 1)] for lr in lrs}
        if k == 0:
            pcaps.append("**البداية**: أربع كرات عند `w = −2` على `L(w) = (w − 3)² + 1`. القاع عند `w = 3`. الفرق الوحيد: طول الخطوة η.")
        else:
            pcaps.append(f"**الخطوة {k}**: η = 0.05 عند {ws[0.05]:.2f} (يزحف)؛ η = 0.3 عند {ws[0.3]:.3f} (وصل تقريبًا)؛ η = 0.9 عند {ws[0.9]:.2f} (يقفز عبر القاع ذهابًا وإيابًا لكنه يقترب)؛ "
                         f"η = 1.05 عند {ws[1.05]:.2f} (كل قفزة **أبعد** من السابقة).")
    animation_player("w05_parabola", [Frame(parabola_svg(paths, k), caption(c), action=f"step {k}",
                                            values=[(f"w (η={lr})", f"{paths[lr][max(0, k - 1)]:.3f}", f"{paths[lr][k]:.3f}") for lr in lrs])
                                      for k, c in enumerate(pcaps)], title_ar="L(w) = (w − 3)² + 1، بداية w = −2", interval_ms=1200)
    h3("لماذا 1.0 هو الحد هنا؟ اشتقاق شرط التقارب", "Why is 1.0 the limit? Deriving the condition")
    steps = _conv_steps()
    cur = animation_player("w05_conv", [Frame("", caption(c), action=f"step {i + 1}", equation=u) for i, (c, u, _) in enumerate(steps)],
                           title_ar="من قاعدة التحديث إلى η < 1/a", interval_ms=2600)
    with st.container(border=True):
        st.markdown(f"**الخطوة {cur + 1} من {len(steps)}**")
        st.latex(steps[cur][2])
    math_note("في الشبكة الحقيقية السطح ليس قطعًا مكافئًا واحدًا، لكن قرب أي قاع يشبه قطعًا مكافئًا متعدد الأبعاد، وأكبر انحناء (أشد اتجاه) هو الذي يحدد η الأقصى. "
              "لذلك: **مدخلات غير محجّمة ⇒ انحناء كبير في اتجاه ⇒ η صغير إجباري ⇒ بطء في كل الاتجاهات الأخرى**. هذا سبب التحجيم الرياضي.")
    code_lab(CodeLab(
        key="w05_gd", title_ar="أربع قيم لـ η على دالة بسيطة: بطء، تقارب، تذبذب، انفجار", code=CODE, level="A",
        before=Before(goal_ar="مشاهدة الأثر الثلاثي لمعدل التعلم بالأرقام على L(w) = (w−3)² + 1، واستنتاج شرط التقارب.", stage_ar="الأسبوع 05: GD وη.", inputs_ar="بداية w = −2، 12 خطوة.", expected_ar="0.05 بطيء (لم يصل)، 0.3 يصل بسلاسة، 0.9 يتذبذب حول 3 ويقترب، 1.05 ينفجر."),
        explain=[("3", "التدرج التحليلي (الأسس 5)."), ("4-9", "لكل η: 12 خطوة من نفس البداية؛ نطبع المسار الأول ونصنّف السلوك."), ("10", "لدالة تربيعية بانحناء a، الشرط η < 1/a. الشبكات ليست تربيعية لكن الحدس نفسه: η كبير مقابل انحناء السطح = انفجار.")],
        run=run_printed(CODE),
        after_ar="- المسار عند 0.9: 3 ± مبالغة تتناقص — هذا «التذبذب المتقارب» الذي تراه كتموّج في منحنى الخسارة.\n- عند 1.05 القيم تكبر بلا حدود: `loss: nan` في Keras بعد حقب قليلة.\n- لا يوجد η صحيح مطلق: يعتمد على انحناء السطح، الذي يعتمد على البيانات وتحجيمها والبنية.",
    ))

    # ------------------------------------------------------------------ SGD noise
    h2("ضوضاء SGD: نفس السطح، ثلاثة أحجام دفعة", "SGD noise: same surface, three batch sizes")
    st.markdown("انحدار خطي على بيانات الأربعين طالبًا (محجّمة)، والمحوران هما الوزن `w` والانحياز `b`. نفس η = 0.1 ونفس البداية؛ الاختلاف حجم الدفعة فقط.")
    nd = sgd_noise()
    ncaps = []
    for ep in range(7):
        if ep == 0:
            ncaps.append("**البداية**: ثلاث نسخ عند نفس النقطة. النقطة الداكنة = الحل الأمثل.")
        else:
            ncaps.append(f"**بعد الحقبة {ep}**: الدفعة الكاملة أجرت {ep} تحديثات فقط (خط أملس بطيء)؛ الدفعة 8 أجرت {ep * 5} (متعرج قليلًا وأسرع)؛ الدفعة 1 أجرت {ep * 40} "
                         + ("(وصلت بسرعة ثم **ترتجف** حول القاع: كل ملاحظة تشد باتجاه مختلف)." if ep >= 2 else "(متعرجة جدًا لكنها تتقدم بسرعة)."))
    animation_player("w05_noise", [Frame(noise_svg(nd, ep), caption(c), action=f"epoch {ep}") for ep, c in enumerate(ncaps)],
                     title_ar="GD مقابل mini-batch مقابل SGD", interval_ms=1600)
    compare_table(["حجم الدفعة", "تحديثات/حقبة", "ضوضاء التدرج", "الذاكرة", "ملاحظة"],
                  [("كل البيانات (GD)", "1", "لا شيء", "الأعلى", "أملس لكنه بطيء لكل حقبة"), ("32–256 (الشائع)", "n/B", "متوسطة", "متوسطة", "توازن جيد، يناسب GPU"),
                   ("1 (SGD الخالص)", "n", "عالية", "الأدنى", "يرتجف حول القاع إلا إذا خُفِّض η")],
                  ["rtl", "ltr", "rtl", "rtl", "rtl"])
    interpretation_note("الضوضاء ليست شرًا خالصًا: تساعد على الخروج من الهضاب والقيعان الضحلة، ويُعتقد أنها تميل إلى قيعان «أعرض» تعمّم أفضل. لكنها تمنع الاستقرار التام — لذلك نخفض η تدريجيًا (الجداول، الدرس 3).")

    # ------------------------------------------------------------------ Keras
    h2("نفس الأثر على شبكة Keras", "The same effect on a Keras network")
    st.markdown("MLP (2 → 8 → 1) على moons بـ **SGD** وأربع قيم لـ η؛ نفس البذرة والدفعات:")
    bs = st.select_slider("batch_size", options=[8, 32, 150, 450], value=32, key="w05_bs")
    try:
        fig = go.Figure(); rows = []
        for lr, color in ((0.01, "#94A3B8"), (0.1, "#2563EB"), (1.0, "#059669"), (3.0, "#DC2626")):
            r = keras_mlp_run((8,), "relu", "sgd", float(lr), 20, int(bs), 0)
            e = np.arange(1, len(r["history"]["loss"]) + 1)
            fig.add_trace(go.Scatter(x=e, y=r["history"]["loss"], name=f"η = {lr}", line=dict(color=color, width=2.5)))
            h = r["history"]["loss"]; rows.append((str(lr), f"{h[0]:.3f}", f"{h[-1]:.3f}", f"{r['history']['val_accuracy'][-1]:.3f}", "انفجار/nan" if (np.isnan(h[-1]) or h[-1] > h[0]) else ("بطيء" if h[-1] > 0.5 else "تقارب")))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="epoch", yaxis_title="loss (train)", legend=dict(orientation="h", x=0, y=1.15))
        st.plotly_chart(fig, width="stretch", key="w05_lr_fig")
        table(["η", "loss الحقبة 1", "loss الحقبة 20", "val_accuracy", "السلوك"], rows, ["num", "num", "num", "num", "rtl"])
    except Exception as exc:  # noqa: BLE001 - TensorFlow missing on this runtime
        warning_note(f"TensorFlow غير متاح هنا ({type(exc).__name__}). التحريكات أعلاه تعرض نفس الأثر بمحرك NumPy.")
    intuition(f"مع batch_size = {bs}: كل حقبة = {int(np.ceil(450 / bs))} تحديثًا. صغّر الدفعة إلى 8 لترى منحنيات أكثر ضوضاء وتعلمًا أسرع لكل حقبة (تحديثات أكثر)؛ كبّرها إلى 450 (دفعة كاملة = GD) لترى منحنى أملس وبطيئًا. η المناسب يتغير مع حجم الدفعة.")
    with st.container(horizontal=True):
        st.button("معمل الانحدار التدرجي", icon=":material/science:", on_click=goto, args=("labs.gradient_descent_lab",), key="w05_lab_gd")
        st.button("معمل معدل التعلم", icon=":material/science:", on_click=goto, args=("labs.learning_rate_lab",), key="w05_lab_lr")
        st.button("محاكي الدفعة/الحقبة", icon=":material/science:", on_click=goto, args=("labs.epoch_batch_simulator",), key="w05_lab_eb")
    why("لماذا نبدأ بـ SGD لا Adam؟ لأن Adam يخفي أثر η جزئيًا (يكيّفه لكل معلمة). من يرى الأثر الخام في SGD يفهم ما يعالجه الزخم وRMSprop في الدرس التالي — ويعرف لماذا يبقى η مهمًا حتى مع Adam.")
    common_mistake("مدخل غير محجّم (دخل بالآلاف بجانب نسبة بين 0 و1) يجعل سطح الخسارة مستطيلًا جدًا: η المناسب لبُعد ينفجر في الآخر. التحجيم (الأسس 8) هو أول علاج لمشاكل η، قبل أي محسّن.")
    quiz("w05.gd", [
        Q("η = 0.9 على L = (w−3)²:", ["ينفجر", "يتذبذب ويتقارب", "لا يتحرك"], 1, "r = 1 − 1.8 = −0.8."),
        Q("`loss: nan` بعد حقبتين مع SGD:", ["η صغير", "η كبير (انفجار)", "بيانات قليلة"], 1, ""),
        Q("batch_size = n (كل البيانات) يعطي…", ["SGD ضوضائيًا", "GD أملس وبطيئًا لكل حقبة", "خطأ"], 1, ""),
        Q("أول علاج لسطح خسارة مستطيل:", ["Adam", "تحجيم المدخلات", "حقب أكثر"], 1, ""),
        Q("L = 4(w − 1)²: أكبر η يضمن التقارب", ["0.25", "0.5", "4"], 0, "a = 4 ⇒ η < 1/4."),
        Q("مع الدفعة 1، المسار قرب القاع…", ["يستقر تمامًا", "يرتجف حوله", "ينفجر"], 1, "ضوضاء التقدير."),
        Q("1000 ملاحظة، batch 50: تحديثات لكل حقبة", ["50", "20", "1000"], 1, ""),
    ])
    takeaway("θ ← θ − η∇L. الخطأ يُضرب في (1 − 2aη) كل خطوة: η < 1/a شرط التقارب، والانحناء الأكبر يحدده. SGD يقدّر التدرج على دفعة: تحديثات أكثر وضوضاء أكثر. التحجيم أولًا؛ η ثانيًا؛ المحسّن ثالثًا.")
    lesson_footer(LESSON, ["أربعة η على قطع مكافئ (تحريك).", "اشتقاق شرط التقارب (تحريك).", "الكود بالأرقام.", "ضوضاء SGD بثلاثة أحجام دفعة (تحريك).", "نفس الأثر على Keras."])
