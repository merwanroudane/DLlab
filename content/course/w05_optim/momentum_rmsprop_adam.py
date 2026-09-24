import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, interpretation_note, intuition, math_note, research_note, takeaway, warning_note, why
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from content.course.w05_optim._viz import adam_steps, race_paths, race_svg
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
    objectives_ar=[
        "مشاهدة أربعة محسّنات تتسابق في وادٍ ضيق خطوة بخطوة ورؤية ما يعالجه كل منها.",
        "المعادلات الثلاث وما تعالجه: الزخم (التذبذب والبطء)، RMSprop (المقاييس المختلفة)، Adam (كلاهما).",
        "حساب ثلاث خطوات من Adam يدويًا بما فيها تصحيح الانحياز.",
        "سباق حقيقي على شبكة Keras، وقواعد الاختيار والافتراضات العملية.",
    ],
    terms=["optimizer", "learning_rate", "gradient", "momentum", "adam", "stochastic_gradient_descent"],
    labs=["labs.optimizer_race"],
    difficulty="intermediate",
    summary_ar="الزخم يراكم اتجاهًا؛ RMSprop يقسّم على جذر متوسط مربعات التدرج لكل معلمة؛ Adam يجمعهما مع تصحيح الانحياز. Adam(1e-3) افتراضي آمن؛ SGD+زخم مع جدول قد يعمّم أفضل. الاختيار زوج (محسّن، η).",
)


def render() -> None:
    lesson_header(LESSON)
    why("SGD الخام يتذبذب في الاتجاهات الحادة ويزحف في المسطحة، ويحتاج η واحدًا لكل المعلمات رغم اختلاف مقاييسها. ثلاثة تحسينات تعالج ذلك — وكلها سطر واحد في compile. المهم أن تعرف **ما يعالجه كل واحد** لتختار وتشخّص.")

    # ------------------------------------------------------------------ the race
    h2("السباق في وادٍ ضيق", "The race in a narrow valley")
    st.markdown("سطح خسارة `L = 0.01x² + y²`: شديد الانحدار في y (انحناء 2) ومسطح تقريبًا في x (انحناء 0.02) — نسبة 100 بين الاتجاهين، تمامًا ما يصنعه مدخلان غير محجّمين. البداية (−9، 2) والقاع (0، 0).")
    rd = race_paths()
    shown = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 70, 80]
    rcaps = []
    for k in shown:
        L = {n: rd["loss"][n][k] for n in rd["loss"]}
        if k == 0:
            rcaps.append("**البداية**: كل المحسّنات عند نفس النقطة. الصعوبة: η كبير يكفي للتقدم في x ينفجر في y؛ وη صغير يكفي لـ y يزحف في x.")
        elif k <= 3:
            rcaps.append(f"**الخطوة {k}**: **SGD** (أزرق) يقفز بين جداري الوادي: η = 0.9 قريب من حد الانفجار في y. **الزخم** (برتقالي) يتجاوز ثم يعود. **RMSprop وAdam** يتحركان في x وy بخطوات متقاربة لأنهما يقسّمان على حجم التدرج في كل اتجاه.")
        elif k <= 20:
            rcaps.append(f"**الخطوة {k}**: SGD خمد تذبذبه في y لكنه **يزحف** في x (تدرج x صغير جدًا). الزخم يراكم السرعة في x فيتقدم أسرع. خسائر: SGD {L['SGD']:.3f}، زخم {L['Momentum']:.3f}، RMSprop {L['RMSprop']:.3f}، Adam {L['Adam']:.3f}.")
        else:
            rcaps.append(f"**الخطوة {k}**: SGD {L['SGD']:.4f} ما زال بعيدًا في x. الزخم {L['Momentum']:.4f} يتموّج حول المحور لكنه يصل. RMSprop {L['RMSprop']:.4f} وAdam {L['Adam']:.4f} في القاع تقريبًا.")
    animation_player("w05_race2d", [Frame(race_svg(rd, k), caption(c), action=f"step {k}") for k, c in zip(shown, rcaps)],
                     title_ar="SGD · Momentum · RMSprop · Adam (مسارات محسوبة بالمعادلات الدقيقة)", interval_ms=1100)
    fl = go.Figure()
    for name, col in (("SGD", "#2563EB"), ("Momentum", "#EA580C"), ("RMSprop", "#059669"), ("Adam", "#DB2777")):
        fl.add_scatter(y=rd["loss"][name], mode="lines", name=name, line=dict(color=col, width=2.5))
    fl.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="step", yaxis_title="loss (log scale)", yaxis_type="log", legend=dict(orientation="h", x=0, y=1.15))
    st.plotly_chart(fl, width="stretch", key="w05_race_loss")
    warning_note("كل محسّن هنا بـ η مختلف اخترناه ليعمل جيدًا على هذا السطح. سطح آخر أو η آخر يغيّر الترتيب — **لا تستنتج «Adam هو الأفضل دائمًا»** من سطح واحد.")

    # ------------------------------------------------------------------ equations
    h2("المعادلات الثلاث", "The three update rules")
    equation(r"v_t = \beta v_{t-1} + \nabla L, \qquad \theta \leftarrow \theta - \eta\, v_t",
             [("v", "سرعة متراكمة: متوسط أسي للتدرجات السابقة."), (r"\beta", "0.9 عادةً: كم تتذكر من الماضي.")], meaning_ar="الاتجاهات المتسقة تتسارع، والمتذبذبة تتلاشى (تلغي بعضها).", example_ar="واد ضيق: SGD يقفز بين الجدارين؛ الزخم يمضي على طول الوادي.", dl_link_ar="`SGD(learning_rate, momentum=0.9)`", title_ar="الزخم")
    math_note("افرد السرعة: `v_t = g_t + βg_{t−1} + β²g_{t−2} + …`. في اتجاه ثابت التدرج g يصبح المجموع g/(1 − β) = **10g** عند β = 0.9: خطوة فعلية أكبر بعشر مرات. وفي اتجاه يتبدل تدرجه (+، −، +…) تلغي الحدود بعضها — لهذا يخمد التذبذب.")
    equation(r"s_t = \rho s_{t-1} + (1-\rho)(\nabla L)^2, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t} + \epsilon}\nabla L",
             [("s", "متوسط أسي لمربع التدرج **لكل معلمة**."), (r"\eta/\sqrt{s}", "خطوة فعلية أصغر للمعلمات ذات التدرجات الكبيرة والعكس.")], meaning_ar="معدل تعلم مكيَّف لكل معلمة: يوازن مقاييس مختلفة.", example_ar="معلمة تدرجها 100 وأخرى 0.01: بلا تكييف، η واحد يهدم الأولى أو يجمّد الثانية.", dl_link_ar="`RMSprop(learning_rate)`", title_ar="RMSprop")
    equation(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L, \quad s_t = \beta_2 s_{t-1} + (1-\beta_2)(\nabla L)^2, \quad \theta \leftarrow \theta - \eta\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}",
             [("m", "زخم (β₁ = 0.9)."), ("s", "تكييف RMSprop (β₂ = 0.999)."), (r"\hat m, \hat s", "تصحيح انحياز البداية (الخطوات الأولى).")], meaning_ar="زخم + تكييف لكل معلمة + تصحيح انحياز: افتراضي قوي بلا ضبط كثير.", example_ar="Adam(1e-3) يعمل «معقولًا» على معظم المسائل من أول محاولة.", dl_link_ar="`Adam(learning_rate=1e-3)`؛ `AdamW` مع اضمحلال أوزان مفصول.", title_ar="Adam")

    # ------------------------------------------------------------------ Adam by hand
    h3("Adam يدويًا: ثلاث خطوات لمعلمة واحدة", "Adam by hand: three steps for one parameter")
    ad = adam_steps()
    acaps = []
    for s in ad:
        acaps.append(f"**الخطوة {s['t']}** (التدرج g = {s['g']:+.1f}): m = {s['m']:.4f}، s = {s['s']:.5f}. بلا تصحيح، m صغيرة جدًا لأنها بدأت من الصفر؛ "
                     f"التصحيح يقسمها على (1 − 0.9^{s['t']}) فيعطي m̂ = {s['mh']:.3f}، ويعطي ŝ = {s['sh']:.3f}. الخطوة = η·m̂/√ŝ = **{s['step']:.4f}** ⇒ θ: {s['theta']:.4f} → {s['new']:.4f}.")
    acaps[0] += " لاحظ: الخطوة الأولى لـ Adam = η تمامًا (m̂/√ŝ = g/|g| = 1) مهما كان حجم التدرج."
    cur = animation_player("w05_adam", [Frame("", caption(c), action=f"t = {s['t']}", values=[("m̂", "", f"{s['mh']:.3f}"), ("ŝ", "", f"{s['sh']:.3f}"), ("θ", f"{s['theta']:.4f}", f"{s['new']:.4f}")])
                                        for c, s in zip(acaps, ad)], title_ar="Adam بأرقام (η = 0.1، β₁ = 0.9، β₂ = 0.999)", interval_ms=3200)
    s = ad[cur]
    worked_steps([
        ("الزخم (متوسط أسي للتدرج)", rf"m_{s['t']} = 0.9\,m_{{{s['t'] - 1}}} + 0.1\,({s['g']:+.1f}) = {s['m']:.4f}"),
        ("متوسط مربع التدرج", rf"s_{s['t']} = 0.999\,s_{{{s['t'] - 1}}} + 0.001\,({s['g']:+.1f})^2 = {s['s']:.5f}"),
        ("تصحيح الانحياز", rf"\hat m = \frac{{{s['m']:.4f}}}{{1-0.9^{s['t']}}} = {s['mh']:.3f},\qquad \hat s = \frac{{{s['s']:.5f}}}{{1-0.999^{s['t']}}} = {s['sh']:.3f}"),
        ("التحديث", rf"\theta \leftarrow {s['theta']:.4f} - 0.1\cdot\frac{{{s['mh']:.3f}}}{{\sqrt{{{s['sh']:.3f}}}}} = {s['new']:.4f}"),
    ], title_ar=f"الخطوة {s['t']} بالرموز")
    interpretation_note("في الخطوة 3 تبدّل التدرج إلى −1 لكن θ **استمر في النقصان**: الزخم m ما زال موجبًا (ذاكرة الخطوتين السابقتين). هذا هو «الاتجاه المتراكم» — ميزة في الوادي الضيق وخطر إن كان η كبيرًا (تجاوز القاع).")

    # ------------------------------------------------------------------ Keras race
    h2("سباق المحسّنات على شبكة Keras", "Optimizer race on a Keras network")
    c1, c2 = st.columns(2)
    with c1:
        lr = st.select_slider("η لكل المحسّنات", options=[0.001, 0.01, 0.1], value=0.01, key="w05_race_lr")
    with c2:
        epochs = st.slider("epochs", 5, 30, 15, 5, key="w05_race_ep")
    try:
        fig = go.Figure(); rows = []
        for name, color in (("sgd", "#2563EB"), ("rmsprop", "#059669"), ("adam", "#DB2777")):
            r = keras_mlp_run((16,), "relu", name, float(lr), int(epochs), 32, 0)
            e = np.arange(1, len(r["history"]["loss"]) + 1)
            fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name=name, line=dict(color=color, width=2.5)))
            rows.append((name, f"{r['history']['loss'][-1]:.3f}", f"{r['history']['val_loss'][-1]:.3f}", f"{r['history']['val_accuracy'][-1]:.3f}", str(int(np.argmin(r["history"]["val_loss"])) + 1)))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="epoch", yaxis_title="val_loss", legend=dict(orientation="h", x=0, y=1.15))
        st.plotly_chart(fig, width="stretch", key="w05_race_fig")
        table(["المحسّن", "loss", "val_loss", "val_accuracy", "أفضل حقبة"], rows, ["code", "num", "num", "num", "num"])
    except Exception as exc:  # noqa: BLE001 - TensorFlow missing on this runtime
        warning_note(f"TensorFlow غير متاح هنا ({type(exc).__name__}). السباق ثنائي الأبعاد أعلاه محسوب بـ NumPy.")
    intuition("عند η = 0.001: SGD يزحف بينما Adam يصل — التكييف يكبّر الخطوات الصغيرة. عند η = 0.1: Adam/RMSprop قد يتذبذبان لأن η كبير لهما (خطوتهما الفعلية ≈ η لكل معلمة) بينما SGD يستفيد. **لا يوجد فائز مطلق؛ يوجد زوج (محسّن، η)**.")
    st.button("معمل سباق المحسّنات (سطح خسارة + تحريك)", icon=":material/science:", type="primary", on_click=goto, args=("labs.optimizer_race",), key="w05_lab_race")
    h2("قواعد الاختيار", "Choosing")
    compare_table(["المحسّن", "يعالج", "لا يعالج", "في Keras"],
                  [("SGD", "—", "التذبذب، البطء، المقاييس", "`SGD(learning_rate)`"), ("الزخم", "التذبذب والبطء في الاتجاهات المتسقة", "المقاييس المختلفة", "`SGD(lr, momentum=0.9)`"),
                   ("RMSprop", "المقاييس المختلفة (η لكل معلمة)", "التسارع", "`RMSprop(lr)`"), ("Adam", "الاثنان + تصحيح البداية", "η كبير جدًا ما زال يتذبذب", "`Adam(lr)` / `AdamW`")],
                  ["code", "rtl", "rtl", "code"])
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
        Q("بزخم β = 0.9 وتدرج ثابت، الخطوة الفعلية تقارب…", ["η·g", "10·η·g", "0.9·η·g"], 1, "1/(1 − 0.9) = 10."),
        Q("الخطوة الأولى لـ Adam تساوي تقريبًا…", ["η·g", "η (مستقلة عن حجم g)", "صفرًا"], 1, "m̂/√ŝ = g/|g|."),
        Q("لماذا تصحيح الانحياز؟", ["لتسريع الحقب الأخيرة", "لأن m وs تبدآن من الصفر فتكونان صغيرتين في الخطوات الأولى", "لتقليل الذاكرة"], 1, ""),
    ])
    takeaway("زخم = اتجاه متراكم (×10 عند β = 0.9)؛ RMSprop = η لكل معلمة؛ Adam = الاثنان + تصحيح البداية. الزوج (محسّن، η) هو الاختيار لا المحسّن وحده. Adam(1e-3) بداية؛ SGD+زخم+جدول عند الضبط.")
    lesson_footer(LESSON, ["سباق في وادٍ ضيق (تحريك) ومنحنيات الخسارة.", "ثلاث معادلات وما تعالجه.", "Adam يدويًا بثلاث خطوات (تحريك).", "سباق حقيقي على Keras.", "قواعد الاختيار."])
