import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table
from labs.fw import keras_mlp_run

LESSON = Lesson(
    id="foundations.frameworks.keras.output_explorer",
    title_ar="مستكشف مخرجات Keras: كيف أقرأ هذا المخرج؟",
    title_en="Keras Output Explorer: How to Read This Output?",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=10,
    prerequisites=["foundations.frameworks.keras.summary_deconstruction", "foundations.frameworks.keras.evaluate_predict"],
    objectives_ar=["رؤية كل مخرج حقيقي من Keras (summary، سجل fit، History.history، evaluate، predict) على نفس النموذج.", "قراءة كل سطر وكل عمود وكل رقم في كل مخرج.", "التمييز بين ما يشير إلى مشكلة وما هو طبيعي."],
    terms=["epoch", "batch", "loss", "iteration"],
    difficulty="beginner",
    summary_ar="خمسة مخرجات لنموذج واحد، ولكل واحد قسم «كيف أقرأ هذا المخرج؟» يفسّر كل عنصر.",
)


def _how(title: str, body_md: str) -> None:
    with st.expander(f"كيف أقرأ هذا المخرج؟ — {title}", icon=":material/visibility:"):
        st.markdown(body_md)


def render() -> None:
    lesson_header(LESSON)
    why("الباحث الجديد يرى `15/15 - 0s - 7ms/step - accuracy: 0.81 - loss: 0.56 - val_accuracy: 0.79 - val_loss: 0.51` ويقرأ منه رقمًا واحدًا. هذا السطر يحوي 8 معلومات مستقلة. هذه الصفحة تشغّل نموذجًا حقيقيًا وتفكك **كل** مخرج.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hidden = st.select_slider("Dense hidden", options=[4, 8, 16, 32], value=8, key="oe_h")
    with c2:
        epochs = st.slider("epochs", 2, 8, 4, key="oe_e")
    with c3:
        bs = st.select_slider("batch_size", options=[16, 32, 64, 150], value=32, key="oe_bs")
    with c4:
        opt = st.selectbox("optimizer", ["adam", "sgd", "rmsprop"], key="oe_opt")
    r = keras_mlp_run((int(hidden),), "relu", opt, 0.01 if opt != "sgd" else 0.3, int(epochs), int(bs), 0)
    n_tr = 450

    h2("1) model.summary()", "1) model.summary()")
    st.code(r["summary"].strip(), language="text")
    _how("summary", f"""
- **`Model: "sequential"`** — اسم النموذج (تلقائي).
- **صف لكل طبقة**: `dense (Dense)` = اسم الطبقة (نوعها). الأسماء تلقائية بترتيب الإنشاء: `dense`, `dense_1`… لو أعدت تشغيل الخلية دون إعادة تشغيل بايثون تصبح `dense_2`, `dense_3` — ليس خطأ.
- **Output Shape** `(None, {hidden})`: `None` = بُعد الدفعة (أي عدد)، `{hidden}` = وحدات الطبقة.
- **Param #**: `2×{hidden}+{hidden} = {3 * hidden}` للأولى، `{hidden}×1+1 = {hidden + 1}` للأخيرة.
- **Total params**: مجموع العمود؛ بين قوسين حجم الذاكرة (float32 = 4 بايت لكل معلمة).
- **Trainable / Non-trainable**: هنا لا BN ولا تجميد فالكل قابل للتدريب.
- **Optimizer params** (يظهر بعد التدريب): حالة المحسّن — Adam يحفظ متوسطين لكل معلمة (2×{r['n_params']} = {2 * r['n_params']}) + عدّاد. ليست معلمات النموذج.
""")

    h2("2) سجل fit (verbose=2)", "2) fit() training log")
    st.code(r["log"].strip(), language="text")
    _how("سجل fit", f"""
سطران لكل حقبة:
- **`Epoch 1/{epochs}`** — عدّاد الحقب: الحالية / الكلية.
- **`{r['steps_per_epoch']}/{r['steps_per_epoch']}`** — الدفعات المنجزة / دفعات الحقبة = ⌈{n_tr}/{bs}⌉ = {r['steps_per_epoch']}. كل واحدة تحديث للمعلمات.
- **`0s - 7ms/step`** — زمن الحقبة وزمن الدفعة الواحدة. الأولى أبطأ (بناء الرسم/الترجمة).
- **`accuracy: …`** — المقياس **التراكمي** على دفعات التدريب في هذه الحقبة (يتغير أثناءها).
- **`loss: …`** — خسارة التدريب التراكمية (الدالة المُشتقة).
- **`val_accuracy`, `val_loss`** — على مجموعة التحقق **كاملة** بعد انتهاء الحقبة، بوضع الاستدلال.
- الترتيب أبجدي في Keras 3 (accuracy قبل loss)، لا بحسب الأهمية.
- **ما يثير القلق**: loss لا ينخفض (η أو الخسارة الخاطئة)، val_loss يرتفع بينما loss ينخفض (فرط تخصيص)، `nan` (انفجار)، accuracy ثابتة عند نسبة الفئة الغالبة (النموذج يتنبأ بفئة واحدة).
""")

    h2("3) History.history", "3) History.history")
    hist = {k: [round(v, 4) for v in vs] for k, vs in r["history"].items()}
    st.code(json.dumps(hist, indent=2), language="json")
    fig = go.Figure()
    e = np.arange(1, len(hist["loss"]) + 1)
    fig.add_trace(go.Scatter(x=e, y=hist["loss"], name="loss", line=dict(color="#2F6FB5", width=3)))
    fig.add_trace(go.Scatter(x=e, y=hist["val_loss"], name="val_loss", line=dict(color="#C8473A", width=3)))
    fig.add_trace(go.Scatter(x=e, y=hist["accuracy"], name="accuracy", line=dict(color="#2F6FB5", width=2, dash="dot"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=e, y=hist["val_accuracy"], name="val_accuracy", line=dict(color="#C8473A", width=2, dash="dot"), yaxis="y2"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis=dict(title="loss"), yaxis2=dict(title="accuracy", overlaying="y", side="right", range=[0, 1]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="oe_fig")
    _how("History", f"""
- **قاموس** مفاتيحه أسماء الخسارة والمقاييس (+ بادئة `val_` للتحقق). كل قيمة **قائمة بطول epochs = {epochs}**.
- العنصر رقم i = قيمة نهاية الحقبة i (نفس أرقام السجل).
- `history.epoch` = `[0, …, {epochs - 1}]` والفهرسة من صفر؛ السجل يعدّ من 1.
- الرسم أعلاه هو منحنيات التعلم (وحدة 19): loss/val_loss على المحور الأيسر، accuracy على الأيمن.
- لا يوجد في History قيم **لكل دفعة** — لذلك تحتاج Callback أو TensorBoard.
""")

    h2("4) evaluate()", "4) evaluate()")
    st.code(f"model.evaluate(X_val, y_val, verbose=0)\n-> {[round(v, 4) for v in r['evaluate']]}\nmodel.metrics_names -> ['loss', 'accuracy']", language="text")
    _how("evaluate", f"""
- قائمة بطول 1 + عدد المقاييس، **بترتيب compile**: `[loss, accuracy]`.
- محسوبة على **كل** البيانات المعطاة دفعةً دفعة ثم مجمّعة — لذلك تساوي `val_loss` لآخر حقبة (= {hist['val_loss'][-1]}) عند تقييم نفس مجموعة التحقق (فروق تقريب فقط).
- بلا مقاييس تعيد عددًا واحدًا لا قائمة.
- `verbose=0` يمنع سطر تقدم؛ `return_dict=True` يعيد قاموسًا بالأسماء بدل القائمة.
""")

    h2("5) predict()", "5) predict()")
    st.code(f"P = model.predict(X_val[:5], verbose=0)\nP.shape -> {tuple(r['predict_shape'])}\nP ->\n{np.array(r['predict'])}\n(P >= 0.5).astype(int).ravel() -> {(np.array(r['predict']) >= 0.5).astype(int).ravel().tolist()}", language="text")
    _how("predict", f"""
- الشكل `{tuple(r['predict_shape'])}` = (عدد الملاحظات، وحدات الطبقة الأخيرة) — هنا `Dense(1, sigmoid)` فعمود واحد باحتمال الفئة 1.
- القيم في (0, 1) لأن التنشيط sigmoid؛ **ليست فئات**. القرار بعتبة (0.5 افتراضيًا؛ وحدة 18 لتغييرها).
- لـ softmax بـ K فئات: (n, K) وكل صف يجمع إلى 1 وargmax يعطي الفئة.
- للانحدار: القيمة بوحدات الهدف (المُحجَّم إن حجّمت y).
- dtype float32: المخرج الافتراضي لـ Keras.
""")
    table(["المخرج", "النوع في بايثون", "أهم ما يُقرأ", "الخطأ الشائع في قراءته"],
          [("summary()", "نص (يُطبع)", "Param # لكل طبقة، None في الشكل", "اعتبار None خطأ"), ("سجل fit", "نص (يُطبع)", "x/y دفعات، loss وval_loss لكل حقبة", "قراءة loss كقيمة آخر دفعة"), ("History", "كائن، `.history` قاموس قوائم", "قائمة لكل مقياس بطول epochs", "الفهرسة من صفر مقابل عدّ السجل من 1"),
           ("evaluate", "قائمة أعداد", "الترتيب [loss, metrics…]", "الخلط بين الخسارة والمقياس"), ("predict", "ndarray (n, units)", "احتمالات لا فئات", "نسيان العتبة/argmax أو بُعد الدفعة")],
          ["code", "rtl", "rtl", "rtl"])
    intuition("غيّر `batch_size` إلى 150 وراقب السجل: `3/3` بدل `15/15` — عدد التحديثات لكل حقبة انخفض 5 مرات، والتعلم في نفس عدد الحقب أبطأ. غيّر المحسّن إلى sgd وراقب loss.")
    common_mistake("«val_accuracy أعلى من accuracy = خطأ». في الحقب الأولى طبيعي: accuracy تراكمية أثناء الحقبة (تبدأ بأوزان سيئة)، أما val_accuracy فبعد الحقبة بأوزان أفضل. ومع Dropout يبقى الفارق (وحدة 20).")
    quiz("keras.outputs", [
        Q("`15/15` في سطر fit تعني…", ["15 حقبة", "15 دفعة (تحديثًا) في الحقبة", "15 ملاحظة"], 1, "⌈n/batch_size⌉."),
        Q("`loss` في سطر الحقبة هو…", ["خسارة آخر دفعة", "المتوسط التراكمي على دفعات الحقبة", "خسارة التحقق"], 1, "تراكمي."),
        Q("`Optimizer params` في summary…", ["معلمات النموذج", "حالة المحسّن (متوسطات Adam)", "المعلمات المجمّدة"], 1, "ليست في النموذج."),
        Q("`history.history['val_loss'][0]` هو…", ["الحقبة 0 في السجل", "الحقبة 1 في السجل", "آخر حقبة"], 1, "الفهرسة من صفر."),
    ])
    takeaway("خمسة مخرجات، كل رقم فيها له معنى محدد: summary (البنية والمعلمات)، السجل (دفعات، خسارة ومقاييس تراكمية، تحقق)، History (قوائم بطول epochs)، evaluate (قائمة بترتيب compile)، predict (احتمالات بشكل (n, units)).")
    lesson_footer(LESSON, ["كل مخرج مع «كيف أقرأه؟».", "الطبيعي مقابل المقلق.", "جدول القراءة السريعة."])
