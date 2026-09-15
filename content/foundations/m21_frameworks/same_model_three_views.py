import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table
from labs.fw import keras_mlp_run, torch_mlp_run

LESSON = Lesson(
    id="foundations.frameworks.same_model_three_views",
    title_ar="نفس النموذج بثلاث رؤى: الرياضيات، Keras، PyTorch",
    title_en="Same Model, Three Views: Mathematics, Keras, PyTorch",
    module="foundations.frameworks",
    order=29,
    prerequisites=["foundations.frameworks.concept_mapping", "foundations.forward.layer_by_layer", "foundations.backprop.backpropagation"],
    objectives_ar=["رؤية MLP واحد (2 → h → 1) على نفس البيانات كمعادلات، ككود Keras، وككود PyTorch.", "ربط كل سطر كود بمكوّنه الرياضي (المعادلة، الخسارة، التدرج، التحديث).", "تشغيل الاثنين فعليًا بنفس المعلمات الفائقة ومقارنة النتائج."],
    terms=["model", "loss", "gradient", "optimizer"],
    difficulty="intermediate",
    summary_ar="المعادلات: z1 = xW1+b1, a1 = relu(z1), z2 = a1W2+b2, p = σ(z2), L = BCE, θ ← θ − η∇L. Keras تصفها؛ PyTorch تكتبها. النتائج متقاربة لأن الرياضيات واحدة.",
)

MATH = [
    ("البيانات", r"X \in \mathbb{R}^{n\times 2},\; y \in \{0,1\}^n"),
    ("الطبقة 1", r"z^{(1)} = X W^{(1)} + b^{(1)},\quad W^{(1)} \in \mathbb{R}^{2\times h}"),
    ("التنشيط", r"a^{(1)} = \max(0, z^{(1)})"),
    ("الطبقة 2", r"z^{(2)} = a^{(1)} W^{(2)} + b^{(2)},\quad W^{(2)} \in \mathbb{R}^{h\times 1}"),
    ("الاحتمال", r"\hat p = \sigma(z^{(2)}) = \frac{1}{1+e^{-z^{(2)}}}"),
    ("الخسارة", r"L = -\frac{1}{n}\sum_i \big[y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\big]"),
    ("التدرج", r"\nabla_\theta L \quad\text{(قاعدة السلسلة، وحدة 14)}"),
    ("التحديث", r"\theta \leftarrow \theta - \eta \, \text{Adam}(\nabla_\theta L)"),
    ("التكرار", r"\text{for epoch: for batch: (forward, L, } \nabla, \text{update)}"),
    ("التقييم", r"\text{val\_loss} = L(X_{val}, y_{val}),\quad \text{acc} = \tfrac{1}{n}\sum \mathbb{1}[\hat p \ge 0.5 = y]"),
]

KERAS = '''import keras
from keras import layers
keras.utils.set_random_seed(0)
model = keras.Sequential([
    layers.Input(shape=(2,)),                       # X ∈ ℝ^{n×2}
    layers.Dense(H, activation="relu"),             # z1 = XW1 + b1 ; a1 = relu(z1)
    layers.Dense(1, activation="sigmoid"),          # z2 = a1W2 + b2 ; p = σ(z2)
])
model.compile(optimizer=keras.optimizers.Adam(LR),  # θ ← θ − η·Adam(∇L)
              loss="binary_crossentropy",           # L = BCE(y, p)
              metrics=["accuracy"])                 # acc
history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                    epochs=EPOCHS, batch_size=BS)   # for epoch: for batch: forward, L, ∇ (autodiff), update ; val'''

TORCH = '''import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, H), nn.ReLU(),                     # z1 = xW1ᵀ + b1 ; a1 = relu(z1)
    nn.Linear(H, 1),                                # z2 = a1W2ᵀ + b2   (logits; σ inside the loss)
)
loss_fn = nn.BCEWithLogitsLoss()                    # L = BCE(y, σ(z2))
opt = torch.optim.Adam(model.parameters(), lr=LR)   # θ ← θ − η·Adam(∇L)
for epoch in range(EPOCHS):                         # for epoch
    model.train()
    for xb, yb in loader:                           #   for batch
        opt.zero_grad()
        out = model(xb)                             #     forward: z1, a1, z2
        loss = loss_fn(out, yb)                     #     L
        loss.backward()                             #     ∇θ L  (autograd)
        opt.step()                                  #     update
    model.eval()
    with torch.no_grad():                           #   val_loss, acc
        val_loss = loss_fn(model(X_va), y_va); acc = ((model(X_va) >= 0).float() == y_va).float().mean()'''


def render() -> None:
    lesson_header(LESSON)
    why("الاختبار الحقيقي لفهم الأطر: هل تستطيع الإشارة إلى **السطر** الذي يقابل كل معادلة؟ هنا نموذج واحد صغير على بيانات moons، بثلاث رؤى، وجدول يربطها سطرًا بسطر — ثم تشغيل حقيقي للاثنين.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        H = st.select_slider("H (وحدات مخفية)", options=[4, 8, 16], value=8, key="tv_h")
    with c2:
        LR = st.select_slider("η (Adam)", options=[0.003, 0.01, 0.03], value=0.01, key="tv_lr")
    with c3:
        EPOCHS = st.slider("epochs", 5, 30, 15, 5, key="tv_ep")
    with c4:
        BS = st.select_slider("batch_size", options=[16, 32, 64], value=32, key="tv_bs")
    tab_m, tab_k, tab_p = st.tabs(["∑ الرياضيات", "Keras / TensorFlow", "PyTorch"])
    with tab_m:
        for name, eq in MATH:
            cc1, cc2 = st.columns([1, 4])
            with cc1:
                st.markdown(f"**{name}**")
            with cc2:
                st.latex(eq)
    with tab_k:
        st.code(KERAS.replace("H,", f"{H},").replace("(LR)", f"({LR})").replace("EPOCHS", str(EPOCHS)).replace("batch_size=BS", f"batch_size={BS}"), language="python")
    with tab_p:
        st.code(TORCH.replace("(2, H)", f"(2, {H})").replace("(H, 1)", f"({H}, 1)").replace("lr=LR", f"lr={LR}").replace("range(EPOCHS)", f"range({EPOCHS})"), language="python")
    h2("سطر ↔ معادلة", "Line ↔ equation")
    table(["المكوّن الرياضي", "Keras", "PyTorch"],
          [("X ∈ ℝ^{n×2}", "`layers.Input(shape=(2,))`", "`nn.Linear(2, H)` (in=2)"), ("z¹ = XW¹ + b¹ ; a¹ = relu", "`Dense(H, activation='relu')`", "`nn.Linear(2, H), nn.ReLU()`"), ("z² = a¹W² + b² ; p = σ(z²)", "`Dense(1, activation='sigmoid')`", "`nn.Linear(H, 1)` + σ داخل `BCEWithLogitsLoss`"),
           ("L = BCE", "`loss='binary_crossentropy'`", "`nn.BCEWithLogitsLoss()`"), ("∇θL", "داخل fit (GradientTape)", "`loss.backward()`"), ("θ ← θ − η·Adam", "`Adam(LR)` داخل fit", "`opt.step()`"), ("for epoch / for batch", "`epochs=`, `batch_size=`", "`for epoch`, `for xb, yb in loader`"),
           ("التقييم", "`validation_data=` → `val_loss`, `val_accuracy`", "`model.eval()` + `no_grad` + حساب يدوي"), ("الاستنساخ", "`set_random_seed(0)`", "`manual_seed(0)`")],
          ["rtl", "code", "code"])
    intuition("لاحظ أين اختفى σ في PyTorch: ليس في النموذج بل داخل الخسارة — لكن المعادلة واحدة. ولاحظ أن `fit` = الأسطر 9–19 في PyTorch.")
    h2("تشغيل فعلي للاثنين", "Actually running both")
    kr = keras_mlp_run((int(H),), "relu", "adam", float(LR), int(EPOCHS), int(BS), 0)
    pr = torch_mlp_run((int(H),), "relu", "adam", float(LR), int(EPOCHS), int(BS), 0)
    table(["", "Keras", "PyTorch"],
          [("المعلمات", str(kr["n_params"]), str(pr["n_params"])), ("تحديثات لكل حقبة", str(kr["steps_per_epoch"]), str(pr["steps_per_epoch"])), ("loss النهائية", f"{kr['history']['loss'][-1]:.4f}", f"{pr['history']['loss'][-1]:.4f}"),
           ("val_loss النهائية", f"{kr['history']['val_loss'][-1]:.4f}", f"{pr['history']['val_loss'][-1]:.4f}"), ("val_accuracy النهائية", f"{kr['history']['val_accuracy'][-1]:.3f}", f"{pr['history']['val_accuracy'][-1]:.3f}"), ("أول 3 تنبؤات (p)", str([round(v[0], 3) for v in kr["predict"][:3]]), str([round(v[0], 3) for v in pr["predict"][:3]]))],
          ["rtl", "num", "num"])
    fig = go.Figure()
    e = np.arange(1, len(kr["history"]["val_loss"]) + 1)
    fig.add_trace(go.Scatter(x=e, y=kr["history"]["val_loss"], name="Keras val_loss", line=dict(color="#C8473A", width=3)))
    fig.add_trace(go.Scatter(x=e, y=pr["history"]["val_loss"], name="PyTorch val_loss", line=dict(color="#7C5CBF", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=e, y=kr["history"]["loss"], name="Keras loss", line=dict(color="#C8473A", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=e, y=pr["history"]["loss"], name="PyTorch loss", line=dict(color="#7C5CBF", width=1.5, dash="dot")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="tv_fig")
    equation(r"\text{Keras} \approx \text{PyTorch} \;\Leftarrow\; \text{same } (X, y, \text{architecture}, L, \text{Adam}, \eta, B, E)",
             [(r"\approx", "متقاربان لا متطابقان: تهيئة الأوزان مختلفة (Glorot مقابل Kaiming-uniform)، ترتيب الخلط مختلف، وتفاصيل float32."), ("=", "لو نسخت الأوزان الابتدائية وترتيب الدفعات لتطابقت الأرقام حتى 1e-6.")],
             meaning_ar="الإطار لا يغيّر النموذج ولا الخوارزمية؛ يغيّر الافتراضات الصغيرة (التهيئة، الخلط) وطريقة الكتابة.",
             example_ar=f"هنا: val_acc {kr['history']['val_accuracy'][-1]:.3f} مقابل {pr['history']['val_accuracy'][-1]:.3f}.", dl_link_ar="عند مقارنة نتائج ورقتين بإطارين مختلفين، الفروق الصغيرة طبيعية؛ الكبيرة تعني اختلافًا في البيانات أو المعلمات الفائقة.", title_ar="لماذا النتائج متقاربة")
    common_mistake("«PyTorch أعطى دقة أعلى إذًا أفضل». مع بذور مختلفة قد ينقلب الترتيب. الفرق الحقيقي بين الإطارين ليس في الدقة بل في **طريقة العمل** (الصفحة التالية).")
    quiz("fw.three", [
        Q("σ في نسخة PyTorch موجود في…", ["nn.Sequential", "داخل BCEWithLogitsLoss", "غير موجود"], 1, "logits."),
        Q("`loss.backward()` يقابل رياضيًا…", ["L", "∇θL", "θ ← θ − η∇L"], 1, "التدرج."),
        Q("لماذا تختلف نتائج الإطارين قليلًا؟", ["رياضيات مختلفة", "تهيئة وخلط مختلفان", "أحدهما خاطئ"], 1, "افتراضات صغيرة."),
        Q("`model.fit` يقابل في PyTorch…", ["nn.Sequential", "حلقتي الحقب والدفعات بخطواتهما", "state_dict"], 1, "الحلقة."),
    ])
    takeaway("نموذج واحد = معادلات واحدة. Keras تصفها في 4 استدعاءات؛ PyTorch تكتبها في 12 سطرًا. الجدول يربط كل سطر بمعادلته، والتشغيل يثبت التقارب.")
    lesson_footer(LESSON, ["ثلاث رؤى في ثلاثة تبويبات.", "جدول سطر ↔ معادلة.", "تشغيل حقيقي ومقارنة."])
