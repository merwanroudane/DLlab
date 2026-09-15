"""Interactive Network Builder — choose layers/units/activation, train a tiny
NumPy MLP on a 2-D toy dataset, watch the decision boundary and loss."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, practical_note, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.network_builder",
    title_ar="معمل بناء الشبكة",
    title_en="Interactive Network Builder",
    category="training",
    description_ar="اختر عدد الطبقات المخفية ووحداتها والتنشيط، درّب شبكة NumPy صغيرة على بيانات ثنائية البُعد (حلقة، حلزون، XOR)، وشاهد حد القرار وعدد المعلمات ومنحنى الخسارة.",
    related_lessons=["foundations.architecture.depth_width_dense", "foundations.architecture.parameter_count", "foundations.neuron.activation_intro"],
)


def _data(kind: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    if kind == "حلقة":
        r = rng.uniform(0, 2.2, n); th = rng.uniform(0, 2 * np.pi, n)
        X = np.column_stack([r * np.cos(th), r * np.sin(th)]); y = (r > 1.3).astype(float)
    elif kind == "حلزون":
        t = np.sqrt(rng.uniform(0.05, 1, n)) * 3 * np.pi; cls = rng.integers(0, 2, n)
        X = np.column_stack([t * np.cos(t + cls * np.pi), t * np.sin(t + cls * np.pi)]) / 4.5 + rng.normal(0, 0.05, (n, 2)); y = cls.astype(float)
    elif kind == "XOR":
        X = rng.uniform(-2, 2, (n, 2)); y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    else:
        X = rng.normal(size=(n, 2)); y = (X[:, 0] + 0.5 * X[:, 1] > 0.2).astype(float)
    return X, y


ACT = {"relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
       "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
       "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))))}


@st.cache_data(max_entries=32, show_spinner=False)
def train(kind: str, n: int, seed: int, hidden: tuple, act: str, epochs: int, lr: float):
    X, y = _data(kind, n, seed)
    rng = np.random.default_rng(seed)
    sizes = [2, *hidden, 1]
    Ws = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
    bs = [np.zeros(s) for s in sizes[1:]]
    f, df = ACT[act]
    losses = []
    for _ in range(epochs):
        acts, zs, a = [X], [], X
        for i in range(len(Ws)):
            z = a @ Ws[i] + bs[i]; zs.append(z)
            a = f(z) if i < len(Ws) - 1 else 1 / (1 + np.exp(-z)); acts.append(a)
        p = a[:, 0]
        losses.append(float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))))
        d = (p - y)[:, None] / len(y)
        for i in reversed(range(len(Ws))):
            gW = acts[i].T @ d; gb = d.sum(0)
            if i > 0:
                d = (d @ Ws[i].T) * df(zs[i - 1])
            Ws[i] -= lr * gW; bs[i] -= lr * gb
    def predict(G):
        a = G
        for i in range(len(Ws)):
            z = a @ Ws[i] + bs[i]; a = f(z) if i < len(Ws) - 1 else 1 / (1 + np.exp(-z))
        return a[:, 0]
    g = np.linspace(-2.6, 2.6, 90); GX, GY = np.meshgrid(g, g)
    P = predict(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    acc = float(((predict(X) >= 0.5) == y).mean())
    nparams = int(sum(W.size + b.size for W, b in zip(Ws, bs)))
    return X, y, g, P, losses, acc, nparams


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3 = st.columns(3)
    with c1:
        kind = st.selectbox("البيانات", ["حلقة", "حلزون", "XOR", "خطي"], key="nb_kind")
        n = st.slider("عدد الملاحظات", 100, 600, 300, 50, key="nb_n")
    with c2:
        n_layers = st.slider("عدد الطبقات المخفية", 0, 4, 1, key="nb_layers")
        units = st.slider("وحدات كل طبقة", 1, 32, 8, key="nb_units")
    with c3:
        act = st.selectbox("التنشيط", ["relu", "tanh", "sigmoid"], key="nb_act")
        epochs = st.slider("الحقب", 100, 2000, 600, 100, key="nb_epochs")
        lr = st.select_slider("η", options=[0.01, 0.05, 0.1, 0.3, 0.5, 1.0], value=0.3, key="nb_lr")
    hidden = tuple([units] * n_layers)
    intuition("ابدأ بصفر طبقات مخفية (= انحدار لوجستي) على الحلقة وراقب فشلها، ثم أضف طبقة واحدة بـ 8 وحدات ReLU.")
    X, y, g, P, losses, acc, nparams = train(kind, n, 0, hidden, act, epochs, lr)
    arch = " → ".join(map(str, [2, *hidden, 1]))
    table(["البنية", "المعلمات", "الدقة (تدريب)", "الخسارة النهائية"], [(arch, f"{nparams:,}", f"{acc:.3f}", f"{losses[-1]:.4f}")], ["ltr", "num", "num", "num"])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure(go.Contour(x=g, y=g, z=P, colorscale=[[0, "#E6F1FB"], [0.5, "#FFFDF9"], [1, "#FBE6E2"]], contours=dict(start=0, end=1, size=0.1), showscale=False, opacity=0.85))
        fig.add_trace(go.Scatter(x=X[y == 1, 0], y=X[y == 1, 1], mode="markers", marker=dict(color="#C8473A", size=6), name="y=1"))
        fig.add_trace(go.Scatter(x=X[y == 0, 0], y=X[y == 0, 1], mode="markers", marker=dict(color="#2F6FB5", size=6), name="y=0"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(scaleanchor="x"), paper_bgcolor="#FFFDF9", plot_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch", key="nb_boundary")
    with c2:
        fig2 = go.Figure(go.Scatter(y=losses, line=dict(color="#C8473A")))
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="BCE", paper_bgcolor="#FFFDF9", plot_bgcolor="#FFFDF9")
        st.plotly_chart(fig2, width="stretch", key="nb_loss")
    if n_layers == 0 and kind in ("حلقة", "حلزون", "XOR"):
        warning_note("بلا طبقة مخفية الحد خط مستقيم: لا يستطيع فصل الحلقة أو XOR. الدقة قرب 50–60% ليست خطأ برمجيًا بل قيدًا بنيويًا.")
    if act == "sigmoid" and n_layers >= 3:
        warning_note("Sigmoid في 3+ طبقات: التدرج يتلاشى (مشتقة ≤ 0.25 لكل طبقة) فيبطئ التدريب أو يتوقف. قارن مع ReLU بنفس الإعدادات.")
    if losses[-1] > losses[0]:
        warning_note("الخسارة ارتفعت: معدل التعلم كبير لهذه البنية. قلّله.")
    practical_note("الحلزون يحتاج عمقًا (2–3 طبقات) أو عرضًا كبيرًا ووقتًا أطول. لاحظ عدد المعلمات مقابل 300 نقطة فقط: الحدود الناعمة تعمّم، والمتعرجة الملتفة حول نقاط فردية علامة حفظ.")
