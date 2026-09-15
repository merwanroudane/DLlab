"""Training Loop Simulator — step through a real training run: every batch
shows shapes, loss, gradient norms and weights before/after (spec §22A)."""

import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, pipeline, table
from labs.tinynet import TinyNet, grad_norm, make_moons

LAB = Lab(
    id="labs.training_loop_simulator",
    title_ar="محاكي حلقة التدريب",
    title_en="Training Loop Simulator",
    category="training",
    description_ar="شغّل حلقة تدريب حقيقية على شبكة صغيرة خطوةً خطوة: لكل دفعة شكل X_batch وy_batch، التنبؤات، الخسارة، معيار التدرج، ووزن قبل/بعد التحديث؛ ثم التحقق في نهاية كل حقبة.",
    related_lessons=["foundations.training_loop.the_loop", "foundations.training_loop.training_loop_code"],
)

STAGES = ["Load batch", "Forward", "Loss", "Backward", "Update", "Validate"]


@st.cache_data(max_entries=16, show_spinner=False)
def _trace(n: int, batch_size: int, epochs: int, lr: float, seed: int):
    X, y = make_moons(n + 100, seed=0)
    Xtr, ytr, Xva, yva = X[:n], y[:n], X[n:], y[n:]
    net = TinyNet([2, 8, 1], "binary", seed=seed)
    rng = np.random.default_rng(seed)
    events = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        for bi, s in enumerate(range(0, n, batch_size), start=1):
            idx = order[s:s + batch_size]
            out, cache = net.forward(Xtr[idx])
            loss = net.loss(out, ytr[idx])
            gW, gb = net.backward(cache, ytr[idx])
            w_before = float(net.W[0][0, 0])
            net.W[0] -= lr * gW[0]; net.b[0] -= lr * gb[0]; net.W[1] -= lr * gW[1]; net.b[1] -= lr * gb[1]
            events.append(dict(epoch=ep, batch=bi, idx=idx.tolist(), xshape=Xtr[idx].shape, yshape=ytr[idx].shape, pred=out[:4, 0].round(3).tolist(),
                               ytrue=ytr[idx][:4].astype(int).tolist(), loss=loss, gnorm=grad_norm(gW, gb), gW0=float(gW[0][0, 0]), w_before=w_before, w_after=float(net.W[0][0, 0])))
        vl = net.loss(net.predict(Xva), yva); va = net.metric(Xva, yva)
        events.append(dict(epoch=ep, batch=None, val_loss=vl, val_acc=va))
    return events


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = st.select_slider("n تدريب", options=[40, 80, 120], value=80, key="tls_n")
    with c2:
        bs = st.select_slider("batch_size", options=[8, 16, 20, 40], value=20, key="tls_bs")
    with c3:
        epochs = st.slider("epochs", 1, 4, 2, key="tls_epochs")
    with c4:
        lr = st.select_slider("η", options=[0.05, 0.1, 0.3, 0.5], value=0.3, key="tls_lr")
    events = _trace(int(n), int(bs), int(epochs), float(lr), 0)
    spe = int(np.ceil(n / bs))
    intuition("كل خطوة في التحريك مرحلة واحدة من الحلقة على دفعة حقيقية. راقب `w[0,0]` قبل/بعد: التغير = −η × تدرجه بالضبط.")
    frames = []
    for e in events:
        if e["batch"] is None:
            frames.append(Frame("", caption(f"**نهاية الحقبة {e['epoch']}** — التحقق بلا تدرج: `val_loss = {e['val_loss']:.4f}`، `val_acc = {e['val_acc']:.3f}`."), action="Validate", values=[("val_loss", "", f"{e['val_loss']:.4f}"), ("val_acc", "", f"{e['val_acc']:.3f}")], highlight=5))
            continue
        where = f"الحقبة {e['epoch']} · الدفعة {e['batch']}/{spe}"
        frames.append(Frame("", caption(f"**{where} — تحميل الدفعة**: `X_batch.shape = {e['xshape']}`, `y_batch.shape = {e['yshape']}`، الفهارس `{e['idx'][:6]}…`"), action="Load batch", highlight=0))
        frames.append(Frame("", caption(f"{where} — **أمامي**: أول 4 تنبؤات `{e['pred']}` مقابل الحقيقة `{e['ytrue']}`."), action="Forward", highlight=1))
        frames.append(Frame("", caption(f"{where} — **الخسارة** (BCE على الدفعة) = **{e['loss']:.4f}**."), action="Loss", values=[("batch loss", "", f"{e['loss']:.4f}")], highlight=2))
        frames.append(Frame("", caption(f"{where} — **خلفي**: معيار التدرج الكلي `{e['gnorm']:.4f}`؛ تدرج `W1[0,0] = {e['gW0']:+.4f}`."), action="Backward", values=[("‖∇L‖", "", f"{e['gnorm']:.4f}")], highlight=3))
        frames.append(Frame("", caption(f"{where} — **تحديث**: `W1[0,0]`: {e['w_before']:.4f} → {e['w_after']:.4f} (= {e['w_before']:.4f} − {lr}×{e['gW0']:+.4f})."), action="Update", equation=f"w ← w − η·g = {e['w_before']:.4f} − {lr}×({e['gW0']:+.4f}) = {e['w_after']:.4f}", values=[("W1[0,0]", f"{e['w_before']:.4f}", f"{e['w_after']:.4f}")], highlight=4))
    animation_player("tls_anim", frames, title_ar="حلقة التدريب مرحلةً مرحلة", stages=STAGES, interval_ms=1100)
    st.markdown("### ملخص الحقب")
    rows = []
    for ep in range(1, epochs + 1):
        bl = [e["loss"] for e in events if e["batch"] is not None and e["epoch"] == ep]; v = [e for e in events if e["batch"] is None and e["epoch"] == ep][0]
        rows.append((str(ep), str(len(bl)), f"{np.mean(bl):.4f}", f"{v['val_loss']:.4f}", f"{v['val_acc']:.3f}"))
    table(["الحقبة", "الدفعات", "متوسط خسارة الدفعات (loss)", "val_loss", "val_acc"], rows, ["num", "num", "num", "num", "num"])
    practical_note("عدد الإطارات = 5 × الدفعات + الحقب. هذه هي «التكرارات» التي تختفي داخل model.fit(): لكل واحدة تمرير وخسارة وخلفي وتحديث.")
