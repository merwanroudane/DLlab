"""Week 03 animation frames: tensors, the Keras workflow, parameter counting,
fit() batch by batch, GradientTape's graph, tf.data's shuffle buffer and the
PyTorch loop. Numbers come from the loan data, exact arithmetic, or real
framework traces (labs.fw), with graceful fallbacks when a framework is missing."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from components import svgkit as K
from labs.datasets import loan_default

NUM4 = ["income", "age", "num_late_payments", "debt_ratio"]


def _num(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


# ------------------------------------------------------------------ A) tensors
def tensor_svg(stage: int) -> str:
    df = loan_default(n=400, seed=7)
    rows = df[NUM4].head(6).to_numpy(float)
    import pandas as pd
    onehot = pd.get_dummies(df[["city", "employment"]]).head(6).to_numpy(int)     # the real 7 one-hot columns
    s = K.svg_open(680, 300)
    cols = [K.BLUE, K.VIOLET, K.AMBER, K.PINK]
    if stage == 0:
        s += K.box(290, 110, 100, 56, f"{rows[0, 3]:.3f}", K.PINK, filled=True, size=16)
        s += K.text(340, 200, "scalar: rank 0, shape ()", size=14, bold=True, color=K.PINK)
        s += K.text(340, 222, "one number: debt_ratio of customer 1001", size=11, color=K.MUTED)
    elif stage == 1:
        for j in range(4):
            s += K.box(160 + j * 92, 110, 86, 50, _num(rows[0, j]), cols[j], size=13, sub=NUM4[j][:11])
        s += K.text(340, 200, "vector: rank 1, shape (4,)", size=14, bold=True, color=K.VIOLET)
        s += K.text(340, 222, "one observation = one customer = 4 features", size=11, color=K.MUTED)
    else:
        n_feat = 4 if stage < 3 else 11
        cw = 56 if n_feat == 4 else 44
        x0 = 340 - n_feat * cw / 2
        for i in range(6):
            for j in range(n_feat):
                if j < 4:
                    v, c = _num(rows[i, j]), cols[j]
                else:
                    v, c = str(onehot[i, j - 4]), K.EMERALD
                s += (f'<rect x="{x0 + j * cw:.1f}" y="{60 + i * 30}" width="{cw - 3}" height="26" rx="5" fill="{K.tint(c, 0.82)}" stroke="{c}"/>'
                      + K.text(x0 + j * cw + (cw - 3) / 2, 78 + i * 30, v, size=10 if n_feat == 11 else 11, color=c, mono=True))
        s += K.text(x0 + n_feat * cw / 2, 255, "⋮  (32 rows per batch)", size=12, color=K.MUTED)
        s += K.arrow(x0 - 18, 60, x0 - 18, 235, color=K.VIOLET, width=2)
        s += K.text(x0 - 30, 150, "axis 0", size=11, color=K.VIOLET, bold=True, anchor="end")
        s += K.text(x0 - 30, 165, "batch", size=10, color=K.VIOLET, anchor="end")
        s += K.arrow(x0, 45, x0 + n_feat * cw - 6, 45, color=K.PINK, width=2)
        s += K.text(x0 + n_feat * cw / 2, 35, "axis 1: features", size=11, color=K.PINK, bold=True)
        label = {2: "matrix: rank 2, shape (32, 4) — a batch", 3: "after one-hot of city + employment: shape (32, 11)",
                 4: "dtype float32 · device CPU:0 — what Dense expects"}[min(stage, 4)]
        s += K.text(340, 285, label, size=14, bold=True, color=K.EMERALD if stage >= 3 else K.BLUE)
    return s + "</svg>"


# ------------------------------------------------------------------ B) Keras workflow
WORKFLOW = [
    ("Sequential", "model = keras.Sequential([...])", "a model object with RANDOM weights", K.VIOLET),
    ("summary()", "model.summary()", "shapes + parameter counts (no training)", K.BLUE),
    ("compile()", "model.compile(optimizer, loss, metrics)", "attaches HOW to learn + WHAT to minimize", K.CYAN),
    ("fit()", "history = model.fit(x, y, ...)", "the training loop → History object", K.EMERALD),
    ("evaluate()", "model.evaluate(x_test, y_test)", "loss + metrics on held-out data, once", K.AMBER),
    ("predict()", "p = model.predict(x_new)", "probabilities for new rows (weights frozen)", K.PINK),
]


def workflow_svg(k: int) -> str:
    s = K.svg_open(680, 250)
    for i, (name, _, _, col) in enumerate(WORKFLOW):
        x = 10 + i * 112
        s += K.box(x, 30, 100, 44, name, col, filled=i == k, size=12)
        if i < len(WORKFLOW) - 1:
            s += K.arrow(x + 101, 52, x + 111, 52, color=col)
    name, code, what, col = WORKFLOW[k]
    s += f'<rect x="60" y="100" width="560" height="44" rx="10" fill="#0F172A"/>'
    s += K.text(340, 127, code, size=13, color="#E2E8F0", mono=True)
    s += K.text(340, 175, what, size=14, bold=True, color=col)
    objs = [["model"], ["model", "summary table"], ["model", "optimizer", "loss", "metrics"], ["model*", "optimizer*", "History"],
            ["[loss, acc, auc]"], ["p ∈ (0,1) per row"]][k]
    for j, o in enumerate(objs):
        s += K.box(340 - len(objs) * 62 + j * 124, 196, 114, 34, o, K.PALETTE[(k + j) % 7], size=11)
    return s + "</svg>"


# ------------------------------------------------------------------ C) parameter counting
def params_svg(stage: int, n_in: int = 11, h: int = 16) -> str:
    s = K.svg_open(680, 300)
    # weight grid W1 (n_in × h)
    cell = 11
    gx, gy = 40, 50
    if stage >= 1:
        for i in range(n_in):
            for j in range(h):
                s += f'<rect x="{gx + j * cell}" y="{gy + i * cell}" width="{cell - 1}" height="{cell - 1}" fill="{K.tint(K.VIOLET, 0.5 + 0.4 * ((i + j) % 3) / 2)}"/>'
        for j in range(h):
            s += f'<rect x="{gx + j * cell}" y="{gy + n_in * cell + 6}" width="{cell - 1}" height="{cell - 1}" fill="{K.AMBER}"/>'
        s += K.text(gx + h * cell / 2, gy - 10, f"W1: {n_in}×{h} = {n_in * h}", size=12, bold=True, color=K.VIOLET)
        s += K.text(gx + h * cell / 2, gy + n_in * cell + 32, f"b1: {h}", size=12, bold=True, color=K.AMBER)
        s += K.text(gx + h * cell / 2, gy + n_in * cell + 52, f"Dense(16): {n_in * h} + {h} = {n_in * h + h}", size=13, bold=True, color=K.INK)
    if stage >= 2:
        g2 = 300
        for j in range(h):
            s += f'<rect x="{g2}" y="{gy + j * cell}" width="{cell - 1}" height="{cell - 1}" fill="{K.tint(K.PINK, 0.5)}"/>'
        s += f'<rect x="{g2}" y="{gy + h * cell + 6}" width="{cell - 1}" height="{cell - 1}" fill="{K.AMBER}"/>'
        s += K.text(g2 + 5, gy - 10, f"W2: {h}×1", size=12, bold=True, color=K.PINK)
        s += K.text(g2 + 60, gy + h * cell + 16, "b2: 1", size=12, bold=True, color=K.AMBER, anchor="start")
        s += K.text(g2 + 5, gy + h * cell + 52, f"Dense(1): {h} + 1 = {h + 1}", size=13, bold=True, color=K.INK)
    if stage == 0:
        s += K.text(340, 140, f"input: {n_in} features per observation — no parameters", size=15, bold=True, color=K.BLUE)
        s += K.text(340, 165, "(Input layer only fixes the shape: (None, 11))", size=12, color=K.MUTED)
    if stage >= 3:
        tot = n_in * h + h + h + 1
        rows = [("Layer (type)", "Output Shape", "Param #"), ("dense (Dense)", "(None, 16)", str(n_in * h + h)), ("dense_1 (Dense)", "(None, 1)", str(h + 1)),
                ("Total params", "", str(tot))]
        for r, (a, b, c) in enumerate(rows):
            yy = 70 + r * 36
            fill = "#0F172A" if r == 0 else ("#F3EEFF" if r < 3 else "#FFE8F3")
            tc = "#E2E8F0" if r == 0 else K.INK
            s += f'<rect x="400" y="{yy}" width="260" height="32" rx="6" fill="{fill}"/>'
            s += K.text(410, yy + 21, a, size=11, color=tc, anchor="start", mono=True) + K.text(560, yy + 21, b, size=11, color=tc, mono=True)
            s += K.text(648, yy + 21, c, size=11, color=K.PINK if r == 3 else tc, anchor="end", mono=True, bold=r == 3)
        s += K.text(530, 240, "model.summary() — the same numbers", size=12, bold=True, color=K.VIOLET)
    return s + "</svg>"


# ------------------------------------------------------------------ D) fit() traced batch by batch (real Keras)
@st.cache_data(ttl=3600, max_entries=2, show_spinner="يتتبع model.fit() بـ Keras…")
def fit_trace() -> dict | None:
    try:
        from labs.fw import keras_fit_trace

        return keras_fit_trace(hidden=8, epochs=5, batch_size=32, lr=0.5, seed=0, n=120)
    except Exception as exc:  # noqa: BLE001 - TensorFlow missing on this runtime
        return {"error": f"{type(exc).__name__}: {exc}"}


def fit_frames(tr: dict) -> list[tuple[str, str, str, list]]:
    """(svg, caption, action, values) per batch / epoch-end event."""
    ev = tr["events"]
    spe = tr["steps_per_epoch"]
    batch_losses: list[float] = []
    out = []
    epoch = 0
    for e in ev:
        if e["kind"] == "epoch_begin":
            epoch = e["epoch"]; continue
        if e["kind"] == "batch":
            batch_losses.append(e["loss"])
        s = K.svg_open(680, 250)
        n_ep = max(x["epoch"] for x in ev if x["kind"] == "epoch_end")
        for ep in range(1, n_ep + 1):
            x0 = 20 + (ep - 1) * 132
            s += K.text(x0 + 60, 28, f"epoch {ep}", size=11, bold=ep == epoch, color=K.VIOLET if ep == epoch else K.MUTED)
            for b in range(1, spe + 1):
                done = ep < epoch or (ep == epoch and (e["kind"] == "epoch_end" or b <= e.get("batch", 0)))
                cur = ep == epoch and e["kind"] == "batch" and b == e["batch"]
                col = K.PALETTE[(ep - 1) % 7]
                s += (f'<rect x="{x0 + (b - 1) * 30}" y="38" width="26" height="26" rx="6" fill="{col if cur else (K.tint(col, 0.6) if done else "#F1F5F9")}" '
                      f'stroke="{col}" stroke-width="{2.5 if cur else 1}"/>')
                s += K.text(x0 + (b - 1) * 30 + 13, 55, str(b), size=10, color="#fff" if cur else col, bold=cur)
        P = K.Panel(60, 95, 380, 120, (0, len([x for x in ev if x["kind"] == "batch"]), 0.3, 0.8))
        s += P.frame("batch loss (each dot = one update of all weights)", "update step")
        s += P.yticks([0.4, 0.5, 0.6, 0.7], "{:.1f}")
        if len(batch_losses) > 1:
            s += P.polyline(range(1, len(batch_losses) + 1), batch_losses, K.PINK, 2)
        if batch_losses:
            s += P.dot(len(batch_losses), batch_losses[-1], K.PINK, 4.5)
        if e["kind"] == "batch":
            bar = "━" * int(20 * e["batch"] / spe) + " " * (20 - int(20 * e["batch"] / spe))
            s += f'<rect x="470" y="95" width="195" height="120" rx="10" fill="#0F172A"/>'
            s += K.text(480, 120, f"Epoch {epoch}/{n_ep}", size=11, color="#A5B4FC", anchor="start", mono=True)
            s += K.text(480, 142, f"{e['batch']}/{spe} {bar[:10]}", size=11, color="#E2E8F0", anchor="start", mono=True)
            s += K.text(480, 164, f"loss: {e['loss']:.4f}", size=11, color="#F9A8D4", anchor="start", mono=True)
            s += K.text(480, 186, f"accuracy: {e['acc']:.4f}", size=11, color="#86EFAC", anchor="start", mono=True)
            s += K.text(480, 206, f"w[0,0]: {e['w_before']:+.3f}→{e['w_after']:+.3f}", size=10, color="#FCD34D", anchor="start", mono=True)
            cap = (f"**الحقبة {epoch}، الدفعة {e['batch']} من {spe}**: 32 ملاحظة ← تمرير أمامي ← خسارة الدفعة {e['loss']:.3f} ← تدرجات ← تحديث **كل** الأوزان مرة واحدة. "
                   f"مثال: الوزن `w[0,0]` تغيّر من {e['w_before']:+.4f} إلى {e['w_after']:+.4f}. «loss» في شريط التقدم متوسط تراكمي منذ بداية الحقبة.")
            vals = [("w[0,0]", f"{e['w_before']:+.4f}", f"{e['w_after']:+.4f}"), ("running loss", "", f"{e['loss']:.4f}")]
            act = f"epoch {epoch} · batch {e['batch']}"
        else:
            s += f'<rect x="470" y="95" width="195" height="120" rx="10" fill="#ECFDF5" stroke="{K.EMERALD}"/>'
            s += K.text(567, 120, f"end of epoch {epoch}", size=12, bold=True, color=K.EMERALD)
            s += K.text(567, 145, f"loss {e['loss']:.4f}", size=11, mono=True, color=K.INK)
            s += K.text(567, 165, f"val_loss {e['val_loss']:.4f}", size=11, mono=True, color=K.AMBER)
            s += K.text(567, 185, f"val_acc {e['val_acc']:.4f}", size=11, mono=True, color=K.AMBER)
            cap = (f"**نهاية الحقبة {epoch}**: مرّت كل ملاحظات التدريب ({tr['n']}) مرة واحدة = {spe} تحديثات. الآن فقط تُقيَّم بيانات التحقق ({tr['n_val']} ملاحظة) "
                   f"**بلا تحديث**: val_loss = {e['val_loss']:.4f}. هذه القيم تُضاف إلى `history.history`.")
            vals = [("loss", "", f"{e['loss']:.4f}"), ("val_loss", "", f"{e['val_loss']:.4f}"), ("val_accuracy", "", f"{e['val_acc']:.4f}")]
            act = f"epoch {epoch} end"
        out.append((s + "</svg>", cap, act, vals))
    return out


# ------------------------------------------------------------------ E) GradientTape graph with exact numbers
def tape_numbers(x=(0.8, 1.5), w=(0.4, -0.3), b=0.1, y=1.0, lr=0.1) -> dict:
    m1, m2 = x[0] * w[0], x[1] * w[1]
    z = m1 + m2 + b
    p = 1 / (1 + math.exp(-z))
    L = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    dL_dp = -(y / p) + (1 - y) / (1 - p)
    dp_dz = p * (1 - p)
    dL_dz = p - y
    g = {"w1": dL_dz * x[0], "w2": dL_dz * x[1], "b": dL_dz}
    return {"x": x, "w": w, "b": b, "y": y, "m1": m1, "m2": m2, "z": z, "p": p, "L": L, "dL_dp": dL_dp, "dp_dz": dp_dz, "dL_dz": dL_dz,
            "g": g, "lr": lr, "new": {"w1": w[0] - lr * g["w1"], "w2": w[1] - lr * g["w2"], "b": b - lr * g["b"]}}


def tape_svg(n: dict, stage: int) -> str:
    """stage 0 recording starts · 1 mul1 · 2 mul2 · 3 add → z · 4 σ → p · 5 loss · 6 dL/dp · 7 dL/dz · 8 dL/dw, dL/db · 9 update"""
    s = K.svg_open(700, 300)
    s += f'<rect x="4" y="4" width="692" height="292" rx="14" fill="{"#FFF7ED" if stage <= 5 else "#EFF6FF"}" stroke="{K.ORANGE if stage <= 5 else K.BLUE}" stroke-dasharray="6 4"/>'
    s += K.text(350, 24, "● REC  with tf.GradientTape() as tape:" if stage <= 5 else "◀ tape.gradient(loss, variables)", size=12, bold=True,
                color=K.RED if stage <= 5 else K.BLUE, mono=True)
    nodes = {"x1": (60, 80, f"x1={n['x'][0]}"), "w1": (60, 130, f"w1={n['w'][0]}"), "x2": (60, 190, f"x2={n['x'][1]}"), "w2": (60, 240, f"w2={n['w'][1]}"),
             "m1": (210, 105, f"×  {n['m1']:.2f}"), "m2": (210, 215, f"×  {n['m2']:.2f}"), "b": (210, 270, f"b={n['b']}"),
             "z": (350, 160, f"+  z={n['z']:.3f}"), "p": (470, 160, f"σ  p={n['p']:.4f}"), "L": (610, 160, f"L={n['L']:.4f}")}
    edges = [("x1", "m1", 1), ("w1", "m1", 1), ("x2", "m2", 2), ("w2", "m2", 2), ("m1", "z", 3), ("m2", "z", 3), ("b", "z", 3), ("z", "p", 4), ("p", "L", 5)]
    active = {0: [], 1: ["m1"], 2: ["m2"], 3: ["z"], 4: ["p"], 5: ["L"]}
    for a, bnode, st_ in edges:
        (x1, y1, _), (x2, y2, _) = nodes[a], nodes[bnode]
        on = stage >= st_
        s += f'<line x1="{x1 + 50}" y1="{y1}" x2="{x2 - 55}" y2="{y2}" stroke="{K.ORANGE if on else "#CBD5E1"}" stroke-width="{2 if on else 1}"/>'
    order = ["x1", "w1", "x2", "w2", "b", "m1", "m2", "z", "p", "L"]
    appear = {"x1": 0, "w1": 0, "x2": 0, "w2": 0, "b": 0, "m1": 1, "m2": 2, "z": 3, "p": 4, "L": 5}
    for k in order:
        x, y, lab = nodes[k]
        shown = stage >= appear[k]
        is_var = k in ("w1", "w2", "b")
        col = K.VIOLET if is_var else (K.BLUE if k.startswith("x") else K.ORANGE)
        if stage <= 5 and k in active.get(stage, []):
            col = K.RED
        s += (f'<rect x="{x - 52}" y="{y - 16}" width="104" height="32" rx="9" fill="{K.tint(col, 0.8) if shown else "#F8FAFC"}" '
              f'stroke="{col if shown else "#E2E8F0"}" stroke-width="{2.2 if (stage <= 5 and k in active.get(stage, [])) else 1.4}"/>')
        s += K.text(x, y + 4, lab if shown else "", size=11, mono=True, color=col)
    # backward annotations
    if stage >= 6:
        s += K.text(610, 205, "∂L/∂L = 1", size=11, mono=True, color=K.BLUE, bold=True)
        s += K.text(540, 125, f"∂L/∂p = {n['dL_dp']:.3f}", size=11, mono=True, color=K.BLUE, bold=stage == 6)
    if stage >= 7:
        s += K.text(410, 125, f"∂L/∂z = p−y = {n['dL_dz']:.4f}", size=11, mono=True, color=K.BLUE, bold=stage == 7)
    if stage >= 8:
        s += K.text(175, 150, f"∂L/∂w1 = {n['g']['w1']:.4f}", size=11, mono=True, color=K.PINK, bold=True, anchor="start")
        s += K.text(175, 258, f"∂L/∂w2 = {n['g']['w2']:.4f}", size=11, mono=True, color=K.PINK, bold=True, anchor="start")
        s += K.text(270, 290, f"∂L/∂b = {n['g']['b']:.4f}", size=11, mono=True, color=K.PINK, bold=True, anchor="start")
    if stage >= 9:
        nw = n["new"]
        s += f'<rect x="420" y="220" width="265" height="62" rx="10" fill="#ECFDF5" stroke="{K.EMERALD}"/>'
        s += K.text(552, 240, f"apply_gradients (η = {n['lr']})", size=11, bold=True, color=K.EMERALD)
        s += K.text(552, 258, f"w1 {n['w'][0]}→{nw['w1']:.4f}  w2 {n['w'][1]}→{nw['w2']:.4f}", size=10, mono=True, color=K.INK)
        s += K.text(552, 274, f"b {n['b']}→{nw['b']:.4f}", size=10, mono=True, color=K.INK)
    return s + "</svg>"


# ------------------------------------------------------------------ F) tf.data shuffle buffer
def shuffle_sim(n: int = 12, buffer: int = 4, batch: int = 4, seed: int = 2) -> list[dict]:
    """Exact simulation of tf.data's shuffle(buffer_size) algorithm."""
    rng = np.random.default_rng(seed)
    src = list(range(n)); buf: list[int] = []; out: list[int] = []; states = []
    while src and len(buf) < buffer:
        buf.append(src.pop(0))
    states.append({"src": list(src), "buf": list(buf), "out": list(out), "pick": None})
    while buf:
        j = int(rng.integers(len(buf))); pick = buf[j]
        out.append(pick)
        if src:
            buf[j] = src.pop(0)
        else:
            buf.pop(j)
        states.append({"src": list(src), "buf": list(buf), "out": list(out), "pick": pick})
    return states


def shuffle_svg(state: dict, batch: int = 4, final: bool = False) -> str:
    s = K.svg_open(680, 250)
    s += K.text(90, 30, "source (in order)", size=11, bold=True, color=K.MUTED)
    for i, v in enumerate(state["src"]):
        s += K.box(20 + i * 30, 40, 26, 26, str(v), K.BLUE, size=10)
    s += K.text(340, 30, "shuffle buffer", size=11, bold=True, color=K.VIOLET)
    for i, v in enumerate(state["buf"]):
        s += K.box(280 + i * 32, 40, 28, 26, str(v), K.VIOLET, size=10)
    s += K.text(560, 30, "picked at random →", size=11, bold=True, color=K.PINK)
    if state["pick"] is not None:
        s += K.box(540, 40, 36, 26, str(state["pick"]), K.PINK, filled=True, size=11)
    s += K.text(340, 110, "output stream (then .batch(4))", size=11, bold=True, color=K.EMERALD)
    for i, v in enumerate(state["out"]):
        bi = i // batch
        col = K.PALETTE[(bi + 3) % 7]
        s += K.box(40 + i * 50 + bi * 12, 125, 42, 30, str(v), col, filled=final, size=11)
    if final:
        for bi in range(math.ceil(len(state["out"]) / batch)):
            x0 = 40 + bi * (batch * 50 + 12)
            s += K.text(x0 + batch * 25 - 4, 180, f"batch {bi + 1}: shape (4, …)", size=11, bold=True, color=K.PALETTE[(bi + 3) % 7])
        s += K.text(340, 225, ".prefetch(AUTOTUNE): prepare the next batch while the model trains on this one", size=11, color=K.MUTED)
    return s + "</svg>"


# ------------------------------------------------------------------ G) PyTorch loop + zero_grad (real torch)
LOOP = [("optimizer.zero_grad()", "امسح التدرجات القديمة المخزّنة في `.grad` (PyTorch **يجمع** التدرجات افتراضيًا).", K.VIOLET),
        ("out = model(xb)", "التمرير الأمامي؛ Autograd يسجّل العمليات (مثل GradientTape).", K.BLUE),
        ("loss = loss_fn(out, yb)", "خسارة الدفعة: رقم واحد (موتر رتبة 0).", K.AMBER),
        ("loss.backward()", "الانتشار العكسي: يملأ `p.grad` لكل معلمة.", K.PINK),
        ("optimizer.step()", "التحديث: `p ← p − η·p.grad` لكل معلمة.", K.EMERALD)]


def loop_svg(k: int) -> str:
    s = K.svg_open(680, 250)
    cx, cy, r = 200, 125, 90
    for i, (code, _, col) in enumerate(LOOP):
        a = -math.pi / 2 + i * 2 * math.pi / len(LOOP)
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        s += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{22 if i == k else 16}" fill="{col if i == k else K.tint(col, 0.75)}" stroke="{col}" stroke-width="2"/>'
        s += K.text(x, y + 5, str(i + 1), size=13, bold=True, color="#fff" if i == k else col)
    s += K.text(cx, cy + 5, "for xb, yb in loader:", size=11, color=K.MUTED, mono=True)
    for i, (code, _, col) in enumerate(LOOP):
        y = 50 + i * 36
        s += f'<rect x="330" y="{y - 18}" width="330" height="30" rx="7" fill="{"#0F172A" if i == k else "#F8FAFC"}" stroke="{col}"/>'
        s += K.text(340, y + 2, f"{i + 1}  {code}", size=12, color="#E2E8F0" if i == k else col, anchor="start", mono=True, bold=i == k)
    return s + "</svg>"


@st.cache_data(ttl=3600, max_entries=2, show_spinner="يجري تجربة zero_grad بـ PyTorch…")
def zero_grad_data() -> dict:
    try:
        from labs.fw import torch_zero_grad_experiment

        return torch_zero_grad_experiment(steps=6, lr=0.1, seed=0)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def zero_grad_svg(d: dict, step: int) -> str:
    s = K.svg_open(680, 260)
    w, wo = d["with"][: step + 1], d["without"][: step + 1]
    mx = max(abs(r["g00"]) for r in d["with"] + d["without"]) * 1.15
    P = K.Panel(60, 30, 560, 180, (0.5, 6.5, -mx, mx))
    s += P.frame("gradient of one weight (W[0,0]) at each step")
    s += f'<line x1="{P.px(0.5)}" y1="{P.py(0)}" x2="{P.px(6.5)}" y2="{P.py(0)}" stroke="#94A3B8"/>'
    for r in w:
        x = P.px(r["step"]) - 22
        s += f'<rect x="{x:.1f}" y="{min(P.py(0), P.py(r["g00"])):.1f}" width="20" height="{abs(P.py(r["g00"]) - P.py(0)):.1f}" rx="3" fill="{K.EMERALD}"/>'
    for r in wo:
        x = P.px(r["step"]) + 2
        s += f'<rect x="{x:.1f}" y="{min(P.py(0), P.py(r["g00"])):.1f}" width="20" height="{abs(P.py(r["g00"]) - P.py(0)):.1f}" rx="3" fill="{K.RED}"/>'
    s += "".join(K.text(P.px(i), 228, f"step {i}", size=10, color=K.MUTED) for i in range(1, 7))
    s += K.legend(200, 250, [(K.EMERALD, "with zero_grad()"), (K.RED, "without zero_grad() — accumulates")], size=11)
    return s + "</svg>"
