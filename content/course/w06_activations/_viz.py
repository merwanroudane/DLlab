"""Week 06 frames: saturation sweep, layer-by-layer vanishing gradients
(real backprop through TinyNet), dead ReLU units, softmax step by step."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from components import svgkit as K


def _sig(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ A) saturation sweep
def saturation_svg(z0: float) -> str:
    s = K.svg_open(700, 260)
    zs = np.linspace(-7, 7, 200)
    for idx, (name, f, df, col, ylim) in enumerate([
        ("sigmoid", _sig, lambda z: _sig(z) * (1 - _sig(z)), K.BLUE, (0, 1)),
        ("tanh", np.tanh, lambda z: 1 - np.tanh(z) ** 2, K.VIOLET, (-1, 1)),
        ("ReLU", lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float), K.EMERALD, (0, 7)),
    ]):
        P = K.Panel(30 + idx * 225, 30, 200, 110, (-7, 7, *ylim))
        s += P.frame(f"{name}(z)")
        s += P.polyline(zs, f(zs), col, 2.4)
        s += P.dot(z0, float(f(np.array([z0]))[0]), K.PINK, 5)
        D = K.Panel(30 + idx * 225, 165, 200, 70, (-7, 7, 0, 1.05))
        s += D.frame(f"{name}′(z)  (what passes back)")
        s += D.polyline(zs, df(zs), col, 2)
        d = float(df(np.array([z0]))[0])
        s += D.dot(z0, d, K.PINK, 5)
        s += K.text(D.x + D.w / 2, 252, f"f′({z0:+.1f}) = {d:.4f}", size=11, mono=True, bold=True, color=K.PINK if d < 0.05 else col)
    return s + "</svg>"


# ------------------------------------------------------------------ B) vanishing gradients through 8 layers (real backprop)
@st.cache_data(ttl=3600, max_entries=2)
def layer_grad_norms(depth: int = 8, width: int = 16) -> dict:
    from labs.tinynet import TinyNet

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8)).astype("float32")
    y = (rng.uniform(size=64) < 0.5).astype("float32")
    out = {}
    for act, init in (("sigmoid", "glorot"), ("tanh", "glorot"), ("relu", "he")):
        net = TinyNet([8] + [width] * depth + [1], "binary", activation=act, seed=0, init=init)
        o, cache = net.forward(X, train=False)
        gW, _ = net.backward(cache, y)
        out[act] = [float(np.linalg.norm(g)) for g in gW]      # layer 1 (first) ... output layer
    return out


def vanish_svg(g: dict, upto: int) -> str:
    """Bars per layer, revealed from the output layer backwards (upto = number of layers revealed)."""
    L = len(g["sigmoid"])
    s = K.svg_open(700, 290)
    cols = {"sigmoid": K.BLUE, "tanh": K.VIOLET, "relu": K.EMERALD}
    lo, hi = -9, 0
    for idx, act in enumerate(("sigmoid", "tanh", "relu")):
        P = K.Panel(40 + idx * 222, 40, 200, 200, (0.5, L + 0.5, lo, hi))
        s += P.frame(f"{act}: log10 ‖∇W‖ per layer")
        for layer in range(L, 0, -1):
            if L - layer + 1 > upto:
                continue
            v = max(lo, math.log10(g[act][layer - 1] + 1e-30))
            x = P.px(layer) - 8
            s += f'<rect x="{x:.1f}" y="{P.py(v):.1f}" width="16" height="{P.py(lo) - P.py(v):.1f}" rx="3" fill="{cols[act]}" fill-opacity="{0.45 + 0.55 * layer / L}"/>'
        s += "".join(K.text(P.px(i), 252, str(i), size=9, color=K.MUTED) for i in range(1, L + 1))
        s += P.yticks([-8, -6, -4, -2, 0])
        if upto >= L:
            ratio = g[act][0] / g[act][-1]
            s += K.text(P.x + P.w / 2, 272, f"first/last = {ratio:.1e}", size=11, mono=True, bold=True, color=K.RED if ratio < 1e-2 else cols[act])
    s += K.text(350, 22, "layer 1 (input side) … layer 9 (output) — gradients flow right → left", size=11, color=K.MUTED)
    return s + "</svg>"


# ------------------------------------------------------------------ C) dead ReLU units (real training with a too-large learning rate)
@st.cache_data(ttl=3600, max_entries=4)
def dead_relu_run(lr: float = 12.0, hidden: int = 10, steps: int = 60) -> dict:
    from labs.tinynet import Optimizer, TinyNet, make_moons

    X, y = make_moons(300, noise=0.2, seed=1)
    out = {}
    for name, rate in (("small η = 0.05", 0.05), (f"large η = {lr}", lr)):
        net = TinyNet([2, hidden, 1], "binary", seed=2)
        opt = Optimizer(net, "sgd", rate)
        rng = np.random.default_rng(0)
        snaps = []
        for step in range(steps + 1):
            H = X @ net.W[0] + net.b[0]
            alive = (H > 0).any(0)
            if step in (0, 5, 15, 30, steps):
                snaps.append({"step": step, "lines": [(float(net.W[0][0, j]), float(net.W[0][1, j]), float(net.b[0][j]), bool(alive[j])) for j in range(hidden)],
                              "dead": int((~alive).sum()), "acc": net.metric(X, y)})
            idx = rng.choice(len(X), 32, replace=False)
            o, cache = net.forward(X[idx], train=True)
            gW, gb = net.backward(cache, y[idx]); opt.step(gW, gb)
        out[name] = snaps
    return {"runs": out, "X": X.tolist(), "y": y.tolist(), "hidden": hidden}


def dead_svg(d: dict, i: int) -> str:
    s = K.svg_open(700, 280)
    ext = (-1.8, 2.8, -1.4, 1.8)
    X, y = np.array(d["X"]), np.array(d["y"])
    for idx, (name, snaps) in enumerate(d["runs"].items()):
        sn = snaps[min(i, len(snaps) - 1)]
        P = K.Panel(30 + idx * 345, 35, 310, 200, ext)
        s += P.frame(f"{name} — step {sn['step']}")
        s += P.points(X, y, r=2)
        s += f'<clipPath id="dc{idx}"><rect x="{P.x}" y="{P.y}" width="{P.w}" height="{P.h}"/></clipPath><g clip-path="url(#dc{idx})">'
        for j, (w1, w2, b, alive) in enumerate(sn["lines"]):
            s += P.line_wb(w1, w2, b, color=K.PALETTE[j % 7] if alive else "#CBD5E1", width=1.8 if alive else 1.2, dash="" if alive else "4 4")
        s += "</g>"
        s += K.text(P.x + P.w / 2, 255, f"dead units: {sn['dead']}/{d['hidden']}   ·   accuracy {sn['acc']:.3f}", size=12, bold=True,
                    color=K.RED if sn["dead"] >= 3 else K.EMERALD)
    return s + "</svg>"


# ------------------------------------------------------------------ D) softmax step by step
def softmax_svg(z: list[float], stage: int, T: float = 1.0) -> str:
    z = np.array(z, float) / T
    e = np.exp(z - z.max())
    p = e / e.sum()
    names = ["repaid", "late", "default"]
    s = K.svg_open(700, 260)
    cols = [K.EMERALD, K.AMBER, K.RED]
    s += K.text(90, 30, "logits z" + (f" / T (T={T:g})" if T != 1 else ""), size=12, bold=True, color=K.VIOLET)
    for i in range(3):
        y0 = 60 + i * 60
        s += K.box(20, y0, 140, 40, f"{names[i]}: {z[i]:+.2f}", cols[i], size=12)
        if stage >= 1:
            s += K.arrow(162, y0 + 20, 215, y0 + 20, color=cols[i])
            s += K.box(220, y0, 140, 40, f"e^(z−max) = {e[i]:.3f}", cols[i], size=11)
        if stage >= 3:
            s += K.arrow(362, y0 + 20, 415, y0 + 20, color=cols[i])
            w = 200 * p[i]
            s += f'<rect x="420" y="{y0 + 4}" width="{w:.1f}" height="32" rx="6" fill="{cols[i]}"/>'
            s += K.text(430 + w, y0 + 25, f"p = {p[i]:.3f}", size=12, bold=True, color=cols[i], anchor="start")
    if stage >= 2:
        s += K.text(290, 245, f"sum = {e.sum():.3f}", size=13, bold=True, color=K.INK, mono=True)
    if stage >= 3:
        s += K.text(530, 245, f"Σp = {p.sum():.3f}", size=13, bold=True, color=K.INK, mono=True)
    return s + "</svg>"
