"""Week 05 frames: learning rate on a parabola, the optimizer race in a
narrow valley, SGD noise by batch size, and Adam step by step. All paths
are computed with the exact update rules (NumPy)."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from components import svgkit as K


# ------------------------------------------------------------------ A) η on a parabola
def parabola_path(lr: float, w0: float = -2.0, steps: int = 12) -> list[float]:
    w, path = w0, [w0]
    for _ in range(steps):
        w = w - lr * 2 * (w - 3)
        path.append(w)
        if abs(w) > 1e6:
            break
    return path


def parabola_svg(paths: dict[float, list[float]], k: int) -> str:
    s = K.svg_open(700, 270)
    cols = [K.MUTED, K.EMERALD, K.AMBER, K.RED]
    for idx, (lr, path) in enumerate(paths.items()):
        P = K.Panel(20 + idx * 172, 40, 155, 170, (-4, 10, 0, 50))
        s += P.frame(f"η = {lr}")
        xs = np.linspace(-4, 10, 80)
        s += P.polyline(xs, (xs - 3) ** 2 + 1, "#CBD5E1", 2)
        pts = path[: k + 1]
        for a, b in zip(pts[:-1], pts[1:]):
            if -4 <= a <= 10 and -4 <= b <= 10:
                s += f'<line x1="{P.px(a):.1f}" y1="{P.py((a - 3) ** 2 + 1):.1f}" x2="{P.px(b):.1f}" y2="{P.py(min(50, (b - 3) ** 2 + 1)):.1f}" stroke="{cols[idx]}" stroke-width="1.4" stroke-opacity=".7"/>'
        w = pts[-1]
        if -4 <= w <= 10:
            s += P.dot(w, min(49, (w - 3) ** 2 + 1), cols[idx], 6)
            s += K.text(P.x + P.w / 2, 232, f"w = {w:.3f}", size=11, mono=True, color=cols[idx], bold=True)
        else:
            s += K.text(P.x + P.w / 2, 125, "out of view!", size=13, bold=True, color=K.RED)
            s += K.text(P.x + P.w / 2, 232, f"w = {w:.3g}", size=11, mono=True, color=K.RED, bold=True)
    s += K.text(350, 262, f"step {k}", size=12, bold=True, color=K.VIOLET)
    return s + "</svg>"


# ------------------------------------------------------------------ B) optimizer race in a narrow valley
A_X, A_Y = 0.01, 1.0               # L(x, y) = 0.01 x² + y²  (curvatures 0.02 and 2: condition number 100)
START = (-9.0, 2.0)
EXT = (-10, 2, -3, 3)


def _loss(x, y):
    return A_X * x ** 2 + A_Y * y ** 2


def _grad(p):
    return np.array([2 * A_X * p[0], 2 * A_Y * p[1]])


@st.cache_data(ttl=3600, max_entries=4)
def race_paths(steps: int = 80) -> dict:
    settings = {"SGD": 0.9, "Momentum": 0.2, "RMSprop": 0.2, "Adam": 0.3}
    out = {}
    # SGD
    p = np.array(START); path = [p.copy()]
    for _ in range(steps):
        p = p - settings["SGD"] * _grad(p); path.append(p.copy())
    out["SGD"] = path
    # Momentum (Keras form: v = β v − η g ; p += v)
    p = np.array(START); v = np.zeros(2); path = [p.copy()]
    for _ in range(steps):
        v = 0.9 * v - settings["Momentum"] * _grad(p); p = p + v; path.append(p.copy())
    out["Momentum"] = path
    # RMSprop
    p = np.array(START); s = np.zeros(2); path = [p.copy()]
    for _ in range(steps):
        g = _grad(p); s = 0.9 * s + 0.1 * g ** 2; p = p - settings["RMSprop"] * g / (np.sqrt(s) + 1e-8); path.append(p.copy())
    out["RMSprop"] = path
    # Adam
    p = np.array(START); m = np.zeros(2); s = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        g = _grad(p); m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g ** 2
        mh, sh = m / (1 - 0.9 ** t), s / (1 - 0.999 ** t)
        p = p - settings["Adam"] * mh / (np.sqrt(sh) + 1e-8); path.append(p.copy())
    out["Adam"] = path
    return {"paths": {k: [list(map(float, q)) for q in v] for k, v in out.items()}, "lr": settings,
            "loss": {k: [float(_loss(*q)) for q in v] for k, v in out.items()}}


RACE_COLORS = {"SGD": K.BLUE, "Momentum": K.ORANGE, "RMSprop": K.EMERALD, "Adam": K.PINK}


def race_svg(rd: dict, k: int, only: list[str] | None = None) -> str:
    s = K.svg_open(700, 300)
    P = K.Panel(30, 30, 420, 240, EXT)
    xx, yy, _ = K.grid(EXT, 80)
    s += P.frame(f"L(x, y) = 0.01·x² + y²   ·   step {k}")
    s += P.image(K.heat_uri(np.log1p(_loss(xx, yy)), stops=["#FFFFFF", "#EDE9FE", "#C4B5FD", "#A78BFA"]), opacity=0.9)
    s += f'<clipPath id="rc"><rect x="{P.x}" y="{P.y}" width="{P.w}" height="{P.h}"/></clipPath><g clip-path="url(#rc)">'
    for lv in (0.02, 0.2, 0.6, 1.5, 3):   # contour ellipses 0.01·x² + y² = lv
        rx, ry = math.sqrt(lv / A_X), math.sqrt(lv / A_Y)
        s += (f'<ellipse cx="{P.px(0):.1f}" cy="{P.py(0):.1f}" rx="{rx / (EXT[1] - EXT[0]) * P.w:.1f}" ry="{ry / (EXT[3] - EXT[2]) * P.h:.1f}" '
              f'fill="none" stroke="#7C3AED" stroke-opacity=".25"/>')
    names = only or list(rd["paths"])
    for name in names:
        path = rd["paths"][name][: k + 1]
        col = RACE_COLORS[name]
        s += P.polyline([q[0] for q in path], [q[1] for q in path], col, 2.2)
        s += P.dot(path[-1][0], path[-1][1], col, 5)
    s += "</g>"
    s += P.dot(0, 0, K.INK, 4)
    s += K.text(P.px(0), P.py(0) + 16, "minimum", size=10, color=K.INK)
    s += P.dot(*START, K.MUTED, 4)
    # scoreboard
    for i, name in enumerate(names):
        L = rd["loss"][name][min(k, len(rd["loss"][name]) - 1)]
        y = 50 + i * 52
        col = RACE_COLORS[name]
        s += K.box(475, y, 205, 42, f"{name} (η={rd['lr'][name]})", col, size=11, sub=f"loss = {L:.4f}")
    return s + "</svg>"


# ------------------------------------------------------------------ C) SGD noise by batch size (linear regression, standardized)
@st.cache_data(ttl=3600, max_entries=2)
def sgd_noise(epochs: int = 6, lr: float = 0.1) -> dict:
    from labs.datasets import study_hours

    df = study_hours()
    x = df["hours"].to_numpy(float); y = df["score"].to_numpy(float)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()          # both standardized: optimum near (w≈0.97, b=0)
    n = len(x)
    runs = {}
    for bs in (n, 8, 1):
        rng = np.random.default_rng(0)
        w, b = -1.0, 1.0
        path = [(w, b)]; per_epoch = [0]
        for _ in range(epochs):
            order = rng.permutation(n)
            for s0 in range(0, n, bs):
                idx = order[s0:s0 + bs]
                e = w * x[idx] + b - y[idx]
                w, b = w - lr * 2 * float(np.mean(e * x[idx])), b - lr * 2 * float(np.mean(e))
                path.append((w, b))
            per_epoch.append(len(path) - 1)
        runs[bs] = {"path": path, "epoch_idx": per_epoch}
    wo, bo = np.polyfit(x, y, 1)
    return {"runs": {str(k): v for k, v in runs.items()}, "opt": (float(wo), float(bo)), "n": n,
            "x": x.tolist(), "y": y.tolist()}


def noise_svg(d: dict, epoch: int) -> str:
    s = K.svg_open(700, 270)
    ext = (-1.3, 1.5, -0.4, 1.2)
    x, y = np.array(d["x"]), np.array(d["y"])
    xx, yy, G = K.grid(ext, 60)
    L = ((G[:, 0][:, None] * x[None, :] + G[:, 1][:, None] - y[None, :]) ** 2).mean(1).reshape(xx.shape)
    labels = {str(d["n"]): f"full batch (GD, {d['n']})", "8": "mini-batch 8", "1": "SGD, batch 1"}
    cols = {str(d["n"]): K.BLUE, "8": K.EMERALD, "1": K.PINK}
    for i, key in enumerate([str(d["n"]), "8", "1"]):
        P = K.Panel(20 + i * 228, 35, 210, 190, ext)
        r = d["runs"][key]
        upto = r["epoch_idx"][min(epoch, len(r["epoch_idx"]) - 1)]
        s += P.frame(labels[key])
        s += P.image(K.heat_uri(np.log1p(L), stops=["#FFFFFF", "#EDE9FE", "#C4B5FD"]), opacity=0.9)
        pts = r["path"][: upto + 1]
        s += P.polyline([q[0] for q in pts], [q[1] for q in pts], cols[key], 1.4)
        s += P.dot(pts[-1][0], pts[-1][1], cols[key], 4.5)
        s += P.dot(*d["opt"], K.INK, 3.5)
        s += K.text(P.x + P.w / 2, 243, f"{upto} updates", size=11, mono=True, color=cols[key], bold=True)
    s += K.text(350, 264, f"after epoch {epoch}   (axes: weight w  ×  bias b; dark dot = optimum)", size=11, color=K.MUTED)
    return s + "</svg>"


# ------------------------------------------------------------------ D) Adam, one parameter, three steps
def adam_steps(grads=(4.0, 3.0, -1.0), lr=0.1, b1=0.9, b2=0.999, eps=1e-8) -> list[dict]:
    m = s = 0.0; theta = 1.0; out = []
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g * g
        mh, sh = m / (1 - b1 ** t), s / (1 - b2 ** t)
        step = lr * mh / (math.sqrt(sh) + eps)
        new = theta - step
        out.append({"t": t, "g": g, "m": m, "s": s, "mh": mh, "sh": sh, "step": step, "theta": theta, "new": new})
        theta = new
    return out
