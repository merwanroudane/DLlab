"""Week 10 frames: a sliding window over the real inflation series, an RNN
cell unrolled through time with its hidden state, gradients travelling back
through time (BPTT), the chronological split and the forecast walk-through."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from components import svgkit as K
from labs.datasets import monthly_inflation
from labs.rnn import bptt_gradient_norms, random_weights, rnn_forward


@st.cache_data(ttl=3600, max_entries=2)
def inflation() -> dict:
    df = monthly_inflation()
    return {"y": df["inflation"].round(2).tolist(), "dates": df["date"].dt.strftime("%Y-%m").tolist()}


# ------------------------------------------------------------------ A) sliding window
def window_svg(series: list[float], dates: list[str], t0: int, window: int = 12) -> str:
    y = np.array(series)
    s = K.svg_open(700, 280)
    P = K.Panel(50, 30, 620, 150, (0, len(y) - 1, float(y.min()) - 0.3, float(y.max()) + 0.3))
    s += P.frame("monthly inflation (%) — 180 months")
    s += P.yticks([round(v) for v in np.linspace(y.min(), y.max(), 4)])
    s += f'<rect x="{P.px(t0):.1f}" y="{P.y}" width="{P.px(t0 + window - 1) - P.px(t0):.1f}" height="{P.h}" fill="{K.tint(K.VIOLET, 0.8)}"/>'
    s += P.polyline(range(len(y)), y, K.BLUE, 1.8)
    for t in range(t0, t0 + window):
        s += P.dot(t, y[t], K.VIOLET, 3)
    s += P.dot(t0 + window, y[t0 + window], K.PINK, 6)
    s += K.text(P.px(t0 + window / 2), P.y + 14, f"window: {dates[t0]} … {dates[t0 + window - 1]}", size=10, bold=True, color=K.VIOLET)
    # the tensor row
    for j in range(window):
        s += K.box(40 + j * 46, 205, 42, 30, f"{y[t0 + j]:.2f}", K.VIOLET, size=10)
    s += K.arrow(40 + window * 46 + 2, 220, 40 + window * 46 + 22, 220, color=K.PINK)
    s += K.box(40 + window * 46 + 24, 205, 58, 30, f"{y[t0 + window]:.2f}", K.PINK, filled=True, size=11)
    s += K.text(320, 256, f"sample #{t0}: X[{t0}] shape (12, 1)  →  y[{t0}] = inflation of {dates[t0 + window]}", size=11, mono=True, color=K.INK)
    return s + "</svg>"


# ------------------------------------------------------------------ B) RNN unrolled
@st.cache_data(ttl=3600, max_entries=2)
def unroll_data(T: int = 6, H: int = 3) -> dict:
    y = np.array(inflation()["y"])
    z = (y - y[:120].mean()) / y[:120].std()
    x = z[100:100 + T]
    W = random_weights("rnn", 1, H, seed=3, scale=0.8)
    steps = rnn_forward(x, W["Wx"], W["Wh"], W["b"])
    return {"x": x.round(3).tolist(), "steps": steps, "Wx": W["Wx"].round(2).tolist(), "Wh": W["Wh"].round(2).tolist()}


def unroll_svg(d: dict, k: int) -> str:
    T = len(d["x"])
    s = K.svg_open(700, 270)
    cw = 100
    for t in range(T):
        x0 = 15 + t * (cw + 12)
        on, done = t == k, t < k
        col = K.VIOLET if on else (K.tint(K.VIOLET, 0.35) if done else "#CBD5E1")
        s += K.box(x0 + 25, 18, 50, 30, f"{d['x'][t]:+.2f}", K.BLUE if t <= k else "#CBD5E1", size=11, sub=None)
        s += K.text(x0 + 50, 62, f"x_{t + 1}", size=10, color=K.BLUE if t <= k else "#CBD5E1")
        if t <= k:
            s += K.arrow(x0 + 50, 49, x0 + 50, 88, color=K.BLUE)
        s += (f'<rect x="{x0 + 5}" y="90" width="{cw - 10}" height="70" rx="12" fill="{K.tint(col, 0.8) if t <= k else "#F8FAFC"}" '
              f'stroke="{col}" stroke-width="{2.6 if on else 1.2}"/>')
        s += K.text(x0 + cw / 2, 108, "tanh(xWx + hWh + b)", size=8, color=col, mono=True)
        if t <= k:
            h = d["steps"][t]["h"]
            for j, v in enumerate(h):
                bh = 22 * abs(v)
                yb = 140 - (bh if v > 0 else 0)
                s += f'<rect x="{x0 + 22 + j * 20}" y="{yb:.1f}" width="14" height="{max(1, bh):.1f}" rx="2" fill="{K.PALETTE[(j + 3) % 7]}"/>'
            s += f'<line x1="{x0 + 18}" y1="140" x2="{x0 + cw - 18}" y2="140" stroke="#94A3B8"/>'
            s += K.text(x0 + cw / 2, 180, "h = [" + ", ".join(f"{v:+.2f}" for v in h) + "]", size=8, mono=True, color=col)
        if t < T - 1:
            s += K.arrow(x0 + cw - 5, 125, x0 + cw + 9, 125, color=K.PINK if t < k else "#CBD5E1", width=2.4)
    s += K.text(350, 210, "the SAME Wx, Wh, b are used at every step — the hidden state h carries memory to the right", size=11, color=K.MUTED)
    s += K.text(350, 235, "Dense(1) reads only the LAST h to predict next month", size=11, bold=True, color=K.PINK)
    return s + "</svg>"


# ------------------------------------------------------------------ C) BPTT: gradients back through time
@st.cache_data(ttl=3600, max_entries=2)
def bptt_norms(T: int = 30) -> dict:
    return {str(sc): bptt_gradient_norms(T, H=8, wh_scale=sc, seed=0) for sc in (0.5, 1.0, 1.5)}


def bptt_svg(d: dict, upto: int) -> str:
    s = K.svg_open(700, 280)
    cols = {"0.5": K.BLUE, "1.0": K.EMERALD, "1.5": K.RED}
    names = {"0.5": "ρ(Wh) = 0.5", "1.0": "ρ(Wh) = 1.0", "1.5": "ρ(Wh) = 1.5"}
    T = len(d["0.5"])
    P = K.Panel(70, 30, 600, 200, (0.5, T + 0.5, -10, 4))
    s += P.frame("log10 ‖∂h_T / ∂h_t‖ — how much of the error at step T reaches step t")
    s += P.yticks([-10, -8, -6, -4, -2, 0, 2, 4])
    s += f'<line x1="{P.px(0.5)}" y1="{P.py(0)}" x2="{P.px(T + 0.5)}" y2="{P.py(0)}" stroke="#94A3B8" stroke-dasharray="4 3"/>'
    for key, norms in d.items():
        pts_t, pts_v = [], []
        for back, v in enumerate(norms[:upto]):
            t = T - back
            pts_t.append(t); pts_v.append(max(-10, min(4, math.log10(v + 1e-30))))
        if len(pts_t) > 1:
            s += P.polyline(pts_t, pts_v, cols[key], 2.6)
        if pts_t:
            s += P.dot(pts_t[-1], pts_v[-1], cols[key], 4.5)
    s += "".join(K.text(P.px(t), 246, f"t={t}", size=9, color=K.MUTED) for t in (1, 5, 10, 15, 20, 25, 30) if t <= T)
    s += K.legend(120, 272, [(cols[k], names[k]) for k in cols], size=11)
    return s + "</svg>"


# ------------------------------------------------------------------ D) chronological split and forecast
def split_svg(n_total: int = 180, n_tr: int = 120, n_va: int = 24, window: int = 12, stage: int = 3) -> str:
    s = K.svg_open(700, 150)
    x0, w = 30, 640
    px = lambda m: x0 + m / n_total * w
    segs = [(0, n_tr, K.VIOLET, "train (fit + μ, σ)"), (n_tr, n_tr + n_va, K.AMBER, "validation"), (n_tr + n_va, n_total, K.EMERALD, "test (once)")]
    for i, (a, b, col, name) in enumerate(segs):
        if stage >= i:
            s += f'<rect x="{px(a):.1f}" y="40" width="{px(b) - px(a):.1f}" height="36" rx="6" fill="{K.tint(col, 0.6)}" stroke="{col}"/>'
            s += K.text((px(a) + px(b)) / 2, 63, name, size=11, bold=True, color=col)
    s += K.arrow(x0, 100, x0 + w, 100, color=K.MUTED)
    s += K.text(x0 + w / 2, 118, "time →   (no shuffling: the future never leaks into the past)", size=11, color=K.MUTED)
    if stage >= 3:
        s += f'<rect x="{px(n_tr + n_va - window):.1f}" y="30" width="{px(window) - px(0):.1f}" height="56" rx="4" fill="none" stroke="{K.PINK}" stroke-dasharray="4 3" stroke-width="2"/>'
        s += K.text(px(n_tr + n_va - window / 2), 25, "first test window starts in validation months", size=10, color=K.PINK)
    return s + "</svg>"


def forecast_svg(r: dict, k: int) -> str:
    true, pred, naive = r["true"], r["pred"], r["naive"]
    n = len(true)
    lo, hi = min(true + pred + naive) - 0.2, max(true + pred + naive) + 0.2
    s = K.svg_open(700, 270)
    P = K.Panel(50, 30, 620, 190, (0, n - 1, lo, hi))
    s += P.frame("test period: true vs RNN vs naive (last value)")
    s += P.yticks([round(v, 1) for v in np.linspace(lo, hi, 4)])
    s += P.polyline(range(k + 1), true[: k + 1], K.INK, 2.4)
    s += P.polyline(range(k + 1), naive[: k + 1], K.MUTED, 1.6, dash="4 3")
    s += P.polyline(range(k + 1), pred[: k + 1], K.PINK, 2.4)
    s += P.dot(k, true[k], K.INK, 4) + P.dot(k, pred[k], K.PINK, 4.5)
    e_r = np.sqrt(np.mean((np.array(pred[: k + 1]) - np.array(true[: k + 1])) ** 2))
    e_n = np.sqrt(np.mean((np.array(naive[: k + 1]) - np.array(true[: k + 1])) ** 2))
    s += K.text(360, 245, f"month {k + 1}/{n}   running RMSE: RNN {e_r:.3f} · naive {e_n:.3f}", size=12, mono=True, bold=True, color=K.PINK if e_r < e_n else K.INK)
    s += K.legend(80, 264, [(K.INK, "true"), (K.PINK, "SimpleRNN"), (K.MUTED, "naive: repeat last month")], size=10)
    return s + "</svg>"
