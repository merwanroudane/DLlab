"""Weeks 11–12 frames: the LSTM cell stage by stage with real gate values,
a memory test (LSTM cell state vs plain RNN), padding & masking, the GRU cell
stage by stage, and the four sequence-task patterns. Gate values come from
labs.rnn.lstm_forward / gru_forward (exact NumPy cells)."""

from __future__ import annotations

import numpy as np

from components import svgkit as K
from labs.rnn import gru_forward, lstm_forward, random_weights, rnn_forward


def _vec(x: float, y: float, v, col: str, label: str, lo: float = -1, hi: float = 1, w: int = 14) -> str:
    """Tiny bar chart of a small vector (values in [lo, hi])."""
    s = K.text(x + len(v) * (w + 3) / 2, y - 6, label, size=10, bold=True, color=col)
    base = y + 26 if lo < 0 else y + 44
    for j, a in enumerate(v):
        h = (abs(a) / max(abs(lo), abs(hi))) * (18 if lo < 0 else 40)
        top = base - h if a >= 0 else base
        s += f'<rect x="{x + j * (w + 3)}" y="{top:.1f}" width="{w}" height="{max(1, h):.1f}" rx="2" fill="{col}" fill-opacity=".85"/>'
    s += f'<line x1="{x - 2}" y1="{base}" x2="{x + len(v) * (w + 3)}" y2="{base}" stroke="#94A3B8"/>'
    s += K.text(x + len(v) * (w + 3) / 2, y + 62, "[" + ", ".join(f"{a:.2f}" for a in v) + "]", size=8, mono=True, color=col)
    return s


# ------------------------------------------------------------------ A) LSTM cell, one timestep, stage by stage
def lstm_step_data(seed: int = 2) -> dict:
    W = random_weights("lstm", 1, 3, seed=seed, scale=0.8)
    x = np.array([0.9, -0.4, 1.3])
    steps = lstm_forward(x, W, 3)
    return {"steps": steps}


def lstm_cell_svg(st_: dict, stage: int) -> str:
    """stage 0 inputs · 1 forget · 2 input+candidate · 3 cell update · 4 output → h."""
    s = K.svg_open(700, 310)
    # conveyor belt (cell state)
    s += f'<line x1="40" y1="60" x2="660" y2="60" stroke="{K.AMBER}" stroke-width="{5 if stage >= 3 else 3}" stroke-opacity=".8"/>'
    s += K.text(350, 30, "cell state c: the conveyor belt (only × and + touch it)", size=12, bold=True, color=K.AMBER)
    s += _vec(40, 80, st_["c_prev"], K.AMBER, "c_{t−1}")
    s += _vec(40, 190, st_["h_prev"], K.CYAN, "h_{t−1}")
    s += K.box(40, 255, 70, 34, f"x = {st_['x'][0]:+.1f}", K.BLUE, size=11)
    cols = {"f": K.RED, "i": K.EMERALD, "g": K.VIOLET, "o": K.PINK}
    if stage >= 1:
        s += K.box(170, 75, 60, 28, "× f", K.RED, filled=stage == 1, size=12)
        s += _vec(160, 140, st_["f"], K.RED, "f = σ(·) forget", lo=0, hi=1)
    if stage >= 2:
        s += K.box(300, 75, 60, 28, "+ i⊙g", K.EMERALD, filled=stage == 2, size=11)
        s += _vec(280, 140, st_["i"], K.EMERALD, "i = σ(·) write", lo=0, hi=1)
        s += _vec(400, 140, st_["g"], K.VIOLET, "g = tanh(·) candidate")
    if stage >= 3:
        s += _vec(500, 88, st_["c"], K.AMBER, "c_t = f⊙c + i⊙g", lo=-1.5, hi=1.5)
    if stage >= 4:
        s += _vec(530, 190, st_["o"], K.PINK, "o = σ(·) expose", lo=0, hi=1)
        s += K.box(580, 262, 110, 34, "h = o⊙tanh(c)", K.CYAN, filled=True, size=11)
        s += K.text(635, 305, "[" + ", ".join(f"{a:.2f}" for a in st_["h"]) + "]", size=9, mono=True, color=K.CYAN)
    return s + "</svg>"


# ------------------------------------------------------------------ B) memory test: remember a signal for T steps
def memory_test(T: int = 20) -> dict:
    """A single spike at t = 0, then zeros. LSTM with a forget-gate bias of +3
    (f ≈ 0.95) and input gate open for the spike vs a tanh RNN with ρ = 0.9.
    Also the gradient path: ∏ f (LSTM cell path) vs ∏ diag(1 − h²) Wh (RNN)."""
    x = np.zeros(T); x[0] = 2.0
    H = 1
    W = {k: np.array([[0.0], [0.0]]) for k in ("Wf", "Wi", "Wc", "Wo")} | {k: np.zeros(1) for k in ("bf", "bi", "bc", "bo")}
    W["bf"] = np.array([3.0]); W["Wi"] = np.array([[3.0], [0.0]]); W["bi"] = np.array([-1.5]); W["Wc"] = np.array([[1.5], [0.0]]); W["bo"] = np.array([2.0])
    ls = lstm_forward(x, W, H)
    rs = rnn_forward(x, np.array([[1.0]]), np.array([[0.9]]), np.zeros(1))
    c = [s_["c"][0] for s_ in ls]; hl = [s_["h"][0] for s_ in ls]; hr = [s_["h"][0] for s_ in rs]
    f = [s_["f"][0] for s_ in ls]
    g_lstm = np.cumprod([1.0] + f[1:]).tolist()
    g_rnn = np.cumprod([1.0] + [(1 - hr[t] ** 2) * 0.9 for t in range(1, T)]).tolist()
    return {"c": c, "h_lstm": hl, "h_rnn": hr, "f": f, "g_lstm": g_lstm, "g_rnn": g_rnn, "T": T}


def memory_svg(m: dict, k: int) -> str:
    T = m["T"]
    s = K.svg_open(700, 280)
    P = K.Panel(60, 30, 280, 190, (0, T - 1, -0.1, 1.2))
    s += P.frame("what is remembered of the spike at t = 1")
    s += P.polyline(range(k + 1), m["c"][: k + 1], K.AMBER, 2.8)
    s += P.polyline(range(k + 1), m["h_rnn"][: k + 1], K.BLUE, 2.4)
    s += P.dot(k, m["c"][k], K.AMBER, 4.5) + P.dot(k, m["h_rnn"][k], K.BLUE, 4.5)
    s += P.yticks([0, 0.5, 1.0])
    Q = K.Panel(410, 30, 260, 190, (0, T - 1, -8, 0.3))
    s += Q.frame("log10 gradient reaching step 1")
    s += Q.yticks([-8, -6, -4, -2, 0])
    lg = [np.log10(max(v, 1e-8)) for v in m["g_lstm"]]; lr_ = [np.log10(max(v, 1e-8)) for v in m["g_rnn"]]
    s += Q.polyline(range(k + 1), lg[: k + 1], K.AMBER, 2.8) + Q.polyline(range(k + 1), lr_[: k + 1], K.BLUE, 2.4)
    s += K.text(200, 240, f"t = {k + 1}:  LSTM c = {m['c'][k]:.3f}   RNN h = {m['h_rnn'][k]:.3f}", size=11, mono=True, bold=True, color=K.INK)
    s += K.text(540, 240, f"∏f = {m['g_lstm'][k]:.2e}  vs  RNN {m['g_rnn'][k]:.2e}", size=10, mono=True, color=K.INK)
    s += K.legend(120, 268, [(K.AMBER, "LSTM cell state (f ≈ 0.95)"), (K.BLUE, "plain RNN hidden state")], size=11)
    return s + "</svg>"


# ------------------------------------------------------------------ C) padding & masking
def padding_svg(stage: int, maxlen: int = 7) -> str:
    seqs = [[5, 3, 8], [2, 9, 4, 4, 1, 7], [6, 1], [3, 3, 5, 2]]
    s = K.svg_open(700, 250)
    names = ["customer A", "customer B", "customer C", "customer D"]
    for r, sq in enumerate(seqs):
        y = 30 + r * 50
        s += K.text(70, y + 22, names[r], size=11, color=K.INK, anchor="end")
        L = len(sq)
        row = sq if stage == 0 else sq + [0] * (maxlen - L)
        for j, v in enumerate(row):
            pad = stage >= 1 and j >= L
            masked = stage >= 2 and pad
            col = "#CBD5E1" if pad else K.PALETTE[r % 7]
            s += f'<rect x="{80 + j * 44}" y="{y}" width="40" height="34" rx="6" fill="{"#F1F5F9" if pad else K.tint(col, 0.75)}" stroke="{col}" stroke-dasharray="{"4 3" if pad else ""}"/>'
            s += K.text(100 + j * 44, y + 22, str(v), size=12, bold=not pad, color="#94A3B8" if pad else col)
            if masked:
                s += K.text(100 + j * 44, y + 8, "✕", size=9, color=K.RED)
        if stage >= 2:
            mask = [1] * L + [0] * (maxlen - L)
            s += K.text(420, y + 22, "mask: " + "".join("1" if m_ else "0" for m_ in mask), size=12, mono=True, anchor="start", color=K.EMERALD)
    lbl = ["different lengths: cannot stack into one tensor", f"post-padding with 0 to maxlen = {maxlen} → tensor (4, {maxlen}, 1)", "Masking(mask_value=0): the LSTM skips padded steps"][min(stage, 2)]
    s += K.text(350, 238, lbl, size=12, bold=True, color=K.VIOLET)
    return s + "</svg>"


# ------------------------------------------------------------------ D) GRU cell, stage by stage
def gru_step_data(seed: int = 4) -> dict:
    W = random_weights("gru", 1, 3, seed=seed, scale=0.8)
    x = np.array([0.9, -0.4, 1.3])
    return {"steps": gru_forward(x, W, 3)}


def gru_cell_svg(st_: dict, stage: int) -> str:
    """stage 0 inputs · 1 reset r · 2 candidate h̃ · 3 update z · 4 interpolation → h."""
    s = K.svg_open(700, 290)
    s += _vec(40, 50, st_["h_prev"], K.CYAN, "h_{t−1}")
    s += K.box(40, 150, 70, 34, f"x = {st_['x'][0]:+.1f}", K.BLUE, size=11)
    if stage >= 1:
        s += _vec(170, 50, st_["r"], K.ORANGE, "r = σ(·) reset", lo=0, hi=1)
        s += K.text(230, 150, "r ⊙ h_{t−1}: how much past to use", size=10, color=K.ORANGE)
    if stage >= 2:
        s += _vec(300, 50, st_["h_tilde"], K.VIOLET, "h̃ = tanh([x, r⊙h]·W)")
    if stage >= 3:
        s += _vec(440, 50, st_["z"], K.EMERALD, "z = σ(·) update", lo=0, hi=1)
    if stage >= 4:
        s += _vec(560, 50, st_["h"], K.PINK, "h = (1−z)⊙h + z⊙h̃")
        s += K.text(350, 230, "h is a per-unit blend: z ≈ 0 keeps the past, z ≈ 1 takes the candidate", size=12, bold=True, color=K.PINK)
    return s + "</svg>"


# ------------------------------------------------------------------ E) sequence-task patterns
PATTERNS = [
    ("many → one (value)", "sequence → one number: next month's inflation from 12 months", K.BLUE, 5, 1),
    ("many → one (class)", "sequence → one class: sentiment of a review from its words", K.VIOLET, 5, 1),
    ("many → many (aligned)", "one output per step: a label for every day (e.g. regime tagging)", K.EMERALD, 5, 5),
    ("many → many (horizon)", "sequence → several future values: demand for the next 3 days", K.AMBER, 5, 3),
]


def pattern_svg(k: int) -> str:
    name, desc, col, n_in, n_out = PATTERNS[k]
    s = K.svg_open(700, 250)
    s += K.text(350, 16, name, size=15, bold=True, color=col)
    for t in range(n_in):
        x = 90 + t * 100
        s += K.box(x, 150, 60, 34, f"x{t + 1}", K.BLUE, size=12)
        s += K.arrow(x + 30, 150, x + 30, 128, color=K.BLUE)
        s += f'<rect x="{x}" y="90" width="60" height="38" rx="10" fill="{K.tint(col, 0.75)}" stroke="{col}"/>'
        s += K.text(x + 30, 114, "cell", size=11, color=col, bold=True)
        if t < n_in - 1:
            s += K.arrow(x + 61, 109, x + 99, 109, color=col)
    outs = range(n_in - n_out, n_in) if n_out != n_in else range(n_in)
    for t in outs:
        x = 90 + t * 100
        s += K.arrow(x + 30, 90, x + 64 - 34, 62, color=K.PINK)
        s += K.box(x, 28, 60, 32, "ŷ" if n_out == 1 else f"ŷ{t - min(outs) + 1}", K.PINK, filled=True, size=12)
    s += K.text(350, 212, desc, size=12, color=K.INK)
    return s + "</svg>"
