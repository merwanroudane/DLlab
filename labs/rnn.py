"""Sequence-model teaching engine (weeks 10–12): sliding windows, an explicit
NumPy RNN cell unrolled through time, LSTM and GRU cells with every gate
value exposed, and BPTT gradient magnitudes for the vanishing/exploding demo."""

from __future__ import annotations

import numpy as np


def make_windows(series: np.ndarray, window: int, horizon: int = 1):
    """(n, window, 1) inputs and (n,) targets: predict series[t+horizon-1] from the previous `window` values."""
    X, y = [], []
    for t in range(window, len(series) - horizon + 1):
        X.append(series[t - window:t]); y.append(series[t + horizon - 1])
    return np.array(X, "float32")[..., None], np.array(y, "float32")


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def rnn_forward(x_seq: np.ndarray, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, h0: np.ndarray | None = None) -> list[dict]:
    """Elman RNN unrolled: h_t = tanh(x_t Wx + h_{t-1} Wh + b). One record per timestep."""
    H = Wh.shape[0]; h = np.zeros(H) if h0 is None else h0.copy(); steps = []
    for t, x in enumerate(x_seq):
        x = np.atleast_1d(x)
        z = x @ Wx + h @ Wh + b; h_new = np.tanh(z)
        steps.append({"t": t, "x": x.tolist(), "h_prev": h.tolist(), "z": z.tolist(), "h": h_new.tolist()})
        h = h_new
    return steps


def lstm_forward(x_seq: np.ndarray, W: dict, H: int) -> list[dict]:
    """LSTM cell unrolled. W holds Wf, Wi, Wc, Wo (each (D+H, H)) and bf, bi, bc, bo.
    Records every gate, candidate, cell and hidden state per timestep."""
    h = np.zeros(H); c = np.zeros(H); steps = []
    for t, x in enumerate(x_seq):
        x = np.atleast_1d(x); xh = np.concatenate([x, h])
        f = _sig(xh @ W["Wf"] + W["bf"]); i = _sig(xh @ W["Wi"] + W["bi"]); g = np.tanh(xh @ W["Wc"] + W["bc"]); o = _sig(xh @ W["Wo"] + W["bo"])
        c_new = f * c + i * g; h_new = o * np.tanh(c_new)
        steps.append({"t": t, "x": x.tolist(), "h_prev": h.tolist(), "c_prev": c.tolist(), "f": f.tolist(), "i": i.tolist(), "g": g.tolist(), "o": o.tolist(), "c": c_new.tolist(), "h": h_new.tolist()})
        h, c = h_new, c_new
    return steps


def gru_forward(x_seq: np.ndarray, W: dict, H: int) -> list[dict]:
    """GRU cell unrolled. W holds Wz, Wr (each (D+H, H)), Wh ((D+H, H)) and bz, br, bh."""
    h = np.zeros(H); steps = []
    for t, x in enumerate(x_seq):
        x = np.atleast_1d(x); xh = np.concatenate([x, h])
        z = _sig(xh @ W["Wz"] + W["bz"]); r = _sig(xh @ W["Wr"] + W["br"])
        h_tilde = np.tanh(np.concatenate([x, r * h]) @ W["Wh"] + W["bh"])
        h_new = (1 - z) * h + z * h_tilde
        steps.append({"t": t, "x": x.tolist(), "h_prev": h.tolist(), "z": z.tolist(), "r": r.tolist(), "h_tilde": h_tilde.tolist(), "h": h_new.tolist()})
        h = h_new
    return steps


def random_weights(kind: str, D: int, H: int, seed: int = 0, scale: float = 0.6) -> dict:
    rng = np.random.default_rng(seed)
    if kind == "rnn":
        return {"Wx": rng.normal(0, scale, (D, H)), "Wh": rng.normal(0, scale, (H, H)), "b": np.zeros(H)}
    if kind == "lstm":
        return {k: rng.normal(0, scale, (D + H, H)) for k in ("Wf", "Wi", "Wc", "Wo")} | {k: np.zeros(H) for k in ("bf", "bi", "bc", "bo")}
    return {k: rng.normal(0, scale, (D + H, H)) for k in ("Wz", "Wr", "Wh")} | {k: np.zeros(H) for k in ("bz", "br", "bh")}


def bptt_gradient_norms(T: int, H: int = 8, wh_scale: float = 0.5, seed: int = 0) -> list[float]:
    """‖∂h_T/∂h_t‖ for t = T..1 in a tanh RNN with random inputs: the product
    of Jacobians diag(1−h²) Whᵀ — shrinks (vanishing) or grows (exploding)
    with wh_scale. Returns norms ordered from t = T (1.0) back to t = 1."""
    rng = np.random.default_rng(seed)
    Wx = rng.normal(0, 0.5, (1, H))
    Wh = rng.normal(0, 1, (H, H)); Wh = Wh / np.abs(np.linalg.eigvals(Wh)).max() * wh_scale     # spectral radius = wh_scale
    x = rng.normal(size=(T, 1)); h = np.zeros(H); hs = []
    for t in range(T):
        h = np.tanh(x[t] @ Wx + h @ Wh); hs.append(h)
    J = np.eye(H); norms = [1.0]
    for t in range(T - 1, 0, -1):
        J = J @ (np.diag(1 - hs[t] ** 2) @ Wh.T)
        norms.append(float(np.linalg.norm(J, 2)))
    return norms


def count_params(kind: str, D: int, H: int) -> int:
    per_gate = (D + H) * H + H
    return {"rnn": per_gate, "lstm": 4 * per_gate, "gru": 3 * per_gate}[kind]
