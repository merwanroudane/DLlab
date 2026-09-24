"""Week 01 shared computations and SVG frames.

Every number shown in the week-01 animations comes from `gd_trace` (real
gradient descent on `labs.datasets.study_hours`) or from the small simulations
below — nothing is typed in by hand.
"""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from components.diagram import svg_arrow, svg_box, svg_defs, svg_text
from labs.datasets import loan_default, study_hours

TEAL, RED, BLUE, GREY, PURPLE, GREEN = "#1F7A78", "#C8473A", "#2F6FB5", "#6B675F", "#7C5CBF", "#2E8B57"


# ------------------------------------------------------------------ training
@st.cache_data(ttl=3600, max_entries=16)
def gd_trace(lr: float, epochs: int) -> dict:
    """Full-batch gradient descent for score = w*hours + b with MSE loss.

    Returns per-epoch w, b, loss, and the gradients used at each step
    (index k = state *after* k epochs; k = 0 is the initial state w = b = 0).
    """
    df = study_hours()
    x, y = df["hours"].to_numpy(float), df["score"].to_numpy(float)
    w = b = 0.0
    ws, bs, losses, gws, gbs = [w], [b], [float(np.mean((w * x + b - y) ** 2))], [], []
    for _ in range(epochs):
        e = w * x + b - y
        gw, gb = float(2 * np.mean(e * x)), float(2 * np.mean(e))
        w, b = w - lr * gw, b - lr * gb
        gws.append(gw); gbs.append(gb)
        ws.append(w); bs.append(b)
        loss = float(np.mean((w * x + b - y) ** 2))
        losses.append(loss if math.isfinite(loss) else float("inf"))
        if not math.isfinite(loss) or abs(w) > 1e12:
            break
    ols_w, ols_b = np.polyfit(x, y, 1)
    return {"x": x.tolist(), "y": y.tolist(), "w": ws, "b": bs, "loss": losses, "gw": gws, "gb": gbs,
            "ols": (float(ols_w), float(ols_b)), "ols_loss": float(np.mean((ols_w * x + ols_b - y) ** 2))}


def frame_epochs(n_done: int) -> list[int]:
    """Epochs to show as animation frames (dense early, sparse late)."""
    base = [0, 1, 2, 3, 5, 8, 12, 20, 35, 50, 80, 120, 200, 300, 400, 500, 600, 800, 1000]
    return [e for e in base if e <= n_done] + ([n_done] if n_done not in base else [])


def fmt(v: float, nd: int = 2) -> str:
    if not math.isfinite(v):
        return "∞"
    if abs(v) >= 1e5:
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def training_svg(tr: dict, k: int) -> str:
    """Left: data + current line. Right: loss curve up to epoch k (log scale)."""
    x, y = tr["x"], tr["y"]
    w, b = tr["w"][k], tr["b"][k]
    s = '<svg viewBox="0 0 660 270" width="100%" style="max-width:660px">'
    # ---- left panel: scatter + line
    L, T, W, H = 45, 30, 270, 200
    sx = lambda v: L + v / 10 * W                      # hours 0..10
    sy = lambda v: T + H - v / 110 * H                 # score 0..110
    s += f'<rect x="{L}" y="{T}" width="{W}" height="{H}" fill="#FFFDF8" stroke="#EADFCD"/>'
    for gy in (0, 25, 50, 75, 100):
        s += f'<line x1="{L}" y1="{sy(gy)}" x2="{L + W}" y2="{sy(gy)}" stroke="#F1EADC"/>' + svg_text(L - 6, sy(gy) + 4, str(gy), size=10, color=GREY, anchor="end")
    for gx in (0, 2, 4, 6, 8, 10):
        s += svg_text(sx(gx), T + H + 14, str(gx), size=10, color=GREY)
    for xi, yi in zip(x, y):
        s += f'<circle cx="{sx(xi):.1f}" cy="{sy(yi):.1f}" r="3.2" fill="{BLUE}" fill-opacity=".75"/>'
    ow, ob = tr["ols"]
    s += f'<line x1="{sx(0)}" y1="{sy(ob):.1f}" x2="{sx(10)}" y2="{sy(ow * 10 + ob):.1f}" stroke="{GREY}" stroke-dasharray="4 4" stroke-width="1.2"/>'
    if math.isfinite(w) and math.isfinite(b):
        y0, y1 = b, w * 10 + b
        # clip the line into the panel so a diverging model stays readable
        def clip(v): return max(-40.0, min(150.0, v))
        s += f'<line x1="{sx(0)}" y1="{sy(clip(y0)):.1f}" x2="{sx(10)}" y2="{sy(clip(y1)):.1f}" stroke="{RED}" stroke-width="2.6"/>'
    s += svg_text(L + W / 2, 18, "data  +  current line ŷ = w·x + b", size=12, bold=True)
    s += svg_text(L + W / 2, T + H + 30, "hours of study (x)", size=11, color=GREY)
    # ---- right panel: loss curve
    L2, W2 = 380, 250
    s += f'<rect x="{L2}" y="{T}" width="{W2}" height="{H}" fill="#FFFDF8" stroke="#EADFCD"/>'
    losses = [v for v in tr["loss"] if math.isfinite(v)]
    n_total = max(1, len(tr["loss"]) - 1)
    lo = max(1e-3, min(losses)) if losses else 1.0
    hi = max(losses) if losses else 10.0
    lo_l, hi_l = math.log10(lo) - 0.1, math.log10(hi) + 0.1
    ly = lambda v: T + H - (math.log10(max(v, 1e-3)) - lo_l) / max(1e-9, hi_l - lo_l) * H
    lx = lambda e: L2 + e / n_total * W2
    pts = [(lx(e), ly(v)) for e, v in enumerate(tr["loss"][: k + 1]) if math.isfinite(v)]
    if len(pts) > 1:
        s += '<polyline fill="none" stroke="#B5462F" stroke-width="2" points="' + " ".join(f"{a:.1f},{c:.1f}" for a, c in pts) + '"/>'
    if pts:
        s += f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4.5" fill="{RED}"/>'
    if math.isfinite(tr["ols_loss"]) and lo_l <= math.log10(tr["ols_loss"]) <= hi_l:
        s += f'<line x1="{L2}" y1="{ly(tr["ols_loss"]):.1f}" x2="{L2 + W2}" y2="{ly(tr["ols_loss"]):.1f}" stroke="{GREY}" stroke-dasharray="4 4"/>'
        s += svg_text(L2 + W2 - 4, ly(tr["ols_loss"]) - 5, "best possible", size=10, color=GREY, anchor="end")
    s += svg_text(L2 + W2 / 2, 18, "loss (MSE, log scale)", size=12, bold=True)
    s += svg_text(L2 + W2 / 2, T + H + 30, f"epoch {k} / {n_total}", size=11, color=GREY, mono=True)
    return s + "</svg>"


# ------------------------------------------------------------------ inference
def inference_svg(stage: int, x_new: float, w: float, b: float) -> str:
    """x → (×w) → (+b) → ŷ with the frozen parameters; `stage` highlights one box."""
    s = '<svg viewBox="0 0 640 170" width="100%" style="max-width:640px">' + svg_defs()
    boxes = [(f"x = {x_new:g} h", "#E6F1FB", BLUE), (f"× w = {w:.2f}", "#FBE6E2", RED),
             (f"+ b = {b:.2f}", "#FBE6E2", RED), (f"ŷ = {w * x_new + b:.1f}", "#DDF5EA", GREEN)]
    for i, (lbl, fill, stroke) in enumerate(boxes):
        on = i <= stage
        s += svg_box(20 + i * 158, 55, 130, 56, lbl if on else "…", fill if on else "#F6F3EE",
                     stroke=stroke if i == stage else "#D9D3C7", font=14, bold=i == stage)
        if i < 3:
            s += svg_arrow(150 + i * 158, 83, 176 + i * 158, 83)
    s += svg_text(320, 30, "inference: parameters are frozen — nothing is learned here", size=12, color=GREY)
    s += svg_text(320, 145, f"w·x = {w * x_new:.2f}" if stage >= 1 else "", size=12, mono=True, color=RED)
    return s + "</svg>"


# ------------------------------------------------------------------ AI ⊃ ML ⊃ DL
def nested_svg() -> str:
    s = '<svg viewBox="0 0 640 300" width="100%" style="max-width:640px">'
    s += '<ellipse cx="320" cy="155" rx="300" ry="138" fill="#F6EEDF" stroke="#C9A96E"/>'
    s += '<ellipse cx="360" cy="170" rx="210" ry="105" fill="#E6F1FB" stroke="#2F6FB5"/>'
    s += '<ellipse cx="400" cy="185" rx="120" ry="68" fill="#EFE9FA" stroke="#7C5CBF"/>'
    s += svg_text(130, 90, "Artificial Intelligence", size=15, bold=True, color="#8A6A2E")
    s += svg_text(130, 110, "rules, search, planning, ML…", size=11, color=GREY)
    s += svg_text(260, 128, "Machine Learning", size=15, bold=True, color=BLUE)
    s += svg_text(260, 146, "learns f from data", size=11, color=GREY)
    s += svg_text(400, 180, "Deep Learning", size=15, bold=True, color=PURPLE)
    s += svg_text(400, 198, "many layers learn", size=11, color=GREY)
    s += svg_text(400, 213, "the features themselves", size=11, color=GREY)
    return s + "</svg>"


def pipelines_svg() -> str:
    """Classic ML (hand-made features) vs DL (learned features)."""
    s = '<svg viewBox="0 0 660 190" width="100%" style="max-width:660px">' + svg_defs()
    rows = [(30, "Classic ML", [("raw data", "#E6F1FB", 110), ("features by hand", "#FFF1D6", 140), ("simple model", "#EFE9FA", 120), ("ŷ", "#DDF5EA", 60)]),
            (115, "Deep Learning", [("raw data", "#E6F1FB", 110), ("layers learn the features + the model", "#EFE9FA", 290), ("ŷ", "#DDF5EA", 60)])]
    for y, name, boxes in rows:
        s += svg_text(8, y + 28, name, size=12, bold=True, anchor="start")
        x = 118
        for i, (lbl, fill, wbox) in enumerate(boxes):
            s += svg_box(x, y, wbox, 46, lbl, fill, font=12)
            if i < len(boxes) - 1:
                s += svg_arrow(x + wbox, y + 23, x + wbox + 22, y + 23)
            x += wbox + 26
    s += svg_text(118 + 110 + 26 + 70, 92, "researcher decides what matters", size=10, color=RED)
    s += svg_text(118 + 110 + 26 + 145, 177, "the network discovers what matters (needs more data)", size=10, color=PURPLE)
    return s + "</svg>"


# ------------------------------------------------------------------ prediction ≠ causation
@st.cache_data(ttl=3600, max_entries=4)
def confounding_demo(n: int = 4000, seed: int = 21) -> dict:
    """Unobserved financial stress S drives both collection calls C and default D.

    A logistic model predicts D from C very well. Then the bank "intervenes"
    and stops all calls (C := 0). The model predicts that defaults collapse;
    the true default rate does not move, because C never caused D.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    stress = rng.normal(size=n)                                   # never observed by the model
    calls = rng.poisson(np.exp(0.3 + 0.9 * stress))               # stress -> calls
    p_default = 1 / (1 + np.exp(-(-2.0 + 1.8 * stress)))          # stress -> default (calls play no role)
    default = (rng.uniform(size=n) < p_default).astype(int)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    clf = LogisticRegression().fit(calls[tr, None], default[tr])
    auc = roc_auc_score(default[te], clf.predict_proba(calls[te, None])[:, 1])
    # intervention: set calls to 0 for everyone; the true mechanism is unchanged
    default_after = (rng.uniform(size=n) < p_default).astype(int)
    pred_before = clf.predict_proba(calls[:, None])[:, 1].mean()
    pred_after = clf.predict_proba(np.zeros((n, 1)))[:, 1].mean()
    return {"auc": float(auc), "coef": float(clf.coef_[0, 0]), "rate_before": float(default.mean()),
            "rate_after": float(default_after.mean()), "pred_before": float(pred_before), "pred_after": float(pred_after),
            "corr": float(np.corrcoef(calls, default)[0, 1])}


def loan_task_svg(stage: int) -> str:
    """Research question → ML task, one box revealed per stage."""
    labels = ["Question", "Observation", "Features", "Target & type", "Data type", "Baseline → DL"]
    s = '<svg viewBox="0 0 660 120" width="100%" style="max-width:660px">' + svg_defs()
    for i, lbl in enumerate(labels):
        on = i <= stage
        s += svg_box(8 + i * 109, 38, 96, 46, lbl, "#E0F3F1" if i == stage else ("#FFFDF8" if on else "#F6F3EE"),
                     stroke=TEAL if i == stage else "#D9D3C7", font=11, bold=i == stage,
                     text_color="#2B2A28" if on else "#B9B2A6")
        if i < len(labels) - 1:
            s += svg_arrow(104 + i * 109, 61, 116 + i * 109, 61)
    return s + "</svg>"


def loan_preview():
    return loan_default().head(6)


# ------------------------------------------------------------------ baseline vs MLP by data size
@st.cache_data(ttl=3600, max_entries=4)
def size_experiment(sizes: tuple[int, ...] = (60, 150, 500, 2000), reps: int = 3) -> list[dict]:
    """Linear regression vs a small MLP on `house_prices`, both scored on the
    same 3000 unseen houses. Price has a multiplicative district effect, so a
    flexible model *can* win — but only once it has enough data."""
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from labs.datasets import house_prices

    test = house_prices(n=3000, seed=99)
    Xte, yte = test.drop(columns="price"), test["price"] / 1e5

    def prep():
        return make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["district"]),
                                       (StandardScaler(), ["area_m2", "rooms", "age_years"]))

    rows = []
    for n in sizes:
        lin_r2, mlp_r2 = [], []
        for rep in range(reps):
            df = house_prices(n=n, seed=100 + rep)
            X, y = df.drop(columns="price"), df["price"] / 1e5
            lin = make_pipeline(prep(), LinearRegression()).fit(X, y)
            mlp = make_pipeline(prep(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=2000, random_state=0)).fit(X, y)
            lin_r2.append(r2_score(yte, lin.predict(Xte))); mlp_r2.append(r2_score(yte, mlp.predict(Xte)))
        rows.append({"n": n, "linear": float(np.mean(lin_r2)), "mlp": float(np.mean(mlp_r2))})
    return rows
