"""Week 02 computations + animation frames. Every number is computed here
from the platform datasets (loan_default, house_prices, make_moons)."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from components import svgkit as K
from labs.datasets import loan_default

FEATS4 = ["income", "age", "num_late_payments", "debt_ratio"]


def _sig(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ data
@st.cache_data(ttl=3600, max_entries=4)
def loan_split(seed: int = 0) -> dict:
    """The week's canonical 60/20/20 stratified split of loan_default (as in the code lab)."""
    from sklearn.model_selection import train_test_split

    ld = loan_default(n=400, seed=7).dropna()
    X = ld[FEATS4].to_numpy(float); y = ld["defaulted"].to_numpy(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    return {"X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va, "X_te": X_te, "y_te": y_te, "mu": mu, "sd": sd}


# ------------------------------------------------------------------ 1) logistic regression trained live
@st.cache_data(ttl=3600, max_entries=8)
def logistic_trace(lr: float = 0.1, epochs: int = 150) -> dict:
    """Full-batch gradient descent on BCE with two standardized features
    (debt_ratio, num_late_payments) — the smallest classifier we can watch."""
    d = loan_split()
    cols = [3, 2]
    Z = (d["X_tr"][:, cols] - d["mu"][cols]) / d["sd"][cols]
    y = d["y_tr"].astype(float)
    w = np.zeros(2); b = 0.0
    hist = {"w1": [0.0], "w2": [0.0], "b": [0.0], "loss": [], "acc": []}

    def stats(w, b):
        p = np.clip(_sig(Z @ w + b), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), float(((p >= 0.5) == y).mean())

    l0, a0 = stats(w, b); hist["loss"].append(l0); hist["acc"].append(a0)
    for _ in range(epochs):
        p = _sig(Z @ w + b)
        gw, gb = Z.T @ (p - y) / len(y), float(np.mean(p - y))     # ∂L/∂z = p − y
        w, b = w - lr * gw, b - lr * gb
        l, a = stats(w, b)
        hist["w1"].append(float(w[0])); hist["w2"].append(float(w[1])); hist["b"].append(float(b))
        hist["loss"].append(l); hist["acc"].append(a)
    hist["Z"] = Z.tolist(); hist["y"] = d["y_tr"].tolist(); hist["base_rate"] = float(y.mean())
    return hist


LOG_EXT = (-2.4, 2.8, -1.4, 3.6)   # standardized debt_ratio (x) × late payments (y)


def logistic_frame(tr: dict, k: int) -> str:
    w1, w2, b = tr["w1"][k], tr["w2"][k], tr["b"][k]
    xx, yy, _ = K.grid(LOG_EXT, 48)
    prob = _sig(w1 * xx + w2 * yy + b)
    s = K.svg_open(680, 290)
    P = K.Panel(50, 30, 300, 220, LOG_EXT)
    s += P.frame("p(default) surface + boundary p = 0.5", "debt_ratio (standardized)", "late payments (std.)")
    s += P.image(K.heat_uri(prob, vmin=0, vmax=1), opacity=0.75)
    Z = np.array(tr["Z"]); y = np.array(tr["y"])
    jitter = np.random.default_rng(0).normal(0, 0.06, len(Z))
    s += P.points(np.column_stack([Z[:, 0], Z[:, 1] + jitter]), y, r=2.6)
    s += P.line_wb(w1, w2, b, color=K.INK, width=2.6)
    # loss curve
    L = K.Panel(420, 30, 230, 220, (0, len(tr["loss"]) - 1, 0.3, max(0.75, max(tr["loss"]) + 0.02)))
    s += L.frame("training log-loss (BCE)", "epoch")
    s += L.yticks([0.4, 0.5, 0.6, 0.7], "{:.1f}")
    s += L.polyline(range(k + 1), tr["loss"][: k + 1], K.PINK, 2.4)
    s += L.dot(k, tr["loss"][k], K.PINK)
    s += K.text(535, 282, f"epoch {k}   loss {tr['loss'][k]:.3f}   accuracy {tr['acc'][k]:.3f}", size=11, mono=True, color=K.VIOLET)
    s += K.legend(60, 282, [(K.BLUE, "repaid (0)"), (K.PINK, "defaulted (1)")], size=10)
    return s + "</svg>"


def two_heads_svg(stage: int, x_vals: list[tuple[str, float]], w: list[float], b: float, head: str) -> str:
    """Neuron with a regression head (identity) or a classification head (sigmoid)."""
    s = K.svg_open(680, 230)
    n = len(x_vals)
    z = b + sum(v * wi for (_, v), wi in zip(x_vals, w))
    for i, ((name, v), wi) in enumerate(zip(x_vals, w)):
        yy = 40 + i * (150 / max(1, n - 1))
        on = stage >= 0
        s += K.box(20, yy - 18, 150, 36, f"{name} = {v:g}", K.PALETTE[i % 7], size=11)
        if stage >= 1:
            s += K.arrow(172, yy, 300, 115, color=K.PALETTE[i % 7])
            s += K.text(235, yy + (115 - yy) / 2 - 4, f"× {wi:+.2f}", size=10, color=K.PALETTE[i % 7], mono=True)
    if stage >= 1:
        s += K.box(305, 88, 130, 54, "Σ + b", K.VIOLET, filled=stage == 1, sub=f"z = {z:.2f}" if stage >= 2 else f"b = {b:+.2f}")
    if stage >= 3:
        s += K.arrow(437, 115, 480, 115, color=K.MUTED)
        if head == "regression":
            s += K.box(485, 88, 170, 54, "identity: ŷ = z", K.EMERALD, filled=True, sub=f"ŷ = {z:.1f}")
        else:
            p = 1 / (1 + math.exp(-z))
            s += K.box(485, 88, 170, 54, "sigmoid: p = σ(z)", K.PINK, filled=True, sub=f"p = {p:.3f}")
    if stage >= 4:
        if head == "regression":
            s += K.text(570, 170, "loss: MSE   metric: RMSE", size=11, color=K.EMERALD, bold=True)
        else:
            p = 1 / (1 + math.exp(-z))
            s += K.text(570, 170, f"class = {int(p >= 0.5)} (threshold 0.5)", size=11, color=K.PINK, bold=True)
            s += K.text(570, 188, "loss: log-loss   metric: recall / AUC", size=11, color=K.PINK)
    return s + "</svg>"


# ------------------------------------------------------------------ 2) splitting + k-fold animations
def split_frames_data(n_show: int = 40, seed: int = 3) -> dict:
    d = loan_split()
    y = np.concatenate([d["y_tr"], d["y_va"], d["y_te"]])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), n_show, replace=False)
    ys = y[idx]
    # stratified 60/20/20 on the shown sample
    order = np.argsort(rng.uniform(size=n_show))
    groups = np.empty(n_show, int)
    for cls in (0, 1):
        members = [i for i in order if ys[i] == cls]
        a, b = round(0.6 * len(members)), round(0.8 * len(members))
        for j, i in enumerate(members):
            groups[i] = 0 if j < a else (1 if j < b else 2)
    return {"y": ys.tolist(), "group": groups.tolist(), "rates": (float(d["y_tr"].mean()), float(d["y_va"].mean()), float(d["y_te"].mean())),
            "sizes": (len(d["y_tr"]), len(d["y_va"]), len(d["y_te"]))}


def split_svg(data: dict, stage: int) -> str:
    """stage 0: all customers; 1: shuffled; 2: stratified into train/val/test; 3: class rates."""
    y, g = data["y"], data["group"]
    s = K.svg_open(680, 252)
    names = [("TRAIN 60%", K.VIOLET), ("VALIDATION 20%", K.AMBER), ("TEST 20%", K.EMERALD)]
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y)) if stage >= 1 else np.arange(len(y))
    counters = [0, 0, 0]
    for pos, i in enumerate(perm):
        if stage < 2:
            cx, cy = 40 + (pos % 20) * 31, 90 + (pos // 20) * 40
        else:
            gi = g[i]; c = counters[gi]; counters[gi] += 1
            base_x = [30, 400, 545][gi]
            per_row = [10, 4, 4][gi]
            cx, cy = base_x + (c % per_row) * 30 + 10, 80 + (c // per_row) * 30
        col = K.PINK if y[i] else K.BLUE
        s += f'<circle cx="{cx}" cy="{cy}" r="11" fill="{K.tint(col, 0.55)}" stroke="{col}" stroke-width="2"/>'
        s += K.text(cx, cy + 4, "1" if y[i] else "0", size=10, bold=True, color=col)
    if stage >= 2:
        for gi, (nm, col) in enumerate(names):
            x0 = [25, 395, 540][gi]; wdt = [315, 125, 125][gi]
            s += f'<rect x="{x0}" y="55" width="{wdt}" height="150" rx="12" fill="none" stroke="{col}" stroke-width="2.2" stroke-dasharray="6 4"/>'
            s += K.text(x0 + wdt / 2, 47, nm, size=12, bold=True, color=col)
            if stage >= 3:
                s += K.text(x0 + wdt / 2, 224, f"n = {data['sizes'][gi]}", size=11, mono=True, color=col)
                s += K.text(x0 + wdt / 2, 240, f"defaults {data['rates'][gi]:.1%}", size=11, mono=True, color=col)
    else:
        s += K.text(340, 40, "customers (0 = repaid, 1 = defaulted)" + ("  — shuffled" if stage == 1 else ""), size=12, bold=True, color=K.VIOLET)
    return s + "</svg>"


@st.cache_data(ttl=3600, max_entries=2)
def kfold_scores(k: int = 5) -> list[float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    d = loan_split()
    X = np.vstack([d["X_tr"], d["X_va"]]); y = np.concatenate([d["y_tr"], d["y_va"]])
    out = []
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        m = LogisticRegression().fit((X[tr] - mu) / sd, y[tr])
        out.append(float(roc_auc_score(y[va], m.predict_proba((X[va] - mu) / sd)[:, 1])))
    return out


def kfold_svg(scores: list[float], fold: int) -> str:
    k = len(scores)
    s = K.svg_open(680, 60 + 34 * k + 40)
    for r in range(k):
        yy = 40 + r * 34
        s += K.text(40, yy + 17, f"round {r + 1}", size=11, color=K.VIOLET if r == fold else K.MUTED, bold=r == fold)
        for c in range(k):
            is_val = c == r
            col = K.AMBER if is_val else K.VIOLET
            active = r == fold
            s += (f'<rect x="{80 + c * 90}" y="{yy}" width="84" height="26" rx="7" fill="{col if (active and is_val) else K.tint(col, 0.8 if active else 0.93)}" '
                  f'stroke="{col}" stroke-width="{2 if active else 1}"/>')
            s += K.text(122 + c * 90, yy + 17, "validate" if is_val else "train", size=10, color="#fff" if (active and is_val) else col, bold=is_val)
        if r <= fold:
            s += K.text(560, yy + 17, f"AUC = {scores[r]:.3f}", size=11, mono=True, color=K.PINK if r == fold else K.MUTED, bold=r == fold, anchor="start")
    if fold == k - 1:
        m, sdv = float(np.mean(scores)), float(np.std(scores))
        s += K.text(340, 60 + 34 * k + 20, f"mean AUC = {m:.3f}  ±  {sdv:.3f}  (the ± is the honest uncertainty)", size=12, bold=True, color=K.EMERALD)
    return s + "</svg>"


# ------------------------------------------------------------------ 3) thresholds, confusion, ROC
@st.cache_data(ttl=3600, max_entries=2)
def val_probs() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve

    d = loan_split()
    Zf = lambda A: (A - d["mu"]) / d["sd"]
    m = LogisticRegression().fit(Zf(d["X_tr"]), d["y_tr"])
    p = m.predict_proba(Zf(d["X_va"]))[:, 1]
    fpr, tpr, _ = roc_curve(d["y_va"], p)
    return {"p": p.tolist(), "y": d["y_va"].tolist(), "auc": float(roc_auc_score(d["y_va"], p)), "fpr": fpr.tolist(), "tpr": tpr.tolist()}


def cm_at(p: list[float], y: list[int], thr: float) -> dict:
    p, y = np.array(p), np.array(y)
    pr = (p >= thr).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
    fn = int(((pr == 0) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (tp and not math.isnan(prec)) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "acc": (tp + tn) / len(y), "prec": prec, "rec": rec, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def threshold_svg(vp: dict, thr: float) -> str:
    c = cm_at(vp["p"], vp["y"], thr)
    s = K.svg_open(680, 260)
    s += K.confusion_svg(c["tn"], c["fp"], c["fn"], c["tp"], x=0, y=40, cell=72)
    s += K.text(115, 30, f"threshold = {thr:.2f}", size=13, bold=True, color=K.VIOLET)
    R = K.Panel(430, 30, 200, 190, (0, 1, 0, 1))
    s += R.frame(f"ROC curve (AUC = {vp['auc']:.3f})", "false positive rate", "recall (TPR)")
    s += f'<line x1="{R.px(0)}" y1="{R.py(0)}" x2="{R.px(1)}" y2="{R.py(1)}" stroke="#CBD5E1" stroke-dasharray="4 4"/>'
    s += R.polyline(vp["fpr"], vp["tpr"], K.VIOLET, 2.4)
    s += R.dot(c["fpr"], c["rec"], K.PINK, 6)
    prec = "—" if math.isnan(c["prec"]) else f"{c['prec']:.2f}"
    s += K.text(330, 248, f"accuracy {c['acc']:.2f}   precision {prec}   recall {c['rec']:.2f}   F1 {c['f1']:.2f}", size=11, mono=True, color=K.INK)
    return s + "</svg>"


# ------------------------------------------------------------------ 4) generalization: tree depth
@st.cache_data(ttl=3600, max_entries=2)
def depth_curve() -> dict:
    from sklearn.tree import DecisionTreeClassifier

    d = loan_split()
    depths = list(range(1, 16))
    tr, va = [], []
    for dp in depths:
        t = DecisionTreeClassifier(max_depth=dp, random_state=0).fit(d["X_tr"], d["y_tr"])
        tr.append(float(t.score(d["X_tr"], d["y_tr"]))); va.append(float(t.score(d["X_va"], d["y_va"])))
    return {"depth": depths, "train": tr, "val": va}


@st.cache_data(ttl=3600, max_entries=2)
def tree_surfaces(depths: tuple = (1, 2, 4, 8, 30)) -> dict:
    """Decision surfaces of trees on 2 standardized features (debt_ratio, late)."""
    from sklearn.tree import DecisionTreeClassifier

    d = loan_split(); cols = [3, 2]
    Z = (d["X_tr"][:, cols] - d["mu"][cols]) / d["sd"][cols]
    Zv = (d["X_va"][:, cols] - d["mu"][cols]) / d["sd"][cols]
    xx, yy, G = K.grid(LOG_EXT, 90)
    out = []
    for dp in depths:
        t = DecisionTreeClassifier(max_depth=dp, random_state=0).fit(Z, d["y_tr"])
        out.append({"depth": dp, "surface": t.predict_proba(G)[:, 1].reshape(xx.shape).tolist(),
                    "train": float(t.score(Z, d["y_tr"])), "val": float(t.score(Zv, d["y_va"])), "leaves": int(t.get_n_leaves())})
    return {"items": out, "Z": Z.tolist(), "y": d["y_tr"].tolist()}


def tree_svg(ts: dict, i: int) -> str:
    it = ts["items"][i]
    s = K.svg_open(680, 280)
    P = K.Panel(60, 30, 320, 220, LOG_EXT)
    s += P.frame(f"tree depth {it['depth'] if it['depth'] < 30 else 'unlimited'} — {it['leaves']} leaves", "debt_ratio (std.)", "late payments (std.)")
    s += P.image(K.heat_uri(np.array(it["surface"]), vmin=0, vmax=1), opacity=0.7)
    Z = np.array(ts["Z"]); jit = np.random.default_rng(0).normal(0, 0.06, len(Z))
    s += P.points(np.column_stack([Z[:, 0], Z[:, 1] + jit]), np.array(ts["y"]), r=2.5)
    B = K.Panel(450, 40, 190, 200, (0, 1, 0.5, 1.0))
    for j, (lab, val, col) in enumerate([("train", it["train"], K.VIOLET), ("validation", it["val"], K.AMBER)]):
        x = 470 + j * 90
        s += f'<rect x="{x}" y="{B.py(val):.1f}" width="60" height="{B.py(0.5) - B.py(val):.1f}" rx="6" fill="{col}"/>'
        s += K.text(x + 30, B.py(val) - 6, f"{val:.3f}", size=12, bold=True, color=col)
        s += K.text(x + 30, 258, lab, size=11, color=col)
    gap = it["train"] - it["val"]
    s += K.text(545, 30, f"gap = {gap:+.3f}", size=13, bold=True, color=K.RED if gap > 0.08 else K.EMERALD)
    return s + "</svg>"


# ------------------------------------------------------------------ 5) ML → DL on moons
MOON_EXT = (-1.6, 2.6, -1.2, 1.7)


@st.cache_data(ttl=3600, max_entries=4)
def moons_training(hidden: int = 16, epochs: int = 150, snap: tuple = (0, 1, 3, 6, 10, 20, 35, 60, 100, 150)) -> dict:
    from sklearn.linear_model import LogisticRegression

    from labs.tinynet import Optimizer, TinyNet, make_moons

    X, y = make_moons(600, noise=0.25, seed=0)
    Xtr, ytr, Xte, yte = X[:420], y[:420], X[420:], y[420:]
    xx, yy, G = K.grid(MOON_EXT, 70)
    log = LogisticRegression().fit(Xtr, ytr)
    net = TinyNet([2, hidden, hidden, 1], "binary", seed=0)
    opt = Optimizer(net, "adam", 0.02)
    rng = np.random.default_rng(0)
    frames = []

    def snapshot(ep):
        frames.append({"epoch": ep, "surface": net.predict(G.astype("float32"))[:, 0].reshape(xx.shape).round(3).tolist(),
                       "train_acc": net.metric(Xtr, ytr), "test_acc": net.metric(Xte, yte),
                       "loss": float(net.loss(net.predict(Xtr), ytr))})

    if 0 in snap:
        snapshot(0)
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        for s0 in range(0, len(Xtr), 32):
            idx = order[s0:s0 + 32]
            out, cache = net.forward(Xtr[idx], train=True)
            gW, gb = net.backward(cache, ytr[idx])
            opt.step(gW, gb)
        if ep in snap:
            snapshot(ep)
    return {"frames": frames, "X": Xtr.tolist(), "y": ytr.tolist(),
            "log_surface": log.predict_proba(G)[:, 1].reshape(xx.shape).round(3).tolist(),
            "log_acc": float(log.score(Xte, yte)), "log_w": (float(log.coef_[0, 0]), float(log.coef_[0, 1]), float(log.intercept_[0])),
            "n_params": int(net.n_params())}


def moons_svg(mt: dict, i: int) -> str:
    f = mt["frames"][i]
    s = K.svg_open(700, 290)
    X, y = np.array(mt["X"]), np.array(mt["y"])
    A = K.Panel(30, 30, 300, 215, MOON_EXT)
    s += A.frame(f"logistic regression — test acc {mt['log_acc']:.3f}")
    s += A.image(K.heat_uri(np.array(mt["log_surface"]), vmin=0, vmax=1), opacity=0.7)
    s += A.points(X, y, r=2.2)
    s += A.line_wb(*mt["log_w"], color=K.INK)
    B = K.Panel(370, 30, 300, 215, MOON_EXT)
    s += B.frame(f"MLP 2-16-16-1 — epoch {f['epoch']} — test acc {f['test_acc']:.3f}")
    s += B.image(K.heat_uri(np.array(f["surface"]), vmin=0, vmax=1), opacity=0.7)
    s += B.points(X, y, r=2.2)
    s += K.text(350, 270, f"MLP train loss {f['loss']:.3f}  ·  train acc {f['train_acc']:.3f}  ·  {mt['n_params']} parameters vs 3 for logistic",
                size=11, mono=True, color=K.VIOLET)
    return s + "</svg>"


@st.cache_data(ttl=3600, max_entries=2)
def hidden_units(n_hidden: int = 6, epochs: int = 300) -> dict:
    """A 2-6-1 ReLU network: each hidden unit is a line (half-plane);
    the output neuron combines the pieces into a curved boundary."""
    from labs.tinynet import Optimizer, TinyNet, make_moons

    X, y = make_moons(400, noise=0.2, seed=2)
    net = TinyNet([2, n_hidden, 1], "binary", seed=3)
    opt = Optimizer(net, "adam", 0.03)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for s0 in range(0, len(X), 32):
            idx = order[s0:s0 + 32]
            out, cache = net.forward(X[idx], train=True)
            gW, gb = net.backward(cache, y[idx]); opt.step(gW, gb)
    xx, yy, G = K.grid(MOON_EXT, 60)
    H = np.maximum(0, G @ net.W[0] + net.b[0])
    units = []
    for j in range(n_hidden):
        units.append({"w1": float(net.W[0][0, j]), "w2": float(net.W[0][1, j]), "b": float(net.b[0][j]),
                      "out_w": float(net.W[1][j, 0]), "map": H[:, j].reshape(xx.shape).round(3).tolist()})
    out = net.predict(G.astype("float32"))[:, 0].reshape(xx.shape)
    return {"units": units, "out": out.round(3).tolist(), "acc": net.metric(X, y), "X": X.tolist(), "y": y.tolist(),
            "b_out": float(net.b[1][0])}


def hidden_svg(hu: dict, i: int) -> str:
    """Frames 0..n-1: one hidden unit each; frame n: the combined output."""
    n = len(hu["units"])
    X, y = np.array(hu["X"]), np.array(hu["y"])
    s = K.svg_open(700, 290)
    # small multiples of all units so far
    for j in range(min(i + 1, n)):
        u = hu["units"][j]
        P = K.Panel(20 + (j % 3) * 112, 30 + (j // 3) * 125, 100, 95, MOON_EXT)
        s += P.frame(f"unit h{j + 1} (w_out {u['out_w']:+.1f})")
        s += P.image(K.heat_uri(np.array(u["map"]), stops=K.SEQUENTIAL, vmin=0), opacity=0.85)
        s += P.line_wb(u["w1"], u["w2"], u["b"], color=K.PALETTE[j % 7], width=2)
    R = K.Panel(390, 30, 290, 215, MOON_EXT)
    if i >= n:
        s += R.frame(f"output = σ(Σ w_out·h + b) — accuracy {hu['acc']:.3f}")
        s += R.image(K.heat_uri(np.array(hu["out"]), vmin=0, vmax=1), opacity=0.75)
    else:
        s += R.frame("the pieces so far (each unit's line)")
    s += R.points(X, y, r=2.1)
    for j in range(min(i + 1, n)):
        u = hu["units"][j]
        s += R.line_wb(u["w1"], u["w2"], u["b"], color=K.PALETTE[j % 7], width=1.8, dash="5 3")
    return s + "</svg>"


# ------------------------------------------------------------------ 6) the two simplest models (same as the code lab)
@st.cache_data(ttl=3600, max_entries=2)
def simple_models() -> dict:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split

    from labs.datasets import house_prices

    hp = house_prices(n=300, seed=11)
    Xr = hp[["area_m2", "rooms", "age_years"]].to_numpy(float); yr = hp["price"].to_numpy(float) / 1000
    Xr_tr, _, yr_tr, _ = train_test_split(Xr, yr, test_size=0.25, random_state=0)
    lin = LinearRegression().fit(Xr_tr, yr_tr)
    ld = loan_default(n=400, seed=7).dropna()
    Xc = ld[["income", "debt_ratio", "num_late_payments"]].to_numpy(float); yc = ld["defaulted"].to_numpy(int)
    Xc_tr, _, yc_tr, _ = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)
    mu, sd = Xc_tr.mean(0), Xc_tr.std(0)
    log = LogisticRegression().fit((Xc_tr - mu) / sd, yc_tr)
    return {"lin_coef": lin.coef_.tolist(), "lin_b": float(lin.intercept_),
            "log_coef": log.coef_[0].tolist(), "log_b": float(log.intercept_[0]), "mu": mu.tolist(), "sd": sd.tolist()}
