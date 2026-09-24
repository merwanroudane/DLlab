"""Week 04 frames: neuron-by-neuron forward pass, matrix-multiplication view,
ReLU universal approximation, and architecture capacity on moons."""

from __future__ import annotations

import numpy as np
import streamlit as st

from components import svgkit as K


# ------------------------------------------------------------------ A) neuron-by-neuron forward pass (2 → 3 → 1)
NET = {
    "x": np.array([0.8, -0.5]),                           # one customer, 2 standardized features
    "names": ["debt_ratio (std.)", "income (std.)"],
    "W1": np.array([[1.2, -0.7, 0.4], [-0.9, 0.5, 1.1]]),  # (2, 3): column j = weights into hidden unit j
    "b1": np.array([0.1, -0.2, 0.05]),
    "W2": np.array([[1.5], [-1.0], [0.8]]),                # (3, 1)
    "b2": np.array([-0.3]),
}


def forward_numbers() -> dict:
    x, W1, b1, W2, b2 = NET["x"], NET["W1"], NET["b1"], NET["W2"], NET["b2"]
    z1 = x @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    p = 1 / (1 + np.exp(-z2))
    return {"z1": z1, "a1": a1, "z2": float(z2[0]), "p": float(p[0])}


def neuron_svg(stage: int) -> str:
    """stage 0 inputs · 1..3 hidden unit j · 4 output z · 5 sigmoid."""
    f = forward_numbers()
    x, W1, b1, W2, b2 = NET["x"], NET["W1"], NET["b1"], NET["W2"], NET["b2"]
    s = K.svg_open(700, 300)
    xi = [(90, 110), (90, 200)]
    hj = [(330, 70), (330, 155), (330, 240)]
    out = (560, 155)
    hcols = [K.VIOLET, K.CYAN, K.AMBER]
    for i, (px, py) in enumerate(xi):
        for j, (qx, qy) in enumerate(hj):
            on = stage == j + 1
            s += (f'<line x1="{px + 34}" y1="{py}" x2="{qx - 34}" y2="{qy}" stroke="{hcols[j] if on else "#E2E8F0"}" stroke-width="{2.6 if on else 1}"/>')
            if on:
                mx, my = px + 34 + (qx - px - 68) * 0.35, py + (qy - py) * 0.35
                s += K.text(mx, my - 4, f"{W1[i, j]:+.1f}", size=11, color=hcols[j], bold=True, mono=True)
    for j, (qx, qy) in enumerate(hj):
        on = stage in (4, 5)
        s += f'<line x1="{qx + 34}" y1="{qy}" x2="{out[0] - 38}" y2="{out[1]}" stroke="{K.PINK if on else "#E2E8F0"}" stroke-width="{2.6 if on else 1}"/>'
        if on:
            s += K.text((qx + out[0]) / 2, (qy + out[1]) / 2 - 6, f"{W2[j, 0]:+.1f}", size=11, color=K.PINK, bold=True, mono=True)
    for i, (px, py) in enumerate(xi):
        s += f'<circle cx="{px}" cy="{py}" r="32" fill="{K.tint(K.BLUE, 0.8)}" stroke="{K.BLUE}" stroke-width="2"/>'
        s += K.text(px, py + 5, f"{x[i]:+.1f}", size=14, bold=True, color=K.BLUE, mono=True)
        s += K.text(px, py + 48, NET["names"][i], size=10, color=K.MUTED)
    for j, (qx, qy) in enumerate(hj):
        done = stage > j
        on = stage == j + 1
        s += (f'<circle cx="{qx}" cy="{qy}" r="32" fill="{hcols[j] if on else (K.tint(hcols[j], 0.75) if done else "#F8FAFC")}" '
              f'stroke="{hcols[j]}" stroke-width="{2.8 if on else 1.4}"/>')
        s += K.text(qx, qy - 3, f"z={f['z1'][j]:+.2f}" if done else f"h{j + 1}", size=10, mono=True, color="#fff" if on else hcols[j], bold=on)
        if done:
            s += K.text(qx, qy + 12, f"a={f['a1'][j]:.2f}", size=11, mono=True, color="#fff" if on else hcols[j], bold=True)
        s += K.text(qx + 40, qy - 26, f"b={b1[j]:+.2f}", size=9, color=K.MUTED, anchor="start", mono=True)
    on_o = stage >= 4
    s += f'<circle cx="{out[0]}" cy="{out[1]}" r="36" fill="{K.PINK if stage == 4 else (K.tint(K.PINK, 0.75) if on_o else "#F8FAFC")}" stroke="{K.PINK}" stroke-width="2"/>'
    if on_o:
        s += K.text(out[0], out[1] + 4, f"z={f['z2']:+.3f}", size=11, mono=True, color="#fff" if stage == 4 else K.PINK, bold=True)
    if stage >= 5:
        s += K.box(620, 130, 72, 50, f"p={f['p']:.3f}", K.EMERALD, filled=True, size=12, sub="σ(z)")
    return s + "</svg>"


# ------------------------------------------------------------------ B) X @ W, cell by cell
def matmul_data() -> dict:
    rng = np.random.default_rng(4)
    X = rng.integers(-2, 3, (4, 3)).astype(float)
    W = rng.integers(-2, 3, (3, 5)).astype(float)
    return {"X": X, "W": W, "Z": X @ W}


def matmul_svg(d: dict, i: int | None, j: int | None, reveal: int) -> str:
    """Highlight row i of X and column j of W; cells revealed up to index `reveal` (row-major)."""
    X, W, Z = d["X"], d["W"], d["Z"]
    s = K.svg_open(700, 250)
    c = 30

    def grid(x0, y0, M, name, color, hl_row=None, hl_col=None, shown=None):
        out = K.text(x0 + M.shape[1] * c / 2, y0 - 10, f"{name}  {M.shape}", size=12, bold=True, color=color)
        for r in range(M.shape[0]):
            for k in range(M.shape[1]):
                on = (hl_row is not None and r == hl_row) or (hl_col is not None and k == hl_col)
                vis = shown is None or (r * M.shape[1] + k) <= shown
                fill = color if (on and shown is None) else (K.tint(color, 0.8) if vis else "#F8FAFC")
                if shown is not None and hl_row is not None and r == hl_row and k == hl_col:
                    fill = color
                out += f'<rect x="{x0 + k * c}" y="{y0 + r * c}" width="{c - 3}" height="{c - 3}" rx="5" fill="{fill}" stroke="{color}"/>'
                if vis:
                    t_col = "#fff" if fill == color else color
                    out += K.text(x0 + k * c + (c - 3) / 2, y0 + r * c + 19, f"{M[r, k]:g}", size=11, color=t_col, mono=True, bold=True)
        return out

    s += grid(30, 40, X, "X (batch, in)", K.BLUE, hl_row=i)
    s += K.text(135, 110, "@", size=22, bold=True, color=K.MUTED)
    s += grid(160, 55, W, "W (in, out)", K.VIOLET, hl_col=j)
    s += K.text(330, 110, "=", size=22, bold=True, color=K.MUTED)
    s += grid(360, 40, Z, "Z = XW (batch, out)", K.PINK, hl_row=i, hl_col=j, shown=reveal)
    if i is not None and j is not None:
        terms = " + ".join(f"({X[i, k]:g})({W[k, j]:g})" for k in range(3))
        s += K.text(350, 205, f"Z[{i},{j}] = {terms} = {Z[i, j]:g}", size=13, mono=True, bold=True, color=K.PINK)
    s += K.text(350, 235, "row of X (one observation)  ·  column of W (one hidden unit)", size=11, color=K.MUTED)
    return s + "</svg>"


# ------------------------------------------------------------------ C) universal approximation with ReLUs
def target_curve(x: np.ndarray) -> np.ndarray:
    """A smooth non-linear 'demand vs price' curve (teaching function)."""
    return 3 * np.exp(-0.35 * x) + 0.6 * np.sin(1.3 * x)


@st.cache_data(ttl=3600, max_entries=2)
def relu_approx(ks: tuple = (1, 2, 3, 5, 8, 16)) -> dict:
    """Best fit of the curve with k ReLU hinges at evenly spaced knots (least squares on the ReLU features)."""
    xs = np.linspace(0, 8, 300)
    y = target_curve(xs)
    out = []
    for k in ks:
        knots = np.linspace(0, 8, k + 1)[:-1]
        H = np.column_stack([np.ones_like(xs), xs] + [np.maximum(0, xs - t) for t in knots[1:]]) if k > 1 else np.column_stack([np.ones_like(xs), xs])
        coef, *_ = np.linalg.lstsq(H, y, rcond=None)
        fit = H @ coef
        out.append({"k": k, "knots": knots.tolist(), "fit": fit.tolist(), "mse": float(np.mean((fit - y) ** 2)), "n_params": int(3 * k + 1)})
    return {"x": xs.tolist(), "y": y.tolist(), "items": out}


def approx_svg(d: dict, i: int) -> str:
    it = d["items"][i]
    s = K.svg_open(680, 270)
    P = K.Panel(60, 30, 560, 200, (0, 8, -0.8, 3.4))
    s += P.frame(f"{it['k']} ReLU unit{'s' if it['k'] > 1 else ''}  ·  MSE = {it['mse']:.4f}", "price (x)", "demand (y)")
    s += P.yticks([0, 1, 2, 3])
    s += P.xticks([0, 2, 4, 6, 8])
    s += P.polyline(d["x"], d["y"], "#94A3B8", 5)
    s += P.polyline(d["x"], it["fit"], K.PINK, 2.6)
    for t in it["knots"][1:]:
        s += f'<line x1="{P.px(t):.1f}" y1="{P.y}" x2="{P.px(t):.1f}" y2="{P.y + P.h}" stroke="{K.VIOLET}" stroke-dasharray="3 4" stroke-opacity=".6"/>'
    s += K.legend(80, 262, [("#94A3B8", "true curve"), (K.PINK, "network: Σ vⱼ·ReLU(x − tⱼ) + linear"), (K.VIOLET, "hinge (kink) positions")], size=10)
    return s + "</svg>"


# ------------------------------------------------------------------ D) capacity on moons
@st.cache_data(ttl=3600, max_entries=2, show_spinner="يدرّب ست بنى على الهلالين…")
def capacity_runs(archs: tuple = ((), (2,), (4,), (16,), (16, 16), (64, 64)), epochs: int = 200, n_train: int = 200) -> dict:
    from labs.tinynet import Optimizer, TinyNet, make_moons

    X, y = make_moons(n_train + 400, noise=0.3, seed=5)
    Xtr, ytr, Xva, yva = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    ext = (-1.7, 2.7, -1.3, 1.8)
    xx, yy, G = K.grid(ext, 60)
    runs = []
    for arch in archs:
        net = TinyNet([2, *arch, 1], "binary", seed=0)
        opt = Optimizer(net, "adam", 0.01)
        rng = np.random.default_rng(0)
        for _ in range(epochs):
            order = rng.permutation(len(Xtr))
            for s0 in range(0, len(Xtr), 32):
                idx = order[s0:s0 + 32]
                out, cache = net.forward(Xtr[idx], train=True)
                gW, gb = net.backward(cache, ytr[idx]); opt.step(gW, gb)
        runs.append({"arch": list(arch), "params": int(net.n_params()), "train": net.metric(Xtr, ytr), "val": net.metric(Xva, yva),
                     "surface": net.predict(G.astype("float32"))[:, 0].reshape(xx.shape).round(3).tolist()})
    return {"runs": runs, "X": Xtr.tolist(), "y": ytr.tolist(), "ext": ext, "n_train": n_train}


def capacity_svg(d: dict, i: int) -> str:
    r = d["runs"][i]
    name = "2 → 1 (logistic)" if not r["arch"] else "2 → " + " → ".join(map(str, r["arch"])) + " → 1"
    s = K.svg_open(680, 280)
    P = K.Panel(40, 30, 330, 230, d["ext"])
    s += P.frame(name)
    s += P.image(K.heat_uri(np.array(r["surface"]), vmin=0, vmax=1), opacity=0.8)
    s += P.points(np.array(d["X"]), np.array(d["y"]), r=2.4)
    B = K.Panel(430, 40, 220, 190, (0, 1, 0.5, 1.0))
    for j, (lab, val, col) in enumerate([("train", r["train"], K.VIOLET), ("validation", r["val"], K.AMBER)]):
        x = 450 + j * 100
        s += f'<rect x="{x}" y="{B.py(val):.1f}" width="70" height="{B.py(0.5) - B.py(val):.1f}" rx="6" fill="{col}"/>'
        s += K.text(x + 35, B.py(val) - 6, f"{val:.3f}", size=12, bold=True, color=col)
        s += K.text(x + 35, 248, lab, size=11, color=col)
    s += K.text(540, 28, f"{r['params']} parameters · gap {r['train'] - r['val']:+.3f}", size=12, bold=True, color=K.PINK)
    return s + "</svg>"
