"""Week 08 frames: the kernel sliding over a real image, multi-channel
convolution, padding/stride geometry, max-pooling, the shape flow through a
CNN and the growth of the receptive field. Every number comes from labs.cnn."""

from __future__ import annotations

import numpy as np

from components import svgkit as K
from labs.cnn import KERNELS, conv_steps, out_size, pool2d, shape_trace, shapes_dataset


def _cell_color(v: float, lo: float, hi: float) -> str:
    """Diverging blue ↔ white ↔ pink for signed values; white→violet for non-negative."""
    if lo < 0:
        m = max(abs(lo), abs(hi), 1e-9)
        t = v / m
        return K.tint(K.PINK, 1 - 0.85 * t) if t > 0 else K.tint(K.BLUE, 1 + 0.85 * t)
    t = (v - lo) / (hi - lo if hi > lo else 1)
    return K.tint(K.VIOLET, 1 - 0.85 * t)


def grid_svg(M: np.ndarray, x0: float, y0: float, c: float, *, lo: float | None = None, hi: float | None = None, fmt: str = "{:.1f}",
             hl: tuple[int, int, int] | None = None, hl_color: str = K.AMBER, shown: int | None = None, size: int = 9) -> str:
    """Draw a matrix as coloured cells. hl = (row, col, k) highlights a k×k window; shown limits cells (row-major)."""
    lo = float(np.min(M)) if lo is None else lo
    hi = float(np.max(M)) if hi is None else hi
    s = ""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            idx = i * M.shape[1] + j
            vis = shown is None or idx < shown
            fill = _cell_color(float(M[i, j]), lo, hi) if vis else "#F8FAFC"
            s += f'<rect x="{x0 + j * c:.1f}" y="{y0 + i * c:.1f}" width="{c - 1.5:.1f}" height="{c - 1.5:.1f}" rx="3" fill="{fill}" stroke="#E2E8F0" stroke-width=".6"/>'
            if vis and size:
                s += K.text(x0 + j * c + (c - 1.5) / 2, y0 + i * c + c / 2 + size * 0.35, fmt.format(M[i, j]), size=size, color=K.INK, mono=True)
    if hl is not None:
        r, cc, k = hl
        s += f'<rect x="{x0 + cc * c - 1:.1f}" y="{y0 + r * c - 1:.1f}" width="{k * c + 0.5:.1f}" height="{k * c + 0.5:.1f}" rx="4" fill="none" stroke="{hl_color}" stroke-width="3"/>'
    return s


# ------------------------------------------------------------------ A) sliding kernel on a real image
def slide_data(kernel: str = "horizontal edge", size: int = 8, seed: int = 4) -> dict:
    X, y = shapes_dataset(12, size=size, seed=seed, noise=0.0)
    idx = int(np.where(y == 2)[0][0]) if (y == 2).any() else 0          # a cross: both edge directions present
    img = X[idx, :, :, 0].round(2)
    steps = conv_steps(img, KERNELS[kernel])
    fmap = np.zeros((size - 2, size - 2))
    for s_ in steps:
        fmap[s_["i"], s_["j"]] = s_["value"]
    return {"img": img, "kernel": KERNELS[kernel], "steps": steps, "fmap": fmap, "name": kernel}


def slide_svg(d: dict, k: int) -> str:
    img, ker, st_, fmap = d["img"], d["kernel"], d["steps"][k], d["fmap"]
    s = K.svg_open(700, 300)
    c = 26
    s += K.text(20 + 4 * c, 22, "image X (8×8)", size=12, bold=True, color=K.BLUE)
    s += grid_svg(img, 20, 32, c, lo=0, hi=1, fmt="{:.0f}", hl=(st_["r0"], st_["c0"], 3), size=9)
    s += K.text(290, 22, f"kernel K: {d['name']}", size=12, bold=True, color=K.AMBER)
    s += grid_svg(ker, 250, 32, 28, fmt="{:+.0f}", size=10)
    prod = np.array(st_["products"])
    s += K.text(290, 150, "window ⊙ K", size=12, bold=True, color=K.VIOLET)
    s += grid_svg(prod, 250, 160, 28, fmt="{:+.0f}", size=10)
    s += K.text(290, 268, f"Σ = {st_['value']:+.0f}", size=16, bold=True, color=K.PINK, mono=True)
    lo, hi = float(fmap.min()), float(fmap.max())
    s += K.text(380 + 3 * 34, 22, "feature map F (6×6)", size=12, bold=True, color=K.PINK)
    s += grid_svg(fmap, 380, 32, 34, lo=min(lo, -1), hi=max(hi, 1), fmt="{:+.0f}", hl=(st_["i"], st_["j"], 1), hl_color=K.PINK, shown=k + 1, size=10)
    s += K.arrow(335, 268, 380 + st_["j"] * 34 + 16, 32 + st_["i"] * 34 + 34, color=K.PINK, dash="4 3")
    return s + "</svg>"


# ------------------------------------------------------------------ B) multi-channel convolution (exact toy numbers)
def channels_data() -> dict:
    rng = np.random.default_rng(7)
    X = rng.integers(0, 3, (3, 3, 3)).astype(float)        # (C=3, 3, 3) one window, three channels
    Kk = rng.integers(-1, 2, (3, 3, 3)).astype(float)       # one kernel = 3 slices
    per = [(X[c] * Kk[c]).sum() for c in range(3)]
    return {"X": X, "K": Kk, "per": per, "b": 1.0, "out": float(sum(per) + 1.0)}


def channels_svg(d: dict, stage: int) -> str:
    s = K.svg_open(700, 260)
    names, cols = ["R", "G", "B"], [K.RED, K.EMERALD, K.BLUE]
    for ch in range(3):
        y0 = 20 + ch * 78
        s += K.text(20, y0 + 40, names[ch], size=16, bold=True, color=cols[ch])
        s += grid_svg(d["X"][ch], 40, y0, 22, lo=0, hi=2, fmt="{:.0f}", size=9)
        if stage >= 1:
            s += K.text(125, y0 + 40, "⊙", size=18, color=K.MUTED)
            s += grid_svg(d["K"][ch], 145, y0, 22, fmt="{:+.0f}", size=9)
        if stage >= 2:
            s += K.arrow(215, y0 + 33, 265, y0 + 33, color=cols[ch])
            s += K.box(270, y0 + 13, 90, 40, f"{d['per'][ch]:+.0f}", cols[ch], size=13, sub=f"sum {names[ch]}")
    if stage >= 3:
        for ch in range(3):
            s += K.arrow(362, 53 + ch * 78, 450, 130, color=cols[ch])
        s += K.box(455, 105, 110, 50, f"{sum(d['per']):+.0f} + b", K.VIOLET, size=13, sub=f"b = {d['b']:+.0f}")
    if stage >= 4:
        s += K.arrow(567, 130, 590, 130, color=K.PINK)
        s += K.box(593, 105, 102, 50, f"{d['out']:+.0f}", K.PINK, filled=True, size=15, sub="1 cell of F")
    return s + "</svg>"


# ------------------------------------------------------------------ C) padding & stride geometry
def geometry_svg(n: int, k: int, p: int, s_: int, step: int | None = None) -> str:
    out = out_size(n, k, p, s_)
    N = n + 2 * p
    c = 24
    s = K.svg_open(700, 290)
    x0, y0 = 30, 30
    for i in range(N):
        for j in range(N):
            is_pad = i < p or j < p or i >= N - p or j >= N - p
            s += f'<rect x="{x0 + j * c}" y="{y0 + i * c}" width="{c - 2}" height="{c - 2}" rx="3" fill="{"#FEF3C7" if is_pad else K.tint(K.BLUE, 0.78)}" stroke="{"#F59E0B" if is_pad else K.BLUE}" stroke-width=".8"/>'
    positions = [(i * s_, j * s_) for i in range(out) for j in range(out)]
    upto = len(positions) if step is None else step + 1
    for q, (r, cc) in enumerate(positions[:upto]):
        on = q == upto - 1
        s += (f'<rect x="{x0 + cc * c - 1}" y="{y0 + r * c - 1}" width="{k * c}" height="{k * c}" rx="5" fill="{K.PINK if on else "none"}" fill-opacity="{0.18 if on else 0}" '
              f'stroke="{K.PINK if on else K.tint(K.PINK, 0.5)}" stroke-width="{2.6 if on else 1}"/>')
    gx = x0 + N * c + 50
    s += K.text(gx + 110, 40, f"n = {n}, k = {k}, p = {p}, s = {s_}", size=13, bold=True, color=K.VIOLET, mono=True)
    s += K.text(gx + 110, 66, f"out = ⌊({n} + 2·{p} − {k}) / {s_}⌋ + 1 = {out}", size=13, bold=True, color=K.PINK, mono=True)
    for i in range(out):
        for j in range(out):
            q = i * out + j
            fill = K.PINK if q == upto - 1 else (K.tint(K.PINK, 0.7) if q < upto else "#F8FAFC")
            s += f'<rect x="{gx + j * 22}" y="{90 + i * 22}" width="20" height="20" rx="3" fill="{fill}" stroke="{K.PINK}" stroke-width=".6"/>'
    s += K.legend(40, 282, [("#F59E0B", "zero padding"), (K.BLUE, "image"), (K.PINK, "kernel positions → output cells")], size=10)
    return s + "</svg>"


# ------------------------------------------------------------------ D) max pooling
def pool_data() -> dict:
    X, y = shapes_dataset(12, size=8, seed=4, noise=0.0)
    img = X[int(np.where(y == 2)[0][0]), :, :, 0]
    from labs.cnn import conv2d

    f = np.maximum(0, conv2d(img, KERNELS["vertical edge"]))[:4, :4].round(1)      # a real ReLU feature-map patch
    return {"f": f, "p": pool2d(f, 2, mode="max"), "a": pool2d(f, 2, mode="avg")}


def pool_svg(d: dict, q: int) -> str:
    f, p = d["f"], d["p"]
    s = K.svg_open(700, 230)
    c = 40
    r, cc = divmod(q, 2)
    s += K.text(20 + 2 * c, 22, "ReLU feature map (4×4)", size=12, bold=True, color=K.VIOLET)
    s += grid_svg(f, 20, 32, c, lo=0, hi=max(1, f.max()), fmt="{:.0f}", hl=(2 * r, 2 * cc, 2), hl_color=K.PINK, size=12)
    s += K.arrow(200, 110, 280, 110, color=K.PINK)
    s += K.text(240, 100, "max 2×2", size=11, color=K.PINK, bold=True)
    s += K.text(290 + c, 22, "max-pool (2×2)", size=12, bold=True, color=K.PINK)
    s += grid_svg(p, 290, 52, c, lo=0, hi=max(1, f.max()), fmt="{:.0f}", hl=(r, cc, 1), hl_color=K.PINK, shown=q + 1, size=12)
    s += K.text(440 + c, 22, "avg-pool (2×2)", size=12, bold=True, color=K.CYAN)
    s += grid_svg(d["a"], 440, 52, c, lo=0, hi=max(1, f.max()), fmt="{:.1f}", hl=(r, cc, 1), hl_color=K.CYAN, shown=q + 1, size=11)
    win = f[2 * r:2 * r + 2, 2 * cc:2 * cc + 2]
    s += K.text(350, 200, f"window {win.ravel().tolist()} → max {win.max():.0f}, mean {win.mean():.2f}", size=12, mono=True, bold=True, color=K.INK)
    return s + "</svg>"


# ------------------------------------------------------------------ E) shape flow and receptive field
ARCH = [{"type": "conv", "k": 3, "filters": 16, "padding": "same"}, {"type": "pool", "size": 2},
        {"type": "conv", "k": 3, "filters": 32, "padding": "same"}, {"type": "pool", "size": 2},
        {"type": "flatten"}, {"type": "dense", "units": 64}, {"type": "dense", "units": 10}]


def shape_rows(hw: int = 28, c_in: int = 1) -> list[dict]:
    return shape_trace(hw, c_in, ARCH)


def flow_svg(rows: list[dict], k: int) -> str:
    s = K.svg_open(700, 250)
    x = 10
    for i, r in enumerate(rows[: k + 1]):
        shape = r["shape"].strip("()").split(",")
        dims = [int(v) for v in shape if v.strip()]
        col = K.PALETTE[i % 7]
        if len(dims) == 3:
            h, _, ch = dims
            side = 18 + h * 3.2
            depth = min(40, 4 + ch * 0.6)
            y0 = 125 - side / 2
            s += f'<rect x="{x + depth}" y="{y0 - depth / 2}" width="{side}" height="{side}" fill="{K.tint(col, 0.85)}" stroke="{col}"/>'
            s += f'<rect x="{x}" y="{y0}" width="{side}" height="{side}" fill="{K.tint(col, 0.6 if i == k else 0.75)}" stroke="{col}" stroke-width="{2 if i == k else 1}"/>'
            w = side + depth
        else:
            n = dims[0]
            hgt = min(190, 20 + n / 12)
            s += f'<rect x="{x}" y="{125 - hgt / 2}" width="16" height="{hgt}" rx="4" fill="{K.tint(col, 0.6)}" stroke="{col}" stroke-width="{2 if i == k else 1}"/>'
            w = 16
        s += K.text(x + w / 2, 232 - (12 if (len(dims) == 1 and i % 2) else 0), r["shape"], size=9, mono=True, color=col, bold=i == k)
        x += max(w, 34) + 16
    return s + "</svg>"


def receptive_svg(n_layers: int) -> str:
    """Receptive field of one output cell after n stacked 3×3 convolutions (stride 1): 2n + 1."""
    s = K.svg_open(700, 240)
    c = 17
    size = 11
    x0, y0 = 40, 25
    rf = 2 * n_layers + 1
    mid = size // 2
    for i in range(size):
        for j in range(size):
            inside = abs(i - mid) <= n_layers and abs(j - mid) <= n_layers
            s += f'<rect x="{x0 + j * c}" y="{y0 + i * c}" width="{c - 2}" height="{c - 2}" rx="3" fill="{K.tint(K.VIOLET, 0.45) if inside else "#F8FAFC"}" stroke="#E2E8F0"/>'
    s += f'<rect x="{x0 + mid * c}" y="{y0 + mid * c}" width="{c - 2}" height="{c - 2}" rx="3" fill="{K.PINK}"/>'
    s += K.text(420, 70, f"after {n_layers} layer{'s' if n_layers > 1 else ''} of 3×3 conv", size=14, bold=True, color=K.VIOLET)
    s += K.text(420, 100, f"receptive field = {rf}×{rf} pixels", size=16, bold=True, color=K.PINK, mono=True)
    s += K.text(420, 130, f"rule: 1 + 2·(number of layers) = {rf}", size=12, color=K.MUTED, mono=True)
    s += K.text(420, 160, "(each 2×2 pooling roughly doubles the growth)", size=11, color=K.MUTED)
    return s + "</svg>"
