"""Week 09 frames: one image through a trained CNN layer by layer, label-
preserving vs label-breaking augmentation, and occlusion sensitivity.
The model is the same small CNN as labs.fw.cnn_shapes_run, trained once and
cached; augmentation transforms are exact NumPy re-implementations."""

from __future__ import annotations

import numpy as np
import streamlit as st

from components import svgkit as K
from labs.cnn import CLASS_NAMES, shapes_dataset


@st.cache_resource(show_spinner="يدرّب CNN صغيرة مرة واحدة لهذا الدرس…")
def trained_cnn(epochs: int = 8, seed: int = 0):
    """Train the week-09 CNN (same architecture as cnn_shapes_run) and keep the model object."""
    from labs.fw import keras

    Kr = keras(); L = Kr.layers
    Kr.utils.set_random_seed(seed)
    X, y = shapes_dataset(600, size=16, seed=seed, noise=0.2)
    Xtr, ytr, Xva, yva, Xte, yte = X[:300], y[:300], X[300:450], y[300:450], X[450:], y[450:]
    model = Kr.Sequential([L.Input(shape=(16, 16, 1)), L.Conv2D(8, 3, padding="same", activation="relu", name="conv1"), L.MaxPooling2D(2, name="pool1"),
                           L.Conv2D(16, 3, padding="same", activation="relu", name="conv2"), L.MaxPooling2D(2, name="pool2"), L.Flatten(name="flatten"),
                           L.Dense(32, activation="relu", name="dense"), L.Dense(3, activation="softmax", name="probs")])
    model.compile(optimizer=Kr.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0)
    acc = float(model.evaluate(Xte, yte, verbose=0)[1])
    return model, Xte, yte, acc


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def layer_trace(index: int = 0) -> dict | None:
    try:
        model, Xte, yte, acc = trained_cnn()
    except Exception as exc:  # noqa: BLE001 - TensorFlow missing
        return {"error": f"{type(exc).__name__}: {exc}"}
    from labs.fw import keras

    Kr = keras()
    names = ["conv1", "pool1", "conv2", "pool2", "dense", "probs"]
    probe = Kr.Model(model.inputs, [model.get_layer(n).output for n in names])
    outs = probe.predict(Xte[index:index + 1], verbose=0)
    return {"image": Xte[index, :, :, 0].tolist(), "true": int(yte[index]), "acc": acc,
            "conv1": np.transpose(outs[0][0], (2, 0, 1)).round(3).tolist(), "pool1": np.transpose(outs[1][0], (2, 0, 1)).round(3).tolist(),
            "conv2": np.transpose(outs[2][0], (2, 0, 1)).round(3).tolist(), "pool2": np.transpose(outs[3][0], (2, 0, 1)).round(3).tolist(),
            "dense": outs[4][0].round(3).tolist(), "probs": outs[5][0].round(4).tolist()}


def _map(P: K.Panel, M) -> str:
    return P.image(K.heat_uri(np.array(M), stops=K.SEQUENTIAL, vmin=0), opacity=1.0)


def trace_svg(t: dict, stage: int) -> str:
    s = K.svg_open(700, 300)
    P0 = K.Panel(15, 40, 110, 110, (0, 1, 0, 1))
    s += P0.frame(f"input 16×16 ({CLASS_NAMES[t['true']]})")
    s += P0.image(K.heat_uri(np.array(t["image"]), stops=["#FFFFFF", "#1E1B4B"], vmin=0, vmax=1, flip_y=False))
    blocks = [("conv1", 8, 16), ("pool1", 8, 8), ("conv2", 16, 16), ("pool2", 16, 4)]
    x = 140
    for bi, (name, n, hw) in enumerate(blocks):
        if stage < bi + 1:
            break
        maps = t[name]
        cols = 2 if n == 8 else 4
        cell = 30 if n == 8 else 22
        s += K.text(x + cols * (cell + 3) / 2, 32, f"{name} ({hw}×{hw}×{n})", size=10, bold=True, color=K.PALETTE[bi])
        for j in range(n):
            r, c = divmod(j, cols)
            P = K.Panel(x + c * (cell + 3), 40 + r * (cell + 3), cell, cell, (0, 1, 0, 1))
            s += f'<rect x="{P.x - 1}" y="{P.y - 1}" width="{cell + 2}" height="{cell + 2}" fill="none" stroke="{K.tint(K.PALETTE[bi], 0.5)}"/>'
            s += P.image(K.heat_uri(np.array(maps[j]), stops=K.SEQUENTIAL, vmin=0, flip_y=False), opacity=1.0)
        x += cols * (cell + 3) + 18
    if stage >= 5:
        d = np.array(t["dense"])
        s += K.text(x + 20, 32, "dense (32)", size=10, bold=True, color=K.ORANGE)
        for j, v in enumerate(d):
            h = 4 + 26 * v / max(d.max(), 1e-6)
            s += f'<rect x="{x + (j % 8) * 6}" y="{40 + (j // 8) * 40 + 30 - h:.1f}" width="5" height="{h:.1f}" fill="{K.ORANGE}"/>'
        x += 60
    if stage >= 6:
        p = t["probs"]
        s += K.text(x + 60, 32, "softmax", size=10, bold=True, color=K.PINK)
        for c, name in enumerate(CLASS_NAMES):
            w = 110 * p[c]
            col = K.EMERALD if c == t["true"] else K.PINK
            s += f'<rect x="{x}" y="{50 + c * 34}" width="{max(1, w):.1f}" height="22" rx="4" fill="{col}"/>'
            s += K.text(x, 45 + c * 34, f"{name}: {p[c]:.3f}", size=10, anchor="start", color=col, bold=c == int(np.argmax(p)))
    return s + "</svg>"


# ------------------------------------------------------------------ augmentation (exact NumPy versions of the Keras layers' effect)
def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(img)
    h, w = img.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def aug_examples() -> list[dict]:
    X, y = shapes_dataset(20, size=16, seed=3, noise=0.1)
    idx = int(np.where(y == 0)[0][0])           # a horizontal bar
    img = X[idx, :, :, 0]
    return [
        {"img": img, "name": "original", "label": 0, "ok": True},
        {"img": _shift(img, 2, 0), "name": "translate ↓2", "label": 0, "ok": True},
        {"img": _shift(img, -2, 3), "name": "translate ↑2 →3", "label": 0, "ok": True},
        {"img": img[:, ::-1], "name": "flip left-right", "label": 0, "ok": True},
        {"img": img[::-1, :], "name": "flip up-down", "label": 0, "ok": True},
        {"img": np.clip(img + np.random.default_rng(1).normal(0, 0.15, img.shape), 0, 1), "name": "extra noise", "label": 0, "ok": True},
        {"img": np.rot90(img), "name": "rotate 90°", "label": 1, "ok": False},
    ]


def aug_svg(ex: list[dict], k: int) -> str:
    s = K.svg_open(700, 230)
    for i, e in enumerate(ex[: k + 1]):
        x = 10 + i * 98
        P = K.Panel(x, 40, 88, 88, (0, 1, 0, 1))
        col = K.EMERALD if e["ok"] else K.RED
        s += f'<rect x="{x - 3}" y="37" width="94" height="94" rx="6" fill="none" stroke="{col}" stroke-width="{3 if i == k else 1.2}"/>'
        s += P.image(K.heat_uri(e["img"], stops=["#FFFFFF", "#1E1B4B"], vmin=0, vmax=1, flip_y=False))
        s += K.text(x + 44, 30, e["name"], size=10, bold=i == k, color=col)
        s += K.text(x + 44, 150, CLASS_NAMES[e["label"]], size=10, color=col, bold=True)
        s += K.text(x + 44, 166, "label kept ✓" if e["ok"] else "label CHANGED ✗", size=10, color=col)
    return s + "</svg>"


# ------------------------------------------------------------------ occlusion sensitivity
@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def occlusion(index: int = 0, patch: int = 4) -> dict | None:
    try:
        model, Xte, yte, acc = trained_cnn()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    img = Xte[index, :, :, 0]
    c = int(yte[index])
    base = float(model.predict(img[None, :, :, None], verbose=0)[0, c])
    batch, pos = [], []
    for r in range(0, 16 - patch + 1, 2):
        for q in range(0, 16 - patch + 1, 2):
            im = img.copy(); im[r:r + patch, q:q + patch] = 0.0
            batch.append(im); pos.append((r, q))
    P = model.predict(np.array(batch)[..., None], verbose=0)[:, c]
    heat = np.zeros((16, 16)); cnt = np.zeros((16, 16))
    for (r, q), p in zip(pos, P):
        heat[r:r + patch, q:q + patch] += base - p; cnt[r:r + patch, q:q + patch] += 1
    heat = heat / np.maximum(cnt, 1)
    return {"image": img.tolist(), "true": c, "base": base, "pos": pos, "p": P.round(4).tolist(), "heat": heat.round(4).tolist(), "patch": patch}


def occl_svg(o: dict, k: int) -> str:
    img = np.array(o["image"])
    r, q = o["pos"][k]
    im = img.copy(); im[r:r + o["patch"], q:q + o["patch"]] = 0.0
    s = K.svg_open(700, 250)
    A = K.Panel(30, 40, 170, 170, (0, 16, 0, 16))
    s += A.frame("input with a blank patch")
    s += A.image(K.heat_uri(im, stops=["#FFFFFF", "#1E1B4B"], vmin=0, vmax=1, flip_y=False))
    s += f'<rect x="{A.x + q * 170 / 16:.1f}" y="{A.y + r * 170 / 16:.1f}" width="{o["patch"] * 170 / 16:.1f}" height="{o["patch"] * 170 / 16:.1f}" fill="none" stroke="{K.PINK}" stroke-width="2.5"/>'
    p = o["p"][k]
    s += K.text(330, 90, f"p({CLASS_NAMES[o['true']]}) = {p:.3f}", size=15, bold=True, color=K.PINK if o["base"] - p > 0.1 else K.EMERALD, mono=True)
    s += K.text(330, 115, f"without patch: {o['base']:.3f}", size=12, color=K.MUTED, mono=True)
    s += K.text(330, 140, f"drop = {o['base'] - p:+.3f}", size=13, bold=True, color=K.VIOLET, mono=True)
    B = K.Panel(480, 40, 170, 170, (0, 16, 0, 16))
    s += B.frame("sensitivity map (so far)")
    heat = np.zeros((16, 16)); cnt = np.zeros((16, 16))
    for (rr, qq), pp in zip(o["pos"][: k + 1], o["p"][: k + 1]):
        heat[rr:rr + o["patch"], qq:qq + o["patch"]] += o["base"] - pp; cnt[rr:rr + o["patch"], qq:qq + o["patch"]] += 1
    heat = np.where(cnt > 0, heat / np.maximum(cnt, 1), 0)
    s += B.image(K.heat_uri(heat, stops=["#FFFFFF", "#FBCFE8", "#DB2777", "#7C3AED"], vmin=0, vmax=max(0.05, float(np.max(o["heat"]))), flip_y=False))
    return s + "</svg>"
