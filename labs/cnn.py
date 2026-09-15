"""CNN teaching engine (weeks 08–09): explicit NumPy 2-D convolution, padding,
stride, pooling, output-shape arithmetic, classic kernels, and a small
synthetic image dataset (bars / boxes / crosses) that trains in seconds."""

from __future__ import annotations

import numpy as np

KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "vertical edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),
    "horizontal edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float),
    "blur (box)": np.full((3, 3), 1 / 9.0),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "diagonal": np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], float),
}


def out_size(n: int, k: int, padding: int, stride: int) -> int:
    """Output length along one axis: floor((n + 2p − k) / s) + 1."""
    return (n + 2 * padding - k) // stride + 1


def pad2d(img: np.ndarray, p: int) -> np.ndarray:
    return np.pad(img, p, mode="constant") if p > 0 else img


def conv2d(img: np.ndarray, kernel: np.ndarray, *, padding: int = 0, stride: int = 1) -> np.ndarray:
    """Cross-correlation (what DL frameworks call convolution) of a 2-D image
    with a 2-D kernel. Returns the feature map."""
    x = pad2d(img, padding); k = kernel.shape[0]
    h = out_size(img.shape[0], k, padding, stride); w = out_size(img.shape[1], k, padding, stride)
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = float((x[i * stride:i * stride + k, j * stride:j * stride + k] * kernel).sum())
    return out


def conv_steps(img: np.ndarray, kernel: np.ndarray, *, padding: int = 0, stride: int = 1) -> list[dict]:
    """One record per kernel position: window coordinates (in the padded
    image), the products, and the resulting value — for the animation."""
    x = pad2d(img, padding); k = kernel.shape[0]
    h = out_size(img.shape[0], k, padding, stride); w = out_size(img.shape[1], k, padding, stride)
    steps = []
    for i in range(h):
        for j in range(w):
            win = x[i * stride:i * stride + k, j * stride:j * stride + k]
            steps.append({"i": i, "j": j, "r0": i * stride, "c0": j * stride, "window": win.tolist(), "products": (win * kernel).tolist(), "value": float((win * kernel).sum())})
    return steps


def pool2d(fmap: np.ndarray, size: int = 2, stride: int | None = None, mode: str = "max") -> np.ndarray:
    stride = stride or size
    h = (fmap.shape[0] - size) // stride + 1; w = (fmap.shape[1] - size) // stride + 1
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            win = fmap[i * stride:i * stride + size, j * stride:j * stride + size]
            out[i, j] = win.max() if mode == "max" else win.mean()
    return out


def conv_params(k: int, c_in: int, c_out: int) -> int:
    return k * k * c_in * c_out + c_out


def shape_trace(input_hw: int, c_in: int, layers: list[dict]) -> list[dict]:
    """Trace (H, W, C) and parameters through a list of layer specs:
    {"type": "conv", "k": 3, "filters": 16, "padding": "same"|"valid", "stride": 1}
    {"type": "pool", "size": 2} · {"type": "flatten"} · {"type": "dense", "units": n}"""
    h = w = input_hw; c = c_in; rows = [{"layer": "Input", "shape": f"({h}, {w}, {c})", "params": 0}]; flat = None
    for L in layers:
        t = L["type"]
        if t == "conv":
            p = (L["k"] - 1) // 2 if L.get("padding", "valid") == "same" else 0; s = L.get("stride", 1)
            h, w = out_size(h, L["k"], p, s), out_size(w, L["k"], p, s); params = conv_params(L["k"], c, L["filters"]); c = L["filters"]
            rows.append({"layer": f"Conv2D({L['filters']}, {L['k']}×{L['k']}, {L.get('padding', 'valid')}, stride {s})", "shape": f"({h}, {w}, {c})", "params": params})
        elif t == "pool":
            h, w = h // L["size"], w // L["size"]
            rows.append({"layer": f"MaxPooling2D({L['size']})", "shape": f"({h}, {w}, {c})", "params": 0})
        elif t == "flatten":
            flat = h * w * c
            rows.append({"layer": "Flatten", "shape": f"({flat},)", "params": 0})
        elif t == "dense":
            n_in = flat if flat is not None else h * w * c
            rows.append({"layer": f"Dense({L['units']})", "shape": f"({L['units']},)", "params": n_in * L["units"] + L["units"]}); flat = L["units"]
    return rows


def shapes_dataset(n: int = 600, size: int = 16, seed: int = 0, noise: float = 0.15):
    """Synthetic 3-class images: 0 = horizontal bar, 1 = vertical bar, 2 = cross.
    Random position/thickness + Gaussian noise. Returns X (n, size, size, 1) float32 in [0,1], y (n,) int."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size), np.float32); y = rng.integers(0, 3, n)
    for i in range(n):
        img = np.zeros((size, size), np.float32); t = int(rng.integers(1, 4))
        r = int(rng.integers(2, size - 2 - t)); c = int(rng.integers(2, size - 2 - t))
        if y[i] in (0, 2):
            img[r:r + t, 2:size - 2] = 1.0
        if y[i] in (1, 2):
            img[2:size - 2, c:c + t] = 1.0
        img += rng.normal(0, noise, (size, size)).astype(np.float32)
        X[i] = np.clip(img, 0, 1)
    return X[..., None], y.astype("int64")


CLASS_NAMES = ["horizontal bar", "vertical bar", "cross"]
