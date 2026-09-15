"""A small, dependable NumPy MLP used by the training / optimization /
regularization labs. Every step is explicit so the labs can expose the
internals (gradients, norms, per-step values) to the learner.

Not a framework — a teaching engine. Deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACT = {
    "relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


@dataclass
class TinyNet:
    sizes: list[int]                       # [d, h1, ..., out]
    task: str = "binary"                   # binary | multiclass | regression
    activation: str = "relu"
    seed: int = 0
    init: str = "he"                       # he | glorot | small | large
    W: list = field(default_factory=list)
    b: list = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W, self.b = [], []
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            scale = {"he": np.sqrt(2 / i), "glorot": np.sqrt(1 / i), "small": 0.01, "large": 1.5}[self.init]
            self.W.append(rng.normal(0, scale, (i, o)))
            self.b.append(np.zeros(o))

    # ------------------------------------------------------------ forward
    def forward(self, X, dropout: float = 0.0, rng=None, train: bool = True):
        f = ACT[self.activation][0]
        a, cache = X, {"a": [X], "z": [], "mask": []}
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            cache["z"].append(z)
            if i < L - 1:
                a = f(z)
                if dropout > 0 and train:
                    mask = (rng.uniform(size=a.shape) >= dropout) / (1 - dropout)
                    a = a * mask
                else:
                    mask = None
                cache["mask"].append(mask)
            else:
                a = {"binary": _sigmoid, "multiclass": _softmax, "regression": lambda v: v}[self.task](z)
            cache["a"].append(a)
        return a, cache

    # --------------------------------------------------------------- loss
    def loss(self, out, y, l2: float = 0.0):
        n = len(y)
        if self.task == "binary":
            p = np.clip(out[:, 0], 1e-7, 1 - 1e-7)
            L = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        elif self.task == "multiclass":
            L = -np.mean(np.log(np.clip(out[np.arange(n), y.astype(int)], 1e-7, 1)))
        else:
            L = np.mean((out[:, 0] - y) ** 2)
        if l2:
            L += l2 * sum((W ** 2).sum() for W in self.W)
        return float(L)

    # ------------------------------------------------------------ backward
    def backward(self, cache, y, l2: float = 0.0):
        df = ACT[self.activation][1]
        n = len(y)
        out = cache["a"][-1]
        if self.task == "binary":
            d = (out[:, 0] - y)[:, None] / n
        elif self.task == "multiclass":
            Y = np.eye(out.shape[1])[y.astype(int)]
            d = (out - Y) / n
        else:
            d = 2 * (out[:, 0] - y)[:, None] / n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = cache["a"][i].T @ d + 2 * l2 * self.W[i]
            gb[i] = d.sum(0)
            if i > 0:
                d = d @ self.W[i].T
                if cache["mask"][i - 1] is not None:
                    d = d * cache["mask"][i - 1]
                d = d * df(cache["z"][i - 1])
        return gW, gb

    def predict(self, X):
        return self.forward(X, train=False)[0]

    def metric(self, X, y):
        out = self.predict(X)
        if self.task == "binary":
            return float(((out[:, 0] >= 0.5) == y).mean())
        if self.task == "multiclass":
            return float((out.argmax(1) == y).mean())
        return float(np.sqrt(np.mean((out[:, 0] - y) ** 2)))

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))


class Optimizer:
    """SGD / momentum / nesterov / rmsprop / adam on a TinyNet, step by step."""

    def __init__(self, net: TinyNet, kind: str = "sgd", lr: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.net, self.kind, self.lr, self.b1, self.b2, self.eps = net, kind, lr, beta1, beta2, eps
        self.vW = [np.zeros_like(W) for W in net.W]; self.vb = [np.zeros_like(b) for b in net.b]
        self.sW = [np.zeros_like(W) for W in net.W]; self.sb = [np.zeros_like(b) for b in net.b]
        self.t = 0

    def step(self, gW, gb, clip: float | None = None):
        self.t += 1
        if clip is not None:
            norm = np.sqrt(sum((g ** 2).sum() for g in gW) + sum((g ** 2).sum() for g in gb))
            if norm > clip:
                gW = [g * clip / norm for g in gW]; gb = [g * clip / norm for g in gb]
        for i in range(len(self.net.W)):
            for P, g, v, s, key in ((self.net.W, gW, self.vW, self.sW, "W"), (self.net.b, gb, self.vb, self.sb, "b")):
                if self.kind == "sgd":
                    P[i] -= self.lr * g[i]
                elif self.kind == "momentum":
                    v[i] = self.b1 * v[i] + g[i]; P[i] -= self.lr * v[i]
                elif self.kind == "nesterov":
                    v_prev = v[i].copy(); v[i] = self.b1 * v[i] + g[i]
                    P[i] -= self.lr * (self.b1 * v[i] + g[i])
                elif self.kind == "rmsprop":
                    s[i] = self.b2 * s[i] + (1 - self.b2) * g[i] ** 2; P[i] -= self.lr * g[i] / (np.sqrt(s[i]) + self.eps)
                elif self.kind == "adam":
                    v[i] = self.b1 * v[i] + (1 - self.b1) * g[i]; s[i] = self.b2 * s[i] + (1 - self.b2) * g[i] ** 2
                    vh = v[i] / (1 - self.b1 ** self.t); sh = s[i] / (1 - self.b2 ** self.t)
                    P[i] -= self.lr * vh / (np.sqrt(sh) + self.eps)


def grad_norm(gW, gb):
    return float(np.sqrt(sum((g ** 2).sum() for g in gW) + sum((g ** 2).sum() for g in gb)))


def train(net: TinyNet, X, y, *, X_val=None, y_val=None, epochs=50, batch_size=32, lr=0.1, optimizer="sgd",
          l2=0.0, dropout=0.0, clip=None, shuffle=True, seed=0, early_stopping_patience=None, record_grad=False):
    """Mini-batch training with a history dict like Keras' History.history."""
    rng = np.random.default_rng(seed)
    opt = Optimizer(net, optimizer, lr)
    hist = {"loss": [], "val_loss": [], "metric": [], "val_metric": [], "grad_norm": [], "lr": []}
    best, bad, best_W = np.inf, 0, None
    n = len(X)
    for ep in range(epochs):
        order = rng.permutation(n) if shuffle else np.arange(n)
        ep_loss, ep_norm, steps = 0.0, 0.0, 0
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            out, cache = net.forward(X[idx], dropout=dropout, rng=rng, train=True)
            ep_loss += net.loss(out, y[idx], l2) * len(idx)
            gW, gb = net.backward(cache, y[idx], l2)
            if record_grad:
                ep_norm += grad_norm(gW, gb); steps += 1
            opt.step(gW, gb, clip=clip)
            if not np.all(np.isfinite(net.W[0])):
                hist["loss"].append(float("nan")); return hist
        hist["loss"].append(ep_loss / n); hist["metric"].append(net.metric(X, y)); hist["lr"].append(lr)
        hist["grad_norm"].append(ep_norm / max(steps, 1) if record_grad else np.nan)
        if X_val is not None:
            vl = net.loss(net.predict(X_val), y_val); hist["val_loss"].append(vl); hist["val_metric"].append(net.metric(X_val, y_val))
            if early_stopping_patience:
                if vl < best - 1e-4:
                    best, bad, best_W = vl, 0, ([W.copy() for W in net.W], [b.copy() for b in net.b])
                else:
                    bad += 1
                    if bad >= early_stopping_patience:
                        net.W, net.b = best_W; hist["stopped_epoch"] = ep + 1; break
    return hist


def make_moons(n=400, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n // 2)
    X1 = np.column_stack([np.cos(t), np.sin(t)]); X2 = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([X1, X2]) + rng.normal(0, noise, (n, 2)); y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    p = rng.permutation(n)
    return X[p].astype(np.float32), y[p].astype(np.float32)
