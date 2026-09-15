"""Framework helpers for the Frameworks Academy (spec §21).

Imports TensorFlow / Keras / PyTorch *lazily* (never at module import time,
spec §64) and captures the textual outputs learners must learn to read:
`model.summary()`, `fit()` logs, printed PyTorch modules, tensor reprs.

All heavy results are returned as plain data (dicts / strings) so they can be
cached with `st.cache_data`.
"""

from __future__ import annotations

import contextlib
import io
import os
from typing import Any

import numpy as np
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def tf():
    import tensorflow as _tf  # noqa: WPS433 - lazy on purpose
    return _tf


def keras():
    import keras as _keras
    return _keras


def torch():
    import torch as _torch
    return _torch


def versions() -> dict[str, str]:
    out = {"numpy": np.__version__}
    try:
        out["tensorflow"] = tf().__version__; out["keras"] = keras().__version__
    except Exception as e:  # noqa: BLE001
        out["tensorflow"] = f"unavailable ({type(e).__name__})"
    try:
        out["torch"] = torch().__version__
    except Exception as e:  # noqa: BLE001
        out["torch"] = f"unavailable ({type(e).__name__})"
    return out


def capture(fn, *args, **kwargs) -> tuple[Any, str]:
    """Run fn, returning (result, everything it printed)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def keras_summary(model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda s, **k: buf.write(s + "\n"))
    return buf.getvalue()


def moons(n: int = 600, noise: float = 0.25, seed: int = 0):
    from labs.tinynet import make_moons
    X, y = make_moons(n, noise=noise, seed=seed)
    k = int(0.75 * n)
    return X[:k], y[:k], X[k:], y[k:]


@st.cache_data(max_entries=32, show_spinner="يشغّل Keras…")
def keras_mlp_run(hidden: tuple, activation: str, optimizer: str, lr: float, epochs: int, batch_size: int, seed: int,
                  n: int = 600, noise: float = 0.25, verbose: int = 2, early_stopping: bool = False) -> dict:
    """Build, compile and fit a small Keras MLP on the moons data. Returns
    summary text, fit log text, history, evaluate/predict outputs and shapes."""
    K = keras(); L = K.layers
    K.utils.set_random_seed(seed)
    Xtr, ytr, Xva, yva = moons(n, noise, seed)
    model = K.Sequential([L.Input(shape=(2,))] + [L.Dense(h, activation=activation) for h in hidden] + [L.Dense(1, activation="sigmoid")])
    opt = {"adam": K.optimizers.Adam, "sgd": K.optimizers.SGD, "rmsprop": K.optimizers.RMSprop}[optimizer](learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    cbs = [K.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)] if early_stopping else []
    hist, log = capture(model.fit, Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=cbs)
    ev = model.evaluate(Xva, yva, verbose=0)
    pred = model.predict(Xva[:5], verbose=0)
    return {
        "summary": keras_summary(model),
        "log": log,
        "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()},
        "evaluate": [float(v) for v in ev],
        "predict": pred.round(4).tolist(),
        "predict_shape": list(pred.shape),
        "n_params": int(model.count_params()),
        "layers": [(l.name, l.__class__.__name__, str(l.output.shape), int(sum(int(np.prod(w.shape)) for w in l.weights))) for l in model.layers],
        "weights0": model.layers[0].get_weights()[0][:2].round(3).tolist(),
        "steps_per_epoch": int(np.ceil(len(Xtr) / batch_size)),
        "epochs_run": len(hist.history["loss"]),
    }


@st.cache_data(max_entries=32, show_spinner="يشغّل PyTorch…")
def torch_mlp_run(hidden: tuple, activation: str, optimizer: str, lr: float, epochs: int, batch_size: int, seed: int,
                  n: int = 600, noise: float = 0.25, zero_grad: bool = True) -> dict:
    """Explicit PyTorch training loop on the moons data with per-epoch logs."""
    T = torch(); nn = T.nn
    T.manual_seed(seed)
    Xtr, ytr, Xva, yva = moons(n, noise, seed)
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation]
    layers, d = [], 2
    for h in hidden:
        layers += [nn.Linear(d, h), act()]; d = h
    layers.append(nn.Linear(d, 1))
    model = nn.Sequential(*layers)
    opt = {"adam": T.optim.Adam, "sgd": T.optim.SGD, "rmsprop": T.optim.RMSprop}[optimizer](model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = T.utils.data.TensorDataset(T.tensor(Xtr), T.tensor(ytr).unsqueeze(1))
    loader = T.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=T.Generator().manual_seed(seed))
    Xv, yv = T.tensor(Xva), T.tensor(yva).unsqueeze(1)
    hist = {"loss": [], "val_loss": [], "val_accuracy": [], "grad_norm": []}
    lines = []
    for ep in range(1, epochs + 1):
        model.train(); tot, gn = 0.0, 0.0
        for xb, yb in loader:
            if zero_grad:
                opt.zero_grad()
            out = model(xb); loss = loss_fn(out, yb); loss.backward()
            gn += float(sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5)
            opt.step(); tot += loss.item() * len(xb)
        model.eval()
        with T.no_grad():
            vo = model(Xv); vl = loss_fn(vo, yv).item(); va = ((vo >= 0).float() == yv).float().mean().item()
        hist["loss"].append(tot / len(ds)); hist["val_loss"].append(vl); hist["val_accuracy"].append(va); hist["grad_norm"].append(gn / len(loader))
        lines.append(f"epoch {ep:3d}/{epochs} - loss: {tot / len(ds):.4f} - val_loss: {vl:.4f} - val_accuracy: {va:.4f}")
    with T.no_grad():
        pred = T.sigmoid(model(Xv[:5]))
    return {
        "model_repr": repr(model),
        "log": "\n".join(lines),
        "history": hist,
        "predict": pred.numpy().round(4).tolist(),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "param_shapes": [(name, list(p.shape)) for name, p in model.named_parameters()],
        "state_dict_keys": list(model.state_dict().keys()),
        "steps_per_epoch": len(loader),
    }


@st.cache_data(max_entries=16, show_spinner="يتتبع model.fit()…")
def keras_fit_trace(hidden: int, epochs: int, batch_size: int, lr: float, seed: int, n: int = 120) -> dict:
    """Trace what `fit()` does internally, batch by batch, with a Callback:
    batch loss/accuracy, a weight before/after each update, validation per epoch."""
    K = keras(); L = K.layers
    K.utils.set_random_seed(seed)
    Xtr, ytr, Xva, yva = moons(n + 100, 0.25, seed)
    Xtr, ytr = Xtr[:n], ytr[:n]
    model = K.Sequential([L.Input(shape=(2,)), L.Dense(hidden, activation="relu"), L.Dense(1, activation="sigmoid")])
    model.compile(optimizer=K.optimizers.SGD(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    events: list[dict] = []

    class Trace(K.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            events.append({"kind": "epoch_begin", "epoch": epoch + 1})

        def on_train_batch_begin(self, batch, logs=None):
            self._w = float(np.asarray(model.layers[0].kernel)[0, 0])

        def on_train_batch_end(self, batch, logs=None):
            events.append({"kind": "batch", "batch": batch + 1, "loss": float(logs["loss"]), "acc": float(logs["accuracy"]),
                           "w_before": self._w, "w_after": float(np.asarray(model.layers[0].kernel)[0, 0])})

        def on_epoch_end(self, epoch, logs=None):
            events.append({"kind": "epoch_end", "epoch": epoch + 1, "loss": float(logs["loss"]), "acc": float(logs["accuracy"]),
                           "val_loss": float(logs["val_loss"]), "val_acc": float(logs["val_accuracy"])})

    hist, log = capture(model.fit, Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[Trace()])
    return {"events": events, "log": log, "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()},
            "steps_per_epoch": int(np.ceil(n / batch_size)), "n": n, "n_val": len(Xva), "x_shape": [batch_size, 2]}


@st.cache_data(max_entries=64, show_spinner="يبني النموذج…")
def keras_summary_info(spec: tuple, input_shape: tuple) -> dict:
    """Build a Sequential model from a spec of (kind, arg) tuples and return
    its summary text plus per-layer facts for the visual deconstruction.
    kinds: dense(units, act) | dropout(rate) | bn | flatten"""
    K = keras(); L = K.layers
    K.utils.set_random_seed(0)
    layers = [L.Input(shape=tuple(input_shape))]
    for item in spec:
        kind = item[0]
        if kind == "dense":
            layers.append(L.Dense(int(item[1]), activation=item[2] if item[2] != "none" else None))
        elif kind == "dropout":
            layers.append(L.Dropout(float(item[1])))
        elif kind == "bn":
            layers.append(L.BatchNormalization())
        elif kind == "flatten":
            layers.append(L.Flatten())
    model = K.Sequential(layers)
    rows = []
    prev = tuple(input_shape)
    for l in model.layers:
        out = tuple(int(d) if d is not None else None for d in l.output.shape)
        inp = tuple(int(d) if d is not None else None for d in l.input.shape)
        trainable = int(sum(int(np.prod(w.shape)) for w in l.trainable_weights))
        non_trainable = int(sum(int(np.prod(w.shape)) for w in l.non_trainable_weights))
        weights = [(w.path.split("/")[-1], tuple(int(d) for d in w.shape)) for w in l.weights]
        rows.append({"name": l.name, "type": l.__class__.__name__, "input": inp, "output": out, "params": trainable + non_trainable,
                     "trainable": trainable, "non_trainable": non_trainable, "weights": weights,
                     "config": {k: v for k, v in l.get_config().items() if k in ("units", "activation", "rate", "momentum", "epsilon")}})
        prev = out
    return {"summary": keras_summary(model), "rows": rows, "total": int(model.count_params()),
            "trainable": int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
            "non_trainable": int(sum(int(np.prod(w.shape)) for w in model.non_trainable_weights))}


@st.cache_data(max_entries=8, show_spinner="يدرّب ويكتب سجلات TensorBoard…")
def keras_tensorboard_run(epochs: int, seed: int = 0) -> dict:
    """Train a small model with the TensorBoard callback + an LR schedule and
    keep what TensorBoard would show: scalars per epoch, weight histograms,
    the learning rate, and the event files written to disk."""
    import glob
    import os
    import tempfile

    K = keras(); L = K.layers
    K.utils.set_random_seed(seed)
    Xtr, ytr, Xva, yva = moons(600, 0.3, seed)
    model = K.Sequential([L.Input(shape=(2,)), L.Dense(32, activation="relu"), L.Dense(16, activation="relu"), L.Dense(1, activation="sigmoid")])
    model.compile(optimizer=K.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=["accuracy"])
    log_dir = os.path.join(tempfile.mkdtemp(prefix="dlia_tb_"), "run1")
    weights, lrs = [], []

    class Snap(K.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            weights.append(np.asarray(model.layers[0].kernel).ravel().tolist())
            lrs.append(float(model.optimizer.learning_rate))

    sched = K.callbacks.LearningRateScheduler(lambda ep, lr: 1e-2 * (0.85 ** ep))
    tb = K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    hist, _ = capture(model.fit, Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0, callbacks=[sched, tb, Snap()])
    files = sorted(os.path.relpath(f, log_dir).replace("\\", "/") for f in glob.glob(os.path.join(log_dir, "**", "*"), recursive=True) if os.path.isfile(f))
    return {"history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}, "weights": weights, "lr": lrs, "log_dir": log_dir, "files": files,
            "layers": [(l.name, l.__class__.__name__, str(tuple(int(d) if d else None for d in l.output.shape))) for l in model.layers]}


@st.cache_data(max_entries=16, show_spinner="يتتبع حلقة PyTorch…")
def torch_loop_trace(hidden: int, epochs: int, batch_size: int, lr: float, seed: int, n: int = 120, zero_grad: bool = True) -> dict:
    """Step-by-step trace of an explicit PyTorch training loop: for every
    batch the shapes, loss, gradient norm/sample and a weight before/after."""
    T = torch(); nn = T.nn
    T.manual_seed(seed)
    Xtr, ytr, Xva, yva = moons(n + 100, 0.25, seed)
    Xtr, ytr = Xtr[:n], ytr[:n]
    model = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = T.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = T.utils.data.TensorDataset(T.tensor(Xtr), T.tensor(ytr).unsqueeze(1))
    loader = T.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=T.Generator().manual_seed(seed))
    Xv, yv = T.tensor(Xva), T.tensor(yva).unsqueeze(1)
    events: list[dict] = []
    for ep in range(1, epochs + 1):
        model.train(); events.append({"kind": "epoch_begin", "epoch": ep, "mode": "train"})
        tot = 0.0
        for bi, (xb, yb) in enumerate(loader, start=1):
            g_before = None if model[0].weight.grad is None else float(model[0].weight.grad[0, 0])
            if zero_grad:
                opt.zero_grad()
            g_after_zero = None if model[0].weight.grad is None else float(model[0].weight.grad[0, 0])
            w_before = model[0].weight[0, 0].item()
            out = model(xb); loss = loss_fn(out, yb); loss.backward()
            gnorm = float(sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5); g00 = float(model[0].weight.grad[0, 0])
            opt.step(); tot += loss.item() * len(xb)
            events.append({"kind": "batch", "epoch": ep, "batch": bi, "x_shape": list(xb.shape), "y_shape": list(yb.shape), "out_shape": list(out.shape),
                           "logits": out[:3, 0].detach().numpy().round(3).tolist(), "loss": loss.item(), "gnorm": gnorm, "g00": g00, "g_before": g_before, "g_after_zero": g_after_zero,
                           "w_before": w_before, "w_after": model[0].weight[0, 0].item()})
        model.eval()
        with T.no_grad():
            vo = model(Xv); vl = loss_fn(vo, yv).item(); va = ((vo >= 0).float() == yv).float().mean().item()
        events.append({"kind": "epoch_end", "epoch": ep, "loss": tot / n, "val_loss": vl, "val_acc": va, "mode": "eval"})
    return {"events": events, "steps_per_epoch": len(loader), "n": n, "n_val": len(Xva), "model_repr": repr(model)}


@st.cache_data(max_entries=8, show_spinner="يجري تجربة zero_grad…")
def torch_zero_grad_experiment(steps: int = 6, lr: float = 0.1, seed: int = 0) -> dict:
    """Same model, same batches: once with optimizer.zero_grad(), once without.
    Records the gradient of one weight and the loss at every step."""
    T = torch(); nn = T.nn
    Xtr, ytr, _, _ = moons(400, 0.25, seed)      # 300 training rows
    xb_all = T.tensor(Xtr); yb_all = T.tensor(ytr).unsqueeze(1)
    out = {}
    for mode in ("with", "without"):
        T.manual_seed(seed)
        model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)); opt = T.optim.SGD(model.parameters(), lr=lr); loss_fn = nn.BCEWithLogitsLoss()
        rows = []
        for s in range(steps):
            idx = slice(s * 50 % 300, s * 50 % 300 + 50)
            xb, yb = xb_all[idx], yb_all[idx]
            if mode == "with":
                opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            fresh = None
            if mode == "without" and model[0].weight.grad is not None:
                g_prev = model[0].weight.grad.clone()      # accumulated so far (for the explanation)
            loss.backward()
            g = float(model[0].weight.grad[0, 0])
            if mode == "without" and s > 0:
                fresh = g - float(g_prev[0, 0])
            gnorm = float(model[0].weight.grad.norm())
            opt.step()
            rows.append({"step": s + 1, "loss": loss.item(), "g00": g, "fresh": fresh, "gnorm": gnorm})
        out[mode] = rows
    return out


@st.cache_data(max_entries=2, show_spinner="يجمع مخرجات حقيقية للمعرض…")
def gallery_outputs() -> dict:
    """Real outputs for the Visual Gallery (module 22), produced by the pinned
    versions: tensor reprs, summary, fit log, PyTorch prints, device info and
    typical error messages. Everything here is text captured from the libraries."""
    import io
    import contextlib

    out: dict = {}
    T = tf(); K = keras(); L = K.layers
    K.utils.set_random_seed(0)
    t = T.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out["tf_tensor"] = f">>> t = tf.constant([[1., 2., 3.], [4., 5., 6.]])\n>>> t\n{t!r}\n>>> t.shape, t.dtype, tf.rank(t).numpy()\n({tuple(t.shape)}, {t.dtype!r}, {int(T.rank(t))})"
    Xtr, ytr, Xva, yva = moons(600, 0.25, 0)
    model = K.Sequential([L.Input(shape=(2,)), L.Dense(16, activation="relu"), L.Dense(8, activation="relu"), L.Dense(1, activation="sigmoid")], name="moons_mlp")
    model.compile(optimizer=K.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=["accuracy"])
    out["keras_summary"] = keras_summary(model)
    hist, log2 = capture(model.fit, Xtr, ytr, validation_data=(Xva, yva), epochs=5, batch_size=32, verbose=2)
    out["keras_fit_log"] = log2
    out["keras_history"] = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    ev = model.evaluate(Xva, yva, verbose=0)
    out["keras_evaluate"] = f">>> model.evaluate(X_val, y_val, verbose=0)\n[{ev[0]:.4f}, {ev[1]:.4f}]\n>>> model.predict(X_val[:3], verbose=0)\n{model.predict(Xva[:3], verbose=0).round(3)!r}"
    out["tf_devices"] = f">>> tf.__version__\n'{T.__version__}'\n>>> tf.config.list_physical_devices('GPU')\n{T.config.list_physical_devices('GPU')!r}\n>>> tf.config.list_physical_devices('CPU')\n{T.config.list_physical_devices('CPU')!r}\n>>> t.device\n'{t.device}'"
    # keras shape error
    bad = K.Sequential([L.Input(shape=(5,)), L.Dense(1)]); bad.compile(optimizer="sgd", loss="mse")
    try:
        bad.fit(Xtr, ytr, epochs=1, verbose=0)
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\x1b[1m", "").replace("\x1b[0m", "")
        out["keras_shape_error"] = f"{type(e).__name__}: {msg.strip()}"
    # torch
    Tt = torch(); nn = Tt.nn
    Tt.manual_seed(0)
    pt = Tt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out["torch_tensor"] = f">>> t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])\n>>> t\n{pt!r}\n>>> t.shape, t.dtype, t.device, t.dim()\n({tuple(pt.shape)}, {pt.dtype}, {pt.device}, {pt.dim()})"
    pmodel = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    out["torch_model"] = f">>> model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))\n>>> print(model)\n{pmodel!r}\n>>> sum(p.numel() for p in model.parameters())\n{sum(p.numel() for p in pmodel.parameters())}"
    r = torch_mlp_run((16, 8), "relu", "adam", 0.01, 5, 32, 0)
    out["torch_log"] = r["log"]
    out["torch_devices"] = f">>> torch.__version__\n'{Tt.__version__}'\n>>> torch.cuda.is_available()\n{Tt.cuda.is_available()}\n>>> device = 'cuda' if torch.cuda.is_available() else 'cpu'; device\n'{'cuda' if Tt.cuda.is_available() else 'cpu'}'\n>>> t.to(device).device\n{pt.to('cpu').device!r}"
    try:
        pmodel(Tt.randn(4, 5))
    except Exception as e:  # noqa: BLE001
        out["torch_shape_error"] = f"Traceback (most recent call last):\n  File \"train.py\", line 12, in <module>\n    out = model(xb)\n  ...\n{type(e).__name__}: {e}"
    try:
        pmodel(Tt.randn(4, 2).double())
    except Exception as e:  # noqa: BLE001
        out["torch_dtype_error"] = f"{type(e).__name__}: {e}"
    try:
        Tt.ones(2).to("cuda") + Tt.ones(2)
    except Exception as e:  # noqa: BLE001
        out["torch_device_error"] = f"{type(e).__name__}: {e}"
    out["torch_device_error_gpu_machine"] = "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"
    return out


@st.cache_data(max_entries=4, show_spinner="يدرّب ويلتقط حالات قبل/أثناء/بعد…")
def before_during_after(epochs: int = 30, seed: int = 0) -> dict:
    """Decision-surface snapshots of the same Keras model before training,
    during (a few epochs) and after — plus loss/accuracy history and sample
    predictions at each stage."""
    K = keras(); L = K.layers
    K.utils.set_random_seed(seed)
    Xtr, ytr, Xva, yva = moons(500, 0.25, seed)
    g = np.linspace(-2, 3, 60).astype("float32"); GX, GY = np.meshgrid(g, g)
    grid = np.column_stack([GX.ravel(), GY.ravel()]).astype("float32")
    model = K.Sequential([L.Input(shape=(2,)), L.Dense(16, activation="relu"), L.Dense(16, activation="relu"), L.Dense(1, activation="sigmoid")])
    model.compile(optimizer=K.optimizers.Adam(5e-3), loss="binary_crossentropy", metrics=["accuracy"])
    snaps = {}

    def snap(name):
        P = model.predict(grid, verbose=0).reshape(GX.shape)
        pv = model.predict(Xva[:6], verbose=0).ravel()
        ev = model.evaluate(Xva, yva, verbose=0)
        snaps[name] = {"P": P.round(3).tolist(), "pred": pv.round(3).tolist(), "true": yva[:6].astype(int).tolist(), "val_loss": float(ev[0]), "val_acc": float(ev[1])}

    snap("before")
    mid = max(1, epochs // 6)

    class Snap(K.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch + 1 == mid:
                snap("during")

    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0, callbacks=[Snap()])
    snap("after")
    return {"grid": g.tolist(), "snaps": snaps, "mid_epoch": mid, "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()},
            "X": Xtr[:200].tolist(), "y": ytr[:200].astype(int).tolist()}


@st.cache_data(max_entries=4, show_spinner="يشغّل مشروع الأسبوع 07…")
def house_project(seed: int = 0, hidden: tuple = (32, 16), epochs: int = 300) -> dict:
    """Week 07 applied project: house-price regression end to end (prep,
    baselines, Keras MLP with early stopping, evaluation in target units,
    permutation importance, residuals). Returns plain data."""
    import pandas as pd
    from labs.datasets import house_prices

    K = keras(); L = K.layers
    K.utils.set_random_seed(seed)
    df = house_prices(n=300, seed=11)
    num = df[["area_m2", "rooms", "age_years"]].to_numpy("float32")
    cat = pd.get_dummies(df["district"]).to_numpy("float32"); cat_names = [f"district_{c}" for c in pd.get_dummies(df["district"]).columns]
    y = (df["price"].to_numpy("float32") / 1000.0)
    idx = np.random.default_rng(seed).permutation(len(df)); tr, va, te = idx[:180], idx[180:240], idx[240:]
    mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-8
    X = np.concatenate([(num - mu) / sd, cat], 1).astype("float32"); names = ["area_m2", "rooms", "age_years"] + cat_names
    ymu, ysd = y[tr].mean(), y[tr].std()
    yz = (y - ymu) / ysd
    # baselines
    base_mean = np.full(len(te), y[tr].mean())
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression().fit(X[tr], y[tr]); base_lin = lin.predict(X[te])
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))); mae = lambda a, b: float(np.mean(np.abs(a - b)))
    model = K.Sequential([L.Input(shape=(X.shape[1],))] + [L.Dense(h, activation="relu") for h in hidden] + [L.Dense(1)])
    model.compile(optimizer=K.optimizers.Adam(3e-3), loss="mse", metrics=["mae"])
    es = K.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    hist = model.fit(X[tr], yz[tr], validation_data=(X[va], yz[va]), epochs=epochs, batch_size=16, verbose=0, callbacks=[es])
    pred_te = model.predict(X[te], verbose=0).ravel() * ysd + ymu
    pred_tr = model.predict(X[tr], verbose=0).ravel() * ysd + ymu
    # permutation importance on validation
    rng = np.random.default_rng(seed)
    base_val = rmse(model.predict(X[va], verbose=0).ravel() * ysd + ymu, y[va])
    imp = []
    for j, nm in enumerate(names):
        Xp = X[va].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append((nm, rmse(model.predict(Xp, verbose=0).ravel() * ysd + ymu, y[va]) - base_val))
    resid = y[te] - pred_te
    worst = np.argsort(-np.abs(resid))[:5]
    return {
        "n": [len(tr), len(va), len(te)], "names": names, "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}, "epochs_run": len(hist.history["loss"]), "n_params": int(model.count_params()),
        "test": {"mean_rmse": rmse(base_mean, y[te]), "mean_mae": mae(base_mean, y[te]), "lin_rmse": rmse(base_lin, y[te]), "lin_mae": mae(base_lin, y[te]), "mlp_rmse": rmse(pred_te, y[te]), "mlp_mae": mae(pred_te, y[te]), "train_rmse": rmse(pred_tr, y[tr])},
        "lin_coef": dict(zip(names, lin.coef_.round(2).tolist())), "importance": imp, "y_te": y[te].tolist(), "pred_te": pred_te.tolist(), "resid": resid.tolist(),
        "worst": [{"idx": int(i), "true": float(y[te][i]), "pred": float(pred_te[i]), "area": float(df["area_m2"].to_numpy()[te][i]), "rooms": int(df["rooms"].to_numpy()[te][i]), "age": int(df["age_years"].to_numpy()[te][i]), "district": str(df["district"].to_numpy()[te][i])} for i in worst],
        "y_mean": float(y[tr].mean()),
    }


@st.cache_data(max_entries=8, show_spinner="يدرّب CNN صغيرة…")
def cnn_shapes_run(epochs: int = 8, augment: bool = False, dropout: float = 0.0, n_train: int = 300, filters: tuple = (8, 16), seed: int = 0) -> dict:
    """Week 09: small CNN on the synthetic bars/cross images. Returns history,
    test metrics, confusion matrix, first-layer kernels/feature maps for one
    image and the worst misclassifications."""
    from labs.cnn import CLASS_NAMES, shapes_dataset

    K = keras(); L = K.layers
    K.utils.set_random_seed(seed)
    X, y = shapes_dataset(n_train + 300, size=16, seed=seed, noise=0.2)
    Xtr, ytr, Xva, yva, Xte, yte = X[:n_train], y[:n_train], X[n_train:n_train + 150], y[n_train:n_train + 150], X[n_train + 150:], y[n_train + 150:]
    aug = [L.RandomTranslation(0.15, 0.15, fill_mode="constant"), L.RandomFlip("horizontal_and_vertical")] if augment else []
    model = K.Sequential([L.Input(shape=(16, 16, 1))] + aug + [L.Conv2D(filters[0], 3, padding="same", activation="relu", name="conv1"), L.MaxPooling2D(2), L.Conv2D(filters[1], 3, padding="same", activation="relu", name="conv2"), L.MaxPooling2D(2), L.Flatten()] + ([L.Dropout(dropout)] if dropout else []) + [L.Dense(32, activation="relu"), L.Dense(3, activation="softmax")])
    model.compile(optimizer=K.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0)
    ev = model.evaluate(Xte, yte, verbose=0)
    P = model.predict(Xte, verbose=0); pred = P.argmax(1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    conv1 = model.get_layer("conv1")
    fm_model = K.Model(model.inputs, conv1.output) if not augment else K.Model(model.inputs, conv1.output)
    img = Xte[:1]
    fmaps = fm_model.predict(img, verbose=0)[0]
    kernels = np.asarray(conv1.kernel)[:, :, 0, :]
    wrong = [i for i in range(len(yte)) if pred[i] != yte[i]][:6]
    return {"history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}, "test_loss": float(ev[0]), "test_acc": float(ev[1]), "cm": cm.tolist(), "classes": CLASS_NAMES,
            "n_params": int(model.count_params()), "summary": keras_summary(model), "image": img[0, :, :, 0].tolist(), "image_class": int(yte[0]), "fmaps": fmaps.transpose(2, 0, 1).tolist(), "kernels": kernels.transpose(2, 0, 1).tolist(),
            "wrong": [{"img": Xte[i, :, :, 0].tolist(), "true": int(yte[i]), "pred": int(pred[i]), "p": float(P[i].max())} for i in wrong], "train_acc": float(hist.history["accuracy"][-1]), "val_acc": float(hist.history["val_accuracy"][-1])}


@st.cache_data(max_entries=24, show_spinner="يدرّب نموذج تسلسل…")
def keras_seq_run(kind: str = "lstm", window: int = 12, units: int = 16, epochs: int = 40, lr: float = 5e-3, seed: int = 0, clipnorm: float | None = None, series: str = "inflation") -> dict:
    """Weeks 10–12: SimpleRNN / LSTM / GRU on the monthly-inflation series
    (chronological split, standardized with training stats). Returns history,
    RMSE in original units vs naive/mean baselines and test predictions."""
    import time

    from labs.datasets import monthly_inflation
    from labs.rnn import make_windows

    K = keras(); L = K.layers
    K.utils.set_random_seed(seed)
    if series == "inflation":
        s = monthly_inflation()["inflation"].to_numpy("float32"); n_tr = 120
    else:                                                    # "seasonal": صناعية بموسمية مركّبة وضوضاء قليلة (400 شهر)
        t = np.arange(400); rng = np.random.default_rng(seed)
        s = (3 + 0.8 * np.sin(2 * np.pi * t / 12) + 0.5 * np.sin(2 * np.pi * t / 5) + 0.002 * t + rng.normal(0, 0.08, 400)).astype("float32"); n_tr = 300
    mu, sd = s[:n_tr].mean(), s[:n_tr].std()
    z = (s - mu) / sd
    X, y = make_windows(z, window)
    k_tr = n_tr - window; k_va = k_tr + 24
    Xtr, ytr, Xva, yva, Xte, yte = X[:k_tr], y[:k_tr], X[k_tr:k_va], y[k_tr:k_va], X[k_va:], y[k_va:]
    if series != "inflation":
        k_va = k_tr + 40; Xva, yva, Xte, yte = X[k_tr:k_va], y[k_tr:k_va], X[k_va:], y[k_va:]
    layer = {"rnn": L.SimpleRNN, "lstm": L.LSTM, "gru": L.GRU}[kind]
    model = K.Sequential([L.Input(shape=(window, 1)), layer(units), L.Dense(1)])
    opt = K.optimizers.Adam(lr, clipnorm=clipnorm) if clipnorm else K.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss="mse")
    t0 = time.perf_counter()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=16, verbose=0, callbacks=[K.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    secs = time.perf_counter() - t0
    pred = model.predict(Xte, verbose=0).ravel() * sd + mu; true = yte * sd + mu
    naive = Xte[:, -1, 0] * sd + mu; mean_b = np.full_like(true, s[:n_tr].mean())
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    return {"history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}, "epochs_run": len(hist.history["loss"]), "n_params": int(model.count_params()), "seconds": round(secs, 1),
            "rmse": rmse(pred, true), "rmse_naive": rmse(naive, true), "rmse_mean": rmse(mean_b, true), "pred": pred.tolist(), "true": true.tolist(), "naive": naive.tolist(),
            "shapes": {"X_train": list(Xtr.shape), "y_train": list(ytr.shape), "X_test": list(Xte.shape)}, "summary": keras_summary(model), "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte))}
