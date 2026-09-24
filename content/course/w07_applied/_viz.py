"""Week 07 frames: one house through data preparation, permutation importance
column by column, and split-to-split variability of the model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import svgkit as K
from labs.datasets import house_prices


@st.cache_data(ttl=3600, max_entries=2)
def prep_example(seed: int = 0) -> dict:
    """Follow one training house through the exact preparation used by labs.fw.house_project."""
    df = house_prices(n=300, seed=11)
    num = df[["area_m2", "rooms", "age_years"]].to_numpy("float32")
    dummies = pd.get_dummies(df["district"])
    y = df["price"].to_numpy("float32") / 1000.0
    idx = np.random.default_rng(seed).permutation(len(df)); tr = idx[:180]
    mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-8
    i = int(tr[0])
    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    return {"raw": {"area_m2": float(num[i, 0]), "rooms": int(num[i, 1]), "age_years": int(num[i, 2]), "district": str(df["district"].iloc[i]), "price_k": float(y[i])},
            "mu": mu.tolist(), "sd": sd.tolist(), "z": ((num[i] - mu) / sd).tolist(), "onehot": dummies.iloc[i].astype(int).to_dict(),
            "ymu": ymu, "ysd": ysd, "yz": (float(y[i]) - ymu) / ysd}


def prep_svg(d: dict, stage: int) -> str:
    raw = d["raw"]
    s = K.svg_open(700, 260)
    s += K.text(350, 22, "one training house through the pipeline", size=12, bold=True, color=K.VIOLET)
    cols = [("area_m2", K.BLUE), ("rooms", K.VIOLET), ("age_years", K.AMBER)]
    for j, (name, col) in enumerate(cols):
        x = 20 + j * 110
        s += K.box(x, 45, 100, 44, f"{raw[name]:g}", col, size=13, sub=name)
        if stage >= 1:
            s += K.arrow(x + 50, 91, x + 50, 118, color=col)
            s += K.box(x, 120, 100, 44, f"{d['z'][j]:+.2f}", col, filled=stage == 1, size=13, sub=f"(x−{d['mu'][j]:.1f})/{d['sd'][j]:.1f}")
    s += K.box(360, 45, 120, 44, raw["district"], K.EMERALD, size=13, sub="district")
    if stage >= 2:
        for k, (name, v) in enumerate(d["onehot"].items()):
            x = 350 + k * 85
            s += K.box(x, 120, 78, 44, str(v), K.EMERALD, filled=v == 1, size=13, sub=name[:9])
        s += K.arrow(420, 91, 420, 118, color=K.EMERALD)
    s += K.box(560, 45, 120, 44, f"{raw['price_k']:.1f}", K.PINK, size=13, sub="price (k DZD)")
    if stage >= 3:
        s += K.arrow(620, 91, 620, 190, color=K.PINK)
        s += K.box(560, 192, 120, 44, f"{d['yz']:+.3f}", K.PINK, filled=True, size=13, sub="standardized y")
    if stage >= 4:
        vec = ", ".join(f"{v:+.2f}" for v in d["z"]) + ", " + ", ".join(str(v) for v in d["onehot"].values())
        s += K.text(270, 215, f"x = [{vec}]", size=12, mono=True, bold=True, color=K.INK)
        s += K.text(270, 238, "7 numbers → Input(shape=(7,))", size=11, color=K.MUTED)
    return s + "</svg>"


def importance_svg(imp: list[tuple[str, float]], k: int) -> str:
    """Bars revealed one feature at a time (in the model's feature order)."""
    s = K.svg_open(700, 250)
    mx = max(max(v for _, v in imp), 1e-6) * 1.15
    P = K.Panel(160, 30, 480, 190, (0, mx, 0, len(imp)))
    s += P.frame("Δ RMSE (k DZD) when one column is shuffled — validation set")
    for i, (name, v) in enumerate(imp):
        y = 30 + i * (190 / len(imp)) + 4
        h = 190 / len(imp) - 8
        on = i <= k
        s += K.text(150, y + h / 2 + 4, name, size=11, anchor="end", color=K.INK if on else "#CBD5E1", mono=True)
        if on:
            col = K.PALETTE[i % 7]
            s += f'<rect x="{P.px(0):.1f}" y="{y:.1f}" width="{max(1.5, P.px(max(0, v)) - P.px(0)):.1f}" height="{h:.1f}" rx="4" fill="{col}" fill-opacity="{1 if i == k else 0.55}"/>'
            s += K.text(P.px(max(0, v)) + 6, y + h / 2 + 4, f"{v:+.1f}", size=11, color=col, anchor="start", bold=i == k)
    return s + "</svg>"


@st.cache_data(ttl=3600, max_entries=2)
def split_variability(n_splits: int = 10) -> dict:
    """Same data, ten different random 240/60 train/test splits: how much do test errors move?
    (scikit-learn MLP as a fast stand-in for the Keras model — same architecture 32-16.)"""
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor

    df = house_prices(n=300, seed=11)
    num = df[["area_m2", "rooms", "age_years"]].to_numpy(float)
    cat = pd.get_dummies(df["district"]).to_numpy(float)
    y = df["price"].to_numpy(float) / 1000.0
    rows = []
    for s in range(n_splits):
        idx = np.random.default_rng(s).permutation(len(df)); tr, te = idx[:240], idx[240:]
        mu, sd = num[tr].mean(0), num[tr].std(0)
        X = np.concatenate([(num - mu) / sd, cat], 1)
        lin = LinearRegression().fit(X[tr], y[tr])
        ymu, ysd = y[tr].mean(), y[tr].std()
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=3000, early_stopping=True, validation_fraction=0.25,
                           n_iter_no_change=25, random_state=s).fit(X[tr], (y[tr] - ymu) / ysd)
        mae_lin = float(np.mean(np.abs(lin.predict(X[te]) - y[te])))
        mae_mlp = float(np.mean(np.abs(mlp.predict(X[te]) * ysd + ymu - y[te])))
        rows.append({"split": s, "lin": mae_lin, "mlp": mae_mlp})
    return {"rows": rows}
