"""CNN Shape Calculator — build a CNN layer by layer and trace (H, W, C) and
parameters through it; compare with Keras' own summary (spec §41)."""

import numpy as np
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.cnn import shape_trace

LAB = Lab(
    id="labs.cnn_shape_calculator",
    title_ar="حاسبة أشكال CNN",
    title_en="CNN Shape Calculator",
    category="cnn",
    description_ar="ابنِ شبكة التفافية طبقةً طبقة (Conv2D بحشو وخطوة، MaxPooling، Flatten، Dense) وشاهد شكل (H, W, C) وعدد المعلمات عند كل طبقة — ثم تحقق من أن Keras يطبع نفس الأرقام.",
    related_lessons=["course.w08.layers_shapes", "course.w09.shape_debugging"],
)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2 = st.columns(2)
    with c1:
        hw = st.select_slider("حجم الصورة (H = W)", options=[16, 28, 32, 64, 128, 224], value=28, key="sc_hw")
    with c2:
        cin = st.select_slider("القنوات (1 رمادي / 3 RGB)", options=[1, 3], value=1, key="sc_cin")
    st.markdown("**الطبقات** (فعّل/عطّل واضبط):")
    layers = []
    for b in range(1, 4):
        cols = st.columns([1, 1, 1, 1, 1])
        with cols[0]:
            on = st.checkbox(f"كتلة {b}: Conv", value=b <= 2, key=f"sc_on{b}")
        with cols[1]:
            f = st.select_slider("filters", options=[8, 16, 32, 64, 128], value=[16, 32, 64][b - 1], key=f"sc_f{b}")
        with cols[2]:
            k = st.select_slider("k", options=[3, 5, 7], value=3, key=f"sc_k{b}")
        with cols[3]:
            pad = st.selectbox("padding", ["same", "valid"], key=f"sc_p{b}")
        with cols[4]:
            pool = st.checkbox("MaxPool 2", value=True, key=f"sc_pool{b}")
        if on:
            layers.append({"type": "conv", "k": int(k), "filters": int(f), "padding": pad, "stride": 1})
            if pool:
                layers.append({"type": "pool", "size": 2})
    cd1, cd2 = st.columns(2)
    with cd1:
        dense = st.select_slider("Dense مخفية (0 = بدون)", options=[0, 32, 64, 128, 256], value=64, key="sc_dense")
    with cd2:
        out = st.select_slider("مخرجات", options=[1, 3, 10, 100], value=10, key="sc_out")
    layers.append({"type": "flatten"})
    if dense:
        layers.append({"type": "dense", "units": int(dense)})
    layers.append({"type": "dense", "units": int(out)})
    rows = shape_trace(int(hw), int(cin), layers)
    total = sum(r["params"] for r in rows)
    table(["الطبقة", "شكل المخرج (H, W, C)", "المعلمات", "كيف حُسبت"],
          [(r["layer"], r["shape"], f"{r['params']:,}", _how(r, prev)) for r, prev in zip(rows, [None] + rows[:-1])], ["code", "code", "num", "rtl"])
    st.markdown(f"**المجموع: {total:,} معلمة** — منها في Dense بعد Flatten: {sum(r['params'] for r in rows if r['layer'].startswith('Dense')):,} ({100 * sum(r['params'] for r in rows if r['layer'].startswith('Dense')) / max(total, 1):.0f}%).")
    if st.button("تحقق بـ Keras (يبني النموذج ويطبع summary)", icon=":material/verified:", key="sc_verify"):
        from labs.fw import keras, keras_summary
        K = keras(); L = K.layers
        ks = [L.Input(shape=(int(hw), int(hw), int(cin)))]
        for Ly in layers:
            if Ly["type"] == "conv":
                ks.append(L.Conv2D(Ly["filters"], Ly["k"], padding=Ly["padding"], activation="relu"))
            elif Ly["type"] == "pool":
                ks.append(L.MaxPooling2D(2))
            elif Ly["type"] == "flatten":
                ks.append(L.Flatten())
            else:
                ks.append(L.Dense(Ly["units"]))
        m = K.Sequential(ks)
        st.code(keras_summary(m).strip(), language="text")
        st.success(f"Keras: {m.count_params():,} معلمة — الحاسبة: {total:,} ✅" if m.count_params() == total else f"اختلاف! Keras {m.count_params():,} مقابل {total:,}", icon="✅" if m.count_params() == total else "❌")
    intuition("لاحظ أين تتركز المعلمات: طبقات Conv رخيصة (k·k·C_in·filters) لأنها تشارك الأوزان عبر الصورة، بينما أول Dense بعد Flatten تبتلع معظم المعلمات (H·W·C × units). لهذا تُقلَّص الصورة بالتجميع قبل Flatten — أو تُستبدل Flatten بـ GlobalAveragePooling.")
    practical_note("قاعدة الأشكال: Conv same يحفظ H وW ويغيّر C إلى filters؛ valid ينقص k−1؛ MaxPool 2 يقسم H وW على 2؛ Flatten يعطي H·W·C؛ Dense تعطي units. المعلمات: Conv = k²·C_in·C_out + C_out؛ Dense = in·out + out.")


def _how(r: dict, prev: dict | None) -> str:
    L = r["layer"]
    if L == "Input":
        return "شكل الصورة الواحدة (بلا بُعد الدفعة)"
    if L.startswith("Conv2D"):
        return "H,W بحسب same/valid؛ C = filters؛ المعلمات = k²·C_in·filters + filters"
    if L.startswith("MaxPooling"):
        return "H/2, W/2؛ C ثابت؛ لا معلمات"
    if L == "Flatten":
        return "H·W·C في متجه واحد؛ لا معلمات"
    return "in×units + units؛ in = طول المتجه السابق"
