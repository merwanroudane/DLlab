"""Padding / Stride / Pooling Lab — see how each hyper-parameter changes the
output size and the feature map on a real image patch (spec §41)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.cnn import KERNELS, conv2d, out_size, pool2d, shapes_dataset

LAB = Lab(
    id="labs.padding_stride_lab",
    title_ar="معمل الحشو والخطوة والتجميع",
    title_en="Padding / Stride / Pooling Lab",
    category="cnn",
    description_ar="غيّر حجم النواة والحشو والخطوة وشاهد أثرها على حجم خريطة الخصائص بالصيغة وبالصورة، ثم طبّق تجميعًا أقصى/متوسطًا وقارن.",
    related_lessons=["course.w08.padding_stride_pooling", "course.w08.layers_shapes"],
)


def _heat(z, title, key):
    fig = go.Figure(go.Heatmap(z=z[::-1], colorscale=[[0, "#E6F1FB"], [0.5, "#FFFDF9"], [1, "#C8473A"]], showscale=False, zmid=0))
    fig.update_layout(height=230, title=dict(text=title, font=dict(size=12), x=0.5), margin=dict(l=5, r=5, t=30, b=5), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key=key)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    X, y = shapes_dataset(12, size=16, seed=3, noise=0.05)
    c0, c1, c2, c3, c4 = st.columns(5)
    with c0:
        which = st.selectbox("صورة (16×16)", list(range(12)), format_func=lambda i: f"#{i} — {['أفقي', 'عمودي', 'صليب'][y[i]]}", key="ps_img")
    with c1:
        k = st.select_slider("حجم النواة k", options=[1, 3, 5, 7], value=3, key="ps_k")
    with c2:
        p = st.select_slider("الحشو p", options=[0, 1, 2, 3], value=0, key="ps_p")
    with c3:
        s = st.select_slider("الخطوة s", options=[1, 2, 3], value=1, key="ps_s")
    with c4:
        kname = st.selectbox("النواة", ["vertical edge", "horizontal edge", "blur (box)", "sharpen"], key="ps_kn")
    img = X[which, :, :, 0]
    base = KERNELS[kname]
    if k == 1:
        kernel = base[1:2, 1:2] if base[1, 1] != 0 else np.ones((1, 1))
    else:
        kernel = np.zeros((k, k)); c = k // 2; kernel[c - 1:c + 2, c - 1:c + 2] = base     # النواة 3×3 في مركز نواة k×k
    n = img.shape[0]; o = out_size(n, k, p, s)
    st.code(f"output = floor((n + 2p − k)/s) + 1 = floor(({n} + 2·{p} − {k})/{s}) + 1 = {o}   → {o}×{o}" + ("   (same: p = (k−1)/2 keeps 16×16 when s = 1)" if p == (k - 1) // 2 and s == 1 else ""), language="text")
    fmap = conv2d(img, kernel, padding=p, stride=s)
    cA, cB, cC = st.columns(3)
    with cA:
        _heat(img, f"input {n}×{n}", "ps_in")
    with cB:
        _heat(fmap, f"conv → {fmap.shape[0]}×{fmap.shape[1]}", "ps_conv")
    with cC:
        mode = st.radio("التجميع", ["max", "avg"], horizontal=True, key="ps_mode")
        size = st.select_slider("نافذة التجميع", options=[2, 3, 4], value=2, key="ps_pool")
        if fmap.shape[0] >= size:
            pooled = pool2d(fmap, size, mode=mode)
            _heat(pooled, f"{mode} pool {size}×{size} → {pooled.shape[0]}×{pooled.shape[1]}", "ps_pool_fig")
        else:
            st.caption("خريطة الخصائص أصغر من نافذة التجميع.")
    rows = []
    for kk in (3, 5):
        for pp in (0, (kk - 1) // 2):
            for ss in (1, 2):
                rows.append((str(kk), str(pp), str(ss), f"{out_size(16, kk, pp, ss)}×{out_size(16, kk, pp, ss)}", "same" if pp == (kk - 1) // 2 and ss == 1 else ("valid" if pp == 0 else "—")))
    st.markdown("### جدول مرجعي لصورة 16×16")
    table(["k", "p", "s", "المخرج", "الاسم في Keras"], rows, ["num", "num", "num", "code", "code"])
    intuition("الحشو يحفظ الحجم (وحواف الصورة)؛ الخطوة تقلّص الحجم بمعامل s (وتقلّل الحساب)؛ التجميع يقلّص بمعامل النافذة ويجعل الكشف أقل حساسية للإزاحة الصغيرة. الثلاثة معًا تحدد كيف تنكمش الصورة عبر الشبكة — وهذا موضوع حاسبة الأشكال.")
    practical_note("في Keras: `Conv2D(filters, 3, padding='same')` تحسب p تلقائيًا؛ `padding='valid'` = بلا حشو؛ `strides=2` بدل التجميع في بعض البنى الحديثة. `MaxPooling2D(2)` = نافذة 2 وخطوة 2.")
