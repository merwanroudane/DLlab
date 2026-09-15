"""GPU / Batch-Memory Concept Lab — estimate the memory a training step needs
(parameters, gradients, optimizer state, activations × batch) and see how
compute time scales with matrix size on this CPU (spec §41)."""

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.cnn import shape_trace

LAB = Lab(
    id="labs.gpu_batch_memory_lab",
    title_ar="معمل مفهوم GPU وذاكرة الدفعة",
    title_en="GPU / Batch Memory Concept Lab",
    category="frameworks",
    description_ar="قدّر ذاكرة خطوة تدريب واحدة (المعلمات + التدرجات + حالة المحسّن + التنشيطات × حجم الدفعة) لشبكة CNN تختارها، وشاهد كيف ينمو زمن ضرب المصفوفات مع الحجم على CPU — ولماذا يهم GPU.",
    related_lessons=["course.w13.cpu_gpu_memory", "course.w13.colab_runtime"],
)

BYTES = {"float32": 4, "float16": 2}


@st.cache_data(max_entries=4, show_spinner=False)
def _matmul_times(sizes: tuple) -> list[float]:
    out = []
    for n in sizes:
        a = np.random.default_rng(0).normal(size=(n, n)).astype("float32"); b = a.copy()
        t = time.perf_counter(); a @ b; out.append(time.perf_counter() - t)
    return out


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    st.markdown("### 1) ذاكرة خطوة التدريب")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hw = st.select_slider("حجم الصورة", options=[32, 64, 128, 224], value=128, key="gm_hw")
    with c2:
        filters = st.select_slider("filters الأساسية", options=[16, 32, 64], value=32, key="gm_f")
    with c3:
        batch = st.select_slider("batch_size", options=[8, 16, 32, 64, 128, 256], value=64, key="gm_bs")
    with c4:
        dtype = st.selectbox("الدقة", ["float32", "float16"], key="gm_dtype")
    layers = [{"type": "conv", "k": 3, "filters": filters, "padding": "same"}, {"type": "pool", "size": 2}, {"type": "conv", "k": 3, "filters": filters * 2, "padding": "same"}, {"type": "pool", "size": 2}, {"type": "conv", "k": 3, "filters": filters * 4, "padding": "same"}, {"type": "pool", "size": 2}, {"type": "flatten"}, {"type": "dense", "units": 256}, {"type": "dense", "units": 10}]
    rows = shape_trace(int(hw), 3, layers)
    params = sum(r["params"] for r in rows)
    acts = 0
    for r in rows:
        shp = [int(v) for v in r["shape"].strip("()").split(",") if v.strip()]
        acts += int(np.prod(shp))
    b = BYTES[dtype]
    mem_params = params * 4; mem_grads = params * 4; mem_adam = params * 8; mem_acts = acts * b * int(batch) * 2   # ×2: تنشيطات + تدرجاتها تقريبًا
    total = mem_params + mem_grads + mem_adam + mem_acts
    mb = lambda x: f"{x / 1e6:,.0f} MB"
    table(["المكوّن", "الحساب", "الذاكرة", "يعتمد على الدفعة؟"],
          [("المعلمات", f"{params:,} × 4 B", mb(mem_params), "لا"), ("التدرجات", f"{params:,} × 4 B", mb(mem_grads), "لا"), ("حالة Adam (m, v)", f"{params:,} × 8 B", mb(mem_adam), "لا"), ("التنشيطات (+ تدرجاتها)", f"{acts:,} قيمة × {b} B × {batch} × 2", mb(mem_acts), "**نعم — خطيًا**"), ("**المجموع التقريبي**", "", f"**{mb(total)}**", "")],
          ["rtl", "code", "num", "rtl"])
    fig = go.Figure()
    bs_axis = [8, 16, 32, 64, 128, 256, 512]
    fig.add_trace(go.Scatter(x=bs_axis, y=[(mem_params + mem_grads + mem_adam + acts * b * bb * 2) / 1e9 for bb in bs_axis], name="GB needed", line=dict(color="#C8473A", width=3), mode="lines+markers"))
    for vram, color in ((4, "#B9B2A6"), (16, "#1F7A78"), (24, "#7C5CBF")):
        fig.add_hline(y=vram, line=dict(color=color, dash="dot"), annotation_text=f"{vram} GB GPU")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="batch_size", yaxis_title="GB", xaxis_type="log", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="gm_fig")
    intuition("المعلمات والتدرجات وحالة المحسّن ثابتة؛ **التنشيطات** تنمو خطيًا مع الدفعة وتربيعيًا مع حجم الصورة. عند تجاوز خط VRAM ترى `ResourceExhaustedError: OOM`. العلاج الأول: batch_size ÷ 2؛ الثاني: float16 (mixed precision)؛ الثالث: صورة أصغر أو شبكة أصغر.")
    st.markdown("### 2) لماذا GPU؟ زمن ضرب المصفوفات على هذا الـ CPU")
    sizes = (128, 256, 512, 1024, 2048)
    times = _matmul_times(sizes)
    table(["n (مصفوفة n×n)", "عمليات (≈2n³)", "الزمن (ms)", "GFLOP/s"], [(str(n), f"{2 * n ** 3 / 1e9:.2f} G", f"{t * 1000:.1f}", f"{2 * n ** 3 / t / 1e9:.1f}") for n, t in zip(sizes, times)], ["num", "num", "num", "num"])
    practical_note("الزمن ينمو ≈ ×8 مع مضاعفة n (تكعيبي). CPU ينجز عشرات GFLOP/s؛ GPU حديث آلافًا — لأن ضرب المصفوفات «متوازٍ بإحراج»: كل عنصر مخرج مستقل، وGPU يملك آلاف الأنوية الصغيرة. الشبكة = آلاف من هذه العمليات لكل دفعة، لذلك يتحول التدريب من ساعات إلى دقائق. الأسبوع 13 يريك كيف تحصل على GPU في Colab.")
