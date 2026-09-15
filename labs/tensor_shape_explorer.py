"""Tensor Shape Explorer (spec §28 Module 4 / §41)."""

import numpy as np
import streamlit as st

from components.callouts import debugging_note, intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.tensor_shape_explorer",
    title_ar="مستكشف أشكال الموترات",
    title_en="Tensor Shape Explorer",
    category="math",
    description_ar="ابنِ موترًا برتبة وأطوال من اختيارك، ثم جرّب reshape وtranspose وexpand/squeeze والتجميع على محور، وشاهد الشكل والذاكرة قبل/بعد.",
    related_lessons=["foundations.linalg.tensors", "foundations.data.shape_axis_rank"],
)

PRESETS = {
    "جدولية (n, d)": (32, 4),
    "سلاسل زمنية (n, T, f)": (16, 30, 3),
    "صور رمادية (n, H, W, 1)": (8, 28, 28, 1),
    "صور ملونة Keras (n, H, W, C)": (8, 32, 32, 3),
    "صور ملونة PyTorch (n, C, H, W)": (8, 3, 32, 32),
    "مخصص": (4, 3),
}


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    intuition("قاعدة ذهبية: **عدد العناصر لا يتغير** بأي عملية شكل. `reshape` يعيد التقسيم، `transpose` يعيد الترتيب، `expand/squeeze` يضيف/يزيل محاور بطول 1.")

    preset = st.selectbox("قالب", list(PRESETS), key="tse_preset")
    base = PRESETS[preset]
    rank = st.slider("الرتبة (عدد المحاور)", 0, 5, len(base), key="tse_rank") if preset == "مخصص" else len(base)
    dims = []
    if rank:
        cols = st.columns(rank)
        for i in range(rank):
            default = base[i] if i < len(base) else 2
            with cols[i]:
                dims.append(st.number_input(f"axis {i}", 1, 512, int(default), key=f"tse_dim_{i}"))
    shape = tuple(int(d) for d in dims)
    dtype = st.selectbox("dtype", ["float32", "float64", "int64", "uint8"], key="tse_dtype")
    t = np.zeros(shape, dtype=dtype)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ndim (rank)", t.ndim); c2.metric("size", f"{t.size:,}"); c3.metric("itemsize", f"{t.itemsize} B"); c4.metric("memory", f"{t.nbytes / 1024:.1f} KB")
    st.code(f"t = np.zeros({shape}, dtype=np.{dtype})\nt.shape = {t.shape}\nt.ndim  = {t.ndim}\nt.size  = {t.size} = {' × '.join(map(str, shape)) if shape else '1'}\nt.nbytes = {t.nbytes} bytes", language="text")

    if rank >= 1:
        st.markdown("### عمليات على الشكل")
        op = st.segmented_control("العملية", ["reshape", "transpose", "expand_dims", "squeeze", "aggregate (axis)"], default="reshape", key="tse_op")
        try:
            if op == "reshape":
                txt = st.text_input("الشكل الجديد (فواصل، -1 مسموح)", value=f"{shape[0]}, -1" if rank > 1 else "-1", key="tse_reshape")
                new = tuple(int(x) for x in txt.split(",") if x.strip())
                r = t.reshape(new)
                st.code(f"t.reshape{new} → {r.shape}   (size {t.size} → {r.size} ✔)", language="text")
                st.markdown("العناصر تُقرأ بالترتيب وتُصبّ في الشكل الجديد. `Flatten` = `reshape(n, -1)`.")
            elif op == "transpose":
                txt = st.text_input("ترتيب المحاور الجديد", value=", ".join(str(i) for i in reversed(range(rank))), key="tse_perm")
                perm = tuple(int(x) for x in txt.split(","))
                r = np.transpose(t, perm)
                st.code(f"np.transpose(t, {perm}) → {r.shape}", language="text")
                st.markdown("كل محور ينتقل إلى موضعه الجديد؛ البيانات يُعاد ترتيبها فعليًا. مثال Keras→PyTorch للصور: `(0, 3, 1, 2)`.")
            elif op == "expand_dims":
                ax = st.slider("axis", 0, rank, 0, key="tse_exp")
                r = np.expand_dims(t, ax)
                st.code(f"np.expand_dims(t, axis={ax}) → {r.shape}", language="text")
                st.markdown("يضيف محورًا بطول 1. `axis=0` يضيف بُعد الدفعة لملاحظة واحدة.")
            elif op == "squeeze":
                r = np.squeeze(t)
                st.code(f"np.squeeze(t) → {r.shape}", language="text")
                st.markdown("يزيل كل المحاور ذات الطول 1. احذر: يزيل بُعد الدفعة أيضًا إن كان 1.")
            else:
                ax = st.slider("axis", 0, rank - 1, 0, key="tse_agg")
                r = t.mean(axis=ax)
                st.code(f"t.mean(axis={ax}) → {r.shape}", language="text")
                st.markdown("المحور المذكور يختفي؛ الباقي يبقى.")
        except Exception as e:  # noqa: BLE001 - show the real error to the learner
            st.error(f"{type(e).__name__}: {e}", icon="❌")
            debugging_note("هذه الرسالة نفسها ستظهر في كود حقيقي. اقرأ الأرقام فيها: غالبًا حاصل ضرب الشكل الجديد لا يساوي عدد العناصر.")

    st.markdown("### مرجع سريع")
    table(["البيانات", "Keras", "PyTorch", "المحور 0", "المعنى"],
          [("جدولية", "(n, d)", "(n, d)", "الدفعة", "d خصائص"), ("سلاسل", "(n, T, f)", "(n, T, f)", "الدفعة", "T خطوات × f متغيرات"),
           ("صور", "(n, H, W, C)", "(n, C, H, W)", "الدفعة", "القناة آخرًا / ثانيًا"), ("نص مرمّز", "(n, T)", "(n, T)", "الدفعة", "T رموز")],
          ["rtl", "code", "code", "rtl", "rtl"])
    practical_note("الذاكرة = size × itemsize. دفعة 256 صورة 224×224×3 بـ float32 ≈ 154 MB قبل أي حساب — سبب أخطاء OOM على GPU.")
