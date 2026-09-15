"""PyTorch Tensor Explorer — create, inspect (shape / dtype / device), reshape,
index and (when available) move between CPU and GPU (spec §21.13)."""

import numpy as np
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.fw import torch

LAB = Lab(
    id="labs.torch_tensor_explorer",
    title_ar="مستكشف موترات PyTorch",
    title_en="PyTorch Tensor Explorer",
    category="frameworks",
    description_ar="أنشئ torch.Tensor بأي شكل وdtype، افحص shape وdtype وdevice وrequires_grad، جرّب view/reshape والفهرسة، وانقله بين CPU وGPU إن توفّر — وقارن بنفس المفاهيم في TensorFlow.",
    related_lessons=["foundations.frameworks.pytorch.tensors", "foundations.frameworks.tensorflow.tensors", "foundations.data.shape_axis_rank"],
)

DTYPES = ["float32", "float16", "float64", "int32", "int64", "bool"]


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    T = torch()
    rank = st.slider("الرتبة", 0, 4, 2, key="ptx_rank")
    names = ["batch", "rows / time", "cols / features", "channels"]
    dims = []
    cols = st.columns(max(rank, 1))
    for i in range(rank):
        with cols[i]:
            dims.append(int(st.number_input(f"محور {i} ({names[i]})", 1, 64, [4, 3, 2, 2][i], key=f"ptx_d{i}")))
    c1, c2, c3 = st.columns(3)
    with c1:
        dtype = st.selectbox("dtype", DTYPES, index=0, key="ptx_dtype")
    with c2:
        fill = st.selectbox("القيم", ["arange", "zeros", "ones", "randn", "randint 0–9"], key="ptx_fill")
    with c3:
        rg = st.toggle("requires_grad", value=False, key="ptx_rg", disabled=not dtype.startswith("float"))
    n = int(np.prod(dims)) if dims else 1
    T.manual_seed(0)
    if fill == "arange":
        x = T.arange(n, dtype=T.float32).reshape(dims)
    elif fill == "zeros":
        x = T.zeros(dims)
    elif fill == "ones":
        x = T.ones(dims)
    elif fill == "randn":
        x = T.randn(dims)
    else:
        x = T.randint(0, 10, dims).float()
    x = x.to(getattr(T, dtype))
    if rg and dtype.startswith("float"):
        x.requires_grad_(True)
    st.markdown("### الكود المكافئ")
    st.code(f"x = torch.{fill.split()[0]}(...).to(torch.{dtype})" + ("\nx.requires_grad_(True)" if rg and dtype.startswith("float") else ""), language="python")
    size = x.numel(); itemsize = x.element_size()
    table(["الخاصية", "الاستدعاء", "القيمة", "المقابل في TensorFlow"],
          [("الشكل", "x.shape / x.size()", str(tuple(x.shape)), "x.shape"), ("الرتبة", "x.dim() / x.ndim", str(x.dim()), "tf.rank(x)"), ("dtype", "x.dtype", str(x.dtype).replace("torch.", ""), "x.dtype"),
           ("عدد العناصر", "x.numel()", str(size), "tf.size(x)"), ("الذاكرة", "numel × element_size", f"{size * itemsize} B", "size × dtype.size"), ("الجهاز", "x.device", str(x.device), "x.device"),
           ("يتتبعه Autograd؟", "x.requires_grad", str(x.requires_grad), "tf.Variable / tape.watch")],
          ["rtl", "code", "code", "code"])
    with st.expander("repr(x)", icon=":material/data_object:"):
        st.code(repr(x)[:1500], language="text")
        st.markdown("PyTorch يطبع `tensor([...])` ثم يذكر dtype فقط إن لم يكن الافتراضي (float32/int64)، وdevice إن لم يكن CPU، و`requires_grad=True` إن كان مفعّلًا.")
    st.markdown("### جرّب: view/reshape، الفهرسة، النقل بين الأجهزة")
    c4, c5, c6 = st.columns(3)
    with c4:
        new_shape = st.text_input("view إلى (مثل -1,2)", value=f"-1,{dims[-1]}" if dims else "1", key="ptx_view")
        try:
            shp = [int(s) for s in new_shape.replace(" ", "").split(",") if s]
            y = x.view(*shp)
            st.code(f"x.view({shp}).shape -> {tuple(y.shape)}\nshares memory with x: {y.data_ptr() == x.data_ptr()}", language="text")
        except Exception as e:  # noqa: BLE001
            st.code(f"{type(e).__name__}: {str(e).splitlines()[0][:150]}", language="text")
            st.caption("view يحتاج نفس عدد العناصر (وذاكرة متصلة؛ وإلا reshape).")
    with c5:
        idx = st.text_input("فهرسة (مثل 0 أو :,0 أو ...,-1)", value="0" if rank else "...", key="ptx_idx")
        try:
            z = eval(f"x[{idx}]", {"x": x})  # noqa: S307 - index expression on a local tensor only
            st.code(f"x[{idx}].shape -> {tuple(z.shape)}\nvalues: {np.asarray(z.detach()).ravel()[:6].tolist()}", language="text")
        except Exception as e:  # noqa: BLE001
            st.code(f"{type(e).__name__}: {str(e).splitlines()[0][:150]}", language="text")
    with c6:
        cuda = T.cuda.is_available()
        st.code(f"torch.cuda.is_available() -> {cuda}\ndevice = 'cuda' if cuda else 'cpu'\nx.to(device).device -> {x.to('cuda' if cuda else 'cpu').device}", language="text")
        if not cuda:
            st.caption("لا GPU على هذه الآلة: `.to('cuda')` سيرفع `AssertionError: Torch not compiled with CUDA enabled` أو `RuntimeError`. في Colab مع GPU يعمل السطر نفسه.")
    intuition("الموتر **علميًا واحد** في NumPy وTensorFlow وPyTorch: شكل + dtype + قيم. ما يختلف هو الواجهة: `x.size()` مقابل `x.shape`، `x.dim()` مقابل `tf.rank`، `.to(device)` مقابل `tf.device`، و`requires_grad` مقابل `tf.Variable`.")
    practical_note("PyTorch: dtype الافتراضي للكسور float32 وللأعداد الصحيحة **int64** (لا int32 كما في TensorFlow). الفهارس والفئات للخسائر مثل `CrossEntropyLoss` يجب أن تكون int64 (`long`).")
