"""TensorFlow Tensor Explorer — create a tensor, change shape / rank / dtype /
values, and read back shape, dtype, rank, size, memory and device (spec §21.9)."""

import numpy as np
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.fw import tf

LAB = Lab(
    id="labs.tf_tensor_explorer",
    title_ar="مستكشف موترات TensorFlow",
    title_en="TensorFlow Tensor Explorer",
    category="frameworks",
    description_ar="أنشئ tf.Tensor بأي شكل ورتبة وdtype وقيم، وشاهد shape وdtype والرتبة وعدد العناصر والذاكرة والجهاز — وكيف تتغير عند reshape وتغيير dtype.",
    related_lessons=["foundations.frameworks.tensorflow.tensors", "foundations.data.shape_axis_rank", "foundations.linalg.tensors"],
)

DTYPES = ["float32", "float16", "float64", "int32", "int64", "bool"]


def _dims_ui() -> list[int]:
    rank = st.slider("الرتبة (عدد المحاور)", 0, 4, 2, key="tfx_rank")
    names = ["batch", "rows / time", "cols / features", "channels"]
    dims = []
    cols = st.columns(max(rank, 1))
    for i in range(rank):
        with cols[i]:
            dims.append(st.number_input(f"محور {i} ({names[i]})", 1, 64, [4, 3, 2, 2][i], key=f"tfx_d{i}"))
    return [int(d) for d in dims]


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    T = tf()
    dims = _dims_ui()
    c1, c2 = st.columns(2)
    with c1:
        dtype = st.selectbox("dtype", DTYPES, index=0, key="tfx_dtype")
    with c2:
        fill = st.selectbox("القيم", ["range (0, 1, 2, …)", "zeros", "ones", "random normal", "random integers 0–9"], key="tfx_fill")
    n = int(np.prod(dims)) if dims else 1
    if fill.startswith("range"):
        arr = np.arange(n, dtype="float64").reshape(dims)
    elif fill == "zeros":
        arr = np.zeros(dims)
    elif fill == "ones":
        arr = np.ones(dims)
    elif fill.startswith("random normal"):
        arr = np.random.default_rng(0).normal(size=dims)
    else:
        arr = np.random.default_rng(0).integers(0, 10, size=dims).astype("float64")
    if dtype == "bool":
        x = T.constant(arr != 0)
    else:
        x = T.cast(T.constant(arr), dtype)
    st.markdown("### الكود المكافئ")
    st.code(f"x = tf.constant(values, dtype=tf.{dtype})   # values.shape = {tuple(dims)}", language="python")
    st.markdown("### ما يعيده TensorFlow")
    size = int(T.size(x)); itemsize = x.dtype.size
    table(["الخاصية", "الاستدعاء", "القيمة", "المعنى"],
          [("الشكل", "x.shape", str(tuple(x.shape.as_list())), "عدد العناصر على كل محور"), ("الرتبة", "tf.rank(x) / x.ndim", str(int(T.rank(x))), "عدد المحاور = طول الشكل"),
           ("dtype", "x.dtype", str(x.dtype.name), f"نوع كل عنصر — {itemsize} بايت"), ("عدد العناصر", "tf.size(x)", str(size), "حاصل ضرب الشكل"),
           ("الذاكرة", "size × dtype.size", f"{size * itemsize} B" + (f" ≈ {size * itemsize / 1024:.1f} KB" if size * itemsize >= 1024 else ""), "تقريب: بدون رأس الكائن"),
           ("الجهاز", "x.device", x.device.split("/")[-1] or "CPU:0", "أين تعيش البيانات")],
          ["rtl", "code", "code", "rtl"])
    with st.expander("repr(x) — كيف يطبع TensorFlow الموتر", icon=":material/data_object:"):
        st.code(repr(x)[:1500], language="text")
        st.markdown("السطر الأول `<tf.Tensor: shape=…, dtype=…, numpy=` ثم المصفوفة كما تطبعها NumPy. الأقواس المتداخلة = الرتبة: `[[` رتبة 2، `[[[` رتبة 3.")
    st.markdown("### جرّب: reshape وcast")
    c3, c4 = st.columns(2)
    with c3:
        new_shape = st.text_input("شكل جديد (مثل 2,-1)", value=f"{n},-1" if n > 1 else "1", key="tfx_reshape")
        try:
            shp = [int(s) for s in new_shape.replace(" ", "").split(",") if s != ""]
            y = T.reshape(x, shp)
            st.code(f"tf.reshape(x, {shp}).shape -> {tuple(y.shape.as_list())}   # نفس {size} عنصرًا، ترتيب مختلف", language="text")
        except Exception as e:  # noqa: BLE001 - show the real error to the learner
            st.code(f"{type(e).__name__}: {str(e).splitlines()[0][:160]}", language="text")
            st.caption("reshape يحتاج نفس عدد العناصر: حاصل ضرب الشكل الجديد يجب أن يساوي " + str(size) + ".")
    with c4:
        to = st.selectbox("cast إلى", DTYPES, index=3, key="tfx_cast")
        z = T.cast(x, to)
        st.code(f"tf.cast(x, tf.{to}).dtype -> {z.dtype.name}\nmemory: {size * itemsize} B -> {size * z.dtype.size} B\nfirst values: {np.asarray(z).ravel()[:6].tolist()}", language="text")
        if dtype.startswith("float") and to.startswith("int"):
            st.caption("cast من كسور إلى أعداد صحيحة **يقتطع** (لا يقرّب): 2.9 → 2.")
    intuition("الموتر مفهوم واحد في كل الأطر (وحدة 4): مصفوفة متعددة الأبعاد بشكل وdtype. ما يضيفه TensorFlow: **الجهاز** الذي تعيش عليه، و**التسجيل** للاشتقاق. غيّر الرتبة إلى 4 بشكل (batch, H, W, C): هذا شكل دفعة صور (الأسبوع 8).")
    practical_note("float32 هو الافتراضي في التعلم العميق: نصف ذاكرة float64 ودقة كافية. float16/bfloat16 لتسريع GPU (تعميق). الأعداد الصحيحة للفهارس والفئات فقط — لا يمكن اشتقاقها.")
