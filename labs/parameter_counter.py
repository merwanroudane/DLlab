"""Parameter Counter — dense / conv / embedding / RNN layer parameter arithmetic."""

import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.parameter_counter",
    title_ar="عدّاد المعلمات",
    title_en="Parameter Counter",
    category="training",
    description_ar="ابنِ شبكة طبقة طبقة (كثيفة، التفافية، تضمين، RNN/LSTM/GRU) وشاهد شكل المخرج وعدد المعلمات لكل طبقة والمجموع — نفس ما يطبعه model.summary().",
    related_lessons=["foundations.architecture.parameter_count"],
)


def _dense(n_in, units):
    return n_in * units + units


def _conv2d(h, w, c_in, filters, k, stride, padding):
    params = k * k * c_in * filters + filters
    if padding == "same":
        ho, wo = -(-h // stride), -(-w // stride)
    else:
        ho, wo = (h - k) // stride + 1, (w - k) // stride + 1
    return params, (ho, wo, filters)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    kind = st.segmented_control("نوع الشبكة", ["كثيفة (جدول)", "التفافية (صورة)", "تكرارية (تسلسل)"], default="كثيفة (جدول)", key="pcnt_kind")
    rows = []; total = 0
    if kind == "كثيفة (جدول)":
        n_in = st.number_input("عدد المدخلات", 1, 10000, 20, key="pcnt_nin")
        txt = st.text_input("وحدات الطبقات (بفواصل)، الأخيرة = الإخراج", "64, 32, 1", key="pcnt_units")
        try:
            units = [int(x) for x in txt.split(",") if x.strip()]
        except ValueError:
            units = [64, 32, 1]
        prev = n_in
        for i, u in enumerate(units, 1):
            p = _dense(prev, u); total += p
            rows.append((f"dense_{i}", f"W ({prev}, {u}) + b ({u},)", f"(None, {u})", f"{p:,}")); prev = u
        intuition("كل طبقة: in×out + out. `None` في الشكل هو بُعد الدفعة غير المحدد — كما في Keras.")
    elif kind == "التفافية (صورة)":
        c1, c2, c3 = st.columns(3)
        with c1:
            h = st.number_input("الارتفاع H", 8, 512, 28, key="pcnt_h"); w = st.number_input("العرض W", 8, 512, 28, key="pcnt_w")
        with c2:
            c_in = st.number_input("القنوات C", 1, 64, 1, key="pcnt_c"); n_conv = st.slider("عدد طبقات Conv", 1, 4, 2, key="pcnt_nconv")
        with c3:
            k = st.selectbox("حجم النواة k", [3, 5, 7], key="pcnt_k"); filters = st.number_input("المرشحات (تتضاعف كل طبقة)", 4, 256, 16, key="pcnt_f")
        padding = st.selectbox("padding", ["valid", "same"], key="pcnt_pad"); pool = st.checkbox("MaxPool 2×2 بعد كل Conv", value=True, key="pcnt_pool")
        dense_units = st.number_input("وحدات الطبقة الكثيفة قبل الإخراج", 1, 1024, 64, key="pcnt_du"); n_classes = st.number_input("عدد الفئات", 1, 1000, 10, key="pcnt_nc")
        shape = (h, w, c_in); f = filters
        for i in range(1, n_conv + 1):
            c_prev = shape[2]
            p, shape = _conv2d(shape[0], shape[1], c_prev, f, k, 1, padding); total += p
            rows.append((f"conv2d_{i}", f"kernel ({k}, {k}, {c_prev}, {f}) + b ({f},)", f"(None, {shape[0]}, {shape[1]}, {f})", f"{p:,}"))
            if pool:
                shape = (shape[0] // 2, shape[1] // 2, f)
                rows.append((f"max_pool_{i}", "—", f"(None, {shape[0]}, {shape[1]}, {f})", "0"))
            f *= 2
        flat = shape[0] * shape[1] * shape[2]
        rows.append(("flatten", "—", f"(None, {flat})", "0"))
        p = _dense(flat, dense_units); total += p; rows.append(("dense_1", f"W ({flat}, {dense_units}) + b", f"(None, {dense_units})", f"{p:,}"))
        p = _dense(dense_units, n_classes); total += p; rows.append(("dense_out", f"W ({dense_units}, {n_classes}) + b", f"(None, {n_classes})", f"{p:,}"))
        intuition("Conv: k×k×C_in×filters + filters — لا يعتمد على حجم الصورة! الطبقة الكثيفة بعد Flatten هي التي تنفجر بحجم الصورة. Pool بلا معلمات.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            vocab = st.number_input("حجم المفردات (0 = بلا تضمين)", 0, 100000, 5000, key="pcnt_vocab"); emb = st.number_input("بُعد التضمين", 2, 512, 32, key="pcnt_emb")
        with c2:
            cell = st.selectbox("الخلية", ["SimpleRNN", "LSTM", "GRU"], key="pcnt_cell"); units = st.number_input("الوحدات", 1, 512, 64, key="pcnt_runits")
        with c3:
            feat = st.number_input("خصائص كل خطوة (إن لم يوجد تضمين)", 1, 100, 3, key="pcnt_feat"); out = st.number_input("وحدات الإخراج", 1, 100, 1, key="pcnt_rout")
        d_in = emb if vocab else feat
        if vocab:
            p = vocab * emb; total += p; rows.append(("embedding", f"({vocab}, {emb})", f"(None, T, {emb})", f"{p:,}"))
        gates = {"SimpleRNN": 1, "LSTM": 4, "GRU": 3}[cell]
        p = gates * (units * (d_in + units) + units); total += p
        rows.append((cell.lower(), f"{gates} × [W ({d_in}, {units}) + U ({units}, {units}) + b ({units},)]", f"(None, {units})", f"{p:,}"))
        p = _dense(units, out); total += p; rows.append(("dense_out", f"W ({units}, {out}) + b", f"(None, {out})", f"{p:,}"))
        intuition("الخلية التكرارية: لكل بوابة مصفوفة للمدخل W، ومصفوفة للحالة U، وانحياز. LSTM أربع بوابات، GRU ثلاث، RNN واحدة. المعلمات لا تعتمد على طول التسلسل T.")
    table(["الطبقة", "المعلمات (شكلها)", "شكل المخرج", "Param #"], rows + [("Total", "", "", f"{total:,}")], ["code", "code", "code", "num"])
    st.code(f"Total params: {total:,}   float32 ≈ {total * 4 / 1e6:.2f} MB", language="text")
    practical_note("قارن المجموع بعدد صفوف بياناتك. ملايين المعلمات على آلاف الصفوف تستدعي تنظيمًا قويًا أو تصميمًا أصغر.")
