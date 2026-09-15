"""Vector / Matrix Lab (spec §41): edit small matrices and see every
operation computed, with shape checks and element-level explanations."""

import numpy as np
import pandas as pd
import streamlit as st

from components.callouts import debugging_note, intuition
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc

LAB = Lab(
    id="labs.vector_matrix",
    title_ar="معمل المتجهات والمصفوفات",
    title_en="Vector / Matrix Lab",
    category="math",
    description_ar="عدّل مصفوفتين صغيرتين واحسب الجمع، الضرب في عدد، المنقولة، الضرب النقطي، ضرب المصفوفات، وهادامارد — مع تفكيك عنصر واحد خطوة بخطوة.",
    related_lessons=["foundations.linalg.dot_product", "foundations.linalg.matrix_multiplication"],
)


def _editor(label: str, default: np.ndarray, key: str) -> np.ndarray:
    st.markdown(f"**{label}**")
    df = pd.DataFrame(default, columns=[f"c{j}" for j in range(default.shape[1])])
    edited = st.data_editor(df, key=key, hide_index=True, num_rows="dynamic", width="stretch")
    arr = edited.to_numpy(dtype=float)
    st.caption(f"shape = {arr.shape}")
    return arr


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    intuition("غيّر الأرقام مباشرة في الجدولين. أضف صفًا لتغيير الشكل وشاهد أي العمليات تبقى ممكنة وأيها تفشل ولماذا.")
    c1, c2 = st.columns(2)
    with c1:
        A = _editor("A", np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), "vm_A")
    with c2:
        B = _editor("B", np.array([[0.5, -1.0, 2.0], [1.0, 0.0, 1.0]]), "vm_B")
    c = st.slider("عدد c", -3.0, 3.0, 2.0, 0.5, key="vm_c")

    st.markdown("### النتائج")
    tabs = st.tabs(["c·A", "Aᵀ", "A @ B", "A ⊙ A (هادامارد)", "A + B", "صف · عمود"])
    with tabs[0]:
        st.code(f"c * A =\n{c * A}", language="text")
    with tabs[1]:
        st.code(f"A.T  shape {A.T.shape} =\n{A.T}", language="text")
    with tabs[2]:
        if A.shape[1] == B.shape[0]:
            C = A @ B
            st.code(f"A{A.shape} @ B{B.shape} → C{C.shape}\n{C}", language="text")
            i = st.number_input("صف i", 0, A.shape[0] - 1, 0, key="vm_i"); j = st.number_input("عمود j", 0, B.shape[1] - 1, 0, key="vm_j")
            terms = " + ".join(f"({A[i, l]:g})({B[l, j]:g})" for l in range(A.shape[1]))
            st.code(f"C[{i},{j}] = A[{i},:] · B[:,{j}] = {terms} = {C[i, j]:g}", language="text")
        else:
            st.error(f"A{A.shape} @ B{B.shape} غير ممكن: أعمدة A ({A.shape[1]}) ≠ صفوف B ({B.shape[0]}).", icon="❌")
            debugging_note("هذا هو `matmul: Input operand 1 has a mismatch in its core dimension`. غيّر عدد أعمدة A أو صفوف B.")
    with tabs[3]:
        st.code(f"A * A =\n{A * A}", language="text")
        st.markdown("الضرب العنصري يتطلب نفس الشكل؛ يظهر في بوابات LSTM والأقنعة.")
    with tabs[4]:
        if A.shape == B.shape:
            st.code(f"A + B =\n{A + B}", language="text")
        else:
            st.error(f"A{A.shape} + B{B.shape} غير ممكن: الجمع يتطلب نفس الشكل (أو بثًا صالحًا).", icon="❌")
    with tabs[5]:
        r = st.number_input("صف من A", 0, A.shape[0] - 1, 0, key="vm_r"); k = st.number_input("عمود من B", 0, B.shape[1] - 1, 0, key="vm_k")
        row, col = A[r], B[:, k]
        if row.shape == col.shape:
            st.code(f"A[{r}] = {row}\nB[:,{k}] = {col}\ndot = Σ = {np.dot(row, col):g}", language="text")
        else:
            st.error(f"الضرب النقطي يتطلب نفس الطول: {row.shape} مقابل {col.shape}.", icon="❌")
