"""Data Split Lab — random / stratified / temporal / group splits, visualised."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.datasets import loan_default

LAB = Lab(
    id="labs.data_split_lab",
    title_ar="معمل تقسيم البيانات",
    title_en="Data Split Lab",
    category="data",
    description_ar="قسّم بيانات القروض عشوائيًا أو طبقيًا أو زمنيًا أو بالمجموعة، وراقب نسب الفئات والتداخل وأثر البذرة على تقدير الأداء.",
    related_lessons=["foundations.prep.splitting", "foundations.ml.baseline_generalization"],
)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    df = loan_default().reset_index(drop=True)
    n = len(df)
    y = df["defaulted"].to_numpy()
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox("الطريقة", ["عشوائي", "طبقي", "زمني (بترتيب الصفوف)", "بالمجموعة (المدينة)"], key="dsl_method")
    with c2:
        test_frac = st.slider("نسبة الاختبار", 0.1, 0.5, 0.2, 0.05, key="dsl_frac")
    with c3:
        seed = st.number_input("seed", 0, 999, 0, key="dsl_seed")
    rng = np.random.default_rng(int(seed))
    if method == "عشوائي":
        idx = rng.permutation(n); k = int((1 - test_frac) * n); tr, te = idx[:k], idx[k:]
    elif method == "طبقي":
        tr, te = [], []
        for cls in (0, 1):
            m = rng.permutation(np.where(y == cls)[0]); k = int((1 - test_frac) * len(m))
            tr += m[:k].tolist(); te += m[k:].tolist()
        tr, te = np.array(tr), np.array(te)
    elif method.startswith("زمني"):
        k = int((1 - test_frac) * n); tr, te = np.arange(k), np.arange(k, n)
    else:
        cities = df["city"].unique(); test_city = rng.choice(cities)
        te = np.where(df["city"] == test_city)[0]; tr = np.where(df["city"] != test_city)[0]
        st.caption(f"مدينة الاختبار: {test_city}")
    intuition("انظر إلى «نسبة التعثر» في العمودين: هل تتشابه؟ غيّر البذرة عدة مرات في العشوائي ولاحظ التذبذب، ثم جرّب الطبقي.")
    table(["المجموعة", "الحجم", "نسبة التعثر", "عدد المتعثرين"],
          [("تدريب", str(len(tr)), f"{y[tr].mean():.3f}", str(int(y[tr].sum()))), ("اختبار", str(len(te)), f"{y[te].mean():.3f}", str(int(y[te].sum())))],
          ["rtl", "num", "num", "num"])
    st.code(f"overlap train∩test = {len(set(tr) & set(te))}   (must be 0)", language="text")
    fig = go.Figure()
    color = np.array(["#B9B2A6"] * n, dtype=object); color[te] = "#C8473A"
    fig.add_trace(go.Scatter(x=np.arange(n), y=y + rng.uniform(-0.08, 0.08, n), mode="markers", marker=dict(color=color.tolist(), size=6), name="rows"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="row index (time order)", yaxis=dict(tickvals=[0, 1], title="defaulted"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", showlegend=False)
    st.plotly_chart(fig, width="stretch", key="dsl_fig")
    st.caption("الأحمر = صفوف الاختبار. في الزمني كلها في النهاية؛ في العشوائي متناثرة.")
    st.markdown("### تذبذب تقدير الأداء بين البذور (خط الأساس)")
    accs = []
    for s in range(30):
        r = np.random.default_rng(s); idx = r.permutation(n); k = int((1 - test_frac) * n); te_s = idx[k:]
        accs.append(1 - y[te_s].mean())
    st.code(f"majority-baseline test accuracy over 30 random seeds: mean={np.mean(accs):.3f}  min={np.min(accs):.3f}  max={np.max(accs):.3f}", language="text")
    warning_note("حتى خط الأساس يتذبذب بعدة نقاط مئوية بين البذور مع اختبار صغير. عند مقارنة نموذجين بفرق 1% تحقق أن الفرق أكبر من هذا التذبذب — أو استخدم التحقق المتقاطع.")
