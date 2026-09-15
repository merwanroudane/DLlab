import numpy as np
import pandas as pd
import streamlit as st

from components.callouts import debugging_note, practical_note, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.datasets import DATASETS_AR, LOADERS, variable_kind

LAB = Lab(
    id="labs.dataset_anatomy",
    title_ar="معمل تشريح مجموعة البيانات",
    title_en="Dataset Anatomy Lab",
    category="data",
    description_ar="اختر مجموعة بيانات، حدد الهدف والخصائص، واقرأ الأشكال والأنواع والقيم المفقودة كما تراها الشبكة.",
    related_lessons=["foundations.data.dataset", "foundations.data.shape_axis_rank", "foundations.data.dtype"],
)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)

    ds_key = st.selectbox("مجموعة البيانات", list(DATASETS_AR), format_func=lambda k: DATASETS_AR[k],
                          key="anatomy_ds")
    df = LOADERS[ds_key]()

    st.markdown("### 1) الجدول كما هو")
    c1, c2, c3 = st.columns(3)
    c1.metric("الملاحظات n", df.shape[0])
    c2.metric("الأعمدة", df.shape[1])
    c3.metric("قيم مفقودة", int(df.isna().sum().sum()))
    st.dataframe(df.head(8), hide_index=True, width="stretch")

    st.markdown("### 2) نوع كل عمود: إحصائيًا وتخزينيًا")
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append((col, str(s.dtype), variable_kind(s), str(s.nunique(dropna=True)), str(int(s.isna().sum()))))
    table(["العمود", "dtype (تخزين)", "النوع الإحصائي (تقديري)", "قيم فريدة", "مفقودة"], rows,
          ["code", "code", "rtl", "num", "num"])
    warning_note("النوع الإحصائي هنا **تقدير آلي**. عمود `rooms` بقيم 1..8 قد يكون عدديًا منفصلًا أو ترتيبيًا حسب سؤالك — القرار لك.")

    st.markdown("### 3) اختر الهدف والخصائص")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target = st.selectbox("الهدف y", numeric_cols, index=len(numeric_cols) - 1, key="anatomy_target")
    feature_options = [c for c in numeric_cols if c != target]
    features = st.multiselect("الخصائص X (عددية فقط في هذا المعمل)", feature_options,
                              default=feature_options[:3], key="anatomy_features")
    if not features:
        st.info("اختر خاصية واحدة على الأقل.")
        return
    drop_na = st.checkbox("حذف الصفوف ذات القيم المفقودة", value=True, key="anatomy_dropna")
    work = df[features + [target]]
    if drop_na:
        work = work.dropna()
    X = work[features].to_numpy(dtype=np.float32)
    y = work[target].to_numpy(dtype=np.float32)

    st.markdown("### 4) كما تراها الشبكة")
    uniq = np.unique(y)
    if len(uniq) == 2 and set(uniq.tolist()) <= {0.0, 1.0}:
        task = "تصنيف ثنائي — الهدف 0/1"
    elif len(uniq) <= 10 and np.all(np.equal(np.mod(uniq, 1), 0)):
        task = f"تصنيف متعدد الفئات ({len(uniq)} فئات) أو عددي منفصل — احكم من المعنى"
    else:
        task = "انحدار — الهدف عددي متصل"
    st.code(
        f"X.shape = {X.shape}    # (n, d): {X.shape[0]} ملاحظة × {X.shape[1]} خصائص\n"
        f"y.shape = {y.shape}    # (n,)\n"
        f"X.dtype = {X.dtype}, y.dtype = {y.dtype}\n"
        f"X.ndim  = {X.ndim}, y.ndim = {y.ndim}\n"
        f"np.unique(y)[:8] = {uniq[:8]}\n"
        f"X.shape[0] == y.shape[0] → {X.shape[0] == y.shape[0]}",
        language="text",
    )
    st.success(f"نوع المسألة: **{task}**", icon="🧭")
    if not drop_na and np.isnan(X).any():
        debugging_note("توجد `NaN` في `X`. تدريب شبكة على هذه المصفوفة يعطي خسارة `NaN` من أول خطوة. حذف أو تعويض القيم المفقودة يسبق أي نموذج.")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("**X — أول 6 صفوف**")
        st.dataframe(pd.DataFrame(X[:6], columns=features), hide_index=True, width="stretch")
    with c2:
        st.markdown("**y — أول 6**")
        st.dataframe(pd.DataFrame({target: y[:6]}), hide_index=True, width="stretch")
    st.markdown("**إحصاءات كل خاصية على المحور 0 (عبر الملاحظات):**")
    stats = pd.DataFrame({"feature": features, "mean": X.mean(axis=0).round(3), "std": X.std(axis=0).round(3),
                          "min": X.min(axis=0), "max": X.max(axis=0)})
    st.dataframe(stats, hide_index=True, width="stretch")
    practical_note(
        "لاحظ فرق النطاقات بين الخصائص (مثلًا الدخل بالآلاف والعمر بالعشرات). هذا سبب حاجتنا إلى التحجيم قبل التدريب "
        "— موضوع وحدة إعداد البيانات ومعمل التحجيم."
    )
