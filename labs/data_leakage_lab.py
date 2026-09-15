"""Data Leakage Lab — toggle three leakage sources on/off and watch test accuracy."""

import numpy as np
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.data_leakage_lab",
    title_ar="معمل تسريب البيانات",
    title_en="Data Leakage Lab",
    category="data",
    description_ar="فعّل أو عطّل ثلاثة مصادر تسريب (خاصية تُعرف بعد الحدث، تحجيم قبل التقسيم، تكرار صفوف بين المجموعات) وشاهد أثر كل منها على دقة الاختبار.",
    related_lessons=["foundations.prep.leakage"],
)


def _logreg(Xtr, ytr, Xte, epochs=300, eta=0.5):
    Xb = np.column_stack([Xtr, np.ones(len(Xtr))]); w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(Xb @ w))); w -= eta * Xb.T @ (p - ytr) / len(ytr)
    pte = 1 / (1 + np.exp(-(np.column_stack([Xte, np.ones(len(Xte))]) @ w)))
    ptr = 1 / (1 + np.exp(-(Xb @ w)))
    return ptr, pte


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    rng = np.random.default_rng(0)
    n = 1500
    income = rng.normal(5000, 1500, n); late = rng.poisson(1.2, n)
    p = 1 / (1 + np.exp(-(-2.8 + 0.7 * late - 0.0003 * income)))
    y = (rng.uniform(size=n) < p).astype(float)
    after_event = np.where(y == 1, rng.integers(1, 12, n), 0).astype(float)   # تُعرف بعد الحدث

    c1, c2, c3 = st.columns(3)
    with c1:
        leak_target = st.toggle("خاصية تُعرف بعد الحدث", value=False, key="dll_target")
    with c2:
        leak_prep = st.toggle("تحجيم على كل البيانات قبل التقسيم", value=False, key="dll_prep")
    with c3:
        leak_dup = st.toggle("تكرار 30% من الصفوف قبل التقسيم", value=False, key="dll_dup")
    intuition("ابدأ بكل المفاتيح مطفأة (نموذج صادق)، ثم فعّل واحدًا في كل مرة. لاحظ أيها يعطي أثرًا هائلًا وأيها أثرًا خفيًا.")

    X = np.column_stack([income, late] + ([after_event] if leak_target else []))
    yy = y.copy()
    if leak_dup:
        dup = rng.choice(n, size=int(0.3 * n), replace=False)
        X = np.vstack([X, X[dup]]); yy = np.concatenate([yy, yy[dup]])
    m = len(X)
    idx = rng.permutation(m); k = int(0.75 * m); tr, te = idx[:k], idx[k:]
    if leak_prep:
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd; Xtr, Xte = Xs[tr], Xs[te]
    else:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ptr, pte = _logreg(Xtr, yy[tr], Xte)
    acc_tr = ((ptr >= 0.5) == yy[tr]).mean(); acc_te = ((pte >= 0.5) == yy[te]).mean()
    base = max(y.mean(), 1 - y.mean())
    # Honest reference: evaluate the same model on a fresh untouched sample
    fresh_income = rng.normal(5000, 1500, 500); fresh_late = rng.poisson(1.2, 500)
    fresh_p = 1 / (1 + np.exp(-(-2.8 + 0.7 * fresh_late - 0.0003 * fresh_income))); fresh_y = (rng.uniform(size=500) < fresh_p).astype(float)
    fresh_X = np.column_stack([fresh_income, fresh_late] + ([np.zeros(500)] if leak_target else []))   # في الإنتاج الخاصية غير معروفة → 0
    _, pfresh = _logreg(Xtr, yy[tr], (fresh_X - mu) / sd)
    acc_fresh = ((pfresh >= 0.5) == fresh_y).mean()
    table(["المقياس", "القيمة"],
          [("خط الأساس (الفئة الأغلب)", f"{base:.3f}"), ("دقة التدريب", f"{acc_tr:.3f}"), ("دقة الاختبار (كما يراها الباحث)", f"{acc_te:.3f}"),
           ("دقة على بيانات إنتاج حقيقية جديدة", f"{acc_fresh:.3f}")], ["rtl", "num"])
    if leak_target:
        warning_note("تسريب الهدف: دقة اختبار قريبة من 100%، لكن على بيانات الإنتاج (حيث الخاصية غير معروفة) ينهار الأداء إلى خط الأساس أو أسوأ. هذا أخطر الأنواع.")
    if leak_dup:
        warning_note("التكرار: نفس الصفوف في التدريب والاختبار ترفع دقة الاختبار قليلًا فوق الحقيقة. الأثر خفي مع نموذج بسيط، وكبير مع نموذج يحفظ (شبكة كبيرة).")
    if leak_prep:
        warning_note("تسريب المعالجة: الأثر هنا صغير جدًا رقميًا لكنه خرق للمبدأ؛ في نماذج أكثر حساسية (اختيار خصائص، تعويض بالهدف) يصبح كبيرًا.")
    if not (leak_target or leak_dup or leak_prep):
        st.success("نموذج صادق: دقة الاختبار ≈ دقة الإنتاج. هذا ما نريده حتى لو كانت الأرقام متواضعة.", icon="✅")
