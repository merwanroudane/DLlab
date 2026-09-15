"""Small packaged synthetic datasets (spec §55): deterministic, offline,
realistic enough for teaching. Every generator takes a seed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

DATASETS_AR = {
    "loan_default": "تعثر القروض (تصنيف ثنائي)",
    "house_prices": "أسعار العقارات (انحدار)",
    "monthly_inflation": "التضخم الشهري (سلسلة زمنية)",
    "study_hours": "ساعات المذاكرة ← درجة الامتحان (انحدار بسيط)",
}


@st.cache_data(ttl=3600, max_entries=8)
def loan_default(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=8.4, sigma=0.45, size=n).round(0)
    age = rng.integers(21, 66, size=n)
    late = rng.poisson(1.2, size=n)
    debt_ratio = np.clip(rng.normal(0.35, 0.15, size=n), 0.02, 0.95).round(3)
    city = rng.choice(["Algiers", "Oran", "Constantine", "Annaba"], size=n, p=[0.45, 0.25, 0.2, 0.1])
    employment = rng.choice(["employed", "self-employed", "unemployed"], size=n, p=[0.65, 0.25, 0.10])
    logit = -2.2 + 1.6 * debt_ratio * 3 + 0.55 * late - 0.00004 * income + (employment == "unemployed") * 1.1
    p = 1 / (1 + np.exp(-logit))
    defaulted = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({
        "customer_id": np.arange(1001, 1001 + n),
        "income": income,
        "age": age,
        "num_late_payments": late,
        "debt_ratio": debt_ratio,
        "city": city,
        "employment": employment,
        "defaulted": defaulted,
    })
    # a few missing values on purpose (teaching)
    idx = rng.choice(n, size=max(3, n // 60), replace=False)
    df.loc[idx, "debt_ratio"] = np.nan
    return df


@st.cache_data(ttl=3600, max_entries=8)
def house_prices(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.normal(120, 35, size=n).clip(35, 320).round(0)
    rooms = np.clip((area / 35).round(0) + rng.integers(-1, 2, size=n), 1, 8).astype(int)
    age_years = rng.integers(0, 45, size=n)
    district = rng.choice(["center", "suburb", "coastal", "rural"], size=n, p=[0.3, 0.4, 0.15, 0.15])
    dist_premium = {"center": 1.35, "suburb": 1.0, "coastal": 1.5, "rural": 0.7}
    price = (area * 900 + rooms * 4000 - age_years * 650) * np.array([dist_premium[d] for d in district])
    price = (price * rng.normal(1, 0.08, size=n)).round(-2)
    return pd.DataFrame({"area_m2": area, "rooms": rooms, "age_years": age_years, "district": district,
                         "price": price})


@st.cache_data(ttl=3600, max_entries=8)
def monthly_inflation(n_months: int = 180, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    trend = 2.5 + 0.004 * t
    season = 0.6 * np.sin(2 * np.pi * t / 12)
    ar = np.zeros(n_months)
    for i in range(1, n_months):
        ar[i] = 0.75 * ar[i - 1] + rng.normal(0, 0.35)
    inflation = (trend + season + ar).round(2)
    rate = (3.0 + 0.5 * np.roll(inflation, 2) + rng.normal(0, 0.2, n_months)).round(2)
    unemployment = (9.0 - 0.3 * np.roll(inflation, 1) + rng.normal(0, 0.3, n_months)).round(2)
    dates = pd.date_range("2011-01-01", periods=n_months, freq="MS")
    return pd.DataFrame({"date": dates, "inflation": inflation, "policy_rate": rate, "unemployment": unemployment})


@st.cache_data(ttl=3600, max_entries=8)
def study_hours(n: int = 40, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0.5, 10, size=n).round(1)
    score = (45 + 4.6 * hours + rng.normal(0, 3.5, size=n)).clip(0, 100).round(1)
    return pd.DataFrame({"hours": hours, "score": score})


LOADERS = {
    "loan_default": loan_default,
    "house_prices": house_prices,
    "monthly_inflation": monthly_inflation,
    "study_hours": study_hours,
}


def variable_kind(series: pd.Series) -> str:
    """Heuristic statistical type used for teaching (not a substitute for judgement)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "زمني (تاريخ)"
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        k = series.nunique(dropna=True)
        return "فئوي ثنائي" if k == 2 else "فئوي اسمي"
    vals = series.dropna().unique()
    if len(vals) == 2 and set(vals) <= {0, 1}:
        return "فئوي ثنائي (0/1)"
    if pd.api.types.is_integer_dtype(series) and len(vals) <= 12:
        return "عددي منفصل (أو فئوي مرمّز — احكم أنت)"
    return "عددي متصل"
