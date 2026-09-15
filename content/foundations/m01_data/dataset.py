import numpy as np
import pandas as pd
import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.data.dataset",
    title_ar="مجموعة البيانات وتشريحها",
    title_en="Dataset & Dataset Anatomy",
    module="foundations.data",
    order=2,
    prerequisites=["foundations.data.what_is_data"],
    objectives_ar=[
        "تعريف مجموعة البيانات كمجموعة ملاحظات بنفس البنية.",
        "التعرف على المكونات: الصفوف، الأعمدة، المدخلات `X`، الهدف `y`.",
        "فحص مجموعة بيانات ببايثون: `shape` و`head()` و`dtypes`.",
    ],
    terms=["dataset", "observation", "feature", "target", "shape"],
    labs=["labs.dataset_anatomy"],
    related=["foundations.data.shape_axis_rank"],
    difficulty="beginner",
    summary_ar="مجموعة البيانات = ملاحظات بنفس الأعمدة. نفصلها إلى X (خصائص) و y (هدف) ونفحص شكلها.",
)

CODE = '''import numpy as np
import pandas as pd

df = pd.DataFrame({
    "income":            [4200, 6100, 2900, 8800, 5100, 3300],
    "num_late_payments": [0, 2, 5, 0, 1, 3],
    "age":               [34, 45, 29, 52, 38, 41],
    "defaulted":         [0, 0, 1, 0, 0, 1],
})

print(df.shape)            # (rows, columns)
print(df.dtypes)           # نوع كل عمود

X = df[["income", "num_late_payments", "age"]].to_numpy(dtype=np.float32)
y = df["defaulted"].to_numpy(dtype=np.float32)

print("X.shape =", X.shape)   # (6, 3): 6 ملاحظات × 3 خصائص
print("y.shape =", y.shape)   # (6,):   هدف واحد لكل ملاحظة'''


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "income": [4200, 6100, 2900, 8800, 5100, 3300],
        "num_late_payments": [0, 2, 5, 0, 1, 3],
        "age": [34, 45, 29, 52, 38, 41],
        "defaulted": [0, 0, 1, 0, 0, 1],
    })


def _controls() -> dict:
    n = st.slider("عدد الصفوف المستخدمة", 2, 6, 6, key="ctrl_dataset_n")
    cols = st.multiselect("أعمدة الخصائص", ["income", "num_late_payments", "age"],
                          default=["income", "num_late_payments", "age"], key="ctrl_dataset_cols")
    return {"n": n, "cols": cols}


def _run(p: dict) -> None:
    df = _df().head(p["n"])
    if not p["cols"]:
        st.warning("اختر عمودًا واحدًا على الأقل.", icon="⚠️")
        return
    X = df[p["cols"]].to_numpy(dtype=np.float32)
    y = df["defaulted"].to_numpy(dtype=np.float32)
    st.code(
        f"df.shape = {df.shape}\n{df.dtypes.to_string()}\n\nX.shape = {X.shape}\ny.shape = {y.shape}\n"
        f"X.dtype = {X.dtype}, y.dtype = {y.dtype}",
        language="text",
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("**X (الخصائص)**")
        st.dataframe(pd.DataFrame(X, columns=p["cols"]), hide_index=True, width="stretch")
    with c2:
        st.markdown("**y (الهدف)**")
        st.dataframe(pd.DataFrame({"defaulted": y}), hide_index=True, width="stretch")


def render() -> None:
    lesson_header(LESSON)
    h2("ما هي؟", "What is it?")
    definition(
        "**مجموعة البيانات** `Dataset` هي مجموعة من **الملاحظات** لها **نفس الأعمدة**. الشرط الأساسي: "
        "كل ملاحظة تُقاس بنفس القياسات. إذا كانت لعميل 4 قيم ولآخر 7 قيم فليست مجموعة بيانات جاهزة بعد."
    )
    why("النموذج يتعلم دالة واحدة تنطبق على كل الملاحظات؛ لذلك يجب أن يكون لكل ملاحظة نفس عدد المدخلات وبنفس الترتيب.")
    h2("التشريح", "Anatomy")
    table(
        ["المكوّن", "English", "الرمز المعتاد", "الشكل", "المعنى"],
        [
            ("الملاحظات", "Observations / rows", "n", "عدد الصفوف", "كم كيانًا لدينا"),
            ("الخصائص", "Features / columns", "d", "عدد أعمدة المدخلات", "كم قياسًا لكل كيان"),
            ("مصفوفة المدخلات", "Input matrix", "X", "(n, d)", "كل الخصائص لكل الملاحظات"),
            ("متجه الهدف", "Target vector", "y", "(n,)", "قيمة واحدة لكل ملاحظة"),
        ],
        ["rtl", "ltr", "code", "code", "rtl"],
    )
    st.markdown(
        """
الفصل إلى `X` و`y` هو أول عملية عملية في أي مشروع: **`X` ما نعرفه، `y` ما نريد معرفته**.
لاحظ أن `X` **مصفوفة** (صفوف وأعمدة) بينما `y` **متجه** (عمود واحد) في حالة الهدف الواحد.
"""
    )
    code_lab(CodeLab(
        key="dataset_xy",
        title_ar="من جدول إلى X و y",
        level="A",
        code=CODE,
        before=Before(
            goal_ar="فحص مجموعة بيانات صغيرة وفصلها إلى مصفوفة خصائص `X` ومتجه هدف `y`.",
            stage_ar="البيانات ← الفحص ← التحضير (قبل أي نموذج).",
            inputs_ar="جدول `pandas.DataFrame` من 6 صفوف و4 أعمدة.",
            expected_ar="`X.shape == (6, 3)` و`y.shape == (6,)` ونوع `float32`.",
            math_ar="$X \\in \\mathbb{R}^{n \\times d}$ و $y \\in \\mathbb{R}^{n}$ مع $n=6$ و $d=3$.",
            objects_ar="`DataFrame` للجدول، ثم `numpy.ndarray` لكل من `X` و`y`.",
        ),
        explain=[
            ("1-2", "`NumPy` للمصفوفات العددية، و`pandas` للجداول ذات الأعمدة المسماة."),
            ("4-9", "نبني الجدول يدويًا. في الواقع يأتي من `pd.read_csv(...)`. كل قائمة عمود، وطولها = عدد الملاحظات."),
            ("11", "`df.shape` يعيد `(6, 4)`: 6 صفوف و4 أعمدة. الترتيب دائمًا (صفوف، أعمدة)."),
            ("12", "`dtypes` يبين نوع تخزين كل عمود (`int64` هنا). سنفرد له درسًا."),
            ("14", "نختار أعمدة الخصائص فقط ونحولها إلى مصفوفة `NumPy` من نوع `float32` — النوع المفضل لأطر التعلم العميق."),
            ("15", "الهدف عمود واحد ← متجه بالشكل `(6,)`. ليس `(6, 1)` — الفرق مهم وسنعود إليه."),
            ("17-18", "الفحص الأخير قبل النموذج: الشكلان متوافقان؟ عدد صفوف `X` = طول `y`؟"),
        ],
        controls=_controls,
        run=_run,
        after_ar=(
            "- `X.shape = (n, d)`: الرقم الأول عدد الملاحظات، الثاني عدد الخصائص. غيّر عدد الصفوف أو الأعمدة ولاحظ أي رقم يتغير.\n"
            "- `y.shape = (n,)`: متجه بطول عدد الملاحظات. لا يوجد رقم ثانٍ لأن الهدف قيمة واحدة لكل ملاحظة.\n"
            "- `float32` هو النوع الذي تحبه `TensorFlow` و`PyTorch` افتراضيًا؛ `int64` قد يسبب تحذيرات أو أخطاء لاحقًا."
        ),
    ))
    common_mistake(
        "الخلط بين `(6,)` و`(6, 1)`. الأول متجه، الثاني مصفوفة بعمود واحد. بعض دوال الخسارة تتوقع الأول، "
        "وبعض الطبقات تنتج الثاني — عدم الانتباه يعطي تحذيرًا صامتًا أو نتائج خاطئة."
    )
    practical_note("جرّب المعمل «تشريح مجموعة البيانات» لتطبيق هذا على بيانات أكبر مع اختيار الهدف والخصائص بنفسك.")
    quiz(
        "data.dataset",
        [
            Q("`df.shape == (250, 9)`. كم ملاحظة؟", ["9", "250", "2250"], 1, "الرقم الأول = الصفوف = الملاحظات.", kind="shape"),
            Q("إذا كان `X.shape == (250, 8)` فما `y.shape` المتوقع للهدف الواحد؟", ["(250,)", "(8,)", "(250, 8)"], 0,
              "قيمة هدف واحدة لكل ملاحظة.", kind="shape"),
            Q("ما الشرط الأساسي لمجموعة البيانات؟", ["أن تكون كبيرة", "أن تكون لكل الملاحظات نفس الأعمدة", "أن تكون أرقامًا فقط"], 1,
              "نفس البنية لكل ملاحظة."),
        ],
    )
    takeaway("مجموعة البيانات = ملاحظات بنفس الأعمدة. X (n, d) ما نعرفه، y (n,) ما نتنبأ به.")
    lesson_footer(LESSON, [
        "Dataset: ملاحظات كثيرة بنفس البنية.",
        "X مصفوفة الخصائص (n, d)، y متجه الهدف (n,).",
        "افحص `shape` و`dtypes` دائمًا قبل أي نموذج.",
    ])
