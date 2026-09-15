import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.python.pandas_basics",
    title_ar="أساسيات pandas: DataFrame وSeries والاختيار",
    title_en="pandas Basics: DataFrame, Series & Selection",
    module="foundations.python",
    order=8,
    prerequisites=["foundations.python.numpy_ndarray", "foundations.data.dataset"],
    objectives_ar=[
        "التمييز بين `DataFrame` (جدول) و`Series` (عمود).",
        "الاختيار بالاسم `df[\"col\"]`، بالشرط `df[df.x > 0]`، وبـ `loc/iloc`.",
        "الفحص السريع: `head`, `shape`, `dtypes`, `describe`, `isna().sum()`، ثم الخروج إلى `NumPy`.",
    ],
    terms=["dataset", "feature", "target"],
    labs=["labs.dataset_anatomy"],
    difficulty="beginner",
    summary_ar="pandas لقراءة الجدول وفحصه وتصفيته؛ NumPy للحساب؛ الطريق بينهما to_numpy().",
)

CODE = '''import numpy as np, pandas as pd

df = pd.DataFrame({
    "income": [4200, 6100, 2900, 8800, 5100, 3300],
    "age":    [34, 45, 29, 52, 38, 41],
    "city":   ["Algiers", "Oran", "Algiers", "Constantine", "Oran", "Annaba"],
    "defaulted": [0, 0, 1, 0, 0, 1],
})
print(df.shape); print(df.dtypes); print(df.head(3))

s = df["income"]                        # Series: عمود واحد باسم وفهرس
print(type(s).__name__, s.mean(), s.max())

print(df[["income", "age"]].describe().round(1))      # إحصاءات وصفية
print(df[df["income"] > 4000])                        # تصفية بشرط
print(df.loc[df["city"] == "Oran", ["income", "defaulted"]])   # loc: تسميات
print(df.iloc[0:2, 0:2])                              # iloc: مواضع
print(df.isna().sum())                                # قيم مفقودة لكل عمود
print(df.groupby("city")["defaulted"].mean())         # معدل التعثر لكل مدينة

X = df[["income", "age"]].to_numpy(dtype=np.float32)
y = df["defaulted"].to_numpy(dtype=np.float32)
print(X.shape, y.shape, X.dtype)'''


def render() -> None:
    lesson_header(LESSON)
    h2("جدول باسم للأعمدة", "A table with named columns")
    definition("**`DataFrame`** جدول ثنائي البُعد أعمدته مسماة وقد تختلف أنواعها. **`Series`** عمود واحد منه. `pandas` للقراءة والفحص والتنظيف؛ الشبكة لا تأخذ `DataFrame` مباشرة بل مصفوفة `NumPy`/موتر بعد التحويل.")
    table(
        ["المهمة", "الكود", "يعيد"],
        [("قراءة ملف", "pd.read_csv(\"data.csv\")", "DataFrame"), ("عمود", "df[\"income\"]", "Series"), ("أعمدة", "df[[\"income\", \"age\"]]", "DataFrame"),
         ("صفوف بشرط", "df[df[\"age\"] > 40]", "DataFrame"), ("بالتسمية", "df.loc[rows, cols]", "حسب الاختيار"), ("بالموضع", "df.iloc[0:5, 0:2]", "حسب الاختيار"),
         ("إلى NumPy", "df[cols].to_numpy(dtype=np.float32)", "ndarray")],
        ["rtl", "code", "rtl"],
    )
    code_lab(CodeLab(
        key="py_pandas", title_ar="فحص واختيار ثم تحويل", code=CODE,
        before=Before(goal_ar="الدورة الكاملة: بناء جدول، فحصه، تصفيته، تجميعه، ثم استخراج `X` و`y`.", stage_ar="البيانات ← الفحص ← التحضير.",
                      inputs_ar="جدول من 6 صفوف و4 أعمدة (أحدها نصي).", expected_ar="أشكال وأنواع، إحصاءات، جداول مصفاة، ومصفوفتان `X (6, 2)` و`y (6,)`."),
        explain=[("3-8", "قاموس أعمدة ← جدول. كل قائمة عمود؛ `city` نصي (`object`)."),
                 ("9", "الفحص الأول دائمًا: الشكل، الأنواع، أول صفوف."),
                 ("11-12", "`df[\"income\"]` عمود واحد = `Series` بدوال إحصائية جاهزة."),
                 ("14", "`describe` يعطي العدد والمتوسط والانحراف والأرباع — أسرع نظرة على المقاييس."),
                 ("15-17", "ثلاث طرق اختيار: قناع منطقي، `loc` بالتسميات (صفوف بشرط + أعمدة بالاسم)، `iloc` بالمواضع كـ NumPy."),
                 ("18-19", "المفقودات لكل عمود، ثم تجميع: معدل التعثر لكل مدينة — تحليل وصفي قبل أي نموذج."),
                 ("21-23", "الخروج إلى `NumPy`: نختار الأعمدة العددية فقط ونحدد `float32`. `city` تحتاج ترميزًا قبل ضمّها.")],
        run=run_printed(CODE),
        after_ar="- `dtypes` أظهر `object` لعمود `city`: لا يدخل `to_numpy(dtype=float32)` دون ترميز.\n- `groupby` يكشف اختلاف معدل التعثر بين المدن — إشارة إلى أن `city` خاصية مفيدة بعد ترميزها.",
    ))
    common_mistake("`df[\"income\", \"age\"]` (بدون قائمة داخلية) يعطي `KeyError`؛ الأعمدة المتعددة تُمرَّر كقائمة `df[[\"income\", \"age\"]]`.")
    practical_note("`loc` يشمل نهاية المدى عند التقطيع بالتسميات، بينما `iloc` لا يشملها كـ Python — مصدر التباس شائع.")
    quiz("py.pandas", [
        Q("`df[\"age\"]` يعيد…", ["DataFrame", "Series", "ndarray"], 1, "عمود واحد."),
        Q("لاختيار الصفوف التي `age > 40` والعمود `income` فقط…", ["df[df.age > 40][\"income\"]", "df.iloc[age > 40]", "df[\"age\" > 40]"], 0, "قناع ثم عمود.", kind="code"),
        Q("لماذا لا يدخل عمود `city` في `to_numpy(dtype=np.float32)`؟", ["لأنه طويل", "لأنه نصي", "لأنه فئوي ترتيبي"], 1, "object ← ترميز أولًا."),
    ])
    takeaway("DataFrame جدول، Series عمود. افحص (shape, dtypes, head, describe, isna)، صفِّ (قناع، loc، iloc)، ثم to_numpy(float32).")
    lesson_footer(LESSON, ["pandas للفحص والتنظيف، NumPy للحساب.", "الأعمدة المتعددة كقائمة.", "object يحتاج ترميزًا قبل الشبكة."])
