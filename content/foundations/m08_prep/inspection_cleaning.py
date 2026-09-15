import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, practical_note, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.prep.inspection_cleaning",
    title_ar="الفحص والتنظيف: مفقودات، تكرار، شواذ، قيم خاطئة، تسميات خاطئة، ضوضاء",
    title_en="Inspection & Cleaning: Missing, Duplicates, Outliers, Wrong Values, Label Errors, Noise",
    module="foundations.prep",
    order=1,
    prerequisites=["foundations.python.pandas_basics", "foundations.data.dataset_anatomy"],
    objectives_ar=[
        "تنفيذ قائمة فحص قياسية على أي جدول قبل النموذج.",
        "اختيار استراتيجية للمفقودات (حذف، تعويض، مؤشر) حسب السبب.",
        "اكتشاف الشواذ والقيم المستحيلة والتكرارات وأخطاء التسميات.",
    ],
    terms=["dataset", "observation", "noise"],
    labs=["labs.dataset_anatomy"],
    difficulty="beginner",
    summary_ar="قبل أي نموذج: مفقودات، تكرار، شواذ، مستحيلات، تسميات؛ لكل منها اكتشاف واستراتيجية.",
)

CODE = '''import numpy as np, pandas as pd
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "income": [4200, 6100, np.nan, 8800, 5100, 3300, 5100, -50, 4200, 9_900_000],
    "age":    [34, 45, 29, 52, 38, 41, 38, 30, 34, 44],
    "city":   ["Algiers", "Oran", "Algiers", "Constantine", "Oran", "Annaba", "Oran", "algiers", "Algiers", "Oran"],
    "defaulted": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
})
print("shape:", df.shape)
print("missing per column:\\n", df.isna().sum())
print("duplicate rows:", df.duplicated().sum())                 # الصف 8 يكرر الصف 0
print("impossible values (income <= 0):", (df["income"] <= 0).sum())
print("category spelling variants:", sorted(df["city"].unique()))

# الشواذ بقاعدة z-score و IQR
inc = df["income"].dropna()
z = (inc - inc.mean()) / inc.std()
q1, q3 = inc.quantile([0.25, 0.75]); iqr = q3 - q1
print("z-score outliers:", inc[np.abs(z) > 3].tolist())
print("IQR outliers    :", inc[(inc < q1 - 1.5*iqr) | (inc > q3 + 1.5*iqr)].tolist())

# التنظيف
clean = df.drop_duplicates().copy()
clean["city"] = clean["city"].str.title()                          # توحيد الإملاء
clean.loc[clean["income"] <= 0, "income"] = np.nan                  # المستحيل → مفقود
clean.loc[clean["income"] > 1_000_000, "income"] = np.nan           # الشاذ الصارخ → مفقود (قرار!)
clean["income_missing"] = clean["income"].isna().astype(int)        # مؤشر الفقد
clean["income"] = clean["income"].fillna(clean["income"].median())  # تعويض بالوسيط
print(clean)'''


def render() -> None:
    lesson_header(LESSON)
    h2("قائمة الفحص", "The checklist")
    compare_table(["المشكلة", "English", "الاكتشاف", "الاستراتيجية"],
                  [("قيم مفقودة", "Missing values", "df.isna().sum()", "حذف الصف/العمود إن كانت قليلة أو عشوائية؛ تعويض بالوسيط/الوسط/الأكثر تكرارًا؛ + عمود مؤشر «كان مفقودًا»"),
                   ("تكرار", "Duplicates", "df.duplicated().sum()", "حذف؛ وتحقق أن الصفوف المكررة ليست في التدريب والاختبار معًا"),
                   ("شواذ", "Outliers", "z-score > 3، IQR، رسم صندوقي", "احتفظ إن كانت حقيقية (ثري فعلًا)؛ قصّ/حوّل (log) إن كانت تشوّه التحجيم؛ أزل إن كانت خطأ إدخال"),
                   ("قيم مستحيلة", "Incorrect values", "شروط منطقية: دخل سالب، عمر 250", "حوّل إلى مفقود ثم عالج"),
                   ("تباين إملائي", "Inconsistent categories", "unique() على النص", "توحيد الحالة والمسافات والمرادفات"),
                   ("تسميات خاطئة", "Label errors", "فحص عينات ذات خسارة عالية بعد نموذج أولي؛ مراجعة يدوية", "تصحيح أو حذف؛ لا تحفظ الضوضاء"),
                   ("ضوضاء", "Noise", "لا تُزال: تباين طبيعي", "اقبلها؛ تحدد الحد الأدنى للخسارة")],
                  ["rtl", "ltr", "code", "rtl"])
    definition("**الضوضاء** جزء من الهدف لا تفسره أي خاصية؛ لا تُنظَّف. **الخطأ** قيمة خاطئة قابلة للاكتشاف؛ يُنظَّف. الخلط بينهما يقود إما إلى حذف بيانات صحيحة أو الاحتفاظ بأخطاء.")
    code_lab(CodeLab(
        key="prep_clean", title_ar="فحص وتنظيف جدول صغير مليء بالمشكلات", code=CODE,
        before=Before(goal_ar="اكتشاف خمسة أنواع من المشكلات في جدول واحد ومعالجتها بقرارات صريحة.", stage_ar="إعداد البيانات ← الفحص والتنظيف.",
                      inputs_ar="10 صفوف فيها مفقود، تكرار، دخل سالب، دخل شاذ، وإملاء مختلف.", expected_ar="تقارير الفحص، ثم جدول نظيف بعمود مؤشر إضافي."),
        explain=[("9-13", "أربعة فحوص سريعة: مفقودات، تكرار، مستحيلات، إملاء. دقيقة واحدة توفر أيامًا."),
                 ("16-20", "قاعدتان للشواذ. z-score حساس للشاذ نفسه (يضخّم الانحراف)؛ IQR أكثر متانة."),
                 ("23-24", "حذف التكرار وتوحيد الإملاء (`algiers` → `Algiers`)."),
                 ("25-26", "المستحيل والشاذ الصارخ يصبحان مفقودًا — **قرار موثّق** لا حذفًا صامتًا."),
                 ("27-28", "مؤشر الفقد يحفظ معلومة «كان مفقودًا» (قد تكون مفيدة للتنبؤ)، ثم التعويض بالوسيط لأنه مقاوم للشواذ.")],
        run=run_printed(CODE),
        after_ar="- الدخل 9.9 مليون قد يكون حقيقيًا؛ حذفه قرار يجب توثيقه في التقرير.\n- التعويض بالوسيط **يجب أن يُحسب من بيانات التدريب فقط** عند وجود تقسيم — الدرس التالي عن التسريب يشرح لماذا.",
    ))
    research_note("في البيانات الاقتصادية الرسمية المفقودات نادرًا ما تكون عشوائية (شركات صغيرة لا تبلّغ، دول في أزمات). عمود المؤشر يحفظ هذه المعلومة، وحذف الصفوف قد يحيّز العينة.")
    debugging_note("خسارة `NaN` من أول خطوة؟ افحص `np.isnan(X).any()`: قيمة مفقودة واحدة تلوّث كل التدرج. الأطر لا تعالج NaN تلقائيًا.")
    common_mistake("تعويض المفقودات ثم حذف الشواذ بترتيب معكوس، أو حساب الوسيط قبل تحويل المستحيلات إلى مفقود: الترتيب مهم. مستحيل → مفقود → تعويض.")
    quiz("prep.clean", [
        Q("دخل = −50 في الجدول…", ["شذوذ حقيقي", "قيمة مستحيلة → مفقود", "ضوضاء"], 1, "الدخل لا يكون سالبًا."),
        Q("لماذا نضيف عمود «كان مفقودًا»؟", ["لزيادة الأعمدة", "لأن الفقد نفسه قد يحمل معلومة", "لتسريع التدريب"], 1, "الفقد غير عشوائي غالبًا."),
        Q("خسارة NaN من الخطوة الأولى، السبب الأرجح…", ["معدل تعلم صغير", "قيم مفقودة في X", "بيانات كثيرة"], 1, "NaN يلوث التدرج."),
        Q("الضوضاء…", ["تُزال بالتنظيف", "تُقبل وتحدد الحد الأدنى للخسارة", "خطأ تسمية"], 1, "لا يمكن التنبؤ بها."),
    ])
    takeaway("افحص خمسة أشياء دائمًا؛ مستحيل → مفقود → مؤشر → تعويض؛ الشاذ الحقيقي يُحتفظ به؛ الضوضاء ليست خطأ.")
    lesson_footer(LESSON, ["isna, duplicated, unique, شروط منطقية، IQR.", "قرارات التنظيف تُوثَّق.", "NaN واحدة تكفي لتخريب التدريب."])
