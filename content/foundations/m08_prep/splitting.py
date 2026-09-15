import streamlit as st

from components.callouts import common_mistake, definition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.prep.splitting",
    title_ar="الخلط والتقسيم والتقسيم الطبقي والزمني",
    title_en="Shuffling, Splitting, Stratification & Temporal Splits",
    module="foundations.prep",
    order=4,
    prerequisites=["foundations.prep.scaling", "foundations.ml.baseline_generalization"],
    objectives_ar=[
        "تنفيذ تقسيم عشوائي وطبقي وزمني من الصفر وفهم متى يُستخدم كل منها.",
        "التعامل مع المجموعات (نفس العميل في صفوف عدة) لمنع تسريب الهوية.",
    ],
    terms=["seed", "observation", "target"],
    labs=["labs.data_split_lab"],
    difficulty="beginner",
    summary_ar="اخلط ثم قسّم (طبقيًا للفئات النادرة)؛ للسلاسل الزمنية لا خلط: الماضي تدريب والمستقبل اختبار؛ المجموعات تبقى معًا.",
)

CODE = '''import numpy as np
rng = np.random.default_rng({seed})
n = 200
y = (rng.uniform(size=n) < 0.08).astype(int)      # فئة نادرة 8%

# 1) تقسيم عشوائي بسيط
idx = rng.permutation(n); cut = int(0.8 * n)
tr, te = idx[:cut], idx[cut:]
print(f"random   : train rate={{y[tr].mean():.3f}} test rate={{y[te].mean():.3f}} (n_test_pos={{y[te].sum()}})")

# 2) تقسيم طبقي: نفس النسبة في كل مجموعة
tr_s, te_s = [], []
for cls in (0, 1):
    members = rng.permutation(np.where(y == cls)[0])
    k = int(0.8 * len(members))
    tr_s += members[:k].tolist(); te_s += members[k:].tolist()
tr_s, te_s = np.array(tr_s), np.array(te_s)
print(f"stratified: train rate={{y[tr_s].mean():.3f}} test rate={{y[te_s].mean():.3f}} (n_test_pos={{y[te_s].sum()}})")

# 3) زمني: بلا خلط
t = np.arange(n)                                   # ترتيب زمني
tr_t, te_t = t[:cut], t[cut:]
print("temporal: train covers", tr_t.min(), "-", tr_t.max(), " test covers", te_t.min(), "-", te_t.max())

# 4) بالمجموعة: كل صفوف العميل في مجموعة واحدة
customer = rng.integers(0, 50, n)                  # 50 عميلًا، عدة صفوف لكل واحد
groups = rng.permutation(50); test_groups = set(groups[:10])
te_g = np.array([i for i in range(n) if customer[i] in test_groups]); tr_g = np.setdiff1d(np.arange(n), te_g)
print("group split: shared customers between train/test =", len(set(customer[tr_g]) & set(customer[te_g])))'''


def _controls() -> dict:
    return {"seed": st.number_input("seed", 0, 999, 3, key="ctrl_split_seed")}


def render() -> None:
    lesson_header(LESSON)
    h2("أربع طرق للتقسيم", "Four ways to split")
    compare_table(["الطريقة", "English", "متى", "الخطر الذي تمنعه"],
                  [("عشوائي بعد خلط", "Random (shuffled)", "جدولية مستقلة الصفوف", "انحياز الترتيب في الملف (مرتب حسب التاريخ أو الفئة)"),
                   ("طبقي", "Stratified", "تصنيف بفئات غير متوازنة", "مجموعة اختبار بلا (أو بقليل من) الفئة النادرة"),
                   ("زمني", "Temporal", "سلاسل زمنية، أي بيانات لها ترتيب زمني", "تسريب المستقبل إلى التدريب"),
                   ("بالمجموعة", "Group", "صفوف متعددة لنفس الكيان (عميل، مريض، شركة)", "حفظ هوية الكيان بدل تعلم النمط")],
                  ["rtl", "ltr", "rtl", "rtl"])
    definition("**الخلط** `Shuffle` يعيد ترتيب الصفوف عشوائيًا قبل التقسيم (وقبل كل حقبة أثناء التدريب). **التقسيم الطبقي** يحافظ على نسب الفئات في كل مجموعة. **الزمني** يقطع عند نقطة زمنية.")
    code_lab(CodeLab(
        key="prep_split", title_ar="أربعة تقسيمات من الصفر", code=CODE, template=True, defaults={"seed": 3},
        before=Before(goal_ar="مقارنة نسبة الفئة النادرة بين تقسيم عشوائي وطبقي، وتنفيذ تقسيم زمني وآخر بالمجموعة.", stage_ar="إعداد البيانات ← التقسيم.",
                      inputs_ar="200 ملاحظة بفئة نادرة 8%، مؤشر زمني، ومعرّفات عملاء.", expected_ar="العشوائي قد يعطي نسبة مختلفة في الاختبار (جرّب بذورًا)؛ الطبقي يثبّتها؛ الزمني فترتان منفصلتان؛ المجموعة صفر عملاء مشتركين."),
        explain=[("6-9", "الخلط بـ `permutation` ثم قطع 80/20. النسبة في الاختبار تتذبذب مع البذرة لأن الفئة النادرة قليلة."),
                 ("12-18", "الطبقي: قسّم كل فئة على حدة بنفس النسبة ثم اجمع. الاختبار يحتوي دائمًا 8% تقريبًا."),
                 ("21-23", "الزمني: بلا خلط؛ الاختبار كله بعد التدريب زمنيًا."),
                 ("26-29", "بالمجموعة: نختار عملاء للاختبار لا صفوفًا؛ لا عميل مشترك بين المجموعتين.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- غيّر البذرة: التقسيم العشوائي قد يعطي اختبارًا بـ 2 أو 6 موجبات من 40 — تقدير دقة غير مستقر. الطبقي يعطي 3 دائمًا.\n- في التقسيم بالمجموعة عدد الصفوف في الاختبار غير ثابت؛ هذا مقبول.",
    ))
    research_note("بيانات اللوحة (Panel) الاقتصادية تجمع الزمن والمجموعة: دول × سنوات. التقسيم الصحيح يعتمد على السؤال: التنبؤ بسنوات جديدة (زمني) أم بدول جديدة (مجموعة)؟ الإجابة تغيّر كل شيء.")
    st.button("افتح معمل التقسيم", icon=":material/science:", on_click=go, args=("labs.data_split_lab",), key="split_lab_btn")
    common_mistake("ملف مرتب حسب التاريخ أو الفئة يُقسَّم بلا خلط: التدريب كله من فئة والاختبار من أخرى. اخلط أولًا (إلا في الزمني).")
    quiz("prep.split", [
        Q("فئة نادرة 3%: التقسيم المناسب…", ["عشوائي", "طبقي", "زمني"], 1, "ضمان وجودها في الاختبار."),
        Q("أسعار يومية للتنبؤ بالغد…", ["اخلط ثم قسّم", "الماضي تدريب، المستقبل اختبار", "طبقي"], 1, "لا تسريب مستقبل."),
        Q("5 صفوف لكل عميل موزعة عشوائيًا بين التدريب والاختبار…", ["صحيح", "تسريب هوية: استخدم تقسيمًا بالمجموعة", "أفضل للتعميم"], 1, "النموذج يحفظ العميل."),
    ])
    takeaway("اخلط ثم قسّم؛ طبقيًا للفئات النادرة؛ زمنيًا للسلاسل بلا خلط؛ بالمجموعة عند تعدد صفوف الكيان.")
    lesson_footer(LESSON, ["الترتيب في الملف ليس عشوائيًا.", "الطبقي يثبّت النسب.", "الزمن والمجموعة يحددان شكل التسريب المحتمل."])
