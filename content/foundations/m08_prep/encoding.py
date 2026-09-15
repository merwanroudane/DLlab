import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.prep.encoding",
    title_ar="ترميز الفئات: one-hot والترتيبي والتضمين",
    title_en="Encoding Categories: One-hot, Ordinal & Embeddings",
    module="foundations.prep",
    order=2,
    prerequisites=["foundations.prep.inspection_cleaning", "foundations.data.variable_types"],
    objectives_ar=[
        "ترميز الفئات الاسمية بـ one-hot والترتيبية بأعداد مرتبة، ومعرفة متى يُفضَّل التضمين.",
        "التعامل مع فئات جديدة في الاختبار لم تظهر في التدريب.",
        "حساب عدد الأعمدة الناتج وأثره على شكل المدخل.",
    ],
    terms=["categorical", "feature", "shape"],
    difficulty="beginner",
    summary_ar="اسمي → one-hot (عمود لكل فئة)، ترتيبي → أعداد مرتبة، فئات كثيرة → تضمين. fit على التدريب فقط.",
)

CODE = '''import numpy as np, pandas as pd

train = pd.DataFrame({"city": ["Algiers", "Oran", "Algiers", "Constantine"], "rating": ["low", "high", "medium", "low"]})
test  = pd.DataFrame({"city": ["Oran", "Annaba"], "rating": ["medium", "high"]})   # Annaba جديدة!

# 1) one-hot يدويًا مع تثبيت الفئات من التدريب
cities = sorted(train["city"].unique())                   # ["Algiers", "Constantine", "Oran"]
def one_hot(col, categories):
    return pd.DataFrame({f"city_{c}": (col == c).astype(int) for c in categories})
print(one_hot(train["city"], cities)); print(one_hot(test["city"], cities))   # Annaba → صف أصفار

# 2) ترتيبي: خريطة صريحة تحفظ الترتيب
order = {"low": 0, "medium": 1, "high": 2}
print(train["rating"].map(order).tolist(), test["rating"].map(order).tolist())

# 3) الشكل النهائي
X_train = np.column_stack([one_hot(train["city"], cities).to_numpy(), train["rating"].map(order).to_numpy()]).astype(np.float32)
print("X_train.shape =", X_train.shape, "  (n, 3 one-hot + 1 ordinal)")

# 4) pandas.get_dummies السريع — احذر اختلاف الأعمدة بين التدريب والاختبار
print(pd.get_dummies(train["city"]).columns.tolist())
print(pd.get_dummies(test["city"]).columns.tolist(), " ← أعمدة مختلفة! استخدم categories ثابتة")'''


def render() -> None:
    lesson_header(LESSON)
    h2("لماذا الترميز؟", "Why encode?")
    why("الشبكة تضرب وتجمع؛ `Oran` لا يُضرب في وزن. ورقم اعتباطي مثل `Oran = 2` يوهم النموذج بترتيب ومسافة لا وجود لهما (درس أنواع المتغيرات).")
    compare_table(["الطريقة", "English", "يناسب", "الناتج", "المزايا / العيوب"],
                  [("عمود لكل فئة", "One-hot", "اسمي بعدد فئات صغير (< ~20)", "k أعمدة من 0/1", "بسيط وصادق؛ يتضخم مع الفئات الكثيرة"),
                   ("أعداد مرتبة", "Ordinal", "ترتيبي (low < medium < high)", "عمود واحد", "يحفظ الترتيب؛ يفترض مسافات متساوية"),
                   ("تضمين", "Embedding", "اسمي بفئات كثيرة (مدن، منتجات، كلمات)", "متجه كثيف بطول d يتعلمه النموذج", "مضغوط ويتعلم تشابه الفئات؛ يحتاج بيانات وطبقة Embedding"),
                   ("هدف/تكرار", "Target / frequency", "حالات خاصة في ML التقليدي", "عمود واحد", "خطر تسريب مع ترميز الهدف")],
                  ["rtl", "ltr", "rtl", "rtl", "rtl"])
    definition("**one-hot**: متجه بطول عدد الفئات فيه 1 في موضع الفئة و0 في الباقي. **الترتيبي**: عدد صحيح يحفظ الترتيب. **التضمين**: جدول بحث `(k, d)` يتعلمه النموذج فتصبح كل فئة متجهًا كثيفًا.")
    code_lab(CodeLab(
        key="prep_encode", title_ar="one-hot وترتيبي مع فئة جديدة في الاختبار", code=CODE,
        before=Before(goal_ar="ترميز عمودين بطريقتين مع تثبيت الفئات من التدريب، ورؤية مشكلة الفئة غير المرئية.", stage_ar="إعداد البيانات ← الترميز.",
                      inputs_ar="جدول تدريب من 4 صفوف واختبار من صفين فيه مدينة جديدة.", expected_ar="مصفوفتا one-hot بنفس الأعمدة، Annaba صف أصفار، ترتيبي 0/1/2، `X_train.shape = (4, 4)`."),
        explain=[("6-9", "الفئات تُحدَّد من **التدريب** وتُثبَّت؛ الاختبار يُرمَّز بنفس القائمة. `Annaba` غير معروفة → أصفار (أو عمود «other»)."),
                 ("12-13", "الترتيبي بخريطة صريحة — لا تترك للمكتبة اختيار الترتيب أبجديًا."),
                 ("16-17", "ضمّ الأعمدة: 3 من one-hot + 1 ترتيبي = 4 مدخلات لأول طبقة."),
                 ("20-21", "`get_dummies` مريحة لكنها تعطي أعمدة مختلفة إن اختلفت الفئات بين المجموعتين — خطأ شكل لاحقًا.")],
        run=run_printed(CODE),
        after_ar="- عدد مدخلات الطبقة الأولى = مجموع أعمدة الترميز؛ سجّله لأن `input_shape` يعتمد عليه.\n- فئة جديدة في الإنتاج ليست خطأ برمجيًا بل قرار تصميمي: أصفار، «other»، أو تضمين لفئة مجهولة.",
    ))
    practical_note("في `Keras` يمكن ترميز النص داخل النموذج بطبقات `StringLookup` + `CategoryEncoding` أو `Embedding`؛ الفكرة نفسها: قاموس يُبنى من التدريب.")
    common_mistake("one-hot لعمود بـ 5000 فئة (رمز بريدي): 5000 عمود شبه فارغة. الحل تضمين بطول 8–32.")
    quiz("prep.encode", [
        Q("عمود `sector` بـ 8 قطاعات بلا ترتيب…", ["ترتيبي", "one-hot", "كما هو"], 1, "اسمي قليل الفئات."),
        Q("فئة في الاختبار لم تظهر في التدريب: الترميز الصحيح…", ["أضف عمودًا جديدًا", "صف أصفار أو فئة «other» بثبات الأعمدة", "احذف الصف دائمًا"], 1, "ثبات الشكل."),
        Q("3 أعمدة one-hot + 2 عددية: `input_shape`…", ["(3,)", "(5,)", "(2,)"], 1, "مجموع الأعمدة.", kind="shape"),
        Q("100k كلمة مختلفة في نصوص: الترميز…", ["one-hot", "تضمين", "ترتيبي"], 1, "فئات كثيرة."),
    ])
    takeaway("اسمي → one-hot، ترتيبي → أعداد مرتبة، كثير → تضمين. الفئات تُثبَّت من التدريب، والاختبار يتبعها.")
    lesson_footer(LESSON, ["لا أرقام اعتباطية للفئات الاسمية.", "ثبّت قائمة الفئات من التدريب.", "عدد الأعمدة الناتج = input_shape."])
