import streamlit as st

from components.callouts import common_mistake, definition, practical_note, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.prep.imbalance_augmentation",
    title_ar="عدم توازن الفئات، إعادة أخذ العينات، وزيادة البيانات",
    title_en="Class Imbalance, Resampling & Data Augmentation",
    module="foundations.prep",
    order=6,
    prerequisites=["foundations.prep.splitting", "foundations.ml.baseline_generalization"],
    objectives_ar=[
        "تشخيص عدم التوازن وأثره على الدقة والخسارة.",
        "مقارنة الحلول: أوزان الفئات، إعادة أخذ العينات، تغيير العتبة، مقاييس مناسبة.",
        "فهم زيادة البيانات كتوليد أمثلة صحيحة إضافية (الصور خاصة) لتحسين التعميم.",
    ],
    terms=["target", "loss", "probability"],
    difficulty="intermediate",
    summary_ar="عدم التوازن يخدع الدقة؛ الحلول: أوزان الفئات، إعادة العينات، العتبة، ومقاييس Precision/Recall. الزيادة تولّد أمثلة صحيحة إضافية.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
n = 3000
y = (rng.uniform(size=n) < 0.05).astype(int)               # 5% إيجابي
x = rng.normal(size=n) + 1.2 * y                            # خاصية واحدة مفيدة

# نموذج بسيط: احتمال = sigmoid(a*x + b) مع تدريب سريع
def train(x, y, w_pos=1.0, epochs=400, eta=0.1):
    a, b = 0.0, 0.0
    wts = np.where(y == 1, w_pos, 1.0)                      # وزن الفئة النادرة
    for _ in range(epochs):
        p = 1/(1+np.exp(-(a*x + b))); g = wts * (p - y)
        a -= eta * (g * x).mean(); b -= eta * g.mean()
    return a, b

def report(name, a, b, thr=0.5):
    p = 1/(1+np.exp(-(a*x + b))); pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    acc = (pred == y).mean(); prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f"{name:<26} acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} predicted_pos={pred.sum()}")

print("baseline: always 0        acc=%.3f" % (1 - y.mean()))
a, b = train(x, y);            report("unweighted, thr=0.5", a, b)
report("unweighted, thr=0.15", a, b, thr=0.15)
a2, b2 = train(x, y, w_pos=19); report("class-weighted (19x)", a2, b2)'''


def render() -> None:
    lesson_header(LESSON)
    h2("المشكلة", "The problem")
    definition("**عدم التوازن** `Class imbalance`: فئة نادرة جدًا (تعثر 5%، احتيال 0.1%). النموذج يستطيع تحقيق دقة عالية بتجاهل الفئة النادرة تمامًا — وهي غالبًا الفئة التي تهمنا.")
    compare_table(["الحل", "English", "الفكرة", "ملاحظات"],
                  [("أوزان الفئات", "Class weights", "ضرب خسارة الفئة النادرة بعامل (مثلًا نسبة الأغلبية/النادرة)", "الأبسط في Keras (`class_weight`) وPyTorch (`pos_weight`)؛ لا يغيّر البيانات"),
                   ("زيادة العينات", "Oversampling", "تكرار/توليد أمثلة من النادرة (SMOTE)", "على التدريب فقط؛ خطر حفظ التكرارات"),
                   ("تقليل العينات", "Undersampling", "حذف جزء من الأغلبية", "يهدر بيانات؛ مفيد مع بيانات ضخمة"),
                   ("تغيير العتبة", "Threshold tuning", "خفض 0.5 حسب كلفة الأخطاء", "لا تدريب إضافي؛ يُضبط على التحقق"),
                   ("مقاييس مناسبة", "Precision / Recall / F1 / AUC", "لا تعتمد على الدقة", "ضروري بغض النظر عن الحل")],
                  ["rtl", "ltr", "rtl", "rtl"])
    code_lab(CodeLab(
        key="prep_imb", title_ar="خط أساس مضلل، ثم عتبة، ثم أوزان فئات", code=CODE,
        before=Before(goal_ar="رؤية أن الدقة تخدع مع 5% إيجابي، ثم تحسين الاستدعاء بتغيير العتبة وبأوزان الفئات.", stage_ar="إعداد البيانات ← عدم التوازن ← التقييم.",
                      inputs_ar="3000 ملاحظة، 5% إيجابي، خاصية واحدة.", expected_ar="خط أساس 95%؛ النموذج غير الموزون بدقة عالية واستدعاء ضعيف؛ العتبة والأوزان يرفعان الاستدعاء على حساب الدقة والصحة (precision)."),
        explain=[("3-5", "5% فقط إيجابي؛ الخاصية تميز جزئيًا."), ("8-14", "انحدار لوجستي مع وزن اختياري للفئة النادرة يضرب تدرجها."),
                 ("16-21", "المقاييس: الدقة (كل الصواب)، الصحة (من الإيجابيات المتنبأة كم صحيح)، الاستدعاء (من الإيجابيات الحقيقية كم التقطنا)."),
                 ("23-26", "أربعة سيناريوهات. لاحظ `predicted_pos`: كم ملاحظة اعتبرها النموذج إيجابية.")],
        run=run_printed(CODE),
        after_ar="- خط الأساس 95% دقة بلا نموذج. النموذج غير الموزون بعتبة 0.5 قد لا يتنبأ بأي إيجابي تقريبًا.\n- خفض العتبة أو الأوزان يرفع الاستدعاء (نلتقط المتعثرين) ويخفض الصحة (إنذارات كاذبة أكثر): مقايضة تحددها كلفة كل خطأ اقتصاديًا.",
    ))
    h2("زيادة البيانات", "Data augmentation")
    definition("**زيادة البيانات** `Augmentation`: توليد أمثلة جديدة صحيحة بتحويلات تحفظ التسمية: للصور (قلب، تدوير، قص، إضاءة)، للنص (مرادفات، حذف كلمات)، للسلاسل (ضوضاء صغيرة، إزاحة). تزيد التنوع وتقلل فرط التخصيص — تُطبَّق على التدريب فقط.")
    practical_note("في الصور: صورة قطة مقلوبة أفقيًا ما زالت قطة؛ رقم 6 مقلوبًا رأسيًا **ليس** 6. التحويل يجب أن يحفظ المعنى في مجالك. في الجداول الزيادة أصعب وأقل شيوعًا.")
    research_note("في البيانات الاقتصادية النادرة (أزمات، تعثر سيادي) لا توجد «زيادة» حقيقية؛ الحلول الواقعية: أوزان الفئات، مقاييس مناسبة، وقبول عدم اليقين في التقرير.")
    common_mistake("زيادة العينات (oversampling) **قبل** التقسيم: نسخ من نفس الملاحظة تقع في التدريب والاختبار ⇒ تسريب وأداء خيالي. الزيادة بعد التقسيم وعلى التدريب فقط.")
    quiz("prep.imb", [
        Q("احتيال 0.2%؛ دقة النموذج 99.8%…", ["ممتاز", "يساوي خط الأساس على الأرجح", "فرط تخصيص"], 1, "الفئة النادرة مُهملة."),
        Q("لرفع الاستدعاء دون إعادة تدريب…", ["زد الحقب", "اخفض العتبة", "احذف خصائص"], 1, "العتبة قرار."),
        Q("قلب صورة رقم 6 رأسيًا كزيادة بيانات…", ["صحيح", "خطأ: يغيّر التسمية", "لا أثر"], 1, "يصبح 9."),
        Q("SMOTE قبل التقسيم…", ["أفضل", "تسريب", "لا فرق"], 1, "نسخ في الاختبار."),
    ])
    takeaway("الدقة تخدع مع عدم التوازن. أوزان، عينات، عتبة، ومقاييس Precision/Recall. الزيادة تحفظ التسمية وتُطبَّق على التدريب فقط.")
    lesson_footer(LESSON, ["قارن دائمًا بخط الأساس.", "العتبة مقايضة اقتصادية.", "الزيادة بعد التقسيم."])
