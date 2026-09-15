import streamlit as st

from components.callouts import common_mistake, definition, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.python.randomness_reproducibility",
    title_ar="العشوائية والبذرة والتكرارية",
    title_en="Random Numbers, Seeds & Reproducibility",
    module="foundations.python",
    order=9,
    prerequisites=["foundations.python.numpy_ndarray"],
    objectives_ar=[
        "فهم أن العشوائية في الحاسوب شبه عشوائية تحددها بذرة `seed`.",
        "معرفة أين تدخل العشوائية في التعلم العميق: تهيئة الأوزان، الخلط، التقسيم، Dropout.",
        "تثبيت البذرة بصورة صحيحة ومعرفة حدود التكرارية على GPU.",
    ],
    terms=["epoch", "batch"],
    difficulty="beginner",
    summary_ar="البذرة تجعل العشوائي قابلًا للتكرار؛ بدونها تختلف النتائج بين تشغيلين بنفس الكود.",
)

CODE = '''import numpy as np

rng_a = np.random.default_rng(seed={seed})
rng_b = np.random.default_rng(seed={seed})
print(rng_a.normal(size=3).round(3))
print(rng_b.normal(size=3).round(3))          # نفس القيم: نفس البذرة

rng_c = np.random.default_rng()               # بلا بذرة: تختلف كل تشغيل
print(rng_c.normal(size=3).round(3))

# أين تظهر العشوائية في التدريب؟
rng = np.random.default_rng({seed})
W_init = rng.normal(0, 0.1, size=(3, 2))       # تهيئة الأوزان
order = rng.permutation(8)                    # خلط الملاحظات قبل الدفعات
split = rng.uniform(size=8) < 0.75            # تقسيم تدريب/اختبار عشوائي
mask = rng.uniform(size=6) > 0.5              # Dropout: إسقاط وحدات عشوائيًا
print(W_init.round(3)); print(order); print(split); print(mask)

# التوزيعات الشائعة
print(rng.uniform(0, 1, 3).round(3))          # منتظم
print(rng.integers(0, 10, 5))                 # أعداد صحيحة
print(rng.choice(["a", "b", "c"], size=4, p=[0.6, 0.3, 0.1]))   # اختيار باحتمالات'''


def _controls() -> dict:
    return {"seed": st.number_input("seed", 0, 9999, 42, key="ctrl_py_seed")}


def render() -> None:
    lesson_header(LESSON)
    h2("عشوائي… لكن قابل للتكرار", "Random, yet reproducible")
    definition("مولد الأعداد العشوائية خوارزمية حتمية تبدأ من **بذرة** `seed`. نفس البذرة ← نفس التسلسل. بلا بذرة يؤخذ وقت النظام فتختلف النتائج كل مرة.")
    why("تجربتان بنفس الكود أعطتا دقة 86% و89%. هل التعديل أفاد أم البذرة اختلفت؟ بدون تثبيت البذرة لا يمكن الإجابة، وبالتالي لا يمكن نشر النتيجة.")
    table(
        ["أين", "ماذا يُعشوائى", "أثر البذرة"],
        [("تهيئة الأوزان", "القيم الأولية لـ W و b", "مسار تدريب مختلف بالكامل"),
         ("الخلط", "ترتيب الملاحظات في كل حقبة", "دفعات مختلفة ← تدرجات مختلفة"),
         ("التقسيم", "أي الملاحظات في التدريب/التحقق/الاختبار", "تقدير أداء مختلف"),
         ("Dropout", "أي الوحدات تُسقط في كل خطوة", "تنظيم مختلف"),
         ("زيادة البيانات", "التحويلات المطبقة على الصور", "أمثلة مختلفة")],
        ["rtl", "rtl", "rtl"],
    )
    code_lab(CodeLab(
        key="py_seed", title_ar="البذرة ومواضع العشوائية", code=CODE, template=True, defaults={"seed": 42},
        before=Before(goal_ar="إثبات أن نفس البذرة تعطي نفس الأرقام، ورؤية المواضع الأربعة التي تدخل فيها العشوائية إلى التدريب.",
                      stage_ar="أساسيات ← التكرارية.", inputs_ar="بذرة عددية.", expected_ar="سطران متطابقان، سطر مختلف، ثم أوزان أولية وترتيب مخلوط وقناع تقسيم وقناع Dropout."),
        explain=[("3-6", "مولّدان بنفس البذرة ← نفس التسلسل تمامًا. هذا هو أساس التكرارية."),
                 ("8-9", "بلا بذرة: نتائج مختلفة في كل تشغيل — غيّر البذرة أعلاه وشغّل مرتين لترى الفرق (السطر الثالث فقط يتغير بين التشغيلين)."),
                 ("12-16", "المواضع الأربعة في التدريب. لاحظ أن كلها تُسحب من **نفس المولّد** بالترتيب؛ تغيير ترتيب الاستدعاءات يغيّر النتائج حتى بنفس البذرة."),
                 ("19-21", "توزيعات شائعة: منتظم، أعداد صحيحة، اختيار موزون — أساس أخذ العينات وتوليد البيانات الاصطناعية.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- السطران الأولان متطابقان؛ الثالث يتغير مع كل تشغيل.\n- `permutation(8)` هو ما يفعله `shuffle=True` كل حقبة.",
    ))
    research_note("على `GPU` بعض العمليات غير حتمية بطبيعتها (ترتيب الجمع المتوازي). تثبيت البذرة يقلل التباين لكنه لا يضمن تطابقًا بتّيًا؛ في البحث أبلغ عن متوسط وانحراف عبر عدة بذور بدل نتيجة واحدة.")
    common_mistake("تثبيت بذرة `NumPy` فقط ونسيان بذرة إطار العمل (`tf.random.set_seed`, `torch.manual_seed`) وبذرة بايثون `random.seed`. كل مكتبة لها مولّدها.")
    quiz("py.seed", [
        Q("نفس البذرة ونفس ترتيب الاستدعاءات يعطي…", ["نفس الأرقام", "أرقامًا مختلفة", "خطأ"], 0, "الحتمية."),
        Q("أي هذه ليس موضع عشوائية في التدريب؟", ["تهيئة الأوزان", "الخلط", "حساب MSE"], 2, "الخسارة حتمية."),
        Q("لماذا يُبلَّغ عن متوسط عدة بذور في البحث؟", ["لأن GPU بطيء", "لأن نتيجة واحدة قد تكون حظًا", "لأن البذرة غير مهمة"], 1, "التباين بين البذور حقيقي."),
    ])
    takeaway("العشوائية حتمية بالبذرة. ثبّت بذور كل المكتبات، وأبلغ عن عدة بذور، ولا تتوقع تطابقًا بتّيًا على GPU.")
    lesson_footer(LESSON, ["default_rng(seed) للتكرارية.", "أربعة مواضع: تهيئة، خلط، تقسيم، Dropout.", "بذرة لكل مكتبة."])
