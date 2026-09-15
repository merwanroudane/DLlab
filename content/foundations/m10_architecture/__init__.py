from core.models import Module

MODULE = Module(
    id="foundations.architecture",
    section="foundations",
    title_ar="الوحدة 10 — بنية الشبكة العصبية",
    title_en="Module 10 — Neural Network Architecture",
    order=10,
    icon=":material/account_tree:",
    prerequisites=["foundations.neuron"],
    purpose_ar="طبقة الإدخال والمخفية والإخراج، الروابط والأوزان والانحيازات، العمق والعرض، الطبقة الكثيفة، وحساب عدد المعلمات لأي شبكة.",
    why_ar="`model.summary()` يطبع أشكالًا وأعدادًا. من يستطيع حسابها يدويًا يعرف أن شبكته مبنية كما قصد؛ ومن لا يستطيع لا يكتشف خطأ `input_shape` إلا بعد ساعات.",
    objectives_ar=["رسم شبكة كثيفة وتسمية طبقاتها وحساب شكل كل طبقة.", "حساب عدد المعلمات لأي شبكة كثيفة يدويًا ومطابقته بالملخص.", "فهم مقايضات العمق مقابل العرض تمهيدًا."],
    challenges_ar=["عدّ طبقة الإدخال كطبقة ذات معلمات.", "نسيان الانحيازات في العدّ.", "الخلط بين عدد الوحدات وعدد المعلمات."],
)

LESSON_MODULES = ["layers", "depth_width_dense", "parameter_count", "build_and_read"]
