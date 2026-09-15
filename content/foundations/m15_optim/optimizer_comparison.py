import streamlit as st

from components.callouts import practical_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.optim.optimizer_comparison",
    title_ar="مقارنة المحسّنات وسباق المحسّنات",
    title_en="Optimizer Comparison & the Optimizer Race",
    module="foundations.optim",
    order=6,
    prerequisites=["foundations.optim.rmsprop_adam"],
    objectives_ar=["جدول مقارنة نهائي: المعادلة، الحالة، المعلمات الفائقة، نقاط القوة والضعف.", "تشغيل سباق حقيقي على شبكة صغيرة ومقارنة المنحنيات بعدل."],
    terms=["optimizer"],
    labs=["labs.optimizer_race"],
    difficulty="intermediate",
    summary_ar="جدول واحد + سباق حقيقي. الحكم: بالتحديثات أو الزمن، وعلى التحقق لا التدريب فقط.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الجدول النهائي", "The final table")
    compare_table(["المحسّن", "قاعدة التحديث", "الحالة لكل معلمة", "المعلمات الفائقة", "القوة", "الضعف"],
                  [("SGD", "θ − η g", "لا شيء", "η", "بسيط، يعمّم جيدًا مع الجدولة", "بطيء في الوديان، حساس لـ η"),
                   ("Momentum", "v = βv + g; θ − ηv", "v", "η, β", "يعبر الوديان والسروج", "قد يتجاوز القاع"),
                   ("Nesterov", "g عند θ − ηβv", "v", "η, β", "تذبذب أقل", "نفس Momentum تقريبًا"),
                   ("RMSprop", "θ − η g/√s", "s", "η, β₂", "مقياس متكيف", "بلا زخم؛ حساس لـ ε أحيانًا"),
                   ("Adam", "θ − η m̂/√ŝ", "m, s", "η, β₁, β₂", "قوي افتراضيًا، سريع", "قد يعمّم أقل من SGD؛ يحتاج تخفيض η أخيرًا"),
                   ("AdamW", "Adam + decay منفصل", "m, s", "η, β₁, β₂, λ", "تنظيم صحيح", "معلمة إضافية")],
                  ["ltr", "code", "code", "code", "rtl", "rtl"])
    h2("قواعد المقارنة العادلة", "Fair comparison rules")
    st.markdown("""
1. **نفس البذرة والبيانات والبنية** لكل محسّن.
2. **اضبط η لكل محسّن على حدة** (مسح صغير) — مقارنة Adam بـ 1e-3 مع SGD بـ 1e-3 ظالمة لـ SGD.
3. قارن بـ **عدد التحديثات أو الزمن**، لا بالحقب فقط إن اختلف حجم الدفعة.
4. انظر إلى **خسارة التحقق** والمقياس، لا خسارة التدريب وحدها.
5. كرر بـ 3 بذور على الأقل قبل إعلان فائز.
""")
    practical_note("سباق المحسّنات في المعمل يطبق هذه القواعد على شبكة صغيرة ببيانات الهلالين: اختر η لكل محسّن، شغّل، وقارن المنحنيات.")
    st.button("افتح سباق المحسّنات", icon=":material/science:", type="primary", on_click=goto, args=("labs.optimizer_race",), key="race_btn")
    quiz("optim.cmp", [
        Q("أي محسّن يحتفظ بحالتين لكل معلمة؟", ["SGD", "Momentum", "Adam"], 2, "m و s."),
        Q("مقارنة عادلة تتطلب…", ["نفس η للجميع", "ضبط η لكل محسّن على حدة", "حقبة واحدة"], 1, "كل محسّن نطاقه."),
        Q("نقطة بداية عملية لأغلب المسائل…", ["SGD بلا زخم", "Adam بـ 1e-3", "RMSprop بـ 0.1"], 1, "الافتراضي."),
    ])
    takeaway("اعرف الجدول، وقارن بعدل: نفس البذرة، η مضبوط لكل محسّن، تحديثات لا حقب، تحقق لا تدريب.")
    lesson_footer(LESSON, ["ستة محسّنات في جدول.", "خمس قواعد للمقارنة.", "المعمل ينفذها."])
