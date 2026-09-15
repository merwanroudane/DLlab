import streamlit as st

from components.callouts import practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.start.self_check",
    title_ar="فحص ذاتي للمتطلبات القبلية",
    title_en="Prerequisite Self-check",
    module="foundations.start",
    order=3,
    prerequisites=["foundations.start.learning_roadmap"],
    objectives_ar=[
        "تقييم صادق لمستواك في البيانات وبايثون والرياضيات والتعلم الآلي.",
        "الحصول على توصية بنقطة البداية المناسبة داخل أكاديمية الأسس.",
    ],
    difficulty="beginner",
    summary_ar="استبيان قصير يحدد أين تبدأ في أكاديمية الأسس.",
)

# (question, weight-category, options as (label, score))
ITEMS = [
    ("هل تستطيع شرح الفرق بين «الملاحظة» و«الخاصية» و«الهدف» في جدول بيانات؟", "data"),
    ("هل تعرف ما الذي تعنيه `X.shape == (120, 4)`؟", "data"),
    ("هل تستطيع كتابة دالة بايثون تأخذ قائمة وتعيد متوسطها؟", "python"),
    ("هل تعرف الفرق بين قائمة بايثون و`numpy.ndarray`؟", "python"),
    ("هل تستطيع حساب مشتقة $f(x)=x^2$ عند $x=3$؟", "math"),
    ("هل تعرف ما هو الضرب النقطي (Dot Product) لمتجهين؟", "math"),
    ("هل تعرف لماذا نقسم البيانات إلى تدريب وتحقق واختبار؟", "ml"),
    ("هل تستطيع تفسير منحنى خسارة يهبط على بيانات التدريب ويصعد على بيانات التحقق؟", "ml"),
]
OPTIONS = ["لا", "تقريبًا", "نعم بثقة"]
CATEGORY_AR = {"data": "البيانات", "python": "بايثون", "math": "الرياضيات", "ml": "التعلم الآلي"}
RECOMMEND = {
    "data": ("الوحدة 1 — أسس البيانات", "foundations.data"),
    "python": ("الوحدة 2 — بايثون للتعلم العميق", "foundations.data"),
    "math": ("الوحدات 3–5 — الرياضيات والجبر الخطي والتفاضل", "foundations.data"),
    "ml": ("الوحدة 7 — أسس التعلم الآلي", "foundations.data"),
}


def render() -> None:
    lesson_header(LESSON)
    h2("أجب بصدق", "Answer honestly")
    st.markdown("لا توجد إجابة «خاطئة». الهدف تحديد نقطة البداية التي توفر عليك وقتًا وإحباطًا.")
    with st.form("self_check_form", border=True):
        answers = []
        for i, (q, cat) in enumerate(ITEMS):
            st.markdown(f"**{i + 1}.** {q}")
            answers.append(
                st.radio("إجابتك", OPTIONS, index=None, key=f"selfcheck_{i}", horizontal=True,
                         label_visibility="collapsed")
            )
        submitted = st.form_submit_button("احسب التوصية", type="primary", icon=":material/insights:")
    if submitted:
        if any(a is None for a in answers):
            st.warning("أجب عن كل الأسئلة أولًا.", icon="⚠️")
            return
        scores: dict[str, list[int]] = {}
        for (q, cat), a in zip(ITEMS, answers):
            scores.setdefault(cat, []).append(OPTIONS.index(a))
        st.markdown("### نتيجتك")
        weakest = None
        for cat, vals in scores.items():
            pct = round(100 * sum(vals) / (2 * len(vals)))
            st.progress(pct / 100, text=f"{CATEGORY_AR[cat]}: {pct}%")
            if weakest is None or pct < weakest[1]:
                weakest = (cat, pct)
        cat, pct = weakest
        if pct >= 75:
            st.success("مستواك جيد في كل المحاور. ابدأ من الوحدة 9 (من الانحدار إلى الخلية العصبية) "
                       "مع مراجعة سريعة للوحدة 17.", icon="✅")
        else:
            label, route = RECOMMEND[cat]
            st.info(f"أضعف محور لديك هو **{CATEGORY_AR[cat]}**. نوصي بالبدء من: **{label}**.", icon="🧭")
            st.button("اذهب إلى نقطة البداية الموصى بها", type="primary", on_click=go, args=(route,),
                      key="selfcheck_go")
    practical_note("يمكنك إعادة الفحص في أي وقت؛ النتيجة لا تُحفظ خارج جلستك الحالية.")
    takeaway("ابدأ من أضعف محور لديك، لا من أول صفحة في الفهرس.")
    lesson_footer(LESSON)
