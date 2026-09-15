from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w15.overview",
    title_ar="نظرة عامة على الأسبوع 15 والاختبار القبلي",
    title_en="Week 15 Overview & Pre-test",
    module="course.w15",
    order=1,
    prerequisites=["course.w14.notebook_template"],
    objectives_ar=["خريطة الأسبوع الأخير.", "اختبار قبلي على العرض والتقييم."],
    terms=["reproducibility"],
    difficulty="beginner",
    summary_ar="العرض ← تقييم الأداء وتحليل النتائج ← نقاط الضعف والمقارنة ← مقترحات التطوير والمناقشة ← الرمز النهائي والاستنساخ.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="15",
        steps=["Presentation", "Performance evaluation", "Results analysis", "Weakness diagnosis", "Choice comparison", "Improvement proposals", "Discussion", "Final rubric", "Post-test"],
        links=[("العرض وتحليل النتائج ونقاط الضعف والمقارنة والمقترحات والمناقشة", "العرض والمناقشة", "الأسبوع 07 (التقرير)؛ الأسبوع 14 (المراحل)"), ("الرمز النهائي وجودة الاستنساخ والعرض", "الرمز النهائي", "الأسبوع 14 (القالب)؛ الأسس 2 (البذرة)")],
        buttons=[("الأسبوع 14 — دليل المشروع", ":material/menu_book:", "course.w14.project_guide"), ("الرمز النهائي", ":material/grading:", "course.w15.rubric")],
        pretest=[Q("أول شريحة بعد العنوان:", ["البنية", "المسألة والقرار والبيانات", "الدقة"], 1, ""), Q("رقم الأداء يُعرض…", ["وحده", "مع خط الأساس و± البذور", "مع الخسارة الموحَّدة"], 1, ""),
                 Q("سؤال «لماذا لم تجرب X؟» يُجاب بـ…", ["الدفاع", "النزاهة: لم أجرب، ولهذا السبب/سأجرب", "تجاهل"], 1, ""), Q("الاستنساخ في الرمز يعني…", ["الكود موجود", "المقيّم يعيد الأرقام من الدفتر", "الدقة نفسها"], 1, "")],
        note_ar="اعرض المنهج لا النموذج. المقيّم يبحث عن خط الأساس والفجوة والأخطاء والاستنساخ قبل الدقة.",
        takeaway_ar="الأسبوع 15 = عرض نزيه، مناقشة تبحث عن الضعف، ورمز معلن يُقيَّم به الجميع.")
