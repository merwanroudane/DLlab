from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w10.overview",
    title_ar="نظرة عامة على الأسبوع 10 والاختبار القبلي",
    title_en="Week 10 Overview & Pre-test",
    module="course.w10",
    order=1,
    prerequisites=["course.w09.shape_debugging", "foundations.data.data_modalities"],
    objectives_ar=["خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على التسلسلات والحالة المخفية."],
    terms=["sequence", "time_series"],
    difficulty="beginner",
    summary_ar="التسلسل والخطوة الزمنية والترتيب ← الحالة المخفية والاتصال التكراري والنشر عبر الزمن ← RNN وBPTT والتلاشي/الانفجار ← تطبيق على التضخم الشهري ← تمهيد LSTM.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="10",
        steps=["Sequence · timestep · order", "Hidden state & recurrence", "Unrolling through time", "BPTT", "Vanishing / exploding", "Time-series application", "LSTM preview", "Post-test"],
        links=[("التسلسل والخطوة الزمنية والترتيب والحالة المخفية", "التسلسلات والحالة المخفية", "الأسس 1: أنماط البيانات (التسلسل)؛ الأسس 4: الموترات (samples, timesteps, features)"), ("الاتصال التكراري والنشر عبر الزمن وBPTT والتلاشي/الانفجار", "RNN وBPTT", "الأسس 14: الانتشار الخلفي، التلاشي والانفجار"),
               ("التطبيق على بيانات زمنية وتمهيد LSTM", "تطبيق على السلاسل الزمنية", "الأسبوع 07 (المنهج)؛ الأسس 8 (التقسيم الزمني)")],
        buttons=[("معمل نشر RNN", ":material/science:", "labs.rnn_unrolling_lab"), ("الأسس 14 — التلاشي والانفجار", ":material/menu_book:", "foundations.backprop.backpropagation.vanishing_exploding")],
        pretest=[Q("شكل دفعة تسلسلات في Keras:", ["(samples, features)", "(samples, timesteps, features)", "(timesteps, samples)"], 1, ""), Q("الحالة المخفية h_t تعتمد على…", ["x_t فقط", "x_t وh_{t−1}", "y_t"], 1, ""),
                 Q("أوزان RNN عبر الخطوات الزمنية…", ["مختلفة لكل خطوة", "نفسها في كل خطوة", "عشوائية"], 1, ""), Q("تقسيم سلسلة زمنية إلى تدريب/اختبار يكون…", ["عشوائيًا", "زمنيًا: الماضي للتدريب والمستقبل للاختبار", "بالتساوي"], 1, "")],
        note_ar="الأسس 14 شرحت تلاشي التدرج عبر الطبقات؛ هنا نفس الظاهرة عبر **الزمن**. المعمل يجعل الخلية ترى التسلسل خطوةً خطوة.",
        takeaway_ar="الأسبوع 10 = ذاكرة عبر الزمن بخلية واحدة مكررة، وحدودها التي يعالجها LSTM/GRU.")
