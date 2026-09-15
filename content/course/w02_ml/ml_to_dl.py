import streamlit as st

from components.callouts import common_mistake, intuition, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="course.w02.ml_to_dl",
    title_ar="من التعلم الآلي إلى التعلم العميق: متى ولماذا تكفي الشبكة؟ + الاختبار البعدي",
    title_en="From Machine Learning to Deep Learning: When and Why a Network — & Post-test",
    module="course.w02",
    order=4,
    prerequisites=["course.w02.evaluation_baselines", "foundations.neuron.line_to_neuron", "foundations.architecture.depth_width_dense"],
    objectives_ar=["رؤية ما تضيفه طبقة مخفية على الانحدار اللوجستي على مسألة غير خطية — وما لا تضيفه على مسألة خطية.", "قواعد عملية: متى الخطي/الأشجار كافية ومتى الشبكة.", "الاختبار البعدي للأسبوع."],
    terms=["model", "feature", "hyperparameter"],
    labs=["labs.network_builder"],
    difficulty="beginner",
    summary_ar="الشبكة = انحدار لوجستي مع خصائص متعلَّمة قبله. تربح عندما تكون العلاقة غير خطية والبيانات كافية؛ وتخسر الشفافية. على جدول صغير خطي: لا فرق — فابدأ بالخطي.",
)

CODE = '''import numpy as np
from labs.tinynet import make_moons, TinyNet, train
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)

def compare(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    log = LogisticRegression().fit(X_tr, y_tr); acc_log = log.score(X_te, y_te)
    net = TinyNet([2, 16, 16, 1], "binary", seed=0)
    train(net, X_tr.astype("float32"), y_tr.astype("float32"), epochs=150, batch_size=32, lr=0.02, optimizer="adam", seed=0)
    acc_net = net.metric(X_te.astype("float32"), y_te.astype("float32"))
    print(f"{name:<28} logistic {acc_log:.3f} | MLP(2-16-16-1) {acc_net:.3f} | gain {acc_net - acc_log:+.3f}")

# (أ) مسألة خطية: حد قرار مستقيم + ضوضاء
X1 = rng.normal(size=(600, 2)); y1 = (X1[:, 0] + 0.8 * X1[:, 1] + rng.normal(0, 0.5, 600) > 0).astype(int)
compare(X1, y1, "linear boundary")
# (ب) مسألة غير خطية: هلالان
X2, y2 = make_moons(600, noise=0.25, seed=0)
compare(X2, y2.astype(int), "moons (non-linear)")
# (ج) غير خطية لكن ببيانات قليلة جدًا
X3, y3 = make_moons(60, noise=0.25, seed=1)
compare(X3, y3.astype(int), "moons, n=60")
# (د) نفس الخطية مع خاصية مهندَسة يدويًا للهلالين: x1² تجعل اللوجستي قادرًا
X2f = np.column_stack([X2, X2[:, 0] ** 2, X2[:, 0] ** 3])
X_tr, X_te, y_tr, y_te = train_test_split(X2f, y2.astype(int), test_size=0.3, random_state=0)
print(f"moons + hand-crafted x1², x1³   logistic {LogisticRegression(max_iter=500).fit(X_tr, y_tr).score(X_te, y_te):.3f}  <- feature engineering closes much of the gap")'''


def render() -> None:
    lesson_header(LESSON)
    why("«الشبكة أقوى» عبارة نصف صحيحة. الأصح: الشبكة **تتعلم خصائص جديدة** قبل الانحدار الأخير. عندما تكون الخصائص الأصلية كافية (علاقة خطية) لا تربح شيئًا وتخسر الشفافية؛ عندما تكون العلاقة غير خطية ولا تعرف الخاصية الصحيحة يدويًا، تربح كثيرًا — بشرط بيانات كافية.")
    code_lab(CodeLab(
        key="w02_mldl", title_ar="لوجستي مقابل MLP على أربع حالات: خطي، غير خطي، قليل البيانات، وهندسة خصائص", code=CODE, level="C",
        before=Before(goal_ar="قياس ما تضيفه طبقتان مخفيتان على الانحدار اللوجستي عبر أربع حالات مصممة لإظهار متى تربح الشبكة ومتى لا.", stage_ar="الأسبوع 02: من ML إلى DL.",
                      inputs_ar="بيانات تركيبية 2-بعدية (خطية وهلالان) بأحجام مختلفة.", expected_ar="خطي: فرق ≈ 0. هلالان: MLP أعلى بوضوح. n=60: الفرق يتقلص أو ينقلب. خصائص مهندَسة: اللوجستي يقترب من MLP.",
                      prerequisites_ar="الأسس 9–10 (من الانحدار إلى الشبكة)، `labs.tinynet` (محرك MLP بـ NumPy)."),
        explain=[("7-13", "دالة مقارنة: نفس التقسيم، لوجستي مقابل MLP صغير (Adam، 150 حقبة) ودقة الاختبار لكل منهما."), ("16-17", "حد قرار مستقيم: الخصائص الأصلية كافية؛ لا مكان للربح."), ("19-20", "الهلالان: لا خط مستقيم يفصلهما؛ الطبقات المخفية تتعلم انحناءً."),
                 ("22-23", "60 ملاحظة فقط: الشبكة (بمئات المعلمات) لا تجد ما تتعلمه بثبات؛ اللوجستي (3 معلمات) أكثر أمانًا."), ("25-27", "هندسة خصائص يدوية (x₁²، x₁³) تمنح اللوجستي انحناءً: ما تفعله الشبكة **تلقائيًا** يمكن أحيانًا فعله يدويًا إن عرفت الخاصية.")],
        run=run_printed(CODE),
        after_ar="- الربح في الحالة (ب) هو **الحجة الوحيدة** للشبكة هنا: تمثيل غير خطي متعلَّم.\n- الحالة (ج) درس الأسس 19: التعقيد يحتاج بيانات.\n- الحالة (د) تفسّر لماذا ما زال الاقتصاد القياسي يعمل: الخبير يهندس الخصائص؛ الشبكة تكتشفها — وتخفيها.",
    ))
    h2("قواعد عملية", "Practical rules")
    compare_table(["الحالة", "ابدأ بـ", "انتقل إلى الشبكة عندما"],
                  [("جدول صغير (< بضعة آلاف صف)، علاقات شبه خطية", "لوجستي/خطي، ثم أشجار متدرجة", "خط الأساس يتوقف وتبقى بنية غير خطية واضحة في البقايا"), ("جدول كبير بتفاعلات معقدة", "أشجار متدرجة (GBM) كخط أساس قوي", "الشبكة تتفوق على GBM على التحقق — ليس دائمًا"),
                   ("صور", "—", "فورًا: CNN (الأسبوع 08) — لا خصائص يدوية معقولة"), ("نصوص/تسلسل زمني طويل", "نماذج ARIMA/خطية كخط أساس", "RNN/LSTM/GRU عندما تهم الأنماط الطويلة (الأسابيع 10–12)"), ("مطلوب تفسير للمعاملات", "خطي/لوجستي", "نادرًا؛ أو شبكة + أدوات تفسير (خارج المقرر)")],
                  ["rtl", "rtl", "rtl"])
    intuition("اقرأ الشبكة كـ«انحدار لوجستي على خصائص لم يعد عليك اختراعها». هذا يفسّر كل شيء: لماذا تحتاج بيانات أكثر (تتعلم الخصائص أيضًا)، لماذا تخسر التفسير (الخصائص المتعلَّمة بلا أسماء)، ولماذا تسحق الطرق التقليدية في الصور والنصوص (لا خصائص يدوية جيدة هناك).")
    research_note("في الأدبيات التطبيقية الاقتصادية، الأشجار المتدرجة تتفوق كثيرًا على الشبكات في الجداول متوسطة الحجم. الشبكات تفوز في البيانات غير المهيكلة وفي الجداول الضخمة. أبلغ دائمًا خط أساس قويًا قبل ادعاء تفوق الشبكة — وهذا معيار في مشروع الأسبوع 14.")
    common_mistake("تدريب شبكة على 200 صف بلا مقارنة، ثم تفسير تذبذب النتائج بين البذور على أنه «حساسية النموذج». هي ببساطة بيانات أقل من أن تعلّم مئات المعلمات — الأسس 19.")
    h2("الاختبار البعدي", "Post-test")
    quiz("w02.posttest", [
        Q("التنبؤ بسعر عقار بالدينار هو…", ["تصنيف", "انحدار", "تجميع"], 1, ""),
        Q("مجموعة التحقق تُستخدم لـ…", ["تدريب المعلمات", "اختيار المعلمات الفائقة والعتبة والإيقاف", "التقرير النهائي"], 1, ""),
        Q("فئة إيجابية 3% فقط: الدقة 97% تعني…", ["نموذج ممتاز", "ربما لا شيء: خط أساس الأغلبية يعطيها", "تسريب"], 1, ""),
        Q("دقة تدريب 1.0 وتحقق 0.6:", ["قصور", "فرط تخصيص", "تعميم ممتاز"], 1, ""),
        Q("الشبكة تتفوق على اللوجستي عندما…", ["دائمًا", "العلاقة غير خطية والبيانات كافية", "البيانات قليلة"], 1, ""),
        Q("ما تفعله الطبقات المخفية مفاهيميًا:", ["تحفظ البيانات", "تتعلم خصائص جديدة قبل الانحدار الأخير", "تقلل حجم الدفعة"], 1, ""),
        Q("خط الأساس الواجب قبل أي شبكة على جدول:", ["لا شيء", "نموذج بسيط (لوجستي/خطي/أشجار)", "شبكة أعمق"], 1, ""),
    ], title_ar="الاختبار البعدي — الأسبوع 02")
    takeaway("الشبكة = لوجستي بخصائص متعلَّمة. تربح مع اللاخطية والبيانات الكافية، وتخسر الشفافية. الجدول الصغير يبدأ بالخطي/الأشجار؛ الصور والتسلسل تبدأ بالشبكة.")
    lesson_footer(LESSON, ["أربع حالات بالكود.", "قواعد الاختيار.", "الاختبار البعدي."])
