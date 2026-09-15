import streamlit as st

from components.callouts import definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.diagnostics import Problem, problem_card
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.prep.leakage",
    title_ar="التسريب: تسريب الخصائص والهدف والمعالجة المسبقة",
    title_en="Leakage: Feature, Target & Preprocessing Leakage",
    module="foundations.prep",
    order=5,
    prerequisites=["foundations.prep.splitting"],
    objectives_ar=[
        "التعرف على الأنواع الثلاثة للتسريب وكيف يظهر كل منها.",
        "تشخيص «الدقة الخيالية» كعرض للتسريب لا كنجاح.",
        "بناء قائمة تحقق لمنع التسريب في أي مشروع.",
    ],
    terms=["feature", "target", "standardization"],
    labs=["labs.data_leakage_lab"],
    difficulty="intermediate",
    summary_ar="التسريب = معلومة لا تتوفر وقت التنبؤ تدخل التدريب؛ أعراضه دقة خيالية؛ ثلاثة أنواع: خاصية، هدف، معالجة.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
n = 2000
income = rng.normal(5000, 1500, n); late = rng.poisson(1.2, n)
p = 1/(1+np.exp(-(-3 + 0.8*late - 0.0003*income)))
y = (rng.uniform(size=n) < p).astype(float)

# خاصية مسرّبة: "عدد أشهر التحصيل" تُعرف فقط بعد التعثر
collection_months = np.where(y == 1, rng.integers(1, 12, n), 0)

def fit_logreg(X, y, epochs=300, eta=0.5):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); Xb = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = 1/(1+np.exp(-(Xb @ w))); w -= eta * Xb.T @ (p - y) / len(y)
    return w, (X.mean(0), X.std(0) + 1e-9)

def accuracy(w, X, y, stats):
    Xz = (X - stats[0]) / stats[1]; Xb = np.column_stack([Xz, np.ones(len(X))])
    return (((1/(1+np.exp(-(Xb @ w)))) >= 0.5) == y).mean()

idx = rng.permutation(n); tr, te = idx[:1500], idx[1500:]
X_honest = np.column_stack([income, late])
X_leaky  = np.column_stack([income, late, collection_months])
for name, X in [("honest", X_honest), ("leaky ", X_leaky)]:
    w, stats = fit_logreg(X[tr], y[tr])
    print(f"{name}: train acc={accuracy(w, X[tr], y[tr], stats):.3f}  test acc={accuracy(w, X[te], y[te], stats):.3f}")
print("baseline (majority):", max(y.mean(), 1-y.mean()).round(3))
print("In production collection_months is unknown at prediction time -> the leaky model is useless.")'''


def render() -> None:
    lesson_header(LESSON)
    h2("ما التسريب؟", "What is leakage?")
    definition("**التسريب** `Leakage` دخول معلومة إلى التدريب أو التقييم **لن تكون متاحة وقت التنبؤ الحقيقي**. النتيجة: أداء ممتاز في التجربة، فشل في الواقع. أخطر أخطاء التعلم الآلي لأنه لا يُنتج رسالة خطأ بل رسالة نجاح.")
    intuition("طالب يحفظ أسئلة الامتحان قبل الامتحان: درجة كاملة في الامتحان، صفر معرفة. الاختبار لم يعد يقيس ما نريد.")
    problem_card(Problem(
        key="leakage", name_ar="التسريب", name_en="Data leakage",
        description_ar="معلومة من الهدف، أو من المستقبل، أو من مجموعة الاختبار تتسرب إلى التدريب أو إلى المعالجة المسبقة.",
        symptoms_ar=["دقة/خسارة أفضل بكثير من خط الأساس ومن المتوقع للمجال.", "خاصية واحدة تفسر كل شيء.", "أداء ممتاز في التحقق وسيئ في الإنتاج."],
        sees_ar=["`val_accuracy: 0.99` في مسألة معروفة الصعوبة.", "ارتباط 0.95+ بين خاصية والهدف.", "أهمية خاصية واحدة تطغى على الباقي."],
        possible_causes_ar=["**تسريب هدف**: خاصية مشتقة من الهدف أو تُعرف بعده (أشهر التحصيل، مبلغ التعويض).",
                            "**تسريب خاصية/زمني**: قيم من المستقبل (متوسط الشهر كله لتنبؤ يوم في منتصفه).",
                            "**تسريب معالجة**: التحجيم/التعويض/الترميز/اختيار الخصائص بمعلمات محسوبة على كل البيانات قبل التقسيم.",
                            "**تسريب تكرار/مجموعة**: نفس الكيان أو صفوف مكررة في التدريب والاختبار."],
        root_causes_ar=["عدم طرح السؤال: «هل تتوفر هذه القيمة لحظة التنبؤ؟»", "تطبيق المعالجة قبل التقسيم لأنه «أسهل»."],
        diagnosis_ar=["اطرح لكل خاصية: متى تُعرف قيمتها زمنيًا مقارنة بالهدف؟", "افحص الارتباطات العالية جدًا مع الهدف.", "درّب بلا الخاصية المشبوهة وقارن.",
                      "تأكد أن `fit` لكل محوّل على التدريب فقط.", "افحص التداخل: صفوف/كيانات مشتركة بين المجموعات.", "قارن أداء التحقق بأداء عينة زمنية لاحقة فعلًا."],
        evidence_ar=["حذف الخاصية يعيد الأداء إلى مستوى معقول.", "إعادة المعالجة بعد التقسيم تغيّر النتيجة."],
        fixes_ar=["احذف الخصائص التي تُعرف بعد الحدث.", "قسّم أولًا، ثم `fit` على التدريب، ثم `transform`.", "تقسيم زمني/بالمجموعة عند الحاجة.", "اجعل خط الأنابيب كائنًا واحدًا يُدرَّب على التدريب فقط."],
        tradeoffs_ar=["حذف خاصية مسرّبة يخفض الأداء المبلَّغ — وهو الأداء الصادق."],
        misdiagnosis_ar=["«النموذج ممتاز» بينما هو تسريب.", "«فرط تخصيص» بينما التحقق نفسه ملوّث."],
        related_ar=["فرط التخصيص", "التقسيم الزمني", "التحجيم"],
        checklist_ar=["كل خاصية متاحة لحظة التنبؤ؟", "التقسيم قبل أي fit؟", "لا كيان مشترك بين المجموعات؟", "الأداء معقول مقارنة بخط الأساس والمجال؟"],
        challenge=[Q("خسارة تحقق 0.01 في التنبؤ بتعثر القروض من بيانات بنكية. الاحتمال الأرجح؟", ["نموذج رائع", "تسريب", "معدل تعلم جيد"], 1, "أداء خيالي = تسريب حتى يثبت العكس.", kind="scenario"),
                   Q("تعويض المفقودات بالوسيط المحسوب على كل البيانات ثم التقسيم…", ["صحيح", "تسريب معالجة", "لا يهم"], 1, "الوسيط رأى الاختبار.", kind="scenario")],
    ))
    code_lab(CodeLab(
        key="prep_leak", title_ar="نموذج صادق مقابل نموذج مسرّب", code=CODE,
        before=Before(goal_ar="مقارنة نموذجين على نفس البيانات: أحدهما يستخدم خاصية تُعرف بعد الحدث. رؤية الدقة الخيالية.", stage_ar="إعداد البيانات ← التشخيص.",
                      inputs_ar="2000 عميل بخاصيتين صادقتين وخاصية مسرّبة.", expected_ar="الصادق ≈ خط الأساس + قليل؛ المسرّب ≈ 100% — ورسالة تذكّر بأن الخاصية غير متاحة في الإنتاج."),
        explain=[("8-9", "`collection_months` صفر لغير المتعثرين وموجب للمتعثرين: تُعرف **بعد** التعثر. كشفٌ مثالي، لكنه لا يتوفر لعميل جديد."),
                 ("11-20", "انحدار لوجستي صغير مع توحيد داخلي (fit على التدريب — لاحظ أن الإحصاءات تُعاد إلى الدالة الأخرى)."),
                 ("22-27", "نفس التقسيم، نموذجان. الفرق في الدقة كله من الخاصية المسرّبة.")],
        run=run_printed(CODE),
        after_ar="- الدقة 100% ليست نجاحًا؛ هي إعادة كتابة للهدف.\n- الصادق قريب من خط الأساس لأن المسألة فعلًا صعبة بخاصيتين — وهذا صادق.",
    ))
    research_note("في الاقتصاد: التنبؤ بالركود باستخدام «مراجعات» البيانات المنشورة لاحقًا (revised data) تسريب زمني كلاسيكي. استخدم البيانات كما كانت متاحة وقتها (vintage data).")
    st.button("افتح معمل التسريب", icon=":material/science:", on_click=go, args=("labs.data_leakage_lab",), key="leak_lab_btn")
    quiz("prep.leak", [
        Q("خاصية «مبلغ التعويض المدفوع» للتنبؤ بوقوع حادث…", ["مفيدة", "تسريب هدف", "ضوضاء"], 1, "تُعرف بعد الحادث."),
        Q("الوقاية من تسريب المعالجة…", ["حجّم كل البيانات معًا", "قسّم ثم fit على التدريب ثم transform", "لا تحجّم"], 1, "الترتيب."),
        Q("أهم عرض للتسريب…", ["خسارة NaN", "أداء أفضل بكثير من المعقول", "تدريب بطيء"], 1, "رسالة نجاح كاذبة."),
    ])
    takeaway("التسريب يعطي نجاحًا كاذبًا. اسأل لكل خاصية: متاحة لحظة التنبؤ؟ قسّم قبل أي fit. أداء خيالي = تسريب حتى يثبت العكس.")
    lesson_footer(LESSON, ["ثلاثة أنواع: هدف، خاصية/زمن، معالجة.", "لا رسالة خطأ؛ فقط دقة مبالغ فيها.", "قائمة التحقق قبل التقرير."])
