import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table, good_vs_bad
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.prep.scaling",
    title_ar="التحجيم: التطبيع والتوحيد القياسي، وfit/transform/fit_transform",
    title_en="Scaling: Normalization vs Standardization; fit / transform / fit_transform",
    module="foundations.prep",
    order=3,
    prerequisites=["foundations.prep.encoding", "foundations.prob.descriptive"],
    objectives_ar=[
        "التمييز بين التطبيع (min-max إلى [0,1]) والتوحيد القياسي (z-score).",
        "فهم لماذا التحجيم ضروري للشبكات وأثره على سطح الخسارة ومعدل التعلم.",
        "تطبيق fit على التدريب فقط ثم transform على الباقي، وفهم معنى كل دالة.",
    ],
    terms=["standardization", "mean", "standard_deviation", "feature"],
    labs=["labs.scaling_lab"],
    difficulty="beginner",
    summary_ar="التوحيد (x−μ)/σ أو التطبيع (x−min)/(max−min)؛ fit من التدريب فقط ثم transform للباقي.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
income = rng.lognormal(8.4, 0.4, 300); age = rng.integers(21, 66, 300).astype(float)
X = np.column_stack([income, age])
idx = rng.permutation(300); X_tr, X_te = X[idx[:240]], X[idx[240:]]
print("raw ranges: income", X_tr[:, 0].min().round(0), "-", X_tr[:, 0].max().round(0), " age", X_tr[:, 1].min(), "-", X_tr[:, 1].max())

class StandardScaler:
    def fit(self, X):                 # يتعلم μ و σ من X (التدريب فقط)
        self.mu, self.sigma = X.mean(axis=0), X.std(axis=0); return self
    def transform(self, X):           # يطبق التحويل بمعلمات محفوظة
        return (X - self.mu) / self.sigma
    def fit_transform(self, X):       # الاثنان معًا — للتدريب فقط
        return self.fit(X).transform(X)

sc = StandardScaler()
Z_tr = sc.fit_transform(X_tr)         # μ, σ من التدريب
Z_te = sc.transform(X_te)             # نفس μ, σ على الاختبار
print("train mean/std:", Z_tr.mean(axis=0).round(3), Z_tr.std(axis=0).round(3))
print("test  mean/std:", Z_te.mean(axis=0).round(3), Z_te.std(axis=0).round(3), " ← ليس بالضبط 0/1، وهذا صحيح")

# التطبيع min-max
mn, mx = X_tr.min(axis=0), X_tr.max(axis=0)
N_te = (X_te - mn) / (mx - mn)
print("min-max test range:", N_te.min(axis=0).round(2), N_te.max(axis=0).round(2), " ← قد يتجاوز [0,1] قليلًا")

# أثر التحجيم على التدرج: تدرجات بمقاييس متباينة قبل التحجيم
w = np.zeros(2); y = 0.001*X_tr[:, 0] - 0.5*X_tr[:, 1]
g_raw = X_tr.T @ (X_tr @ w - y) / 240; g_std = Z_tr.T @ (Z_tr @ w - (y - y.mean())/y.std()) / 240
print("gradient scale raw   :", np.abs(g_raw).round(1))
print("gradient scale scaled:", np.abs(g_std).round(3))'''


def render() -> None:
    lesson_header(LESSON)
    h2("لماذا التحجيم؟", "Why scale?")
    why("الدخل بالآلاف والعمر بالعشرات: تدرج وزن الدخل أكبر بمئات المرات من تدرج وزن العمر. معدل تعلم مناسب للعمر يفجّر الدخل، ومعدل يناسب الدخل يجمّد العمر. سطح الخسارة وعاء إهليلجي ممدود — رأيت تعرجه في معمل التدرج. التحجيم يجعله أقرب إلى الدائرة.")
    intuition("تخيل قيادة سيارة تستجيب عجلتها اليمنى 100 مرة أقوى من اليسرى. التحجيم يجعل الاستجابتين متساويتين فتصبح القيادة (التدريب) مستقرة.")
    compare_table(["الطريقة", "English", "الصيغة", "الناتج", "حساسية للشواذ", "متى"],
                  [("التوحيد القياسي", "Standardization", "(x − μ) / σ", "متوسط 0، انحراف 1، غير محدود", "متوسطة", "الافتراضي للجداول"),
                   ("التطبيع", "Normalization (min-max)", "(x − min) / (max − min)", "[0, 1] على التدريب", "عالية (شاذ واحد يضغط الباقي)", "الصور (بكسل/255)، عندما يلزم نطاق محدد"),
                   ("تحويل لوغاريتمي", "Log transform", "log(1 + x)", "يضغط الذيل الطويل", "يقللها", "الدخل، الأسعار، العدّ قبل التوحيد"),
                   ("متين", "Robust scaling", "(x − median) / IQR", "حول 0", "منخفضة", "بيانات بشواذ كثيرة")],
                  ["rtl", "ltr", "code", "rtl", "rtl", "rtl"])
    equation(r"z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}, \qquad x' = \frac{x - \min_{\text{train}}}{\max_{\text{train}} - \min_{\text{train}}}",
             [(r"\mu_{\text{train}}, \sigma_{\text{train}}", "تُحسب من **بيانات التدريب فقط** وتُحفظ."), ("x", "أي قيمة: من التدريب أو التحقق أو الاختبار أو الإنتاج — نفس المعلمات.")],
             meaning_ar="التحويل دالة ثابتة تُتعلم مرة واحدة ثم تُطبق على الجميع.",
             example_ar="μ = 5000، σ = 1500: دخل 8000 → z = 2؛ دخل 3500 → z = −1.",
             dl_link_ar="**fit** = تعلّم المعلمات (μ, σ). **transform** = تطبيقها. **fit_transform** = الاثنان، ويُستخدم **مرة واحدة على التدريب فقط**.", title_ar="التحجيم")
    good_vs_bad("الترتيب الصحيح", "قسّم ← fit على التدريب ← transform على الكل.", "التسريب الشائع", "fit_transform على كل البيانات ثم التقسيم: الاختبار أثّر في μ وσ.",
                good_code="X_tr, X_te = split(X)\nsc.fit(X_tr)\nZ_tr = sc.transform(X_tr)\nZ_te = sc.transform(X_te)",
                bad_code="Z = sc.fit_transform(X)     # كل البيانات!\nZ_tr, Z_te = split(Z)",
                verdict_ar="الفرق صغير في الأرقام لكنه خرق مبدئي: أي معلومة من الاختبار تدخل التدريب تجعل التقييم متفائلًا.")
    code_lab(CodeLab(
        key="prep_scale", title_ar="محجّم من الصفر + أثر التحجيم على التدرج", code=CODE,
        before=Before(goal_ar="بناء `StandardScaler` بثلاث دوال، تطبيقه بالترتيب الصحيح، مقارنة التطبيع، ورؤية تباين مقاييس التدرج قبل/بعد.", stage_ar="إعداد البيانات ← التحجيم.",
                      inputs_ar="دخل (لوغاريتمي-طبيعي) وعمر لـ 300 ملاحظة.", expected_ar="تدريب موحد بالضبط 0/1، اختبار قريبًا منها، min-max قد يتجاوز [0,1]، وتدرجات بمقاييس تختلف بمئات المرات قبل التحجيم ومتقاربة بعده."),
        explain=[("8-14", "الفئة تفصل التعلم (`fit`) عن التطبيق (`transform`) — نفس تصميم scikit-learn وطبقة `Normalization` في Keras."),
                 ("16-20", "التدريب يعطي 0/1 بالضبط لأن معلماته منه؛ الاختبار قريب لا مطابق — الطبيعي والصحيح."),
                 ("23-25", "min-max على الاختبار قد يخرج عن [0, 1] لأن الأقصى/الأدنى من التدريب."),
                 ("28-31", "التدرج الخام لوزن الدخل بمقياس آلاف مقابل عشرات للعمر؛ بعد التحجيم متقاربان.")],
        run=run_printed(CODE),
        after_ar="- نسبة مقاييس التدرج قبل التحجيم ≈ نسبة مقاييس الخصائص²: هذا ما يجعل معدل تعلم واحد مستحيلًا.\n- الهدف أيضًا يُحجَّم في الانحدار (رأيت هذا في المحاكي: /100) وتُعكس النتيجة عند التقرير.",
    ))
    st.button("افتح معمل التحجيم", icon=":material/science:", on_click=go, args=("labs.scaling_lab",), key="scaling_lab_btn")
    common_mistake("تحجيم أعمدة one-hot (0/1): غير ضار غالبًا لكنه بلا معنى؛ وتحجيم الهدف الفئوي: خطأ. حجّم الخصائص العددية فقط.")
    quiz("prep.scale", [
        Q("`fit_transform` يُستدعى على…", ["كل البيانات", "التدريب فقط", "الاختبار فقط"], 1, "التعلم من التدريب."),
        Q("متوسط الاختبار بعد التوحيد بمعلمات التدريب…", ["0 بالضبط", "قريب من 0", "1"], 1, "معلمات من مجموعة أخرى."),
        Q("خاصية بذيل طويل (الدخل): قبل التوحيد يُفضَّل…", ["min-max", "log(1+x)", "لا شيء"], 1, "يضغط الذيل."),
        Q("لماذا يُصعّب عدم التحجيم اختيار معدل التعلم؟", ["يبطئ الحساب", "تدرجات بمقاييس متباينة جدًا", "يسبب NaN دائمًا"], 1, "وعاء إهليلجي."),
    ])
    takeaway("حجّم الخصائص العددية: توحيد افتراضيًا، min-max للصور، log للذيول. fit على التدريب مرة، transform على الجميع.")
    lesson_footer(LESSON, ["التحجيم يوحّد مقاييس التدرجات.", "fit / transform / fit_transform معانٍ محددة.", "التقسيم قبل fit."])
