import numpy as np
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table, good_vs_bad
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.loss.cce_sparse",
    title_ar="الإنتروبيا المتقاطعة الفئوية: CCE وSparse CCE",
    title_en="Categorical Cross-Entropy: CCE & Sparse CCE",
    module="foundations.loss",
    order=4,
    prerequisites=["foundations.loss.bce", "foundations.activations.softmax_output"],
    objectives_ar=["صيغة CCE مع one-hot وتبسيطها إلى −log p̂(الصحيحة).", "الفرق العملي بين CCE (هدف one-hot) وSparse CCE (هدف أرقام فئات).", "القيمة المرجعية ln K والخطأ الشائع في شكل الهدف."],
    terms=["cross_entropy", "softmax", "target"],
    labs=["labs.loss_lab"],
    difficulty="intermediate",
    summary_ar="CCE = −Σ y_k log p̂_k = −log p̂(الصحيحة)؛ Sparse نفس الرقم بهدف أرقام؛ مرجع ln K.",
)

CODE = '''import numpy as np
def softmax(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

logits = np.array([[2.0, 0.5, -1.0], [0.2, 0.1, 0.3], [-1.0, 2.5, 0.0]])
y_ids = np.array([0, 2, 1])                     # أرقام الفئات (sparse)
Y_1h = np.eye(3)[y_ids]                         # one-hot
P = softmax(logits)

cce = -np.mean((Y_1h * np.log(P)).sum(axis=1))                 # مع one-hot
sparse = -np.mean(np.log(P[np.arange(3), y_ids]))              # مع أرقام
print("P =\\n", P.round(3))
print("CCE =", cce.round(4), "  Sparse CCE =", sparse.round(4), " (identical)")
print("per-sample -log p(correct):", (-np.log(P[np.arange(3), y_ids])).round(3))
print("reference ln K (K=3):", np.log(3).round(4))

# الخطأ الشائع: أرقام الفئات مع صيغة one-hot
try:
    wrong = -np.mean((y_ids[:, None] * np.log(P)).sum(axis=1))    # يعامل الرقم كوزن!
    print("WRONG (ids fed as one-hot):", wrong.round(4), " <- meaningless (class 0 contributes nothing, class 2 doubled)")
except Exception as e:
    print(e)

# label smoothing: هدف ناعم يقلل الثقة المطلقة
eps = 0.1; Y_smooth = Y_1h * (1 - eps) + eps / 3
print("smoothed target row 0:", Y_smooth[0].round(3), " CCE(smooth) =", (-np.mean((Y_smooth * np.log(P)).sum(1))).round(4))'''


def render() -> None:
    lesson_header(LESSON)
    h2("الصيغة", "The formula")
    equation(r"\text{CCE} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\log \hat{p}_{ik} \;=\; -\frac{1}{n}\sum_{i=1}^{n}\log \hat{p}_{i,\,c_i}",
             [("y_{ik}", "one-hot: 1 للفئة الصحيحة $c_i$ و0 للباقي."), (r"\hat p_{ik}", "احتمال Softmax للفئة $k$."), (r"\hat p_{i, c_i}", "بعد التبسيط: احتمال الفئة الصحيحة فقط.")],
             meaning_ar="مفاجأة النموذج من الفئة الصحيحة، في المتوسط. الفئات الخاطئة لا تدخل مباشرة — لكنها تدخل عبر مقام Softmax.",
             example_ar="3 فئات، الصحيحة 0، $\\hat p = (0.7, 0.2, 0.1)$: $-\\ln 0.7 = 0.357$.",
             dl_link_ar="**CCE** تتوقع هدفًا `(n, K)` one-hot؛ **Sparse CCE** تتوقع `(n,)` أرقام فئات. نفس الرقم تمامًا؛ الفرق شكل الهدف فقط. تدرجها بالنسبة لـ logits: $\\hat p - y$.", title_ar="CCE")
    compare_table(["", "categorical_crossentropy", "sparse_categorical_crossentropy"],
                  [("شكل الهدف", "(n, K) one-hot", "(n,) أعداد صحيحة 0..K−1"), ("الذاكرة", "K أضعاف", "أقل"), ("متى", "أهداف ناعمة (label smoothing، mixup)", "الافتراضي العملي"), ("PyTorch", "—", "nn.CrossEntropyLoss (يأخذ logits + أرقامًا)")],
                  ["rtl", "code", "code"])
    code_lab(CodeLab(
        key="loss_cce", title_ar="CCE وSparse متطابقتان، الخطأ الشائع، والتنعيم", code=CODE,
        before=Before(goal_ar="حساب CCE بالصيغتين، إثبات تطابقهما، رؤية ما يحدث عند تمرير أرقام الفئات إلى صيغة one-hot، وتجربة label smoothing.", stage_ar="الخسارة.",
                      inputs_ar="logits (3, 3) وأهداف بصيغتين.", expected_ar="رقمان متطابقان ≈ 0.77؛ نتيجة «خاطئة» بلا خطأ تشغيل؛ مرجع ln 3 = 1.0986."),
        explain=[("4-7", "نفس الهدف بشكلين: أرقام ثم one-hot عبر `np.eye`."), ("9-10", "الصيغتان: الضرب في one-hot يختار عمودًا؛ الفهرسة المتقدمة تختاره مباشرة."),
                 ("16-20", "تمرير أرقام الفئات إلى صيغة one-hot: الرقم 0 يلغي مساهمة ملاحظته، والرقم 2 يضاعفها. **لا رسالة خطأ** لأن الأشكال تُبثّ. الأطر قد تعطي خطأ شكل أو تُدرّب على هراء."),
                 ("23-25", "التنعيم: بدل (1, 0, 0) نستخدم (0.93, 0.03, 0.03). يمنع الثقة المطلقة ويحسّن المعايرة.")],
        run=run_printed(CODE),
        after_ar="- الملاحظة الثانية (logits شبه متساوية) تساهم بـ ≈1.07 ≈ ln 3: النموذج جاهل هناك.\n- التنعيم يرفع الخسارة قليلًا عمدًا؛ الحد الأدنى لم يعد صفرًا.",
    ))
    good_vs_bad("هدف أرقام + Sparse CCE", "`y.shape == (n,)` من أعداد صحيحة.", "هدف أرقام + CCE", "الأطر تتوقع (n, K)؛ خطأ شكل، أو أسوأ: بث صامت.",
                good_code="model.compile(loss='sparse_categorical_crossentropy')\nmodel.fit(X, y_ids)          # y_ids.shape == (n,)",
                bad_code="model.compile(loss='categorical_crossentropy')\nmodel.fit(X, y_ids)          # expects (n, K) one-hot!",
                verdict_ar="اطبع `y.shape` و`y.dtype` واختر الخسارة وفقهما. أو حوّل بـ `to_categorical(y_ids)` واستخدم CCE.")
    definition("**القيمة المرجعية**: $\\ln K$ لمصنف جاهل ($K$ فئات). 10 فئات: 2.303. 100 فئة: 4.6. ابدأ التدريب وتوقع خسارة قريبة منها.")
    debugging_note("`ValueError: Shapes (32, 1) and (32, 10) are incompatible` مع CCE: الهدف أرقام `(32, 1)` والنموذج يخرج `(32, 10)`. الحل: Sparse CCE مع `(32,)`، أو one-hot.")
    common_mistake("فئات مرقمة من 1 إلى K بدل 0 إلى K−1 مع Sparse CCE: الفئة K خارج المدى ⇒ خطأ فهرسة (أو `nan` في بعض الإصدارات). رقّم من الصفر.")
    quiz("loss.cce", [
        Q("Sparse CCE مقابل CCE…", ["خسارتان مختلفتان", "نفس الرقم؛ شكل الهدف مختلف", "Sparse أدق"], 1, "الفرق تمثيلي."),
        Q("مصنف 10 فئات، خسارة ابتدائية معقولة…", ["0.1", "≈ 2.3", "10"], 1, "ln 10."),
        Q("هدف بأرقام 0..9 بالشكل (n,): الخسارة…", ["categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy"], 1, "أرقام."),
        Q("label smoothing…", ["يرفع الثقة", "يخفض الثقة المطلقة ويحسّن المعايرة", "يسرّع التدريب"], 1, "هدف ناعم."),
    ])
    takeaway("CCE = −log p̂(الصحيحة) في المتوسط. one-hot ↔ CCE، أرقام ↔ Sparse. مرجع ln K. اطبع شكل y قبل اختيار الخسارة.")
    lesson_footer(LESSON, ["التبسيط إلى حد واحد.", "Sparse للأرقام، CCE للناعم.", "ln K نقطة البداية."])
