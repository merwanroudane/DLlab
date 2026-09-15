import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import definition, intuition, takeaway
from components.diagnostics import Problem, problem_card
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.optim.learning_rate",
    title_ar="معدل التعلم: صغير جدًا، كبير جدًا، تذبذب، تباعد، NaN",
    title_en="Learning Rate: Too Small, Too Large, Oscillation, Divergence, NaN",
    module="foundations.optim",
    order=3,
    prerequisites=["foundations.optim.gd_variants"],
    objectives_ar=["فهم معدل التعلم كحجم الخطوة وعلاقته بانحناء السطح (η < 2/انحناء للاستقرار).", "التعرف على الأعراض الخمسة وتشخيصها بإطار كامل.", "مشاهدة الحالات في تحريك ومعمل."],
    terms=["learning_rate", "hyperparameter"],
    labs=["labs.learning_rate_lab"],
    difficulty="intermediate",
    summary_ar="η حجم الخطوة؛ صغير = بطء، كبير = تذبذب ثم تباعد ثم NaN. الحد النظري للوعاء التربيعي η < 2/k.",
)


def _path(eta: float, k: float = 1.0, steps: int = 12, x0: float = 4.0):
    xs = [x0]
    for _ in range(steps):
        x = xs[-1]; xs.append(x - eta * k * x)   # L = k/2 x²  → L' = kx
    return xs


def _svg(xs, k):
    w, h = 460, 200
    lo, hi = -6, 6
    def sx(x): return 20 + (x - lo) / (hi - lo) * (w - 40)
    def sy(y): return h - 20 - min(y, 20) / 20 * (h - 50)
    curve = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(k / 2 * x * x):.1f}" for i, x in enumerate(np.linspace(lo, hi, 80)))
    pts = " ".join(f"{'M' if i == 0 else 'L'}{sx(np.clip(x, lo - 1, hi + 1)):.1f},{sy(k / 2 * x * x):.1f}" for i, x in enumerate(xs))
    s = f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px"><path d="{curve}" fill="none" stroke="#2F6FB5" stroke-width="2"/>'
    s += f'<path d="{pts}" fill="none" stroke="#C8473A" stroke-width="2" stroke-dasharray="4 3"/>'
    for i, x in enumerate(xs):
        s += f'<circle cx="{sx(np.clip(x, lo - 1, hi + 1)):.1f}" cy="{sy(k / 2 * x * x):.1f}" r="{5 if i == len(xs) - 1 else 3}" fill="{"#1F7A78" if i == len(xs) - 1 else "#C8473A"}"/>'
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ما معدل التعلم؟", "What is the learning rate?")
    definition("**معدل التعلم** $\\eta$ معلمة فائقة تضرب التدرج لتحديد **حجم الخطوة**: $\\theta \\leftarrow \\theta - \\eta\\nabla L$. أهم معلمة فائقة على الإطلاق: خطأ فيها يفسد أي بنية.")
    equation(r"L(x) = \tfrac{k}{2}x^2 \;\Rightarrow\; x_{t+1} = x_t - \eta k x_t = (1 - \eta k)\,x_t",
             [("k", "انحناء الوعاء (المشتقة الثانية)."), ("1 - \\eta k", "عامل الضرب في كل خطوة: بين 0 و1 نزول ناعم؛ بين −1 و0 تذبذب متقارب؛ أقل من −1 تباعد.")],
             meaning_ar="الاستقرار يتطلب $|1 - \\eta k| < 1$ أي $\\eta < 2/k$: كلما زاد الانحناء وجب صغر الخطوة. الخطوة المثالية $\\eta = 1/k$ تصل في خطوة واحدة.",
             example_ar="$k = 1$: $\\eta = 0.5$ نزول ناعم، $\\eta = 1.5$ تذبذب متقارب، $\\eta = 2.5$ تباعد.",
             dl_link_ar="الشبكة لها انحناءات مختلفة في اتجاهات مختلفة (رأيتها في معمل التحجيم)؛ $\\eta$ واحد يجب أن يحترم أشدها. لهذا تفيد المحسّنات التكيفية.", title_ar="التحليل على وعاء تربيعي")
    h3("تحريك: أربعة معدلات على نفس الوعاء", "Animation")
    frames = []
    for eta, label in [(0.1, "صغير جدًا: بطيء"), (0.8, "جيد: نزول سريع"), (1.7, "كبير: تذبذب متقارب"), (2.3, "كبير جدًا: تباعد")]:
        xs = _path(eta)
        frames.append(Frame(_svg(xs, 1.0), caption(f"**η = {eta}** — {label}. بعد 12 خطوة: `x = {xs[-1]:.3g}`، `L = {0.5 * xs[-1] ** 2:.3g}`. العامل `1 − ηk = {1 - eta:+.1f}`."),
                            action=f"η = {eta}", equation=f"x_(t+1) = (1 − {eta}) x_t", values=[("x after 12 steps", "4.0", f"{xs[-1]:.3g}")]))
    animation_player("lr_anim", frames, title_ar="أثر معدل التعلم على وعاء بانحناء 1", interval_ms=2500)
    intuition("على الوعاء المثالي كل شيء محسوب. في الشبكة الحقيقية الانحناء يتغير من مكان لآخر ومن اتجاه لآخر، لذا نجرّب: ابدأ بـ 1e-3 لـ Adam أو 1e-2 لـ SGD، واستخدم البحث اللوغاريتمي (3e-4, 1e-3, 3e-3, …).")
    problem_card(Problem(
        key="lr_bad", name_ar="معدل تعلم غير مناسب", name_en="Bad learning rate",
        description_ar="الخطوة أصغر أو أكبر مما يحتمله انحناء سطح الخسارة.",
        symptoms_ar=["**صغير**: الخسارة تهبط ببطء شديد وبخط شبه مستقيم؛ حقب كثيرة بلا تقدم يُذكر.", "**كبير**: تذبذب من حقبة لأخرى؛ أو قفزة ثم ارتفاع ثم `nan`.", "**متوسط سيئ**: نزول سريع ثم استقرار فوق الحد الأدنى (تذبذب حول القاع لا يهبط إليه)."],
        sees_ar=["منحنى loss مستوٍ منحدر قليلًا (صغير).", "منحنى مسنّن أو منفجر (كبير).", "`loss: nan` في السجل."],
        possible_causes_ar=["η غير مضبوط للمحسّن (SGD يحتاج أكبر من Adam).", "مدخلات غير محجّمة تغيّر الانحناء.", "حجم دفعة صغير مع η كبير."],
        root_causes_ar=["|1 − ηk| بعيد عن الصفر: k صغير ⇒ بطء، k كبير ⇒ تباعد."],
        diagnosis_ar=["شغّل مسح لوغاريتمي قصير (5 قيم × حقبتين) وارسم الخسارة.", "راقب معيار التدرج: قفزاته مع الخسارة تعني η كبير.", "تحقق من التحجيم أولًا."],
        evidence_ar=["منحنى «حرف U» لخسارة الحقبة الثانية مقابل log(η): الأدنى هو نقطة البداية."],
        fixes_ar=["اختر η من المسح.", "جدولة تنازلية للوصول إلى القاع بعد التذبذب.", "warm-up في البداية.", "Adam إن اختلفت الانحناءات كثيرًا.", "قصّ التدرج عند الانفجار."],
        tradeoffs_ar=["η صغير آمن لكن مكلف؛ الجدولة تجمع السرعة أولًا والدقة أخيرًا."],
        misdiagnosis_ar=["«النموذج صغير» (بينما η صغير).", "«البيانات فيها NaN» (بينما η كبير)."],
        related_ar=["انفجار التدرج", "الجداول", "التحجيم"],
        checklist_ar=["مسح لوغاريتمي؟", "المدخلات محجّمة؟", "η مناسب للمحسّن؟", "جدولة؟"],
        challenge=[Q("الخسارة: 2.3، 2.29، 2.28، 2.27 عبر 4 حقب على MNIST. التشخيص؟", ["كبير", "صغير جدًا (أو تلاشٍ)", "مثالي"], 1, "بطء شديد.", kind="curve"),
                   Q("الخسارة: 0.9، 0.4، 1.8، 0.3، 2.5…", ["صغير", "كبير: تذبذب", "جيد"], 1, "مسنّن.", kind="curve")],
    ))
    st.button("افتح معمل معدل التعلم", icon=":material/science:", type="primary", on_click=goto, args=("labs.learning_rate_lab",), key="lr_lab_btn")
    quiz("optim.lr", [
        Q("على وعاء بانحناء k، شرط الاستقرار…", ["η < k", "η < 2/k", "η > 2/k"], 1, "|1 − ηk| < 1."),
        Q("Adam يبدأ عادةً بـ η…", ["1", "1e-3", "1e-8"], 1, "أصغر من SGD."),
        Q("نزول سريع ثم استقرار فوق القاع مع تذبذب…", ["η صغير", "η كبير للمرحلة النهائية: جدولة تنازلية", "تلاشٍ"], 1, "يقفز حول القاع."),
    ])
    takeaway("η = حجم الخطوة مقابل الانحناء. صغير بطء، كبير تذبذب/تباعد/NaN. مسح لوغاريتمي + جدولة تنازلية.")
    lesson_footer(LESSON, ["η < 2/k.", "الأعراض الخمسة في إطار واحد.", "المسح اللوغاريتمي قبل أي شيء."])
