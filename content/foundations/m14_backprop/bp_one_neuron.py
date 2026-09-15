import numpy as np
import streamlit as st

from components.callouts import intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import worked_steps
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.backprop.backpropagation.one_neuron",
    title_ar="مثال بخلية واحدة: أمامي، خلفي، تحديث، وتحسّن",
    title_en="One-Neuron Example: Forward, Backward, Update, Improvement",
    module="foundations.backprop",
    parent="foundations.backprop.backpropagation",
    order=3,
    prerequisites=["foundations.backprop.backpropagation.gradient_flow"],
    objectives_ar=["تنفيذ دورة كاملة (أمامي → خلفي → تحديث) لخلية واحدة بالأرقام.", "إثبات أن الخسارة تنخفض بعد التحديث."],
    terms=["weight", "bias", "learning_rate"],
    difficulty="intermediate",
    summary_ar="خلية Sigmoid على مدخلين: نحسب z, p, L، ثم δ = p − y، ثم gW = x·δ، gb = δ، نحدّث، ونعيد الحساب لنرى الخسارة تهبط.",
)

CODE = '''import numpy as np
sigmoid = lambda z: 1/(1+np.exp(-z))
x = np.array([1.0, 2.0]); y = 1.0
w = np.array([0.5, -0.3]); b = 0.1; eta = {eta}

for step in range(1, 4):
    z = w @ x + b; p = sigmoid(z); L = -(y*np.log(p) + (1-y)*np.log(1-p))     # أمامي
    delta = p - y                                                             # ∂L/∂z للثنائي مع Sigmoid+BCE
    gw, gb = x * delta, delta                                                 # ∂L/∂w = x·δ, ∂L/∂b = δ
    print(f"step {{step}}: z={{z:+.3f}} p={{p:.3f}} L={{L:.4f}} | delta={{delta:+.3f}} gw={{gw.round(3)}} gb={{gb:+.3f}}")
    w, b = w - eta*gw, b - eta*gb                                             # تحديث
print("after 3 updates: w =", w.round(3), " b =", round(b, 3))'''


def _controls() -> dict:
    return {"eta": st.select_slider("η", options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0, key="ctrl_bp1_eta")}


def render() -> None:
    lesson_header(LESSON)
    h2("الإعداد", "Setup")
    st.markdown("خلية Sigmoid بمدخلين، هدف $y = 1$، أوزان أولية $w = (0.5, -0.3)$، $b = 0.1$، مدخل $x = (1, 2)$.")
    z0 = 0.5 * 1 + (-0.3) * 2 + 0.1; p0 = 1 / (1 + np.exp(-z0)); L0 = -np.log(p0); d0 = p0 - 1
    worked_steps([("أمامي: z", rf"z = 0.5(1) + (-0.3)(2) + 0.1 = {z0:+.3f}"), ("أمامي: p", rf"p = \sigma({z0:+.3f}) = {p0:.3f}"), ("الخسارة", rf"L = -\ln {p0:.3f} = {L0:.4f}"),
                  ("خلفي: δ", rf"\delta = p - y = {p0:.3f} - 1 = {d0:+.3f}"), ("تدرج الأوزان", rf"g_w = x\,\delta = ({d0:+.3f},\ {2 * d0:+.3f}),\quad g_b = {d0:+.3f}"),
                  ("تحديث بـ η = 1", rf"w \leftarrow (0.5, -0.3) - ({d0:+.3f}, {2 * d0:+.3f}) = ({0.5 - d0:.3f}, {-0.3 - 2 * d0:.3f})")])
    intuition("δ سالب لأن p < y: النموذج «قلّل» — فالتحديث يزيد z، أي يزيد الأوزان في اتجاه x. الخاصية الثانية (x₂ = 2) تحصل على تدرج مضاعف لأنها ساهمت أكثر.")
    code_lab(CodeLab(
        key="bp_one", title_ar="ثلاث دورات كاملة", code=CODE, template=True, defaults={"eta": 1.0},
        before=Before(goal_ar="تنفيذ أمامي/خلفي/تحديث ثلاث مرات ومشاهدة الخسارة تنخفض والاحتمال يقترب من 1.", stage_ar="الانتشار الخلفي ← مثال.",
                      inputs_ar="خلية واحدة، ملاحظة واحدة، معدل تعلم.", expected_ar="L تهبط كل خطوة (مع η معقول)، وp يرتفع نحو 1؛ مع η = 5 قد يقفز."),
        explain=[("7", "الأمامي في سطر: z، p، BCE."), ("8", "الاختصار الأنيق: مع Sigmoid+BCE، ∂L/∂z = p − y مباشرة (درس الانحدار اللوجستي)."), ("9", "القواعد: تدرج الوزن = المدخل × δ، تدرج الانحياز = δ."), ("11", "التحديث بالمحسّن الأبسط (SGD).")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- الخسارة تنخفض في كل خطوة: الدليل أن التدرج يشير إلى الاتجاه الصحيح.\n- الوزن الثاني يتغير ضعف الأول تمامًا كما حسبنا.",
    ))
    quiz("bp.one", [
        Q("δ = p − y = −0.55؛ اتجاه تحديث w…", ["يقلل z", "يزيد z", "لا يغيّره"], 1, "نطرح η·x·δ (سالب) فتزيد."),
        Q("تدرج الوزن المرتبط بـ x₂ = 2 مقارنة بـ x₁ = 1…", ["نصفه", "ضعفه", "متساويان"], 1, "x·δ."),
    ])
    takeaway("دورة واحدة كاملة بالأرقام: z, p, L؛ δ = p − y؛ gW = x·δ؛ تحديث؛ الخسارة تهبط.")
    lesson_footer(LESSON, ["الاختصار p − y.", "المدخل الأكبر يحصل على تدرج أكبر.", "التحديث يخفض الخسارة فعلًا."])
