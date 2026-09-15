import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.optim.rmsprop_adam",
    title_ar="RMSprop وAdam: معدل تعلم لكل معلمة",
    title_en="RMSprop & Adam: a Learning Rate per Parameter",
    module="foundations.optim",
    order=5,
    prerequisites=["foundations.optim.momentum_nesterov"],
    objectives_ar=["فهم فكرة التكيّف: قسمة التدرج على جذر متوسط مربعاته لتوحيد الخطوة عبر الاتجاهات.", "معادلات RMSprop وAdam (زخم + تكيف + تصحيح الانحياز).", "معرفة القيم الافتراضية ومتى لا يكون Adam الخيار الأفضل."],
    terms=["optimizer", "learning_rate"],
    labs=["labs.optimizer_race"],
    difficulty="intermediate",
    summary_ar="RMSprop: g / √(متوسط g²). Adam: زخم للتدرج + RMSprop للمقياس + تصحيح انحياز. خطوة ≈ η لكل معلمة مهما كان مقياس تدرجها.",
)

CODE = '''import numpy as np
def grad(p): return np.array([1.0 * p[0], 25.0 * p[1]])      # نفس الوادي الضيق
def loss(p): return 0.5 * p[0]**2 + 12.5 * p[1]**2

def run(kind, eta, steps=60, b1=0.9, b2=0.999, eps=1e-8):
    p = np.array([-6.0, 1.0]); m = np.zeros(2); s = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        g = grad(p)
        if kind == "sgd":
            p = p - eta * g
        elif kind == "rmsprop":
            s = b2 * s + (1 - b2) * g**2
            p = p - eta * g / (np.sqrt(s) + eps)
        elif kind == "adam":
            m = b1 * m + (1 - b1) * g;  s = b2 * s + (1 - b2) * g**2
            m_hat = m / (1 - b1**t);    s_hat = s / (1 - b2**t)          # تصحيح الانحياز
            p = p - eta * m_hat / (np.sqrt(s_hat) + eps)
        path.append(p.copy())
    return np.array(path)

for kind, eta in [("sgd", 0.05), ("rmsprop", {eta_adaptive}), ("adam", {eta_adaptive})]:
    path = run(kind, eta)
    step_sizes = np.abs(np.diff(path, axis=0))
    print(f"{{kind:<8}} eta={{eta:<5}} final loss={{loss(path[-1]):9.5f}}  mean |step| x={{step_sizes[:,0].mean():.4f}} y={{step_sizes[:,1].mean():.4f}}")
print("SGD: y-steps 25x larger than x-steps (raw gradient scale). Adaptive: both ~eta.")'''


def _controls() -> dict:
    return {"eta_adaptive": st.select_slider("η لـ RMSprop/Adam", options=[0.01, 0.05, 0.1, 0.3, 0.5], value=0.1, key="ctrl_adam_eta")}


def render() -> None:
    lesson_header(LESSON)
    h2("الفكرة: طبّع الخطوة", "The idea: normalize the step")
    intuition("في الوادي الضيق تدرج y أكبر 25 مرة من تدرج x. SGD يعطي y خطوات 25 مرة أكبر. ماذا لو قسمنا كل مكوّنة على حجمها المعتاد؟ تصبح كل الخطوات بنفس الحجم تقريبًا — معدل تعلم فعلي **لكل معلمة**.")
    equation(r"\text{RMSprop:}\quad s_t = \beta_2 s_{t-1} + (1-\beta_2)\,g_t^2, \qquad \theta_{t+1} = \theta_t - \eta\,\frac{g_t}{\sqrt{s_t} + \epsilon}",
             [("s_t", "متوسط متحرك لمربع التدرج، **لكل معلمة على حدة** (عنصريًا)."), (r"\beta_2", "0.999 عادةً: ذاكرة طويلة."), (r"g_t/\sqrt{s_t}", "التدرج مقسومًا على مقياسه المعتاد: النتيجة بحجم ~1."), (r"\epsilon", "1e-8 لتفادي القسمة على صفر.")],
             meaning_ar="المعلمات ذات التدرجات الكبيرة تُكبح، وذات التدرجات الصغيرة تُضخَّم. الخطوة ≈ η للجميع.",
             example_ar="تدرج y ≈ 25، تدرج x ≈ 1: بعد التطبيع كلاهما ≈ ±1.",
             dl_link_ar="`RMSprop` افتراضي جيد للـ RNN تاريخيًا. الفكرة نفسها في Adagrad (بلا نسيان) وAdadelta.", title_ar="RMSprop")
    equation(r"\text{Adam:}\quad m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2, \quad \hat m_t = \frac{m_t}{1-\beta_1^t}, \ \hat s_t = \frac{s_t}{1-\beta_2^t}, \quad \theta_{t+1} = \theta_t - \eta\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}",
             [("m_t", "زخم (متوسط متحرك للتدرج) بـ $\\beta_1 = 0.9$."), ("s_t", "مقياس (متوسط متحرك لمربعه) بـ $\\beta_2 = 0.999$."), (r"\hat m, \hat s", "تصحيح الانحياز: في الخطوات الأولى المتوسطات المتحركة تبدأ من صفر وتكون منحازة نحوه؛ القسمة على $1 - \\beta^t$ تصحح ذلك."), (r"\eta", "0.001 افتراضيًا.")],
             meaning_ar="Adam = زخم + RMSprop + تصحيح للبداية. الخطوة لكل معلمة ≈ η × (اتجاه ثابت؟ ±1 : أقل).",
             example_ar="t = 1: $\\hat m_1 = g_1$ بالضبط (بدل $0.1 g_1$) بفضل التصحيح.",
             dl_link_ar="`Adam(learning_rate=1e-3)` الافتراضي الأكثر شيوعًا. AdamW يفصل تنظيم weight decay عن التدرج (تعمّق).", title_ar="Adam")
    code_lab(CodeLab(
        key="optim_adam", title_ar="حجم الخطوة في كل اتجاه: SGD مقابل RMSprop مقابل Adam", code=CODE, template=True, defaults={"eta_adaptive": 0.1},
        before=Before(goal_ar="قياس متوسط حجم الخطوة في الاتجاهين لكل محسّن على الوادي الضيق.", stage_ar="التحسين ← المحسّنات التكيفية.",
                      inputs_ar="η لكل نوع.", expected_ar="SGD: خطوات y أكبر بكثير من x؛ RMSprop/Adam: الاثنان ≈ η — والخسارة النهائية أصغر."),
        explain=[("9-11", "SGD خام."), ("12-14", "RMSprop: متوسط g² ثم القسمة على جذره."), ("15-18", "Adam: زخم + مقياس + تصحيح الانحياز بـ `t`."), ("21-24", "متوسط |الخطوة| في كل اتجاه يكشف الفرق الجوهري.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- Adam بـ η = 0.1 على هذا الوادي: خطوات x وy متقاربة، والتقدم في x سريع رغم صغر تدرجه.\n- الثمن: قرب القاع التدرجات صغيرة فيُضخَّم الضجيج — لذلك يُخفَّض η في النهاية (جدولة).",
    ))
    compare_table(["المحسّن", "زخم", "تكيف المقياس", "η الافتراضي", "متى"],
                  [("SGD", "لا", "لا", "0.01–0.1", "مرجع؛ مع زخم وجدولة قد يعمّم أفضل في الصور الكبيرة"), ("SGD+Momentum", "نعم", "لا", "0.01–0.1", "CNN الكلاسيكية"),
                   ("RMSprop", "لا", "نعم", "1e-3", "RNN؛ تدرجات متفاوتة المقياس"), ("Adam", "نعم", "نعم", "1e-3", "الافتراضي العملي لأغلب المسائل"), ("AdamW", "نعم", "نعم", "1e-3", "Adam مع weight decay صحيح؛ Transformers")],
                  ["ltr", "rtl", "rtl", "code", "rtl"])
    research_note("Adam ليس دائمًا الأفضل تعميمًا: في بعض مهام الصور يتفوق SGD+Momentum مع جدولة جيدة على مجموعة الاختبار رغم تفوق Adam في سرعة تقليل خسارة التدريب. للمقرر: ابدأ بـ Adam، وجرّب SGD+Momentum عند البحث عن آخر نقطة مئوية.")
    common_mistake("«Adam لا يحتاج ضبط معدل التعلم». يحتاج، لكن نطاقه المعقول أضيق (1e-4 إلى 3e-3). 1e-3 نقطة بداية لا نهاية.")
    quiz("optim.adam", [
        Q("RMSprop يقسم التدرج على…", ["متوسطه", "جذر متوسط مربعه", "عدد الخطوات"], 1, "المقياس."),
        Q("تصحيح الانحياز في Adam يعالج…", ["التلاشي", "انحياز المتوسطات نحو الصفر في البداية", "الانفجار"], 1, "1 − β^t."),
        Q("β₁ وβ₂ الافتراضيان…", ["0.5، 0.5", "0.9، 0.999", "0.99، 0.9"], 1, "القيم القياسية."),
        Q("Adam يعطي خطوة ≈ η…", ["لأكبر معلمة فقط", "لكل معلمة بغض النظر عن مقياس تدرجها", "لا يعطي"], 1, "التكيف."),
    ])
    takeaway("التكيف = قسمة كل مكوّنة على مقياسها. Adam = زخم + تكيف + تصحيح. 1e-3 بداية، وخفّض في النهاية.")
    lesson_footer(LESSON, ["RMSprop: g/√s.", "Adam: m̂/√ŝ.", "الجدول يحدد الاختيار."])
