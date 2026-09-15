import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.optim.momentum_nesterov",
    title_ar="الزخم ونسترُف",
    title_en="Momentum & Nesterov",
    module="foundations.optim",
    order=4,
    prerequisites=["foundations.optim.learning_rate"],
    objectives_ar=["فهم الزخم كمتوسط متحرك للتدرجات يراكم الاتجاه الثابت ويلغي التذبذب.", "معادلة الزخم ومعنى β، وتعديل نسترُف («انظر قبل أن تقفز»).", "رؤية أثره في وادٍ ضيق."],
    terms=["optimizer", "gradient"],
    labs=["labs.optimizer_race"],
    difficulty="intermediate",
    summary_ar="v ← βv + ∇L، θ ← θ − ηv: كرة تتدحرج تراكم السرعة في الاتجاه الثابت وتلغي التذبذب العرضي. نسترُف يحسب التدرج بعد خطوة الزخم.",
)

CODE = '''import numpy as np
# وادٍ ضيق: انحناء 1 في x، و25 في y  → SGD يتذبذب في y ويبطئ في x
def grad(p): return np.array([1.0 * p[0], 25.0 * p[1]])
def loss(p): return 0.5 * p[0]**2 + 12.5 * p[1]**2

def run(kind, eta, beta=0.9, steps=60):
    p, v = np.array([-6.0, 1.0]), np.zeros(2); path = [p.copy()]
    for _ in range(steps):
        if kind == "sgd":       g = grad(p); p = p - eta * g
        elif kind == "momentum": g = grad(p); v = beta * v + g; p = p - eta * v
        elif kind == "nesterov": g = grad(p - eta * beta * v); v = beta * v + g; p = p - eta * v
        path.append(p.copy())
        if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6: break
    return np.array(path)

eta = {eta}
for kind in ["sgd", "momentum", "nesterov"]:
    path = run(kind, eta)
    print(f"{{kind:<9}} steps={{len(path)-1:>3}} final loss={{loss(path[-1]):10.5f}}  final p={{path[-1].round(3)}}  y-oscillations={{int((np.sign(path[1:,1]) != np.sign(path[:-1,1])).sum())}}")'''


def _controls() -> dict:
    return {"eta": st.select_slider("η", options=[0.01, 0.03, 0.05, 0.07, 0.08, 0.1], value=0.05, key="ctrl_mom_eta")}


def render() -> None:
    lesson_header(LESSON)
    h2("المشكلة التي يحلها الزخم", "The problem momentum solves")
    intuition("وادٍ ضيق: جدرانه شديدة الانحدار (اتجاه y) وقاعه ينحدر ببطء نحو الحل (اتجاه x). SGD يقفز بين الجدارين ويتقدم قليلًا في x. الزخم يلاحظ أن تدرج y يتبدل إشارته (فيُلغى بالمتوسط) وأن تدرج x ثابت (فيتراكم).")
    equation(r"\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + \nabla L(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta\,\mathbf{v}_t",
             [(r"\mathbf{v}", "«السرعة»: مجموع مرجّح للتدرجات السابقة."), (r"\beta", "الزخم (0.9 عادةً): كم من السرعة السابقة نحتفظ به. الحد الأقصى للتضخيم $1/(1-\\beta) = 10$ لاتجاه ثابت."), (r"\eta", "معدل التعلم يضرب السرعة لا التدرج.")],
             meaning_ar="كرة ثقيلة تتدحرج: تتسارع في الاتجاه الثابت ولا تتأثر كثيرًا بالدفعات العرضية المتعاكسة.",
             example_ar="تدرج ثابت g لـ 20 خطوة بـ β = 0.9: السرعة تقترب من 10g — عشرة أضعاف SGD.",
             dl_link_ar="`SGD(momentum=0.9)` في Keras وPyTorch. جزء الزخم في Adam هو نفس الفكرة بتطبيع.", title_ar="الزخم")
    equation(r"\text{Nesterov:}\quad \mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + \nabla L\big(\theta_t - \eta\beta\,\mathbf{v}_{t-1}\big), \qquad \theta_{t+1} = \theta_t - \eta\,\mathbf{v}_t",
             [(r"\theta_t - \eta\beta\mathbf{v}_{t-1}", "«نظرة إلى الأمام»: نحسب التدرج عند النقطة التي سيصل إليها الزخم، لا عند النقطة الحالية.")],
             meaning_ar="إن كان الزخم سيدفعنا فوق القاع، فالتدرج المحسوب هناك يفرمل مبكرًا. تذبذب أقل قرب الحل.",
             example_ar="`SGD(momentum=0.9, nesterov=True)`.",
             dl_link_ar="تحسين صغير ومستقر؛ شائع في تدريب الشبكات الالتفافية الكبيرة مع جداول معدل تعلم.", title_ar="نسترُف")
    code_lab(CodeLab(
        key="optim_mom", title_ar="SGD مقابل الزخم مقابل نسترُف في وادٍ ضيق", code=CODE, template=True, defaults={"eta": 0.05},
        before=Before(goal_ar="مقارنة الثلاثة على وعاء إهليلجي (انحناء 1 مقابل 25) بعدد الخطوات والتذبذبات والخسارة النهائية.", stage_ar="التحسين ← المحسّنات.",
                      inputs_ar="η واحد للثلاثة.", expected_ar="SGD يتذبذب في y كثيرًا ويتقدم ببطء في x؛ الزخم يصل أقرب بكثير بنفس الخطوات؛ نسترُف تذبذبات أقل. مع η = 0.08+ يتباعد SGD (2/25 = 0.08) بينما قد يصمد الزخم بصعوبة."),
        explain=[("3-4", "وادٍ: انحناءان مختلفان بنسبة 25 — الحد الآمن لـ SGD هو η < 2/25 = 0.08."), ("6-13", "الثلاث خوارزميات في دالة واحدة؛ الفرق سطر واحد لكل منها."),
                 ("16-19", "عدّاد تبدل إشارة y يقيس التذبذب.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- الزخم يصل إلى خسارة أصغر بمراتب بنفس عدد الخطوات: يراكم في x ويلغي في y.\n- الزخم ليس مجانيًا: قرب القاع قد يتجاوزه ويعود (تذبذب زخمي)؛ نسترُف يخفف ذلك.",
    ))
    research_note("β = 0.9 قيمة افتراضية معقولة جدًا وقلّما تُضبط. ما يُضبط هو η — ومع الزخم يُخفَّض غالبًا لأن الخطوة الفعلية تصل إلى η/(1−β).")
    common_mistake("إبقاء نفس η عند إضافة الزخم: الخطوة الفعلية في الاتجاه الثابت تصبح 10 أضعاف فينفجر التدريب. قلّل η أو ابدأ بـ warm-up.")
    quiz("optim.mom", [
        Q("الزخم يضخّم…", ["التدرجات المتذبذبة", "التدرجات ثابتة الاتجاه", "كل التدرجات بالتساوي"], 1, "المتوسط يلغي المتعاكس."),
        Q("β = 0.9: الحد الأقصى لتضخيم اتجاه ثابت…", ["0.9", "10", "2"], 1, "1/(1−β)."),
        Q("نسترُف يحسب التدرج…", ["عند النقطة الحالية", "عند النقطة بعد خطوة الزخم", "في المتوسط"], 1, "نظرة إلى الأمام."),
    ])
    takeaway("الزخم: متوسط متحرك للتدرجات — يسرّع الثابت ويلغي المتذبذب. نسترُف ينظر قبل أن يقفز. قلّل η عند إضافته.")
    lesson_footer(LESSON, ["v ← βv + g؛ θ ← θ − ηv.", "β = 0.9، تضخيم حتى 10×.", "الوادي الضيق هو الحالة النموذجية."])
