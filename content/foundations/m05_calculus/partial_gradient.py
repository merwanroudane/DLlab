import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.calculus.partial_gradient",
    title_ar="المشتقة الجزئية والتدرج واتجاهه",
    title_en="Partial Derivatives, the Gradient & its Direction",
    module="foundations.calculus",
    order=3,
    prerequisites=["foundations.calculus.derivative", "foundations.linalg.scalars_vectors"],
    objectives_ar=[
        "اشتقاق دالة متعددة المتغيرات بالنسبة لمتغير واحد مع تثبيت الباقي.",
        "تجميع المشتقات الجزئية في متجه التدرج وفهم أنه يشير إلى اتجاه أسرع صعود.",
        "استنتاج قاعدة التحديث: تحرك عكس التدرج.",
    ],
    terms=["gradient", "learning_rate", "loss", "parameter"],
    labs=["labs.gradient_lab"],
    difficulty="intermediate",
    summary_ar="المشتقة الجزئية تثبّت الباقي؛ التدرج متجه كل الجزئيات؛ يشير إلى أسرع صعود فنتحرك عكسه.",
)

CODE = '''import numpy as np

def L(w, b):                       # خسارة بمتغيرين: وعاء مركزه (2, -1)
    return (w - 2)**2 + 3*(b + 1)**2

def grad(w, b):                    # المشتقات الجزئية رمزيًا
    return np.array([2*(w - 2), 6*(b + 1)])

def grad_numeric(w, b, h=1e-5):    # نفس الشيء عدديًا: ثبّت أحدهما وحرّك الآخر
    dw = (L(w + h, b) - L(w - h, b)) / (2*h)
    db = (L(w, b + h) - L(w, b - h)) / (2*h)
    return np.array([dw, db])

w, b = {w0}, {b0}
print("L      =", L(w, b))
print("grad   =", grad(w, b), " numeric:", grad_numeric(w, b).round(5))

eta = {eta}
for step in range(6):
    g = grad(w, b)
    w, b = w - eta*g[0], b - eta*g[1]          # الخطوة عكس التدرج
    print(f"step {{step+1}}: w={{w:7.4f}} b={{b:7.4f}} L={{L(w, b):9.5f}} |grad|={{np.linalg.norm(grad(w, b)):.4f}}")'''


def _controls() -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        w0 = st.slider("w₀", -3.0, 6.0, 5.0, 0.5, key="ctrl_grad_w0")
    with c2:
        b0 = st.slider("b₀", -4.0, 3.0, 2.0, 0.5, key="ctrl_grad_b0")
    with c3:
        eta = st.select_slider("η", options=[0.01, 0.05, 0.1, 0.2, 0.3, 0.35], value=0.1, key="ctrl_grad_eta")
    return {"w0": w0, "b0": b0, "eta": eta}


def _surface(w0: float, b0: float) -> None:
    w = np.linspace(-3, 6, 60); b = np.linspace(-4, 3, 60)
    W, B = np.meshgrid(w, b)
    Z = (W - 2) ** 2 + 3 * (B + 1) ** 2
    g = np.array([2 * (w0 - 2), 6 * (b0 + 1)])
    fig = go.Figure(go.Contour(x=w, y=b, z=Z, colorscale="YlGnBu", contours=dict(showlabels=True), showscale=False))
    fig.add_trace(go.Scatter(x=[w0], y=[b0], mode="markers", marker=dict(size=12, color="#C8473A"), name="(w₀, b₀)"))
    s = 0.15
    fig.add_trace(go.Scatter(x=[w0, w0 + s * g[0]], y=[b0, b0 + s * g[1]], mode="lines+markers", line=dict(color="#C8473A", width=3), name="التدرج ∇L (أسرع صعود)"))
    fig.add_trace(go.Scatter(x=[w0, w0 - s * g[0]], y=[b0, b0 - s * g[1]], mode="lines+markers", line=dict(color="#1F7A78", width=3), name="−∇L (اتجاه التحديث)"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="w", yaxis_title="b", legend=dict(orientation="h"), paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="grad_contour")


def render() -> None:
    lesson_header(LESSON)
    h2("المشتقة الجزئية", "Partial derivative")
    definition("لدالة بعدة متغيرات $L(w, b)$، **المشتقة الجزئية** $\\frac{\\partial L}{\\partial w}$ هي مشتقة $L$ بالنسبة لـ $w$ **مع تثبيت** $b$ كأنه ثابت. الرمز $\\partial$ («دِل» المستديرة) يذكّرك أن هناك متغيرات أخرى مثبتة.")
    equation(r"L(w, b) = (w-2)^2 + 3(b+1)^2 \quad\Rightarrow\quad \frac{\partial L}{\partial w} = 2(w-2), \qquad \frac{\partial L}{\partial b} = 6(b+1)",
             [(r"\partial L/\partial w", "اشتق كأن $b$ رقم ثابت: الحد $3(b+1)^2$ يختفي."), (r"\partial L/\partial b", "اشتق كأن $w$ ثابت: الحد $(w-2)^2$ يختفي.")],
             meaning_ar="كل جزئية تسأل: إن حرّكت هذا المتغير وحده، كم يتغير المخرج؟",
             example_ar="عند $(w, b) = (5, 2)$: $\\partial L/\\partial w = 6$، $\\partial L/\\partial b = 18$.",
             dl_link_ar="الشبكة لها ملايين المعلمات؛ لكل واحدة مشتقة جزئية للخسارة. الانتشار الخلفي يحسبها كلها في تمريرة واحدة.", title_ar="مشتقتان جزئيتان")
    h2("التدرج", "The gradient")
    equation(r"\nabla L = \left(\frac{\partial L}{\partial w},\ \frac{\partial L}{\partial b}\right)",
             [(r"\nabla", "نابلا: «متجه كل المشتقات الجزئية»."), (r"\nabla L", "متجه بطول عدد المعلمات.")],
             meaning_ar="التدرج يجمع الجزئيات في متجه واحد يشير إلى **اتجاه أسرع زيادة** للدالة، وطوله معدل تلك الزيادة.",
             example_ar="عند $(5, 2)$: $\\nabla L = (6, 18)$؛ الدالة تصعد أسرع في اتجاه $b$ لأن معامله 3.",
             dl_link_ar="قاعدة التحديث $\\theta \\leftarrow \\theta - \\eta\\nabla L$: نتحرك **عكس** التدرج لأننا نريد النزول لا الصعود.", title_ar="التدرج")
    intuition("قف على سفح جبل مغمض العينين. التدرج هو اتجاه أشد انحدار صعودًا تحت قدميك. للنزول إلى الوادي (أقل خسارة) امشِ عكسه. معدل التعلم طول خطوتك.")
    why("لماذا عكس التدرج تحديدًا ولا اتجاه آخر؟ لأن التدرج هو الاتجاه الذي تكون فيه المشتقة الاتجاهية قصوى؛ عكسه هو أسرع نزول محلي. أي اتجاه آخر ينزل أبطأ.")
    h3("شاهد الاتجاه", "See the direction")
    p = _controls()
    _surface(p["w0"], p["b0"])
    code_lab(CodeLab(
        key="calc_grad", title_ar="التدرج رمزيًا وعدديًا + ست خطوات نزول", code=CODE, template=True, defaults=p,
        before=Before(goal_ar="حساب التدرج بطريقتين والتحقق من تطابقهما، ثم تنفيذ ست خطوات انحدار تدريجي.", stage_ar="تفاضل ← التحسين.",
                      inputs_ar="نقطة بداية `(w0, b0)` ومعدل تعلم `η` (من عناصر التحكم أعلاه).", expected_ar="تدرجان متطابقان، ثم خسارة تتناقص ومعيار تدرج يقترب من الصفر (أو يتباعد إن كان η كبيرًا).",
                      math_ar="$\\theta \\leftarrow \\theta - \\eta \\nabla L(\\theta)$."),
        explain=[("3-4", "وعاء إهليلجي: أدنى نقطة عند (2, −1). المعامل 3 يجعل الاتجاه b أشد انحدارًا."),
                 ("6-7", "الجزئيتان من الجدول، مجمّعتين في متجه."),
                 ("9-12", "عدديًا: حرّك `w` وحده ثم `b` وحده. الفكرة الحرفية للمشتقة الجزئية."),
                 ("14-16", "قيم البداية والمقارنة."),
                 ("18-22", "ست خطوات: كل خطوة تطرح `η × التدرج`. راقب `|grad|`: يجب أن يتناقص عند الاقتراب من القاع.")],
        run=run_printed(CODE, template=True),
        after_ar="- مع `η = 0.1` تتناقص الخسارة بسرعة؛ `b` يصل قبل `w` لأن تدرجه أكبر.\n- جرّب `η = 0.35`: الاتجاه b يتذبذب ويتباعد (6 × 0.35 > 1 يقلب الإشارة ويكبّرها) — مثال حي على معدل تعلم كبير.",
    ))
    common_mistake("«التدرج يشير إلى الحد الأدنى». لا: يشير إلى أسرع **صعود** محليًا. نأخذ سالبه، وحتى سالبه لا يشير مباشرة إلى القاع في الأوعية الإهليلجية (لذلك يتعرج المسار).")
    st.button("افتح معمل التدرج", icon=":material/science:", on_click=goto, args=("labs.gradient_lab",), key="grad_lab_btn")
    quiz("calc.gradient", [
        Q("$L = w^2 b$؛ $\\partial L / \\partial w$ هي…", ["$2wb$", "$w^2$", "$2w$"], 0, "b ثابت.", kind="equation"),
        Q("التدرج يشير إلى…", ["أسرع نزول", "أسرع صعود", "الحد الأدنى دائمًا"], 1, "لذلك نأخذ سالبه."),
        Q("عند القاع تمامًا، التدرج…", ["أكبر ما يمكن", "صفر", "سالب"], 1, "لا ميل عند الحد الأدنى."),
        Q("شبكة بمليون معلمة: طول متجه التدرج؟", ["1", "مليون", "يعتمد على البيانات"], 1, "جزئية لكل معلمة.", kind="shape"),
    ])
    takeaway("الجزئية: ثبّت الباقي. التدرج: متجه الجزئيات، اتجاه أسرع صعود. التحديث: عكسه بخطوة η.")
    lesson_footer(LESSON, ["∂ تعني متغيرات أخرى مثبتة.", "∇L بطول عدد المعلمات.", "θ ← θ − η∇L؛ η كبير = تذبذب."])
