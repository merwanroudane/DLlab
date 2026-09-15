import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.activations.relu_family",
    title_ar="ReLU وعائلته: Leaky ReLU وELU وSELU",
    title_en="ReLU & its Family: Leaky ReLU, ELU, SELU",
    module="foundations.activations",
    order=3,
    prerequisites=["foundations.activations.sigmoid_tanh"],
    objectives_ar=["صيغة ReLU ومشتقتها ولماذا حلّت مشكلة التلاشي.", "فهم Dead ReLU وكيف تعالجه البدائل بميل صغير للسالب.", "معرفة متى تختار ReLU ومتى Leaky/ELU/SELU."],
    terms=["derivative", "gradient"],
    labs=["labs.activation_lab"],
    difficulty="intermediate",
    summary_ar="ReLU = max(0, z): مشتقة 1 للموجب (لا تلاشٍ) و0 للسالب (خطر الموت). البدائل تعطي السالب ميلًا صغيرًا.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
relu = lambda z: np.maximum(0, z); drelu = lambda z: (np.asarray(z) > 0).astype(float)
leaky = lambda z, a=0.1: np.where(z > 0, z, a*z); dleaky = lambda z, a=0.1: np.where(z > 0, 1.0, a)
elu = lambda z, a=1.0: np.where(z > 0, z, a*(np.exp(np.minimum(z, 0)) - 1))

for z in [-3, -1, 0, 1, 3]:
    print(f"z={z:>2}  relu={relu(z):.2f} d={drelu(z):.0f}   leaky={leaky(z):.2f} d={dleaky(z):.1f}   elu={elu(z):.2f}")

# Dead ReLU: وحدة ذات انحياز سالب كبير لا تُفعَّل لأي مدخل
X = rng.normal(size=(1000, 4)); w = rng.normal(size=4)
for b in [0.0, -2.0, -8.0]:
    z = X @ w + b
    print(f"bias={b:>5}: active fraction={(z > 0).mean():.3f}  mean gradient={drelu(z).mean():.3f}")
print("with bias=-8 the unit is dead: gradient 0 for every input -> it can never recover")

# التدرج عبر 10 طبقات ReLU (للوحدات النشطة): لا تلاشٍ
print("10 active ReLU layers gradient factor =", 1.0**10)'''


def render() -> None:
    lesson_header(LESSON)
    h2("ReLU", "ReLU")
    equation(r"\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}",
             [(r"\max(0, z)", "يمرر الموجب كما هو ويصفّر السالب."), ("1", "للموجب: التدرج يمر **دون تضعيف** — لا إشباع من هذا الجانب."), ("0", "للسالب: لا تدرج إطلاقًا.")],
             meaning_ar="أبسط لاخطية ممكنة: «إما أن تمرر أو تسكت». رخيصة حسابيًا ولا تُشبع للموجب.",
             example_ar="$\\text{ReLU}(3) = 3$، $\\text{ReLU}(-2) = 0$.",
             dl_link_ar="التنشيط الافتراضي للطبقات المخفية في MLP وCNN. حاصل ضرب مشتقات وحدات نشطة = 1 مهما كان العمق — حل عملي لتلاشي التدرج.", title_ar="ReLU")
    intuition("ReLU مفتاح: مفتوح (يمرر التدرج كاملًا) أو مغلق (يمرر صفرًا). الشبكة تتعلم أي المفاتيح تفتح لأي مدخل، فتصبح دالة خطية مختلفة في كل منطقة — «خطي قطعيًا».")
    h2("المشكلة: Dead ReLU", "Dead ReLU")
    definition("**Dead ReLU**: وحدة أصبح مجموعها الموزون سالبًا لكل المدخلات (غالبًا بعد تحديث كبير دفع الانحياز بعيدًا في السالب). مخرجها 0 دائمًا و**مشتقتها 0 دائمًا** فلا يصلها تدرج ولا تتعافى أبدًا. مع معدل تعلم كبير قد يموت جزء كبير من الشبكة.")
    compare_table(["الدالة", "الصيغة للسالب", "المشتقة للسالب", "يحل الموت؟", "ملاحظة"],
                  [("ReLU", "0", "0", "لا", "الافتراضي؛ راقب نسبة الوحدات الميتة"), ("Leaky ReLU", "α·z (α ≈ 0.01–0.1)", "α", "نعم", "تدرج صغير لكنه غير صفري"),
                   ("PReLU", "α·z، α متعلَّم", "α", "نعم", "معلمة إضافية لكل وحدة"), ("ELU", "α(e^z − 1)", "α·e^z", "نعم", "ناعمة، متوسط قريب من الصفر، أبطأ قليلًا"),
                   ("SELU", "λ·ELU بثوابت محددة", "—", "نعم", "«تطبيع ذاتي» بشروط (تهيئة LeCun، بلا Dropout عادي)")],
                  ["ltr", "code", "code", "rtl", "rtl"])
    z = np.linspace(-4, 4, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=np.maximum(0, z), name="ReLU", line=dict(color="#1F7A78", width=3)))
    fig.add_trace(go.Scatter(x=z, y=np.where(z > 0, z, 0.1 * z), name="Leaky (0.1)", line=dict(color="#2E8B57", width=2)))
    fig.add_trace(go.Scatter(x=z, y=np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1), name="ELU", line=dict(color="#7C5CBF", width=2)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="relu_fig")
    code_lab(CodeLab(
        key="act_relu", title_ar="ReLU وبدائله + وحدة ميتة", code=CODE,
        before=Before(goal_ar="جدول قيم ومشتقات، ثم محاكاة وحدة تموت بانحياز سالب كبير، ثم عامل التدرج عبر 10 طبقات نشطة.", stage_ar="التنشيط ← التشخيص.",
                      inputs_ar="قيم z، مدخلات عشوائية، ثلاثة انحيازات.", expected_ar="مشتقة ReLU 0/1؛ نسبة التفعيل تهبط إلى 0 مع b = −8 ومتوسط التدرج 0؛ عامل التدرج عبر 10 طبقات = 1."),
        explain=[("3-5", "الثلاثة وصيغ مشتقاتها."), ("7-8", "عند السالب: ReLU 0 (ميت)، Leaky 0.1 (حي)، ELU ناعم."),
                 ("11-15", "انحياز −8: لا مدخل يجعل z موجبًا؛ `mean gradient = 0` — الوحدة ميتة ولا شيء يوقظها لأن تحديث الانحياز نفسه يحتاج تدرجًا."),
                 ("18", "المقارنة مع Sigmoid (1e-9): الوحدات النشطة تمرر التدرج كاملًا.")],
        run=run_printed(CODE),
        after_ar="- الموت غالبًا نتيجة معدل تعلم كبير في بداية التدريب؛ الحلول: معدل أصغر، Leaky/ELU، أو تهيئة He.\n- نسبة الوحدات النشطة (~50% عند b=0) مؤشر صحي؛ راقبها في التشخيص.",
    ))
    why("لماذا يبقى ReLU الافتراضي رغم الموت؟ لأنه أسرع، وبتهيئة جيدة (He) ومعدل تعلم معقول يموت القليل فقط، والشبكة تحتمل ذلك. البدائل تُجرَّب عندما يظهر الموت في التشخيص.")
    common_mistake("ReLU في طبقة الإخراج لانحدار هدفه قد يكون سالبًا (تغير الأسعار): النموذج لا يستطيع أبدًا التنبؤ بسالب. الإخراج خطي.")
    quiz("act.relu", [
        Q("مشتقة ReLU لـ z = −0.5…", ["−0.5", "0", "1"], 1, "الجانب المسطح."),
        Q("وحدة ReLU ميتة…", ["تتعافى تدريجيًا", "لا تتعافى لأن تدرجها صفر", "تصبح Sigmoid"], 1, "لا إشارة."),
        Q("أبسط علاج لـ Dead ReLU…", ["زيادة معدل التعلم", "Leaky ReLU أو معدل تعلم أصغر", "Sigmoid"], 1, "ميل صغير للسالب."),
        Q("عامل التدرج عبر 20 طبقة ReLU نشطة…", ["1", "0.25^20", "20"], 0, "لا تلاشٍ من التنشيط."),
    ])
    takeaway("ReLU: مشتقة 1 للموجب فلا تلاشٍ، 0 للسالب فخطر موت. Leaky/ELU/SELU يعالجان الموت. ReLU للمخفية افتراضيًا، لا للإخراج الحر.")
    lesson_footer(LESSON, ["max(0, z) وقطعيّة الخطية.", "Dead ReLU: انحياز سالب كبير + معدل تعلم كبير.", "راقب نسبة التفعيل."])
