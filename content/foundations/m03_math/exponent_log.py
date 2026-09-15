import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.math.exponent_log",
    title_ar="الأس والدالة الأسية واللوغاريتم الطبيعي",
    title_en="Exponents, the Exponential & the Natural Log",
    module="foundations.math",
    order=3,
    prerequisites=["foundations.math.functions"],
    objectives_ar=[
        "قراءة $x^n$ و$e^x$ و$\\ln x$ وفهم أن اللوغاريتم عكس الأس.",
        "معرفة خواص $e^x$ (موجب دائمًا، يكبر بسرعة) و$\\ln x$ (معرّف للموجب فقط، يضغط الكبير ويمدّ الصغير).",
        "رؤية لماذا Softmax تستخدم $e^x$ ولماذا الإنتروبيا المتقاطعة تستخدم $\\ln$.",
    ],
    terms=["loss"],
    difficulty="beginner",
    summary_ar="e^x يحوّل أي عدد إلى موجب؛ ln يحوّل احتمالًا إلى «مفاجأة». Softmax والإنتروبيا مبنيتان عليهما.",
)

CODE = '''import numpy as np

print(2 ** 3, 2 ** -1, 9 ** 0.5)               # أسس
print(np.e, np.exp(1), np.exp(0), np.exp(-2))    # الدالة الأسية
print(np.log(np.e), np.log(1), np.log(0.5))      # اللوغاريتم الطبيعي (ln)
print(np.exp(np.log(7.0)), np.log(np.exp(3.0)))  # عكس بعضهما

# Softmax: تحويل أرقام حرة إلى احتمالات موجبة مجموعها 1
logits = np.array([2.0, 1.0, -1.0])
probs = np.exp(logits) / np.exp(logits).sum()
print(probs.round(4), probs.sum())

# الإنتروبيا المتقاطعة لمثال واحد: -log(احتمال الفئة الصحيحة)
for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
    print(f"p={p:<5} loss=-log(p)={-np.log(p):.3f}")

print(np.log(0))            # -inf مع تحذير: سبب خسارة NaN/inf الشهير
print(-np.log(1e-7))        # الحل: قصّ الاحتمال بعيدًا عن الصفر'''


def render() -> None:
    lesson_header(LESSON)
    h2("الأس", "Exponent")
    definition("$x^n$ هو $x$ مضروبًا في نفسه $n$ مرة. $x^{-1} = 1/x$ و$x^{1/2} = \\sqrt{x}$ و$x^0 = 1$.")
    h2("الدالة الأسية", "The exponential")
    equation(r"f(x) = e^{x}, \qquad e \approx 2.718", [("e", "ثابت أويلر؛ الأساس «الطبيعي» لأن مشتقة $e^x$ هي $e^x$ نفسها."), ("x", "أي عدد حقيقي، سالبًا أو موجبًا.")],
             meaning_ar="مخرج **موجب دائمًا**: $e^{-100}$ صغير جدًا لكنه أكبر من الصفر، و$e^{10} \\approx 22026$ يكبر بسرعة هائلة.",
             example_ar="$e^0 = 1$، $e^1 \\approx 2.72$، $e^{-2} \\approx 0.135$.",
             dl_link_ar="Softmax تأخذ أرقامًا حرة (`logits`) وتمررها في $e^x$ لتصبح موجبة، ثم تقسم على المجموع لتصبح احتمالات.", title_ar="الدالة الأسية")
    h2("اللوغاريتم الطبيعي", "Natural log")
    equation(r"\ln(x) = y \iff e^{y} = x, \qquad x > 0", [("\\ln", "اللوغاريتم الطبيعي (أساسه $e$)؛ في الكود `np.log`."), ("x", "يجب أن يكون **موجبًا**؛ $\\ln 0 = -\\infty$ وللسالب غير معرّف.")],
             meaning_ar="عكس الأس. $\\ln 1 = 0$، الأعداد الأصغر من 1 لوغاريتمها سالب، ويكبر ببطء شديد للأعداد الكبيرة.",
             example_ar="$\\ln(e) = 1$، $\\ln(0.5) \\approx -0.69$، $\\ln(1000) \\approx 6.9$.",
             dl_link_ar="خسارة التصنيف $-\\ln(p_{\\text{correct}})$: احتمال 0.99 ← خسارة 0.01؛ احتمال 0.01 ← خسارة 4.6. اللوغاريتم يعاقب الثقة الخاطئة بشدة.", title_ar="اللوغاريتم الطبيعي")
    intuition("$-\\ln p$ هو «مقدار المفاجأة»: حدث احتماله 1 لا يفاجئ (0)، وحدث احتماله شبه صفر يفاجئ بلا حدود. الخسارة تقيس مفاجأة النموذج من الإجابة الصحيحة.")
    h3("الشكل", "The curves")
    x1 = np.linspace(-3, 3, 200); x2 = np.linspace(0.02, 3, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=np.exp(x1), name="e^x", line=dict(color="#7C5CBF", width=3)))
    fig.add_trace(go.Scatter(x=x2, y=np.log(x2), name="ln(x)", line=dict(color="#C8473A", width=3)))
    fig.add_trace(go.Scatter(x=x2, y=-np.log(x2), name="−ln(x)  (loss)", line=dict(color="#C77A1A", width=2, dash="dot")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[-4, 8]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="explog_fig")
    table(["الخاصية", "e^x", "ln(x)"],
          [("المجال", "كل الأعداد", "الموجب فقط"), ("المدى", "الموجب فقط", "كل الأعداد"), ("عند 0 / 1", "e^0 = 1", "ln(1) = 0"),
           ("الاتجاه", "متزايدة بسرعة", "متزايدة ببطء"), ("العلاقة", "ln(e^x) = x", "e^{ln x} = x")], ["rtl", "rtl", "rtl"])
    code_lab(CodeLab(
        key="math_explog", title_ar="الأس واللوغاريتم ← Softmax والخسارة", code=CODE,
        before=Before(goal_ar="حساب الأسس واللوغاريتمات، ثم بناء Softmax وخسارة −log من الصفر ورؤية فخ log(0).", stage_ar="رياضيات أساسية ← الخسارة.",
                      inputs_ar="أرقام صغيرة و`logits` ثلاثة.", expected_ar="قيم أسية ولوغاريتمية، احتمالات مجموعها 1، جدول خسارة يتزايد كلما صغر الاحتمال، و`-inf` مع تحذير."),
        explain=[("3-6", "الأساسيات؛ لاحظ أن `exp` و`log` يلغي أحدهما الآخر."),
                 ("9-11", "Softmax في سطر: `exp` ثم القسمة على المجموع. الأكبر يحصل على أكبر احتمال، والمجموع 1 دائمًا."),
                 ("14-15", "الخسارة لمثال واحد. $p=0.5$ يعطي 0.693 (= ln 2)؛ $p = 0.01$ يعطي 4.6."),
                 ("17-18", "`log(0)` = `-inf`؛ إن وصل احتمال الفئة الصحيحة إلى صفر بالضبط تصبح الخسارة لانهائية ثم `NaN`. الأطر تقصّ الاحتمال عند `1e-7` تقريبًا.")],
        run=run_printed(CODE),
        after_ar="- `probs.sum()` يساوي 1.0 (قد يظهر 0.9999999 بسبب دقة التمثيل العشري — طبيعي).\n- سلسلة الخسارة 0.01 → 0.105 → 0.693 → 2.3 → 4.6: ليست خطية؛ الثقة الخاطئة مكلفة جدًا.",
    ))
    debugging_note("خسارة `nan` أو `inf` من أول حقبة في التصنيف؟ غالبًا `log(0)`: احتمال صار صفرًا بالضبط (مثلًا Sigmoid مشبعة أو مخرج بلا تنشيط ممرر إلى خسارة تتوقع احتمالات). استخدم `from_logits=True` أو قصّ الاحتمالات.")
    common_mistake("الخلط بين `np.log` (طبيعي، أساس e) و`np.log10`. في التعلم العميق «log» دائمًا طبيعي ما لم يُذكر غير ذلك.")
    quiz("math.explog", [
        Q("$e^x$ لأي $x$ يكون…", ["موجبًا دائمًا", "أكبر من 1 دائمًا", "أقل من 1 أحيانًا سالبًا"], 0, "المدى (0, ∞)."),
        Q("$-\\ln(0.01)$ تقريبًا…", ["0.01", "4.6", "−4.6"], 1, "مفاجأة كبيرة.", kind="equation"),
        Q("لماذا Softmax تستخدم $e^x$؟", ["لأنه سريع", "ليصبح كل مخرج موجبًا قبل القسمة على المجموع", "لأنه خطي"], 1, "احتمالات تحتاج موجبًا."),
        Q("خسارة تصنيف = inf في الحقبة الأولى. السبب الأرجح؟", ["معدل تعلم صغير", "log(0): احتمال صار صفرًا", "بيانات كثيرة"], 1, "قصّ الاحتمالات / from_logits.", kind="error"),
    ])
    takeaway("e^x: كل عدد ← موجب (Softmax). ln: احتمال ← مفاجأة (خسارة). ln(0) = −∞ هو مصدر NaN الكلاسيكي.")
    lesson_footer(LESSON, ["x^n, e^x, ln x وعلاقة العكس.", "Softmax = exp / sum(exp).", "−ln p يعاقب الثقة الخاطئة؛ احذر log(0)."])
