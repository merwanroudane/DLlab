import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.optim.loss_surface",
    title_ar="سطح الخسارة: الحد الأدنى المحلي والعام، نقطة السرج، المحدب وغير المحدب",
    title_en="The Loss Surface: Local/Global Minima, Saddle Points, Convex vs Nonconvex",
    module="foundations.optim",
    order=1,
    prerequisites=["foundations.calculus.partial_gradient", "foundations.calculus.jacobian_hessian"],
    objectives_ar=["تخيل الخسارة كسطح فوق فضاء المعلمات والتدريب كنزول عليه.", "التمييز بين الحد الأدنى المحلي والعام ونقطة السرج والهضبة.", "فهم لماذا سطح الشبكات غير محدب وماذا يعنيه ذلك عمليًا."],
    terms=["gradient", "loss", "hessian"],
    labs=["labs.gradient_lab"],
    difficulty="intermediate",
    summary_ar="الخسارة سطح فوق المعلمات؛ التدريب نزول؛ الشبكات غير محدبة: حدود محلية وسروج وهضاب — وغالبًا يكفي حد محلي جيد.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الصورة الذهنية", "The mental picture")
    definition("**سطح الخسارة** `Loss surface`: الدالة $L(\\theta)$ لكل قيم المعلمات. بمعلمتين هو تضاريس فوق مستوى؛ بمليون معلمة لا نراه لكن الكلمات نفسها تنطبق. **التدريب** = السير على هذا السطح نزولًا.")
    compare_table(["المصطلح", "English", "التعريف", "التدرج هناك", "الهسي"],
                  [("حد أدنى محلي", "Local minimum", "أدنى من كل جيرانه", "0", "قيم ذاتية موجبة"), ("حد أدنى عام", "Global minimum", "أدنى نقطة على السطح كله", "0", "موجبة"),
                   ("نقطة سرج", "Saddle point", "أدنى في اتجاه وأعلى في آخر", "0", "مختلطة الإشارة"), ("هضبة", "Plateau", "منطقة شبه مستوية", "≈ 0", "≈ 0"),
                   ("محدب", "Convex", "وعاء واحد: أي حد محلي عام", "—", "موجبة في كل مكان"), ("غير محدب", "Nonconvex", "وديان وقمم متعددة", "—", "متغيرة")],
                  ["rtl", "ltr", "rtl", "code", "rtl"])
    h2("شاهد السطوح", "See the surfaces")
    kind = st.segmented_control("السطح", ["محدب (وعاء)", "غير محدب (وديان)", "سرج", "هضبة"], default="غير محدب (وديان)", key="ls_kind")
    g = np.linspace(-3, 3, 70); X, Y = np.meshgrid(g, g)
    if kind.startswith("محدب"):
        Z = X ** 2 + 0.5 * Y ** 2
    elif kind.startswith("غير"):
        Z = np.sin(1.5 * X) * np.cos(1.5 * Y) + 0.1 * (X ** 2 + Y ** 2)
    elif kind.startswith("سرج"):
        Z = X ** 2 - Y ** 2
    else:
        Z = np.tanh(0.3 * X) ** 2 + 0.02 * Y ** 2
    fig = go.Figure(go.Surface(x=g, y=g, z=Z, colorscale="YlGnBu", showscale=False))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), scene=dict(xaxis_title="θ₁", yaxis_title="θ₂", zaxis_title="L"), paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="ls_fig")
    intuition("ادفع الرسم بالفأرة. في الوعاء كل طريق نزول يصل إلى القاع نفسه. في الوديان يصل إلى **أقرب** قاع، والتهيئة تحدد أيّها. عند السرج يتوقف التدرج مؤقتًا رغم أن الطريق مفتوح في اتجاه آخر. على الهضبة الخطوات صغيرة جدًا لأن الميل شبه معدوم.")
    research_note("لماذا لا تعلق الشبكات العميقة في حدود محلية سيئة كما يُخشى؟ نتائج نظرية وتجريبية تشير إلى أن أغلب النقاط الحرجة في الأبعاد العالية سروج لا حدود دنيا، وأن الحدود المحلية التي يصل إليها SGD متقاربة الجودة. العدو العملي هو السروج والهضاب (بطء)، لا الحدود السيئة. الزخم وAdam مصممان جزئيًا لعبورها.")
    common_mistake("«الخسارة توقفت عن الانخفاض إذن وصلنا إلى الحد الأدنى». قد تكون هضبة أو سرج: راقب معيار التدرج وجرّب زخمًا أو رفع معدل التعلم قليلًا قبل الحكم.")
    st.button("افتح معمل التدرج (سطوح متعددة)", icon=":material/science:", on_click=goto, args=("labs.gradient_lab",), key="ls_lab")
    quiz("optim.surface", [
        Q("نقطة تدرجها صفر وهسيها مختلط الإشارة…", ["حد أدنى", "سرج", "هضبة"], 1, "أدنى في اتجاه وأعلى في آخر."),
        Q("في دالة محدبة…", ["حدود محلية كثيرة", "أي حد محلي هو العام", "لا حدود"], 1, "وعاء واحد."),
        Q("سطح الشبكات العصبية…", ["محدب دائمًا", "غير محدب", "خطي"], 1, "وديان وسروج."),
        Q("الخسارة ثابتة ومعيار التدرج ≈ 0 مبكرًا جدًا…", ["حد عام", "هضبة/سرج محتمل", "تسريب"], 1, "قبل الحكم جرّب زخمًا."),
    ])
    takeaway("الخسارة سطح؛ التدريب نزول. غير محدب: أقرب قاع لا أفضلها؛ السروج والهضاب تبطئ أكثر مما تضلل الحدود المحلية.")
    lesson_footer(LESSON, ["مفردات السطح.", "الرسم ثلاثي الأبعاد للأنواع الأربعة.", "التدرج صفر لا يعني القاع."])
