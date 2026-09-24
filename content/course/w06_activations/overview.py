import numpy as np
import streamlit as st

from components import svgkit as K
from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w06.overview",
    title_ar="نظرة عامة على الأسبوع 06 والاختبار القبلي",
    title_en="Week 06 Overview & Pre-test",
    module="course.w06",
    order=1,
    prerequisites=["course.w05.loss_surface_convergence", "foundations.activations.why_nonlinearity"],
    objectives_ar=["رؤية لماذا نحتاج اللاخطية أصلًا: طبقات خطية متراكمة تنطوي في طبقة واحدة (تحريك).", "خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على دوال التنشيط."],
    terms=["softmax", "derivative", "activation_function", "relu", "sigmoid"],
    difficulty="beginner",
    summary_ar="بلا تنشيط تنطوي الطبقات في خط واحد. الأسبوع: الدوال ومشتقاتها، التشبع والتلاشي، ReLU الميت، softmax، والاختيار حسب المهمة.",
)


def _fold_svg(stage: int) -> str:
    """1-D input x → two linear layers (no activation) vs with ReLU: the output curve."""
    s = K.svg_open(700, 250)
    xs = np.linspace(-3, 3, 120)
    w1, b1 = np.array([1.0, -1.0, 1.5]), np.array([0.5, 0.5, -1.0])
    w2 = np.array([1.2, 0.8, -1.5])
    lin = (xs[:, None] * w1 + b1) @ w2
    relu = np.maximum(0, xs[:, None] * w1 + b1) @ w2
    P = K.Panel(60, 30, 280, 190, (-3, 3, -6, 6))
    Q_ = K.Panel(390, 30, 280, 190, (-3, 3, -6, 6))
    s += P.frame("no activation: f(x) = W2(W1 x + b1)", "x")
    s += Q_.frame("with ReLU: f(x) = W2 ReLU(W1 x + b1)", "x")
    if stage >= 1:
        s += P.polyline(xs, lin, K.BLUE, 3)
        s += K.text(200, 240, "always a straight line", size=12, bold=True, color=K.BLUE)
    if stage >= 2:
        s += Q_.polyline(xs, relu, K.PINK, 3)
        s += K.text(530, 240, "bends at each ReLU hinge", size=12, bold=True, color=K.PINK)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Sigmoid · Tanh · Softmax", "ReLU · LeakyReLU", "Swish · ELU · SELU", "Derivatives & saturation", "Dead ReLU", "Choosing by task", "Post-test"], active=0)
    h2("لماذا التنشيط أصلًا؟", "Why activations at all?")
    caps = ["**نفس الشبكة الصغيرة مرتين**: مدخل واحد x، طبقة مخفية من 3 وحدات، مخرج واحد. الأوزان نفسها في الحالتين.",
            "**بلا تنشيط**: تركيب دالتين خطيتين = دالة خطية: `W2(W1x + b1) = (W2W1)x + W2b1`. مهما أضفت من طبقات يبقى الناتج خطًا مستقيمًا — العمق بلا فائدة.",
            "**مع ReLU**: كل وحدة تنكسر عند نقطة مختلفة، والمخرج يجمع الخطوط المكسورة فينحني. هذه هي القدرة التي نشتريها بالتنشيط (الأسبوع 04: التقريب الشامل)."]
    animation_player("w06_why", [Frame(_fold_svg(i), caption(c), action=["same weights", "linear", "ReLU"][i]) for i, c in enumerate(caps)],
                     title_ar="طبقتان خطيتان تنطويان في خط واحد", interval_ms=2600)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("Sigmoid وTanh وSoftmax", "الدوال ومشتقاتها", "الأسس 11: sigmoid/tanh، softmax"), ("ReLU وLeaky ReLU وELU وSELU وSwish", "الدوال ومشتقاتها", "الأسس 11: عائلة ReLU، Swish/GELU"),
           ("المشتقات والتشبع وتلاشي التدرج وReLU الميت", "الدوال ومشتقاتها", "الأسس 11: المشتقات والتشبع؛ الأسس 14: التلاشي"), ("الاختيار حسب المهمة", "اختيار التنشيط", "الأسس 11 و13: توافق المخرج والخسارة")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تقرأ رسم مشتقة أي دالة تنشيط وتتنبأ بما يمر من التدرج.\n"
        "- أن تشرح تلاشي التدرج كحاصل ضرب وتقيسه طبقة طبقة.\n"
        "- أن تكتشف الوحدات الميتة وتعالجها.\n"
        "- أن تختار تنشيط المخرج وخسارته من نوع الهدف، وتنشيط المخفي وتهيئته."
    )
    practical_note("الأسس 11 تشرح كل دالة؛ هذا الأسبوع يقيس أثرها على شبكة عميقة حقيقية ويعطيك قاعدة الاختيار.")
    with st.container(horizontal=True):
        st.button("الأسس 11 — التنشيط", icon=":material/menu_book:", on_click=go, args=("foundations.activations",), key="w06_go_act")
        st.button("معمل التنشيط", icon=":material/science:", on_click=go, args=("labs.activation_lab",), key="w06_go_lab")
    intuition("التنشيط يؤدي وظيفتين متعارضتين: يصنع الانحناء (أماميًا)، ويتحكم في مرور التدرج (خلفيًا). الدالة الجيدة تنحني دون أن تخنق التدرج.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w06.pretest", [
        Q("مشتقة sigmoid القصوى:", ["1", "0.25", "0.5"], 1, "عند z = 0: 0.5 × 0.5."),
        Q("ReLU الميت:", ["وحدة مخرجها دائمًا 0 وتدرجها 0", "وحدة بطيئة", "وحدة بلا انحياز"], 0, ""),
        Q("تنشيط المخرج لتصنيف 5 فئات متنافية:", ["sigmoid", "softmax", "ReLU"], 1, ""),
        Q("Tanh مقابل sigmoid: المخرج", ["(0,1)", "(−1,1)", "(0,∞)"], 1, ""),
        Q("ثلاث طبقات Dense بلا تنشيط تكافئ…", ["شبكة عميقة", "طبقة خطية واحدة", "لا شيء"], 1, "التركيب الخطي خطي."),
    ], title_ar="الاختبار القبلي — الأسبوع 06")
    takeaway("الأسبوع 06 = الدوال بمشتقاتها، أثرها على العمق، وقاعدة اختيار بحسب المهمة.")
    lesson_footer(LESSON, ["لماذا التنشيط (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
