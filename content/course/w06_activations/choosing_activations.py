import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, research_note, takeaway, why
from components.comparison import compare_table, good_vs_bad
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from core.rtl import table
from labs.fw import keras_mlp_run

LESSON = Lesson(
    id="course.w06.choosing_activations",
    title_ar="اختيار التنشيط المناسب حسب المهمة: المخرج والمخفي، ومقارنة حقيقية + الاختبار البعدي",
    title_en="Choosing the Right Activation by Task: Output & Hidden, a Real Comparison — & Post-test",
    module="course.w06",
    order=3,
    prerequisites=["course.w06.functions_derivatives", "foundations.loss.loss_activation_compatibility", "foundations.activations.softmax_output"],
    objectives_ar=["قاعدة المخرج: المهمة تحدد التنشيط والخسارة معًا.", "قاعدة المخفي: ReLU افتراضيًا، وبدائله عند مشاكل محددة.", "مقارنة sigmoid/tanh/ReLU في المخفي على شبكة Keras حقيقية، والاختبار البعدي."],
    terms=["softmax", "cross_entropy", "probability"],
    labs=["labs.activation_lab", "labs.loss_lab"],
    difficulty="intermediate",
    summary_ar="المخرج: انحدار = بلا تنشيط + MSE؛ ثنائي = sigmoid + BCE؛ متعدد = softmax + CCE؛ متعدد الوسوم = sigmoid لكل وسم. المخفي: ReLU افتراضيًا؛ Leaky/ELU عند الموت؛ tanh في RNN.",
)


def render() -> None:
    lesson_header(LESSON)
    why("سؤالان منفصلان: (1) ماذا في **المخرج**؟ الجواب يفرضه نوع الهدف ويُقرن بالخسارة — خطأ هنا خطأ صامت. (2) ماذا في **الطبقات المخفية**؟ الجواب افتراضي (ReLU) مع استثناءات لمشاكل التدريب. لا تخلط السؤالين.")
    h2("قاعدة المخرج", "The output rule")
    compare_table(["المهمة", "الهدف", "وحدات المخرج", "التنشيط", "الخسارة (Keras)", "القرار"],
                  [("انحدار", "عددي", "1 (أو k)", "بلا (linear)", "mse / mae / huber", "القيمة نفسها"), ("انحدار موجب فقط (سعر، طلب)", "عددي ≥ 0", "1", "بلا + هدف محوَّل log، أو softplus", "mse على log", "عكس التحويل"),
                   ("تصنيف ثنائي", "0/1", "1", "sigmoid", "binary_crossentropy", "عتبة (الأسس 18)"), ("متعدد الفئات متنافٍ", "0..K−1", "K", "softmax", "sparse_categorical_crossentropy", "argmax"),
                   ("متعدد الوسوم (عدة صحيحة معًا)", "متجه 0/1 بطول K", "K", "sigmoid لكل وحدة", "binary_crossentropy", "عتبة لكل وسم"), ("احتمال محدود بين حدين", "في [a, b]", "1", "sigmoid ثم تحجيم", "mse", "—")],
                  ["rtl", "rtl", "num", "code", "code", "rtl"])
    good_vs_bad("متعدد الفئات", "`Dense(K, softmax)` + `sparse_categorical_crossentropy` مع هدف أعداد صحيحة.", "خلط", "`Dense(K, sigmoid)` + `categorical_crossentropy`: الاحتمالات لا تجمع إلى 1 والخسارة تفترض ذلك — يعمل بصمت ويعطي نتائج أسوأ.",
                good_code="layers.Dense(5, activation='softmax')\nmodel.compile(loss='sparse_categorical_crossentropy')", bad_code="layers.Dense(5, activation='sigmoid')\nmodel.compile(loss='categorical_crossentropy')   # silent mismatch",
                verdict_ar="التنشيط والخسارة زوج لا يفترق؛ جدول التوافق في الأسس 13.")
    h2("قاعدة المخفي", "The hidden rule")
    compare_table(["الحالة", "الاختيار", "لماذا"],
                  [("افتراضي (MLP، CNN)", "ReLU", "بسيط، بلا تشبع للموجب، سريع"), ("وحدات ميتة كثيرة (مخرجات صفر ثابتة)", "Leaky ReLU / ELU", "ميل للسالب يبقي التدرج"), ("شبكة كثيفة عميقة جدًا بلا BN", "SELU + lecun_normal", "تطبيع ذاتي (بشروط)"),
                   ("داخل RNN/LSTM/GRU", "tanh (وsigmoid للبوابات)", "مدى محدود يثبّت الحالة (الأسابيع 10–12)"), ("نماذج حديثة كبيرة", "Swish / GELU", "أداء أفضل قليلًا تجريبيًا"), ("sigmoid في المخفي", "تجنّبها", "تشبع + مشتقة ≤ 0.25 = تلاشٍ")],
                  ["rtl", "code", "rtl"])
    h2("مقارنة حقيقية في الطبقات المخفية", "A real hidden-activation comparison")
    c1, c2 = st.columns(2)
    with c1:
        depth = st.slider("عدد الطبقات المخفية", 1, 4, 2, key="w06_depth")
    with c2:
        epochs = st.slider("epochs", 5, 40, 10, 5, key="w06_ep")
    fig = go.Figure(); rows = []
    for act, color in (("sigmoid", "#2F6FB5"), ("tanh", "#7C5CBF"), ("relu", "#1F7A78")):
        r = keras_mlp_run(tuple([16] * depth), act, "adam", 0.003, int(epochs), 32, 0)
        e = np.arange(1, len(r["history"]["loss"]) + 1)
        fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name=act, line=dict(color=color, width=2.5)))
        rows.append((act, f"{r['history']['loss'][-1]:.3f}", f"{r['history']['val_loss'][-1]:.3f}", f"{r['history']['val_accuracy'][-1]:.3f}"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="val_loss", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="w06_cmp_fig")
    table(["التنشيط المخفي", "loss", "val_loss", "val_accuracy"], rows, ["code", "num", "num", "num"])
    intuition("بعمق 1 الفروق صغيرة. بعمق 4 يتأخر sigmoid بوضوح (تلاشٍ عبر الطبقات) وtanh في الوسط وReLU أسرع. جرّب η أصغر (الأسبوع 05) لترى أن sigmoid يحتاج حقبًا أكثر بكثير لا أنه «لا يعمل».")
    research_note("في الجداول الصغيرة (مشاريع المقرر) أثر التنشيط المخفي أقل من أثر التحجيم وη والبنية. أعطه الأولوية فقط عند علامات محددة: وحدات ميتة، تدريب بطيء مع العمق، أو مخرجات متشبعة.")
    common_mistake("تغيير التنشيط المخفي لعلاج فرط التخصيص: التنشيط لا يعالج التعميم (الأسبوع 02/الأسس 19–20). علاجه التنظيم والبيانات والحجم.")
    post_test("course.w06", "06", [
        Q("مشتقة sigmoid القصوى:", ["1", "0.25", "0.5"], 1, ""),
        Q("ReLU الميت:", ["وحدة مخرجها دائمًا 0 وتدرجها 0", "وحدة بطيئة", "وحدة بلا انحياز"], 0, ""),
        Q("مخرج تصنيف 5 فئات متنافية:", ["sigmoid", "softmax", "ReLU"], 1, ""),
        Q("Tanh مخرجه في", ["(0,1)", "(−1,1)", "(0,∞)"], 1, ""),
        Q("متعدد الوسوم (عدة فئات صحيحة معًا): المخرج", ["softmax", "sigmoid لكل وحدة + BCE", "linear"], 1, ""),
        Q("شبكة عميقة بـ sigmoid في المخفي تعاني من…", ["انفجار", "تلاشي التدرج", "ReLU الميت"], 1, ""),
        Q("ظهور وحدات ميتة كثيرة → جرّب", ["sigmoid", "Leaky ReLU / ELU", "softmax"], 1, ""),
        Q("الانحدار على سعر: تنشيط المخرج", ["sigmoid", "بلا تنشيط (linear)", "softmax"], 1, ""),
    ])
    takeaway("المخرج تفرضه المهمة مع خسارته (زوج لا يفترق). المخفي: ReLU افتراضيًا، Leaky/ELU عند الموت، tanh داخل RNN، sigmoid لا. التنشيط لا يعالج التعميم.")
    lesson_footer(LESSON, ["قاعدة المخرج وجدولها.", "قاعدة المخفي وبدائلها.", "مقارنة حقيقية والاختبار البعدي."])
