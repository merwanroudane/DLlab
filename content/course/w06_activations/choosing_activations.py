import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components import svgkit as K
from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, intuition, research_note, takeaway, warning_note, why
from components.comparison import compare_table, good_vs_bad
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.fw import keras_mlp_run

LESSON = Lesson(
    id="course.w06.choosing_activations",
    title_ar="اختيار التنشيط المناسب حسب المهمة: المخرج والمخفي، ومقارنة حقيقية + الاختبار البعدي",
    title_en="Choosing the Right Activation by Task: Output & Hidden, a Real Comparison — & Post-test",
    module="course.w06",
    order=3,
    prerequisites=["course.w06.functions_derivatives", "foundations.loss.loss_activation_compatibility", "foundations.activations.softmax_output"],
    objectives_ar=[
        "قاعدة المخرج كشجرة قرار (تحريك): المهمة تحدد التنشيط والخسارة معًا.",
        "قاعدة المخفي والتهيئة المرافقة: ReLU افتراضيًا، وبدائله عند مشاكل محددة.",
        "مقارنة sigmoid/tanh/ReLU في المخفي على شبكة Keras حقيقية، والاختبار البعدي.",
    ],
    terms=["softmax", "cross_entropy", "probability", "sigmoid", "relu", "activation_function", "regression", "classification"],
    labs=["labs.activation_lab", "labs.loss_lab"],
    difficulty="intermediate",
    summary_ar="المخرج تفرضه المهمة مع خسارته (زوج لا يفترق). المخفي: ReLU افتراضيًا، Leaky/ELU عند الموت، tanh داخل RNN، sigmoid لا. التنشيط لا يعالج التعميم.",
)

_TREE = [
    ("target?", "**السؤال الأول: ما الهدف y؟** رقم متصل أم فئة؟ هذا السؤال وحده يحدد نصف الجواب.", K.VIOLET),
    ("number", "**رقم** ⇒ انحدار: وحدة (أو k وحدات) **بلا تنشيط** + `mse`/`mae`/`huber`. إن كان موجبًا دائمًا (سعر، طلب): حوّل الهدف بـ log أو استعمل softplus.", K.EMERALD),
    ("2 classes", "**فئتان** ⇒ وحدة واحدة **sigmoid** + `binary_crossentropy`. القرار بعتبة (ليست 0.5 بالضرورة — الأسبوع 02).", K.BLUE),
    ("K exclusive", "**K فئات متنافية** (واحدة فقط صحيحة) ⇒ K وحدات **softmax** + `sparse_categorical_crossentropy` (هدف أعداد صحيحة) أو `categorical_crossentropy` (one-hot). القرار argmax.", K.AMBER),
    ("K labels", "**K وسوم** (عدة صحيحة معًا، مثل وسوم مقال) ⇒ K وحدات **sigmoid** مستقلة + `binary_crossentropy` لكل وسم.", K.PINK),
    ("hidden", "**الطبقات المخفية** سؤال منفصل تمامًا: ReLU افتراضيًا مهما كانت المهمة.", K.CYAN),
]


def _tree_svg(k: int) -> str:
    s = K.svg_open(700, 240)
    s += K.box(280, 12, 140, 40, "what is y?", K.VIOLET, filled=k == 0, size=12)
    leaves = [(20, "number", "linear + mse", K.EMERALD), (190, "2 classes", "sigmoid + BCE", K.BLUE), (360, "K exclusive", "softmax + CCE", K.AMBER), (530, "K labels", "K sigmoid + BCE", K.PINK)]
    for i, (x, name, rule, col) in enumerate(leaves):
        on = k == i + 1
        s += K.arrow(350, 52, x + 75, 98, color=col if k >= i + 1 else "#CBD5E1")
        s += K.box(x, 100, 150, 40, name, col, filled=on, size=12)
        if k >= i + 1:
            s += K.box(x, 150, 150, 36, rule, col, size=11)
    if k >= 5:
        s += K.box(200, 200, 300, 34, "hidden layers: ReLU (default)", K.CYAN, filled=True, size=12)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    why("سؤالان منفصلان: (1) ماذا في **المخرج**؟ الجواب يفرضه نوع الهدف ويُقرن بالخسارة — خطأ هنا خطأ صامت. (2) ماذا في **الطبقات المخفية**؟ الجواب افتراضي (ReLU) مع استثناءات لمشاكل التدريب. لا تخلط السؤالين.")
    h2("قاعدة المخرج كشجرة قرار", "The output rule as a decision tree")
    animation_player("w06_tree", [Frame(_tree_svg(i), caption(t), action=n) for i, (n, t, _) in enumerate(_TREE)], title_ar="من الهدف إلى التنشيط والخسارة", interval_ms=2400)
    compare_table(["المهمة", "الهدف", "وحدات المخرج", "التنشيط", "الخسارة (Keras)", "القرار"],
                  [("انحدار", "عددي", "1 (أو k)", "بلا (linear)", "mse / mae / huber", "القيمة نفسها"), ("انحدار موجب فقط (سعر، طلب)", "عددي ≥ 0", "1", "بلا + هدف محوَّل log، أو softplus", "mse على log", "عكس التحويل"),
                   ("تصنيف ثنائي", "0/1", "1", "sigmoid", "binary_crossentropy", "عتبة (الأسس 18)"), ("متعدد الفئات متنافٍ", "0..K−1", "K", "softmax", "sparse_categorical_crossentropy", "argmax"),
                   ("متعدد الوسوم (عدة صحيحة معًا)", "متجه 0/1 بطول K", "K", "sigmoid لكل وحدة", "binary_crossentropy", "عتبة لكل وسم"), ("احتمال محدود بين حدين", "في [a, b]", "1", "sigmoid ثم تحجيم", "mse", "—")],
                  ["rtl", "rtl", "num", "code", "code", "rtl"])
    good_vs_bad("متعدد الفئات", "`Dense(K, softmax)` + `sparse_categorical_crossentropy` مع هدف أعداد صحيحة.", "خلط", "`Dense(K, sigmoid)` + `categorical_crossentropy`: الاحتمالات لا تجمع إلى 1 والخسارة تفترض ذلك — يعمل بصمت ويعطي نتائج أسوأ.",
                good_code="layers.Dense(5, activation='softmax')\nmodel.compile(loss='sparse_categorical_crossentropy')", bad_code="layers.Dense(5, activation='sigmoid')\nmodel.compile(loss='categorical_crossentropy')   # silent mismatch",
                verdict_ar="التنشيط والخسارة زوج لا يفترق؛ جدول التوافق في الأسس 13.")
    compare_table(["أمثلة اقتصادية/إدارية", "نوع المخرج", "الإعداد"],
                  [("سعر عقار", "رقم موجب", "`Dense(1)` على log(price) + mse"), ("تعثر العميل", "فئتان", "`Dense(1, 'sigmoid')` + BCE"), ("قسم الشكوى (5 أقسام)", "K متنافية", "`Dense(5, 'softmax')` + sparse CCE"),
                   ("وسوم مقال اقتصادي (تضخم، بطالة، تجارة…)", "K وسوم", "`Dense(K, 'sigmoid')` + BCE"), ("الطلب على 3 منتجات معًا", "3 أرقام", "`Dense(3)` + mse")],
                  ["rtl", "rtl", "code"])
    h2("قاعدة المخفي", "The hidden rule")
    compare_table(["الحالة", "الاختيار", "لماذا"],
                  [("افتراضي (MLP، CNN)", "ReLU", "بسيط، بلا تشبع للموجب، سريع"), ("وحدات ميتة كثيرة (مخرجات صفر ثابتة)", "Leaky ReLU / ELU", "ميل للسالب يبقي التدرج"), ("شبكة كثيفة عميقة جدًا بلا BN", "SELU + lecun_normal", "تطبيع ذاتي (بشروط)"),
                   ("داخل RNN/LSTM/GRU", "tanh (وsigmoid للبوابات)", "مدى محدود يثبّت الحالة (الأسابيع 10–12)"), ("نماذج حديثة كبيرة", "Swish / GELU", "أداء أفضل قليلًا تجريبيًا"), ("sigmoid في المخفي", "تجنّبها", "تشبع + مشتقة ≤ 0.25 = تلاشٍ")],
                  ["rtl", "code", "rtl"])
    compare_table(["التنشيط", "التهيئة المرافقة في Keras", "لماذا"],
                  [("ReLU / Leaky", "`he_normal` أو `he_uniform`", "تعوّض أن نحو نصف المخرجات صفر فتبقي التباين ثابتًا عبر الطبقات"), ("tanh / sigmoid", "`glorot_uniform` (الافتراضي)", "متماثلة حول الصفر"), ("SELU", "`lecun_normal`", "شرط التطبيع الذاتي")],
                  ["code", "code", "rtl"])
    h2("مقارنة حقيقية في الطبقات المخفية", "A real hidden-activation comparison")
    c1, c2 = st.columns(2)
    with c1:
        depth = st.slider("عدد الطبقات المخفية", 1, 4, 2, key="w06_depth")
    with c2:
        epochs = st.slider("epochs", 5, 40, 10, 5, key="w06_ep")
    try:
        fig = go.Figure(); rows = []
        for act, color in (("sigmoid", "#2563EB"), ("tanh", "#7C3AED"), ("relu", "#059669")):
            r = keras_mlp_run(tuple([16] * depth), act, "adam", 0.003, int(epochs), 32, 0)
            e = np.arange(1, len(r["history"]["loss"]) + 1)
            fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name=act, line=dict(color=color, width=2.5)))
            rows.append((act, f"{r['history']['loss'][-1]:.3f}", f"{r['history']['val_loss'][-1]:.3f}", f"{r['history']['val_accuracy'][-1]:.3f}"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="epoch", yaxis_title="val_loss", legend=dict(orientation="h", x=0, y=1.15))
        st.plotly_chart(fig, width="stretch", key="w06_cmp_fig")
        table(["التنشيط المخفي", "loss", "val_loss", "val_accuracy"], rows, ["code", "num", "num", "num"])
    except Exception as exc:  # noqa: BLE001 - TensorFlow missing on this runtime
        warning_note(f"TensorFlow غير متاح هنا ({type(exc).__name__}). تحريك التلاشي في الدرس السابق يعرض نفس الأثر بمحرك NumPy.")
    intuition("بعمق 1 الفروق صغيرة. بعمق 4 يتأخر sigmoid بوضوح (تلاشٍ عبر الطبقات) وtanh في الوسط وReLU أسرع. جرّب η أصغر (الأسبوع 05) لترى أن sigmoid يحتاج حقبًا أكثر بكثير لا أنه «لا يعمل».")
    research_note("في الجداول الصغيرة (مشاريع المقرر) أثر التنشيط المخفي أقل من أثر التحجيم وη والبنية. أعطه الأولوية فقط عند علامات محددة: وحدات ميتة، تدريب بطيء مع العمق، أو مخرجات متشبعة.")
    common_mistake("تغيير التنشيط المخفي لعلاج فرط التخصيص: التنشيط لا يعالج التعميم (الأسبوع 02/الأسس 19–20). علاجه التنظيم والبيانات والحجم.")
    common_mistake("انحدار على سعر بمخرج sigmoid: كل التنبؤات محصورة بين 0 و1 فلا يتعلم النموذج شيئًا مفيدًا. الانحدار = بلا تنشيط في المخرج.")
    with st.container(horizontal=True):
        st.button("معمل التنشيط", icon=":material/science:", on_click=goto, args=("labs.activation_lab",), key="w06_lab_act2")
        st.button("معمل الخسارة", icon=":material/science:", on_click=goto, args=("labs.loss_lab",), key="w06_lab_loss")
    post_test("course.w06", "06", [
        Q("مشتقة sigmoid القصوى:", ["1", "0.25", "0.5"], 1, "عند z = 0."),
        Q("ReLU الميت:", ["وحدة مخرجها دائمًا 0 وتدرجها 0", "وحدة بطيئة", "وحدة بلا انحياز"], 0, ""),
        Q("مخرج تصنيف 5 فئات متنافية:", ["sigmoid", "softmax", "ReLU"], 1, ""),
        Q("Tanh مخرجه في", ["(0,1)", "(−1,1)", "(0,∞)"], 1, ""),
        Q("متعدد الوسوم (عدة فئات صحيحة معًا): المخرج", ["softmax", "sigmoid لكل وحدة + BCE", "linear"], 1, "الوسوم مستقلة."),
        Q("شبكة عميقة بـ sigmoid في المخفي تعاني من…", ["انفجار", "تلاشي التدرج", "ReLU الميت"], 1, ""),
        Q("ظهور وحدات ميتة كثيرة → جرّب", ["sigmoid", "Leaky ReLU / ELU", "softmax"], 1, ""),
        Q("الانحدار على سعر: تنشيط المخرج", ["sigmoid", "بلا تنشيط (linear)", "softmax"], 1, ""),
        Q("التهيئة المناسبة مع ReLU:", ["glorot", "he_normal", "zeros"], 1, ""),
        Q("softmax بفئتين يكافئ…", ["ReLU", "sigmoid على الفرق بين logits", "tanh"], 1, ""),
    ])
    takeaway("المخرج تفرضه المهمة مع خسارته (زوج لا يفترق): رقم → linear+mse، فئتان → sigmoid+BCE، K متنافية → softmax+CCE، K وسوم → K sigmoid+BCE. المخفي: ReLU+He افتراضيًا، Leaky/ELU عند الموت، tanh داخل RNN، sigmoid لا. التنشيط لا يعالج التعميم.")
    lesson_footer(LESSON, ["شجرة قرار المخرج (تحريك) وجدولها وأمثلة اقتصادية.", "قاعدة المخفي والتهيئة المرافقة.", "مقارنة حقيقية والاختبار البعدي."])
