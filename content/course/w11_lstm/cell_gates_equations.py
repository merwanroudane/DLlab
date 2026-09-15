import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.rnn import lstm_forward, random_weights, rnn_forward

LESSON = Lesson(
    id="course.w11.cell_gates_equations",
    title_ar="حالة الخلية والبوابات الثلاث: المعادلات، المعنى، والتحريك بوابةً بوابة",
    title_en="Cell State & the Three Gates: Equations, Meaning, Gate-by-Gate Animation",
    module="course.w11",
    order=3,
    prerequisites=["course.w11.data_prep_windows_padding", "course.w10.rnn_bptt_vanishing", "foundations.activations.sigmoid_tanh"],
    objectives_ar=["المعادلات الست لـ LSTM وتفسير كل رمز.", "معنى كل بوابة بقيمة sigmoid: نسيان/إبقاء، كتابة/تجاهل، كشف/إخفاء.", "لماذا يحفظ مسار الجمع في c_t التدرج — بمقارنة رقمية مع RNN البسيط."],
    terms=["gradient", "hadamard_product"],
    labs=["labs.lstm_gates_lab"],
    difficulty="advanced",
    summary_ar="f, i, o = σ([x, h]W + b)؛ g = tanh(…)؛ c_t = f⊙c_{t−1} + i⊙g؛ h_t = o⊙tanh(c_t). البوابات تتعلم متى تنسى/تكتب/تكشف. c يمر بجمع لا بضرب متكرر: التدرج يبقى.",
)


def render() -> None:
    lesson_header(LESSON)
    definition("**LSTM** (Long Short-Term Memory): خلية تكرارية بحالتين: **حالة الخلية** c_t (ذاكرة طويلة تمر بجمع) و**الحالة المخفية** h_t (ما يُكشف للخارج). ثلاث **بوابات** بقيم في (0, 1) من sigmoid تقرر: **النسيان** f (كم نبقي من c القديمة)، **الإدخال** i (كم نكتب من المرشَّح g)، **الإخراج** o (كم نكشف من c). كل بوابة دالة في [x_t, h_{t−1}] بأوزانها الخاصة.")
    why("RNN البسيط يضرب الحالة في Wh وtanh كل خطوة فيتلاشى أثر الماضي (الأسبوع 10). LSTM تعطي الماضي **طريقًا سريعًا**: c_t = f ⊙ c_{t−1} + … — إذا تعلمت الشبكة f ≈ 1 مرت المعلومة (والتدرج) بلا تلاشٍ عشرات الخطوات.")
    h2("المعادلات", "The equations")
    equation(r"f_t = \sigma\big([x_t, h_{t-1}]\,W_f + b_f\big), \quad i_t = \sigma\big([x_t, h_{t-1}]\,W_i + b_i\big), \quad o_t = \sigma\big([x_t, h_{t-1}]\,W_o + b_o\big)",
             [(r"[x_t, h_{t-1}]", "المدخل الحالي والحالة السابقة متجاوران في متجه واحد بطول D + H."), (r"W_f, W_i, W_o", "ثلاث مصفوفات (D+H)×H: بوابة لكل وظيفة."), (r"\sigma", "sigmoid: القيمة بين 0 (مغلقة) و1 (مفتوحة) لكل مكوّن.")],
             meaning_ar="ثلاث «صمامات» بقيم مستمرة تتعلمها الشبكة من البيانات.", example_ar="f = 0.95 يعني «احتفظ بـ 95% من الذاكرة القديمة».", dl_link_ar="Keras تخزّن الأربع مصفوفات (f, i, c, o) متجاورة في `kernel` بشكل (D, 4H) و`recurrent_kernel` (H, 4H).", title_ar="البوابات الثلاث")
    equation(r"\tilde c_t = \tanh\big([x_t, h_{t-1}]\,W_c + b_c\big), \qquad c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t, \qquad h_t = o_t \odot \tanh(c_t)",
             [(r"\tilde c_t", "المرشَّح: ماذا يمكن كتابته (في (−1, 1))."), (r"c_t", "الذاكرة: جزء من القديم + جزء من الجديد — **جمع**."), (r"h_t", "المخرج: جزء مكشوف من الذاكرة، يعود كمدخل في الخطوة التالية ويُعطى للرأس.")],
             meaning_ar="المعادلة الوسطى هي قلب LSTM: ⊙ ضرب عنصري (الأسس 4) و+ جمع؛ لا Wh متكررة على c.", example_ar="c = 0.95×2.0 + 0.1×0.5 = 1.95: الذاكرة تبقى 1.95 بعد خطوة كاملة.", dl_link_ar="`layers.LSTM(units)`؛ `nn.LSTM` في PyTorch.", title_ar="المرشَّح والذاكرة والمخرج")
    h2("البوابات بالكلمات", "Gates in words")
    compare_table(["البوابة", "≈ 0", "≈ 1", "مثال اقتصادي"],
                  [("النسيان f", "امسح الذاكرة القديمة", "احتفظ بها كاملة", "بداية سنة مالية جديدة: انسَ الموسمية السابقة؟"), ("الإدخال i", "تجاهل هذه الخطوة", "اكتب المرشَّح كاملًا", "صدمة سعرية: اكتبها في الذاكرة"), ("الإخراج o", "لا تكشف شيئًا للمخرج", "اكشف الذاكرة", "الذاكرة مهمة لاحقًا لا الآن: لا تكشف بعد")],
                  ["rtl", "rtl", "rtl", "rtl"])
    h2("لماذا يبقى التدرج: مقارنة رقمية", "Why the gradient survives: a numeric comparison")
    T = st.slider("طول التسلسل", 5, 40, 20, 5, key="w11_T")
    xs = np.zeros(T); xs[0] = 2.0
    Wr = random_weights("rnn", 1, 3, seed=1, scale=0.6); Wl = random_weights("lstm", 1, 3, seed=1, scale=0.6)
    Wl["bf"] = np.full(3, 2.0)                       # بوابة نسيان مفتوحة تقريبًا (σ(2) ≈ 0.88): تحيز شائع في التهيئة
    r_steps = rnn_forward(xs, **Wr); l_steps = lstm_forward(xs, Wl, 3)
    rows = [(str(t), f"{abs(r_steps[t]['h'][0]):.4f}", f"{abs(l_steps[t]['c'][0]):.4f}", f"{l_steps[t]['f'][0]:.2f}") for t in [0, 1, 2, 5, 10, T - 1] if t < T]
    table(["t", "|h_t[0]| في RNN البسيط", "|c_t[0]| في LSTM", "f_t[0]"], rows, ["num", "num", "num", "num"])
    st.markdown("**التجربة**: إشارة واحدة x₀ = 2 ثم أصفار. في RNN البسيط تتلاشى الحالة أسيًا (ضرب متكرر في Wh وtanh)؛ في LSTM تبقى c قريبة من قيمتها لأن f ≈ 0.88 (جمع مع نسيان بطيء). التدرج يسلك المسار نفسه: يُضرب في f لا في Whᵀ·tanh′.")
    st.button("افتح معمل بوابات LSTM (تحريك بوابةً بوابة)", icon=":material/science:", type="primary", on_click=goto, args=("labs.lstm_gates_lab",), key="w11_lab_gates")
    intuition("LSTM ليست «أذكى» من RNN؛ هي RNN مع **مسار جمع** ومفاتيح متعلَّمة. كل ما تعلمته عن المجموع الموزون وsigmoid وtanh والضرب العنصري يُستخدم هنا بلا مفهوم جديد — فقط تركيب جديد.")
    research_note("**تعميق**: تهيئة انحياز النسيان بـ 1 (Keras: `unit_forget_bias=True` افتراضيًا) تجعل f ≈ 0.73 في البداية فتتذكر الشبكة قبل أن تتعلم متى تنسى. المعلمات = 4 × ((D + H) × H + H): أربع مرات RNN البسيط — وهذا ثمن الذاكرة.")
    common_mistake("قراءة c_t وh_t كشيء واحد: h_t هو ما يراه الرأس والخطوة التالية، أما c_t فذاكرة داخلية قد لا تُكشف (o ≈ 0). `LSTM(units, return_state=True)` يعيد الاثنين إن أردت فحصهما.")
    quiz("w11.gates", [
        Q("c_t = ", ["tanh(c_{t−1} Wh)", "f⊙c_{t−1} + i⊙g", "o⊙h_{t−1}"], 1, ""),
        Q("f ≈ 1 وi ≈ 0 يعني…", ["استبدال الذاكرة", "الاحتفاظ بالذاكرة القديمة كما هي", "مسح الذاكرة"], 1, ""),
        Q("المسار الذي يحفظ التدرج عبر الزمن:", ["h_t عبر tanh", "c_t عبر الجمع وf", "المخرج"], 1, ""),
        Q("LSTM(16) على D=4: المعلمات", ["336", "1,344", "84"], 1, "4×((4+16)×16+16)."),
    ])
    takeaway("ست معادلات: ثلاث بوابات sigmoid، مرشَّح tanh، ذاكرة بجمع، مخرج مكشوف. f/i/o = انسَ/اكتب/اكشف. الجمع في c هو ما يحفظ التدرج. ×4 معلمات.")
    lesson_footer(LESSON, ["المعادلات بالرموز.", "البوابات بالكلمات والأمثلة.", "مقارنة رقمية مع RNN."])
