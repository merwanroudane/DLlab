import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="course.w12.update_reset_gates",
    title_ar="GRU: بوابة التحديث وبوابة إعادة الضبط والحالة المخفية",
    title_en="GRU: Update Gate, Reset Gate & Hidden State",
    module="course.w12",
    order=2,
    prerequisites=["course.w12.overview", "course.w11.cell_gates_equations"],
    objectives_ar=["معادلات GRU الأربع ومعنى z وr.", "مقابلة GRU بـ LSTM: ما دُمج وما حُذف.", "تنفيذ خلية GRU يدويًا ومطابقتها بـ Keras بنفس الأوزان."],
    terms=["hadamard_product", "gradient"],
    labs=["labs.gru_gates_lab"],
    difficulty="advanced",
    summary_ar="z = σ(…Wz)، r = σ(…Wr)، h̃ = tanh([x, r⊙h]Wh)، h = (1−z)⊙h + z⊙h̃. z تدمج النسيان والإدخال (نسيان = 1−z)؛ r تقرر كم من الماضي يدخل في المرشَّح؛ لا حالة خلية منفصلة. 3 مجموعات أوزان بدل 4.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
from labs.rnn import gru_forward, random_weights, count_params

H = 3; xs = np.array([1.5, 0.0, 0.0, -1.0], "float32")
W = random_weights("gru", D=1, H=H, seed=0, scale=0.8)
steps = gru_forward(xs, W, H)
for s_ in steps:
    print(f"t={s_['t']} x={s_['x'][0]:+.1f}  z={np.round(s_['z'], 2).tolist()}  r={np.round(s_['r'], 2).tolist()}  h~={np.round(s_['h_tilde'], 2).tolist()}  h={np.round(s_['h'], 2).tolist()}")
print("params (hand):", count_params("gru", 1, H), "= 3 × ((1 + 3) × 3 + 3)")

# نفس الأوزان في Keras GRU (reset_after=False يطابق الصيغة الكلاسيكية أعلاه)
gru = layers.GRU(H, reset_after=False)
inp = keras.Input(shape=(len(xs), 1)); out = gru(inp); m = keras.Model(inp, out)
# ترتيب Keras للبوابات في kernel: [z, r, h] بشكل (D, 3H)؛ recurrent_kernel (H, 3H)؛ bias (3H,)
# تنبيه: Keras تعرّف z معكوسة (h = z⊙h_prev + (1−z)⊙h~)؛ نقلب إشارة أوزان z لأن σ(−a) = 1 − σ(a)
kernel = np.concatenate([-W["Wz"][:1], W["Wr"][:1], W["Wh"][:1]], axis=1)            # صف المدخل من كل مصفوفة
rkernel = np.concatenate([-W["Wz"][1:], W["Wr"][1:], W["Wh"][1:]], axis=1)            # صفوف الحالة
bias = np.concatenate([-W["bz"], W["br"], W["bh"]])
gru.set_weights([kernel.astype("float32"), rkernel.astype("float32"), bias.astype("float32")])
h_keras = m.predict(xs[None, :, None], verbose=0)[0]
print("Keras final h:", np.round(h_keras, 4).tolist(), "| hand final h:", np.round(steps[-1]["h"], 4).tolist(), "| match:", np.allclose(h_keras, steps[-1]["h"], atol=1e-5))
print("Keras params:", m.count_params())'''


def render() -> None:
    lesson_header(LESSON)
    definition("**GRU** (Gated Recurrent Unit): خلية تكرارية ببوابتين وحالة واحدة h. **بوابة التحديث** z تقرر كم نستبدل من الحالة القديمة بالمرشَّح (تدمج نسيان LSTM وإدخالها: الإبقاء = 1−z). **بوابة إعادة الضبط** r تقرر كم من الحالة القديمة يشارك في **حساب** المرشَّح. لا حالة خلية c منفصلة: h نفسها تمر بمسار الجمع.")
    why("LSTM ثلاث بوابات وحالتان و4 مجموعات أوزان. سؤال بحثي مشروع: هل كل ذلك ضروري؟ GRU تجيب: بوابتان وحالة واحدة تكفيان غالبًا — بمعلمات أقل بالربع وتدريب أسرع. متى تكفي؟ سؤال تجريبي (الدرس التالي).")
    equation(r"z_t = \sigma\big([x_t, h_{t-1}] W_z + b_z\big), \qquad r_t = \sigma\big([x_t, h_{t-1}] W_r + b_r\big)",
             [("z_t", "التحديث: 0 = أبقِ القديم كما هو (تخطٍّ)، 1 = استبدله بالمرشَّح."), ("r_t", "إعادة الضبط: 0 = تجاهل الماضي عند اقتراح الجديد، 1 = استخدمه كاملًا.")],
             meaning_ar="بوابتان بنفس شكل بوابات LSTM (sigmoid على [x, h]).", example_ar="z = 0.1: الحالة تتغير 10% فقط في هذه الخطوة — ذاكرة طويلة.", dl_link_ar="`layers.GRU(units)`؛ `nn.GRU`.", title_ar="البوابتان")
    equation(r"\tilde h_t = \tanh\big([x_t,\; r_t \odot h_{t-1}]\, W_h + b_h\big), \qquad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t",
             [(r"\tilde h_t", "المرشَّح: يُحسب من المدخل ومن **جزء** من الماضي تحدده r."), ("(1 - z_t)", "نصيب القديم؛ z_t نصيب الجديد: متوسط موزون (تحدّب) — لا يمكن أن «ينفجر» الحجم."), ("h_t", "الحالة الوحيدة: تُكشف كاملة وتعود للخطوة التالية.")],
             meaning_ar="مسار الجمع (1−z)⊙h يحفظ التدرج كما تفعل c في LSTM؛ وبما أن (1−z) + z = 1، الحالة مزيج بين القديم والجديد.", example_ar="h_prev = 2، h̃ = −1، z = 0.25 → h = 0.75×2 + 0.25×(−1) = 1.25.", dl_link_ar="Keras 3 افتراضيًا `reset_after=True` (صيغة CuDNN؛ فرق طفيف في موضع r).", title_ar="المرشَّح والمزج")
    h2("GRU مقابل LSTM", "GRU vs LSTM")
    compare_table(["الجانب", "LSTM", "GRU"],
                  [("الحالات", "c (ذاكرة) + h (مكشوفة)", "h فقط"), ("البوابات", "f، i، o", "z، r"), ("النسيان/الإدخال", "مستقلان (f، i)", "مرتبطان: (1−z)، z"), ("التحكم في الكشف", "o", "لا (h مكشوفة دائمًا)"), ("مصفوفات الأوزان", "4", "3"), ("المعلمات (D=1, H=32)", "4,352", "3,264"),
                   ("الأداء", "ميزة طفيفة مع اعتماديات طويلة جدًا", "مماثل غالبًا؛ أسرع قليلًا"), ("متى", "تسلسلات طويلة، بيانات كثيرة", "بيانات أقل، سرعة، افتراضي معقول")],
                  ["rtl", "rtl", "rtl"])
    code_lab(CodeLab(
        key="w12_gru", title_ar="خلية GRU يدويًا ثم نفس الأوزان في Keras: تطابق الأرقام", code=CODE, level="B",
        before=Before(goal_ar="تمرير تسلسل قصير عبر GRU يدوية وقراءة z وr وh̃ وh عند كل خطوة، ثم زرع نفس الأوزان في layers.GRU والتحقق من تطابق الحالة النهائية والمعلمات.", stage_ar="الأسبوع 12: بوابتا GRU.",
                      inputs_ar="تسلسل (1.5, 0, 0, −1) وH = 3.", expected_ar="أربعة أسطر بقيم البوابات؛ 45 معلمة؛ Keras == اليد: True."),
        explain=[("6-11", "خلية المنصة (`labs.rnn.gru_forward`) بأوزان عشوائية؛ نطبع كل بوابة. عدّ المعلمات: 3 × ((D+H)×H + H)."), ("14-16", "Keras GRU بـ `reset_after=False` لتطابق الصيغة الكلاسيكية."), ("17-22", "Keras تخزّن البوابات الثلاث متجاورة: `kernel (D, 3H)`، `recurrent_kernel (H, 3H)`، `bias (3H,)` بترتيب [z, r, h]. **فرق اصطلاحي**: Keras تستخدم z كنصيب القديم (h = z⊙h_prev + (1−z)⊙h̃)؛ قلب إشارة أوزان z يعادل 1−z فتتطابق الصيغتان."), ("23-25", "التطابق حتى 1e-5: لا فرق بين المعادلات و`layers.GRU`.")],
        run=run_printed(CODE),
        after_ar="- عند x = 0 (الخطوتان الوسطى) راقب z: صغيرة → h تكاد لا تتغير: الذاكرة تُحمل عبر الجمع.\n- ترتيب [z, r, h] في kernel هو ما تراه لو فحصت `gru.get_weights()` في مشروعك — واصطلاح z معكوس في Keras (نفس الرياضيات بتسمية مقلوبة).\n- `reset_after=True` (الافتراضي) يطبّق r بعد الضرب في الأوزان المتكررة؛ الفكرة نفسها والأرقام تختلف قليلًا.",
    ))
    st.button("افتح معمل بوابات GRU (تحريك)", icon=":material/science:", type="primary", on_click=goto, args=("labs.gru_gates_lab",), key="w12_lab_gates")
    intuition("اقرأ z كـ«مقبض السرعة»: z ≈ 0 الحالة تتجمد (تتذكر)، z ≈ 1 تُستبدل (تنسى وتتعلم الجديد). وr كـ«هل أنظر إلى الماضي قبل أن أقترح؟». بوابتان فقط، وكلاهما متعلَّم من البيانات.")
    common_mistake("اعتبار z في GRU مثل i في LSTM. z تقرر **نسبة الاستبدال** (تربط النسيان بالإدخال)؛ في LSTM يمكن نسيان القديم وعدم كتابة الجديد معًا (f = 0، i = 0) — في GRU لا: ما يُنسى يُستبدل.")
    quiz("w12.gates", [
        Q("z = 0 في GRU يعني…", ["استبدال كامل", "الاحتفاظ بالحالة القديمة كما هي", "مسح الحالة"], 1, ""),
        Q("r تؤثر في…", ["المزج النهائي", "حساب المرشَّح h̃", "المخرج فقط"], 1, ""),
        Q("GRU(16) على D=1: المعلمات", ["1,152", "864", "288"], 1, "3×((1+16)×16+16)."),
        Q("ما يقابل مسار c في LSTM داخل GRU:", ["لا شيء", "(1−z)⊙h_{t−1} في المزج", "r"], 1, ""),
    ])
    takeaway("GRU = بوابتان (z تحديث، r إعادة ضبط) وحالة واحدة بمزج تحدّبي. ثلاث مصفوفات بدل أربع، أداء مماثل غالبًا. المعادلات = Keras بالأرقام.")
    lesson_footer(LESSON, ["المعادلات الأربع.", "المقابلة مع LSTM.", "التطابق مع Keras بالكود."])
