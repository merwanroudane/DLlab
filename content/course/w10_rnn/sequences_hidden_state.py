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
    id="course.w10.sequences_hidden_state",
    title_ar="التسلسل والخطوة الزمنية والترتيب، الحالة المخفية والاتصال التكراري",
    title_en="Sequence, Timestep, Order, Hidden State & the Recurrent Connection",
    module="course.w10",
    order=2,
    prerequisites=["course.w10.overview", "foundations.data.data_modalities", "foundations.linalg.tensors"],
    objectives_ar=["تحويل سلسلة إلى نوافذ بشكل (samples, timesteps, features) وفهم لماذا الترتيب مهم.", "خلية RNN: h_t = tanh(x_t Wx + h_{t−1} Wh + b) بالمعادلة وباليد وبالتحريك.", "نفس الأوزان في كل خطوة: المعلمات لا تعتمد على طول التسلسل."],
    terms=["sequence", "time_series", "shape", "weight"],
    labs=["labs.rnn_unrolling_lab"],
    difficulty="intermediate",
    summary_ar="النافذة الزمنية هي الملاحظة: (samples, timesteps, features). RNN: خلية واحدة تُطبَّق على كل خطوة وتحمل حالة h_t تلخّص الماضي. الأوزان مشتركة عبر الزمن كما تشارك CNN الأوزان عبر المكان.",
)

CODE = '''import numpy as np
from labs.datasets import monthly_inflation
from labs.rnn import make_windows, rnn_forward, random_weights

s = monthly_inflation()["inflation"].to_numpy("float32")            # 180 شهرًا
print("series:", s.shape, "| first 6 months:", s[:6])
# 1) النوافذ: كل ملاحظة = 12 شهرًا سابقة، الهدف = الشهر التالي
X, y = make_windows(s, window=12)
print("windows X:", X.shape, "(samples, timesteps, features) | y:", y.shape)
print("X[0, :, 0] =", X[0, :, 0], "-> y[0] =", y[0], "== s[12] =", s[12])
print("X[1, :, 0] starts at s[1]: overlapping windows shift by one month")
# الترتيب مهم: نفس القيم بترتيب مختلف = ملاحظة مختلفة
print("shuffled window equals original?", np.array_equal(X[0, :, 0], np.random.default_rng(0).permutation(X[0, :, 0])))

# 2) خلية RNN يدويًا على أول نافذة (موحَّدة)
z = (X[0, :, 0] - s[:120].mean()) / s[:120].std()
W = random_weights("rnn", D=1, H=3, seed=0, scale=0.6)
steps = rnn_forward(z, **W)
print("\\nWx:", W["Wx"].shape, "Wh:", W["Wh"].shape, "b:", W["b"].shape, "-> params:", W["Wx"].size + W["Wh"].size + W["b"].size, "(for ANY window length)")
for st_ in steps[:4]:
    print(f"  t={st_['t']:>2} x={st_['x'][0]:+.2f} h_prev={np.round(st_['h_prev'], 2).tolist()} -> h={np.round(st_['h'], 2).tolist()}")
print("  ...")
print(f"  t={steps[-1]['t']:>2} final h = {np.round(steps[-1]['h'], 2).tolist()}  <- a 3-number summary of 12 months, fed to Dense(1)")
# 3) الحالة تتذكر: نفس x الأخير بتاريخ مختلف يعطي h مختلفة
z2 = z.copy(); z2[:6] = -z2[:6]
h_a = rnn_forward(z, **W)[-1]["h"]; h_b = rnn_forward(z2, **W)[-1]["h"]
print("same last 6 inputs, different first 6 -> final h differs by only", np.round(np.abs(np.array(h_a) - np.array(h_b)).max(), 3), "(memory exists but is WEAK: early inputs fade -> next lesson)")'''


def render() -> None:
    lesson_header(LESSON)
    h2("التسلسل كملاحظة", "The sequence as an observation")
    definition("**التسلسل** بيانات يهم فيها **الترتيب**: سلسلة زمنية (تضخم شهري)، نص (كلمات)، إشارة. **الخطوة الزمنية** موضع في التسلسل. الملاحظة في مسائل التسلسل هي **نافذة** من الخطوات: X بشكل `(samples, timesteps, features)` — رتبة 3. الهدف: القيمة التالية (انحدار) أو فئة (تصنيف).")
    compare_table(["البعد", "المعنى", "مثال التضخم", "مثال نص"],
                  [("samples", "عدد النوافذ", "168 نافذة من 180 شهرًا", "عدد الجمل"), ("timesteps", "طول النافذة", "12 شهرًا", "عدد الكلمات (بحشو)"), ("features", "قيم كل خطوة", "1 (التضخم) أو 3 (+ الفائدة والبطالة)", "طول تضمين الكلمة")],
                  ["code", "rtl", "rtl", "rtl"])
    why("MLP على نافذة مسطّحة (12 رقمًا) ممكن — لكنه يعامل «الشهر الأول» و«الشهر الثاني عشر» كخصائص مستقلة بلا معنى للترتيب، ويحتاج إعادة تدريب لأي طول آخر. RNN تقرأ الخطوات **بالترتيب** بخلية واحدة تحمل حالة — تعمل لأي طول وتتشارك الأوزان عبر الزمن كما تتشاركها CNN عبر المكان.")
    h2("الخلية والحالة المخفية", "The cell and the hidden state")
    equation(r"h_t = \tanh\big(x_t W_x + h_{t-1} W_h + b\big), \qquad \hat y = h_T W_y + b_y",
             [("x_t", "مدخل الخطوة t (features,)."), ("h_{t-1}", "الحالة المخفية السابقة (H,): ذاكرة ملخَّصة لكل ما قبل t."), ("W_x, W_h, b", "أوزان **واحدة** لكل الخطوات: (features×H)، (H×H)، (H,)."), (r"\hat y", "الرأس: Dense على الحالة الأخيرة (أو على كل حالة).")],
             meaning_ar="نفس معادلة الخلية (مجموع موزون + تنشيط) مع إضافة واحدة: مدخل ثانٍ هو الحالة السابقة. التكرار = الاتصال من h_{t−1} إلى h_t.",
             example_ar="H = 3، features = 1: المعلمات 1×3 + 3×3 + 3 = 15 لأي طول نافذة.", dl_link_ar="`layers.SimpleRNN(units=H)` في Keras؛ `nn.RNN` في PyTorch.", title_ar="خلية RNN")
    code_lab(CodeLab(
        key="w10_seq", title_ar="النوافذ الزمنية، خلية RNN يدويًا خطوةً خطوة، والذاكرة", code=CODE, level="A",
        before=Before(goal_ar="تحويل التضخم الشهري إلى نوافذ (samples, 12, 1)، تمرير نافذة عبر خلية RNN بأوزان صغيرة وقراءة h عند كل خطوة، ثم إثبات أن الحالة النهائية تتذكر البداية.", stage_ar="الأسبوع 10: التسلسل والحالة.",
                      inputs_ar="180 شهرًا من التضخم (مجموعة المنصة).", expected_ar="X (168, 12, 1)؛ y[0] = s[12]؛ 15 معلمة لأي طول؛ h يتغير كل خطوة؛ تغيير أول 6 مدخلات يغيّر h النهائية بفارق صغير (ذاكرة ضعيفة)."),
        explain=[("7-11", "`make_windows`: نافذة متزحلقة بطول 12 والهدف الشهر التالي. الشكل رتبة 3. الترتيب داخل النافذة جزء من الملاحظة."), ("16-23", "الخلية يدويًا: أوزان بأشكال (1,3)، (3,3)، (3,) — 15 معلمة **مستقلة عن طول النافذة**. h تتطور خطوةً خطوة."), ("25-28", "الذاكرة: نفس آخر 6 مدخلات مع بداية مختلفة → h نهائية مختلفة لكن **بفارق صغير جدًا**: الذاكرة موجودة وضعيفة. المدخلات القديمة تتلاشى عبر tanh وWh المتكررين — موضوع الدرس التالي.")],
        run=run_printed(CODE),
        after_ar="- `X[1]` تبدأ من `s[1]`: النوافذ متداخلة، وهذا مهم عند التقسيم (لا تخلط: تسريب من المستقبل).\n- h النهائية (3 أرقام) هي «ملخص» 12 شهرًا يُعطى للرأس Dense(1) للتنبؤ بالشهر 13.\n- الأوزان عشوائية هنا؛ التدريب (الدرس الرابع) يجعل الملخص مفيدًا.\n- الفارق 0.015 هو أول دليل على **تلاشي** أثر الماضي البعيد في RNN البسيط.",
    ))
    st.button("افتح معمل نشر RNN عبر الزمن", icon=":material/science:", type="primary", on_click=goto, args=("labs.rnn_unrolling_lab",), key="w10_lab_unroll")
    intuition("تخيّل قراءة جملة كلمةً كلمة مع ورقة صغيرة تكتب عليها ملخصك حتى الآن وتمحو وتعيد الكتابة بنفس القاعدة عند كل كلمة. الورقة هي h، القاعدة هي (Wx, Wh, b)، والجملة هي التسلسل. الورقة صغيرة — وهنا تبدأ المشكلة (الدرس التالي).")
    common_mistake("`train_test_split(X, y, shuffle=True)` على نوافذ زمنية: نافذة من 2024 في التدريب ونافذة متداخلة معها من 2024 في الاختبار — النموذج «رأى» المستقبل. التقسيم زمني دائمًا: الماضي للتدريب، ثم التحقق، ثم الاختبار الأحدث.")
    quiz("w10.seq", [
        Q("نافذة 24 شهرًا بثلاث متغيرات لكل شهر: شكل X لـ 100 نافذة", ["(100, 24, 3)", "(100, 72)", "(24, 100, 3)"], 0, ""),
        Q("SimpleRNN(units=8) على features=2: المعلمات", ["16", "88", "80"], 1, "2·8 + 8·8 + 8."),
        Q("h_t هي…", ["المخرج النهائي", "ذاكرة ملخَّصة للخطوات حتى t", "الوزن"], 1, ""),
        Q("مضاعفة طول النافذة تجعل معلمات RNN…", ["تتضاعف", "ثابتة", "تتربع"], 1, ""),
    ])
    takeaway("النافذة ملاحظة رتبة 3. RNN = خلية واحدة بأوزان مشتركة عبر الزمن تحمل حالة h تلخّص الماضي. المعلمات مستقلة عن الطول. التقسيم زمني بلا خلط.")
    lesson_footer(LESSON, ["النوافذ والأشكال.", "الخلية والحالة بالمعادلة واليد.", "الذاكرة والتقسيم الزمني."])
