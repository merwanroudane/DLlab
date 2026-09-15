import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.architecture.parameter_count",
    title_ar="حساب عدد المعلمات",
    title_en="Counting Parameters",
    module="foundations.architecture",
    order=3,
    prerequisites=["foundations.architecture.layers", "foundations.linalg.matrix_multiplication"],
    objectives_ar=["حساب معلمات أي طبقة كثيفة: `in × out + out`.", "جمعها لشبكة كاملة ومطابقة النتيجة بـ `model.summary()`.", "فهم علاقة عدد المعلمات بحجم البيانات المطلوب وبالذاكرة."],
    terms=["parameter", "matrix"],
    labs=["labs.parameter_counter"],
    difficulty="beginner",
    summary_ar="لكل طبقة كثيفة: in×out أوزان + out انحيازات. المجموع هو ما يطبعه summary — تحقق منه دائمًا.",
)

CODE = '''def dense_params(n_in, n_out):
    return n_in * n_out + n_out            # أوزان + انحيازات

def count(layers):
    """layers = [input_dim, units_1, units_2, ..., output_units]"""
    total = 0
    print(f"{{'layer':<10}}{{'shape W':<12}}{{'shape b':<10}}{{'params':>8}}")
    for i in range(1, len(layers)):
        n_in, n_out = layers[i-1], layers[i]
        p = dense_params(n_in, n_out); total += p
        print(f"dense_{{i:<4}}{{str((n_in, n_out)):<12}}{{str((n_out,)):<10}}{{p:>8,}}")
    print(f"{{'total':<32}}{{total:>8,}}")
    return total

count({layers})
print()
print("MNIST MLP 784 -> 128 -> 10:"); count([784, 128, 10])
print("memory float32 (MB):", round(count([784, 128, 10]) * 4 / 1e6, 3))'''


def _controls() -> dict:
    txt = st.text_input("الطبقات: input, units..., output", value="4, 64, 32, 1", key="ctrl_pc_layers")
    try:
        layers = [int(x) for x in txt.split(",") if x.strip()]
        if len(layers) < 2:
            raise ValueError
    except ValueError:
        layers = [4, 64, 32, 1]
        st.warning("صيغة غير صالحة؛ استُخدم 4, 64, 32, 1.")
    return {"layers": layers}


def render() -> None:
    lesson_header(LESSON)
    h2("القاعدة", "The rule")
    equation(r"\text{params}(\text{Dense}) = n_{\text{in}} \times n_{\text{out}} + n_{\text{out}}",
             [(r"n_{\text{in}} \times n_{\text{out}}", "عناصر مصفوفة الأوزان `(in, out)`: وزن لكل رابط."), (r"n_{\text{out}}", "انحياز لكل وحدة.")],
             meaning_ar="عدّ الروابط وأضف عدد الوحدات.", example_ar="`Dense(64)` بعد مدخل من 4: $4 \\times 64 + 64 = 320$.",
             dl_link_ar="هذا الرقم هو عمود `Param #` في `model.summary()` وما يعدّه `sum(p.numel() for p in model.parameters())` في PyTorch.", title_ar="معلمات الطبقة الكثيفة")
    worked_steps([("الشبكة 4 → 64 → 32 → 1", r"\text{ثلاث طبقات كثيفة}"), ("dense_1", r"4\times64 + 64 = 320"), ("dense_2", r"64\times32 + 32 = 2{,}080"), ("dense_3", r"32\times1 + 1 = 33"), ("المجموع", r"320 + 2{,}080 + 33 = 2{,}433")])
    code_lab(CodeLab(
        key="arch_params", title_ar="عدّاد معلمات لأي شبكة كثيفة", code=CODE, template=True, defaults={"layers": [4, 64, 32, 1]},
        before=Before(goal_ar="حساب معلمات كل طبقة والمجموع لبنية تحددها، ثم مثال MNIST وذاكرته.", stage_ar="البنية ← الفحص.",
                      inputs_ar="قائمة أحجام الطبقات.", expected_ar="جدول بشكل W وb ومعلمات كل طبقة، المجموع، ثم 101,770 لـ MNIST MLP و≈0.4 MB."),
        explain=[("1-2", "القاعدة في دالة."), ("4-13", "الحلقة على الأزواج المتتالية (in, out). طباعة تشبه `model.summary()` عمدًا."), ("15", "بنيتك من عنصر التحكم."), ("17-18", "784 = 28×28 بكسل مفرودة؛ 101,770 معلمة × 4 بايت (float32) ≈ 0.41 MB.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- قارن جدولك بما سيطبعه `model.summary()` عندما تبني نفس الشبكة في Keras (الأسبوع 03): يجب أن يتطابق رقمًا رقمًا.\n- الذاكرة: المعلمات وحدها صغيرة؛ ما يستهلك ذاكرة GPU هو **تنشيطات الدفعة** والتدرجات (الأسبوع 13).",
    ))
    definition("**القاعدة الإرشادية للبيانات**: بيانات قليلة مع معلمات كثيرة = حفظ. لا توجد نسبة سحرية، لكن 2,433 معلمة على 300 صف مثلًا تستدعي تنظيمًا قويًا أو شبكة أصغر.")
    debugging_note("`Param #` أصغر من توقعك في الطبقة الأولى؟ `input_shape` خاطئ (مثلًا 3 بدل 4 بعد نسيان عمود one-hot). أكبر؟ ربما دخل عمود لا يجب أن يدخل (المعرّف، الهدف).")
    st.button("افتح عدّاد المعلمات التفاعلي", icon=":material/science:", type="primary", on_click=go, args=("labs.parameter_counter",), key="pc_lab_btn")
    common_mistake("عدّ طبقة الإدخال: «784 معلمة للإدخال». لا — الإدخال بلا معلمات؛ 784 هو `n_in` لأول طبقة كثيفة.")
    quiz("arch.params", [
        Q("`Dense(10)` بعد مدخل من 20: المعلمات…", ["200", "210", "30"], 1, "20×10 + 10.", kind="shape"),
        Q("شبكة 3 → 5 → 2: المجموع…", ["32", "25", "22"], 0, "(15+5) + (10+2).", kind="shape"),
        Q("Param # لأول طبقة أقل من المتوقع. السبب الأرجح…", ["معدل التعلم", "input_shape خاطئ", "التنشيط"], 1, "n_in.", kind="error"),
    ])
    takeaway("in×out + out لكل طبقة كثيفة؛ اجمع؛ طابق مع summary. الإدخال بلا معلمات. المعلمات مقابل حجم البيانات مؤشر مبكر لفرط التخصيص.")
    lesson_footer(LESSON, ["القاعدة والمثال الكامل.", "summary يجب أن يطابق حسابك.", "الذاكرة الحقيقية في التنشيطات لا المعلمات."])
