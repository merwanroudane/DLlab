import numpy as np
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.regularization.dropout",
    title_ar="Dropout: التدريب مقابل الاستدلال",
    title_en="Dropout: Training vs Inference",
    module="foundations.regularization",
    order=2,
    prerequisites=["foundations.regularization.l1_l2_weight_decay", "foundations.python.randomness_reproducibility"],
    objectives_ar=["ما يفعله Dropout في التدريب (إسقاط عشوائي) ولماذا يعمّم (منع الاعتماد المتبادل، تجميع ضمني).", "القلب المعكوس والفرق بين وضعي التدريب والاستدلال.", "اختيار معدله ومكانه، وأثره على المنحنيات."],
    terms=["seed", "probability"],
    labs=["labs.dropout_lab"],
    difficulty="intermediate",
    summary_ar="في التدريب: صفّر كل وحدة باحتمال p واقسم الباقي على (1−p). في الاستدلال: لا شيء. loss قد تبدو أعلى من val_loss — طبيعي.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
a = np.ones((4, 8))                       # تنشيطات دفعة من 4 ملاحظات × 8 وحدات (كلها 1 للتوضيح)
p = {p}

# التدريب: قناع برنولي + قلب معكوس
mask = (rng.uniform(size=a.shape) >= p) / (1 - p)
a_train = a * mask
print("mask (0 = dropped, else scale):\\n", mask.round(2))
print("mean activation train =", a_train.mean().round(3), "  (≈ 1: the scaling keeps the expectation)")

# الاستدلال: لا إسقاط ولا تحجيم
a_eval = a
print("mean activation eval  =", a_eval.mean().round(3))

# لماذا القلب المعكوس؟ بدونه يختلف مقياس المخرج بين التدريب والاستدلال
naive = a * (rng.uniform(size=a.shape) >= p)
print("without inverted scaling, train mean =", naive.mean().round(3), " vs eval 1.0  -> the next layer sees a different scale")

# أثر التجميع الضمني: نفس المدخل، أقنعة مختلفة → مخرجات مختلفة (تدريب) ومخرج واحد (استدلال)
w = rng.normal(size=8)
outs = [float((a[0] * ((rng.uniform(size=8) >= p) / (1 - p))) @ w) for _ in range(5)]
print("5 training-mode outputs for one input:", np.round(outs, 2), "  eval output:", round(float(a[0] @ w), 2))'''


def _controls() -> dict:
    return {"p": st.select_slider("معدل الإسقاط p", options=[0.1, 0.2, 0.3, 0.5, 0.7], value=0.5, key="ctrl_do_p")}


def render() -> None:
    lesson_header(LESSON)
    h2("ما هو؟", "What is it?")
    definition("**Dropout**: أثناء التدريب، في كل تمريرة، تُصفَّر كل وحدة في الطبقة باحتمال $p$ عشوائيًا، وتُقسم الوحدات الباقية على $(1-p)$ (**القلب المعكوس**) لتبقى القيمة المتوقعة ثابتة. أثناء **الاستدلال** لا يحدث شيء: كل الوحدات تعمل.")
    equation(r"\text{train:}\quad \tilde{a} = \frac{m \odot a}{1-p}, \ m_j \sim \text{Bernoulli}(1-p) \qquad\qquad \text{eval:}\quad \tilde{a} = a",
             [("m", "قناع 0/1 عشوائي جديد لكل دفعة."), ("1 - p", "احتمال الإبقاء؛ القسمة عليه تحفظ $\\mathbb{E}[\\tilde a] = a$."), ("p", "معدل الإسقاط: 0.2–0.5 للطبقات الكثيفة.")],
             meaning_ar="كل تمريرة تدريب تستخدم «شبكة فرعية» عشوائية مختلفة؛ الاستدلال يستخدم الشبكة الكاملة كمتوسط لكل الشبكات الفرعية.",
             example_ar="p = 0.5، 8 وحدات: ~4 تُصفَّر، والباقي يُضرب في 2.",
             dl_link_ar="`Dropout(0.3)` في Keras، `nn.Dropout(0.3)` في PyTorch. الأطر تبدّل السلوك تلقائيًا بحسب `training`/`model.eval()` — لهذا وضع eval إلزامي عند التقييم.", title_ar="Dropout")
    why("لماذا يعمّم؟ (1) لا تستطيع وحدة الاعتماد على وحدة أخرى بعينها (قد تختفي) فتتعلم خصائص أكثر متانة. (2) تجميع ضمني لعدد هائل من الشبكات الفرعية. (3) ضوضاء في التنشيطات تعمل كتنظيم.")
    intuition("فريق يتدرب وكل يوم يغيب نصف أعضائه عشوائيًا: لا يستطيع أحد الاعتماد على زميل بعينه، فيتعلم الجميع تغطية بعضهم. يوم المباراة يحضر الجميع — أقوى من أي تشكيلة تدرّبت وحدها.")
    code_lab(CodeLab(
        key="reg_dropout", title_ar="القناع والقلب المعكوس ووضعا التدريب/الاستدلال", code=CODE, template=True, defaults={"p": 0.5},
        before=Before(goal_ar="رؤية قناع Dropout، لماذا القلب المعكوس، وكيف يعطي نفس المدخل مخرجات مختلفة في التدريب ومخرجًا واحدًا في الاستدلال.", stage_ar="التنظيم.",
                      inputs_ar="تنشيطات من الآحاد ومعدل p.", expected_ar="قناع بأصفار و1/(1−p)؛ متوسط التدريب ≈ 1 = الاستدلال؛ بلا القلب المعكوس المتوسط (1−p)؛ خمسة مخرجات مختلفة مقابل واحد."),
        explain=[("6-9", "القناع: 0 للمسقَط، 1/(1−p) للباقي. المتوسط يبقى ≈ 1."), ("12-13", "الاستدلال: التنشيطات كما هي."), ("16-17", "بلا القلب المعكوس: مقياس مختلف بين الوضعين — الطبقة التالية تدرّبت على مقياس ولا ترى غيره عند الاستدلال. (النسخة القديمة كانت تضرب في (1−p) عند الاستدلال بدلًا من ذلك.)"),
                 ("20-22", "نفس المدخل خمس مرات في التدريب: خمس شبكات فرعية. الاستدلال: قيمة واحدة حتمية.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- p = 0.7 يسقط معظم الوحدات: تنظيم قوي جدًا قد يسبب قصور تعلم.\n- نتيجة الاستدلال حتمية وأدق: لهذا تُقاس كل المقاييس في وضع eval.",
    ))
    debugging_note("`loss` (تدريب) أعلى من `val_loss` طوال التدريب مع Dropout؟ طبيعي: خسارة التدريب تُحسب بشبكة فرعية مشوَّشة، والتحقق بالشبكة الكاملة. الفجوة المعكوسة ليست تسريبًا هنا.")
    st.button("افتح معمل Dropout", icon=":material/science:", type="primary", on_click=goto, args=("labs.dropout_lab",), key="do_lab")
    common_mistake("التنبؤ في وضع التدريب (نسيان `model.eval()` في PyTorch): تنبؤات عشوائية مختلفة لنفس المدخل. في Keras `model.predict` يستخدم وضع الاستدلال تلقائيًا؛ `model(x, training=True)` لا.")
    quiz("reg.do", [
        Q("في الاستدلال، Dropout…", ["يسقط بنفس p", "لا يفعل شيئًا", "يضرب في p"], 1, "الشبكة الكاملة."),
        Q("لماذا نقسم على (1 − p) في التدريب؟", ["للسرعة", "للحفاظ على القيمة المتوقعة للتنشيط", "لتقليل p"], 1, "القلب المعكوس."),
        Q("loss > val_loss مع Dropout…", ["تسريب", "طبيعي: التدريب مشوَّش", "خطأ"], 1, "شبكة فرعية مقابل كاملة."),
        Q("p = 0.8 على كل الطبقات…", ["تنظيم مثالي", "قصور تعلم محتمل", "لا أثر"], 1, "إسقاط مفرط."),
    ])
    takeaway("Dropout = إسقاط عشوائي مع قلب معكوس في التدريب، لا شيء في الاستدلال. يمنع الاعتماد المتبادل ويجمّع ضمنيًا. eval mode إلزامي.")
    lesson_footer(LESSON, ["القناع والقلب المعكوس.", "لماذا يعمّم.", "الفجوة المعكوسة طبيعية."])
