import streamlit as st

from components.callouts import common_mistake, definition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.neuron.neuron_perceptron",
    title_ar="الخلية العصبية، الوحدة، العقدة، البيرسبترون",
    title_en="Neuron, Unit, Node, Perceptron",
    module="foundations.neuron",
    order=4,
    prerequisites=["foundations.neuron.activation_intro"],
    objectives_ar=["رسم الخلية العصبية وتسمية أجزائها.", "معرفة أن الأسماء الأربعة تشير إلى الشيء نفسه (مع فرق تاريخي للبيرسبترون).", "تنفيذ خلية كاملة في NumPy وفي «شكل الإطار» كصف واحد."],
    terms=["weight", "bias", "dot_product"],
    labs=["labs.neuron_lab"],
    difficulty="beginner",
    summary_ar="خلية = مدخلات × أوزان + انحياز → تنشيط. وحدة/عقدة مرادفان؛ البيرسبترون النسخة التاريخية بعتبة حادة.",
)

CODE = '''import numpy as np

class Neuron:
    def __init__(self, n_inputs, activation="relu", seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.5, n_inputs)     # أوزان أولية عشوائية صغيرة
        self.b = 0.0
        self.activation = activation
    def forward(self, x):
        self.z = self.w @ x + self.b              # مجموع موزون (نحفظه للانتشار الخلفي لاحقًا)
        if self.activation == "relu":    self.a = max(0.0, self.z)
        elif self.activation == "sigmoid": self.a = 1 / (1 + np.exp(-self.z))
        else:                            self.a = self.z
        return self.a

x = np.array([1.2, -0.5, -0.8])
for act in ["linear", "relu", "sigmoid"]:
    n = Neuron(3, activation=act)
    a = n.forward(x)
    print(f"{act:<8} w={n.w.round(2)} z={n.z:+.3f} -> a={a:+.3f}")

# البيرسبترون التاريخي (1958): عتبة حادة بدل تنشيط ناعم
def perceptron(x, w, b): return 1 if w @ x + b > 0 else 0
print("perceptron:", perceptron(x, np.array([-0.8, 1.5, 1.1]), -0.5))'''


def _neuron_svg() -> str:
    s = '<svg viewBox="0 0 620 230" width="100%" style="max-width:620px">' + svg_defs()
    for i, (lbl, y) in enumerate([("x₁", 40), ("x₂", 100), ("x₃", 160)]):
        s += f'<circle cx="60" cy="{y + 15}" r="20" fill="#E6F1FB" stroke="#2F6FB5"/>' + svg_text(60, y + 20, lbl, size=13, bold=True)
        s += svg_arrow(82, y + 15, 232, 112)
        s += svg_text(150, (y + 15 + 112) / 2 - 6, f"w{i + 1}", size=12, color="#7C5CBF", bold=True)
    s += '<circle cx="270" cy="112" r="36" fill="#EFE9FA" stroke="#7C5CBF" stroke-width="2"/>' + svg_text(270, 108, "Σ wⱼxⱼ + b", size=12, bold=True) + svg_text(270, 124, "= z", size=12)
    s += svg_text(270, 175, "b", size=13, color="#7C5CBF", bold=True) + svg_arrow(270, 165, 270, 150)
    s += svg_arrow(308, 112, 372, 112)
    s += '<rect x="375" y="87" width="90" height="50" rx="12" fill="#FFF1DC" stroke="#C77A1A" stroke-width="2"/>' + svg_text(420, 117, "f(z)", size=14, bold=True)
    s += svg_arrow(467, 112, 540, 112) + svg_text(575, 117, "a", size=16, bold=True, color="#2E8B57")
    s += svg_text(310, 215, "inputs → weights → weighted sum + bias → activation → output", size=12, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("الرسم القياسي", "The standard diagram")
    diagram("خلية عصبية واحدة", _neuron_svg(), what_ar="ثلاثة مدخلات على اليسار، كل منها يمر عبر وزن إلى دائرة الجمع، يُضاف الانحياز، ثم يمر الناتج بدالة التنشيط ليخرج a.",
            how_ar="اتبع الأسهم: مدخل × وزن، اجمع، أضف b، طبّق f. هذا الرسم يتكرر بالضبط لكل خلية في كل طبقة من أي شبكة.",
            takeaway_ar="الخلية آلة صغيرة بثلاث عمليات: ضرب نقطي، إزاحة، تنشيط. لا شيء «عصبي» فيها بيولوجيًا سوى الاسم.", title_en="One neuron",
            legend=[("#E6F1FB", "مدخلات"), ("#EFE9FA", "معلمات ومجموع"), ("#FFF1DC", "تنشيط"), ("#DDF5EA", "مخرج")])
    definition("**الخلية العصبية** `Neuron` = **الوحدة** `Unit` = **العقدة** `Node`: نفس الشيء بثلاثة أسماء (Keras تقول units، الرسوم تقول nodes). **البيرسبترون** `Perceptron` النموذج التاريخي (1958) بتنشيط عتبة حادة (0 أو 1)؛ اليوم تُستخدم الكلمة أحيانًا لأي خلية، و«MLP» (multi-layer perceptron) اسم الشبكة الكثيفة.")
    compare_table(["الاسم", "English", "السياق", "ملاحظة"],
                  [("خلية عصبية", "Neuron", "الشرح العام", "استعارة بيولوجية فقط"), ("وحدة", "Unit", "Keras: `Dense(units=64)`", "الاسم البرمجي"), ("عقدة", "Node", "الرسوم البيانية للشبكة", "نقطة في الرسم"),
                   ("بيرسبترون", "Perceptron", "تاريخي / MLP", "عتبة حادة؛ غير قابل للاشتقاق فلا يعمل مع التدرج")], ["rtl", "ltr", "rtl", "rtl"])
    code_lab(CodeLab(
        key="neuron_class", title_ar="خلية كصنف بايثون + بيرسبترون تاريخي", code=CODE,
        before=Before(goal_ar="تغليف الخلية في صنف له `forward`، مقارنة ثلاثة تنشيطات على نفس المدخل، ورؤية البيرسبترون بعتبة حادة.", stage_ar="الخلية العصبية.",
                      inputs_ar="متجه من 3 خصائص.", expected_ar="نفس z للثلاثة (نفس البذرة) ومخرجات مختلفة: خطي = z، ReLU = max(0,z)، Sigmoid في (0,1)؛ ثم 0/1 من البيرسبترون."),
        explain=[("3-8", "الأوزان الأولية عشوائية صغيرة (سنشرح لماذا في التهيئة)؛ الانحياز صفر."), ("9-14", "`forward` يحفظ `z` و`a` — الانتشار الخلفي سيحتاجهما. هذا نمط `nn.Module.forward` في PyTorch."),
                 ("16-20", "ثلاثة تنشيطات على نفس z."), ("23-24", "البيرسبترون: مقارنة مع عتبة. مشتقة العتبة صفر في كل مكان (وغير معرفة عند القفز) — لذلك لا يُدرَّب بالتدرج، وهذا سبب استبداله بتنشيطات ناعمة.")],
        run=run_printed(CODE),
        after_ar="- `z` واحد، ثلاث «قراءات» له حسب التنشيط.\n- البيرسبترون يعطي 0 لأن z سالب؛ نفس القرار الذي يعطيه Sigmoid عند العتبة 0.5 لكن بلا احتمال ولا تدرج.",
    ))
    research_note("الاستعارة البيولوجية (تشابك، إطلاق) ألهمت الاسم لكنها لا تفسر شيئًا في الرياضيات. لا تستخدم «الشبكة تحاكي الدماغ» في ورقة علمية؛ قل «نموذج مركّب من وحدات لاخطية».")
    st.button("افتح معمل الخلية العصبية", icon=":material/science:", type="primary", on_click=go, args=("labs.neuron_lab",), key="neuron_lab_btn")
    common_mistake("`Dense(64)` تعني 64 خلية، لكل منها أوزانها الخاصة بطول عدد المدخلات وانحيازها. ليست خلية واحدة بـ 64 وزنًا.")
    quiz("neuron.names", [
        Q("`Dense(units=32)` تنشئ…", ["32 وزنًا", "32 خلية", "32 طبقة"], 1, "وحدة = خلية."),
        Q("لماذا لا يُدرَّب البيرسبترون بالتدرج؟", ["بطيء", "تنشيط العتبة مشتقته صفر", "يحتاج GPU"], 1, "لا إشارة تدرج."),
        Q("ما الذي تحفظه `forward` للاحقًا؟", ["المدخلات فقط", "z وa للانتشار الخلفي", "لا شيء"], 1, "الوسائط."),
    ])
    takeaway("خلية = وحدة = عقدة: ضرب نقطي + انحياز + تنشيط. البيرسبترون النسخة الحادة القديمة. forward يحفظ z وa.")
    lesson_footer(LESSON, ["رسم واحد يتكرر لكل خلية.", "الأسماء الأربعة.", "العتبة الحادة لا تُشتق."])
