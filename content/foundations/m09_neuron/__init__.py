from core.models import Module

MODULE = Module(
    id="foundations.neuron",
    section="foundations",
    title_ar="الوحدة 9 — من الانحدار إلى الخلية العصبية",
    title_en="Module 9 — From Regression to Neuron",
    order=9,
    icon=":material/hub:",
    prerequisites=["foundations.ml", "foundations.linalg"],
    purpose_ar="المسار التدريجي: $y = \\beta_0 + \\beta_1 x$ ← $z = wx + b$ ← $z = \\sum w_j x_j + b$ ← $a = f(z)$. الوزن، الانحياز، المجموع الموزون، التنشيط، ثم الخلية/الوحدة/العقدة/البيرسبترون.",
    why_ar="الخلية العصبية ليست مفهومًا جديدًا بل انحدار بتنشيط. رؤية هذا التطابق تزيل الغموض عن كل ما بعده.",
    objectives_ar=["كتابة الخلية العصبية معادلةً وكودًا ورسمًا.", "تفسير الوزن والانحياز والتنشيط بأمثلة رقمية.", "التمييز بين المصطلحات الأربعة المترادفة: خلية، وحدة، عقدة، بيرسبترون."],
    challenges_ar=["الاعتقاد أن الخلية «تفكر».", "نسيان الانحياز.", "الخلط بين z (قبل التنشيط) وa (بعده)."],
)

LESSON_MODULES = ["line_to_neuron", "weighted_sum_bias", "activation_intro", "neuron_perceptron"]
