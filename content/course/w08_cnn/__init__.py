from core.models import Module

MODULE = Module(
    id="course.w08",
    section="course",
    title_ar="الأسبوع 08 — الشبكات العصبية الالتفافية CNN: المفاهيم الأساسية",
    title_en="Week 08 — Convolutional Neural Networks: Core Concepts",
    order=8,
    icon=":material/grid_on:",
    prerequisites=["course.w07", "foundations.linalg", "foundations.architecture"],
    purpose_ar="الصورة كموتر، البكسل والقناة، النواة/المرشّح، الالتفاف، خريطة الخصائص، الحشو، الخطوة، التجميع، الطبقات الأساسية، البنى النموذجية، حساب شكل المخرج، وتحريك حركة النواة.",
    why_ar="الصورة 28×28 = 784 خاصية؛ MLP يعاملها كأرقام مستقلة ويهدر الجوار. CNN تبني على الجوار ومشاركة الأوزان — نفس MLP بقيود ذكية تجعل الصور قابلة للتعلم بمعلمات أقل بكثير.",
    objectives_ar=["قراءة صورة كموتر (H, W, C) ودفعة (B, H, W, C).", "تنفيذ الالتفاف يدويًا وفهم النواة وخريطة الخصائص والحشو والخطوة والتجميع.", "حساب أشكال ومعلمات CNN كاملة يدويًا ومطابقتها بـ Keras."],
    challenges_ar=["الخلط بين القناة والدفعة في الأشكال.", "نسيان أن Dense بعد Flatten تحمل معظم المعلمات."],
)

LESSON_MODULES = ["overview", "convolution", "padding_stride_pooling", "layers_shapes"]
