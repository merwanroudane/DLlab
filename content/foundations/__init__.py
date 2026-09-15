"""Foundations Academy — the prerequisite curriculum (spec §28)."""

from core.models import Section

SECTION = Section(
    id="foundations",
    title_ar="أكاديمية الأسس",
    title_en="Deep Learning Foundations Academy",
    icon=":material/foundation:",
    order=1,
    description_ar=(
        "قسم تأسيسي داخل المنصة يبني كل ما يحتاجه الباحث قبل الشبكات العصبية: "
        "البيانات، بايثون، الرياضيات، الجبر الخطي، التفاضل، الاحتمال، التعلم الآلي، "
        "ثم الخلية العصبية والشبكة والتدريب والتشخيص، وأخيرًا أطر العمل "
        "`Keras` و`TensorFlow` و`PyTorch`. كل مفهوم يُشرح قبل أن يُستخدم."
    ),
)

# Ordered module packages. Add a package here once it has at least one lesson.
MODULE_PACKAGES = [
    "m00_start_here",
    "m01_data",
    "m02_python",
    "m03_math",
    "m04_linalg",
    "m05_calculus",
    "m06_prob",
    "m07_ml",
    "m08_prep",
    "m09_neuron",
    "m10_architecture",
    "m11_activations",
    "m12_forward",
    "m13_loss",
    "m14_backprop",
    "m15_optim",
    "m16_training_loop",
    "m17_batch_epoch",
    "m18_eval",
    "m19_generalization",
    "m20_regularization",
    "m21_frameworks",
    "m22_gallery",
    "m23_ready",
]
