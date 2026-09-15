"""Official course — Weeks 01–15 (spec §29). All fifteen weeks are mandatory."""

from core.models import Section

SECTION = Section(
    id="course",
    title_ar="المقرر الرسمي — التعلم العميق",
    title_en="Official Deep Learning Course — Weeks 01–15",
    icon=":material/school:",
    order=2,
    description_ar=(
        "البرنامج الرسمي للمادة على خمسة عشر أسبوعًا، من مقدمة التعلم العميق إلى "
        "الشبكات الالتفافية والتكرارية ومشروع التخرج. كل أسبوع يربط محتواه بدروس "
        "أكاديمية الأسس التي يعتمد عليها."
    ),
)

MODULE_PACKAGES = [
    "w01_intro",
    "w02_ml",
    "w03_frameworks",
    "w04_ffnn",
    "w05_optim",
    "w06_activations",
    "w07_applied",
    "w08_cnn",
    "w09_cnn_apps",
    "w10_rnn",
    "w11_lstm",
    "w12_gru",
    "w13_gpu",
    "w14_project",
    "w15_presentation",
]
