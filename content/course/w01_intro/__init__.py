from core.models import Module

MODULE = Module(
    id="course.w01",
    section="course",
    title_ar="الأسبوع 01 — مقدمة في التعلم العميق",
    title_en="Week 01 — Introduction to Deep Learning",
    order=1,
    icon=":material/looks_one:",
    prerequisites=["foundations.start"],
    purpose_ar="المفاهيم الأساسية، الفرق بين التعلم الآلي والتعلم العميق، وأبرز التطبيقات في التخصصات الاقتصادية والإدارية.",
    why_ar="هذا الأسبوع يثبّت اللغة المشتركة للمقرر كله: نموذج، بنية، تدريب، استدلال، وحدود ما يمكن للتعلم العميق ادعاؤه.",
    objectives_ar=[
        "التمييز بين الذكاء الاصطناعي والتعلم الآلي والتعلم العميق.",
        "شرح مفاهيم النموذج والبنية والتدريب والاستدلال بأمثلة مبسطة.",
        "ذكر تطبيقات في الاقتصاد والإدارة وتحديد نوع المسألة لكل منها.",
    ],
    challenges_ar=["الخلط بين التنبؤ والسببية.", "توقع أن التعلم العميق يحل كل مسألة بيانات."],
)

LESSON_MODULES = [
    "overview",
    "core_concepts",
    "applications",
]
