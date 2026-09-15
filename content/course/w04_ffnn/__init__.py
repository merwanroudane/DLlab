from core.models import Module

MODULE = Module(
    id="course.w04",
    section="course",
    title_ar="الأسبوع 04 — الشبكات العصبية الأمامية",
    title_en="Week 04 — Feedforward Neural Networks",
    order=4,
    icon=":material/looks_4:",
    prerequisites=["course.w03", "foundations.architecture", "foundations.forward"],
    purpose_ar="مفهوم التغذية الأمامية، MLP، طبقات المدخل/المخفية/المخرج، الأوزان والانحيازات، التمرير الأمامي، بناء نماذج متعددة الطبقات في Keras، عدّ المعلمات، والتدفق المرئي.",
    why_ar="MLP هو النموذج الذي تُبنى عليه كل البنى اللاحقة (CNN وRNN تضيف بنية على نفس الفكرة). من يفهم التدفق والأشكال والمعلمات هنا يقرأ أي بنية لاحقًا.",
    objectives_ar=["تتبع دفعة عبر MLP بأشكالها ومعادلاتها.", "بناء شبكات بأعماق وعروض مختلفة في Keras وعدّ معلماتها.", "قياس أثر العمق والعرض على مسألة حقيقية واختيار بنية بأدلة."],
    challenges_ar=["الاعتقاد أن الأعمق أفضل دائمًا.", "الخلط بين عدد الطبقات وعدد المعلمات."],
)

LESSON_MODULES = ["overview", "forward_flow", "depth_width"]
