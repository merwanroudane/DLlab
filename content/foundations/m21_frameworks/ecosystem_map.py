import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from labs.fw import versions

LESSON = Lesson(
    id="foundations.frameworks.ecosystem_map",
    title_ar="خريطة المنظومة: أين يقع كل شيء؟",
    title_en="Framework Ecosystem Map",
    module="foundations.frameworks",
    order=2,
    prerequisites=["foundations.frameworks.from_python_to_framework"],
    objectives_ar=["وضع Python وNumPy وTensorFlow وKeras وPyTorch وscikit-learn وJupyter/Colab وTensorBoard في أماكنها الصحيحة.", "عدم الخلط بين إطار العمل والمنصة/الدفتر والأداة.", "معرفة النسخ المثبتة في هذا المشروع."],
    terms=["ndarray", "tensor"],
    difficulty="beginner",
    summary_ar="طبقات: اللغة (Python) ← الأساس العددي (NumPy) ← أطر العمل (TensorFlow، PyTorch) ← الواجهة العليا (Keras فوق TensorFlow) — وبجانبها: scikit-learn (ML تقليدي)، Jupyter/Colab (بيئات تشغيل)، TensorBoard (أداة مراقبة).",
)


def _map_svg() -> str:
    s = '<svg viewBox="0 0 760 380" width="100%" style="max-width:760px">' + svg_defs()
    # layers (bottom to top)
    s += svg_box(40, 300, 460, 44, "Python  (language)", "#F1EFEA", font=14, bold=True)
    s += svg_box(40, 240, 460, 44, "NumPy  (numerical arrays on CPU)", "#E6F1FB", stroke="#2F6FB5", font=14, bold=True)
    s += svg_box(40, 150, 220, 70, "", "#FBE6E2", stroke="#C8473A") + svg_text(150, 178, "TensorFlow", size=15, bold=True) + svg_text(150, 200, "framework · tensors · autodiff · GPU", size=11, color="#6B675F")
    s += svg_box(280, 150, 220, 70, "", "#EFE9F8", stroke="#7C5CBF") + svg_text(390, 178, "PyTorch", size=15, bold=True) + svg_text(390, 200, "framework · tensors · autograd · GPU", size=11, color="#6B675F")
    s += svg_box(40, 80, 220, 50, "", "#E3F3F0", stroke="#1F7A78") + svg_text(150, 100, "Keras", size=15, bold=True) + svg_text(150, 119, "high-level API (backend: TensorFlow here)", size=11, color="#6B675F")
    s += svg_arrow(150, 132, 150, 148) + svg_arrow(150, 222, 150, 238) + svg_arrow(390, 222, 390, 238) + svg_arrow(270, 282, 270, 298)
    # side: not frameworks
    s += svg_box(540, 80, 200, 56, "", "#FFF3D6", stroke="#D9A21B") + svg_text(640, 102, "scikit-learn", size=14, bold=True) + svg_text(640, 122, "classical ML library (reference)", size=11, color="#6B675F")
    s += svg_box(540, 160, 200, 56, "", "#FFF3D6", stroke="#D9A21B") + svg_text(640, 182, "Jupyter / Google Colab", size=14, bold=True) + svg_text(640, 202, "runtimes / notebooks — not frameworks", size=11, color="#6B675F")
    s += svg_box(540, 240, 200, 56, "", "#FFF3D6", stroke="#D9A21B") + svg_text(640, 262, "TensorBoard", size=14, bold=True) + svg_text(640, 282, "visualization / monitoring tool", size=11, color="#6B675F")
    s += svg_text(640, 330, "these sit BESIDE the stack, not inside it", size=12, color="#D9A21B", bold=True)
    s += svg_text(270, 30, "Deep learning stack (this course: Keras on TensorFlow, and PyTorch)", size=13, bold=True)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    diagram("خريطة المنظومة", _map_svg(), what_ar="المكدس على اليسار من الأسفل إلى الأعلى: لغة ← مصفوفات ← أطر عمل ← واجهة عليا. الصناديق الصفراء على اليمين ليست أطر تعلم عميق: مكتبة ML تقليدية، بيئات تشغيل، وأداة مراقبة.",
            how_ar="السهم يعني «يُبنى فوق / يعتمد على». Keras فوق TensorFlow (في هذا المقرر). PyTorch إطار مستقل بجانب TensorFlow، وكلاهما فوق NumPy/Python للتبادل. ما على اليمين يمكن استخدامه مع أي إطار — أو بدون أي إطار.",
            takeaway_ar="سؤالان يضعان أي اسم في مكانه: هل يوفّر موترات واشتقاقًا تلقائيًا؟ (إطار) وهل يعمل بداخله الكود أم يُستدعى منه؟ (بيئة/مكتبة).",
            legend=[("#E3F3F0", "واجهة عليا"), ("#FBE6E2", "إطار عمل"), ("#EFE9F8", "إطار عمل"), ("#E6F1FB", "أساس عددي"), ("#FFF3D6", "ليس إطار تعلم عميق")], title_en="Ecosystem map")
    h2("كل اسم في سطر", "Each name in one line")
    compare_table(["الاسم", "ما هو", "ما ليس هو", "دوره في هذا المقرر"],
                  [("Python", "لغة البرمجة", "ليس إطارًا ولا يعرف الشبكات", "كل شيء يُكتب بها"),
                   ("NumPy", "مكتبة مصفوفات عددية على CPU", "لا اشتقاق تلقائي، لا GPU", "الأساس (وحدات 2–20)؛ الأطر تتبادل معه"),
                   ("TensorFlow", "إطار تعلم عميق: موترات، اشتقاق تلقائي، tf.data، أجهزة، حفظ", "ليس Keras — Keras واجهة فوقه", "الخلفية التي تشغّل Keras في المقرر"),
                   ("Keras", "واجهة عليا لبناء وتدريب النماذج بأسطر قليلة", "ليس إطارًا مستقلًا؛ يحتاج خلفية", "الواجهة الأساسية للمقرر (Keras 3 بخلفية TensorFlow)"),
                   ("PyTorch", "إطار تعلم عميق بحلقة تدريب صريحة", "ليس «أصعب» ولا «أفضل»؛ أسلوب مختلف", "المسار الثاني الكامل"),
                   ("scikit-learn", "مكتبة ML تقليدي (انحدار، أشجار، تقسيم، مقاييس)", "ليس إطار تعلم عميق", "مرجع ومساعد: train_test_split، المقاييس، خطوط الأساس"),
                   ("Jupyter / Colab", "بيئات تشغيل (دفاتر) — Colab يوفر GPU مجانيًا", "ليست أطر عمل؛ لا تغيّر الكود", "مكان تشغيل مشاريع الأسابيع"),
                   ("TensorBoard", "أداة تصوّر ومراقبة للسجلات (منحنيات، رسوم، هيستوغرامات)", "ليست TensorFlow نفسه؛ تعمل مع PyTorch أيضًا", "مراقبة التدريب (درس لاحق)")],
                  ["ltr", "rtl", "rtl", "rtl"])
    definition("**Keras 3** في هذا المشروع يعمل بخلفية **TensorFlow** — وهذا مُصرَّح به في كل درس. تدعم Keras 3 خلفيات أخرى (JAX، PyTorch) عبر `KERAS_BACKEND`؛ نذكر ذلك كمعلومة تعميق فقط ولا يغيّر شيئًا في المسار الأساسي.")
    h2("النسخ المثبتة في هذا المشروع", "Versions pinned in this project")
    v = versions()
    st.code("\n".join(f"{k:<12} {val}" for k, val in v.items()), language="text")
    practical_note("واجهات API تتغير بين النسخ (مثلًا Keras 2 ↔ Keras 3، أو `torch.inference_mode`). كل كود في هذه الوحدة يعمل على النسخ أعلاه بالضبط، وهي مثبّتة في `requirements.txt`. عند قراءة درس قديم على الإنترنت تحقق أولًا من النسخة.")
    common_mistake("«أستخدم Colab إذًا أستخدم TensorFlow». Colab بيئة تشغيل تأتي مثبّتًا فيها TensorFlow وPyTorch معًا؛ اختيار الإطار قرارك أنت داخل الدفتر.")
    quiz("fw.map", [
        Q("Keras هي…", ["إطار مستقل", "واجهة عليا تعمل فوق خلفية (TensorFlow هنا)", "بيئة تشغيل"], 1, "high-level API."),
        Q("TensorBoard…", ["هو TensorFlow", "أداة مراقبة/تصوّر تعمل مع أطر عدة", "إطار عمل"], 1, "أداة."),
        Q("scikit-learn…", ["إطار تعلم عميق أساسي", "مكتبة ML تقليدي مرجعية", "خلفية Keras"], 1, "ML تقليدي."),
        Q("Jupyter وColab…", ["أطر عمل", "بيئات تشغيل/دفاتر", "مكتبات مصفوفات"], 1, "runtime."),
    ])
    takeaway("المكدس: Python ← NumPy ← {TensorFlow, PyTorch} ← Keras (فوق TensorFlow). بجانبه لا داخله: scikit-learn، Jupyter/Colab، TensorBoard.")
    lesson_footer(LESSON, ["الخريطة: مكدس + جانب.", "Keras 3 بخلفية TensorFlow في هذا المقرر.", "النسخ مثبّتة وموثّقة."])
