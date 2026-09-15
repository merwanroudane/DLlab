import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.from_python_to_framework",
    title_ar="قبل أطر العمل: من بايثون إلى إطار تعلم عميق",
    title_en="Before Frameworks: From Python to a Deep Learning Framework",
    module="foundations.frameworks",
    order=1,
    prerequisites=["foundations.training_loop.training_loop_code", "foundations.backprop.backpropagation.implementation"],
    objectives_ar=["رسم المسار: بايثون ← NumPy ← موترات ← اشتقاق تلقائي ← طبقات/نماذج ← خسارة+محسّن ← حلقة تدريب ← CPU/GPU ← إطار العمل.", "التفريق بين لغة برمجة ومكتبة وإطار عمل وواجهة عليا وخلفية وبيئة تشغيل ومسرّع.", "فهم لماذا لا نكتب ضرب المصفوفات والتدرجات ونوى GPU يدويًا في كل مشروع."],
    terms=["tensor", "ndarray", "vectorization", "backpropagation"],
    difficulty="beginner",
    summary_ar="إطار العمل = كل ما بنيته يدويًا في الوحدات 9–20 (موترات، اشتقاق تلقائي، طبقات، محسّنات، حلقة تدريب) جاهزًا وسريعًا وقابلًا للتشغيل على GPU. ما يبقى لك: البيانات، التصميم، التشخيص، التفسير.",
)

CODE = '''import numpy as np, time
# ما فعلناه يدويًا في الوحدة 14: تمرير أمامي + خلفي لطبقة كثيفة
rng = np.random.default_rng(0)
X = rng.normal(size=(256, 64)).astype(np.float32); y = rng.normal(size=(256, 1)).astype(np.float32)
W = rng.normal(0, 0.1, (64, 1)).astype(np.float32); b = np.zeros((1,), np.float32)

t = time.perf_counter()
for step in range(200):
    y_hat = X @ W + b                       # forward (ضرب مصفوفات + بث)
    dz = 2 * (y_hat - y) / len(X)           # ∂L/∂z لخسارة MSE  (اشتقاق يدوي)
    gW, gb = X.T @ dz, dz.sum(0)            # backward يدوي
    W -= 0.1 * gW; b -= 0.1 * gb            # تحديث يدوي
print(f"manual NumPy: 200 steps in {1000*(time.perf_counter()-t):.1f} ms, final MSE = {((X @ W + b - y)**2).mean():.4f}")
print()
print("ما الذي كتبناه بأنفسنا؟")
print("  1. ضرب المصفوفات        -> NumPy يفعله (BLAS مترجم)، لكن على CPU فقط")
print("  2. مشتقة الخسارة        -> يدويًا: يجب اشتقاق كل خسارة وكل طبقة وكل تنشيط")
print("  3. التمرير الخلفي       -> يدويًا: خطأ صغير = تدرج خاطئ بصمت")
print("  4. قاعدة التحديث        -> يدويًا: Adam يحتاج ~10 أسطر إضافية لكل معلمة")
print("  5. الدفعات/الخلط/التحقق -> يدويًا")
print("  6. GPU                  -> مستحيل هنا بدون إعادة كتابة كل شيء")
print("إطار العمل يعطيك 1-6 جاهزة ومختبَرة. يبقى لك: البيانات، التصميم، التشخيص.")'''


def _path_svg() -> str:
    steps = [("Python", "#F1EFEA"), ("NumPy arrays", "#E6F1FB"), ("Tensors", "#E6F1FB"), ("Automatic|differentiation", "#FBE6E2"), ("Layers / Models", "#E3F3F0"),
             ("Loss + Optimizer", "#FBE6E2"), ("Training loop", "#EFE9F8"), ("CPU / GPU", "#FFF3D6"), ("Framework", "#1F7A78")]
    s = '<svg viewBox="0 0 720 300" width="100%" style="max-width:720px">' + svg_defs()
    for i, (lbl, fill) in enumerate(steps):
        col, row = i % 5, i // 5
        x, y = 20 + col * 140, 30 + row * 130
        if "|" in lbl:
            a, b_ = lbl.split("|")
            s += svg_box(x, y, 120, 60, "", fill, stroke="#B9B2A6") + svg_text(x + 60, y + 26, a, size=12, bold=True) + svg_text(x + 60, y + 44, b_, size=12, bold=True)
        else:
            s += svg_box(x, y, 120, 60, lbl, fill, stroke="#B9B2A6" if i < 8 else "#1F7A78", font=13, bold=True, text_color="#2B2A28" if i < 8 else "#FFFFFF")
        if i < 8:
            if col < 4:
                s += svg_arrow(x + 122, y + 30, x + 138, y + 30)
            else:
                s += f'<path d="M{x + 60},{y + 62} L{x + 60},{y + 95} L80,{y + 95} L80,{y + 128}" fill="none" stroke="#6B675F" stroke-width="2" marker-end="url(#arrowhead)"/>'
    s += svg_text(360, 285, "each layer hides the layer beneath it — but you have already built every one of them by hand", size=12, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("المسار الذي قطعته بالفعل", "The path you have already walked")
    diagram("من بايثون إلى إطار العمل", _path_svg(), what_ar="تسع طبقات من التجريد. كل واحدة تخفي التي قبلها: NumPy يخفي حلقات C، الموترات تخفي الأجهزة، الاشتقاق التلقائي يخفي قاعدة السلسلة…",
            how_ar="اقرأ من اليسار إلى اليمين ثم إلى السطر الثاني. اسأل عند كل صندوق: «في أي وحدة بنيت هذا بيدي؟» (NumPy: وحدة 2، الموترات: وحدة 4، الاشتقاق: وحدة 14، الطبقات: 10–12، الخسارة والمحسّن: 13 و15، الحلقة: 16).",
            takeaway_ar="إطار العمل ليس معرفة جديدة — بل تغليف لمعرفة تملكها الآن.", title_en="From Python to a framework")
    why("في الوحدة 14 كتبت `backward` كاملًا وتحققت منه عدديًا. تخيّل فعل ذلك لشبكة من 50 طبقة بأنواع مختلفة، ثم إعادة كتابته لتشغيله على GPU، ثم إضافة Adam والقصّ والإيقاف المبكر… لكل مشروع. إطار العمل يجعل ذلك **مكتوبًا مرة واحدة، مختبَرًا، وسريعًا** — ويترك لك ما لا يمكن أتمتته: فهم البيانات، تصميم النموذج، تشخيص المشاكل، وتفسير النتائج.")
    h2("المصطلحات: لغة، مكتبة، إطار عمل، واجهة عليا، خلفية، بيئة تشغيل، مسرّع", "Language, library, framework, high-level API, backend, runtime, accelerator")
    compare_table(["المصطلح", "التعريف", "مثال", "ما الذي يعطيك التحكم به؟"],
                  [("لغة برمجة", "قواعد وصيغة لكتابة التعليمات؛ لا تعرف شيئًا عن الشبكات العصبية", "Python", "كل شيء — وتكتب كل شيء"),
                   ("مكتبة", "مجموعة دوال **تستدعيها** أنت متى شئت؛ أنت تملك تدفق البرنامج", "NumPy, pandas", "تدفق البرنامج بالكامل"),
                   ("إطار عمل", "بنية تُحدد **كيف** يُبنى النموذج ويُدرَّب؛ أنت تملأ الفراغات وهو يدير التدفق (انعكاس التحكم)", "TensorFlow, PyTorch", "الأجزاء التي يصمّمها الباحث: البنية، الخسارة، البيانات"),
                   ("واجهة عليا (High-level API)", "طبقة فوق الإطار تختصر الشائع في أسطر قليلة", "Keras (model.fit)", "قرارات قليلة صريحة؛ التفاصيل مخفية"),
                   ("خلفية (Backend)", "المحرك الذي ينفذ العمليات فعليًا تحت الواجهة العليا", "TensorFlow under Keras (this course)", "لا تلمسه عادةً؛ يحدد السرعة والأجهزة"),
                   ("بيئة التشغيل (Runtime)", "المكان/العملية التي يعمل فيها الكود", "Python process, Colab, Jupyter", "ليست إطار عمل"),
                   ("مسرّع (Accelerator)", "عتاد يوازي ضرب المصفوفات على آلاف الأنوية", "GPU, TPU", "تُنقل إليه الموترات؛ الحساب نفسه لا يتغير")],
                  ["rtl", "rtl", "ltr", "rtl"])
    intuition("**مكتبة**: أنت تستدعي الكود. **إطار عمل**: الكود يستدعيك — تعطيه `forward` وخسارة وبيانات، وهو يدير الحلقة والذاكرة والأجهزة. Keras تدفع ذلك إلى أقصاه: تصف النموذج وتقول `fit`. PyTorch يبقي الحلقة في يدك ويؤتمت الباقي (الاشتقاق، الطبقات، المحسّنات، GPU).")
    definition("**إطار التعلم العميق** `Deep learning framework`: برمجية تقدّم (1) موترات على CPU/GPU، (2) اشتقاقًا تلقائيًا، (3) طبقات ونماذج، (4) خسائر ومقاييس ومحسّنات، (5) أدوات بيانات وتدريب وحفظ — بحيث يكتب الباحث **ما هو خاص بمشكلته فقط**.")
    code_lab(CodeLab(
        key="fw_manual_vs", title_ar="ما كنا نكتبه يدويًا — وما يتولاه الإطار", code=CODE,
        before=Before(goal_ar="إعادة تشغيل انحدار خطي يدوي بـ NumPy ثم تعداد ما كتبناه بأنفسنا وما سيتولاه إطار العمل.", stage_ar="مقدمة أطر العمل.",
                      inputs_ar="X بشكل (256, 64)، y بشكل (256, 1).", expected_ar="زمن التنفيذ وMSE النهائي، ثم قائمة بستة أشياء كتبناها يدويًا."),
        explain=[("3-5", "بيانات ومعلمات كما في الوحدة 16."), ("8-12", "الحلقة اليدوية: أمامي، مشتقة الخسارة (اشتقاق ورقي)، خلفي، تحديث. أربعة أسطر لطبقة واحدة وخسارة واحدة — و**كل** تغيير في البنية يعني إعادة الاشتقاق."),
                 ("15-22", "قائمة المسؤوليات. لاحظ البند 6: NumPy لا يعمل على GPU؛ نقل هذا الكود إلى GPU يعني إعادة كتابته بالكامل.")],
        run=run_printed(CODE),
        after_ar="- الزمن صغير هنا لأن الشبكة صغيرة؛ مع صور وملايين المعلمات تصبح أسطر الحلقة نفسها لكن الحساب أبطأ 1000× دون GPU.\n- إطار العمل لا يغيّر **الرياضيات**: `X @ W + b`، `2(ŷ−y)/n`، `Xᵀδ` هي ما ينفذه بالضبط داخله.",
    ))
    h2("لماذا لا نكتب كل شيء يدويًا؟", "Why not write it all by hand?")
    st.markdown("""
- **ضرب المصفوفات**: NumPy يستدعي BLAS المترجم — لكن على CPU. نوى GPU (CUDA kernels) لغة أخرى وهندسة أخرى؛ إعادة كتابتها لكل عملية ولكل مشروع غير عملي.
- **حساب التدرجات**: الاشتقاق اليدوي صحيح لطبقة واحدة، وخطأ شبه مؤكد لشبكة بـ 50 طبقة وتفرّعات. الاشتقاق التلقائي يطبق قاعدة السلسلة آليًا على **أي** رسم حسابي (وحدة 14).
- **المحسّنات والتنظيم**: Adam، القصّ، الجداول، Dropout، BN — كلها مكتوبة ومختبَرة ومتّسقة.
- **التكرار والأخطاء**: أخطاء التدرج صامتة (وحدة 14: التحقق العددي). الإطار مختبَر على ملايين الاستخدامات.
- **الأجهزة**: نفس الكود يعمل على CPU وGPU وTPU بتغيير سطر.
""")
    common_mistake("«سأتعلم الإطار بدل الرياضيات». الإطار يخفي الحساب لا الفهم: عندما تنفجر الخسارة أو يتلاشى التدرج أو تتعطل الأشكال، لا يخبرك `fit()` لماذا. الوحدات 9–20 هي أدوات التشخيص؛ هذه الوحدة تعلّمك كيف تطلب من الإطار ما تفهمه.")
    quiz("fw.intro", [
        Q("الفرق الجوهري بين المكتبة وإطار العمل…", ["الحجم", "من يملك تدفق البرنامج: أنت (مكتبة) أم هو (إطار)", "اللغة"], 1, "انعكاس التحكم."),
        Q("Google Colab هو…", ["إطار تعلم عميق", "بيئة تشغيل (runtime)", "خلفية Keras"], 1, "مكان يعمل فيه الكود."),
        Q("ماذا يبقى مسؤولية الباحث مع إطار العمل؟", ["لا شيء", "البيانات والتصميم والتشخيص والتفسير", "كتابة نوى GPU"], 1, "ما لا يمكن أتمتته."),
        Q("الاشتقاق التلقائي…", ["يغيّر الرياضيات", "يطبق قاعدة السلسلة آليًا على الرسم الحسابي", "يعمل فقط للطبقات الكثيفة"], 1, "وحدة 14 آليًا."),
    ])
    takeaway("إطار العمل = موترات + اشتقاق تلقائي + طبقات + محسّنات + حلقة تدريب + CPU/GPU، مكتوبة مرة واحدة. لغة ≠ مكتبة ≠ إطار ≠ واجهة عليا ≠ خلفية ≠ بيئة تشغيل ≠ مسرّع.")
    lesson_footer(LESSON, ["المسار التسعي من بايثون إلى الإطار.", "سبعة مصطلحات لا تُخلط.", "ما يتولاه الإطار وما يبقى لك."])
