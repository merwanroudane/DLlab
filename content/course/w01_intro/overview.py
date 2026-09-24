import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, research_note, takeaway
from components.comparison import compare_table
from components.diagram import svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w01.overview",
    title_ar="نظرة عامة على الأسبوع وربطه بالأسس",
    title_en="Week Overview & Links to Foundations",
    module="course.w01",
    order=1,
    prerequisites=["foundations.start.ai_ml_dl", "foundations.start.model_training_prediction"],
    objectives_ar=[
        "معرفة ما يغطيه الأسبوع الأول وما يُتوقع منك في نهايته.",
        "فهم لماذا أصبح التعلم العميق ممكنًا الآن (بيانات، حساب، خوارزميات، برمجيات) عبر خط زمني.",
        "إجراء اختبار قبلي سريع (Pre-test) لتحديد ما تحتاج مراجعته من الأسس.",
    ],
    terms=["deep_learning", "model", "training", "inference", "gpu"],
    difficulty="beginner",
    summary_ar="خريطة الأسبوع الأول، لماذا الآن؟ (خط زمني من 1943 إلى اليوم)، الاختبار القبلي، وروابط الأسس التي يعتمد عليها.",
)

# (year, milestone EN, Arabic explanation, what it unlocked)
_TIMELINE = [
    (1943, "McCulloch & Pitts neuron", "**1943 — عصبون McCulloch و Pitts**: أول نموذج رياضي لعصبون: يجمع مدخلات موزونة ويطلق إشارة إذا تجاوز المجموع عتبة. لا تعلّم بعد — الأوزان تُضبط يدويًا.", "neuron = weighted sum + threshold"),
    (1958, "Rosenblatt's perceptron", "**1958 — بيرسبترون Rosenblatt**: أول عصبون **يتعلم أوزانه من البيانات** بقاعدة تحديث بسيطة. هذه بذرة التعلم الآلي بالشبكات.", "weights learned from data"),
    (1969, "Minsky & Papert, Perceptrons", "**1969 — كتاب Perceptrons لـ Minsky و Papert**: أثبت أن عصبونًا بطبقة واحدة لا يستطيع حل مسائل بسيطة مثل `XOR`. تراجع الاهتمام بالشبكات سنوات طويلة.", "limits of one layer"),
    (1986, "Backpropagation (Rumelhart, Hinton & Williams)", "**1986 — الانتشار العكسي**: Rumelhart و Hinton و Williams نشروا طريقة فعالة لحساب تدرجات شبكة **متعددة الطبقات** بقاعدة السلسلة — فأمكن تدريب طبقات مخفية تحل XOR وأكثر.", "training hidden layers"),
    (1997, "LSTM (Hochreiter & Schmidhuber)", "**1997 — LSTM**: خلية ذاكرة ببوابات تحافظ على المعلومات عبر تسلسلات طويلة وتخفف تلاشي التدرج. ستدرسها في الأسبوع 11.", "memory over long sequences"),
    (1998, "LeNet-5 CNN (LeCun et al.)", "**1998 — LeNet-5**: شبكة التفافية `CNN` من LeCun وزملائه قرأت الأرقام المكتوبة بخط اليد (استُعملت في قراءة الشيكات). ستدرسها في الأسبوع 08.", "convolution for images"),
    (2012, "AlexNet wins ImageNet", "**2012 — AlexNet**: Krizhevsky و Sutskever و Hinton درّبوا CNN عميقة على **معالجات رسومية** `GPU` فتفوقت بفارق كبير في مسابقة ImageNet. بداية الموجة الحديثة.", "GPUs + big data + depth"),
    (2017, "Transformer (Vaswani et al.)", "**2017 — Transformer**: بنية تعتمد على «الانتباه» `attention` بدل التكرار، أساس نماذج اللغة الكبيرة الحالية.", "attention replaces recurrence"),
    (2022, "Large language models go public", "**2022 وما بعدها — نماذج اللغة الكبيرة**: نماذج Transformer بمليارات المعلمات مدرَّبة على نصوص هائلة أصبحت أدوات يومية. نفس حلقة التدريب التي ستتعلمها — على نطاق ضخم.", "same loop, huge scale"),
]


def _timeline_svg(k: int) -> str:
    x0, x1, y = 40, 620, 110
    sx = lambda yr: x0 + (yr - 1940) / (2025 - 1940) * (x1 - x0)
    s = '<svg viewBox="0 0 660 190" width="100%" style="max-width:660px">'
    s += f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#B9B2A6" stroke-width="2"/>'
    for dec in range(1940, 2030, 10):
        s += f'<line x1="{sx(dec)}" y1="{y - 4}" x2="{sx(dec)}" y2="{y + 4}" stroke="#B9B2A6"/>' + svg_text(sx(dec), y + 20, str(dec), size=10, color="#6B675F")
    for i, (yr, name, _, _) in enumerate(_TIMELINE):
        if i > k:
            s += f'<circle cx="{sx(yr):.1f}" cy="{y}" r="5" fill="#fff" stroke="#D9D3C7" stroke-width="1.5"/>'
            continue
        on = i == k
        s += f'<circle cx="{sx(yr):.1f}" cy="{y}" r="{9 if on else 6}" fill="{"#1F7A78" if on else "#9FD3CF"}" stroke="#1F7A78"/>'
        if on:
            s += f'<line x1="{sx(yr):.1f}" y1="{y - 10}" x2="{sx(yr):.1f}" y2="62" stroke="#1F7A78" stroke-dasharray="3 3"/>'
            anchor = "start" if sx(yr) < 200 else ("end" if sx(yr) > 480 else "middle")
            s += svg_text(sx(yr), 38, str(yr), size=18, bold=True, color="#1F7A78", anchor=anchor)
            s += svg_text(sx(yr), 57, name, size=12, bold=True, anchor=anchor)
            s += svg_text(sx(yr), 160, _TIMELINE[i][3], size=12, color="#C8473A", anchor=anchor, mono=True)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Intro to DL", "Core concepts", "ML vs DL", "Training live", "Applications", "Post-test"], active=0)
    table(
        ["الموضوع", "الدرس في المقرر", "الدرس التأسيسي الذي يشرحه"],
        [
            ("ما التعلم العميق؟ الفرق عن التعلم الآلي", "المفاهيم الأساسية", "الأسس 0: AI vs ML vs DL"),
            ("نموذج / بنية / تدريب / استدلال", "المفاهيم الأساسية", "الأسس 0: النموذج والتدريب والتنبؤ"),
            ("مشاهدة التدريب: خسارة، تدرج، معدل تعلم", "المفاهيم الأساسية", "الأسس 5 و15: المشتقة والنزول بالتدرج"),
            ("لماذا بيانات + رياضيات + تحسين", "المفاهيم الأساسية", "الأسس 0: الأركان الثلاثة"),
            ("تطبيقات اقتصادية وإدارية، وحدود التفسير", "التطبيقات", "الأسس 1: الهدف ونوع المسألة"),
        ],
        ["rtl", "rtl", "rtl"],
    )
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تشرح لزميل الفرق بين AI و ML و DL بمثال من تخصصك.\n"
        "- أن تصف ما يحدث في حقبة تدريب واحدة (تنبؤ ← خسارة ← تدرج ← تحديث) وأن تتعرف على معدل تعلم كبير من منحنى الخسارة.\n"
        "- أن تحوّل سؤالًا بحثيًا إلى مسألة تعلم آلي وتختار عائلة البنية المناسبة — أو تقرر أن الشبكة غير ضرورية.\n"
        "- أن تفرّق بين قدرة النموذج على التنبؤ وادعاء السببية."
    )
    practical_note("إن لم تكن قد أنهيت الوحدة 0 من الأسس، فافعل ذلك أولًا؛ هذا الأسبوع يفترضها.")
    with st.container(horizontal=True):
        st.button("الوحدة 0 — ابدأ من هنا", icon=":material/flag:", on_click=go, args=("foundations.start",), key="w01_go_start")
        st.button("الوحدة 1 — أسس البيانات", icon=":material/table_chart:", on_click=go, args=("foundations.data",), key="w01_go_data")

    h2("لماذا الآن؟ ثمانون عامًا في دقيقة", "Why now? Eighty years in one minute")
    st.markdown("فكرة الشبكات العصبية قديمة؛ ما تغيّر هو اجتماع الشروط. شغّل الخط الزمني:")
    frames = [Frame(_timeline_svg(i), caption(txt), action=str(yr), highlight=None) for i, (yr, _, txt, _) in enumerate(_TIMELINE)]
    animation_player("w01_timeline", frames, title_ar="محطات التعلم العميق", interval_ms=2600)
    compare_table(
        ["الشرط", "قبل 2010 تقريبًا", "اليوم"],
        [
            ("البيانات", "آلاف الأمثلة الموسومة", "ملايين الصور والنصوص والمعاملات الرقمية"),
            ("الحساب", "معالجات مركزية `CPU` بطيئة لضرب المصفوفات", "معالجات رسومية `GPU` تنفذ آلاف العمليات بالتوازي"),
            ("الخوارزميات", "تدرجات تتلاشى في الشبكات العميقة", "`ReLU`، تهيئة أفضل، `Dropout`، `Adam`، `BatchNorm`"),
            ("البرمجيات", "كل باحث يكتب المشتقات بيده", "TensorFlow/Keras (2015) و PyTorch (2016): اشتقاق آلي مجاني"),
        ],
        ["rtl", "rtl", "rtl"],
    )
    intuition("ستجد في هذا المقرر الشروط الأربعة: البيانات (الأسس 1)، الحساب (الأسبوع 13)، الخوارزميات (الأسابيع 05–06)، والبرمجيات (الأسبوع 03).")

    h2("الاختبار القبلي", "Pre-test")
    st.markdown("اختبار قصير **قبل** الدرس لقياس نقطة انطلاقك. لا يُحتسب؛ استخدمه لتعرف ما تراجعه.")
    quiz(
        "w01.pretest",
        [
            Q("التعلم العميق هو…", ["مرادف للذكاء الاصطناعي", "فرع من التعلم الآلي يستخدم شبكات متعددة الطبقات", "أي برنامج يستخدم بيانات"], 1,
              "AI ⊃ ML ⊃ DL. راجع درس AI vs ML vs DL في الأسس 0 إن أخطأت."),
            Q("أثناء الاستدلال، المعلمات…", ["تتغير", "مجمّدة", "تُحذف"], 1, "المعلمات تتغير في التدريب فقط. راجع درس النموذج والتدريب."),
            Q("ما الذي يقلّله التدريب؟", ["عدد الخصائص", "الخسارة", "حجم الدفعة"], 1, "التدريب = تقليل الخسارة بتحديث المعلمات."),
            Q("لماذا ارتبطت نهضة التعلم العميق بعام 2012 تقريبًا؟", ["اختراع الشبكات العصبية في 2012", "اجتماع بيانات كبيرة + GPU + خوارزميات أفضل", "توقف الإحصاء عن العمل"], 1,
              "الشبكات أقدم بكثير (1958)؛ الجديد هو اجتماع الشروط (AlexNet 2012)."),
            Q("نموذج يتنبأ بالتضخم من الفائدة بدقة عالية. هل يثبت أن الفائدة تسبب التضخم؟", ["نعم", "لا؛ التنبؤ لا يعني السببية", "فقط إذا كانت الدقة 100%"], 1,
              "نزاهة البحث: تنبؤ ≠ سببية. سترى محاكاة تثبت ذلك في درس التطبيقات."),
        ],
        title_ar="الاختبار القبلي — الأسبوع 01",
    )
    research_note("سنعود إلى أسئلة مماثلة في الاختبار البعدي نهاية الأسبوع لقياس التقدم.")
    takeaway("الأسبوع 01 يبني اللغة المشتركة. الشبكات فكرة قديمة نجحت حين اجتمعت البيانات والحساب والخوارزميات والبرمجيات. الاختبار القبلي يوجهك إلى ما تراجعه في الأسس.")
    lesson_footer(LESSON, ["الأسبوع الأول = مفاهيم + فرق ML/DL + تدريب مرئي + تطبيقات.", "محطات: 1958 بيرسبترون، 1986 انتشار عكسي، 2012 AlexNet، 2017 Transformer.", "أنهِ الوحدة 0 من الأسس أولًا."])
