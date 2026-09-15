import streamlit as st

from components.callouts import common_mistake, intuition, practical_note, research_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.framework_choice",
    title_ar="دليل اختيار الإطار، وتوجيه اختياري في المنظومة الأوسع",
    title_en="Framework Choice Guide & Optional Ecosystem Orientation",
    module="foundations.frameworks",
    order=30,
    prerequisites=["foundations.frameworks.same_model_three_views"],
    objectives_ar=["مقارنة بلا تحيز: منحنى التعلم، مستوى التجريد، صراحة الحلقة، أسلوب التصحيح، سير العمل البحثي، الإنتاج، ومتطلب المقرر مقابل التفضيل.", "أداة قرار قصيرة تعطي توصية بحسب سياقك.", "تحديد موقع JAX وscikit-learn وHugging Face وONNX وLightning في المنظومة دون تحويل المقرر إلى مقارنة مكتبات."],
    terms=[],
    difficulty="beginner",
    summary_ar="لا إطار «أفضل مطلقًا». Keras: أسرع للقياسي وأقل كودًا؛ PyTorch: أوضح لغير القياسي وأكثر شيوعًا في الأبحاث. متطلب المقرر أولًا، ثم السياق. كلاهما ينفذ نفس الرياضيات.",
)


def render() -> None:
    lesson_header(LESSON)
    why("السؤال «أي إطار أتعلم؟» يُطرح قبل فهم الأطر، وجوابه بعد فهمها بسيط: **كلاهما** ينفذ نفس الوحدات 9–20. ما يختلف: كم تكتب، وكم ترى، وكيف تصحح، ومن حولك. هذه الصفحة تعرض المعايير بلا انحياز ثم تعطيك أداة قرار.")
    h2("المعايير", "The criteria")
    compare_table(["المعيار", "Keras (فوق TensorFlow)", "PyTorch", "ملاحظة منصفة"],
                  [("منحنى التعلم", "أقصر للنموذج الأول: 4 استدعاءات", "أطول قليلًا: تكتب الحلقة", "بعد هذه الوحدة الفرق صغير — أنت تعرف الحلقة أصلًا"),
                   ("مستوى التجريد", "عالٍ افتراضيًا؛ يمكن النزول (GradientTape)", "منخفض افتراضيًا؛ يمكن الصعود (Lightning وغيرها)", "كلاهما يغطي المستويين"),
                   ("صراحة حلقة التدريب", "مخفية في fit (استدعاءات للتدخل)", "مكشوفة سطرًا سطرًا", "الصراحة تعني تحكمًا أكثر وأخطاء صامتة أكثر"),
                   ("أسلوب التصحيح", "أخطاء من طبقة TensorFlow تحت Keras؛ الرسم المجمَّع يؤخر بعض الأخطاء", "فوري بايثوني: print وpdb في أي سطر", "TF2 فوري أيضًا خارج fit"),
                   ("سير العمل البحثي", "ممتاز للقياسي؛ غير القياسي يحتاج Subclassing/حلقة مخصصة", "الافتراضي في معظم الأوراق الحديثة والمستودعات المفتوحة", "توفر الكود المرجعي عامل عملي مهم"),
                   ("الإنتاج والمنظومة", "SavedModel، TF Serving، TFLite، TF.js، TPU", "TorchServe، ONNX، TorchScript/export، دعم واسع من المكتبات", "كلاهما ناضج؛ يعتمد على البنية التحتية للجهة"),
                   ("الأجهزة", "GPU/TPU تلقائي", "GPU صريح (`.to`)، TPU عبر XLA", "—"),
                   ("الجدولية الصغيرة (مثل هذا المقرر)", "كود أقل", "كود أكثر لكنه تعليمي", "المقرر يستخدم الاثنين عن قصد")],
                  ["rtl", "rtl", "rtl", "rtl"])
    intuition("Keras: «أريد النتيجة بأقل قرارات». PyTorch: «أريد أن أرى كل قرار». الباحث الذي يفهم الرياضيات يستطيع الاثنين ويختار بحسب المهمة — لا بحسب الهوية.")
    h2("أداة قرار", "Decision helper")
    st.markdown("أجب عن الأسئلة، والتوصية تُحسب من إجاباتك (استرشادية لا حكمًا):")
    c1, c2 = st.columns(2)
    with c1:
        req = st.radio("هل يفرض المقرر/المشرف/الفريق إطارًا؟", ["لا", "Keras/TensorFlow", "PyTorch"], key="fc_req", horizontal=True)
        task = st.radio("نوع المسألة", ["قياسية (جدولية، صور/نصوص شائعة بنماذج معروفة)", "غير قياسية (خسائر/حلقات/بنى مخصصة)"], key="fc_task")
        ref = st.radio("الكود المرجعي الذي ستبني عليه", ["لا يوجد", "متوفر بـ Keras/TF", "متوفر بـ PyTorch"], key="fc_ref", horizontal=True)
    with c2:
        deploy = st.radio("النشر", ["لا نشر (بحث/دراسة)", "موبايل/متصفح (TFLite/TF.js)", "خادم عام / ONNX"], key="fc_deploy")
        debug = st.radio("تفضيلك في التصحيح", ["أقل كود، أقبل الصندوق", "أريد رؤية كل خطوة"], key="fc_debug")
    score = 0  # >0 → PyTorch, <0 → Keras
    reasons = []
    if req == "PyTorch":
        score += 10; reasons.append("متطلب صريح: PyTorch — ينهي النقاش.")
    elif req == "Keras/TensorFlow":
        score -= 10; reasons.append("متطلب صريح: Keras/TensorFlow — ينهي النقاش.")
    if task.startswith("غير"):
        score += 2; reasons.append("غير قياسية → الحلقة الصريحة تسهّل التعديل (+PyTorch).")
    else:
        score -= 1; reasons.append("قياسية → fit + callbacks توفر وقتًا (+Keras).")
    if ref.endswith("PyTorch"):
        score += 3; reasons.append("كود مرجعي بـ PyTorch → لا تترجم (+PyTorch).")
    elif ref.endswith("Keras/TF"):
        score -= 3; reasons.append("كود مرجعي بـ Keras → لا تترجم (+Keras).")
    if deploy.startswith("موبايل"):
        score -= 2; reasons.append("TFLite/TF.js مسار ناضج في TensorFlow (+Keras).")
    elif deploy.startswith("خادم"):
        reasons.append("النشر على خادم/ONNX متاح للاثنين (محايد).")
    if debug.startswith("أريد"):
        score += 1; reasons.append("رؤية كل خطوة → PyTorch (+1).")
    else:
        score -= 1; reasons.append("أقل كود → Keras (+1).")
    verdict = "PyTorch" if score > 1 else ("Keras / TensorFlow" if score < -1 else "أيهما — الفرق هامشي في حالتك؛ ابدأ بما يملك كودًا مرجعيًا أو بما يعرفه من حولك")
    with st.container(border=True):
        st.markdown(f"### التوصية: **{verdict}**")
        st.markdown("\n".join(f"- {r}" for r in reasons))
        st.caption("النقاط: " + str(score) + " (موجب = PyTorch، سالب = Keras). الأداة ترجّح ولا تقرر؛ متطلب المقرر يتقدم على كل شيء.")
    practical_note("**متطلب المقرر مقابل التفضيل الشخصي**: هذا المقرر يستخدم Keras 3 بخلفية TensorFlow في مشاريع الأسابيع، ويقدّم PyTorch كمسار كامل مواز. سلّم المشروع بما يطلبه المقرر؛ وتعلّم الآخر لأن سوق البحث يستخدمهما معًا.")
    common_mistake("«سأتعلم إطارًا واحدًا فقط». الأوراق والمستودعات لا تسألك عن تفضيلك. بعد هذه الوحدة، قراءة كود بالإطار الآخر مسألة ترجمة (خريطة المفاهيم) لا تعلّم من الصفر.")
    h2("توجيه اختياري: ما حول الإطارين", "Optional orientation: around the two frameworks")
    research_note("**غير أساسي** — لتحديد المواقع فقط، لا للمقارنة:")
    compare_table(["الاسم", "ما هو", "موقعه", "هل تحتاجه في المقرر؟"],
                  [("JAX", "مكتبة تحويلات دوال (grad, jit, vmap) على NumPy-like مع XLA", "إطار بحثي بجانب TensorFlow/PyTorch؛ خلفية ممكنة لـ Keras 3", "لا"), ("scikit-learn", "ML تقليدي: تقسيم، تحجيم، مقاييس، خطوط أساس", "مساعد لأي إطار", "نعم كمساعد (train_test_split، المقاييس)"),
                   ("PyTorch Lightning / Keras-like wrappers", "طبقة عليا فوق PyTorch تخفي الحلقة (fit)", "فوق PyTorch", "لا؛ لكن ستفهمها فورًا: هي fit لـ PyTorch"), ("Hugging Face Transformers", "نماذج لغة/رؤية مدرّبة مسبقًا بواجهة واحدة لـ PyTorch (وTF/JAX)", "فوق الإطارين", "ربما في المشاريع المتقدمة"),
                   ("ONNX", "صيغة تبادل للنماذج بين الأطر ومحركات الاستدلال", "بين الأطر", "لا"), ("TensorBoard / Weights & Biases", "مراقبة وتتبع تجارب", "بجانب أي إطار", "TensorBoard نعم (درسه)"), ("Jupyter / Colab", "بيئات تشغيل", "ليست أطرًا", "نعم للتشغيل")],
                  ["ltr", "rtl", "rtl", "rtl"])
    quiz("fw.choice", [
        Q("العامل الذي يتقدم على كل المعايير:", ["السرعة", "متطلب المقرر/الفريق", "شهرة الإطار"], 1, "الالتزام."),
        Q("مسألة غير قياسية بخسارة مخصصة تميل إلى…", ["Keras فقط", "PyTorch (أو حلقة مخصصة في TF)", "لا يمكن"], 1, "الصراحة."),
        Q("PyTorch Lightning هو…", ["إطار مستقل", "طبقة عليا تخفي حلقة PyTorch (مثل fit)", "بديل لـ NumPy"], 1, "تجريد أعلى."),
        Q("«الإطار X أفضل مطلقًا»:", ["صحيح لـ PyTorch", "صحيح لـ Keras", "غير صحيح: يعتمد على السياق"], 2, "بلا تحيز."),
    ])
    takeaway("لا أفضل مطلقًا. متطلب المقرر أولًا، ثم: قياسي/كود أقل → Keras؛ غير قياسي/رؤية كل خطوة/كود مرجعي → PyTorch. كلاهما نفس الرياضيات، والترجمة بينهما مهارة تملكها الآن.")
    lesson_footer(LESSON, ["ثمانية معايير بلا تحيز.", "أداة قرار استرشادية.", "خريطة ما حول الإطارين."])
