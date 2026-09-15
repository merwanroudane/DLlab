import streamlit as st

from components.callouts import intuition, practical_note, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from content.foundations.m23_ready._scoring import scored_quiz
from core.models import Lesson
from core.progress import module_progress
from core.registry import get_registry
from core.routing import go

LESSON = Lesson(
    id="foundations.ready.prerequisite_check",
    title_ar="فحص المتطلبات والاختبار التشخيصي",
    title_en="Prerequisite Check & Diagnostic Quiz",
    module="foundations.ready",
    order=2,
    prerequisites=["foundations.ready.concept_map"],
    objectives_ar=["تقييم ذاتي صريح لكل وحدة: «أستطيع أن…» مع رابط العودة.", "اختبار تشخيصي من 20 سؤالًا يغطي الوحدات 1–21 ويُسجَّل لحالة الجاهزية.", "قراءة النتيجة كخريطة فجوات لا كحكم."],
    terms=[],
    difficulty="beginner",
    summary_ar="21 عبارة «أستطيع» + 20 سؤالًا تشخيصيًا. النتيجة تُحفظ في الجلسة وتُستخدم في صفحة حالة الجاهزية مع التحديات.",
)

CAN = [
    ("foundations.data", "أشرح الفرق بين الملاحظة والخاصية والهدف، وأقرأ شكل (batch, features) وأحدد الرتبة."),
    ("foundations.python", "أنشئ ndarray وأستخدم البث والفهرسة وأثبّت البذرة."),
    ("foundations.math", "أقرأ Σ والأس واللوغاريتم وأفرّق الدالة الخطية عن غير الخطية."),
    ("foundations.linalg", "أحسب حاصل الضرب النقطي وضرب المصفوفات وأتحقق من توافق الأشكال."),
    ("foundations.calculus", "أشتق دالة بسيطة، أفسّر التدرج كاتجاه، وأطبّق قاعدة السلسلة."),
    ("foundations.prob", "أحسب المتوسط والتباين وأفسّر الاحتمال والانتروبيا المتقاطعة."),
    ("foundations.ml", "أميّز الانحدار عن التصنيف، والمعلمة عن المعلمة الفائقة، وأبني خط أساس."),
    ("foundations.prep", "أقسّم قبل التحجيم، أرمّز الفئوي، وأكشف التسريب."),
    ("foundations.neuron", "أكتب z = wx + b ثم a = f(z) وأفسّر الوزن والانحياز."),
    ("foundations.architecture", "أعدّ معلمات شبكة كثيفة من أشكال طبقاتها."),
    ("foundations.activations", "أختار التنشيط المناسب للمخرج وأعرف مشكلة التشبّع وReLU الميت."),
    ("foundations.forward", "أتتبع دفعة عبر الطبقات بأشكالها وأميّز logits عن الاحتمال."),
    ("foundations.loss", "أطابق الخسارة مع التنشيط وترميز الهدف (MSE/BCE/CCE/sparse)."),
    ("foundations.backprop", "أشرح δ وقاعدة السلسلة إلى الخلف وأتحقق عدديًا من تدرج."),
    ("foundations.optim", "أشرح معدل التعلم والزخم وAdam وأثر η الكبير/الصغير."),
    ("foundations.training_loop", "أكتب حلقة تدريب بخطواتها الخمس والتحقق في نهاية الحقبة."),
    ("foundations.batch_epoch", "أحسب عدد التحديثات من n وbatch_size وepochs."),
    ("foundations.eval", "أقرأ مصفوفة التباس وأحسب الصحة والاستدعاء وأختار العتبة."),
    ("foundations.generalization", "أشخّص القصور وفرط التخصيص من منحنيي التدريب/التحقق."),
    ("foundations.regularization", "أستخدم L2 وDropout والإيقاف المبكر وأعرف سلوك BN في الوضعين."),
    ("foundations.frameworks", "أقرأ summary وسجل fit، وأكتب حلقة PyTorch، وأترجم بين الإطارين."),
]

DIAG = [
    Q("مصفوفة بيانات بشكل (500, 12): ماذا يعني الرقمان؟", ["500 خاصية × 12 ملاحظة", "500 ملاحظة × 12 خاصية", "500 دفعة"], 1, "وحدة 1."),
    Q("`(3,) + (2, 3)` في NumPy…", ["خطأ", "بث إلى (2, 3)", "(5, 3)"], 1, "وحدة 2."),
    Q("log(1) =", ["1", "0", "e"], 1, "وحدة 3."),
    Q("(4, 3) @ (3, 5) →", ["(4, 5)", "(3, 3)", "خطأ"], 0, "وحدة 4."),
    Q("d/dx (x²·3) عند x = 2 =", ["6", "12", "4"], 1, "وحدة 5."),
    Q("الانتروبيا المتقاطعة −log p̂(y) تكون صغيرة عندما…", ["p̂(y) قريب من 0", "p̂(y) قريب من 1", "دائمًا"], 1, "وحدة 6."),
    Q("معدل التعلم هو…", ["معلمة تُتعلَّم", "معلمة فائقة تُضبط", "مقياس"], 1, "وحدة 7."),
    Q("إحصاءات التحجيم تُحسب على…", ["كل البيانات", "التدريب فقط بعد التقسيم", "الاختبار"], 1, "وحدة 8."),
    Q("خلية بلا تنشيط تعطي…", ["دالة خطية", "دالة غير خطية", "احتمالًا"], 0, "وحدة 9."),
    Q("Dense(8) على مدخل بـ 5 خصائص: المعلمات", ["40", "48", "13"], 1, "وحدة 10."),
    Q("softmax في المخرج يناسب…", ["الانحدار", "التصنيف المتعدد الفئات المتنافي", "الثنائي فقط"], 1, "وحدة 11."),
    Q("logit = 0 يقابل احتمال sigmoid =", ["0", "0.5", "1"], 1, "وحدة 12."),
    Q("هدف أعداد صحيحة 0..K−1 مع softmax: الخسارة", ["mse", "categorical_crossentropy", "sparse_categorical_crossentropy"], 2, "وحدة 13."),
    Q("الانتشار الخلفي…", ["يحدّث الأوزان", "يحسب التدرجات فقط", "يحسب الخسارة"], 1, "وحدة 14."),
    Q("η كبير جدًا يسبب…", ["تقاربًا بطيئًا", "تذبذبًا أو انفجارًا", "لا شيء"], 1, "وحدة 15."),
    Q("n = 1000, batch_size = 50, epochs = 10: التحديثات", ["200", "20", "500"], 0, "وحدة 17."),
    Q("val_loss يرتفع بينما loss ينخفض:", ["قصور", "فرط تخصيص", "تسريب"], 1, "وحدة 19."),
    Q("Dropout في الاستدلال…", ["فعّال", "معطّل", "نصف فعّال"], 1, "وحدة 20."),
    Q("`(None, 16)` في summary:", ["16 ملاحظة", "أي دفعة × 16 وحدة", "16 طبقة"], 1, "وحدة 21."),
    Q("في PyTorch، قبل `loss.backward()` في كل دفعة…", ["`optimizer.step()`", "`optimizer.zero_grad()`", "`model.eval()`"], 1, "وحدة 21."),
]


def render() -> None:
    lesson_header(LESSON)
    why("الاختبار الذاتي بلا مواجهة يُنتج ثقة زائفة. هنا خطوتان: أولًا تقول «أستطيع» عبارةً عبارة، ثم تختبر ذلك بأسئلة تُصحَّح. الفارق بين الاثنين هو خريطة العودة.")
    h2("1) فحص المتطلبات: أستطيع أن…", "1) Prerequisite check: I can…")
    reg = get_registry()
    st.session_state.setdefault("ready_can", {})
    can = st.session_state["ready_can"]
    checked = 0
    for mid, text in CAN:
        p = module_progress(mid)
        c1, c2 = st.columns([5, 1.6], vertical_alignment="center")
        with c1:
            can[mid] = st.checkbox(f"{text}", value=can.get(mid, False), key=f"can_{mid}")
        with c2:
            st.button(f"{reg.modules[mid].title_ar.split('—')[0].strip()} ({p.done}/{p.total})", key=f"can_go_{mid}", type="tertiary", on_click=go, args=(mid,))
        checked += int(can[mid])
    st.progress(checked / len(CAN), text=f"{checked} / {len(CAN)} عبارة")
    practical_note("علامة «أستطيع» على وحدة لم تزر دروسها (0/n) تناقض يستحق التوقف: إما تعرفها من قبل — فاختبر ذلك أدناه — أو تظن ذلك.")
    h2("2) الاختبار التشخيصي (20 سؤالًا)", "2) Diagnostic quiz (20 questions)")
    scored_quiz("diagnostic", "ready.diagnostic", DIAG, "foundations", title_ar="الاختبار التشخيصي")
    sc = st.session_state.get("ready_scores", {}).get("diagnostic")
    if sc:
        pct = round(100 * sc["correct"] / sc["total"])
        if pct >= 85:
            st.success(f"{pct}% — أساس متين. انتقل إلى التحديات.", icon="✅")
        elif pct >= 60:
            st.warning(f"{pct}% — جيد مع فجوات: راجع الوحدات المذكورة في تفسير الأسئلة الخاطئة ثم أعد المحاولة.", icon="🟡")
        else:
            st.error(f"{pct}% — عُد إلى الوحدات المذكورة قبل المقرر الرسمي؛ حالة الجاهزية ستوجّهك.", icon="🔁")
    intuition("كل سؤال يحمل رقم وحدته في التفسير. السؤال الخاطئ ليس نقطة ناقصة بل عنوان درس تعود إليه.")
    takeaway("عبارات «أستطيع» ثم اختبار يُصحَّح. النتيجة تُسجَّل في الجلسة وتدخل في حساب الجاهزية مع التحديات.")
    lesson_footer(LESSON, ["21 عبارة بروابط العودة.", "20 سؤالًا تشخيصيًا مُسجَّلًا.", "تفسير النتيجة."])
