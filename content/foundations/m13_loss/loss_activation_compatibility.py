import streamlit as st

from components.callouts import takeaway
from components.comparison import compare_table
from components.diagnostics import Problem, problem_card
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.loss.loss_activation_compatibility",
    title_ar="توافق الخسارة مع تنشيط الإخراج وشكل الهدف",
    title_en="Loss / Output-activation / Target-shape Compatibility",
    module="foundations.loss",
    order=5,
    prerequisites=["foundations.loss.cce_sparse", "foundations.loss.mse_mae_huber"],
    objectives_ar=["امتلاك جدول التوافق الكامل والرجوع إليه قبل كل `compile`.", "تشخيص عدم التوافق بإطار المشكلة الكامل."],
    terms=["loss", "softmax"],
    labs=["labs.loss_lab"],
    difficulty="intermediate",
    summary_ar="جدول واحد: المهمة ← وحدات الإخراج ← التنشيط ← الخسارة ← شكل y. أي خلط = تعلم خاطئ بصمت.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الجدول النهائي", "The definitive table")
    compare_table(["المهمة", "وحدات الإخراج", "التنشيط", "الخسارة (Keras)", "شكل y", "dtype y", "PyTorch"],
                  [("انحدار", "1", "خطي", "mse / mae / huber", "(n,) أو (n,1)", "float32", "MSELoss / L1Loss / HuberLoss"),
                   ("ثنائي", "1", "sigmoid (أو from_logits)", "binary_crossentropy", "(n,) أو (n,1) من 0/1", "float32", "BCEWithLogitsLoss"),
                   ("متعدد الفئات", "K", "softmax (أو from_logits)", "sparse_categorical_crossentropy", "(n,) أرقام 0..K−1", "int", "CrossEntropyLoss (logits + أرقام)"),
                   ("متعدد الفئات (one-hot)", "K", "softmax", "categorical_crossentropy", "(n, K)", "float32", "CrossEntropyLoss بأهداف احتمالية"),
                   ("متعدد التسميات", "K", "sigmoid", "binary_crossentropy", "(n, K) من 0/1", "float32", "BCEWithLogitsLoss")],
                  ["rtl", "num", "ltr", "code", "code", "code", "code"])
    problem_card(Problem(
        key="loss_mismatch", name_ar="عدم توافق الخسارة مع الإخراج", name_en="Loss / activation / target mismatch",
        description_ar="الخسارة تتوقع نوعًا من المخرج (احتمال، logits، توزيع) أو شكلًا من الهدف غير الذي يُعطى لها. غالبًا بلا رسالة خطأ.",
        symptoms_ar=["الخسارة لا تنخفض تحت قيمة معينة (مثلًا 0.5 مع Sigmoid مزدوجة).", "دقة عالية وخسارة غريبة، أو العكس.", "خسارة سالبة أو `nan`.", "خطأ شكل عند `fit`."],
        sees_ar=["`Shapes (32, 1) and (32, 10) are incompatible`.", "خسارة تبدأ بعيدًا عن ln K / ln 2.", "`loss: -3.2` (سالبة) في السجل."],
        possible_causes_ar=["Softmax مع BCE أو Sigmoid مع CCE.", "أرقام فئات مع CCE (تتوقع one-hot).", "تنشيط في الطبقة + `from_logits=True` (مزدوج).", "MSE للتصنيف.", "هدف خارج المدى (فئات من 1، أو احتمالات كأهداف مع Sparse).", "ReLU في إخراج انحدار بقيم سالبة."],
        root_causes_ar=["اختيار كل عنصر (تنشيط، خسارة، شكل y) على حدة بدل اختيار صف كامل من الجدول."],
        diagnosis_ar=["اطبع `y.shape`, `y.dtype`, `np.unique(y)[:10]`.", "اطبع `model.output_shape` وتنشيط الطبقة الأخيرة.", "قارن الخسارة الابتدائية بالمرجع (ln K، ln 2، تباين الهدف).", "شغّل حقبة واحدة على 32 ملاحظة: يجب أن تنخفض الخسارة بوضوح."],
        evidence_ar=["تصحيح صف الجدول يعيد الخسارة الابتدائية إلى مرجعها ويجعلها تنخفض."],
        fixes_ar=["اختر صفًا كاملًا من الجدول.", "حوّل الهدف إلى الشكل/النوع المطلوب.", "إما تنشيط في الطبقة أو `from_logits=True` — لا الاثنين."],
        misdiagnosis_ar=["«معدل التعلم صغير».", "«البيانات صعبة»."],
        related_ar=["Softmax", "BCE/CCE", "أشكال الهدف"],
        checklist_ar=["y.shape وdtype مطابقان للصف؟", "التنشيط الأخير مطابق؟", "الخسارة مطابقة؟", "from_logits ليس مزدوجًا؟", "الخسارة الابتدائية قرب المرجع؟"],
        challenge=[Q("مصنف 5 فئات، `Dense(5, 'softmax')`، `loss='binary_crossentropy'`، الخسارة تبدأ 0.5 ولا تنخفض كثيرًا. المشكلة؟", ["معدل التعلم", "Softmax مع BCE: صف مختلط", "قلة البيانات"], 1, "استخدم sparse_categorical_crossentropy.", kind="scenario"),
                   Q("انحدار على تغيرات أسعار (موجبة وسالبة) بـ `Dense(1, 'relu')` وMSE؛ التنبؤات كلها ≥ 0. المشكلة؟", ["MSE خاطئة", "ReLU في الإخراج", "الهدف غير محجّم"], 1, "خطي.", kind="scenario")],
    ))
    st.button("افتح معمل دوال الخسارة", icon=":material/science:", type="primary", on_click=go, args=("labs.loss_lab",), key="compat_lab_btn")
    quiz("loss.compat", [
        Q("قبل `compile` تطبع…", ["فقط X.shape", "y.shape وdtype وقيمها الفريدة", "لا شيء"], 1, "الصف يعتمد على y."),
        Q("Sigmoid في الطبقة + from_logits=True…", ["صحيح", "مزدوج: خطأ صامت", "أسرع"], 1, "اختر واحدًا."),
    ])
    takeaway("اختر صفًا كاملًا من جدول التوافق؛ تحقق من y.shape/dtype والخسارة الابتدائية مقابل مرجعها. الخلط لا يعطي خطأ بل نموذجًا يتعلم الشيء الخطأ.")
    lesson_footer(LESSON, ["جدول التوافق الكامل.", "مشكلة مُشخَّصة بالكامل.", "المرجع الابتدائي أداة تحقق مجانية."])
