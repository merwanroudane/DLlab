import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.ml.task_types",
    title_ar="أنواع المهام: انحدار، تصنيف ثنائي، متعدد الفئات، متعدد التسميات",
    title_en="Task Types: Regression, Binary, Multiclass & Multilabel Classification",
    module="foundations.ml",
    order=2,
    prerequisites=["foundations.ml.paradigms", "foundations.data.target"],
    objectives_ar=[
        "تحديد نوع المهمة من نوع الهدف، واستنتاج طبقة الإخراج والخسارة والمقياس المناسبين.",
        "التمييز بين «فئة واحدة من k» و«عدة تسميات معًا».",
    ],
    terms=["target", "loss", "softmax", "cross_entropy"],
    difficulty="beginner",
    summary_ar="نوع الهدف ← المهمة ← طبقة الإخراج + الخسارة + المقياس. جدول واحد تعود إليه دائمًا.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الجدول المرجعي", "The reference table")
    definition("**الانحدار** `Regression` هدف عددي متصل. **التصنيف الثنائي** فئتان. **متعدد الفئات** `Multiclass` فئة واحدة من $k$. **متعدد التسميات** `Multilabel` أي عدد من $k$ تسميات معًا.")
    compare_table(["المهمة", "الهدف", "شكل y", "طبقة الإخراج", "التنشيط", "الخسارة", "المقاييس"],
                  [("انحدار", "عدد متصل", "(n,) أو (n,1)", "Dense(1)", "خطي (بلا)", "MSE / MAE / Huber", "RMSE, MAE, R²"),
                   ("تصنيف ثنائي", "0/1", "(n,)", "Dense(1)", "Sigmoid", "Binary cross-entropy", "Accuracy, Precision, Recall, F1, AUC"),
                   ("متعدد الفئات", "فئة من k", "(n,) أرقام أو (n,k) one-hot", "Dense(k)", "Softmax", "(Sparse) categorical cross-entropy", "Accuracy, F1-macro, مصفوفة الالتباس"),
                   ("متعدد التسميات", "متجه 0/1 بطول k", "(n,k)", "Dense(k)", "Sigmoid (لكل وحدة)", "Binary cross-entropy", "F1 لكل تسمية, Hamming")],
                  ["rtl", "rtl", "code", "code", "ltr", "ltr", "ltr"])
    practical_note("احفظ الصفوف الأربعة كوحدة واحدة: تغيير المهمة يغيّر الأعمدة الأربعة معًا. أغلب أخطاء «الخسارة لا تنخفض» في التصنيف سببها صف مختلط: Softmax مع BCE، أو Sigmoid مع CCE.")
    h2("أمثلة للتصنيف", "Classify these")
    compare_table(["السؤال", "المهمة", "لماذا"],
                  [("كم سيبلغ سعر السهم غدًا؟", "انحدار", "عدد متصل"), ("هل سيرتفع السهم غدًا؟", "تصنيف ثنائي", "نعم/لا"),
                   ("إلى أي قطاع تنتمي الشركة؟", "متعدد الفئات", "قطاع واحد من عدة"), ("ما مواضيع هذا التقرير (قد تكون عدة)؟", "متعدد التسميات", "أكثر من تسمية معًا"),
                   ("كم عدد الوحدات المباعة؟", "انحدار (عدّ)", "عدد صحيح كبير ~ متصل"), ("تقييم الخدمة من 1 إلى 5", "متعدد الفئات (أو ترتيبي)", "فئات مرتبة")],
                  ["rtl", "rtl", "rtl"])
    common_mistake("معاملة هدف ترتيبي (1..5 نجوم) كانحدار خالص أو كفئات مستقلة تمامًا: كلاهما يفقد معلومة. في المقرر نعامله كمتعدد الفئات مع الانتباه إلى أن الأخطاء المتجاورة أخف.")
    quiz("ml.tasks", [
        Q("هدف بقيم مثل 0.0، 1250.5، 300.2…", ["انحدار", "تصنيف ثنائي", "متعدد التسميات"], 0, "متصل."),
        Q("Softmax + Binary cross-entropy…", ["زوج صحيح", "خطأ: Softmax مع CCE، Sigmoid مع BCE", "يعتمد على البيانات"], 1, "الصف المختلط."),
        Q("كل مقال قد يحمل عدة مواضيع…", ["متعدد الفئات", "متعدد التسميات", "انحدار"], 1, "عدة تسميات معًا."),
        Q("طبقة الإخراج لتصنيف 7 فئات…", ["Dense(1) Sigmoid", "Dense(7) Softmax", "Dense(7) Sigmoid"], 1, "احتمالات مجموعها 1."),
    ])
    takeaway("نوع الهدف يقرر الصف كله: طبقة الإخراج + التنشيط + الخسارة + المقياس. لا تخلط الصفوف.")
    lesson_footer(LESSON, ["أربع مهام، أربعة صفوف.", "Sigmoid↔BCE، Softmax↔CCE، خطي↔MSE.", "الترتيبي حالة خاصة."])
