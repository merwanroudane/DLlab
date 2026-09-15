import streamlit as st

from components.callouts import common_mistake, practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.registry import get_registry
from core.routing import go
from core.rtl import term_table

LESSON = Lesson(
    id="foundations.start.terminology_orientation",
    title_ar="توجيه في المصطلحات",
    title_en="Terminology Orientation",
    module="foundations.start",
    order=7,
    prerequisites=["foundations.start.model_training_prediction"],
    objectives_ar=[
        "معرفة نظام المصطلحات الموحد في المنصة (ترجمة واحدة لكل مصطلح).",
        "التعرف المبكر على أكثر 20 مصطلحًا ستقابلها في الوحدات الأولى.",
        "استخدام القاموس المركزي والبحث ثنائي اللغة.",
    ],
    terms=["dataset", "observation", "feature", "target", "shape", "tensor", "epoch", "batch", "batch_size", "iteration"],
    difficulty="beginner",
    summary_ar="خريطة مبكرة لأهم المصطلحات وترجمتها الموحدة وكيفية استخدام القاموس.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("قاعدة الترجمة الواحدة", "One term, one translation")
    st.markdown(
        """
كل مصطلح إنجليزي في هذه المنصة له **ترجمة عربية واحدة معتمدة** تُستخدم في كل الدروس، ويُذكر معه
المصطلح الإنجليزي بين علامات الكود ليبقى بالاتجاه الصحيح. مثال: نقول دائمًا **الحقبة** `Epoch`،
ولا نقول أحيانًا «الدورة» وأحيانًا «الجولة».

عند أول ظهور لمصطلح مهم ستجد إما درسًا كاملًا له أو رابطًا إلى الدرس الذي يشرحه — **لا نكتفي بتلميح من سطر واحد**.
"""
    )
    h2("المصطلحات العشرون الأولى", "First twenty terms")
    term_table([
        ("البيانات", "Data", "قياسات أو تسجيلات عن العالم، خام أو منظمة", "df"),
        ("مجموعة البيانات", "Dataset", "مجموعة منظمة من الملاحظات بنفس البنية", "X, y"),
        ("الملاحظة", "Observation / Sample / Instance", "صف واحد: كيان واحد وقيمه", "X[0]"),
        ("الخاصية", "Feature", "عمود مدخل يصف الملاحظة", "X[:, 2]"),
        ("الهدف", "Target / Label", "القيمة التي نريد التنبؤ بها", "y"),
        ("الشكل", "Shape", "عدد العناصر على كل محور", "(120, 4)"),
        ("الرتبة", "Rank", "عدد المحاور في الموتر", "X.ndim"),
        ("الموتر", "Tensor", "مصفوفة متعددة المحاور: التمثيل الأساسي للبيانات في الشبكات", "tf.Tensor"),
        ("النموذج", "Model", "دالة بمعلمات قابلة للتعديل", "model"),
        ("المعلمة", "Parameter", "قيمة يتعلمها النموذج (وزن، انحياز)", "w, b"),
        ("المعلمة الفائقة", "Hyperparameter", "قيمة يحددها الباحث قبل التدريب", "learning_rate"),
        ("الخسارة", "Loss", "رقم يقيس سوء التنبؤ؛ التدريب يقلّله", "loss=\"mse\""),
        ("التدرج", "Gradient", "اتجاه ومعدل تغير الخسارة بالنسبة للمعلمات", "∇L"),
        ("معدل التعلم", "Learning Rate", "حجم خطوة التحديث", "lr=0.01"),
        ("الحقبة", "Epoch", "مرور كامل على بيانات التدريب", "epochs=10"),
        ("الدفعة", "Batch", "جزء من بيانات التدريب يُمرَّر معًا", "Batch 1"),
        ("حجم الدفعة", "Batch Size", "عدد الملاحظات في الدفعة", "batch_size=32"),
        ("التكرار / الخطوة", "Iteration / Step", "تحديث واحد للمعلمات على دفعة واحدة", "step 4"),
        ("التنبؤ", "Prediction", "مخرج النموذج لمدخل معين", "model.predict(x)"),
        ("التعميم", "Generalization", "أداء النموذج على بيانات لم يرها", "val_loss"),
    ])
    practical_note("لا تحفظ الجدول. عد إليه من القاموس كلما احتجت؛ كل صف له درس كامل في مكانه من المسار.")
    common_mistake(
        "«Sample» تعني في التعلم الآلي **ملاحظة واحدة**، بينما في الإحصاء تعني **عينة كاملة**. "
        "في المنصة نستخدم «ملاحظة» للصف الواحد و«عينة» لمجموعة الصفوف عند الحديث الإحصائي، ونوضح السياق دائمًا."
    )
    h2("جرّب البحث", "Try the search")
    q = st.text_input("ابحث عن مصطلح بالعربية أو الإنجليزية", key="term_orient_search", placeholder="مثال: epoch أو الحقبة")
    if q:
        reg = get_registry()
        hits = reg.search(q, limit=8)
        if not hits:
            st.info("لا نتائج. جرّب كلمة أقصر.")
        for kind, hid, label in hits:
            icon = {"term": ":material/dictionary:", "lesson": ":material/menu_book:", "lab": ":material/science:"}[kind]
            route = hid if kind != "term" else "glossary"
            st.button(label, key=f"to_{kind}_{hid}", type="tertiary", icon=icon, on_click=go, args=(route,))
    quiz(
        "start.terminology",
        [
            Q("ما الترجمة المعتمدة لـ `Epoch` في المنصة؟", ["الدورة", "الحقبة", "الجولة"], 1, "الحقبة = مرور كامل على بيانات التدريب."),
            Q("`X.shape == (120, 4)`: كم ملاحظة وكم خاصية؟", ["4 ملاحظات و120 خاصية", "120 ملاحظة و4 خصائص", "480 ملاحظة"], 1,
              "المحور الأول ملاحظات، الثاني خصائص.", kind="shape"),
            Q("أيها معلمة فائقة؟", ["الوزن w", "الانحياز b", "معدل التعلم"], 2, "يحددها الباحث ولا يتعلمها النموذج."),
        ],
    )
    takeaway("مصطلح واحد = ترجمة واحدة. القاموس المركزي هو مرجعك، وكل مصطلح له درس.")
    lesson_footer(LESSON, [
        "ترجمة موحدة لكل مصطلح عبر المنصة.",
        "المصطلح الإنجليزي يظهر دائمًا بالاتجاه الصحيح بين علامات الكود.",
        "القاموس والبحث ثنائيا اللغة.",
    ])
