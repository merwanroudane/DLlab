import numpy as np
import streamlit as st

from components.callouts import definition, intuition, research_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.data.data_modalities",
    title_ar="أنواع البيانات: جدولية، زمنية، نصية، صور، صوت",
    title_en="Data Modalities: Tabular, Time Series, Text, Image, Audio",
    module="foundations.data",
    order=6,
    prerequisites=["foundations.data.target"],
    objectives_ar=[
        "التمييز بين البيانات المهيكلة وغير المهيكلة.",
        "معرفة الشكل الموتري المعتاد لكل نوع بيانات.",
        "ربط نوع البيانات بالبنية المناسبة (MLP, CNN, RNN) في المقرر.",
    ],
    terms=["tensor", "shape", "sequence", "time_series"],
    difficulty="beginner",
    summary_ar="كل نوع بيانات له شكل موتري نموذجي وبنية شبكية مناسبة.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("مهيكلة أم غير مهيكلة؟", "Structured vs unstructured")
    definition(
        "**البيانات المهيكلة** `Structured` تأتي في جدول بأعمدة ذات معنى محدد (الدخل، العمر). "
        "**البيانات غير المهيكلة** `Unstructured` ليس لها أعمدة جاهزة: نص، صورة، صوت. "
        "التعلم العميق تفوق تاريخيًا في الثانية لأنه يتعلم التمثيل بنفسه."
    )
    intuition("الجدول يشبه استمارة مملوءة؛ الصورة تشبه لوحة. لا يوجد عمود اسمه «قطة» في الصورة — على الشبكة أن تكتشفه.")
    h2("الشكل الموتري لكل نوع", "Tensor shape by modality")
    compare_table(
        ["النوع", "English", "مثال", "الشكل النموذجي", "المعنى", "بنية مناسبة"],
        [
            ("جدولية", "Tabular", "بيانات عملاء", "(n, d)", "n ملاحظات × d خصائص", "MLP"),
            ("سلسلة زمنية", "Time series", "أسعار يومية", "(n, T, f)", "n نوافذ × T خطوة زمنية × f متغير", "RNN / LSTM / GRU"),
            ("تسلسل نصي", "Text sequence", "مراجعات", "(n, T)", "n جملة × T رمز (كلمة) كأعداد", "RNN / Transformer"),
            ("صورة", "Image", "صور رمادية 28×28", "(n, H, W, C)", "n صورة × ارتفاع × عرض × قنوات", "CNN"),
            ("صوت", "Audio", "مقاطع صوتية", "(n, T) أو (n, T, F)", "n مقطع × عينات زمنية (× ترددات)", "CNN / RNN"),
        ],
        ["rtl", "ltr", "rtl", "code", "rtl", "ltr"],
    )
    research_note(
        "ترتيب المحاور اصطلاحي ويختلف بين الأطر: `Keras` تستخدم `(n, H, W, C)` (القنوات أخيرًا) بينما `PyTorch` "
        "يستخدم `(n, C, H, W)` (القنوات أولًا). خطأ الترتيب لا يعطي دائمًا رسالة خطأ — أحيانًا يعطي نموذجًا يتدرب على هراء."
    )
    h2("جرّب الأشكال", "See the shapes")
    kind = st.segmented_control("اختر نوع البيانات", ["جدولية", "سلسلة زمنية", "صورة", "نص"], default="جدولية", key="modality_pick")
    rng = np.random.default_rng(0)
    if kind == "جدولية":
        arr = rng.normal(size=(5, 3)).round(2)
        st.code(f"X.shape = {arr.shape}   # 5 ملاحظات × 3 خصائص\n{arr}", language="text")
    elif kind == "سلسلة زمنية":
        arr = rng.normal(size=(2, 4, 3)).round(2)
        st.code(f"X.shape = {arr.shape}   # 2 نافذة × 4 خطوات زمنية × 3 متغيرات\n{arr}", language="text")
    elif kind == "صورة":
        arr = rng.integers(0, 256, size=(1, 4, 4, 1))
        st.code(f"X.shape = {arr.shape}   # 1 صورة × 4 ارتفاع × 4 عرض × 1 قناة (رمادي)\n{arr[0, :, :, 0]}", language="text")
    else:
        arr = np.array([[12, 45, 7, 0, 0], [3, 99, 45, 12, 8]])
        st.code(f"X.shape = {arr.shape}   # 2 جملة × 5 رموز (الصفر = حشو Padding)\n{arr}", language="text")
    quiz(
        "data.modalities",
        [
            Q("`(64, 30, 5)` لسلسلة زمنية يعني…", ["64 خطوة، 30 نافذة، 5 متغيرات", "64 نافذة، 30 خطوة زمنية، 5 متغيرات", "64 صورة"], 1,
              "(n, T, f).", kind="shape"),
            Q("صور ملونة 32×32 في `Keras`، ما الشكل؟", ["(n, 3, 32, 32)", "(n, 32, 32, 3)", "(n, 1024)"], 1, "القنوات أخيرًا في Keras.", kind="shape"),
            Q("أي البيانات «غير مهيكلة»؟", ["جدول رواتب", "صور فواتير", "أسعار إغلاق يومية"], 1, "الصورة لا أعمدة لها."),
        ],
    )
    takeaway("نوع البيانات ← شكل موتري ← بنية شبكية. تعلّم قراءة الشكل قبل اختيار الطبقة الأولى.")
    lesson_footer(LESSON, [
        "مهيكلة (جدول) مقابل غير مهيكلة (نص، صورة، صوت).",
        "جدولية (n, d)، زمنية (n, T, f)، صور (n, H, W, C) في Keras.",
        "ترتيب المحاور اصطلاحي ويختلف بين Keras وPyTorch.",
    ])
