import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.training_loop.validation_in_loop",
    title_ar="التحقق داخل الحلقة ومتى نتوقف",
    title_en="Validation Inside the Loop & When to Stop",
    module="foundations.training_loop",
    order=2,
    prerequisites=["foundations.training_loop.the_loop", "foundations.ml.baseline_generalization"],
    objectives_ar=["ماذا يُحسب على التحقق (خسارة ومقياس بلا تدرج) ولماذا في وضع «تقييم».", "معايير التوقف: عدد حقب ثابت، إيقاف مبكر بصبر، استعادة أفضل الأوزان.", "قراءة السجل (log) سطرًا سطرًا."],
    terms=["epoch", "loss"],
    difficulty="intermediate",
    summary_ar="كل حقبة: قيّم على التحقق بلا تدرج وفي وضع eval؛ توقف عندما لا يتحسن val_loss لـ patience حقب وأعد أفضل الأوزان.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا يحدث في خطوة التحقق؟", "What happens in validation?")
    definition("في نهاية كل حقبة نمرر مجموعة التحقق كاملة عبر النموذج **بلا انتشار خلفي ولا تحديث** ونحسب الخسارة والمقياس. في الأطر يوضع النموذج في **وضع التقييم** (`model.eval()` / `training=False`) لأن Dropout وBatchNorm تتصرف بصورة مختلفة أثناء التدريب.")
    why("لماذا بلا تدرج؟ لأن أي تعلم من التحقق يجعله جزءًا من التدريب فيفقد دوره كمقياس للتعميم. ولماذا كل حقبة لا كل دفعة؟ لأن التحقق مكلف نسبيًا ولأن حقبة واحدة وحدة تقدم معقولة.")
    h2("قراءة السجل", "Reading the log")
    st.code("Epoch 7/50\n25/25 ━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3121 - accuracy: 0.8712 - val_loss: 0.3580 - val_accuracy: 0.8467", language="text")
    compare_table(["الجزء", "المعنى"],
                  [("Epoch 7/50", "الحقبة السابعة من خمسين مخططة"), ("25/25", "25 دفعة في الحقبة (n/batch_size)"), ("3ms/step", "زمن الدفعة الواحدة"),
                   ("loss", "متوسط خسارة دفعات التدريب خلال الحقبة (متوسط متحرك؛ المعلمات كانت تتغير أثناءه)"), ("accuracy", "مقياس التدريب بنفس الطريقة"),
                   ("val_loss / val_accuracy", "على مجموعة التحقق كاملة بالمعلمات في **نهاية** الحقبة — لذلك قد يبدو التحقق أفضل من التدريب في الحقب الأولى")],
                  ["code", "rtl"])
    h2("متى نتوقف؟", "When to stop?")
    compare_table(["الاستراتيجية", "English", "الفكرة", "المعلمات", "الخطر"],
                  [("عدد ثابت", "Fixed epochs", "epochs=50", "epochs", "قليل = قصور، كثير = فرط تخصيص"), ("إيقاف مبكر", "Early stopping", "توقف إن لم يتحسن val_loss لـ patience حقب؛ استعد أفضل الأوزان", "monitor, patience, min_delta, restore_best_weights", "صبر قصير يوقف قبل هضبة مؤقتة"),
                   ("حفظ الأفضل", "Model checkpoint", "احفظ الأوزان عند كل تحسن في التحقق", "save_best_only", "لا يوقف؛ يُستخدم مع الأولى")],
                  ["rtl", "ltr", "rtl", "code", "rtl"])
    practical_note("`EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)` مع `epochs` كبير (200): اترك للتحقق أن يقرر الطول. لا تراقب `val_accuracy` للإيقاف في التصنيف غير المتوازن؛ الخسارة أنعم.")
    common_mistake("الإيقاف المبكر بمراقبة خسارة **التدريب**: تنخفض دائمًا تقريبًا فلا يتوقف أبدًا في الوقت الصحيح. راقب التحقق.")
    quiz("loop.val", [
        Q("أثناء التحقق…", ["نحسب التدرج ولا نحدّث", "لا تدرج ولا تحديث، ووضع eval", "نحدّث بمعدل صغير"], 1, "قياس فقط."),
        Q("val_loss أفضل من loss في الحقبة الأولى…", ["خطأ", "طبيعي: loss متوسط أثناء التغير وval في النهاية", "تسريب"], 1, "التوقيت."),
        Q("restore_best_weights=True يعني…", ["الاحتفاظ بآخر أوزان", "العودة إلى أوزان أفضل حقبة تحقق", "حذف الأوزان"], 1, "الأفضل لا الأخير."),
    ])
    takeaway("التحقق كل حقبة بلا تدرج وفي وضع eval. توقف بالإيقاف المبكر على val_loss مع استعادة الأفضل. اقرأ السجل بوعي بالتوقيت.")
    lesson_footer(LESSON, ["eval mode للتحقق.", "السجل: loss متوسط متحرك، val في النهاية.", "EarlyStopping على التحقق."])
