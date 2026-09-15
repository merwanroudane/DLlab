import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, research_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.fw import cnn_shapes_run

LESSON = Lesson(
    id="course.w09.overfit_augment",
    title_ar="فرط التخصيص في الصور وزيادة البيانات",
    title_en="Overfitting on Images & Data Augmentation",
    module="course.w09",
    order=3,
    prerequisites=["course.w09.build_train", "foundations.regularization.augmentation_simplification", "foundations.regularization.dropout"],
    objectives_ar=["إحداث فرط تخصيص بتقليل بيانات التدريب إلى 60 صورة وقراءته من المنحنيات.", "علاجه بزيادة البيانات (إزاحة/قلب) وبـ Dropout، ومقارنة الثلاثة على الاختبار.", "قواعد اختيار تحويلات الزيادة المناسبة للمسألة."],
    terms=["hyperparameter"],
    labs=["labs.overfitting_lab", "labs.dropout_lab"],
    difficulty="intermediate",
    summary_ar="بيانات قليلة + CNN = حفظ: train 1.0 وval أقل بكثير. زيادة البيانات (layers.RandomTranslation/RandomFlip داخل النموذج، على التدريب فقط) تخلق تنوعًا حقيقيًا وترفع الاختبار. Dropout يساعد أقل هنا. التحويلات يجب أن تحفظ الفئة.",
)

CODE = '''# زيادة البيانات كطبقات داخل النموذج: تعمل في التدريب فقط وتُعطَّل تلقائيًا في evaluate/predict
model = keras.Sequential([
    layers.Input(shape=(16, 16, 1)),
    layers.RandomTranslation(0.15, 0.15, fill_mode="constant"),   # إزاحة عشوائية حتى 15%
    layers.RandomFlip("horizontal_and_vertical"),                 # قلب: يحفظ الفئة هنا (شريط/صليب)
    layers.Conv2D(8, 3, padding="same", activation="relu"), layers.MaxPooling2D(2),
    layers.Conv2D(16, 3, padding="same", activation="relu"), layers.MaxPooling2D(2),
    layers.Flatten(), layers.Dropout(0.0), layers.Dense(32, activation="relu"), layers.Dense(3, activation="softmax"),
])
# التدريب على 60 صورة فقط لـ 20 حقبة (الزيادة تحتاج حقبًا أكثر لأن كل حقبة ترى صورًا مختلفة)'''


def render() -> None:
    lesson_header(LESSON)
    why("الشبكة التي بلغت دقة 1.0 في الدرس السابق دُرّبت على 300 صورة. في الواقع الاقتصادي/الإداري قد تملك 60 صورة موسومة. ماذا يحدث؟ وما العلاج الذي لا يحتاج جمع بيانات جديدة؟")
    h2("التجربة: 60 صورة، ثلاثة إعدادات", "The experiment: 60 images, three settings")
    epochs = st.slider("epochs", 10, 40, 20, 5, key="w09_of_ep")
    runs = {"بلا علاج": cnn_shapes_run(epochs=int(epochs), augment=False, n_train=60), "زيادة بيانات": cnn_shapes_run(epochs=int(epochs), augment=True, n_train=60), "Dropout 0.5": cnn_shapes_run(epochs=int(epochs), augment=False, n_train=60, dropout=0.5)}
    fig = go.Figure(); rows = []
    for (name, r), color in zip(runs.items(), ("#C8473A", "#1F7A78", "#7C5CBF")):
        e = np.arange(1, len(r["history"]["loss"]) + 1)
        fig.add_trace(go.Scatter(x=e, y=r["history"]["loss"], name=f"{name} — loss", line=dict(color=color, width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name=f"{name} — val_loss", line=dict(color=color, width=3)))
        rows.append((name, f"{r['train_acc']:.3f}", f"{r['val_acc']:.3f}", f"{r['test_acc']:.3f}", f"{r['train_acc'] - r['test_acc']:+.3f}"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="loss", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="w09_of_fig")
    table(["الإعداد", "train acc", "val acc", "test acc", "الفجوة train−test"], rows, ["rtl", "num", "num", "num", "num"])
    with st.expander("كيف أقرأ؟", expanded=True, icon=":material/visibility:"):
        st.markdown("""
- **بلا علاج**: loss يهبط نحو 0 (حفظ 60 صورة) بينما val_loss يتوقف ثم يصعد — الفجوة الكبيرة هي فرط التخصيص (الأسس 19).
- **زيادة البيانات**: loss التدريب أعلى (الصور تتغير كل حقبة فلا يمكن حفظها) لكن val/test أعلى — النموذج تعلّم «الشكل» لا «الصورة».
- **Dropout**: يضيّق الفجوة قليلًا لكنه لا يعوّض نقص التنوع؛ في الصور، الزيادة أقوى.
- غيّر الحقب: الزيادة تحتاج حقبًا أكثر لتتفوق (كل حقبة ترى بيانات «جديدة»).
""")
    h2("زيادة البيانات في Keras", "Augmentation in Keras")
    st.code(CODE, language="python")
    compare_table(["التحويل", "الطبقة", "يحفظ الفئة في…", "يكسرها في…"],
                  [("إزاحة", "`RandomTranslation(0.1, 0.1)`", "معظم الصور", "صور حيث الموضع هو الهدف"), ("قلب أفقي", "`RandomFlip('horizontal')`", "أشياء متماثلة (وجوه، منتجات)", "أرقام/حروف/نصوص، إشارات مرور"), ("قلب عمودي", "`RandomFlip('vertical')`", "أنماط مجردة، صور جوية", "معظم الصور الطبيعية"),
                   ("تدوير", "`RandomRotation(0.1)`", "زوايا صغيرة", "6 ↔ 9؛ رسوم بيانية"), ("تكبير/قص", "`RandomZoom(0.1)`", "الأشياء بأحجام مختلفة", "عندما يهم الحجم المطلق"), ("سطوع/تباين", "`RandomBrightness`/`RandomContrast`", "إضاءة متغيرة", "عندما اللون هو الهدف")],
                  ["rtl", "code", "rtl", "rtl"])
    intuition("سؤال واحد قبل كل تحويل: «لو أريتُ الصورة المحوَّلة لخبير، هل سيعطيها **نفس الوسم**؟» إن كان الجواب لا فالتحويل يعلّم النموذج أخطاء. قلب رسم بياني لسعر عموديًا يحوّل صعودًا إلى هبوط — كارثة لمسألة اتجاه السوق.")
    research_note("الزيادة كطبقات داخل النموذج (Keras 3) تعمل على GPU وتُعطَّل تلقائيًا في الاستدلال (`training=False`). البديل القديم `ImageDataGenerator` مهجور. للصور الحقيقية المنقولة من نماذج مدرَّبة مسبقًا (transfer learning) الزيادة تبقى مفيدة لكن الأثر الأكبر من الأوزان المسبقة — خارج نطاق هذا الأسبوع.")
    with st.container(horizontal=True):
        st.button("معمل فرط التخصيص", icon=":material/science:", on_click=goto, args=("labs.overfitting_lab",), key="w09_lab_of")
        st.button("الأسس 20 — الزيادة والتبسيط", icon=":material/menu_book:", on_click=goto, args=("foundations.regularization.augmentation_simplification",), key="w09_go_aug")
    common_mistake("تطبيق الزيادة على مجموعة التحقق/الاختبار (بمعالجة مسبقة يدوية على كل البيانات): تقيس على صور مشوّهة وتُبلّغ رقمًا خاطئًا. الطبقات داخل النموذج تحل ذلك تلقائيًا؛ مع خطوط tf.data طبّق `map(augment)` على التدريب فقط.")
    quiz("w09.aug", [
        Q("train 1.0 وtest 0.65 على 60 صورة:", ["قصور", "فرط تخصيص", "تسريب"], 1, ""),
        Q("لماذا loss التدريب أعلى مع الزيادة؟", ["النموذج أسوأ", "الصور تتغير كل حقبة فلا تُحفظ", "η أصغر"], 1, ""),
        Q("قلب عمودي لصورة رسم بياني لسعر:", ["زيادة مناسبة", "يكسر الفئة (صعود ↔ هبوط)", "لا أثر"], 1, ""),
        Q("طبقات الزيادة في Keras أثناء predict…", ["تعمل", "معطّلة تلقائيًا", "تحتاج إزالة"], 1, ""),
    ])
    takeaway("بيانات قليلة + CNN = حفظ. الزيادة تخلق تنوعًا حقيقيًا (على التدريب فقط، كطبقات) وتحتاج حقبًا أكثر؛ Dropout مساعد. اختر تحويلات تحفظ الفئة.")
    lesson_footer(LESSON, ["تجربة ثلاثية حية.", "طبقات الزيادة وجدول التحويلات.", "قاعدة «نفس الوسم»."])
