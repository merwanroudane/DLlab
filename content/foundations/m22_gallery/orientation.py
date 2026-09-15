import streamlit as st

from components.callouts import common_mistake, definition, intuition, practical_note, takeaway, why
from components.comparison import compare_table
from components.gallery import Frame6, gallery_item, notebook_cell
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.gallery.orientation",
    title_ar="توجيه المنصات: الدفتر، الخلية، وحدة التحكم — ولا واجهة رسومية للأطر",
    title_en="Platform Orientation: Notebook, Cell, Console — and No Framework GUI",
    module="foundations.gallery",
    order=1,
    prerequisites=["foundations.frameworks.ecosystem_map"],
    objectives_ar=["فهم أن Keras/TensorFlow/PyTorch مكتبات تعمل داخل بايثون ولا تملك واجهة رسومية؛ ما تراه هو بيئة التشغيل.", "معرفة عناصر خلية الدفتر: المطالبة In/Out، الكود، المخرج النصي، الرسم، والخطأ.", "الأسئلة الستة لقراءة أي لقطة، وقاعدة «إعادة بناء تعليمية»."],
    terms=[],
    difficulty="beginner",
    summary_ar="ما تراه على الشاشة هو Jupyter/Colab/IDE/طرفية — بيئة تشغيل. الإطار يطبع نصًا ويعيد كائنات؛ كل لقطة في هذا المعرض إعادة بناء تعليمية من مخرجات حقيقية، تُقرأ بستة أسئلة.",
)

CELL_CODE = '''import numpy as np
x = np.arange(6).reshape(2, 3)
print("shape:", x.shape, "dtype:", x.dtype)
x * 10'''
CELL_OUT = '''shape: (2, 3) dtype: int64
array([[ 0, 10, 20],
       [30, 40, 50]])'''

COLAB_CODE = '''!nvidia-smi -L        # أمر نظام (علامة ! في Colab/Jupyter)
import tensorflow as tf, torch
print(tf.__version__, torch.__version__)
print("TF GPUs:", tf.config.list_physical_devices("GPU"))
print("torch cuda:", torch.cuda.is_available())'''
COLAB_OUT = '''GPU 0: Tesla T4 (UUID: GPU-3f1e...)
2.20.0 2.14.0+cu126
TF GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
torch cuda: True'''


def render() -> None:
    lesson_header(LESSON)
    definition("**القاعدة العلمية للمعرض**: Keras وTensorFlow وPyTorch **ليست برامج بواجهة رسومية**. هي مكتبات بايثون: تستدعيها في كود، فتعيد كائنات وتطبع نصًا وترسم عبر مكتبات رسم. ما تراه على الشاشة هو **بيئة التشغيل**: دفتر Jupyter، Google Colab، محرر (VS Code/PyCharm)، أو طرفية. لذلك لا يخترع هذا المعرض واجهة وهمية ويُنسبها إلى إطار؛ كل لقطة **إعادة بناء تعليمية** لسطح حقيقي (خلية، وحدة تحكم، لوحة TensorBoard) مبنية من **مخرجات حقيقية** أنتجتها النسخ المثبتة في هذا المشروع، ومُعلَّمة بذلك.")
    why("الباحث الجديد يفتح Colab، يشغّل `model.fit`، ويرى 40 سطرًا من الأرقام فيسأل: أين النموذج؟ أين النتيجة؟ هل هذا خطأ؟ المعرض يعطيه خريطة قراءة لكل سطح قبل أن يقف أمامه وحده.")
    h2("الأسطح الأربعة", "The four surfaces")
    compare_table(["السطح", "ما هو", "ماذا ترى فيه", "ليس"],
                  [("خلية دفتر (Jupyter / Colab)", "محرر تفاعلي: كود في خلية، تنفيذ، مخرج تحت الخلية", "`In [n]:` الكود · `Out[n]:` قيمة آخر تعبير · ما طُبع بـ print · الرسوم · الأخطاء بالأحمر", "إطار عمل"),
                   ("وحدة التحكم / الطرفية", "تشغيل ملف `python train.py` أو REPL `>>>`", "نص فقط: ما يُطبع، وأشرطة التقدم، والأخطاء", "إطار عمل"),
                   ("المحرر (IDE)", "VS Code / PyCharm: ملفات، تصحيح، طرفية مدمجة", "الكود ملفًا؛ المخرج في الطرفية المدمجة", "إطار عمل"),
                   ("TensorBoard", "برنامج منفصل في المتصفح يقرأ سجلات", "منحنيات، رسوم، هيستوغرامات", "TensorFlow نفسه")],
                  ["rtl", "rtl", "rtl", "rtl"])
    h2("تشريح خلية دفتر", "Anatomy of a notebook cell")
    gallery_item("خلية بايثون أساسية في Jupyter", "Python / Jupyter basic code cell", notebook_cell(CELL_CODE, CELL_OUT, n=1),
                 Frame6(where_ar="داخل دفتر Jupyter (أو Colab — نفس التشريح). لا إطار تعلم عميق بعد: NumPy فقط.",
                        what_ar="خلية واحدة: مطالبة `In [1]:` (رقم التنفيذ)، الكود على خلفية رمادية، ثم منطقة المخرج.",
                        code_where_ar="الأسطر الأربعة بجانب `In [1]:`. يُنفَّذ بالضغط Shift+Enter.",
                        output_where_ar="تحت الكود: السطر الأول من `print` (يظهر دائمًا)، ثم `array([...])` هو **قيمة آخر تعبير** في الخلية (`x * 10`) — الدفتر يعرضها تلقائيًا بلا print.",
                        numbers_ar="`(2, 3)` شكل المصفوفة، `int64` نوعها (NumPy)، المصفوفة المطبوعة بصفين وثلاثة أعمدة.",
                        notice_ar="رقم `In [1]` يزداد مع كل تنفيذ (حتى لنفس الخلية): `In [7]` يعني أن الدفتر نُفّذ سبع مرات، وترتيب الأرقام لا يطابق ترتيب الخلايا بالضرورة — مصدر شائع لحالة مخفية.",
                        extra_ar=["آخر تعبير فقط يُعرض تلقائيًا؛ ما قبله يحتاج print.", "إعادة تشغيل النواة (Runtime → Restart) تمسح كل المتغيرات."]))
    gallery_item("دفتر Google Colab مع GPU", "Google Colab notebook", notebook_cell(COLAB_CODE, COLAB_OUT, n=2, kind="Google Colab notebook (GPU runtime)"),
                 Frame6(where_ar="Google Colab في المتصفح، بعد اختيار Runtime → Change runtime type → GPU. هذه اللقطة إعادة بناء لما يظهر على جهاز بـ GPU (على هذه الآلة القائمة فارغة — انظر لقطة الأجهزة في صفحة PyTorch).",
                        what_ar="خلية تبدأ بـ `!` (أمر نظام لا بايثون) ثم استيراد الإطارين وطباعة النسخ والأجهزة.",
                        code_where_ar="الخلية نفسها. `!nvidia-smi -L` يسأل نظام التشغيل عن بطاقات الرسوم.",
                        output_where_ar="أربعة أسطر: اسم البطاقة (Tesla T4)، النسخ، قائمة أجهزة TensorFlow، وهل CUDA متاح لـ PyTorch.",
                        numbers_ar="`+cu126` في نسخة torch = مبنية بدعم CUDA 12.6 (مقابل `+cpu` محليًا). `GPU:0` = أول بطاقة.",
                        notice_ar="نفس الكود يعمل على CPU وGPU؛ ما يتغير هو هذه الأسطر والسرعة. إن كانت القائمة `[]` في Colab فلم تُفعّل GPU في إعدادات وقت التشغيل.",
                        extra_ar=["Colab يثبّت الإطارين مسبقًا؛ محليًا تثبّتها أنت.", "جلسة Colab تنتهي بعد فترة خمول: احفظ النماذج والملفات إلى Drive."]))
    h2("الأسئلة الستة", "The six questions")
    st.markdown("""
كل لقطة في المعرض تُقرأ بالترتيب نفسه:

1. **أين نحن؟** — أي سطح (دفتر، طرفية، TensorBoard) وأي مرحلة (قبل/أثناء/بعد التدريب).
2. **ماذا نرى؟** — وصف العناصر المرئية.
3. **أين الكود؟** — ما الذي أنتج هذه اللقطة.
4. **أين المخرج؟** — وما نوعه: قيمة معروضة، نص مطبوع، رسم، أو خطأ.
5. **ماذا تعني الأرقام؟** — كل رقم بدلالته ووحدته.
6. **ما الذي يجب أن يلاحظه الباحث؟** — العلامة التي تفرق بين الطبيعي والمقلق.
""")
    intuition("اللقطة ليست صورة تُحفظ بل نص يُقرأ: كل سطر أنتجه سطر كود يمكنك الإشارة إليه. عندما تستطيع الإجابة عن الأسئلة الستة لأي لقطة تصادفك، انتهى دور هذا المعرض.")
    practical_note("اللقطات في هذه الوحدة مبنية **آنيًا** من الكود والنسخ المثبتة في المشروع (لا صور ثابتة)، لذلك تطابق ما ستراه على جهازك بهذه النسخ. ما يُختلف عنه في Colab (مثل GPU) مُشار إليه صراحةً.")
    common_mistake("«أين نافذة Keras؟» — لا توجد. وإن رأيت لقطة بواجهة أزرار ونوافذ منسوبة إلى Keras أو PyTorch فهي إما أداة طرف ثالث أو تلفيق. الأداة الرسومية الوحيدة في المقرر هي TensorBoard (وهي منفصلة).")
    quiz("gallery.orient", [
        Q("`Out[3]:` في الدفتر يعرض…", ["كل ما طُبع", "قيمة آخر تعبير في الخلية", "الأخطاء"], 1, "آخر تعبير."),
        Q("`!nvidia-smi` في خلية Colab…", ["كود بايثون", "أمر نظام تشغيل", "دالة TensorFlow"], 1, "علامة !."),
        Q("Keras/PyTorch لهما واجهة رسومية…", ["نعم", "لا: مكتبات داخل بايثون؛ ما تراه بيئة التشغيل", "فقط في Colab"], 1, "القاعدة."),
        Q("اللقطات هنا هي…", ["صور رسمية", "إعادات بناء تعليمية من مخرجات حقيقية ومُعلَّمة", "رسوم تخيلية"], 1, "Educational recreation."),
    ])
    takeaway("الإطار يطبع نصًا داخل بيئة تشغيل (دفتر/طرفية/IDE). لا واجهة رسومية له. كل لقطة إعادة بناء تعليمية من مخرجات حقيقية، وتُقرأ بالأسئلة الستة.")
    lesson_footer(LESSON, ["الأسطح الأربعة.", "تشريح الخلية.", "الأسئلة الستة."])
