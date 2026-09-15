import streamlit as st

from components.callouts import common_mistake, debugging_note, intuition, practical_note, takeaway, why
from components.code_lab import exec_source
from components.comparison import compare_table
from components.gallery import Frame6, gallery_item, notebook_cell
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="course.w13.colab_runtime",
    title_ar="Google Colab: اختيار وقت التشغيل، اكتشاف GPU، معلومات الجهاز، تشخيص OOM، وملاحظات الأداء + الاختبار البعدي",
    title_en="Google Colab: Runtime Choice, GPU Detection, Device Info, OOM Diagnostics & Performance Notes — & Post-test",
    module="course.w13",
    order=3,
    prerequisites=["course.w13.cpu_gpu_memory", "foundations.gallery.orientation"],
    objectives_ar=["خطوات Colab: فتح دفتر، اختيار GPU، التحقق منه، رفع البيانات، الحفظ إلى Drive.", "اكتشاف GPU ومعلومات الجهاز في TensorFlow وPyTorch، وقراءة nvidia-smi.", "تشخيص OOM وملاحظات الأداء العملية، والاختبار البعدي."],
    terms=["batch_size"],
    difficulty="beginner",
    summary_ar="Runtime → Change runtime type → GPU (T4). تحقق: list_physical_devices('GPU') / cuda.is_available() / !nvidia-smi. OOM: قلّل الدفعة، float16، أعد تشغيل وقت التشغيل. الأداء: tf.data + prefetch، دفعة مناسبة، لا print في الحلقة، احفظ نقاط الحفظ في Drive.",
)

DETECT = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf, torch
print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
print("PyTorch   ", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0), "| VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print("-> CPU only on this machine; in Colab with a GPU runtime the two lines above list a T4/L4/A100")'''

NVSMI = '''Tue Sep 15 12:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   42C    P0             27W /   70W |    3121MiB /  15360MiB |     63%      Default |
+-----------------------------------------+------------------------+----------------------+'''

OOM = '''ResourceExhaustedError: Graph execution error:
... OOM when allocating tensor with shape[256,224,224,64] and type float on /job:localhost/replica:0/task:0/device:GPU:0
by allocator GPU_0_bfc [Op:Conv2D]'''


def render() -> None:
    lesson_header(LESSON)
    why("Colab يمنحك GPU مجانًا (بحدود) في المتصفح — وهو بيئة تشغيل لا إطار (الوحدة 22). المطلوب ثلاث مهارات: تفعيل GPU والتحقق منه، إدارة الملفات والجلسة، وقراءة OOM ونصائح الأداء.")
    h2("1) خطوات Colab", "1) Colab steps")
    compare_table(["الخطوة", "أين", "ملاحظة"],
                  [("افتح دفترًا", "colab.research.google.com → New notebook (أو ارفع .ipynb)", "يعمل بحساب Google"), ("فعّل GPU", "Runtime → Change runtime type → Hardware accelerator: **GPU** (T4 مجانًا)", "أعد تشغيل الجلسة بعد التغيير"), ("تحقق", "خلية: `!nvidia-smi` + الكود أدناه", "قبل أي تدريب"),
                   ("البيانات", "`from google.colab import drive; drive.mount('/content/drive')` أو رفع ملف من اللوحة اليسرى", "الملفات المرفوعة تُحذف مع الجلسة؛ Drive يبقى"), ("الحفظ", "`model.save('/content/drive/MyDrive/proj/m.keras')` + `ModelCheckpoint` هناك", "الجلسة المجانية تنتهي بعد خمول أو ~12 ساعة"), ("الحزم", "مثبّتة: tensorflow, keras, torch, sklearn, pandas… `!pip install x` للباقي", "النسخ قد تختلف عن المشروع: اطبعها")],
                  ["rtl", "rtl", "rtl"])
    h2("2) اكتشاف GPU ومعلومات الجهاز", "2) GPU detection & device info")
    st.code(DETECT, language="python")
    st.code(exec_source(DETECT).rstrip(), language="text")
    st.caption("المخرج أعلاه من هذه الآلة (CPU). في Colab مع GPU تظهر البطاقة وذاكرتها.")
    gallery_item("!nvidia-smi في Colab", "nvidia-smi output", notebook_cell("!nvidia-smi", NVSMI, n=1, kind="Google Colab (GPU runtime)"),
                 Frame6(where_ar="خلية Colab بأمر نظام (علامة !) بعد تفعيل GPU. إعادة بناء تعليمية لما يظهر مع T4.", what_ar="جدول من تعريف NVIDIA: النسخ في الأعلى، ثم سطر لكل بطاقة.",
                        code_where_ar="السطر الوحيد `!nvidia-smi`.", output_where_ar="الجدول كاملًا.", numbers_ar="`Tesla T4`: اسم البطاقة. `3121MiB / 15360MiB`: الذاكرة المستخدمة/الكلية (≈15 GB VRAM). `63%`: نسبة الاستغلال لحظيًا. `CUDA Version 12.4`: أقصى CUDA يدعمه التعريف.",
                        notice_ar="Memory-Usage قبل التدريب يجب أن يكون صغيرًا؛ إن كان ممتلئًا من جلسة سابقة → Runtime → Restart. GPU-Util منخفض أثناء التدريب = عنق زجاجة في البيانات (CPU/القرص) لا في النموذج.", extra_ar=["شغّله في خلية منفصلة أثناء التدريب لمراقبة الذاكرة."]))
    h2("3) تشخيص OOM", "3) OOM diagnostics")
    st.code(OOM, language="text")
    compare_table(["ما تقرؤه", "المعنى", "التصرف"],
                  [("`shape[256,224,224,64]`", "تنشيط دفعة 256 صورة 224×224 بـ 64 قناة = 3.2 مليار قيمة × 4 B ≈ 13 GB", "الدفعة ÷ 4 أو أكثر"), ("`[Op:Conv2D]`", "الطبقة التي طلبت الذاكرة", "الطبقات الأولى (دقة كاملة) هي الأثقل تنشيطًا"), ("`device:GPU:0`", "على البطاقة لا RAM", "لا يفيد تكبير RAM"),
                   ("يظهر بعد حقب عدة لا فورًا", "تسرّب: تجميع موترات (loss بلا item/float) أو نماذج متعددة في الذاكرة", "`float(loss)`؛ `keras.backend.clear_session()`؛ أعد التشغيل"), ("يظهر في التقييم فقط", "دفعة التقييم أكبر أو بلا no_grad (PyTorch)", "`batch_size` في evaluate/predict؛ `torch.no_grad()`")],
                  ["code", "rtl", "rtl"])
    debugging_note("ترتيب علاج OOM: (1) الدفعة ÷ 2 حتى يعمل؛ (2) `mixed_float16`؛ (3) صورة/نافذة أصغر؛ (4) شبكة أصغر أو GlobalAveragePooling بدل Flatten؛ (5) تجميع التدرجات على دفعات صغيرة (الوحدة 21) إن أردت دفعة فعلية كبيرة. وبعد OOM أعد تشغيل وقت التشغيل: الذاكرة قد تبقى محجوزة.")
    h2("4) ملاحظات الأداء", "4) Performance notes")
    compare_table(["الملاحظة", "لماذا", "كيف"],
                  [("GPU-Util منخفض", "الدفعات لا تصل بسرعة (تحميل/تحويل على CPU)", "`tf.data` مع `map(num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)`، أو `DataLoader(num_workers>0)`"), ("الحقبة الأولى بطيئة", "بناء الرسم/الترجمة (tf.function)", "طبيعي؛ لا تقس على الأولى"), ("دفعة صغيرة جدًا", "استغلال ضعيف للتوازي", "كبّرها مع تعديل η"),
                   ("print/رسم داخل الحلقة", "مزامنة GPU↔CPU كل خطوة", "سجّل كل حقبة أو استخدم callbacks/TensorBoard"), ("نقل البيانات كل خطوة (PyTorch)", "`.to(device)` بطيء للمصفوفات الكبيرة", "انقل مرة واحدة إن اتسعت أو استخدم `pin_memory=True`"), ("انقطاع الجلسة", "حدود Colab المجاني", "`ModelCheckpoint` إلى Drive + استئناف بـ `load_model`")],
                  ["rtl", "rtl", "code"])
    intuition("GPU سريع في الحساب وبطيء في الانتظار: إن كان CPU يجهّز الدفعات ببطء أو تطبع كل خطوة، تجلس البطاقة عاطلة. راقب GPU-Util: الهدف > 80% أثناء التدريب.")
    practical_note("مشروع الأسبوع 14: طوّر على عينة صغيرة محليًا (CPU) حتى يعمل الكود بلا أخطاء، ثم انقله إلى Colab للتدريب الكامل. الكود نفسه؛ ما يتغير هو مسارات الملفات والدفعة.")
    with st.container(horizontal=True):
        st.button("لقطات Colab وGPU (الوحدة 22)", icon=":material/photo_library:", on_click=goto, args=("foundations.gallery.pytorch",), key="w13_go_gal")
        st.button("أخطاء الأجهزة في الوحدة 21", icon=":material/bug_report:", on_click=goto, args=("foundations.frameworks.tensorflow.tensorboard_errors",), key="w13_go_err")
    common_mistake("تدريب ساعتين في Colab دون حفظ نقاط إلى Drive، ثم تنتهي الجلسة. أول سطر في أي دفتر تدريب طويل: `ModelCheckpoint('/content/drive/MyDrive/.../ckpt.keras', save_best_only=True)`.")
    post_test("course.w13", "13", [
        Q("GPU يسرّع التدريب لأنه…", ["أدق", "ينفذ عمليات الضرب بالتوازي", "يملك ذاكرة أكبر"], 1, ""),
        Q("`OOM when allocating tensor with shape[256,224,224,64]`:", ["خطأ كود", "تنشيط دفعة كبيرة لا يتسع في VRAM: قلّل الدفعة", "GPU غائب"], 1, ""),
        Q("GPU يغيّر…", ["الدقة", "السرعة", "الخسارة"], 1, ""),
        Q("تفعيل GPU في Colab:", ["pip", "Runtime → Change runtime type", "تلقائي"], 1, ""),
        Q("GPU-Util منخفض أثناء التدريب يشير إلى…", ["نموذج صغير جدًا", "عنق زجاجة في تغذية البيانات", "η كبير"], 1, ""),
        Q("الملفات المرفوعة مباشرة إلى Colab…", ["تبقى", "تُحذف مع الجلسة؛ استخدم Drive", "تُحفظ في GitHub"], 1, ""),
        Q("الجزء من الذاكرة الذي ينمو مع الدفعة:", ["المعلمات", "التنشيطات", "حالة Adam"], 1, ""),
        Q("مضاعفة الدفعة تستدعي…", ["إبقاء η", "تعديل η", "تقليل الحقب"], 1, ""),
    ])
    takeaway("Colab: فعّل GPU، تحقق (list_physical_devices / cuda.is_available / nvidia-smi)، احفظ إلى Drive. OOM: الشكل في الرسالة يخبرك؛ الدفعة أولًا. الأداء: غذّ GPU بسرعة ولا تطبع كل خطوة.")
    lesson_footer(LESSON, ["خطوات Colab.", "الاكتشاف وnvidia-smi.", "OOM والأداء والاختبار البعدي."])
