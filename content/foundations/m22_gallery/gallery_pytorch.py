import streamlit as st

from components.callouts import intuition, takeaway
from components.gallery import Frame6, console, gallery_item, notebook_cell
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from labs.fw import gallery_outputs

LESSON = Lesson(
    id="foundations.gallery.pytorch",
    title_ar="معرض PyTorch: الموتر، طباعة النموذج، سجل حلقة التدريب، مخرجات CPU/GPU",
    title_en="PyTorch Gallery: Tensor, Printed Model, Training-Loop Log, CPU/GPU Output",
    module="foundations.gallery",
    order=3,
    prerequisites=["foundations.gallery.keras_tf", "foundations.frameworks.pytorch.training_loop"],
    objectives_ar=["قراءة repr موتر PyTorch وما يظهر وما لا يظهر فيه.", "قراءة النموذج المطبوع وحساب معلماته يدويًا.", "قراءة سجل حلقة تدريب صريحة في الطرفية، ومخرجات فحص الأجهزة على CPU وعلى GPU."],
    terms=["tensor", "model", "epoch"],
    difficulty="beginner",
    summary_ar="أربع لقطات حقيقية: موتر (repr مختصر)، print(model) (شجرة وحدات بلا أشكال مخرجات)، سجل حلقة في الطرفية (سطر لكل حقبة كتبته أنت)، وفحص الأجهزة على CPU مع مقابله على GPU.",
)

DEVICE_CODE = '''import torch
print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
x = torch.ones(2, 3).to(device)
print(x.device)'''
DEVICE_GPU_OUT = '''2.14.0+cu126
True
cuda
cuda:0'''


def render() -> None:
    lesson_header(LESSON)
    o = gallery_outputs()
    st.caption("كل مخرج أنتجته النسخة المثبتة من PyTorch لحظة فتح الصفحة؛ الإطار المحيط إعادة بناء تعليمية.")
    h2("1) مخرج موتر PyTorch", "1) PyTorch tensor output")
    t_out = o["torch_tensor"].split("\n>>> t\n")[1].split("\n>>> t.shape")[0]
    props = o["torch_tensor"].split("\n")[-1]
    gallery_item("موتر PyTorch في خلية", "PyTorch tensor output", notebook_cell("import torch\nt = torch.tensor([[1., 2., 3.], [4., 5., 6.]])\nt", t_out, n=1) + notebook_cell("t.shape, t.dtype, t.device, t.dim()", props, n=2),
                 Frame6(where_ar="خلية دفتر بعد استيراد torch.", what_ar="repr مختصر: `tensor([[...]])` فقط — بلا shape ولا dtype.",
                        code_where_ar="الخلية الأولى تنشئ وتعرض؛ الثانية تسأل عن الخصائص صراحةً.", output_where_ar="`Out` الأولى: القيم. `Out` الثانية: صف من أربع خصائص.",
                        numbers_ar="`torch.Size([2, 3])` يُطبع هنا كـ (2, 3)؛ `torch.float32` الافتراضي للكسور؛ `cpu` الجهاز؛ `2` الرتبة.",
                        notice_ar="بخلاف TensorFlow، repr PyTorch يذكر dtype **فقط إن لم يكن الافتراضي** (float32/int64) والجهاز فقط إن لم يكن CPU، و`requires_grad=True` إن كان مفعّلًا. غياب المعلومة يعني «الافتراضي» لا «غير معروف».",
                        extra_ar=["`tensor([1, 2, 3])` بلا فاصلة → int64 ولا يُطبع dtype.", "`tensor([...], device='cuda:0')` يظهر على GPU."]))
    h2("2) النموذج المطبوع", "2) Printed model architecture")
    m_code = "model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))\nprint(model)\nsum(p.numel() for p in model.parameters())"
    m_out = o["torch_model"].split("\n>>> print(model)\n")[1].replace("\n>>> sum(p.numel() for p in model.parameters())\n", "\n")
    gallery_item("print(model) في PyTorch", "PyTorch printed model architecture", notebook_cell(m_code, m_out, n=3),
                 Frame6(where_ar="بعد بناء نموذج Sequential.", what_ar="شجرة: اسم الصنف الخارجي ثم وحدة لكل سطر بفهرسها `(0)`, `(1)`… وبمعاملاتها.",
                        code_where_ar="`print(model)` يطبع؛ آخر تعبير (المجموع) يظهر كـ Out.", output_where_ar="الشجرة مطبوعة ثم الرقم 193 كقيمة.",
                        numbers_ar="`in_features=2, out_features=16` أشكال الطبقة؛ `bias=True` انحياز. المعلمات: 2×16+16 = 48، 16×8+8 = 136، 8×1+1 = 9 → 193 (نفس نموذج Keras في المعرض السابق).",
                        notice_ar="لا أشكال مخرجات ولا عدّ معلمات في الطباعة (بخلاف summary). للتحقق من الأشكال مرّر دفعة وهمية. الفهارس `(0)…(4)` هي طريقة الوصول: `model[0].weight`.",
                        extra_ar=["في صنف مخصص تظهر أسماء الخصائص (`(fc1): Linear(...)`) بدل الفهارس."]))
    h2("3) سجل حلقة التدريب في الطرفية", "3) Training-loop console output")
    gallery_item("تشغيل train.py من الطرفية", "PyTorch training loop console/log output", console(o["torch_log"] + "\nsaved model to model.pt", prompt="$ python train.py"),
                 Frame6(where_ar="طرفية (لا دفتر): شغّلنا ملفًا بـ `python train.py`.", what_ar="سطر لكل حقبة بثلاثة أرقام، ثم رسالة حفظ.",
                        code_where_ar="ليس على الشاشة — في الملف `train.py`: حلقة صريحة تطبع `f\"epoch {ep}/5 - loss: … - val_loss: … - val_accuracy: …\"` (درس حلقة التدريب).",
                        output_where_ar="كل ما تراه هو ما طبعته أنت بـ print؛ PyTorch لا يطبع شيئًا من تلقاء نفسه أثناء التدريب.",
                        numbers_ar="loss متوسط موزون لخسارة التدريب في الحقبة؛ val_loss وval_accuracy على التحقق بوضع eval. الشكل يشبه سطر Keras لأننا كتبناه كذلك.",
                        notice_ar="غياب شريط تقدم أو زمن ليس عطلًا: في PyTorch كل سطر مطبوع قرارك (أضف `tqdm` لشريط تقدم). loss ينخفض من 0.66 إلى 0.27 وval_accuracy يرتفع: تدريب سليم.",
                        extra_ar=["لو رأيت `UserWarning` باللون الأصفر قبل الأسطر فهو تحذير لا خطأ — لكن اقرأه (بث الأهداف!)."]))
    h2("4) مخرجات الأجهزة: CPU وGPU", "4) CPU/GPU device output")
    cpu_out = "\n".join(l for l in o["torch_devices"].split("\n") if not l.startswith(">>>")).replace("'", "")
    c1, c2 = st.columns(2)
    with c1:
        st.html(notebook_cell(DEVICE_CODE, cpu_out, n=4, kind="This machine (CPU-only install)"))
    with c2:
        st.html(notebook_cell(DEVICE_CODE, DEVICE_GPU_OUT, n=4, kind="Colab with GPU runtime (recreated)"))
    gallery_item("نفس الكود على آلة CPU وعلى Colab بـ GPU", "CPU/GPU device output", "",
                 Frame6(where_ar="يسار: هذه الآلة (تثبيت CPU). يمين: إعادة بناء لما يظهر في Colab مع GPU.", what_ar="نفس الخلية بسبعة أسطر؛ يختلف المخرج في كل سطر تقريبًا.",
                        code_where_ar="الخلية نفسها في الجهتين. السطر الرابع هو النمط القياسي لاختيار الجهاز.", output_where_ar="أربعة أسطر مطبوعة في كل جهة.",
                        numbers_ar="`+cpu` مقابل `+cu126` في النسخة؛ `False/True` لتوفر CUDA؛ `cpu/cuda`؛ `cpu/cuda:0` جهاز الموتر بعد `.to`.",
                        notice_ar="الكود لم يتغير حرفًا: هذا معنى «الجهاز صريح لكن قابل للنقل». على CPU `.to('cuda')` كان سيفشل — لذلك لا تكتب `cuda` حرفيًا بل `device`.",
                        extra_ar=["`cuda:0` = أول بطاقة؛ مع بطاقتين ترى `cuda:1` أيضًا.", "TensorFlow يعطي المعلومة نفسها بـ `list_physical_devices`."]))
    intuition("PyTorch «صامت» افتراضيًا: لا شريط تقدم ولا سجل ولا ملخص أشكال. كل ما يظهر على الشاشة في مشروع PyTorch كتبه المؤلف — وهذه ميزة تقرأ منها ما يهمه بالضبط.")
    quiz("gallery.pt", [
        Q("`tensor([[1., 2., 3.], [4., 5., 6.]])` بلا dtype ظاهر يعني…", ["dtype غير معروف", "float32 الافتراضي", "float64"], 1, "الافتراضي لا يُطبع."),
        Q("print(model) في PyTorch يعرض…", ["أشكال المخرجات والمعلمات", "شجرة الوحدات ومعاملاتها فقط", "المنحنيات"], 1, "بلا أشكال."),
        Q("أسطر سجل التدريب في PyTorch مصدرها…", ["الإطار تلقائيًا", "print الذي كتبه المؤلف", "TensorBoard"], 1, "صريح."),
        Q("`2.14.0+cu126` يعني…", ["نسخة CPU", "مبنية بدعم CUDA 12.6", "نسخة تجريبية"], 1, "لاحقة البناء."),
    ])
    takeaway("موتر: repr مختصر (الافتراضي لا يُطبع). print(model): شجرة بلا أشكال. السجل: ما كتبته أنت. الأجهزة: نفس الكود، مخرج مختلف — لذلك `device` لا `'cuda'`.")
    lesson_footer(LESSON, ["أربع لقطات حقيقية.", "CPU مقابل GPU جنبًا إلى جنب.", "PyTorch صامت افتراضيًا."])
