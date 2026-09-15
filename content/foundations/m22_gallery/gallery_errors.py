import streamlit as st

from components.callouts import debugging_note, intuition, takeaway
from components.comparison import compare_table
from components.gallery import Frame6, console, gallery_item, notebook_cell
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from labs.fw import gallery_outputs

LESSON = Lesson(
    id="foundations.gallery.errors",
    title_ar="معرض الأخطاء: خطأ شكل نموذجي، وخطأ dtype/جهاز نموذجي — كما تظهر فعليًا",
    title_en="Error Gallery: A Typical Shape Error and a Typical dtype/Device Error, As They Really Appear",
    module="foundations.gallery",
    order=4,
    prerequisites=["foundations.gallery.pytorch", "foundations.frameworks.keras.errors", "foundations.frameworks.pytorch.saving_errors"],
    objectives_ar=["رؤية شكل الخطأ الكامل في الدفتر (الـ traceback بالأحمر) وتحديد السطر المفيد بسرعة.", "خطأ شكل في Keras وفي PyTorch جنبًا إلى جنب.", "خطأ dtype وخطأ جهاز في PyTorch، والفرق بين تحذير وخطأ."],
    terms=["shape", "dtype"],
    difficulty="beginner",
    summary_ar="الخطأ يظهر بالأحمر تحت الخلية: traceback طويل ثم آخر سطر هو الرسالة. Keras: expected … but received …. PyTorch: mat1 and mat2 shapes cannot be multiplied (a×b and c×d)؛ dtype: Double and Float؛ device: same device.",
)

KERAS_ERR_CODE = 'model = keras.Sequential([layers.Input(shape=(5,)), layers.Dense(1)])   # يتوقع 5 خصائص\nmodel.compile(optimizer="sgd", loss="mse")\nmodel.fit(X_train, y_train, epochs=1)                                   # X_train.shape == (450, 2)'
TORCH_ERR_CODE = 'model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))\nxb = torch.randn(4, 5)          # 4 ملاحظات × 5 خصائص، لكن Linear يتوقع 2\nout = model(xb)'
TORCH_DTYPE_CODE = 'X = np.random.default_rng(0).normal(size=(4, 2))   # NumPy: float64\nxb = torch.from_numpy(X)                           # torch.float64 (Double)\nout = model(xb)                                    # أوزان float32 (Float)'
TORCH_DEVICE_CODE = 'model = model.to("cuda")\nxb = torch.randn(4, 2)          # بقي على CPU!\nout = model(xb)'


def _tb(lines: str, last: str) -> str:
    return "---------------------------------------------------------------------------\n" + lines + "\n" + last


def render() -> None:
    lesson_header(LESSON)
    o = gallery_outputs()
    st.caption("رسائل الأخطاء أدناه حقيقية (من النسخ المثبتة)؛ الـ traceback المحيط مختصر وإعادة بناء تعليمية لشكله في الدفتر.")
    h2("1) خطأ شكل نموذجي — Keras", "1) Typical shape error — Keras")
    k_err = o["keras_shape_error"]
    k_tb = _tb("ValueError                                Traceback (most recent call last)\nCell In[6], line 3\n      1 model = keras.Sequential([layers.Input(shape=(5,)), layers.Dense(1)])\n      2 model.compile(optimizer=\"sgd\", loss=\"mse\")\n----> 3 model.fit(X_train, y_train, epochs=1)\n\nFile .../keras/src/utils/traceback_utils.py:122, in filter_traceback.<locals>.error_handler(*args, **kwargs)\n    119     filtered_tb = _process_traceback_frames(e.__traceback__)\n    120     # To get the full stack trace, call:\n    121     # `keras.config.disable_traceback_filtering()`\n--> 122     raise e.with_traceback(filtered_tb) from None\n    123 finally:\n    124     del filtered_tb\n", k_err)
    gallery_item("خطأ شكل المدخل في fit (Keras)", "Typical shape error (Keras)", notebook_cell(KERAS_ERR_CODE, "", n=6, stderr=k_tb),
                 Frame6(where_ar="خلية دفتر عند استدعاء fit؛ الخطأ ظهر فورًا قبل أي حقبة.", what_ar="كتلة حمراء: اسم الخطأ في الأعلى، سهم `---->` يشير إلى سطر الخلية المسؤول، إطارات داخلية من Keras، ثم **الرسالة في الأسفل**.",
                        code_where_ar="السطر 3 (`model.fit`) كما يشير السهم؛ لكن **السبب** في السطر 1 (`shape=(5,)`).", output_where_ar="لا مخرج طبيعي؛ كل ما تحت الخلية هو الخطأ (stderr).",
                        numbers_ar="`expected axis -1 of input shape to have value 5` = آخر بُعد المتوقع 5؛ `received input with shape (None, 2)` = ما وصل فعلًا: أي عدد من الملاحظات × 2 خاصية.",
                        notice_ar="اقرأ من الأسفل: `ValueError` + الشكلان. `None` ليس المشكلة (بُعد الدفعة)؛ المشكلة 5 مقابل 2. الإطارات الوسطى من مكتبة Keras لا تهمك — Keras يرشّحها أصلًا (`traceback filtering`).",
                        extra_ar=["الحل: `Input(shape=(X_train.shape[1],))`.", "شرح قراءة رسائل Keras بالخطوات الخمس في الوحدة 21."]))
    h2("2) خطأ شكل نموذجي — PyTorch", "2) Typical shape error — PyTorch")
    p_last = o["torch_shape_error"].split("\n")[-1]
    p_tb = _tb("RuntimeError                              Traceback (most recent call last)\nCell In[7], line 3\n      1 model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))\n      2 xb = torch.randn(4, 5)\n----> 3 out = model(xb)\n\nFile .../torch/nn/modules/module.py:1775, in Module._wrapped_call_impl(self, *args, **kwargs)\n   ...\nFile .../torch/nn/modules/linear.py:134, in Linear.forward(self, input)\n    133 def forward(self, input: Tensor) -> Tensor:\n--> 134     return F.linear(input, self.weight, self.bias)\n", p_last)
    gallery_item("خطأ شكل المدخل في forward (PyTorch)", "Typical shape error (PyTorch)", notebook_cell(TORCH_ERR_CODE, "", n=7, stderr=p_tb),
                 Frame6(where_ar="خلية دفتر عند أول تمرير أمامي.", what_ar="نفس البنية: اسم الخطأ، سهم إلى `model(xb)`، إطارات من `torch/nn/modules/linear.py` تُظهر أن الفشل داخل `F.linear`، ثم الرسالة.",
                        code_where_ar="السطر 3؛ السبب في السطر 2 (5 خصائص) أو في `nn.Linear(2, …)` — أحدهما خاطئ.", output_where_ar="الخطأ فقط.",
                        numbers_ar="`(4x5 and 2x16)`: الأول مدخلك (4 ملاحظات × 5 خصائص)، الثاني الوزن منقولًا (in=2 × out=16). الرقمان الداخليان 5 و2 يجب أن يتساويا.",
                        notice_ar="PyTorch لا يرشّح الـ traceback: ستمر بإطارات `module.py` و`linear.py` — تجاهلها وانزل إلى آخر سطر. اسم الملف `linear.py` يخبرك أي طبقة فشلت (Linear) لا أيّ واحدة من طبقاتك؛ لمعرفة ذلك مرّر دفعة وهمية طبقة طبقة.",
                        extra_ar=["الحل: `nn.Linear(xb.shape[1], 16)` أو إصلاح البيانات."]))
    h2("3) خطأ dtype وخطأ جهاز — PyTorch", "3) dtype and device errors — PyTorch")
    c1, c2 = st.columns(2)
    with c1:
        st.html(notebook_cell(TORCH_DTYPE_CODE, "", n=8, stderr=_tb("RuntimeError                              Traceback (most recent call last)\nCell In[8], line 3\n----> 3 out = model(xb)\n", o["torch_dtype_error"]), kind="dtype error (this machine)"))
    with c2:
        st.html(notebook_cell(TORCH_DEVICE_CODE, "", n=9, stderr=_tb("RuntimeError                              Traceback (most recent call last)\nCell In[9], line 3\n----> 3 out = model(xb)\n", o["torch_device_error_gpu_machine"]), kind="device error (GPU machine, recreated)"))
    gallery_item("dtype (يسار) وجهاز (يمين)", "Typical dtype / device error", "",
                 Frame6(where_ar="يسار: هذه الآلة — بيانات NumPy float64 دخلت نموذجًا بأوزان float32. يمين: آلة بـ GPU — النموذج على cuda والدفعة على cpu (على هذه الآلة يظهر بدلًا منه `AssertionError: Torch not compiled with CUDA enabled` عند `.to('cuda')`).",
                        what_ar="خطآن قصيران؛ الرسالة في كل منهما سطر واحد واضح.", code_where_ar="السطر الذي أنشأ الموتر (`from_numpy` / `torch.randn` بلا `.to(device)`)، لا سطر `model(xb)` الذي يشير إليه السهم.",
                        output_where_ar="الخطأ فقط.", numbers_ar="`Double and Float` = float64 مقابل float32. `cuda:0 and cpu` = الجهازان المتخالفان.",
                        notice_ar="كلاهما «عدم تطابق» بين طرفي عملية: طابق النوعين (`xb.float()`) أو الجهازين (`xb.to(device)`). السهم يشير إلى مكان **الانفجار** لا مكان **السبب**.",
                        extra_ar=["TensorFlow يعطي المقابل: `cannot compute AddV2 … int32 … float` للنوع، وتحذير `Could not load dynamic library` للجهاز (تحذير لا خطأ)."]))
    h2("تحذير أم خطأ؟", "Warning or error?")
    st.html(console("UserWarning: Using a target size (torch.Size([32])) that is different to the input size (torch.Size([32, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.\n  return F.mse_loss(input, target, reduction=self.reduction)\nepoch 1/5 - loss: 1.0412\nepoch 2/5 - loss: 1.0398", prompt="$ python train.py"))
    compare_table(["", "تحذير (Warning)", "خطأ (Error / Exception)"],
                  [("اللون في الدفتر", "أصفر/وردي فاتح", "أحمر مع traceback"), ("هل يتوقف التنفيذ؟", "لا — يكمل", "نعم — يتوقف عند السطر"), ("هل يجب قراءته؟", "**نعم**: كثير منها يشير إلى خطأ صامت (البث أعلاه: النتائج خاطئة والكود «يعمل»)", "نعم: آخر سطر أولًا"),
                   ("أمثلة", "target size mismatch، Could not load cudart، deprecation", "ValueError، RuntimeError، InvalidArgumentError، OOM"), ("لتحويله إلى خطأ أثناء التطوير", "`warnings.simplefilter('error')`", "—")],
                  ["rtl", "rtl", "rtl"])
    debugging_note("أخطر سطر في اللقطة الأخيرة هو الأصفر لا الأحمر: التدريب «يعمل» بخسارة ≈ 1.04 لا تنخفض — لأن (32,1)−(32,) بُثّت إلى (32,32). لو كان خطأً لتوقف؛ لأنه تحذير، تجاهله كثيرون وأضاعوا ساعات.")
    intuition("كل الأخطاء الأربعة «عدم تطابق» بين طرفين: شكل، نوع، جهاز. الرسالة تذكر الطرفين دائمًا. مهمتك: أي الطرفين هو المقصود، ثم أصلح الآخر عند **مصدره**.")
    quiz("gallery.err", [
        Q("في traceback الدفتر، السطر الذي تقرؤه أولًا:", ["الأول", "الأخير (الرسالة)", "الأوسط"], 1, "من الأسفل."),
        Q("السهم `---->` يشير إلى…", ["سبب الخطأ", "السطر الذي انفجر فيه", "الحل"], 1, "مكان الانفجار."),
        Q("`(4x5 and 2x16)`: الخلاف بين", ["4 و16", "5 و2", "4 و2"], 1, "الداخليان."),
        Q("تحذير target size mismatch…", ["يوقف التدريب", "يكمل بنتائج خاطئة (بث)", "بلا أثر"], 1, "خطأ صامت."),
    ])
    takeaway("traceback: اقرأ آخر سطر، السهم مكان الانفجار لا السبب، الإطارات الوسطى من المكتبة. الأخطاء عدم تطابق طرفين (شكل/نوع/جهاز). التحذيرات الصفراء قد تكون أخطر من الأحمر.")
    lesson_footer(LESSON, ["خطأ شكل في الإطارين.", "dtype وجهاز.", "تحذير مقابل خطأ."])
