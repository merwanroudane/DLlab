import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.frameworks.pytorch.tensors",
    title_ar="torch.Tensor: الإنشاء، الشكل، dtype، الجهاز، CPU↔GPU، العمليات، البث",
    title_en="torch.Tensor: Creation, Shape, dtype, Device, CPU↔GPU, Ops & Broadcasting",
    module="foundations.frameworks",
    parent="foundations.frameworks.pytorch",
    order=20,
    prerequisites=["foundations.frameworks.pytorch", "foundations.frameworks.tensorflow.tensors"],
    objectives_ar=["إنشاء موترات PyTorch وفحص shape/dtype/device.", "النقل بين CPU وGPU بـ .to(device) وفهم خطأ عدم تطابق الجهاز.", "العمليات والبث وview/reshape والتحويل من/إلى NumPy، مع الفروق عن TensorFlow."],
    terms=["tensor", "shape", "dtype", "broadcasting"],
    labs=["labs.torch_tensor_explorer"],
    difficulty="intermediate",
    summary_ar="torch.tensor/zeros/ones/randn/arange؛ .shape .dtype .device؛ .to(device)؛ عمليات كـ NumPy مع بث؛ view/reshape/unsqueeze؛ .numpy() و torch.from_numpy. dtype الافتراضي float32/int64. المفهوم واحد، الواجهة تختلف.",
)

CODE = '''import numpy as np, torch
torch.manual_seed(0)
# الإنشاء
a = torch.tensor([[1, 2, 3], [4, 5, 6]])                 # int64 (لا int32 كما في TensorFlow)
b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])     # float32
z = torch.zeros(2, 3); o = torch.ones(3); r = torch.randn(2, 3); ar = torch.arange(6).reshape(2, 3)
n = torch.from_numpy(np.arange(6.0).reshape(2, 3))       # يشارك الذاكرة مع NumPy! float64
for name, t in [("tensor int", a), ("tensor float", b), ("zeros", z), ("ones", o), ("randn", r), ("arange", ar), ("from_numpy", n)]:
    print(f"{name:<13} shape={tuple(t.shape)!s:<8} dim={t.dim()} dtype={str(t.dtype)[6:]:<8} device={t.device} numel={t.numel()}")

# العمليات والبث (كما في NumPy)
print("b + ones(3)      ->", (b + o).tolist())
print("b.mean(dim=0)    ->", b.mean(dim=0).tolist(), "| b.sum() ->", b.sum().item())
print("b @ b.T          ->", (b @ b.T).tolist())
print("b[:, 1]          ->", b[:, 1].tolist(), "| b.view(3, 2) ->", b.view(3, 2).tolist())
print("unsqueeze/squeeze:", b[:, 1].unsqueeze(1).shape, "->", b[:, 1].unsqueeze(1).squeeze(1).shape)

# dtype: PyTorch يرقّي int + float تلقائيًا (بخلاف TensorFlow) لكن ليس دائمًا ما تريد
print("a(int64) + b(float32) ->", (a + b).dtype, "(promoted)")
print("a / 2 ->", (a / 2).dtype, "| a // 2 ->", (a // 2).dtype, "| a.float() ->", a.float().dtype)

# الجهاز
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| cuda available:", torch.cuda.is_available())
x_dev = b.to(device)                                      # نسخة على الجهاز (على CPU: نفس الموتر)
print("x_dev.device:", x_dev.device)
try:
    cpu_t = torch.ones(2, 3); gpu_t = torch.ones(2, 3).to("cuda")   # على آلة بلا CUDA يفشل هنا
    cpu_t + gpu_t
except Exception as e:
    print("mixing devices ->", type(e).__name__ + ":", str(e).splitlines()[0][:90])

# التبادل مع NumPy
print("b.numpy() ->", type(b.numpy()).__name__, "| shares memory:", np.shares_memory(b.numpy(), b.numpy()))
n_np = n.numpy(); n_np[0, 0] = 99.0
print("from_numpy shares memory: tensor sees the NumPy write ->", n[0, 0].item())'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`torch.Tensor`**: مصفوفة متعددة الأبعاد بشكل وdtype و**جهاز** (`cpu` أو `cuda:0`) وعلامة `requires_grad`. تُنشأ بـ `torch.tensor` (من بيانات) أو `zeros/ones/randn/arange` (بشكل)، وتُنقل بين الأجهزة بـ `.to(device)`. كل ما يشارك في عملية واحدة يجب أن يكون على **نفس الجهاز**.")
    code_lab(CodeLab(
        key="pt_tensors", title_ar="إنشاء، فحص، عمليات، dtype، جهاز، تبادل مع NumPy", code=CODE, level="B",
        before=Before(goal_ar="نفس جولة موترات TensorFlow لكن بـ PyTorch، مع إبراز الفروق: int64 افتراضيًا، ترقية dtype تلقائية، `.to(device)`، ومشاركة الذاكرة مع NumPy.", stage_ar="PyTorch: الموترات.",
                      inputs_ar="موترات صغيرة يدوية.", expected_ar="جدول الخصائص لسبع طرق إنشاء؛ بث وview وunsqueeze؛ الترقية التلقائية؛ device=cpu؛ خطأ عند خلط الأجهزة (إن وُجد GPU) أو خطأ CUDA غير متاح؛ from_numpy يرى تعديل NumPy.",
                      objects_ar="`torch.tensor`, `torch.zeros/ones/randn/arange`, `torch.from_numpy`, `.to(device)`."),
        explain=[("3-9", "سبع طرق إنشاء. **int64** للأعداد الصحيحة (TensorFlow: int32). `from_numpy` **يشارك** الذاكرة (لا نسخ)."), ("12-16", "البث والاختزال بـ `dim=` (لا `axis=`) وmatmul بـ `@` والفهرسة و`view`. `unsqueeze(1)` يضيف محورًا بطول 1: (2,) → (2,1) — ستحتاجه لأشكال الهدف."),
                 ("19-20", "**فرق عن TensorFlow**: PyTorch يرقّي int64 + float32 إلى float32 تلقائيًا. القسمة `/` تعطي float دائمًا؛ `//` تبقي int."), ("23-31", "`device` كسلسلة، `.to(device)` للنقل. خلط جهازين يرفع `RuntimeError: Expected all tensors to be on the same device`. بلا CUDA يفشل `.to('cuda')` نفسه."),
                 ("34-36", "`.numpy()` على CPU يشارك الذاكرة أيضًا: تعديل أحدهما يظهر في الآخر. على GPU تحتاج `.cpu().numpy()`، ومع requires_grad تحتاج `.detach()`.")],
        run=run_printed(CODE),
        after_ar="- `dim=0` في PyTorch = `axis=0` في NumPy/TensorFlow.\n- الترقية التلقائية مريحة لكنها قد تخفي int حيث تريد float: للفئات (`CrossEntropyLoss`) تحتاج **int64**، وللانحدار float32 — تحقق من `.dtype` دائمًا.\n- مشاركة الذاكرة مع NumPy تعني: لا تعدّل مصفوفة NumPy التي بنيت منها موترًا وأنت تتوقع بقاء الموتر.",
    ))
    st.button("افتح مستكشف موترات PyTorch", icon=":material/science:", type="primary", on_click=go, args=("labs.torch_tensor_explorer",), key="pt_tensors_lab")
    h2("نفس المفهوم، واجهتان", "One concept, two APIs")
    compare_table(["", "TensorFlow", "PyTorch"],
                  [("إنشاء من بيانات", "`tf.constant([...])`", "`torch.tensor([...])`"), ("أعداد صحيحة افتراضيًا", "int32", "**int64**"), ("الرتبة", "`tf.rank(x)`", "`x.dim()` / `x.ndim`"), ("عدد العناصر", "`tf.size(x)`", "`x.numel()`"),
                   ("اختزال على محور", "`tf.reduce_mean(x, axis=0)`", "`x.mean(dim=0)`"), ("إعادة تشكيل", "`tf.reshape(x, s)`", "`x.view(s)` / `x.reshape(s)`"), ("إضافة محور", "`tf.expand_dims(x, 1)` / `x[:, None]`", "`x.unsqueeze(1)` / `x[:, None]`"),
                   ("خلط dtype", "خطأ", "ترقية تلقائية"), ("الجهاز", "تلقائي؛ `with tf.device`", "`x.to(device)` صريح"), ("معلمة قابلة للاشتقاق", "`tf.Variable`", "`requires_grad=True` / `nn.Parameter`"), ("إلى NumPy", "`x.numpy()`", "`x.detach().cpu().numpy()`")],
                  ["rtl", "code", "code"])
    intuition("قاعدة الجهاز في PyTorch: **البيانات والنموذج على نفس الجهاز**. النمط القياسي: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`، ثم `model.to(device)` مرة واحدة، و`xb, yb = xb.to(device), yb.to(device)` داخل الحلقة لكل دفعة.")
    debugging_note("`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` — أشهر خطأ PyTorch على GPU. أحد الطرفين (عادةً دفعة بيانات أو موتر أنشأته داخل forward بـ `torch.zeros`) بقي على CPU. أضف `.to(device)` أو أنشئه بـ `device=x.device`.")
    common_mistake("`x.view(...)` على موتر غير متصل في الذاكرة (بعد `transpose` مثلًا) يرفع `RuntimeError: view size is not compatible`. استخدم `reshape` (ينسخ عند الحاجة) أو `.contiguous().view(...)`.")
    quiz("pt.tensors", [
        Q("`torch.tensor([1, 2]).dtype`", ["int32", "int64", "float32"], 1, "بخلاف TensorFlow."),
        Q("`x.mean(dim=0)` يقابل في NumPy", ["`x.mean(axis=0)`", "`x.mean(axis=1)`", "`x.sum()`"], 0, "dim = axis."),
        Q("موتر على cuda + موتر على cpu:", ["يُنقل تلقائيًا", "RuntimeError: same device", "يعمل ببطء"], 1, "صريح."),
        Q("`torch.from_numpy(a)`…", ["ينسخ a", "يشارك ذاكرة a", "يحوّل إلى float32"], 1, "بلا نسخ."),
    ])
    takeaway("torch.Tensor = شكل + dtype + جهاز + requires_grad. int64 افتراضي، ترقية تلقائية، dim= بدل axis=، .to(device) صريح لكل شيء، .detach().cpu().numpy() للخروج. المفهوم واحد والواجهة مختلفة.")
    lesson_footer(LESSON, ["سبع طرق إنشاء وخصائص.", "الفروق الأربعة عن TensorFlow.", "قاعدة الجهاز الواحد."])
