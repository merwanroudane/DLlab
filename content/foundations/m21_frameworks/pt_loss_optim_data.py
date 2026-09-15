import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.pytorch.loss_optim_data",
    title_ar="الخسائر والمحسّنات والبيانات: nn.*Loss وtorch.optim وDataset وDataLoader",
    title_en="Losses, Optimizers & Data: nn.*Loss, torch.optim, Dataset & DataLoader",
    module="foundations.frameworks",
    parent="foundations.frameworks.pytorch",
    order=23,
    prerequisites=["foundations.frameworks.pytorch.nn_module", "foundations.loss.loss_activation_compatibility", "foundations.optim.rmsprop_adam", "foundations.frameworks.tensorflow.tf_data"],
    objectives_ar=["اختيار الخسارة الصحيحة في PyTorch بحسب المهمة وشكل/نوع الهدف (BCEWithLogits، CrossEntropy، MSE).", "إنشاء محسّن من torch.optim بمعلمات النموذج وفهم step وzero_grad وparam_groups.", "Dataset (len/getitem) وDataLoader (batch_size, shuffle) وما يعيده كل تكرار."],
    terms=["loss", "optimizer", "batch", "batch_size", "cross_entropy"],
    difficulty="intermediate",
    summary_ar="الخسائر كائنات تأخذ (input, target) بأشكال محددة: BCEWithLogitsLoss (n,1) float؛ CrossEntropyLoss (n,K) logits مع (n,) int64؛ MSELoss نفس الشكل. optim.SGD/Adam(model.parameters(), lr). Dataset يعرّف __len__/__getitem__؛ DataLoader يجمّع ويخلط.",
)

CODE = '''import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
torch.manual_seed(0)

# --- الخسائر: كل واحدة لها شكل ونوع هدف محدد ---
logits_bin = torch.randn(4, 1); y_bin = torch.tensor([[1.], [0.], [1.], [0.]])            # (4,1) float
logits_mc  = torch.randn(4, 3); y_mc  = torch.tensor([0, 2, 1, 2])                        # (4,3) logits + (4,) int64
pred_reg   = torch.randn(4, 1); y_reg = torch.randn(4, 1)
print("BCEWithLogitsLoss:", round(nn.BCEWithLogitsLoss()(logits_bin, y_bin).item(), 4), "| CrossEntropyLoss:", round(nn.CrossEntropyLoss()(logits_mc, y_mc).item(), 4), "| MSELoss:", round(nn.MSELoss()(pred_reg, y_reg).item(), 4))
try:
    nn.CrossEntropyLoss()(logits_mc, y_mc.float())                 # هدف float بدل int64
except RuntimeError as e:
    print("CrossEntropy with float target ->", str(e)[:80])
try:
    nn.BCEWithLogitsLoss()(logits_bin.squeeze(1), y_bin)           # (4,) مقابل (4,1)
except ValueError as e:
    print("BCE shape mismatch ->", str(e)[:80])
import warnings; warnings.simplefilter("ignore")
print("MSE (4,) vs (4,1) broadcasts silently ->", round(nn.MSELoss()(pred_reg.squeeze(1), y_reg).item(), 4), "(wrong! averages a 4x4 matrix)")

# --- المحسّنات: يستلم المعلمات ويملك step/zero_grad ---
model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
print("optimizer:", type(opt).__name__, "| lr =", opt.param_groups[0]["lr"], "| n param tensors:", len(opt.param_groups[0]["params"]))
opt.param_groups[0]["lr"] = 5e-4                                   # تغيير η يدويًا (ما تفعله الجداول)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)  # جدول: η × 0.5 كل حقبتين
lrs = []
for epoch in range(5):
    opt.step(); sched.step(); lrs.append(round(opt.param_groups[0]["lr"], 6))        # sched.step() مرة لكل حقبة
print("lr per epoch with StepLR:", lrs)

# --- Dataset و DataLoader ---
class MyData(Dataset):                                             # أي مصدر: مصفوفات، ملفات، صور...
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)                          # عدد الأمثلة
    def __getitem__(self, i): return self.X[i], self.y[i]          # مثال واحد (بلا بُعد دفعة)
X = torch.arange(10, dtype=torch.float32).unsqueeze(1); y = torch.arange(10)
ds = MyData(X, y)
print("len(ds) =", len(ds), "| ds[3] =", (ds[3][0].tolist(), ds[3][1].item()))
loader = DataLoader(ds, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
print("len(loader) =", len(loader), "batches (10/4 -> 3, last has 2)")
for epoch in range(2):
    print(f"epoch {epoch+1}:", [yb.tolist() for xb, yb in loader], "| xb.shape of first =", tuple(next(iter(loader))[0].shape))
print("drop_last=True ->", len(DataLoader(ds, batch_size=4, drop_last=True)), "batches")
print("TensorDataset shortcut:", len(TensorDataset(X, y)), "| val loader: shuffle=False")'''


def render() -> None:
    lesson_header(LESSON)
    h2("الخسائر", "Losses")
    definition("خسائر PyTorch **كائنات** في `torch.nn` تُستدعى بـ `loss_fn(input, target)` وتعيد موترًا عدديًا (متوسط الدفعة افتراضيًا، `reduction='mean'`). كل واحدة تفرض **شكل ونوع** الهدف — ومخالفتها إما خطأ صريح أو **بث صامت** خاطئ.")
    compare_table(["المهمة", "الخسارة", "المدخل (input)", "الهدف (target)", "المقابل في Keras"],
                  [("ثنائي", "`nn.BCEWithLogitsLoss()`", "logits (n, 1) float", "(n, 1) float 0/1", "sigmoid + binary_crossentropy"), ("متعدد الفئات", "`nn.CrossEntropyLoss()`", "logits (n, K) float", "(n,) **int64** بقيم 0..K−1", "softmax + sparse_categorical_crossentropy"),
                   ("انحدار", "`nn.MSELoss()` / `nn.L1Loss()` / `nn.HuberLoss()`", "(n, 1) float", "(n, 1) float **نفس الشكل**", "mse / mae / huber"), ("ثنائي باحتمالات", "`nn.BCELoss()` (بعد sigmoid)", "(n, 1) في (0,1)", "(n, 1) float", "غير مستحب: استخدم WithLogits")],
                  ["rtl", "code", "code", "code", "rtl"])
    intuition("`WithLogits` و`CrossEntropyLoss` تأخذان **logits** وتطبّقان sigmoid/softmax داخليًا بصيغة مستقرة عدديًا (وحدة 13). لا تضع تنشيط مخرج في `forward` للتصنيف في PyTorch.")
    code_lab(CodeLab(
        key="pt_loss_optim_data", title_ar="ثلاث خسائر بأشكالها وأخطائها، محسّن وجدول، Dataset وDataLoader", code=CODE, level="B",
        before=Before(goal_ar="لمس المكوّنات الثلاثة التي تغذّي حلقة التدريب: خسائر (مع خطأين صريحين وبث صامت)، محسّن بمعلماته وجدول معدل تعلم، وDataset/DataLoader مع ما يعيده كل تكرار.", stage_ar="PyTorch: الخسارة، المحسّن، البيانات.",
                      inputs_ar="موترات صغيرة و10 أمثلة مرقّمة.", expected_ar="ثلاث قيم خسارة؛ خطأ CrossEntropy مع float؛ خطأ شكل BCE؛ MSE يبث بصمت؛ η يتناقص بالجدول؛ 3 دفعات بترتيب مختلف كل حقبة.",
                      objects_ar="`nn.BCEWithLogitsLoss`, `nn.CrossEntropyLoss`, `nn.MSELoss`, `torch.optim.Adam`, `StepLR`, `Dataset`, `DataLoader`, `TensorDataset`."),
        explain=[("6-9", "ثلاث خسائر بأشكالها الصحيحة. لاحظ y_mc أعداد صحيحة (int64) بلا بُعد ثانٍ."), ("10-17", "خطآن صريحان مفيدان: CrossEntropy ترفض float؛ BCE ترفض (4,) مقابل (4,1)."),
                 ("18-19", "**الخطر**: MSE مع (4,) و(4,1) **يبث** إلى (4,4) ويعطي رقمًا معقولًا المظهر وخاطئًا. `unsqueeze(1)` للهدف دائمًا."), ("22-25", "المحسّن يستلم `model.parameters()` وη و`weight_decay` (L2، وحدة 20). `param_groups` يكشف η ويسمح بتغييره."),
                 ("26-30", "جدول معدل تعلم: `sched.step()` مرة كل **حقبة** (لا كل دفعة) → η × 0.5 كل حقبتين."), ("33-39", "`Dataset`: `__len__` و`__getitem__` يعيد **مثالًا واحدًا**. هذا كل ما يحتاجه DataLoader."),
                 ("40-45", "`DataLoader` يجمّع (يضيف بُعد الدفعة)، يخلط كل حقبة، ويعطي آخر دفعة ناقصة ما لم `drop_last`. `TensorDataset` اختصار للمصفوفات؛ التحقق بلا خلط.")],
        run=run_printed(CODE),
        after_ar="- أشكال الهدف الثلاثة تختلف: (n,1) float للثنائي، (n,) int64 للمتعدد، (n,1) float للانحدار. احفظها.\n- الجدول يعمل على `param_groups`؛ وهذا ما يمكنك فعله يدويًا أيضًا.\n- `DataLoader` هو نظير `tf.data`: نفس الأفكار (خلط قبل التجميع، آخر دفعة ناقصة).",
    ))
    h2("المحسّنات", "Optimizers")
    compare_table(["المحسّن", "الاستدعاء", "أهم المعاملات"],
                  [("SGD (+momentum)", "`torch.optim.SGD(params, lr=0.01, momentum=0.9)`", "`lr`, `momentum`, `nesterov`, `weight_decay`"), ("Adam", "`torch.optim.Adam(params, lr=1e-3)`", "`lr`, `betas=(0.9, 0.999)`, `weight_decay`"), ("AdamW", "`torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)`", "اضمحلال أوزان مفصول (أفضل مع Adam)"), ("RMSprop", "`torch.optim.RMSprop(params, lr=1e-3)`", "`alpha`"),
                   ("جداول", "`lr_scheduler.StepLR / ReduceLROnPlateau / CosineAnnealingLR`", "`sched.step()` كل حقبة (ReduceLROnPlateau يأخذ `val_loss`)")],
                  ["rtl", "code", "rtl"])
    debugging_note("خسارة لا تتغير أبدًا؟ تحقق: (1) `opt = SGD(model.parameters(), ...)` أُنشئ **بعد** بناء النموذج وعلى نفس النموذج، (2) `opt.step()` يُستدعى، (3) `loss.backward()` قبله، (4) المعلمات مسجّلة (درس nn.Module). و`torch.optim.lr_scheduler` بلا `sched.step()` = η ثابت.")
    common_mistake("`y = torch.tensor([0, 1, 1]).float()` ثم `CrossEntropyLoss` → خطأ dtype. أو `y.unsqueeze(1)` مع CrossEntropy → خطأ شكل. القاعدة: CrossEntropy تريد (n,) int64؛ BCEWithLogits تريد (n,1) float.")
    quiz("pt.lod", [
        Q("هدف `CrossEntropyLoss`:", ["(n, K) one-hot float", "(n,) int64", "(n, 1) float"], 1, "فئات كأعداد."),
        Q("`MSELoss(pred (n,1), y (n,))`…", ["خطأ", "بث صامت إلى (n,n)", "يعمل صحيحًا"], 1, "unsqueeze."),
        Q("`sched.step()` يُستدعى…", ["كل دفعة", "كل حقبة (لمعظم الجداول)", "مرة واحدة"], 1, "حقبة."),
        Q("`__getitem__` في Dataset يعيد…", ["دفعة", "مثالًا واحدًا", "كل البيانات"], 1, "DataLoader يجمّع."),
    ])
    takeaway("الخسارة تفرض شكل الهدف ونوعه (احذر البث الصامت). المحسّن يستلم parameters() ويملك step/zero_grad/param_groups؛ الجدول يستدعي step كل حقبة. Dataset = len + getitem لمثال واحد؛ DataLoader = دفعات + خلط.")
    lesson_footer(LESSON, ["جدول الخسائر بالأشكال.", "المحسّن والجدول.", "Dataset وDataLoader."])
