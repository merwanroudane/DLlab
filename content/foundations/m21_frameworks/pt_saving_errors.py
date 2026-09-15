import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagnostics import Problem, problem_card
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.pytorch.saving_errors",
    title_ar="الحفظ والتحميل (state_dict) وأخطاء PyTorch الشائعة",
    title_en="Saving/Loading (state_dict) & Common PyTorch Errors",
    module="foundations.frameworks",
    parent="foundations.frameworks.pytorch",
    order=27,
    prerequisites=["foundations.frameworks.pytorch.train_eval", "foundations.frameworks.keras.errors"],
    objectives_ar=["حفظ وتحميل state_dict للنموذج والمحسّن، ونقطة حفظ كاملة لاستئناف التدريب.", "قراءة أخطاء PyTorch: عدم تطابق الجهاز، الشكل، dtype؛ التراكم غير المقصود؛ نسيان train/eval؛ OOM.", "قائمة فحص قبل كل حلقة تدريب."],
    terms=["parameter", "shape", "dtype"],
    difficulty="intermediate",
    summary_ar="torch.save(model.state_dict(), 'm.pt') ثم model = Net(); model.load_state_dict(torch.load('m.pt')); model.eval(). نقطة حفظ = {model, optimizer, epoch}. الأخطاء: mat1 and mat2 shapes (شكل)، same dtype (نوع)، same device (جهاز)، backward twice، does not require grad، size mismatch عند التحميل.",
)

CODE = '''import os, tempfile, torch, torch.nn as nn
torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1 = nn.Linear(4, 8); self.fc2 = nn.Linear(8, 1)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x)))

model = Net(); opt = torch.optim.Adam(model.parameters(), lr=1e-2)
x = torch.randn(16, 4); y = torch.randn(16, 1)
for step in range(5):                                                        # تدريب قصير
    opt.zero_grad(); loss = nn.MSELoss()(model(x), y); loss.backward(); opt.step()

# 1) state_dict: قاموس {اسم المعلمة: موتر}
sd = model.state_dict()
print("state_dict keys:", list(sd.keys()))
print("shapes:", {k: tuple(v.shape) for k, v in sd.items()})
path = os.path.join(tempfile.mkdtemp(), "model.pt")
torch.save(sd, path)                                                         # الأوزان فقط (الطريقة الموصى بها)
print("saved", os.path.getsize(path), "bytes")

# 2) التحميل: ابنِ نفس البنية ثم حمّل الأوزان
fresh = Net(); fresh.load_state_dict(torch.load(path)); fresh.eval()
with torch.no_grad():
    print("same predictions after reload:", torch.allclose(fresh(x), model(x)))

# 3) نقطة حفظ كاملة لاستئناف التدريب (النموذج + المحسّن + الحقبة)
ckpt = {"epoch": 5, "model": model.state_dict(), "optimizer": opt.state_dict(), "loss": loss.item()}
torch.save(ckpt, path.replace("model.pt", "ckpt.pt"))
loaded = torch.load(path.replace("model.pt", "ckpt.pt"))
model2 = Net(); opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
model2.load_state_dict(loaded["model"]); opt2.load_state_dict(loaded["optimizer"])
print("resume from epoch", loaded["epoch"], "| optimizer state restored (Adam moments):", len(opt2.state_dict()["state"]) > 0)

# 4) خطأ التحميل ببنية مختلفة
class Bigger(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1 = nn.Linear(4, 16); self.fc2 = nn.Linear(16, 1)
try:
    Bigger().load_state_dict(torch.load(path))
except RuntimeError as e:
    print("load into different architecture ->", str(e).splitlines()[1].strip()[:95])
# 5) GPU -> CPU: map_location
sd_cpu = torch.load(path, map_location="cpu")                                # يحمّل أوزانًا حُفظت على GPU إلى CPU
print("map_location='cpu' ok:", all(v.device.type == "cpu" for v in sd_cpu.values()))'''


def render() -> None:
    lesson_header(LESSON)
    h2("الحفظ والتحميل", "Saving & loading")
    definition("**`state_dict`**: قاموس مرتّب من `{اسم المعلمة/المخزن: موتر}` لكل ما يملكه `nn.Module` (الأوزان، الانحيازات، متوسطات BN المتحركة). الطريقة القياسية للحفظ: `torch.save(model.state_dict(), path)`؛ وللتحميل: ابنِ **نفس البنية** ثم `model.load_state_dict(torch.load(path))` ثم `model.eval()`. المحسّن له `state_dict` أيضًا (متوسطات Adam) — احفظه لاستئناف التدريب.")
    code_lab(CodeLab(
        key="pt_save", title_ar="state_dict للنموذج والمحسّن، نقطة حفظ، وخطأ البنية المختلفة", code=CODE, level="B",
        before=Before(goal_ar="حفظ أوزان نموذج مدرّب، تحميلها في نسخة جديدة والتحقق من تطابق التنبؤات، حفظ نقطة كاملة (نموذج + محسّن + حقبة) واستئنافها، ورؤية خطأ التحميل في بنية مختلفة وحل مشكلة GPU→CPU.", stage_ar="PyTorch: الحفظ.",
                      inputs_ar="Net صغير مدرّب 5 خطوات.", expected_ar="مفاتيح fc1.weight/bias, fc2.weight/bias؛ تطابق التنبؤات True؛ استئناف من الحقبة 5 مع حالة Adam؛ خطأ size mismatch؛ map_location يعمل.",
                      objects_ar="`state_dict`, `torch.save`, `torch.load`, `load_state_dict`, `map_location`."),
        explain=[("14-19", "المفاتيح = أسماء الخصائص في `__init__` + `.weight/.bias`. الحفظ بصيغة pickle داخل zip؛ الامتداد `.pt` أو `.pth` عرف."), ("22-24", "التحميل يتطلب **الكود** (الصنف Net) — state_dict لا يحوي البنية. ثم `eval()`."),
                 ("27-32", "نقطة حفظ كاملة: قاموس تصممه أنت. حالة المحسّن ضرورية لاستئناف Adam بمتوسطاته لا من الصفر."), ("35-41", "بنية بأشكال مختلفة → `size mismatch for fc1.weight`. الرسالة تذكر الشكلين — قارنهما بـ summary/print."),
                 ("43-44", "أوزان حُفظت على GPU وتُحمَّل على آلة بلا GPU: `map_location='cpu'` وإلا خطأ CUDA.")],
        run=run_printed(CODE),
        after_ar="- Keras `.keras` يحفظ البنية والأوزان معًا؛ PyTorch القياسي يحفظ الأوزان فقط وتحتاج الكود. (`torch.save(model)` يحفظ الكائن كاملًا لكنه هش عند تغيير الكود — غير موصى به.)\n- لا تنسَ `model.eval()` بعد التحميل للاستدلال.\n- احفظ مع النموذج: النسخة، معلمات التحجيم، والعتبة (نفس نصيحة Keras).",
    ))
    compare_table(["المهمة", "PyTorch", "Keras"],
                  [("حفظ الأوزان", "`torch.save(model.state_dict(), 'm.pt')`", "`model.save_weights('m.weights.h5')`"), ("تحميل الأوزان", "`model = Net(); model.load_state_dict(torch.load('m.pt'))`", "`model = build(); model.load_weights(...)`"), ("بنية + أوزان", "الكود + state_dict (أو `torch.save(model)` هش)", "`model.save('m.keras')`"),
                   ("استئناف التدريب", "قاموس {model, optimizer, epoch}", "`.keras` يحوي حالة المحسّن"), ("النشر", "`torch.jit.script` / `torch.export` / ONNX (تعميق)", "`model.export()` SavedModel")],
                  ["rtl", "code", "code"])
    h2("أخطاء PyTorch الشائعة", "Common PyTorch errors")
    compare_table(["الرسالة (السطر المهم)", "العائلة", "السبب", "الحل"],
                  [("`mat1 and mat2 shapes cannot be multiplied (3x5 and 4x8)`", "شكل", "المدخل 5 خصائص وLinear يتوقع 4 (in_features)", "طابق `in_features` مع `x.shape[1]`؛ Flatten للصور"),
                   ("`mat1 and mat2 must have the same dtype, but got Double and Float`", "dtype", "بيانات NumPy float64 (Double) مع أوزان float32", "`x.float()` أو `torch.tensor(a, dtype=torch.float32)`"),
                   ("`… got Long and Float`", "dtype", "مدخل أعداد صحيحة", "`x.float()`"),
                   ("`Expected all tensors to be on the same device, but found … cuda:0 and cpu`", "جهاز", "الدفعة أو موتر داخلي بقي على CPU", "`xb.to(device)`، `torch.zeros(..., device=x.device)`"),
                   ("`Torch not compiled with CUDA enabled`", "جهاز", "`.to('cuda')` على تثبيت CPU", "`device = 'cuda' if torch.cuda.is_available() else 'cpu'`"),
                   ("`expected target dtype to be Long or Byte, but got Float`", "dtype", "هدف CrossEntropy كسور", "`y.long()`"),
                   ("`Target 5 is out of bounds`", "شكل/بيانات", "فئة ≥ عدد المخرجات", "وحدات المخرج = عدد الفئات؛ الفئات من 0"),
                   ("`Trying to backward through the graph a second time`", "Autograd", "backward مرتين على نفس الرسم (خسارة محفوظة عبر الدفعات)", "أعد الأمامي كل دفعة؛ لا تجمع الخسائر كموترات"),
                   ("`element 0 of tensors does not require grad`", "Autograd", "الخسارة مفصولة عن المعلمات (detach/no_grad/NumPy)", "احسب الخسارة من مخرج النموذج مباشرة داخل الرسم"),
                   ("`a Tensor with 3 elements cannot be converted to Scalar`", "شكل", "`.item()` على موتر غير عددي", "`.mean().item()` أو `.tolist()`"),
                   ("`CUDA out of memory. Tried to allocate …`", "جهاز", "ذاكرة GPU ممتلئة", "قلّل batch_size، `.item()` عند التجميع، `no_grad` للتحقق، `torch.cuda.empty_cache()`")],
                  ["code", "rtl", "rtl", "rtl"])
    h2("الأخطاء الصامتة الثلاثة", "The three silent errors")
    problem_card(Problem(
        key="pt_silent", name_ar="الأخطاء الصامتة في حلقة PyTorch", name_en="Silent PyTorch loop errors",
        description_ar="ثلاثة أخطاء لا ترفع رسالة: تراكم التدرجات (نسيان zero_grad)، نسيان الوضع (train/eval)، وخسارة تُبث بصمت (أشكال الهدف).",
        symptoms_ar=["الخسارة تتذبذب أو تنفجر بعد حقب قليلة (تراكم).", "نتائج التحقق تتقلب بين استدعاءين أو أقل من المتوقع (Dropout/BN في train).", "خسارة «معقولة» لكن النموذج لا يتعلم شيئًا مفيدًا (بث (n,) مقابل (n,1))."],
        sees_ar=["سجل تدريب طبيعي المظهر.", "`UserWarning: Using a target size (torch.Size([32])) that is different to the input size (torch.Size([32, 1]))` — التحذير الوحيد، وسهل تجاهله."],
        possible_causes_ar=["نسخ قالب ناقص.", "إعادة ترتيب أسطر الحلقة.", "هدف من pandas بشكل (n,)."], root_causes_ar=["PyTorch يثق بك: لا يفرض ترتيب الحلقة ولا الوضع ولا الأشكال المتطابقة."],
        diagnosis_ar=["اطبع معيار التدرج كل دفعة: ينمو خطيًا؟ → تراكم.", "اطبع `model.training` قبل التحقق وبعده.", "اطبع `out.shape, yb.shape` قبل الخسارة: يجب أن يتطابقا (أو (n,K) مع (n,) لـ CrossEntropy فقط)."],
        evidence_ar=["‖grad‖ يزداد مع الخطوات.", "`model.training == True` أثناء التحقق.", "تحذير target size."],
        fixes_ar=["`optimizer.zero_grad()` أول سطر في حلقة الدفعات.", "`model.train()` بداية كل حقبة، `model.eval()` + `no_grad` للتحقق.", "`y.unsqueeze(1)` / `y.float()` / `y.long()` بحسب الخسارة — واجعل التحذيرات أخطاء أثناء التطوير: `warnings.simplefilter('error')`."],
        checklist_ar=["zero_grad قبل backward.", "train() في بداية الحقبة وeval() للتحقق.", "أشكال out وyb متطابقة (أو (n,K)/(n,) للمتعدد).", "dtype: float32 للمدخل، long لفئات CrossEntropy.", "كل شيء على نفس الجهاز.", ".item() عند التجميع."],
        challenge=[Q("‖grad‖ ينمو خطيًا مع الدفعات:", ["η كبير", "نسيان zero_grad", "نسيان eval"], 1, "تراكم."), Q("تنبؤات مختلفة لنفس المدخل:", ["نسيان eval (Dropout)", "نسيان zero_grad", "dtype"], 0, "عشوائية Dropout.")],
    ))
    intuition("PyTorch يعطيك أخطاء **صريحة** ممتازة للأشكال والأنواع والأجهزة — وأخطاء **صامتة** خطيرة لما يخص ترتيب الحلقة والوضع والبث. الأولى تقرؤها، والثانية تمنعها بقائمة الفحص.")
    debugging_note("رسائل الشكل في PyTorch تذكر الشكلين دائمًا (`3x5 and 4x8`): الأول مدخلك (batch × features)، الثاني الوزن منقولًا (in × out). الرقم الداخلي (5 مقابل 4) هو الخلاف.")
    practical_note("قبل أول حقبة: مرّر **دفعة واحدة** يدويًا واطبع `out.shape`, `yb.shape`, `loss.item()`، ثم `loss.backward()` واطبع معيار التدرج. دقيقة واحدة تكشف 90% من الأخطاء قبل ساعات التدريب.")
    common_mistake("`torch.save(model)` (الكائن كاملًا) ثم تغيير اسم الصنف أو الملف → `AttributeError: Can't get attribute 'Net'` عند التحميل. احفظ `state_dict` وأبقِ الكود.")
    quiz("pt.save", [
        Q("`torch.save(model.state_dict())` يحفظ…", ["البنية والأوزان", "الأوزان (والمخازن) فقط", "الكود"], 1, "تحتاج الصنف."),
        Q("`mat1 and mat2 shapes cannot be multiplied (3x5 and 4x8)`:", ["batch 3 خطأ", "in_features=4 لكن المدخل 5 خصائص", "المخرج 8 خطأ"], 1, "البعد الداخلي."),
        Q("`got Double and Float`:", ["أوزان float64", "بيانات NumPy float64 مع أوزان float32", "GPU"], 1, "`.float()`."),
        Q("لاستئناف تدريب Adam تحتاج حفظ…", ["النموذج فقط", "النموذج والمحسّن والحقبة", "الخسارة فقط"], 1, "حالة المحسّن."),
    ])
    takeaway("state_dict للنموذج والمحسّن؛ الكود + الأوزان = النموذج. الأخطاء الصريحة: طابق الشكلين/النوعين/الجهازين من الرسالة. الصامتة: zero_grad، train/eval، أشكال الهدف — قائمة الفحص قبل الحلقة.")
    lesson_footer(LESSON, ["الحفظ الثلاثي: أوزان، نقطة حفظ، map_location.", "جدول الأخطاء الصريحة.", "الأخطاء الصامتة الثلاثة وقائمة الفحص."])
