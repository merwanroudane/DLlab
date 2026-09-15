import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table, good_vs_bad
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.pytorch.nn_module",
    title_ar="nn.Module: __init__ وforward وnn.Linear والتنشيطات والمعلمات",
    title_en="nn.Module: __init__, forward, nn.Linear, Activations & Parameters",
    module="foundations.frameworks",
    parent="foundations.frameworks.pytorch",
    order=22,
    prerequisites=["foundations.frameworks.pytorch.autograd", "foundations.frameworks.keras.layers_models", "foundations.architecture.parameter_count"],
    objectives_ar=["كتابة نموذج كصنف يرث nn.Module: الطبقات في __init__، الحساب في forward.", "nn.Linear وأشكال أوزانه (out, in)، والتنشيطات كوحدات (nn.ReLU) أو دوال (F.relu).", "model.parameters() وnamed_parameters() وعدّ المعلمات وطباعة النموذج، وnn.Sequential للبسيط."],
    terms=["model", "parameter", "shape"],
    labs=["labs.parameter_counter"],
    difficulty="intermediate",
    summary_ar="class Net(nn.Module): __init__ يعرّف الطبقات (nn.Linear...)، forward يصف الحساب. المعلمات تُسجَّل تلقائيًا عند إسناد الوحدات إلى self. nn.Linear(in, out).weight بشكل (out, in). print(model) وsum(p.numel()) بديل summary.",
)

CODE = '''import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self, n_in=4, hidden=8, n_out=3):
        super().__init__()                                  # إلزامي: يهيّئ آلية تسجيل المعلمات
        self.fc1 = nn.Linear(n_in, hidden)                  # W1 (8, 4) + b1 (8,)  -> 40 معلمة
        self.act = nn.ReLU()                                # وحدة بلا معلمات
        self.fc2 = nn.Linear(hidden, n_out)                 # W2 (3, 8) + b2 (3,)  -> 27 معلمة

    def forward(self, x):                                   # الحساب: x (batch, 4) -> logits (batch, 3)
        h = self.act(self.fc1(x))
        return self.fc2(h)                                  # logits بلا softmax (الخسارة تتولاه)

model = MLP()
print(model)                                                # بديل summary: الوحدات المسجلة
print("n params:", sum(p.numel() for p in model.parameters()), "(hand: 40 + 27 = 67)")
for name, p in model.named_parameters():
    print(f"  {name:<10} shape={tuple(p.shape)!s:<8} requires_grad={p.requires_grad}")

x = torch.randn(5, 4)                                       # دفعة من 5
out = model(x)                                              # يستدعي forward عبر __call__ (لا تستدعِ forward مباشرة)
print("out.shape:", tuple(out.shape), "| grad_fn:", type(out.grad_fn).__name__)
print("probabilities (softmax on logits) row 0:", F.softmax(out, dim=1)[0].detach().numpy().round(3), "sum =", F.softmax(out, dim=1)[0].sum().item())

# nn.Sequential: نفس الشبكة بلا صنف (للتسلسلات البسيطة)
seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
print(seq); print("seq params:", sum(p.numel() for p in seq.parameters()), "| access a layer: seq[0].weight.shape =", tuple(seq[0].weight.shape))

# خطأ شائع: طبقات في قائمة بايثون لا تُسجَّل
class Broken(nn.Module):
    def __init__(self):
        super().__init__(); self.layers = [nn.Linear(4, 8), nn.Linear(8, 3)]     # قائمة عادية!
    def forward(self, x):
        return self.layers[1](torch.relu(self.layers[0](x)))
print("Broken params:", sum(p.numel() for p in Broken().parameters()), "<- optimizer would train nothing; use nn.ModuleList / nn.Sequential")'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`nn.Module`**: الصنف الأساس لكل طبقة ونموذج في PyTorch. تكتب صنفًا يرثه: في **`__init__`** تنشئ الوحدات الفرعية (`nn.Linear`، `nn.ReLU`، …) وتسندها إلى `self` فتُسجَّل معلماتها تلقائيًا؛ في **`forward(x)`** تصف الحساب بأي بايثون تريده. `model(x)` يستدعي `forward` (عبر `__call__` الذي يضيف الخطافات). الطبقة **هي** نموذج صغير، والنموذج طبقة كبيرة — نفس الصنف.")
    why("في Keras، الافتراضي قائمة طبقات (Sequential) والصنف تعميق. في PyTorch العكس: الصنف هو الطريقة القياسية لأن `forward` بايثون حر — شروط، حلقات، تفرّعات — وهذا ما يفضّله الباحثون. `nn.Sequential` موجود للحالات البسيطة فقط.")
    code_lab(CodeLab(
        key="pt_module", title_ar="MLP كصنف، طباعته، معلماته، استدعاؤه، وSequential، وخطأ القائمة", code=CODE, level="B",
        before=Before(goal_ar="كتابة نفس MLP (4→8→3) الذي بنيته في Keras كصنف PyTorch، فحص معلماته وأشكالها، استدعاؤه على دفعة، ثم النسخة Sequential، ثم فخ شائع يُخفي المعلمات عن المحسّن.", stage_ar="PyTorch: nn.Module.",
                      inputs_ar="دفعة (5, 4).", expected_ar="طباعة النموذج بوحداته؛ 67 معلمة بأشكال (8,4),(8,),(3,8),(3,)؛ مخرج (5, 3) بـ grad_fn؛ softmax يجمع إلى 1؛ Broken بـ 0 معلمة.",
                      objects_ar="`nn.Module`, `nn.Linear`, `nn.ReLU`, `F.softmax`, `nn.Sequential`, `named_parameters`."),
        explain=[("4-9", "`__init__`: `super().__init__()` أولًا، ثم الوحدات كخصائص. `nn.Linear(in, out)` تنشئ `weight (out, in)` و`bias (out,)` وتهيّئهما."), ("11-13", "`forward`: مخرج بلا softmax — `CrossEntropyLoss` تطبّقه داخليًا (وحدة 13: من logits)."),
                 ("15-19", "`print(model)` يعرض الشجرة (لا الأشكال ولا العدّ — لهذا نعدّ بـ `numel`). `named_parameters` = أسماء بنمط `layer.weight`."), ("21-24", "استدعاء `model(x)` لا `model.forward(x)`. المخرج يحمل `grad_fn`: الرسم مبني من x إلى out."),
                 ("27-28", "`nn.Sequential` يقبل الوحدات بترتيبها ويُفهرس كقائمة."), ("31-36", "**فخ**: قائمة بايثون عادية لا تُسجَّل؛ `parameters()` فارغ؛ المحسّن لا يحدّث شيئًا **بصمت**. الحل `nn.ModuleList` أو `nn.Sequential`.")],
        run=run_printed(CODE),
        after_ar="- الأشكال (8, 4) لا (4, 8): PyTorch `Linear` يحسب `x @ W.T + b`. عند نقل أوزان من Keras انقلها منقولة.\n- `requires_grad=True` لكل معلمة تلقائيًا: `nn.Parameter` هو موتر ورقة مراقَب.\n- الفخ الأخير من أكثر الأخطاء الصامتة شيوعًا في PyTorch: خسارة لا تنخفض وكل شيء «يبدو» صحيحًا.",
    ))
    h2("المكوّنات", "The pieces")
    compare_table(["المكوّن", "PyTorch", "المقابل في Keras", "ملاحظة"],
                  [("طبقة كثيفة", "`nn.Linear(in, out)`", "`Dense(units)`", "PyTorch يحتاج `in` صريحًا؛ Keras يستنتجه"), ("تنشيط كوحدة", "`nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()`", "`activation='relu'`", "بلا معلمات؛ تظهر في print(model)"),
                   ("تنشيط كدالة", "`F.relu(x)`, `torch.sigmoid(x)`", "`keras.activations.relu`", "داخل forward مباشرة"), ("Dropout / BN", "`nn.Dropout(p)`, `nn.BatchNorm1d(f)`", "`Dropout(p)`, `BatchNormalization()`", "سلوكها يعتمد على train/eval"),
                   ("قائمة وحدات", "`nn.ModuleList([...])`", "قائمة بايثون تكفي", "إلزامي للتسجيل"), ("تسلسل", "`nn.Sequential(...)`", "`Sequential([...])`", "للبسيط فقط"),
                   ("المعلمات", "`model.parameters()` / `named_parameters()`", "`model.weights` / `trainable_weights`", "المحسّن يستلم `parameters()`"), ("ملخص", "`print(model)` + `sum(p.numel())`", "`model.summary()`", "لا أشكال مخرجات في print")],
                  ["rtl", "code", "code", "rtl"])
    intuition("`__init__` = «ما لدي من أدوات» (الطبقات بمعلماتها)، `forward` = «كيف أستخدمها» على دفعة. الفصل يسمح باستخدام نفس الطبقة مرتين (أوزان مشتركة) أو بمسار مشروط — أشياء تحتاج Subclassing في Keras.")
    good_vs_bad("logits من forward", "أخرج logits وادفع بها إلى `nn.CrossEntropyLoss` (تطبّق log-softmax داخليًا، مستقرة عدديًا).", "softmax داخل forward + CrossEntropyLoss", "تطبيق softmax مرتين: خسارة خاطئة وتدرجات ضعيفة — بلا رسالة خطأ.",
                good_code="def forward(self, x):\n    return self.fc2(F.relu(self.fc1(x)))   # logits\nloss = nn.CrossEntropyLoss()(model(x), y_int)", bad_code="def forward(self, x):\n    return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)\nloss = nn.CrossEntropyLoss()(model(x), y_int)   # double softmax",
                verdict_ar="القاعدة نفسها في Keras مع `from_logits=True`؛ في PyTorch هي الافتراضي: الخسائر تأخذ logits.")
    debugging_note("`print(model)` لا يعطي أشكال المخرجات. لتتبع الأشكال طبقةً طبقة مرّر دفعة وهمية واطبع: `x = torch.zeros(1, 4); for m in model.children(): x = m(x); print(type(m).__name__, tuple(x.shape))`. (حزمة `torchinfo` تعطي ملخصًا شبيهًا بـ Keras — اختياري.)")
    common_mistake("نسيان `super().__init__()`: `AttributeError: cannot assign module before Module.__init__() call`. أول سطر في `__init__` دائمًا.")
    quiz("pt.module", [
        Q("`nn.Linear(4, 8).weight.shape`", ["(4, 8)", "(8, 4)", "(8,)"], 1, "(out, in)."),
        Q("الحساب يُكتب في…", ["`__init__`", "`forward`", "`__call__`"], 1, "init للأدوات."),
        Q("طبقات في قائمة بايثون عادية داخل Module…", ["تُسجَّل", "لا تُسجَّل: المحسّن لا يراها", "تُسجَّل بأسماء عشوائية"], 1, "ModuleList."),
        Q("مخرج forward للتصنيف المتعدد مع CrossEntropyLoss:", ["softmax", "logits", "argmax"], 1, "الخسارة تتولى softmax."),
    ])
    takeaway("صنف يرث nn.Module: __init__ للوحدات (تسجيل تلقائي عبر self)، forward للحساب، model(x) للاستدعاء. Linear (out, in). ModuleList للقوائم. logits للخسارة. print + numel بديل summary.")
    lesson_footer(LESSON, ["الصنف والوحدات والمعلمات.", "جدول المكوّنات مقابل Keras.", "فخ القائمة وsoftmax المزدوج."])
