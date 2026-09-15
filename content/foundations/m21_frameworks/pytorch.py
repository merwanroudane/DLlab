import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.pytorch",
    title_ar="ما هو PyTorch؟ خريطة المسار الكامل",
    title_en="What is PyTorch? The Full Track Map",
    module="foundations.frameworks",
    order=19,
    prerequisites=["foundations.frameworks.tensorflow.tensorboard_errors"],
    objectives_ar=["فهم PyTorch كإطار تعلم عميق: torch.Tensor، Autograd، nn.Module، الخسائر، المحسّنات، Dataset/DataLoader، حلقة تدريب صريحة، الأجهزة، الحفظ، الاستدلال.", "التثبيت والاستيراد والتحقق من النسخة والجهاز.", "خريطة الصفحات الفرعية."],
    terms=["tensor", "gradient", "model", "optimizer"],
    difficulty="intermediate",
    summary_ar="PyTorch: موترات بـ Autograd، نماذج كأصناف ترث nn.Module مع forward، خسارة ومحسّن ككائنات، DataLoader للدفعات، وحلقة تدريب تكتبها أنت سطرًا سطرًا: zero_grad → forward → loss → backward → step.",
)

CODE = '''import torch, torch.nn as nn
print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "| device:", "cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# البيانات: y = 3x1 - 2x2 + 1 + ضوضاء (نفس مثال Keras الأول)
X = torch.randn(200, 2); y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + 0.1 * torch.randn(200)
y = y.unsqueeze(1)                                            # (200,) -> (200, 1) ليطابق مخرج الطبقة

model = nn.Linear(2, 1)                                       # 1) الطبقة/النموذج: z = xWᵀ + b
loss_fn = nn.MSELoss()                                        # 2) الخسارة ككائن
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)       # 3) المحسّن يستلم المعلمات

for epoch in range(20):                                       # 4) حلقة التدريب — أنت تكتبها
    for i in range(0, 200, 20):                               #    دفعات من 20
        xb, yb = X[i:i+20], y[i:i+20]
        optimizer.zero_grad()                                 #    (أ) صفّر التدرجات المتراكمة
        y_hat = model(xb)                                     #    (ب) أمامي
        loss = loss_fn(y_hat, yb)                             #    (ج) خسارة
        loss.backward()                                       #    (د) خلفي: يملأ .grad لكل معلمة
        optimizer.step()                                      #    (هـ) تحديث
print("learned W :", model.weight.data.numpy().round(3), "(true: [3, -2])")
print("learned b :", model.bias.data.numpy().round(3), "(true: 1)")
print("final loss:", round(loss.item(), 5))
model.eval()
with torch.no_grad():                                         # 5) استدلال بلا تسجيل
    print("predict([1, 1]):", model(torch.tensor([[1.0, 1.0]])).numpy().round(3), "(true: 2)")'''


def _map_svg() -> str:
    items = [("torch.Tensor", "#E6F1FB"), ("Autograd", "#FBE6E2"), ("nn.Module", "#E3F3F0"), ("Loss (nn.*Loss)", "#FBE6E2"), ("torch.optim", "#EFE9F8"), ("Dataset + DataLoader", "#FFF3D6"), ("Training loop (yours)", "#EFE9F8"), ("device / .to()", "#F1EFEA"), ("state_dict save/load", "#F1EFEA")]
    s = '<svg viewBox="0 0 760 210" width="100%" style="max-width:760px">' + svg_defs()
    for i, (lbl, fill) in enumerate(items):
        col, row = i % 5, i // 5
        x, y = 12 + col * 150, 20 + row * 90
        s += svg_box(x, y, 138, 54, lbl, fill, font=12, bold=True)
        if i < 8:
            if col < 4:
                s += svg_arrow(x + 140, y + 27, x + 148, y + 27)
            else:
                s += f'<path d="M{x + 69},{y + 56} L{x + 69},{y + 75} L81,{y + 75} L81,{y + 88}" fill="none" stroke="#6B675F" stroke-width="2" marker-end="url(#arrowhead)"/>'
    s += svg_text(380, 200, "the loop in the middle is explicit — every line is yours", size=11, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    definition("**PyTorch**: إطار تعلم عميق يقدّم **موترات** (`torch.Tensor`) على CPU/GPU مع **اشتقاق تلقائي** (Autograd)، **وحدات** (`nn.Module`) تُركَّب منها النماذج، خسائر ومحسّنات ككائنات، و`Dataset`/`DataLoader` للدفعات. الفرق الجوهري عن Keras: **حلقة التدريب صريحة** — تكتب أنت `zero_grad`, `forward`, `loss`, `backward`, `step`. سير العمل فوري (eager): كل سطر ينفذ لحظة كتابته، والرسم الحسابي يُبنى ديناميكيًا أثناء التمرير الأمامي.")
    diagram("خريطة PyTorch", _map_svg(), what_ar="تسعة مكوّنات؛ الوسط منها هو الحلقة التي تكتبها بنفسك.", how_ar="الموترات والاشتقاق أساس؛ nn.Module والخسارة والمحسّن كائنات تركّبها؛ DataLoader يغذّي الحلقة؛ الجهاز والحفظ يحيطان بكل شيء.", takeaway_ar="لا يوجد fit: الحلقة هي الكود الذي تراه. ما أخفته Keras يصبح مكشوفًا.", title_en="PyTorch map")
    why("لماذا مسار PyTorch كاملًا لا صفحة مقارنة؟ لأن نصف الأبحاث الحديثة تُنشر به، ولأن الحلقة الصريحة تجعل كل ما تعلمته في الوحدات 12–17 **مرئيًا في الكود**: هنا ترى `backward` كسطر، و`step` كسطر، وتصفير التدرجات كسطر — أشياء كانت تختفي داخل `fit`.")
    h2("التثبيت والاستيراد", "Installing & importing")
    st.markdown("""
- **التثبيت**: `pip install torch` (CPU). لـ GPU اختر الأمر من صفحة PyTorch الرسمية بحسب نسخة CUDA. في Colab مثبّت مسبقًا. هذا المشروع يثبّت `torch` CPU مثبّتًا في `requirements.txt`.
- **الاستيراد القياسي**: `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `from torch.utils.data import Dataset, DataLoader`.
- **التحقق**: `torch.__version__`, `torch.cuda.is_available()`.
""")
    code_lab(CodeLab(
        key="pt_first", title_ar="أول نموذج PyTorch: نفس الانحدار الخطي، بحلقة صريحة", code=CODE, level="B",
        before=Before(goal_ar="حل نفس المسألة التي حلّتها Keras في 3 أسطر — لكن هنا ترى الحلقة كاملة. التحقق من الأوزان المتعلَّمة.", stage_ar="PyTorch: الأساسيات.",
                      inputs_ar="X (200, 2)، y (200, 1) من علاقة خطية معلومة.", expected_ar="النسخة والجهاز؛ W ≈ [3, −2]، b ≈ 1؛ خسارة صغيرة؛ تنبؤ ≈ 2.",
                      objects_ar="`nn.Linear`, `nn.MSELoss`, `torch.optim.SGD`, `torch.no_grad`."),
        explain=[("1-3", "الاستيراد، النسخة، وهل يوجد GPU (`cuda`). البذرة بـ `torch.manual_seed`."), ("6-7", "الموترات مباشرة من torch. **`unsqueeze(1)`**: y بشكل (200,) يجب أن يصبح (200, 1) ليطابق مخرج `Linear(2, 1)` — وإلا بث صامت خاطئ."),
                 ("9-11", "الطبقة (نموذج بطبقة واحدة)، الخسارة ككائن، والمحسّن **يستلم قائمة المعلمات** التي سيحدّثها."), ("13-21", "الحلقة الصريحة بخطواتها الخمس (أ–هـ). كل واحدة سطر. هذا ما ستكرره في كل مشروع PyTorch."),
                 ("22-24", "المعلمات في `model.weight` و`model.bias`؛ `.data` قيمتها، `.item()` للعدد الواحد."), ("25-27", "`model.eval()` + `torch.no_grad()` للاستدلال: بلا تسجيل للرسم، ذاكرة أقل، وطبقات Dropout/BN بوضع الاستدلال.")],
        run=run_printed(CODE),
        after_ar="- W بشكل (1, 2) لا (2, 1): PyTorch يخزّن `Linear` كـ (out, in) ويحسب xWᵀ + b. عكس Keras (in, out).\n- لو حذفت `zero_grad()` لتراكمت التدرجات (صفحة zero_grad).\n- قارن بالوحدة 16: هذا **هو** كود الحلقة اليدوية، مع استبدال الاشتقاق اليدوي بـ `backward()` والتحديث اليدوي بـ `step()`.",
    ))
    h2("PyTorch مقابل Keras في نظرة", "PyTorch vs Keras at a glance")
    compare_table(["", "Keras (فوق TensorFlow)", "PyTorch"],
                  [("بناء النموذج", "قائمة طبقات / رسم Functional", "صنف يرث `nn.Module` مع `forward` (أو `nn.Sequential` للبسيط)"), ("التدريب", "`compile` + `fit`", "حلقة صريحة تكتبها"), ("الاشتقاق", "داخل fit (`GradientTape` تحته)", "`loss.backward()` تستدعيه أنت"),
                   ("التدرجات", "تُحسب وتُطبَّق وتُنسى داخل الخطوة", "تُخزَّن في `.grad` وتتراكم حتى تصفّرها"), ("وضع الطبقات", "تلقائي (fit/evaluate/predict)", "`model.train()` / `model.eval()` مسؤوليتك"), ("الجهاز", "تلقائي", "`model.to(device)`, `x.to(device)` مسؤوليتك"),
                   ("الأسلوب", "تصريحي: تصف وتطلب", "أمري: تكتب ما يحدث")],
                  ["rtl", "rtl", "rtl"])
    intuition("Keras تسألك «ماذا تريد؟» وPyTorch تسألك «ماذا يحدث؟». لا أحدهما أفضل: الأول أسرع للقياسي، والثاني أوضح لغير القياسي. بعد الوحدات السابقة أنت مؤهل للاثنين.")
    h2("خريطة الصفحات الفرعية", "Sub-page map")
    st.markdown("""
1. **الموترات** — الإنشاء، الشكل، dtype، الجهاز، CPU↔GPU، العمليات، البث + **مستكشف موترات PyTorch**.
2. **Autograd** — requires_grad، .grad، الرسم الحسابي، backward + **محرك Autograd المرئي**.
3. **nn.Module** — `__init__`، `forward`، nn.Linear، التنشيطات، البنية، المعلمات.
4. **الخسائر والمحسّنات والبيانات** — nn.*Loss، torch.optim، Dataset، DataLoader، batch، shuffle.
5. **حلقة التدريب** — المحرك المرئي الإلزامي: كل خطوة بأشكالها وخسارتها وتدرجها ووزنها.
6. **لماذا zero_grad()؟** — تجربة التراكم.
7. **model.train() مقابل model.eval()** — تجربة Dropout.
8. **الحفظ والأخطاء الشائعة** — state_dict، عدم تطابق الجهاز/الشكل/dtype، التراكم، نسيان الوضع، OOM.
""")
    common_mistake("`y` بشكل (n,) مع مخرج (n, 1) في `MSELoss`: PyTorch يحذّر (`UserWarning: Using a target size ... different to the input size`) ثم **يبث** (n,1)−(n,) → (n,n) ويحسب خسارة خاطئة تمامًا. `unsqueeze(1)` دائمًا.")
    quiz("pt.intro", [
        Q("الفرق الجوهري عن Keras:", ["لا اشتقاق تلقائي", "حلقة التدريب صريحة تكتبها أنت", "لا GPU"], 1, "explicit loop."),
        Q("`optimizer = SGD(model.parameters(), lr)`: المحسّن يستلم…", ["البيانات", "المعلمات التي سيحدّثها", "الخسارة"], 1, "قائمة المعلمات."),
        Q("`nn.Linear(2, 1).weight.shape`", ["(2, 1)", "(1, 2)", "(2,)"], 1, "(out, in)."),
        Q("الخطوات الخمس بالترتيب:", ["forward, zero_grad, loss, step, backward", "zero_grad, forward, loss, backward, step", "step, backward, loss, forward, zero_grad"], 1, "أ–هـ."),
    ])
    takeaway("PyTorch = موترات + Autograd + nn.Module + خسارة/محسّن ككائنات + DataLoader + حلقة صريحة (zero_grad, forward, loss, backward, step) + الجهاز والوضع مسؤوليتك.")
    lesson_footer(LESSON, ["التعريف والخريطة.", "أول نموذج بحلقة صريحة.", "المقارنة السريعة مع Keras."])
