import html as _html

import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table
from labs.fw import torch_loop_trace

LESSON = Lesson(
    id="foundations.frameworks.pytorch.training_loop",
    title_ar="حلقة التدريب في PyTorch: المحرك المرئي سطرًا سطرًا",
    title_en="The PyTorch Training Loop: Line-by-Line Visualizer",
    module="foundations.frameworks",
    parent="foundations.frameworks.pytorch",
    order=24,
    prerequisites=["foundations.frameworks.pytorch.loss_optim_data", "foundations.training_loop.the_loop", "foundations.training_loop.validation_in_loop"],
    objectives_ar=["كتابة حلقة التدريب الكاملة: for epoch → model.train() → for batch → zero_grad → forward → loss → backward → step → تجميع → model.eval() + no_grad للتحقق.", "عند كل خطوة: السطر المضاء، الأشكال، الخسارة، معيار التدرج، والوزن قبل/بعد التحديث — بتتبع حقيقي.", "تجميع الحقبة (متوسط موزون) وحلقة التحقق."],
    terms=["epoch", "batch", "iteration", "gradient"],
    labs=["labs.training_loop_simulator"],
    difficulty="intermediate",
    summary_ar="الحلقة القياسية: لكل حقبة model.train()؛ لكل دفعة: zero_grad → out = model(xb) → loss → loss.backward() → optimizer.step() → تجميع؛ ثم model.eval() وwith no_grad للتحقق. كل سطر يقابل خطوة في الوحدة 16.",
)

LOOP = [
    "for epoch in range(epochs):",
    "    model.train()",
    "    for xb, yb in train_loader:",
    "        optimizer.zero_grad()",
    "        out = model(xb)",
    "        loss = loss_fn(out, yb)",
    "        loss.backward()",
    "        optimizer.step()",
    "        running += loss.item() * len(xb)",
    "    model.eval()",
    "    with torch.no_grad():",
    "        val_loss, val_acc = evaluate(model, val_loader)",
]
STAGES = ["epoch", "train()", "batch", "zero_grad", "forward", "loss", "backward", "step", "metrics", "eval()", "validation"]

CODE = '''import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from labs.tinynet import make_moons
torch.manual_seed(0)
X, y = make_moons(400, noise=0.25, seed=0)
X = torch.tensor(X); y = torch.tensor(y).unsqueeze(1)                       # (400,2) float32, (400,1) float32
train_loader = DataLoader(TensorDataset(X[:300], y[:300]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[300:], y[300:]), batch_size=100)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def evaluate(model, loader):                                                  # حلقة التحقق
    model.eval(); tot, correct, n = 0.0, 0, 0
    with torch.no_grad():                                                     # بلا رسم، بلا تدرج
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb); tot += loss_fn(out, yb).item() * len(xb)
            correct += ((out >= 0).float() == yb).sum().item(); n += len(xb)
    return tot / n, correct / n

history = {"loss": [], "val_loss": [], "val_acc": []}
for epoch in range(1, 9):
    model.train(); running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(xb)                                     # مجموع موزون بحجم الدفعة
    train_loss = running / len(train_loader.dataset)                          # تجميع الحقبة
    val_loss, val_acc = evaluate(model, val_loader)
    history["loss"].append(train_loss); history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
    print(f"epoch {epoch}/8 - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.3f}")
print("history keys:", list(history), "| len:", len(history["loss"]), "| updates per epoch:", len(train_loader))'''


def _code_html(active: int) -> str:
    rows = []
    for i, line in enumerate(LOOP):
        style = "background:#FBE6E2;border-inline-start:3px solid #C8473A;font-weight:600" if i == active else ""
        rows.append(f'<div style="padding:1px 6px;{style}"><span style="color:#B9B2A6;display:inline-block;width:1.6em">{i + 1}</span>{_html.escape(line)}</div>')
    return '<pre dir="ltr" style="text-align:left;font-family:JetBrains Mono,Consolas,monospace;font-size:12.5px;line-height:1.45;margin:0;background:#FFFDF9;border:1px solid #E5E0D6;border-radius:8px;padding:6px">' + "".join(rows) + "</pre>"


def render() -> None:
    lesson_header(LESSON)
    definition("**حلقة التدريب في PyTorch** ليست دالة تستدعيها بل كود تكتبه: حلقة الحقب، تبديل الوضع إلى تدريب، حلقة الدفعات بخطواتها الخمس (تصفير، أمامي، خسارة، خلفي، تحديث)، تجميع المقاييس، ثم التحقق بوضع الاستدلال وبلا تدرج. **كل سطر يقابل خطوة في الوحدة 16** — لا يوجد ما يخفيه.")
    why("هذه الصفحة إلزامية لأن كل مشروع PyTorch يعيد هذه الأسطر الاثني عشر. من يفهمها سطرًا سطرًا يستطيع تعديل أي شيء (خسائر مركّبة، تجميع تدرجات، تسجيل مخصص)؛ ومن يحفظها يتعطل عند أول خطأ. المحرك أدناه يشغّل حلقة **حقيقية** ويعرض عند كل سطر ما يحدث فعلًا.")
    h2("المحرك المرئي", "The visualizer")
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.slider("epochs", 1, 3, 2, key="ptl_e")
    with c2:
        bs = st.select_slider("batch_size", options=[20, 40, 60, 120], value=40, key="ptl_bs")
    with c3:
        lr = st.select_slider("lr (SGD)", options=[0.1, 0.5, 1.0], value=0.5, key="ptl_lr")
    tr = torch_loop_trace(8, int(epochs), int(bs), float(lr), 0)
    spe = tr["steps_per_epoch"]
    frames = []
    for e in tr["events"]:
        if e["kind"] == "epoch_begin":
            frames.append(Frame(_code_html(0), caption(f"**الحقبة {e['epoch']}/{epochs}** تبدأ. `train_loader` سيعطي {spe} دفعة (n = {tr['n']}, batch_size = {bs})."), action="epoch", highlight=0))
            frames.append(Frame(_code_html(1), caption("**`model.train()`**: وضع التدريب — Dropout فعّال وBN بإحصاءات الدفعة (هنا لا شيء منهما، لكن السطر يُكتب دائمًا)."), action="train()", highlight=1))
        elif e["kind"] == "batch":
            b = e["batch"]; where = f"الحقبة {e['epoch']} · الدفعة {b}/{spe}"
            frames.append(Frame(_code_html(2), caption(f"**{where} — تحميل الدفعة**: `xb.shape = {tuple(e['x_shape'])}`, `yb.shape = {tuple(e['y_shape'])}` (DataLoader أضاف بُعد الدفعة وخلط الترتيب)."), action="batch", highlight=2))
            gz = "لا تدرج بعد (`.grad` هو None)" if e["g_before"] is None else f"`weight.grad[0,0]` كان {e['g_before']:+.4f} من الدفعة السابقة → أصبح `None` (مُسح؛ الافتراضي `set_to_none=True`)"
            frames.append(Frame(_code_html(3), caption(f"{where} — **`optimizer.zero_grad()`**: مسح `.grad` المتراكم: {gz}. بدونه تُجمع تدرجات كل الدفعات."), action="zero_grad", highlight=3))
            frames.append(Frame(_code_html(4), caption(f"{where} — **أمامي**: `model(xb)` → `out.shape = {tuple(e['out_shape'])}` (logits). أول 3: `{e['logits']}`. الرسم الحسابي يُبنى الآن."), action="forward", highlight=4))
            frames.append(Frame(_code_html(5), caption(f"{where} — **الخسارة** `BCEWithLogitsLoss(out, yb)` = **{e['loss']:.4f}** (عدد واحد بـ grad_fn)."), action="loss", values=[("loss", "", f"{e['loss']:.4f}")], highlight=5))
            frames.append(Frame(_code_html(6), caption(f"{where} — **`loss.backward()`**: Autograd يملأ `.grad` لكل معلمة. معيار التدرج الكلي `{e['gnorm']:.4f}`؛ عينة `model[0].weight.grad[0,0] = {e['g00']:+.4f}`."), action="backward", values=[("‖∇L‖", "", f"{e['gnorm']:.4f}"), ("weight.grad[0,0]", "", f"{e['g00']:+.4f}")], highlight=6))
            frames.append(Frame(_code_html(7), caption(f"{where} — **`optimizer.step()`**: لكل معلمة `p -= lr * p.grad`. `weight[0,0]`: {e['w_before']:+.4f} → {e['w_after']:+.4f} (= {e['w_before']:+.4f} − {lr}×{e['g00']:+.4f})."), action="step", equation=f"w ← w − η·g = {e['w_before']:+.4f} − {lr}×({e['g00']:+.4f}) = {e['w_after']:+.4f}", values=[("weight[0,0]", f"{e['w_before']:+.4f}", f"{e['w_after']:+.4f}")], highlight=7))
            frames.append(Frame(_code_html(8), caption(f"{where} — **تجميع**: `running += loss.item() * len(xb)` = {e['loss']:.4f} × {e['x_shape'][0]}. `.item()` يفصل العدد عن الرسم (لا تجمع الموتر نفسه!)."), action="metrics", highlight=8))
        else:
            frames.append(Frame(_code_html(9), caption(f"**نهاية الحقبة {e['epoch']}** — `model.eval()`: وضع الاستدلال. خسارة التدريب المجمّعة = {e['loss']:.4f}."), action="eval()", values=[("epoch loss", "", f"{e['loss']:.4f}")], highlight=9))
            frames.append(Frame(_code_html(10), caption(f"**`with torch.no_grad()`** + حلقة التحقق على {tr['n_val']} ملاحظة: بلا رسم وبلا تحديث. `val_loss = {e['val_loss']:.4f}`, `val_acc = {e['val_acc']:.3f}`."), action="validation", values=[("val_loss", "", f"{e['val_loss']:.4f}"), ("val_acc", "", f"{e['val_acc']:.3f}")], highlight=10))
    animation_player("ptl_anim", frames, title_ar="حلقة التدريب: السطر المضاء وما يحدث فعلًا", stages=STAGES, interval_ms=1300)
    rows = []
    for ep in range(1, epochs + 1):
        bl = [x["loss"] for x in tr["events"] if x["kind"] == "batch" and x["epoch"] == ep]; v = [x for x in tr["events"] if x["kind"] == "epoch_end" and x["epoch"] == ep][0]
        rows.append((str(ep), str(len(bl)), f"{np.mean(bl):.4f}", f"{v['val_loss']:.4f}", f"{v['val_acc']:.3f}"))
    table(["الحقبة", "التحديثات", "loss (متوسط الدفعات)", "val_loss", "val_acc"], rows, ["num"] * 5)
    intuition(f"عدد الإطارات = 2 + 7×{spe} لكل حقبة + 2. هذه هي «التكرارات» التي كانت مخفية في `fit`: هنا كل واحدة سطر تراه وتستطيع أن تطبع منه ما تريد.")
    h2("الحلقة الكاملة كما تكتبها", "The full loop as you write it")
    code_lab(CodeLab(
        key="pt_loop", title_ar="حلقة تدريب وتحقق كاملة بـ History يدوي", code=CODE, level="B",
        before=Before(goal_ar="الحلقة القياسية كاملة مع الجهاز، والتجميع الموزون، ودالة تحقق منفصلة، وقاموس History تملؤه بنفسك — لتطابق ما يعطيه Keras.", stage_ar="PyTorch: حلقة التدريب.",
                      inputs_ar="moons: 300 تدريب / 100 تحقق.", expected_ar="8 أسطر بشكل سطر Keras؛ الخسارة تنخفض وval_acc يرتفع؛ 10 تحديثات لكل حقبة.",
                      objects_ar="`DataLoader`, `nn.Sequential`, `BCEWithLogitsLoss`, `Adam`, `model.train/eval`, `torch.no_grad`."),
        explain=[("5-9", "البيانات كموترات؛ `unsqueeze(1)` للهدف؛ DataLoader للتدريب (خلط) وللتحقق (بلا خلط)؛ الجهاز."), ("11-13", "النموذج **على الجهاز**، الخسارة، المحسّن."),
                 ("15-22", "دالة التحقق: `eval()` + `no_grad()`، تجميع موزون، دقة من logits (`out >= 0` ⇔ p ≥ 0.5)."), ("24-26", "حلقة الحقب: `train()` وتصفير المجمّع."),
                 ("27-34", "الخطوات الخمس + نقل الدفعة إلى الجهاز + التجميع بـ `.item()`."), ("35-38", "متوسط الحقبة = المجموع الموزون / عدد الأمثلة؛ التحقق؛ الإلحاق بـ History؛ سطر سجل بشكل Keras.")],
        run=run_printed(CODE),
        after_ar="- السطر المطبوع يطابق سطر Keras حرفيًا لأن الحساب نفسه.\n- `running += loss.item() * len(xb)` ثم القسمة على n = المتوسط الصحيح حتى مع دفعة أخيرة أصغر (بخلاف متوسط متوسطات الدفعات).\n- الدقة من logits بلا sigmoid: `sigmoid(z) ≥ 0.5 ⇔ z ≥ 0` (وحدة 12).",
    ))
    compare_table(["سطر الحلقة", "المقابل في الوحدة 16", "ما الذي يفعله فعلًا"],
                  [("`model.train()`", "—", "يضبط `training=True` في كل وحدة (Dropout/BN)"), ("`optimizer.zero_grad()`", "—", "`p.grad = None` (أو صفر) لكل معلمة"), ("`out = model(xb)`", "`forward(X_batch)`", "أمامي + بناء الرسم"),
                   ("`loss = loss_fn(out, yb)`", "`loss(out, y_batch)`", "عدد واحد بـ grad_fn"), ("`loss.backward()`", "`backward(cache, y_batch)`", "قاعدة السلسلة → `.grad`"), ("`optimizer.step()`", "`W -= lr * gW`", "قاعدة التحديث للمحسّن المختار"),
                   ("`running += loss.item() * len(xb)`", "`losses.append(loss)`", "تجميع بلا إبقاء الرسم"), ("`model.eval()` + `no_grad`", "`predict(X_val)`", "وضع استدلال بلا تسجيل")],
                  ["code", "code", "rtl"])
    common_mistake("`running += loss` (بلا `.item()`): يبقي الرسم الحسابي لكل دفعة في الذاكرة حتى نهاية الحقبة — ذاكرة GPU تنفد ببطء (OOM في الحقبة الثالثة). دائمًا `.item()` أو `.detach()` عند التجميع.")
    quiz("pt.loop", [
        Q("الترتيب الصحيح داخل الدفعة:", ["forward → zero_grad → backward → step", "zero_grad → forward → loss → backward → step", "backward → forward → step"], 1, "الخطوات الخمس."),
        Q("`loss.item()` في التجميع…", ["يبطئ", "يفصل العدد عن الرسم فيحرر الذاكرة", "يغيّر الخسارة"], 1, "بلا رسم."),
        Q("التحقق يحتاج…", ["`model.train()`", "`model.eval()` + `torch.no_grad()`", "`optimizer.step()`"], 1, "استدلال."),
        Q("متوسط خسارة الحقبة الصحيح =", ["متوسط متوسطات الدفعات", "Σ(loss × حجم الدفعة) / n", "آخر دفعة"], 1, "موزون."),
    ])
    takeaway("12 سطرًا: epoch → train() → batch → zero_grad → forward → loss → backward → step → تجميع بـ item() → eval() → no_grad → تحقق. كل سطر خطوة في الوحدة 16 مكشوفة.")
    lesson_footer(LESSON, ["المحرك: السطر المضاء + الأشكال + الخسارة + التدرج + الوزن.", "الحلقة الكاملة بـ History.", "جدول المطابقة مع الوحدة 16."])
