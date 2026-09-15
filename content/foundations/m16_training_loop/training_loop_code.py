import streamlit as st

from components.callouts import practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.training_loop.training_loop_code",
    title_ar="حلقة التدريب كودًا: NumPy مع تحقق وإيقاف مبكر وسجل",
    title_en="The Training Loop in Code: NumPy with Validation, Early Stopping & History",
    module="foundations.training_loop",
    order=3,
    prerequisites=["foundations.training_loop.validation_in_loop", "foundations.backprop.backpropagation.implementation"],
    objectives_ar=["كتابة الحلقة الكاملة في ~40 سطرًا مع خلط ودفعات وتحقق وإيقاف مبكر وسجل.", "مطابقة كل كتلة بمقابلها في Keras وPyTorch."],
    terms=["epoch", "batch", "backpropagation"],
    labs=["labs.training_loop_simulator"],
    difficulty="advanced",
    summary_ar="حلقتان متداخلتان + تحقق + إيقاف مبكر + history — وهذا ما تفعله fit() حرفيًا.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
# بيانات: هلالان
t = rng.uniform(0, np.pi, 300)
X = np.vstack([np.c_[np.cos(t), np.sin(t)], np.c_[1-np.cos(t), 0.5-np.sin(t)]]) + rng.normal(0, 0.2, (600, 2))
y = np.r_[np.zeros(300), np.ones(300)]; p_ = rng.permutation(600); X, y = X[p_], y[p_]
X_tr, y_tr, X_val, y_val = X[:450], y[:450], X[450:], y[450:]

relu, drelu, sig = (lambda z: np.maximum(0, z)), (lambda z: (z > 0).astype(float)), (lambda z: 1/(1+np.exp(-z)))
W1 = rng.normal(0, np.sqrt(2/2), (2, 16)); b1 = np.zeros(16); W2 = rng.normal(0, np.sqrt(2/16), (16, 1)); b2 = np.zeros(1)   # 1) تهيئة

def forward(X): z1 = X @ W1 + b1; a1 = relu(z1); z2 = a1 @ W2 + b2; return sig(z2)[:, 0], (z1, a1)
def bce(p, y): p = np.clip(p, 1e-7, 1-1e-7); return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

eta, batch_size, epochs, patience = {eta}, {bs}, 200, 8
history = {{"loss": [], "val_loss": [], "val_acc": []}}
best, bad, best_w = np.inf, 0, None
for epoch in range(1, epochs + 1):
    order = rng.permutation(len(X_tr))                                   # 2) خلط
    ep_loss = 0.0
    for s in range(0, len(X_tr), batch_size):                            # 3) دفعة
        idx = order[s:s+batch_size]; Xb, yb = X_tr[idx], y_tr[idx]
        p, (z1, a1) = forward(Xb)                                        # 4) أمامي
        ep_loss += bce(p, yb) * len(idx)                                 # 5) خسارة
        d2 = (p - yb)[:, None] / len(idx)                                # 6) خلفي
        gW2 = a1.T @ d2; gb2 = d2.sum(0)
        d1 = (d2 @ W2.T) * drelu(z1); gW1 = Xb.T @ d1; gb1 = d1.sum(0)
        W1 -= eta*gW1; b1 -= eta*gb1; W2 -= eta*gW2; b2 -= eta*gb2       # 7) تحديث (SGD)
    history["loss"].append(ep_loss / len(X_tr))                          # 9) نهاية الحقبة
    pv, _ = forward(X_val)                                               # 10) تحقق (بلا تدرج)
    vl = bce(pv, y_val); history["val_loss"].append(vl); history["val_acc"].append(((pv >= 0.5) == y_val).mean())
    if vl < best - 1e-4: best, bad, best_w = vl, 0, (W1.copy(), b1.copy(), W2.copy(), b2.copy())
    else: bad += 1
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {{epoch:3d}}: loss={{history['loss'][-1]:.4f}} val_loss={{vl:.4f}} val_acc={{history['val_acc'][-1]:.3f}}")
    if bad >= patience:                                                  # 12) توقف مبكر
        W1, b1, W2, b2 = best_w; print(f"early stopping at epoch {{epoch}}, restored best (val_loss={{best:.4f}})"); break
print("history keys:", list(history), " epochs run:", len(history["loss"]))'''


def _controls() -> dict:
    c1, c2 = st.columns(2)
    with c1:
        eta = st.select_slider("η", options=[0.03, 0.1, 0.3, 0.5], value=0.3, key="ctrl_tlc_eta")
    with c2:
        bs = st.select_slider("batch_size", options=[8, 16, 32, 64, 450], value=32, key="ctrl_tlc_bs")
    return {"eta": eta, "bs": bs}


def render() -> None:
    lesson_header(LESSON)
    h2("الكود الكامل", "The full code")
    code_lab(CodeLab(
        key="loop_code", title_ar="حلقة تدريب كاملة من الصفر", code=CODE, template=True, defaults={"eta": 0.3, "bs": 32},
        before=Before(goal_ar="تدريب شبكة 2 → 16 → 1 على الهلالين بحلقة كاملة: خلط، دفعات، أمامي، خسارة، خلفي، تحديث، تحقق، إيقاف مبكر، سجل.", stage_ar="حلقة التدريب ← الكود.",
                      inputs_ar="600 نقطة (450 تدريب، 150 تحقق).", expected_ar="سطر كل 10 حقب؛ val_acc يرتفع إلى ~0.95+؛ إيقاف مبكر غالبًا قبل 200 مع استعادة الأفضل.", math_ar="الخطوات 1–12 من الدرس الأول مرقّمة في التعليقات."),
        explain=[("10", "الخطوة 1: تهيئة He."), ("15-17", "المعلمات الفائقة والسجل: نفس مفاتيح `History.history`."), ("19-20", "الحلقة الخارجية + الخلط كل حقبة."),
                 ("21-28", "الحلقة الداخلية: الخطوات 3–7 على كل دفعة. لاحظ تراكم `ep_loss` مرجّحًا بحجم الدفعة (الأخيرة قد تكون جزئية)."),
                 ("29-32", "نهاية الحقبة: متوسط الخسارة، ثم التحقق بلا تدرج."), ("33-37", "الإيقاف المبكر بصبر 8 واستعادة أفضل الأوزان — نفس `EarlyStopping(restore_best_weights=True)`.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- `batch_size=450` (Batch GD): تحديث واحد لكل حقبة فيبطئ؛ `8`: 57 تحديثًا لكل حقبة لكن بضوضاء.\n- إن كانت `loss` أقل من `val_loss` بفارق كبير ومتزايد فهذا فرط تخصيص؛ الإيقاف المبكر يلتقطه.",
    ))
    table(["كتلة NumPy", "Keras", "PyTorch"],
          [("التهيئة", "Sequential([Dense(16,'relu'), Dense(1,'sigmoid')])", "nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,1))"), ("الخلط والدفعات", "fit(shuffle=True, batch_size=32)", "DataLoader(ds, batch_size=32, shuffle=True)"),
           ("أمامي + خسارة", "داخل fit (compile(loss=...))", "p = model(Xb); loss = loss_fn(p, yb)"), ("خلفي", "داخل fit", "optimizer.zero_grad(); loss.backward()"), ("تحديث", "داخل fit (compile(optimizer=...))", "optimizer.step()"),
           ("تحقق", "fit(validation_data=...)", "model.eval(); with torch.no_grad(): ..."), ("إيقاف مبكر + سجل", "callbacks=[EarlyStopping(...)]; history = fit(...)", "يدوي كما هنا")],
          ["rtl", "code", "code"])
    practical_note("هذا الكود هو المرجع الذهني لكل ما سيأتي في أكاديمية أطر العمل: كل سطر Keras/PyTorch يقابل سطرًا هنا.")
    quiz("loop.code", [
        Q("لماذا نضرب `bce(p, yb)` في `len(idx)` عند التراكم؟", ["للسرعة", "لترجيح الدفعة الجزئية الأخيرة بحجمها الصحيح", "خطأ"], 1, "متوسط مرجّح."),
        Q("`optimizer.zero_grad()` في PyTorch يقابل هنا…", ["لا مقابل: نحسب التدرجات من الصفر كل دفعة", "التحديث", "الخلط"], 0, "لا تراكم عندنا."),
        Q("الإيقاف المبكر هنا يراقب…", ["loss", "val_loss", "val_acc"], 1, "التحقق."),
    ])
    takeaway("40 سطرًا تعادل fit(): حلقتان، تحقق بلا تدرج، إيقاف مبكر بأفضل الأوزان، وسجل. احتفظ به مرجعًا.")
    lesson_footer(LESSON, ["الخطوات 1–12 مرقّمة في الكود.", "جدول المقابلات.", "الدفعة الجزئية تُرجَّح بحجمها."])
