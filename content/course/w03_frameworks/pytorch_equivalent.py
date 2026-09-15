import streamlit as st

from components.callouts import common_mistake, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="course.w03.pytorch_equivalent",
    title_ar="المكافئ في PyTorch: نفس الشبكة الأولى بـ nn.Module وحلقة صريحة، والأخطاء الخمسة + الاختبار البعدي",
    title_en="The PyTorch Equivalent: The Same First Network with nn.Module & an Explicit Loop, the Five Errors — & Post-test",
    module="course.w03",
    order=4,
    prerequisites=["course.w03.under_the_hood", "foundations.frameworks.pytorch.training_loop", "foundations.frameworks.same_model_three_views"],
    objectives_ar=["كتابة نفس MLP (تعثر القروض) في PyTorch: nn.Module، خسارة، محسّن، DataLoader، حلقة تدريب وتحقق، تنبؤ.", "مقارنة المخرجات بمخرجات Keras سطرًا بسطر.", "الأخطاء الخمسة الشائعة برسائلها في الإطارين، والاختبار البعدي."],
    terms=["model", "loss", "optimizer", "epoch"],
    labs=["labs.torch_tensor_explorer"],
    difficulty="intermediate",
    summary_ar="Mini-lab: نفس MLP في Keras وPyTorch. PyTorch: صنف + حلقة (zero_grad/forward/loss/backward/step) + eval/no_grad. الأخطاء: شكل، ترميز الهدف، خسارة/تنشيط، dtype، جهاز.",
)

CODE = '''import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from labs.datasets import loan_default
torch.manual_seed(0)

# --- نفس البيانات ونفس التقسيم كما في درس Keras ---
df = loan_default(n=400, seed=7)
num = df[["income", "age", "num_late_payments", "debt_ratio"]].to_numpy("float32")
cat = pd.get_dummies(df[["city", "employment"]]).to_numpy("float32")
y = df["defaulted"].to_numpy("float32")
idx = np.random.default_rng(0).permutation(len(df)); tr, va, te = idx[:240], idx[240:320], idx[320:]
med = np.nanmedian(num[tr], 0); num = np.where(np.isnan(num), med, num)
mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-8
X = torch.tensor(np.concatenate([(num - mu) / sd, cat], 1)); Y = torch.tensor(y).unsqueeze(1)    # (400, 11), (400, 1)
loader = DataLoader(TensorDataset(X[tr], Y[tr]), batch_size=32, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1) النموذج: صنف بدل قائمة ---
class LoanMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__(); self.fc1 = nn.Linear(n_in, 16); self.out = nn.Linear(16, 1)
    def forward(self, x):
        return self.out(torch.relu(self.fc1(x)))          # logits (σ داخل الخسارة)
model = LoanMLP(X.shape[1]).to(device)
print(model); print("params:", sum(p.numel() for p in model.parameters()), "(Keras: 209)")

# --- 2) الخسارة والمحسّن ككائنات منفصلة (لا compile) ---
loss_fn = nn.BCEWithLogitsLoss(); opt = torch.optim.Adam(model.parameters(), lr=3e-3)

# --- 3) الحلقة = fit + EarlyStopping مكتوبان يدويًا ---
best, best_state, patience, bad = float("inf"), None, 8, 0
for epoch in range(1, 101):
    model.train(); running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); out = model(xb); loss = loss_fn(out, yb); loss.backward(); opt.step()
        running += loss.item() * len(xb)
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X[va].to(device)), Y[va].to(device)).item()
    if val_loss < best:
        best, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= patience:
            break
model.load_state_dict(best_state)                                   # restore_best_weights
print(f"epochs run: {epoch} | best val_loss: {best:.4f}")

# --- 4) evaluate + predict ---
model.eval()
with torch.no_grad():
    logits = model(X[te].to(device)); p = torch.sigmoid(logits).cpu()
    test_loss = loss_fn(logits, Y[te].to(device)).item(); acc = ((p >= 0.5).float() == Y[te]).float().mean().item()
print(f"test: loss={test_loss:.4f} accuracy={acc:.3f} | baseline={max(y[te].mean(), 1 - y[te].mean()):.3f}")
print("probabilities:", p[:5].numpy().ravel().round(3), "| decision@0.5:", (p[:5] >= 0.5).int().numpy().ravel(), "| truth:", y[te][:5].astype(int))'''


def render() -> None:
    lesson_header(LESSON)
    why("نفس الشبكة الأولى، نفس البيانات ونفس التقسيم — بـ PyTorch. الهدف ليس تعلم إطار ثانٍ من الصفر بل رؤية أن كل سطر في Keras له مقابل مكشوف هنا. زر «المكافئ في PyTorch» في دروس المقرر يقود إلى هذا النمط.")
    code_lab(CodeLab(
        key="w03_pt", title_ar="Mini-lab: نفس MLP في PyTorch مع إيقاف مبكر يدوي", code=CODE, level="C",
        before=Before(goal_ar="إعادة بناء درس «أول شبكة» في PyTorch بالكامل، بما فيه الإيقاف المبكر واسترجاع أفضل أوزان، وقراءة نفس المخرجات.", stage_ar="الأسبوع 03: المكافئ في PyTorch.",
                      inputs_ar="loan_default بنفس التحضير (11 خاصية).", expected_ar="print(model) بطبقتين و209 معلمة؛ توقف قبل 100 حقبة؛ test accuracy قريبة من نسخة Keras؛ 5 احتمالات وقرارات.", prerequisites_ar="الوحدة 21: PyTorch (كل الصفحات)."),
        explain=[("6-15", "نفس التحضير، ثم الموترات: `unsqueeze(1)` للهدف (الأسس: شكل الهدف)، DataLoader بدل `batch_size=` في fit، والجهاز."), ("19-25", "**Layers → Model**: صنف بطبقتين؛ `forward` يعيد logits (σ داخل `BCEWithLogitsLoss`). `print(model)` بدل summary، والعدّ يدوي = 209."),
                 ("28", "**compile** لا يوجد: الخسارة والمحسّن كائنان منفصلان."), ("31-48", "**fit + EarlyStopping** يدويًا: حلقة الدفعات الخمسية، تحقق بـ eval/no_grad، تتبع أفضل val_loss ونسخ state_dict وصبر 8 حقب، ثم استرجاع الأفضل."), ("51-56", "**evaluate/predict**: sigmoid على logits للاحتمالات، دقة يدوية، خط الأساس بجانبها.")],
        run=run_printed(CODE),
        after_ar="- الأرقام قريبة من Keras لا مطابقة: تهيئة وخلط مختلفان (الوحدة 21: ثلاث رؤى).\n- الأسطر 31–48 هي بالضبط ما يخفيه `fit(..., callbacks=[EarlyStopping])`: 15 سطرًا مقابل معامل واحد.\n- ما ربحته: كل شيء مرئي وقابل للتعديل. ما خسرته: 15 سطرًا يمكن أن تخطئ فيها (الأخطاء الصامتة).",
    ))
    h2("سطرًا بسطر: Keras مقابل PyTorch لنفس الشبكة", "Line by line")
    compare_table(["المرحلة", "Keras (درس أول شبكة)", "PyTorch (هنا)"],
                  [("النموذج", "`Sequential([Input, Dense(16, relu), Dense(1, sigmoid)])`", "`class LoanMLP(nn.Module)` + `forward` → logits"), ("الملخص", "`model.summary()`", "`print(model)` + `sum(numel)`"), ("الربط", "`compile(Adam(3e-3), 'binary_crossentropy', metrics)`", "`BCEWithLogitsLoss()`, `Adam(model.parameters(), 3e-3)`"),
                   ("الدفعات", "`batch_size=32` في fit", "`DataLoader(..., batch_size=32, shuffle=True)`"), ("التدريب", "`fit(..., epochs=100, validation_data, callbacks=[EarlyStopping])`", "حلقتان + zero_grad/forward/loss/backward/step + تحقق + صبر يدوي"), ("الوضع", "تلقائي", "`model.train()` / `model.eval()` + `no_grad`"),
                   ("أفضل أوزان", "`restore_best_weights=True`", "نسخ `state_dict` وإرجاعه"), ("التقييم", "`evaluate(X_te, y_te)`", "أمامي تحت no_grad + خسارة ودقة يدويًا"), ("التنبؤ", "`predict` → احتمالات", "`sigmoid(model(x))`")],
                  ["rtl", "code", "code"])
    h2("الأخطاء الخمسة في الإطارين", "The five errors in both frameworks")
    compare_table(["الخطأ", "Keras / TensorFlow", "PyTorch", "الحل"],
                  [("عدم تطابق الشكل", "`expected axis -1 … value 5, but received (None, 11)`", "`mat1 and mat2 shapes cannot be multiplied (32x11 and 5x16)`", "شكل المدخل من البيانات: `X.shape[1]`"),
                   ("ترميز الهدف الخاطئ", "`target and output must have the same rank` (categorical مع أعداد صحيحة)", "`expected target dtype to be Long` / بث صامت (n,) مع (n,1)", "sparse للأعداد الصحيحة؛ `unsqueeze(1)`/`long()`"),
                   ("خسارة/تنشيط غير متوافقين", "sigmoid+categorical (صامت) أو softmax(1)", "softmax داخل forward + CrossEntropy (صامت)", "جدول التوافق (الأسس 13)؛ logits للخسائر في PyTorch"),
                   ("dtype", "`cannot compute AddV2 … int32 … float`", "`must have the same dtype, but got Double and Float`", "`astype('float32')` / `.float()`"),
                   ("الجهاز/GPU", "تحذير cudart (يعمل على CPU)؛ OOM", "`Expected all tensors to be on the same device`", "Keras تلقائي؛ PyTorch `.to(device)` للنموذج والدفعة")],
                  ["rtl", "code", "code", "rtl"])
    with st.container(horizontal=True):
        st.button("الأخطاء في المعرض (كما تظهر فعليًا)", icon=":material/bug_report:", on_click=go, args=("foundations.gallery.errors",), key="w03_go_errors")
        st.button("خريطة المفاهيم بين الإطارين", icon=":material/compare_arrows:", on_click=go, args=("foundations.frameworks.concept_mapping",), key="w03_go_map")
        st.button("مستكشف موترات PyTorch", icon=":material/science:", on_click=go, args=("labs.torch_tensor_explorer",), key="w03_lab_pt")
    intuition("الأسبوع الرسمي يبقى Keras؛ PyTorch هنا **مرآة**: كل ما يبدو سحريًا في fit يظهر في المرآة سطرًا. من يرى الاثنين لا يخاف من أي منهما.")
    common_mistake("ترجمة `Dense(1, activation='sigmoid')` إلى `nn.Linear(16, 1)` + `nn.Sigmoid()` ثم `BCEWithLogitsLoss`: sigmoid مرتين → خسارة خاطئة بصمت. في PyTorch: logits + WithLogits.")
    h2("الاختبار البعدي", "Post-test")
    quiz("w03.posttest", [
        Q("Keras في هذا المقرر…", ["إطار مستقل", "واجهة عليا فوق TensorFlow", "بيئة تشغيل"], 1, ""),
        Q("`Input(shape=(12,))`:", ["12 ملاحظة", "كل ملاحظة 12 خاصية", "12 طبقة"], 1, ""),
        Q("`compile()`…", ["يدرّب", "يربط المحسّن والخسارة والمقاييس", "يحفظ"], 1, ""),
        Q("`history.history['val_loss']` طولها", ["عدد الدفعات", "عدد الحقب الفعلية", "عدد الملاحظات"], 1, ""),
        Q("ما يقابل fit في PyTorch:", ["compile", "حلقة تدريب تكتبها", "summary"], 1, ""),
        Q("`tape.gradient(loss, vars)` يقابل في PyTorch…", ["`optimizer.step()`", "`loss.backward()` ثم `.grad`", "`model.eval()`"], 1, ""),
        Q("`mat1 and mat2 shapes cannot be multiplied (32x11 and 5x16)`: أصلح…", ["batch_size", "`in_features` ليصبح 11", "المخرج"], 1, ""),
        Q("في PyTorch للتصنيف الثنائي المخرج يكون…", ["sigmoid داخل forward + BCELoss (الأفضل)", "logits + BCEWithLogitsLoss", "softmax"], 1, ""),
    ], title_ar="الاختبار البعدي — الأسبوع 03")
    takeaway("نفس الشبكة الأولى في المرآة: صنف + كائنات + حلقة + وضع. الأخطاء الخمسة لها رسائل معروفة في الإطارين. الأسبوع الرسمي يستمر بـ Keras.")
    lesson_footer(LESSON, ["Mini-lab كامل.", "جدول سطر بسطر.", "الأخطاء الخمسة والاختبار البعدي."])
