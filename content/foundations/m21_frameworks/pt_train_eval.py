import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.frameworks.pytorch.train_eval",
    title_ar="model.train() مقابل model.eval(): الوضع يغيّر سلوك الطبقات",
    title_en="model.train() vs model.eval(): Mode Changes Layer Behaviour",
    module="foundations.frameworks",
    parent="foundations.frameworks.pytorch",
    order=26,
    prerequisites=["foundations.frameworks.pytorch.zero_grad", "foundations.regularization.dropout", "foundations.regularization.batch_normalization"],
    objectives_ar=["إثبات تجريبيًا أن Dropout وBatchNorm يتصرفان بشكل مختلف بحسب model.training.", "التمييز بين الوضع (train/eval) وتعطيل التدرج (no_grad): شيئان مستقلان.", "أعراض نسيان eval() ونسيان العودة إلى train()."],
    terms=["standardization", "mean", "variance"],
    labs=["labs.dropout_lab"],
    difficulty="intermediate",
    summary_ar="model.train()/eval() يضبطان علامة training في كل وحدة: Dropout يسقط في التدريب وهوية في الاستدلال؛ BN بإحصاءات الدفعة في التدريب وبالمتحركة في الاستدلال. no_grad شيء آخر: يمنع بناء الرسم فقط.",
)

CODE = '''import torch, torch.nn as nn
torch.manual_seed(0)
r = lambda t: [round(v, 2) for v in t.detach().flatten().tolist()]      # طباعة مختصرة
x = torch.randn(8, 4)                                          # دفعة من 8
x1 = x[:1]                                                     # ملاحظة واحدة (نفس المدخل دائمًا)

# --- Dropout ---
drop = nn.Dropout(p=0.5)
drop.train()
print("Dropout train mode, same input 3 times:", [r(drop(x1)) for _ in range(3)], "<- random masks, survivors scaled by 1/(1-p)=2")
drop.eval()
print("Dropout eval  mode, same input 3 times:", [r(drop(x1)) for _ in range(3)], "<- identity")
print("input was                             :", r(x1))

# --- BatchNorm ---
bn = nn.BatchNorm1d(4)
bn.train()
for _ in range(20):                                            # تدريب: تحديث المتوسطات المتحركة
    _ = bn(torch.randn(8, 4) * 3 + 5)                          # مدخلات بمتوسط 5 وانحراف 3
print("BN running_mean after training:", r(bn.running_mean), "| running_var:", r(bn.running_var))
bn.train();  y_train = bn(x)                                   # يستخدم إحصاءات x نفسها
bn.eval();   y_eval  = bn(x)                                   # يستخدم المتحركة
print("BN train-mode output mean/std per feature:", r(y_train.mean(0)), r(y_train.std(0)), "<- ~0/1 by construction")
print("BN eval -mode output mean per feature    :", r(y_eval.mean(0)), "<- shifted: x has mean~0 but running_mean~5")
try:
    bn.train(); bn(x1)                                         # ملاحظة واحدة في وضع التدريب
except ValueError as e:
    print("BN train mode with batch of 1 ->", str(e)[:70])
bn.eval(); print("BN eval mode with batch of 1  -> works:", tuple(bn(x1).shape))

# --- الوضع مستقل عن no_grad ---
model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 1))
model.eval()
out = model(x1)
print("eval() alone: requires_grad =", out.requires_grad, "(graph still built!)")
with torch.no_grad():
    out = model(x1)
print("eval() + no_grad: requires_grad =", out.requires_grad, "| model.training =", model.training)
model.train(); print("back to train(): model.training =", model.training, "| submodule flags:", [m.training for m in model])'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`model.train()` / `model.eval()`**: يضبطان العلم `self.training` في النموذج وكل وحداته الفرعية (`True`/`False`). بعض الوحدات تقرأ هذا العلم وتغيّر سلوكها: **Dropout** (إسقاط عشوائي في التدريب، هوية في الاستدلال)، **BatchNorm** (إحصاءات الدفعة في التدريب، المتوسطات المتحركة في الاستدلال). الوحدات الأخرى (Linear، ReLU) لا تتأثر. هذا **مستقل تمامًا** عن `torch.no_grad()` الذي يتحكم في بناء الرسم الحسابي.")
    why("«train للتدريب وeval للاختبار» جملة صحيحة وغير كافية. السؤال: **ما الذي يتغير فعلًا؟** الجواب التجريبي أدناه: نفس المدخل يعطي مخرجات مختلفة في كل استدعاء بوضع التدريب (Dropout)، ومخرجات منزاحة بوضع الاستدلال إن كانت BN دُرّبت على توزيع آخر — وملاحظة واحدة تكسر BN في وضع التدريب.")
    code_lab(CodeLab(
        key="pt_train_eval", title_ar="Dropout وBatchNorm في الوضعين، والاستقلال عن no_grad", code=CODE, level="B",
        before=Before(goal_ar="إعطاء نفس المدخل لوحدة Dropout ولوحدة BatchNorm في وضع التدريب ثم الاستدلال، ومشاهدة الفرق بالأرقام؛ ثم إثبات أن eval() لا يعطّل الرسم وأن no_grad لا يغيّر الوضع.", stage_ar="PyTorch: الوضع.",
                      inputs_ar="دفعة (8, 4) وملاحظة واحدة (1, 4).", expected_ar="Dropout: ثلاث مخرجات مختلفة (أصفار وقيم مضاعفة) ثم ثلاث متطابقة = المدخل. BN: متوسطات متحركة ≈ 5؛ مخرج التدريب ~0/1؛ مخرج الاستدلال منزاح؛ خطأ مع دفعة من 1 في التدريب. eval وحده يبقي requires_grad=True.",
                      objects_ar="`nn.Dropout`, `nn.BatchNorm1d`, `.train()`, `.eval()`, `torch.no_grad`, `.training`."),
        explain=[("8-13", "Dropout: في `train()` كل استدعاء قناع عشوائي جديد (أصفار) والباقون ×2 (القسمة على 1−p، وحدة 20). في `eval()` هوية تامة."),
                 ("16-19", "BN: 20 دفعة بمتوسط 5 تدفع `running_mean` نحو 5 و`running_var` نحو 9 (زخم 0.1)."), ("20-23", "نفس x: في التدريب يُطبَّع بإحصاءاته هو (~0/1)؛ في الاستدلال بالمتحركة (≈5) فيخرج منزاحًا — **هذا هو الفرق الذي يُفسد نتائجك لو نسيت eval()** أو لو اختلف توزيع الاستدلال."),
                 ("24-28", "دفعة من ملاحظة واحدة: لا يمكن حساب تباين في التدريب → خطأ؛ في الاستدلال يعمل لأن المتحركة جاهزة."), ("31-38", "الاستقلال: `eval()` وحده يبقي الرسم (requires_grad=True) — ذاكرة مهدرة؛ `no_grad` وحده لا يغيّر الوضع. للاستدلال الصحيح **كلاهما**. ولا تنسَ العودة إلى `train()`.")],
        run=run_printed(CODE),
        after_ar="- `.training` علم لكل وحدة؛ `model.train()` يمر عليها كلها.\n- Keras تفعل كل هذا داخل fit/evaluate/predict (`training=True/False`). في PyTorch مسؤوليتك — وأول ما يُنسى.\n- خطأ BN مع دفعة من 1 يظهر عادةً عند آخر دفعة ناقصة في التدريب: `drop_last=True` أو batch أكبر.",
    ))
    st.button("افتح معمل Dropout (التدريب مقابل الاستدلال)", icon=":material/science:", type="primary", on_click=go, args=("labs.dropout_lab",), key="pt_te_lab")
    h2("جدول السلوك", "Behaviour table")
    compare_table(["الوحدة", "train()", "eval()", "no_grad يؤثر؟"],
                  [("`nn.Dropout(p)`", "يصفّر كل عنصر باحتمال p ويقسّم الباقي على 1−p (عشوائي)", "هوية", "لا"), ("`nn.BatchNorm*`", "يطبّع بإحصاءات الدفعة ويحدّث المتحركة", "يطبّع بالمتحركة (حتمي)", "لا"),
                   ("`nn.Linear`, `nn.Conv*`, `nn.ReLU`", "نفس الحساب", "نفس الحساب", "لا (فقط الرسم)"), ("بناء الرسم / الذاكرة", "يُبنى", "يُبنى!", "**نعم**: no_grad يمنعه")],
                  ["code", "rtl", "rtl", "rtl"])
    intuition("سؤالان مستقلان قبل كل تمرير: (1) هل أريد سلوك التدريب أم الاستدلال للطبقات؟ → `train()/eval()`. (2) هل سأستدعي `backward()` بعده؟ → إن لا، `no_grad()`. الاستدلال = (eval, no_grad). التدريب = (train, رسم). التحقق داخل التدريب = (eval, no_grad) ثم **العودة إلى train**.")
    debugging_note("**نسيت eval()**: نتائج `predict` تتغير بين استدعاء وآخر (Dropout)، أو دقة التحقق أقل من المتوقع وتتقلب (BN بإحصاءات دفعة التحقق). **نسيت العودة إلى train()** بعد التحقق: من الحقبة الثانية Dropout معطّل وBN لا تحدّث متوسطاتها — التدريب «يعمل» لكن التنظيم اختفى بصمت.")
    common_mistake("`model.eval()` بلا `no_grad` في حلقة استدلال على آلاف الدفعات: الرسم يُبنى لكل دفعة والذاكرة تتراكم إن احتفظت بالمخرجات. `torch.inference_mode()` بديل أسرع لـ `no_grad` في النسخ الحديثة عندما لا تحتاج التدرج إطلاقًا.")
    quiz("pt.te", [
        Q("Dropout في `eval()`…", ["يسقط بنسبة p", "هوية", "يضاعف القيم"], 1, "لا إسقاط."),
        Q("BN في `train()` تستخدم…", ["المتوسطات المتحركة", "إحصاءات الدفعة الحالية", "لا شيء"], 1, "الدفعة."),
        Q("`model.eval()` وحده…", ["يمنع بناء الرسم", "يغيّر سلوك Dropout/BN فقط", "يعطّل التدرج"], 1, "الوضع فقط."),
        Q("بعد التحقق داخل حلقة التدريب يجب…", ["لا شيء", "`model.train()`", "`zero_grad()`"], 1, "العودة."),
    ])
    takeaway("train()/eval() = علم يغيّر Dropout وBN فقط؛ no_grad = لا رسم. الاستدلال يحتاج الاثنين؛ التدريب يعود إلى train(). نسيان أحدهما خطأ صامت.")
    lesson_footer(LESSON, ["التجربة بالأرقام لـ Dropout وBN.", "جدول السلوك.", "الاستقلال عن no_grad."])
