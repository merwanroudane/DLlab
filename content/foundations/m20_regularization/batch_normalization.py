import numpy as np
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.regularization.batch_normalization",
    title_ar="Batch Normalization بشرح دقيق",
    title_en="Batch Normalization, Precisely",
    module="foundations.regularization",
    order=5,
    prerequisites=["foundations.regularization.dropout", "foundations.prob.descriptive", "foundations.backprop.backpropagation.vanishing_exploding"],
    objectives_ar=["ما يُطبَّع بالضبط (كل خاصية/قناة على الدفعة)، ومعلمتا γ وβ المتعلَّمتان.", "سلوك التدريب (إحصاءات الدفعة) مقابل الاستدلال (متوسطات متحركة) ولماذا.", "الفوائد (تدريب أسرع، معدل تعلم أكبر، تنظيم جانبي) والحدود (دفعات صغيرة، تسلسلات)."],
    terms=["standardization", "mean", "variance"],
    difficulty="advanced",
    summary_ar="BN تطبّع z لكل خاصية على الدفعة ثم تعيد تحجيمه بـ γ وβ متعلَّمين. في الاستدلال تستخدم متوسطات متحركة من التدريب.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
z = rng.normal(5.0, 3.0, (32, 4)); z[:, 1] *= 10           # دفعة 32 × 4 خصائص بمقاييس مختلفة
gamma, beta, eps = np.ones(4), np.zeros(4), 1e-5

# التدريب: إحصاءات الدفعة الحالية
mu_b, var_b = z.mean(0), z.var(0)
z_hat = (z - mu_b) / np.sqrt(var_b + eps)
y = gamma * z_hat + beta
print("batch mean per feature :", mu_b.round(2)); print("batch std per feature  :", np.sqrt(var_b).round(2))
print("after BN mean/std      :", y.mean(0).round(3), y.std(0).round(3))

# المتوسطات المتحركة (تُحدَّث كل دفعة في التدريب، وتُستخدم في الاستدلال)
momentum = 0.9; run_mu, run_var = np.zeros(4), np.ones(4)
for step in range(50):
    zb = rng.normal(5.0, 3.0, (32, 4)); zb[:, 1] *= 10
    run_mu = momentum * run_mu + (1 - momentum) * zb.mean(0); run_var = momentum * run_var + (1 - momentum) * zb.var(0)
print("running mean (eval)    :", run_mu.round(2), " running std:", np.sqrt(run_var).round(2))

# الاستدلال: ملاحظة واحدة — لا يمكن حساب إحصاءات دفعة، فنستخدم المتحركة
x1 = rng.normal(5.0, 3.0, (1, 4)); x1[:, 1] *= 10
y_eval = gamma * (x1 - run_mu) / np.sqrt(run_var + eps) + beta
print("single-sample eval output:", y_eval.round(2), " (well-scaled thanks to running stats)")
# لو استخدمنا إحصاءات "دفعة" من ملاحظة واحدة: var = 0 → قسمة على ~0
print("if we used batch stats on 1 sample: var =", x1.var(0).round(3), " -> output would be 0/undefined")'''


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا تفعل؟", "What does it do?")
    definition("**Batch Normalization**: طبقة توضع عادةً بعد الجزء الخطي وقبل التنشيط. لكل خاصية (أو قناة) تحسب متوسط الدفعة وتباينها، تطبّع القيم إلى متوسط 0 وانحراف 1، ثم تعيد التحجيم بمعلمتين **متعلَّمتين** $\\gamma$ (مقياس) و$\\beta$ (إزاحة) حتى لا تفقد الشبكة القدرة على تمثيل أي مقياس تريده.")
    equation(r"\hat z_j = \frac{z_j - \mu_{B,j}}{\sqrt{\sigma^2_{B,j} + \epsilon}}, \qquad y_j = \gamma_j \hat z_j + \beta_j",
             [(r"\mu_{B,j}, \sigma^2_{B,j}", "متوسط وتباين الخاصية $j$ **على الدفعة الحالية** (في التدريب)."), (r"\epsilon", "ثابت صغير للاستقرار."), (r"\gamma_j, \beta_j", "معلمتان لكل خاصية تتعلمهما الشبكة بالتدرج مثل الأوزان.")],
             meaning_ar="توحيد قياسي لكل خاصية داخل الشبكة، مع حرية استعادة أي مقياس.",
             example_ar="خاصية بمتوسط 50 وانحراف 30 على الدفعة تصبح متوسط 0 وانحراف 1، ثم γ = 2، β = 1 تعطي متوسط 1 وانحراف 2.",
             dl_link_ar="`BatchNormalization()` في Keras، `nn.BatchNorm1d/2d` في PyTorch. معلماتها: 2 متعلَّمتان + 2 متحركتان (غير قابلتين للتدريب) لكل خاصية — لهذا ترى «non-trainable params» في الملخص.", title_ar="Batch Normalization")
    intuition("بلا BN، توزيع مدخلات كل طبقة يتغير مع تغير أوزان الطبقات السابقة (تحوّل التوزيع الداخلي)، فتطارد كل طبقة هدفًا متحركًا. BN تثبّت المقياس فتصبح الأسطح أنعم ويمكن استخدام معدل تعلم أكبر.")
    code_lab(CodeLab(
        key="reg_bn", title_ar="BN في التدريب والاستدلال + مشكلة الملاحظة الواحدة", code=CODE,
        before=Before(goal_ar="تطبيع دفعة بإحصاءاتها، تحديث المتوسطات المتحركة، ثم استدلال على ملاحظة واحدة يعتمد عليها.", stage_ar="التنظيم ← BN.",
                      inputs_ar="دفعة 32 × 4 بمقاييس مختلفة جدًا.", expected_ar="بعد BN متوسط 0 وانحراف 1 لكل خاصية؛ متوسطات متحركة تقترب من (5, 50, 5, 5)؛ مخرج استدلال محجّم؛ تباين ملاحظة واحدة = 0."),
        explain=[("6-10", "التدريب: إحصاءات الدفعة نفسها. لاحظ أن الخاصية الثانية (مقياس 10×) تُوحَّد مع الباقي."), ("13-17", "المتوسطات المتحركة تُحدَّث كل دفعة بزخم (0.9–0.99). هذه ما يُحفظ مع النموذج."),
                 ("20-22", "الاستدلال بملاحظة واحدة: نستخدم المتحركة. النتيجة حتمية ومستقلة عن بقية الدفعة."), ("24", "لو استخدمنا إحصاءات الملاحظة الواحدة: تباين 0 — ولهذا **يجب** التبديل إلى وضع eval.")],
        run=run_printed(CODE),
        after_ar="- سلوك مختلف بين الوضعين ليس خللًا بل تصميم: التدريب يستفيد من ضوضاء الدفعة (تنظيم جانبي)، الاستدلال يحتاج حتمية.\n- الدفعات الصغيرة جدًا (< 8) تعطي إحصاءات ضجيج: استخدم LayerNorm أو GroupNorm بدلًا.",
    ))
    compare_table(["الجانب", "التدريب", "الاستدلال"],
                  [("μ, σ²", "من الدفعة الحالية", "المتوسطات المتحركة المحفوظة"), ("الحتمية", "لا (تعتمد على زملاء الدفعة)", "نعم"), ("γ, β", "تُحدَّث بالتدرج", "ثابتة"), ("التبديل", "training=True / model.train()", "training=False / model.eval()")],
                  ["rtl", "rtl", "rtl"])
    research_note("التفسير الأصلي (تقليل «التحوّل الداخلي للتوزيع») نوقش لاحقًا؛ الأدلة الأحدث تشير إلى أن BN تنعّم سطح الخسارة وتحسّن تكييف التدرج. عمليًا الأثر ثابت: تدريب أسرع، معدل تعلم أكبر، حساسية أقل للتهيئة، وتنظيم خفيف.")
    debugging_note("النموذج ممتاز في التدريب وسيئ جدًا عند `predict`؟ تحقق من وضع eval ومن أن المتوسطات المتحركة تدرّبت (حقب كافية) وأن دفعة التدريب لم تكن صغيرة جدًا. مع `batch_size=1` BN غير قابل للاستخدام في التدريب.")
    common_mistake("وضع Dropout **قبل** BN مباشرة: Dropout يغيّر تباين التنشيطات في التدريب فتصبح إحصاءات BN غير متطابقة مع الاستدلال. الترتيب الشائع: Dense → BN → Activation، وDropout بعد التنشيط أو في نهاية الكتلة.")
    quiz("reg.bn", [
        Q("BN تطبّع…", ["كل ملاحظة عبر خصائصها", "كل خاصية عبر الدفعة", "الأوزان"], 1, "على المحور 0."),
        Q("γ وβ…", ["ثابتتان", "متعلَّمتان بالتدرج", "متوسطات متحركة"], 1, "لاستعادة أي مقياس."),
        Q("في الاستدلال تستخدم BN…", ["إحصاءات الدفعة", "المتوسطات المتحركة", "لا شيء"], 1, "حتمية."),
        Q("دفعة من 2 مع BN…", ["ممتازة", "إحصاءات ضجيج: استخدم LayerNorm", "لا فرق"], 1, "عينة صغيرة."),
    ])
    takeaway("BN = توحيد كل خاصية على الدفعة + γ وβ متعلَّمتان. تدريب بإحصاءات الدفعة، استدلال بالمتحركة. تسرّع وتثبّت وتنظّم قليلًا؛ تحتاج دفعات ≥ 16.")
    lesson_footer(LESSON, ["الصيغة والمعلمات الأربع لكل خاصية.", "وضعان بسلوكين.", "الترتيب مع Dropout والتنشيط."])
