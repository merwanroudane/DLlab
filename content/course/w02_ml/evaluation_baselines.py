import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, warning_note, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline

LESSON = Lesson(
    id="course.w02.evaluation_baselines",
    title_ar="تقييم الأداء: تدريب/تحقق/اختبار، المقاييس، خطوط الأساس، والتعميم",
    title_en="Evaluating Performance: Train/Validation/Test, Metrics, Baselines & Generalization",
    module="course.w02",
    order=3,
    prerequisites=["course.w02.model_types", "foundations.prep.splitting", "foundations.eval.classification_metrics", "foundations.generalization.under_overfitting"],
    objectives_ar=["تقسيم ثلاثي صحيح ودور كل مجموعة.", "اختيار المقياس بحسب القرار (لا الدقة دائمًا) وبناء خط أساس قبل أي نموذج.", "قياس فجوة التعميم وتشخيص القصور/فرط التخصيص بنموذج شجري متزايد التعقيد."],
    terms=["dataset", "target", "loss"],
    labs=["labs.data_split_lab", "labs.confusion_matrix_lab", "labs.overfitting_lab"],
    difficulty="beginner",
    summary_ar="التدريب للمعلمات، التحقق للقرارات، الاختبار للتقرير مرة واحدة. خط الأساس أولًا. المقياس بحسب كلفة الخطأ. فجوة التدريب−التحقق تشخّص التعميم.",
)

CODE = '''import numpy as np
from labs.datasets import loan_default
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ld = loan_default(n=400, seed=7).dropna()
X = ld[["income", "age", "num_late_payments", "debt_ratio"]].to_numpy(float); y = ld["defaulted"].to_numpy(int)
# 1) تقسيم ثلاثي: 60 / 20 / 20 مع حفظ نسبة الفئات
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)
mu, sd = X_tr.mean(0), X_tr.std(0); Z = lambda A: (A - mu) / sd
print("sizes:", len(X_tr), len(X_va), len(X_te), "| default rate:", y_tr.mean().round(3), y_va.mean().round(3), y_te.mean().round(3))

# 2) خطوط الأساس
majority = np.zeros_like(y_va) + int(y_tr.mean() >= 0.5)
print(f"baseline majority   -> val accuracy {accuracy_score(y_va, majority):.3f} | recall {recall_score(y_va, majority, zero_division=0):.3f}")
rule = (X_va[:, 3] > 0.4).astype(int)                                  # قاعدة خبير: نسبة دين > 40%
print(f"baseline rule       -> val accuracy {accuracy_score(y_va, rule):.3f} | recall {recall_score(y_va, rule):.3f} | precision {precision_score(y_va, rule):.3f}")

# 3) نموذج: انحدار لوجستي، مقاييس متعددة على التحقق
log = LogisticRegression().fit(Z(X_tr), y_tr); p_va = log.predict_proba(Z(X_va))[:, 1]
for thr in (0.5, 0.35):
    pr = (p_va >= thr).astype(int)
    print(f"logistic @thr={thr:<4} -> acc {accuracy_score(y_va, pr):.3f} | precision {precision_score(y_va, pr):.3f} | recall {recall_score(y_va, pr):.3f} | F1 {f1_score(y_va, pr):.3f}")
print("logistic AUC (threshold-free):", round(roc_auc_score(y_va, p_va), 3), "| confusion @0.5 [[TN FP][FN TP]]:", confusion_matrix(y_va, (p_va >= 0.5).astype(int)).tolist())

# 4) التعميم: تعقيد متزايد (عمق شجرة) -> فجوة تدريب/تحقق
print("\\ndepth  train_acc  val_acc  gap")
for depth in (1, 2, 4, 8, None):
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    a_tr, a_va = t.score(X_tr, y_tr), t.score(X_va, y_va)
    print(f"{str(depth):<6} {a_tr:.3f}      {a_va:.3f}    {a_tr - a_va:+.3f}")

# 5) الاختبار: مرة واحدة، للنموذج المختار على التحقق
best = LogisticRegression().fit(Z(X_tr), y_tr)
print(f"\\nFINAL (test, once): accuracy {best.score(Z(X_te), y_te):.3f} | AUC {roc_auc_score(y_te, best.predict_proba(Z(X_te))[:, 1]):.3f}")'''


def render() -> None:
    lesson_header(LESSON)
    pipeline(["Split 60/20/20", "Baseline", "Choose metric", "Train", "Validate & tune", "Test once", "Report"], active=1)
    definition("**ثلاث مجموعات بثلاثة أدوار**: التدريب يقدّر المعلمات؛ التحقق يقيس أثناء التطوير ويختار المعلمات الفائقة والعتبة والنموذج؛ الاختبار يُلمس **مرة واحدة** في النهاية لتقدير الأداء على بيانات جديدة. **خط الأساس** أبسط تنبؤ ممكن (الفئة الغالبة، المتوسط، قاعدة خبير) يجب على أي نموذج التفوق عليه. **المقياس** يُختار بحسب كلفة الخطأ لا بحسب العادة.")
    why("بدون خط أساس لا تعرف إن كانت دقة 66% إنجازًا أو فشلًا؛ بدون تحقق منفصل تختار العتبة والنموذج على الاختبار فتُبلّغ رقمًا متفائلًا؛ بدون فجوة التعميم لا تعرف إن كان النموذج يحفظ أو يتعلم. الثلاثة معًا هي «النزاهة العلمية» في التعلم الآلي.")
    code_lab(CodeLab(
        key="w02_eval", title_ar="تقسيم ثلاثي، خطا أساس، مقاييس بعتبتين، فجوة التعميم بعمق الشجرة، ثم الاختبار مرة واحدة", code=CODE, level="C",
        before=Before(goal_ar="تشغيل منهج التقييم كاملًا على تعثر القروض: خطوط أساس، مقاييس متعددة، تشخيص التعميم عبر تعقيد متزايد، ثم رقم الاختبار النهائي.", stage_ar="الأسبوع 02: التقييم.",
                      inputs_ar="loan_default (394 صفًا بعد حذف المفقود)، 4 خصائص.", expected_ar="أحجام 236/79/79 بنسب فئات متقاربة؛ خط أساس الأغلبية بلا استدعاء؛ قاعدة الخبير بدقة عالية (قد تتفوق على اللوجستي عند 0.5!)؛ اللوجستي بعتبة 0.35 يعطي استدعاء أعلى بكثير؛ فجوة تكبر مع العمق؛ رقم اختبار واحد في النهاية.",
                      prerequisites_ar="الأسس 8، 18، 19."),
        explain=[("10-14", "تقسيم على مرحلتين للحصول على ثلاث مجموعات؛ `stratify` يحفظ نسبة التعثر في الثلاث؛ التحجيم بإحصاءات التدريب فقط."), ("17-20", "خطا أساس: الأغلبية (دقة = نسبة الفئة الغالبة، استدعاء 0) وقاعدة خبير بسيطة. أي نموذج لا يتفوق عليهما بلا قيمة."),
                 ("23-27", "اللوجستي بعتبتين: خفض العتبة إلى 0.35 يرفع الاستدعاء (يلتقط متعثرين أكثر) ويخفض الصحة — القرار بحسب كلفة التفويت (الأسس 18). AUC مستقل عن العتبة. مصفوفة التباس بترتيب [[TN, FP], [FN, TP]]."),
                 ("30-34", "شجرة قرار بعمق متزايد: دقة التدريب ترتفع نحو 1 بينما التحقق يتوقف أو ينخفض — الفجوة = فرط تخصيص (الأسس 19). العمق 1–2 قصور."), ("37-38", "الاختبار **مرة واحدة** للنموذج المختار على التحقق. هذا الرقم هو ما يُبلَّغ.")],
        run=run_printed(CODE),
        after_ar="- الأغلبية تعطي دقة قريبة من نسبة الفئة الغالبة بلا أي استدعاء: الدقة وحدها مضلّلة.\n- **مفاجأة صحية**: قاعدة الخبير (نسبة دين > 40%) تتفوق على اللوجستي في الدقة عند عتبة 0.5. هذا بالضبط لماذا نبني خط الأساس — ولماذا لا نحكم بمقياس واحد: عند 0.35 يلتقط اللوجستي 85% من المتعثرين مقابل 51% للقاعدة. للبنك الذي يخشى التفويت هذا أفضل رغم الصحة الأقل.\n- الشجرة غير المحدودة: دقة تدريب 1.0 وتحقق أقل من اللوجستي — حفظ لا تعلم.\n- رقم الاختبار قد يكون أدنى قليلًا من التحقق: طبيعي؛ التحقق استُخدم للاختيار.",
    ))
    h2("اختيار المقياس بحسب القرار", "Choosing the metric by the decision")
    compare_table(["الحالة", "المقياس المناسب", "لماذا"],
                  [("فئات متوازنة وكلفة الخطأين متساوية", "الدقة", "بسيطة ومفهومة"), ("فئة نادرة (احتيال 1%)", "الصحة/الاستدعاء/F1، AUC-PR", "الدقة تكافئ التنبؤ بالفئة الغالبة"), ("تفويت الإيجابي أغلى (تعثر، مرض)", "الاستدعاء عند صحة مقبولة", "التقط أكثر ولو بإنذارات كاذبة"),
                   ("الإنذار الكاذب أغلى (حظر عميل)", "الصحة عند استدعاء مقبول", "لا تتهم بلا دليل"), ("مقارنة نماذج بلا عتبة", "AUC", "جودة الترتيب"), ("انحدار بقيم متطرفة", "MAE / Huber", "RMSE يضخّم المتطرفة"), ("انحدار يُبلَّغ بوحدة الهدف", "RMSE / MAE", "قابل للتفسير")],
                  ["rtl", "rtl", "rtl"])
    intuition("اسأل دائمًا: «ما الذي يحدث لو أخطأ النموذج في كل اتجاه؟» الجواب يحدد المقياس والعتبة. رقم واحد لا يكفي لمسألة قرار؛ أبلغ مصفوفة التباس أو منحنى.")
    with st.container(horizontal=True):
        st.button("معمل مصفوفة التباس", icon=":material/science:", on_click=go, args=("labs.confusion_matrix_lab",), key="w02_lab_cm")
        st.button("معمل فرط التخصيص", icon=":material/science:", on_click=go, args=("labs.overfitting_lab",), key="w02_lab_over")
        st.button("معمل التقسيم", icon=":material/science:", on_click=go, args=("labs.data_split_lab",), key="w02_lab_split")
    warning_note("**التسريب** يُفسد كل ما سبق: تحجيم قبل التقسيم، خصائص محسوبة من الهدف، أو نفس العميل في التدريب والاختبار. راجع الأسس 8 (التسريب) قبل مشروع الأسبوع 07.")
    common_mistake("ضبط العتبة أو اختيار النموذج على مجموعة الاختبار «لأنها أكبر». الاختبار يُلمس مرة واحدة؛ وإلا صار مجموعة تحقق ثانية ورقمك متفائلًا بلا رقيب.")
    quiz("w02.eval", [
        Q("أي مجموعة تختار عليها العتبة؟", ["التدريب", "التحقق", "الاختبار"], 1, "قرارات التطوير."),
        Q("خط أساس الأغلبية يعطي دقة 0.7 واستدعاء 0. نموذجك دقة 0.72 واستدعاء 0.1:", ["ممتاز", "بالكاد أفضل من لا شيء", "أسوأ"], 1, "قارن بخط الأساس والاستدعاء."),
        Q("دقة تدريب 1.0 وتحقق 0.62:", ["قصور", "فرط تخصيص", "تعميم ممتاز"], 1, "فجوة كبيرة."),
        Q("خفض العتبة من 0.5 إلى 0.35…", ["يرفع الصحة", "يرفع الاستدعاء ويخفض الصحة", "لا يغيّر AUC ولا شيء آخر"], 1, "مقايضة."),
    ])
    takeaway("ثلاث مجموعات بأدوار لا تتبادل. خط أساس قبل أي نموذج. المقياس بحسب كلفة الخطأ. الفجوة تشخّص التعميم. الاختبار مرة واحدة.")
    lesson_footer(LESSON, ["المنهج كاملًا بالكود.", "جدول اختيار المقياس.", "التسريب والاختبار المرة الواحدة."])
