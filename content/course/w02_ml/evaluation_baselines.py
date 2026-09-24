import math

import numpy as np
import plotly.graph_objects as go_fig
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, interpretation_note, intuition, math_note, research_note, takeaway, warning_note, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from content.course.w02_ml._viz import (cm_at, depth_curve, kfold_scores, kfold_svg, split_frames_data, split_svg, threshold_svg,
                                        tree_surfaces, tree_svg, val_probs)
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
    objectives_ar=[
        "تقسيم ثلاثي طبقي صحيح ودور كل مجموعة — ومتى نستعمل التحقق المتقاطع K-fold بدلًا منه.",
        "بناء خط أساس قبل أي نموذج، واختيار المقياس بحسب كلفة الخطأ لا بحسب العادة.",
        "قراءة مصفوفة الالتباس ومنحنى ROC وتحريك العتبة لرؤية مقايضة الصحة/الاستدعاء حيّة.",
        "قياس فجوة التعميم وتشخيص القصور وفرط التخصيص بشجرة قرار متزايدة العمق.",
    ],
    terms=["dataset", "target", "loss", "train_test_split", "validation_set", "baseline", "accuracy", "precision", "recall",
           "confusion_matrix", "roc_auc", "threshold", "generalization", "overfitting", "underfitting", "data_leakage"],
    labs=["labs.data_split_lab", "labs.confusion_matrix_lab", "labs.overfitting_lab"],
    difficulty="beginner",
    summary_ar="التدريب للمعلمات، التحقق للقرارات، الاختبار للتقرير مرة واحدة. خط الأساس أولًا. المقياس والعتبة بحسب كلفة الخطأ. فجوة التدريب−التحقق تشخّص التعميم.",
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

    # ------------------------------------------------------------------ split animation
    h2("التقسيم الطبقي خطوة بخطوة", "Stratified splitting, step by step")
    sd = split_frames_data()
    caps = [
        ("**40 عميلًا من بيانات القروض** (عيّنة للعرض). الرقم داخل الدائرة هو الهدف: 1 = تعثر (وردي)، 0 = سداد (أزرق).", "data"),
        ("**الخلط أولًا**: البيانات غالبًا مرتبة (بتاريخ التسجيل، بالفرع…). بلا خلط قد يقع كل عملاء فرع واحد في الاختبار. `random_state` يجعل الخلط قابلًا للتكرار.", "shuffle"),
        ("**التقسيم الطبقي** `stratify=y`: نقسم كل فئة على حدة 60/20/20، فتحصل كل مجموعة على نفس نسبة المتعثرين تقريبًا.", "stratify"),
        (f"**التحقق من النسب** على البيانات الكاملة: التدريب {sd['sizes'][0]} عميلًا، التحقق {sd['sizes'][1]}، الاختبار {sd['sizes'][2]}. نسب التعثر شبه متطابقة — هذا ما يضمنه `stratify`.", "check rates"),
    ]
    animation_player("w02_split", [Frame(split_svg(sd, i), caption(c), action=a, highlight=i) for i, (c, a) in enumerate(caps)],
                     title_ar="من جدول واحد إلى ثلاث مجموعات", stages=["Data", "Shuffle", "Stratify", "Check"], interval_ms=2400)
    compare_table(["المجموعة", "من يلمسها؟", "ماذا نقرر بها؟", "كم مرة؟"],
                  [("التدريب", "خوارزمية التحسين", "قيم المعلمات (الأوزان)", "آلاف المرات"),
                   ("التحقق", "الباحث", "المعلمات الفائقة، العتبة، الإيقاف المبكر، اختيار النموذج", "عشرات المرات"),
                   ("الاختبار", "الباحث في النهاية", "لا شيء — فقط التقرير", "مرة واحدة")],
                  ["rtl", "rtl", "rtl", "rtl"])
    warning_note("**بيانات زمنية** (تضخم، مبيعات شهرية): لا خلط! التقسيم يكون **زمنيًا**: الماضي للتدريب، ثم التحقق، ثم الأحدث للاختبار. الخلط يسرّب المستقبل إلى التدريب (الأسبوع 10).")

    # ------------------------------------------------------------------ k-fold
    h3("عندما تكون البيانات قليلة: التحقق المتقاطع K-fold", "K-fold cross-validation")
    scores = kfold_scores()
    kcaps = [f"**الجولة {r + 1}**: الجزء {r + 1} يُحجز للتحقق، والأجزاء الأربعة الأخرى للتدريب. AUC = {s:.3f}." for r, s in enumerate(scores)]
    kcaps[-1] += f" **النتيجة**: المتوسط {np.mean(scores):.3f} ± {np.std(scores):.3f}. التشتت بين الجولات يكشف كم كان رقم تحقق واحد سيخدعنا."
    animation_player("w02_kfold", [Frame(kfold_svg(scores, i), caption(c), action=f"fold {i + 1}") for i, c in enumerate(kcaps)],
                     title_ar="5-fold على التدريب + التحقق (315 عميلًا)", interval_ms=2000)
    equation(r"\text{CV score} = \frac{1}{K}\sum_{k=1}^{K} \text{score}_k \qquad \text{spread} = \operatorname{sd}(\text{score}_1,\dots,\text{score}_K)",
             [("K", "عدد الأجزاء (5 أو 10 عادة)."), (r"\text{score}_k", "مقياس الجولة k على الجزء المحجوز.")],
             meaning_ar="كل ملاحظة تُستعمل للتحقق مرة واحدة بالضبط؛ المتوسط أكثر استقرارًا من تقسيم واحد، والتشتت يعطي حسًّا بعدم اليقين.",
             example_ar=f"هنا: {', '.join(f'{s:.2f}' for s in scores)} ⇒ {np.mean(scores):.3f} ± {np.std(scores):.3f}. الفرق بين أفضل وأسوأ جولة {max(scores) - min(scores):.2f}!",
             dl_link_ar="مع الشبكات العميقة K-fold مكلف (K تدريبات)، فيُستعمل غالبًا مع البيانات الصغيرة فقط؛ والاختبار يبقى محجوزًا في كل الأحوال.", title_ar="التحقق المتقاطع")
    interpretation_note("لاحظ أن AUC تراوح بين الجولات تراوحًا كبيرًا على نفس النموذج ونفس البيانات. مع 79 عميلًا فقط في التحقق، فرق 0.03 بين نموذجين **ليس دليلًا** على تفوق أحدهما.")

    # ------------------------------------------------------------------ baselines
    h2("خط الأساس: «هل تفوّقت على اللاشيء؟»", "Baselines")
    compare_table(["نوع المسألة", "خط الأساس الأبسط", "خط أساس أقوى"],
                  [("تصنيف", "الفئة الغالبة دائمًا", "قاعدة خبير (نسبة دين > 40%)، انحدار لوجستي"),
                   ("انحدار", "متوسط التدريب لكل الملاحظات", "انحدار خطي"),
                   ("سلسلة زمنية", "القيمة الأخيرة (naive): ŷₜ₊₁ = yₜ", "القيمة الموسمية السابقة، ARIMA"),
                   ("صور/نص", "الفئة الغالبة", "نموذج مدرَّب مسبقًا بسيط")],
                  ["rtl", "rtl", "rtl"])
    intuition("خط الأساس ليس إهانة للنموذج بل **مسطرة**. «دقة 66%» بلا مسطرة لا تعني شيئًا؛ «66% مقابل 51% للفئة الغالبة و64% لقاعدة الخبير» جملة علمية.")

    # ------------------------------------------------------------------ threshold explorer
    h2("المقاييس والعتبة: مختبر حي", "Metrics and threshold: a live explorer")
    vp = val_probs()
    st.markdown("انحدار لوجستي على 4 خصائص، والاحتمالات محسوبة لـ 79 عميلًا في **مجموعة التحقق**. العتبة تحوّل الاحتمال إلى قرار: «متعثر» إذا p ≥ العتبة.")
    thrs = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tframes = []
    for t in thrs:
        c = cm_at(vp["p"], vp["y"], t)
        if t <= 0.2:
            cap = f"**عتبة {t:.2f} (متساهلة جدًا)**: نتهم تقريبًا الجميع. الاستدعاء {c['rec']:.2f} (نلتقط كل المتعثرين) لكن الإنذارات الكاذبة FP = {c['fp']}."
        elif t >= 0.8:
            cap = f"**عتبة {t:.2f} (متشددة جدًا)**: لا نتهم إلا الواضحين. FP = {c['fp']} فقط، لكننا نفوّت FN = {c['fn']} متعثرًا — الاستدعاء {c['rec']:.2f}."
        else:
            cap = f"**عتبة {t:.2f}**: TP = {c['tp']}، FP = {c['fp']}، FN = {c['fn']}، TN = {c['tn']}. النقطة الوردية على ROC هي هذه العتبة."
        tframes.append(Frame(threshold_svg(vp, t), caption(cap), action=f"threshold {t:.2f}",
                             values=[("recall", "", f"{c['rec']:.2f}"), ("precision", "", "—" if math.isnan(c["prec"]) else f"{c['prec']:.2f}"), ("FP / FN", "", f"{c['fp']} / {c['fn']}")]))
    animation_player("w02_thr", tframes, title_ar="تحريك العتبة من 0.1 إلى 0.9", interval_ms=1500)
    thr = st.slider("اختر عتبتك", 0.05, 0.95, 0.5, 0.05, key="w02_thr_slider")
    c = cm_at(vp["p"], vp["y"], thr)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{c['acc']:.3f}")
    m2.metric("Precision", "—" if math.isnan(c["prec"]) else f"{c['prec']:.3f}")
    m3.metric("Recall", f"{c['rec']:.3f}")
    m4.metric("F1", f"{c['f1']:.3f}")
    worked_steps([
        (f"مصفوفة الالتباس عند العتبة {thr:.2f}", rf"TP={c['tp']},\ FP={c['fp']},\ FN={c['fn']},\ TN={c['tn']}"),
        ("الدقة `accuracy` = نسبة القرارات الصحيحة", rf"\frac{{TP+TN}}{{n}} = \frac{{{c['tp']}+{c['tn']}}}{{{c['tp'] + c['tn'] + c['fp'] + c['fn']}}} = {c['acc']:.3f}"),
        ("الصحة `precision` = من بين من اتهمناهم، كم كان متعثرًا فعلًا؟", rf"\frac{{TP}}{{TP+FP}} = \frac{{{c['tp']}}}{{{c['tp'] + c['fp']}}}" + ("" if math.isnan(c["prec"]) else rf" = {c['prec']:.3f}")),
        ("الاستدعاء `recall` = من بين المتعثرين فعلًا، كم التقطنا؟", rf"\frac{{TP}}{{TP+FN}} = \frac{{{c['tp']}}}{{{c['tp'] + c['fn']}}} = {c['rec']:.3f}"),
        ("F1 = المتوسط التوافقي للصحة والاستدعاء (يعاقب اختلال أحدهما)", rf"F_1 = \frac{{2PR}}{{P+R}} = {c['f1']:.3f}"),
    ], title_ar="الحساب عند عتبتك")
    math_note(f"AUC = {vp['auc']:.3f} لا تتغير مع العتبة: هي احتمال أن يعطي النموذج متعثرًا عشوائيًا احتمالًا أعلى من مسدّد عشوائي. 0.5 = تخمين، 1.0 = ترتيب تام. "
              "استعملها لمقارنة النماذج، واستعمل العتبة + مصفوفة الالتباس لاتخاذ القرار.")
    h3("اختيار المقياس بحسب القرار", "Choosing the metric by the decision")
    compare_table(["الحالة", "المقياس المناسب", "لماذا"],
                  [("فئات متوازنة وكلفة الخطأين متساوية", "الدقة", "بسيطة ومفهومة"), ("فئة نادرة (احتيال 1%)", "الصحة/الاستدعاء/F1، AUC-PR", "الدقة تكافئ التنبؤ بالفئة الغالبة"), ("تفويت الإيجابي أغلى (تعثر، مرض)", "الاستدعاء عند صحة مقبولة", "التقط أكثر ولو بإنذارات كاذبة"),
                   ("الإنذار الكاذب أغلى (حظر عميل)", "الصحة عند استدعاء مقبول", "لا تتهم بلا دليل"), ("مقارنة نماذج بلا عتبة", "AUC", "جودة الترتيب"), ("انحدار بقيم متطرفة", "MAE / Huber", "RMSE يضخّم المتطرفة"), ("انحدار يُبلَّغ بوحدة الهدف", "RMSE / MAE", "قابل للتفسير")],
                  ["rtl", "rtl", "rtl"])
    h3("العتبة بالتكلفة: مثال بنكي", "Cost-based threshold")
    c_fn = st.number_input("كلفة تفويت متعثر (FN) بآلاف الدنانير", 1, 200, 50, key="w02_cfn")
    c_fp = st.number_input("كلفة رفض عميل جيد (FP) بآلاف الدنانير", 1, 200, 10, key="w02_cfp")
    grid_t = np.round(np.arange(0.05, 0.96, 0.05), 2)
    costs = [cm_at(vp["p"], vp["y"], t)["fn"] * c_fn + cm_at(vp["p"], vp["y"], t)["fp"] * c_fp for t in grid_t]
    best_t = float(grid_t[int(np.argmin(costs))])
    fc = go_fig.Figure()
    fc.add_scatter(x=grid_t, y=costs, mode="lines+markers", line=dict(color="#EA580C", width=3), name="total cost")
    fc.add_vline(x=best_t, line_dash="dot", line_color="#7C3AED")
    fc.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="threshold", yaxis_title="total cost on validation (thousand DZD)", showlegend=False)
    st.plotly_chart(fc, width="stretch", key="w02_cost_fig")
    interpretation_note(f"بهذه الكلف، أفضل عتبة على التحقق ≈ **{best_t:.2f}** — لا 0.5. كلما زادت كلفة التفويت مقارنة بالإنذار الكاذب انخفضت العتبة المثلى. "
                        "العتبة قرار اقتصادي يُتخذ على التحقق، ثم يُقاس أثره مرة واحدة على الاختبار.")

    # ------------------------------------------------------------------ generalization
    h2("التعميم: شجرة تحفظ بدل أن تتعلم", "Generalization: a tree that memorizes")
    ts = tree_surfaces()
    gcaps = []
    for it in ts["items"]:
        dp = it["depth"]
        if dp == 1:
            gcaps.append(f"**العمق 1**: سؤال واحد فقط («هل نسبة الدين أكبر من كذا؟»). حدّ بسيط جدًا: تدريب {it['train']:.3f}، تحقق {it['val']:.3f}. نموذج بسيط لا يحفظ.")
        elif dp <= 4:
            gcaps.append(f"**العمق {dp}**: {it['leaves']} ورقة. المستطيلات تتكاثر. التدريب يرتفع إلى {it['train']:.3f} لكن التحقق {it['val']:.3f} — بدأت الفجوة.")
        else:
            gcaps.append(f"**العمق {'غير محدود' if dp >= 30 else dp}**: {it['leaves']} ورقة — جزر صغيرة حول نقاط فردية. التدريب {it['train']:.3f}، التحقق {it['val']:.3f}: **حفظ الضجيج**. هذا فرط التخصيص.")
    animation_player("w02_tree", [Frame(tree_svg(ts, i), caption(c), action=f"depth {it['depth']}") for i, (c, it) in enumerate(zip(gcaps, ts["items"]))],
                     title_ar="سطح القرار مع تعمّق الشجرة (خاصيتان)", interval_ms=2400)
    dc = depth_curve()
    fd = go_fig.Figure()
    fd.add_scatter(x=dc["depth"], y=dc["train"], mode="lines+markers", name="train accuracy", line=dict(color="#7C3AED", width=3))
    fd.add_scatter(x=dc["depth"], y=dc["val"], mode="lines+markers", name="validation accuracy", line=dict(color="#D97706", width=3))
    best_d = dc["depth"][int(np.argmax(dc["val"]))]
    fd.add_vline(x=best_d, line_dash="dot", line_color="#059669")
    fd.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="tree depth (model complexity)", yaxis_title="accuracy", legend=dict(orientation="h", x=0, y=1.15))
    st.plotly_chart(fd, width="stretch", key="w02_depth_fig")
    compare_table(["المنطقة", "التدريب", "التحقق", "التشخيص", "العلاج"],
                  [("يسار المنحنى", "منخفض", "منخفض", "قصور `underfitting`", "نموذج أعقد، خصائص أفضل"),
                   ("عند أفضل تحقق", "جيد", "الأعلى", "توازن", "اختر هذا التعقيد"),
                   ("يمين المنحنى", "≈ 1.0", "ينخفض", "فرط تخصيص `overfitting`", "تبسيط، تنظيم، بيانات أكثر")],
                  ["rtl", "rtl", "rtl", "rtl", "rtl"])
    interpretation_note(f"أفضل عمق على التحقق هنا = {best_d}. لاحظ أن منحنى التحقق **متعرج**: 79 ملاحظة تحقق فقط تعني ضجيجًا في كل نقطة — سبب إضافي لـ K-fold مع البيانات الصغيرة. "
                        "نفس المنحنى سيظهر مع الشبكات حين يكون «التعقيد» عدد الحقب (الأسبوع 05) أو عدد الوحدات (الأسبوع 04).")

    # ------------------------------------------------------------------ code lab
    h2("المنهج كاملًا بالكود", "The full protocol in code")
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
    with st.container(horizontal=True):
        st.button("معمل مصفوفة التباس", icon=":material/science:", on_click=go, args=("labs.confusion_matrix_lab",), key="w02_lab_cm")
        st.button("معمل فرط التخصيص", icon=":material/science:", on_click=go, args=("labs.overfitting_lab",), key="w02_lab_over")
        st.button("معمل التقسيم", icon=":material/science:", on_click=go, args=("labs.data_split_lab",), key="w02_lab_split")
    warning_note("**التسريب** يُفسد كل ما سبق: تحجيم قبل التقسيم، خصائص محسوبة من الهدف، أو نفس العميل في التدريب والاختبار. راجع الأسس 8 (التسريب) قبل مشروع الأسبوع 07.")
    common_mistake("ضبط العتبة أو اختيار النموذج على مجموعة الاختبار «لأنها أكبر». الاختبار يُلمس مرة واحدة؛ وإلا صار مجموعة تحقق ثانية ورقمك متفائلًا بلا رقيب.")
    common_mistake("إبلاغ الدقة وحدها لمسألة غير متوازنة. أبلغ دائمًا: نسبة الفئة الإيجابية، خط الأساس، مصفوفة الالتباس أو الصحة/الاستدعاء، وAUC.")
    research_note("قالب التقرير: «قسّمنا البيانات طبقيًا 60/20/20 (بذرة 0). اخترنا المعلمات الفائقة والعتبة على التحقق. خط الأساس: … . على الاختبار (مرة واحدة): AUC = …، الاستدعاء = … عند عتبة … .»")
    quiz("w02.eval", [
        Q("أي مجموعة تختار عليها العتبة؟", ["التدريب", "التحقق", "الاختبار"], 1, "قرارات التطوير."),
        Q("خط أساس الأغلبية يعطي دقة 0.7 واستدعاء 0. نموذجك دقة 0.72 واستدعاء 0.1:", ["ممتاز", "بالكاد أفضل من لا شيء", "أسوأ"], 1, "قارن بخط الأساس والاستدعاء."),
        Q("دقة تدريب 1.0 وتحقق 0.62:", ["قصور", "فرط تخصيص", "تعميم ممتاز"], 1, "فجوة كبيرة."),
        Q("خفض العتبة من 0.5 إلى 0.35…", ["يرفع الصحة", "يرفع الاستدعاء ويخفض الصحة", "لا يغيّر AUC ولا شيء آخر"], 1, "مقايضة."),
        Q("لماذا `stratify=y`؟", ["لتسريع التدريب", "لتحافظ كل مجموعة على نسبة الفئات", "لخلط الأعمدة"], 1, ""),
        Q("بيانات تضخم شهرية: كيف نقسم؟", ["خلط عشوائي", "زمنيًا: الماضي تدريب والأحدث اختبار", "طبقيًا"], 1, "لا خلط في السلاسل الزمنية."),
        Q("كلفة FN خمسة أضعاف FP. العتبة المثلى غالبًا…", ["أعلى من 0.5", "أقل من 0.5", "0.5 دائمًا"], 1, "نتهم أسهل لنفوّت أقل."),
        Q("AUC = 0.5 يعني", ["نموذج ممتاز", "ترتيب عشوائي", "كل التنبؤات صحيحة"], 1, ""),
    ])
    takeaway("ثلاث مجموعات بأدوار لا تتبادل؛ K-fold مع البيانات الصغيرة. خط أساس قبل أي نموذج. المقياس والعتبة بحسب كلفة الخطأ. الفجوة تشخّص التعميم. الاختبار مرة واحدة.")
    lesson_footer(LESSON, ["التقسيم الطبقي (تحريك) و K-fold (تحريك).", "خطوط الأساس لكل نوع مسألة.", "مختبر العتبة: مصفوفة التباس، ROC، الصيغ، وعتبة بالتكلفة.",
                           "التعميم: أسطح شجرة تتعمق ومنحنى التعقيد.", "المنهج كاملًا بالكود."])
