import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, interpretation_note, intuition, research_note, takeaway, warning_note, why
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from content.course.w07_applied._viz import importance_svg, split_variability
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.fw import house_project

LESSON = Lesson(
    id="course.w07.diagnose_interpret",
    title_ar="التقييم والتشخيص والتفسير وتحليل الأخطاء والتقرير + الاختبار البعدي",
    title_en="Evaluation, Diagnostics, Interpretation, Error Analysis & the Report — & Post-test",
    module="course.w07",
    order=3,
    prerequisites=["course.w07.build_pipeline", "foundations.eval.regression_metrics", "foundations.generalization.curves"],
    objectives_ar=[
        "تقييم نهائي على الاختبار بوحدة الهدف مقابل خطي الأساس.",
        "تشخيص التعميم من الفجوة والمنحنيات والبواقي (بما فيها البواقي مقابل التنبؤ).",
        "تفسير الخصائص بأهمية التبديل عمودًا عمودًا (تحريك)، وتحليل أسوأ الأخطاء.",
        "رؤية تذبذب المقارنة بين عشرة تقسيمات مختلفة، وقالب تقرير قابل للاستنساخ، والاختبار البعدي.",
    ],
    terms=["reproducibility", "seed", "generalization", "overfitting", "baseline"],
    labs=["labs.curves_diagnostic_lab"],
    difficulty="intermediate",
    summary_ar="RMSE/MAE بآلاف الدنانير مقابل المتوسط والخطي؛ فجوة تدريب/اختبار وبواقٍ؛ أهمية التبديل (لا سببية)؛ أسوأ 5 أخطاء وأنماطها؛ تذبذب بين التقسيمات؛ تقرير بالبذرة والنسخ والقرارات.",
)


def render() -> None:
    lesson_header(LESSON)
    r = house_project(); t = r["test"]
    h2("6) التقييم على الاختبار — مرة واحدة", "6) Evaluation on the test set — once")
    table(["النموذج", "RMSE (ألف دينار)", "MAE (ألف دينار)", "تحسّن MAE عن المتوسط"],
          [("المتوسط", f"{t['mean_rmse']:.1f}", f"{t['mean_mae']:.1f}", "—"), ("انحدار خطي", f"{t['lin_rmse']:.1f}", f"{t['lin_mae']:.1f}", f"{100 * (1 - t['lin_mae'] / t['mean_mae']):.0f}%"), ("MLP 7→32→16→1", f"{t['mlp_rmse']:.1f}", f"{t['mlp_mae']:.1f}", f"{100 * (1 - t['mlp_mae'] / t['mean_mae']):.0f}%")], ["rtl", "num", "num", "num"])
    gain = t["lin_mae"] - t["mlp_mae"]
    st.markdown(f"**القراءة**: الشبكة تخطئ في المتوسط بـ ≈ {t['mlp_mae']:.0f} ألف دينار لعقار متوسطه ≈ {r['y_mean']:.0f} ألفًا ({100 * t['mlp_mae'] / r['y_mean']:.0f}%). الربح عن الخطي {gain:+.1f} ألفًا في MAE — " + ("ربح حقيقي يستحق التعقيد." if gain > 1.5 else "ربح صغير: الخطي مرشح جيد لأنه أبسط وأشفّ.") + " كل هذه الأرقام على 60 عقارًا محجوزًا لم يُلمسا في أي قرار.")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=r["y_te"], y=r["pred_te"], mode="markers", marker=dict(color="#7C3AED", size=7), name="test"))
        lo, hi = min(r["y_te"]), max(r["y_te"])
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#94A3B8", dash="dot"), name="y = ŷ"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="predicted vs true (test)", xaxis_title="true (k DZD)", yaxis_title="predicted", showlegend=False)
        st.plotly_chart(fig, width="stretch", key="w07_pvt")
    with c2:
        fig2 = go.Figure(go.Histogram(x=r["resid"], nbinsx=20, marker=dict(color="#DB2777")))
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="residuals y − ŷ (test)", xaxis_title="k DZD")
        st.plotly_chart(fig2, width="stretch", key="w07_resid")
    fig4 = go.Figure(go.Scatter(x=r["pred_te"], y=r["resid"], mode="markers", marker=dict(color="#0891B2", size=7)))
    fig4.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
    fig4.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="residuals vs predicted (test)", xaxis_title="predicted (k DZD)", yaxis_title="residual (k DZD)")
    st.plotly_chart(fig4, width="stretch", key="w07_resid_pred")
    interpretation_note("مخطط البواقي مقابل التنبؤ يكشف ما لا يكشفه الهيستوغرام: إن اتسع التشتت مع السعر (شكل قمع) فالأخطاء **نسبية** لا مطلقة — علاجها تحويل log للهدف. إن ظهر منحنى منتظم فالنموذج أخطأ الشكل الوظيفي. المثالي: غيمة أفقية حول الصفر.")

    h2("7) التشخيص", "7) Diagnostics")
    gap = t["mlp_rmse"] - t["train_rmse"]
    table(["المؤشر", "القيمة", "القراءة"],
          [("RMSE تدريب", f"{t['train_rmse']:.1f}", "ما يحفظه النموذج"), ("RMSE اختبار", f"{t['mlp_rmse']:.1f}", "ما يعمّمه"), ("الفجوة", f"{gap:+.1f}", "فرط تخصيص معتدل" if gap > 0.25 * t["train_rmse"] else "تعميم جيد"), ("الحقب الفعلية", str(r["epochs_run"]), "الإيقاف المبكر عمل قبل الحد الأعلى 300"),
           ("البواقي", "متمركزة حول 0؟ ذيول؟", "ذيل يمين = عقارات غالية يقلّل النموذج تقديرها")], ["rtl", "num", "rtl"])
    intuition(f"منحنى التدريب (الدرس السابق) + الفجوة + شكل البواقي = التشخيص الكامل. فجوة معتدلة مع 180 صفًا و{r['n_params']} معلمة متوقعة؛ العلاج إن أردت: Dropout صغير، L2، أو تقليل العرض (الأسس 20).")

    h2("8) التفسير: أهمية التبديل", "8) Interpretation: permutation importance")
    imp = r["importance"]
    icaps = []
    for i, (nm, v) in enumerate(imp):
        icaps.append(f"**نخلط عمود `{nm}`** بين عقارات التحقق (نكسر علاقته بالسعر ونُبقي توزيعه)، ثم نعيد التنبؤ: RMSE يزيد بـ **{v:+.1f}** ألف دينار."
                     + (f" أكبر تدهور: النموذج يعتمد على `{nm}` أكثر من أي خاصية أخرى." if v == max(x[1] for x in imp) else ""))
    animation_player("w07_perm", [Frame(importance_svg(imp, i), caption(c), action=nm) for i, (c, (nm, _)) in enumerate(zip(icaps, imp))],
                     title_ar="أهمية التبديل عمودًا عمودًا (مجموعة التحقق)", interval_ms=1800)
    st.markdown("**الطريقة**: نخلط عمودًا واحدًا في التحقق (نكسر علاقته بالهدف) ونقيس كم يسوء RMSE. كلما كبر التدهور اعتمد النموذج على الخاصية أكثر. **ما تقوله**: على ماذا يعتمد النموذج. **ما لا تقوله**: أن المساحة «تسبب» السعر (الأسبوع 01) أو أن الخصائص المترابطة مستقلة الأثر.")
    warning_note("الخصائص المترابطة (المساحة والغرف) تتقاسم الأهمية: خلط الغرف وحدها يؤذي قليلًا لأن المساحة ما زالت تحمل معلومتها. لا تقرأ أهمية منخفضة على أنها «لا علاقة».")
    research_note("للمقارنة مع التفسير التقليدي: معاملات الانحدار الخطي (الدرس السابق) تعطي اتجاهًا وحجمًا لكل خاصية؛ أهمية التبديل تعطي حجمًا بلا اتجاه لكنها تعمل لأي نموذج. في التقرير قدّم الاثنين وناقش اتفاقهما.")

    h2("9) تحليل الأخطاء", "9) Error analysis")
    table(["#", "الحقيقي", "المتنبأ", "الخطأ", "المساحة", "الغرف", "العمر", "المنطقة"], [(str(i + 1), f"{w['true']:.0f}", f"{w['pred']:.0f}", f"{w['true'] - w['pred']:+.0f}", f"{w['area']:.0f}", str(w["rooms"]), str(w["age"]), w["district"]) for i, w in enumerate(r["worst"])], ["num", "num", "num", "num", "num", "num", "num", "ltr"])
    st.markdown("**اقرأ الأنماط لا الأرقام**: هل أسوأ الأخطاء في منطقة واحدة؟ في العقارات الكبيرة/الغالية (قليلة في التدريب)؟ في القديمة جدًا؟ كل نمط اقتراح: بيانات أكثر من تلك الفئة، خاصية جديدة (تفاعل منطقة×مساحة)، أو تحويل الهدف (log للسعر).")
    warning_note("أسوأ الأخطاء قد تكون **أخطاء في البيانات** (سعر مسجَّل خطأ) لا في النموذج. تحليل الأخطاء يعيدك أحيانًا إلى الأسس 8 (التنظيف) قبل أي تعديل في النموذج.")

    h3("هل الربح حقيقي؟ عشرة تقسيمات مختلفة", "Is the gain real? Ten different splits")
    sv = split_variability()["rows"]
    fv = go.Figure()
    fv.add_scatter(x=[x["split"] for x in sv], y=[x["lin"] for x in sv], mode="lines+markers", name="linear regression", line=dict(color="#2563EB", width=3))
    fv.add_scatter(x=[x["split"] for x in sv], y=[x["mlp"] for x in sv], mode="lines+markers", name="MLP 32-16", line=dict(color="#DB2777", width=3))
    fv.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="random split (seed)", yaxis_title="test MAE (k DZD)", legend=dict(orientation="h", x=0, y=1.15))
    st.plotly_chart(fv, width="stretch", key="w07_splits")
    lm, ls = np.mean([x["lin"] for x in sv]), np.std([x["lin"] for x in sv])
    mm, ms = np.mean([x["mlp"] for x in sv]), np.std([x["mlp"] for x in sv])
    wins = sum(x["mlp"] < x["lin"] for x in sv)
    interpretation_note(f"نفس البيانات، عشرة تقسيمات عشوائية (MLP من scikit-learn بنفس البنية كبديل سريع لنموذج Keras). الخطي: MAE = {lm:.1f} ± {ls:.1f}؛ الشبكة: {mm:.1f} ± {ms:.1f}. "
                        f"الشبكة تفوز في {wins} من 10 تقسيمات — ربح حقيقي في المتوسط، **لكن** تشتتها أكبر، وفي بعض التقسيمات تخسر بوضوح. رقم تقسيم واحد كان سيكفي لرواية أي من القصتين.")

    h2("10) التقرير القابل للاستنساخ", "10) The reproducible report")
    compare_table(["البند", "ما يُكتب"],
                  [("المسألة", "السؤال، الملاحظة، الهدف ووحدته، نوع المهمة، حدود الادعاء"), ("البيانات", "المصدر، الحجم، الأعمدة، المفقود والمتطرف وكيف عولجا، التقسيم (60/20/20، البذرة)"), ("الإعداد", "التوحيد بإحصاءات التدريب، الترميز، تحويل الهدف"),
                   ("خطوط الأساس", "المتوسط والخطي بأرقامهما على الاختبار"), ("النموذج", "البنية والمعلمات، الخسارة، المحسّن وη، الدفعة، الإيقاف المبكر (patience)، البذرة، النسخ (keras/tensorflow)"), ("النتائج", "RMSE/MAE بوحدة الهدف على الاختبار (مرة واحدة)، الفجوة، منحنى التدريب، البواقي"),
                   ("الاستقرار", "المتوسط ± الانحراف عبر عدة بذور أو تقسيمات"),
                   ("التفسير", "أهمية التبديل + معاملات الخطي، مع التحفظ السببي"), ("الأخطاء", "أسوأ 5 وأنماطها واقتراحات"), ("الاستنساخ", "الكود كاملًا، البذرة، الأمر الذي يعيد كل الأرقام")],
                  ["rtl", "rtl"])
    with st.container(horizontal=True):
        st.button("معمل تشخيص المنحنيات", icon=":material/science:", on_click=goto, args=("labs.curves_diagnostic_lab",), key="w07_lab_curves")
        st.button("الأسس 20 — التنظيم", icon=":material/menu_book:", on_click=goto, args=("foundations.regularization",), key="w07_go_reg")
    common_mistake("«الشبكة أفضل من الخطي بـ 2 ألف دينار» من تشغيل واحد ببذرة واحدة على 60 عقارًا. كرر بعدة بذور أو تقسيمات وأبلغ المتوسط ± الانحراف؛ إن تداخلا فلا تدّعِ التفوق.")
    post_test("course.w07", "07", [
        Q("أول نموذج في مشروع:", ["أعمق شبكة", "خط أساس بسيط", "LSTM"], 1, ""),
        Q("RMSE يُبلَّغ…", ["كنسبة", "بوحدة الهدف", "كخسارة موحَّدة"], 1, ""),
        Q("val_loss يرتفع بينما loss ينخفض:", ["قصور", "فرط تخصيص", "تسريب"], 1, ""),
        Q("أهمية التبديل العالية تعني…", ["سببية", "النموذج يعتمد على الخاصية للتنبؤ", "الخاصية مقاسة جيدًا"], 1, ""),
        Q("أسوأ الأخطاء كلها في منطقة واحدة قليلة العينات:", ["زد الطبقات", "بيانات أكثر من تلك المنطقة أو خاصية تفاعل", "أزل المنطقة"], 1, ""),
        Q("الادعاء بتفوق الشبكة يحتاج…", ["تشغيلًا واحدًا", "عدة بذور/تقسيمات ومقارنة بخط أساس على الاختبار", "دقة تدريب عالية"], 1, ""),
        Q("مجموعة الاختبار تُلمس…", ["كل حقبة", "مرة واحدة في النهاية", "لاختيار η"], 1, ""),
        Q("البواقي تتسع مع السعر (شكل قمع):", ["فرط تخصيص", "أخطاء نسبية: جرّب log للهدف", "تسريب"], 1, ""),
        Q("خلط عمود «الغرف» لا يؤذي كثيرًا رغم ارتباطه بالسعر لأن…", ["الغرف بلا علاقة", "المساحة المترابطة تحمل نفس المعلومة", "خطأ في الحساب"], 1, ""),
    ])
    takeaway("تقييم بوحدة الهدف مقابل خطي الأساس؛ فجوة + منحنى + بواقٍ للتشخيص؛ أهمية التبديل للتفسير بلا سببية؛ أسوأ الأخطاء للأنماط؛ عدة تقسيمات قبل أي ادعاء؛ تقرير بالبذرة والنسخ. هذا قالب الأسبوع 14.")
    lesson_footer(LESSON, ["التقييم والبواقي (ثلاثة مخططات).", "التشخيص.", "أهمية التبديل عمودًا عمودًا (تحريك).", "تحليل الأخطاء وتذبذب عشرة تقسيمات.", "قالب التقرير والاختبار البعدي."])
