import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, research_note, takeaway, warning_note, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
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
    objectives_ar=["تقييم نهائي على الاختبار بوحدة الهدف مقابل خطي الأساس.", "تشخيص التعميم من الفجوة والمنحنيات، وتفسير الخصائص بأهمية التبديل، وتحليل أسوأ الأخطاء.", "قالب تقرير قابل للاستنساخ، والاختبار البعدي."],
    terms=["reproducibility", "seed"],
    labs=["labs.curves_diagnostic_lab"],
    difficulty="intermediate",
    summary_ar="RMSE/MAE بآلاف الدنانير مقابل المتوسط والخطي؛ فجوة تدريب/اختبار؛ أهمية التبديل (لا سببية)؛ أسوأ 5 أخطاء وأنماطها؛ تقرير بالبذرة والنسخ والقرارات.",
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
        fig = go.Figure(go.Scatter(x=r["y_te"], y=r["pred_te"], mode="markers", marker=dict(color="#1F7A78", size=7), name="test"))
        lo, hi = min(r["y_te"]), max(r["y_te"])
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#B9B2A6", dash="dot"), name="y = ŷ"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="predicted vs true (test)", xaxis_title="true (k DZD)", yaxis_title="predicted", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", showlegend=False)
        st.plotly_chart(fig, width="stretch", key="w07_pvt")
    with c2:
        fig2 = go.Figure(go.Histogram(x=r["resid"], nbinsx=20, marker=dict(color="#7C5CBF")))
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="residuals y − ŷ (test)", xaxis_title="k DZD", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
        st.plotly_chart(fig2, width="stretch", key="w07_resid")
    h2("7) التشخيص", "7) Diagnostics")
    gap = t["mlp_rmse"] - t["train_rmse"]
    table(["المؤشر", "القيمة", "القراءة"],
          [("RMSE تدريب", f"{t['train_rmse']:.1f}", "ما يحفظه النموذج"), ("RMSE اختبار", f"{t['mlp_rmse']:.1f}", "ما يعمّمه"), ("الفجوة", f"{gap:+.1f}", "فرط تخصيص معتدل" if gap > 0.25 * t["train_rmse"] else "تعميم جيد"), ("الحقب الفعلية", str(r["epochs_run"]), "الإيقاف المبكر عمل قبل الحد الأعلى 300"),
           ("البواقي", "متمركزة حول 0؟ ذيول؟", "ذيل يمين = عقارات غالية يقلّل النموذج تقديرها")], ["rtl", "num", "rtl"])
    intuition("منحنى التدريب (الدرس السابق) + الفجوة + شكل البواقي = التشخيص الكامل. فجوة معتدلة مع 180 صفًا و801 معلمة متوقعة؛ العلاج إن أردت: Dropout صغير، L2، أو تقليل العرض (الأسس 20).")
    h2("8) التفسير: أهمية التبديل", "8) Interpretation: permutation importance")
    imp = sorted(r["importance"], key=lambda x: -x[1])
    fig3 = go.Figure(go.Bar(x=[v for _, v in imp], y=[n for n, _ in imp], orientation="h", marker=dict(color="#C8473A")))
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="زيادة RMSE عند خلط الخاصية (validation)", xaxis_title="Δ RMSE (k DZD)", yaxis=dict(autorange="reversed"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig3, width="stretch", key="w07_imp")
    st.markdown("**الطريقة**: نخلط عمودًا واحدًا في التحقق (نكسر علاقته بالهدف) ونقيس كم يسوء RMSE. كلما كبر التدهور اعتمد النموذج على الخاصية أكثر. **ما تقوله**: على ماذا يعتمد النموذج. **ما لا تقوله**: أن المساحة «تسبب» السعر (الأسبوع 01) أو أن الخصائص المترابطة مستقلة الأثر.")
    research_note("للمقارنة مع التفسير التقليدي: معاملات الانحدار الخطي (الدرس السابق) تعطي اتجاهًا وحجمًا لكل خاصية؛ أهمية التبديل تعطي حجمًا بلا اتجاه لكنها تعمل لأي نموذج. في التقرير قدّم الاثنين وناقش اتفاقهما.")
    h2("9) تحليل الأخطاء", "9) Error analysis")
    table(["#", "الحقيقي", "المتنبأ", "الخطأ", "المساحة", "الغرف", "العمر", "المنطقة"], [(str(i + 1), f"{w['true']:.0f}", f"{w['pred']:.0f}", f"{w['true'] - w['pred']:+.0f}", f"{w['area']:.0f}", str(w["rooms"]), str(w["age"]), w["district"]) for i, w in enumerate(r["worst"])], ["num", "num", "num", "num", "num", "num", "num", "ltr"])
    st.markdown("**اقرأ الأنماط لا الأرقام**: هل أسوأ الأخطاء في منطقة واحدة؟ في العقارات الكبيرة/الغالية (قليلة في التدريب)؟ في القديمة جدًا؟ كل نمط اقتراح: بيانات أكثر من تلك الفئة، خاصية جديدة (تفاعل منطقة×مساحة)، أو تحويل الهدف (log للسعر).")
    warning_note("أسوأ الأخطاء قد تكون **أخطاء في البيانات** (سعر مسجَّل خطأ) لا في النموذج. تحليل الأخطاء يعيدك أحيانًا إلى الأسس 8 (التنظيف) قبل أي تعديل في النموذج.")
    h2("10) التقرير القابل للاستنساخ", "10) The reproducible report")
    compare_table(["البند", "ما يُكتب"],
                  [("المسألة", "السؤال، الملاحظة، الهدف ووحدته، نوع المهمة، حدود الادعاء"), ("البيانات", "المصدر، الحجم، الأعمدة، المفقود والمتطرف وكيف عولجا، التقسيم (60/20/20، البذرة)"), ("الإعداد", "التوحيد بإحصاءات التدريب، الترميز، تحويل الهدف"),
                   ("خطوط الأساس", "المتوسط والخطي بأرقامهما على الاختبار"), ("النموذج", "البنية والمعلمات، الخسارة، المحسّن وη، الدفعة، الإيقاف المبكر (patience)، البذرة، النسخ (keras/tensorflow)"), ("النتائج", "RMSE/MAE بوحدة الهدف على الاختبار (مرة واحدة)، الفجوة، منحنى التدريب، البواقي"),
                   ("التفسير", "أهمية التبديل + معاملات الخطي، مع التحفظ السببي"), ("الأخطاء", "أسوأ 5 وأنماطها واقتراحات"), ("الاستنساخ", "الكود كاملًا، البذرة، الأمر الذي يعيد كل الأرقام")],
                  ["rtl", "rtl"])
    with st.container(horizontal=True):
        st.button("معمل تشخيص المنحنيات", icon=":material/science:", on_click=goto, args=("labs.curves_diagnostic_lab",), key="w07_lab_curves")
        st.button("الأسس 20 — التنظيم", icon=":material/menu_book:", on_click=goto, args=("foundations.regularization",), key="w07_go_reg")
    common_mistake("«الشبكة أفضل من الخطي بـ 2 ألف دينار» من تشغيل واحد ببذرة واحدة على 60 عقارًا. كرر بـ 5 بذور وأبلغ المتوسط ± الانحراف؛ إن تداخلا فلا تدّعِ التفوق.")
    post_test("course.w07", "07", [
        Q("أول نموذج في مشروع:", ["أعمق شبكة", "خط أساس بسيط", "LSTM"], 1, ""),
        Q("RMSE يُبلَّغ…", ["كنسبة", "بوحدة الهدف", "كخسارة موحَّدة"], 1, ""),
        Q("val_loss يرتفع بينما loss ينخفض:", ["قصور", "فرط تخصيص", "تسريب"], 1, ""),
        Q("أهمية التبديل العالية تعني…", ["سببية", "النموذج يعتمد على الخاصية للتنبؤ", "الخاصية مقاسة جيدًا"], 1, ""),
        Q("أسوأ الأخطاء كلها في منطقة واحدة قليلة العينات:", ["زد الطبقات", "بيانات أكثر من تلك المنطقة أو خاصية تفاعل", "أزل المنطقة"], 1, ""),
        Q("الادعاء بتفوق الشبكة يحتاج…", ["تشغيلًا واحدًا", "عدة بذور ومقارنة بخط أساس على الاختبار", "دقة تدريب عالية"], 1, ""),
        Q("مجموعة الاختبار تُلمس…", ["كل حقبة", "مرة واحدة في النهاية", "لاختيار η"], 1, ""),
    ])
    takeaway("تقييم بوحدة الهدف مقابل خطي الأساس؛ فجوة + منحنى + بواقٍ للتشخيص؛ أهمية التبديل للتفسير بلا سببية؛ أسوأ الأخطاء للأنماط؛ تقرير بالبذرة والنسخ. هذا قالب الأسبوع 14.")
    lesson_footer(LESSON, ["التقييم والتشخيص بالأرقام.", "التفسير وتحليل الأخطاء.", "قالب التقرير والاختبار البعدي."])
