import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.ml.baseline_generalization",
    title_ar="خط الأساس والتعميم والتقسيم تدريب/تحقق/اختبار",
    title_en="Baseline, Generalization & the Train/Validation/Test Split",
    module="foundations.ml",
    order=4,
    prerequisites=["foundations.ml.objective_loss_metric"],
    objectives_ar=[
        "بناء خط أساس تافه (متوسط، الفئة الأغلب) ومقارنة أي نموذج به.",
        "تعريف التعميم كأداء على بيانات لم يرها النموذج، ولماذا نحتاج ثلاث مجموعات لا اثنتين.",
        "قراءة الفجوة تدريب/تحقق كأول إشارة تشخيصية.",
    ],
    terms=["observation", "loss", "hyperparameter"],
    labs=["labs.data_split_lab"],
    difficulty="beginner",
    summary_ar="خط الأساس يحدد «هل تعلمنا شيئًا؟»؛ التعميم يُقاس على بيانات محجوزة؛ التحقق للضبط والاختبار للتقرير.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
n = 1000
y = (rng.uniform(size=n) < 0.12).astype(int)            # 12% تعثر: عدم توازن
x = rng.normal(size=(n, 3))

# خط أساس للتصنيف: تنبأ دائمًا بالفئة الأغلب
majority = np.bincount(y).argmax()
print("majority class =", majority, " baseline accuracy =", (y == majority).mean().round(3))

# خط أساس للانحدار: تنبأ دائمًا بالمتوسط
price = rng.normal(200_000, 50_000, n)
baseline_rmse = np.sqrt(((price - price.mean())**2).mean())
print("regression baseline RMSE =", baseline_rmse.round(0), "(= std of target)")

# تقسيم 70/15/15 بعد الخلط
idx = rng.permutation(n)
n_tr, n_va = int(0.7*n), int(0.15*n)
tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
print("sizes:", len(tr), len(va), len(te), " overlap:", len(set(tr) & set(te)))
print("default rate per split:", y[tr].mean().round(3), y[va].mean().round(3), y[te].mean().round(3))'''


def render() -> None:
    lesson_header(LESSON)
    h2("خط الأساس", "Baseline")
    definition("**خط الأساس** `Baseline` أبسط نموذج ممكن: للتصنيف «تنبأ دائمًا بالفئة الأغلب»، وللانحدار «تنبأ دائمًا بالمتوسط». أي نموذج لا يتفوق عليه **لم يتعلم شيئًا** مهما بدت أرقامه.")
    intuition("دقة 88% في التنبؤ بالتعثر تبدو رائعة — حتى تعرف أن 88% من العملاء لا يتعثرون أصلًا. خط الأساس يحقق 88% بلا أي تعلم.")
    h2("التعميم", "Generalization")
    definition("**التعميم** `Generalization` قدرة النموذج على الأداء الجيد على **بيانات جديدة** لم يرها في التدريب. هذا هو الهدف الحقيقي؛ الأداء على بيانات التدريب قد يكون حفظًا.")
    pipeline(["All data", "shuffle", "Train 70%", "Validation 15%", "Test 15%"], active=2)
    st.markdown("""
- **التدريب** `Train`: يتعلم منه النموذج المعلمات.
- **التحقق** `Validation`: نقيّم عليه أثناء التطوير لاختيار المعلمات الفائقة والإيقاف المبكر. **يتأثر باختياراتنا**، فلا يصلح للتقرير النهائي.
- **الاختبار** `Test`: يُفتح مرة واحدة في النهاية. يعطي تقديرًا غير متحيز للأداء على بيانات جديدة.

لماذا ثلاث لا اثنتان؟ لأنك إن اخترت أفضل نموذج من عشرة بالنظر إلى «الاختبار»، فقد اخترت الأنسب لتلك العينة تحديدًا — والرقم متفائل.
""")
    code_lab(CodeLab(
        key="ml_baseline", title_ar="خطا أساس + تقسيم ثلاثي مع فحص التسريب والتوازن", code=CODE,
        before=Before(goal_ar="حساب خط أساس للتصنيف وللانحدار، ثم تقسيم البيانات ثلاثيًا والتحقق من عدم التداخل ومن تشابه نسب الفئات.", stage_ar="ML ← إعداد البيانات.",
                      inputs_ar="1000 ملاحظة مولّدة بنسبة تعثر 12%.", expected_ar="خط أساس 88%؛ RMSE = الانحراف المعياري؛ أحجام 700/150/150 بلا تداخل ونسب متقاربة."),
        explain=[("7-9", "الفئة الأغلب ونسبتها = دقة خط الأساس. اكتبها في أول سطر من أي تقرير."),
                 ("11-14", "للانحدار، التنبؤ بالمتوسط يعطي RMSE يساوي الانحراف المعياري للهدف — أي نموذج يجب أن يهزمه."),
                 ("16-19", "الخلط ثم التقطيع. `overlap = 0` شرط لا يُناقش."),
                 ("21", "نسب الفئات متقاربة بالصدفة هنا؛ مع بيانات أصغر أو أكثر اختلالًا نحتاج **تقسيمًا طبقيًا** `stratified` يضمن ذلك.")],
        run=run_printed(CODE),
        after_ar="- إن كانت دقة نموذجك 89% وخط الأساس 88% فقد تعلمت 1% فقط — وربما لا شيء إحصائيًا.\n- نسبة التعثر في الاختبار قد تختلف عن التدريب؛ التقسيم الطبقي يثبّتها.",
    ))
    h2("الفجوة الأولى للتشخيص", "The first diagnostic gap")
    st.markdown("""
| خسارة التدريب | خسارة التحقق | التشخيص الأولي |
|---|---|---|
| منخفضة | منخفضة وقريبة | صحي |
| منخفضة | مرتفعة | **فرط تخصيص** `Overfitting`: حفظ لا تعلّم |
| مرتفعة | مرتفعة وقريبة | **قصور تعلّم** `Underfitting`: قدرة ناقصة أو تدريب غير كافٍ |
| مرتفعة | أقل من التدريب | غالبًا Dropout/تنظيم يعمل في التدريب فقط، أو تسريب في التحقق |
""")
    research_note("في السلاسل الزمنية **لا تخلط قبل التقسيم**: التدريب من الماضي والاختبار من المستقبل، وإلا تسرّب المستقبل إلى النموذج. سنعود لهذا في الأسبوع 10.")
    common_mistake("فتح مجموعة الاختبار مبكرًا «لمجرد النظر» ثم تعديل النموذج. بعد ذلك لم تعد اختبارًا؛ صارت تحققًا ثانيًا.")
    quiz("ml.baseline", [
        Q("فئة أغلب 95%. نموذج بدقة 95%…", ["ممتاز", "يساوي خط الأساس: لم يتعلم", "فرط تخصيص"], 1, "قارن دائمًا بخط الأساس."),
        Q("على أي مجموعة نختار عدد الحقب الأفضل؟", ["التدريب", "التحقق", "الاختبار"], 1, "التحقق للضبط."),
        Q("خسارة تدريب 0.05 وتحقق 0.9…", ["قصور تعلم", "فرط تخصيص", "صحي"], 1, "فجوة كبيرة = حفظ."),
        Q("سلسلة زمنية: التقسيم الصحيح…", ["خلط ثم تقطيع", "الماضي تدريب والمستقبل اختبار", "لا فرق"], 1, "منع تسريب المستقبل."),
    ])
    takeaway("اهزم خط الأساس أولًا. التعميم يُقاس على بيانات محجوزة: تحقق للضبط، اختبار للتقرير مرة واحدة. الفجوة تدريب/تحقق أول تشخيص.")
    lesson_footer(LESSON, ["خط الأساس = الفئة الأغلب / المتوسط.", "ثلاث مجموعات لثلاثة أدوار.", "الفجوة تخبرك: فرط تخصيص أم قصور تعلم."])
