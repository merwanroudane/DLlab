import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="course.w02.model_types",
    title_ar="أنواع النماذج: الانحدار والتصنيف بمثال اقتصادي واحد",
    title_en="Model Types: Regression & Classification on One Economic Example",
    module="course.w02",
    order=2,
    prerequisites=["course.w02.overview", "foundations.ml.linear_regression", "foundations.ml.logistic_regression"],
    objectives_ar=["تصنيف المهام: انحدار، تصنيف ثنائي/متعدد، وأنواع أخرى (تجميع، تسلسل) بأمثلة اقتصادية وإدارية.", "تشغيل انحدار خطي وانحدار لوجستي على بيانات حقيقية-الشكل وقراءة معاملاتهما.", "فهم أن كل نموذج = دالة بمعلمات + خسارة + خوارزمية، وأن الشبكة العصبية تعميم لهذين."],
    terms=["model", "target", "numerical", "categorical", "parameter"],
    labs=["labs.neuron_lab"],
    difficulty="beginner",
    summary_ar="الهدف عددي → انحدار (MSE)؛ فئوي → تصنيف (احتمالات + cross-entropy). الانحدار الخطي واللوجستي أبسط نموذجين — والخلية العصبية هي أحدهما بالضبط.",
)

CODE = '''import numpy as np, pandas as pd
from labs.datasets import loan_default, house_prices
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# --- انحدار: سعر العقار من المساحة والغرف والعمر ---
hp = house_prices(n=300, seed=11)
Xr = hp[["area_m2", "rooms", "age_years"]].to_numpy(float); yr = hp["price"].to_numpy(float) / 1000          # بآلاف الدنانير
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=0)
lin = LinearRegression().fit(Xr_tr, yr_tr)
print("REGRESSION  price = b0 + b1*area + b2*rooms + b3*age")
print("  coefficients:", dict(zip(["area_m2", "rooms", "age_years"], lin.coef_.round(2).tolist())), "| intercept:", round(lin.intercept_, 1))
pred = lin.predict(Xr_te)
print(f"  test RMSE = {np.sqrt(((pred - yr_te) ** 2).mean()):.1f} (thousand DZD) | first 3 predictions vs truth: {pred[:3].round(0)} vs {yr_te[:3].round(0)}")

# --- تصنيف: التعثر من الدخل ونسبة الدين والتأخر ---
ld = loan_default(n=400, seed=7).dropna()
Xc = ld[["income", "debt_ratio", "num_late_payments"]].to_numpy(float); yc = ld["defaulted"].to_numpy(int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)
mu, sd = Xc_tr.mean(0), Xc_tr.std(0)                                   # تحجيم بإحصاءات التدريب
log = LogisticRegression().fit((Xc_tr - mu) / sd, yc_tr)
print("\\nCLASSIFICATION  p(default) = sigmoid(b0 + b1*income + b2*debt_ratio + b3*late)")
print("  coefficients (standardized):", dict(zip(["income", "debt_ratio", "late"], log.coef_[0].round(2).tolist())), "| intercept:", round(log.intercept_[0], 2))
p = log.predict_proba((Xc_te - mu) / sd)[:, 1]
print(f"  test accuracy = {((p >= 0.5) == yc_te).mean():.3f} | first 3 probabilities: {p[:3].round(2)} -> classes {(p[:3] >= 0.5).astype(int)} vs truth {yc_te[:3]}")
print("  positive rate in test:", round(yc_te.mean(), 3))'''


def render() -> None:
    lesson_header(LESSON)
    h2("أنواع المهام", "Task types")
    compare_table(["المهمة", "الهدف", "المخرج", "الخسارة النموذجية", "مثال اقتصادي/إداري"],
                  [("انحدار", "عددي متصل", "رقم", "MSE / MAE / Huber", "سعر عقار، مبيعات الشهر القادم، تكلفة مشروع"), ("تصنيف ثنائي", "فئتان", "احتمال واحد", "Binary cross-entropy", "تعثر/سداد، احتيال/سليم، مغادرة عميل"),
                   ("تصنيف متعدد", "K فئات متنافية", "K احتمالات", "Categorical cross-entropy", "تصنيف شكاوى إلى أقسام، تصنيف قطاع الشركة"), ("انحدار متعدد المخرجات", "عدة أرقام", "متجه", "MSE", "توقع الطلب لعدة منتجات معًا"),
                   ("تسلسل → قيمة", "قيمة تالية", "رقم/احتمال", "بحسب الهدف", "التضخم الشهري القادم (الأسبوع 10)"), ("تجميع (غير مُشرف)", "لا هدف", "مجموعات", "—", "تقسيم العملاء إلى شرائح — خارج نطاق المقرر")],
                  ["rtl", "rtl", "rtl", "ltr", "rtl"])
    definition("**نوع المهمة يحدده الهدف** `y`: عددي متصل → انحدار؛ فئوي → تصنيف (ثنائي أو متعدد). ثم يحدد نوعُ المهمة **مخرج النموذج وخسارته ومقياسه** (الأسس 13 و18). كل ما بعد ذلك — خطي أو شبكة عميقة — تفصيل في شكل الدالة.")
    equation(r"\text{Regression: } \hat y = \mathbf{x}^\top \boldsymbol\beta + \beta_0 \qquad\qquad \text{Classification: } \hat p = \sigma(\mathbf{x}^\top \boldsymbol\beta + \beta_0)",
             [(r"\mathbf{x}^\top\boldsymbol\beta + \beta_0", "نفس المجموع الموزون في الحالتين — هو الخلية العصبية (الأسس 9)."), (r"\sigma", "sigmoid تحوّل الرقم إلى احتمال للتصنيف."), (r"\boldsymbol\beta", "المعلمات: تُقدَّر بتقليل الخسارة (MSE أو cross-entropy).")],
             meaning_ar="الانحدار الخطي واللوجستي نموذجان بنفس الجزء الخطي؛ الفرق في التنشيط الأخير والخسارة. الشبكة العصبية تضيف طبقات مخفية قبل هذا الجزء.",
             example_ar="تعثر القروض: p̂ = σ(β₀ + β₁·income + β₂·debt_ratio + β₃·late).", dl_link_ar="`Dense(1)` = انحدار خطي؛ `Dense(1, sigmoid)` = لوجستي؛ أضف `Dense(16, relu)` قبلها = شبكة.", title_ar="النموذجان الأبسط")
    code_lab(CodeLab(
        key="w02_models", title_ar="انحدار خطي (عقارات) وانحدار لوجستي (قروض) بـ scikit-learn", code=CODE, level="C",
        before=Before(goal_ar="حل مسألتي انحدار وتصنيف بأبسط نموذجين، قراءة المعاملات، والتقييم على اختبار محجوز — لنعرف لاحقًا ما الذي تضيفه الشبكة.", stage_ar="الأسبوع 02: أنواع النماذج.",
                      inputs_ar="house_prices (300 صف) وloan_default (400 صف) من مجموعات المنصة.", expected_ar="معاملات الانحدار بوحدات مفسَّرة وRMSE؛ معاملات لوجستية موحَّدة، احتمالات، دقة اختبار ونسبة الإيجابيات.",
                      prerequisites_ar="الأسس 7 (الانحدار)، 8 (التقسيم والتحجيم)."),
        explain=[("6-9", "انحدار: ثلاث خصائص عددية؛ تقسيم 75/25 ببذرة؛ `LinearRegression` تقدّر β بالمربعات الصغرى (نفس هدف MSE)."), ("10-14", "المعاملات بوحدات: ألف دينار لكل م²، لكل غرفة، لكل سنة عمر (سالب). RMSE بوحدة الهدف."),
                 ("17-20", "تصنيف: `stratify` يحفظ نسبة الفئات؛ التحجيم بإحصاءات التدريب فقط (الأسس 8)."), ("21-24", "المعاملات الموحَّدة قابلة للمقارنة بينها: أكبرها قيمةً مطلقة أهم (بحذر). `predict_proba` يعطي الاحتمال، والعتبة 0.5 تعطي الفئة."), ("25-26", "الدقة مقابل نسبة الإيجابيات — قارنها بخط الأساس في الدرس التالي.")],
        run=run_printed(CODE),
        after_ar="- RMSE للانحدار بوحدة الهدف: قل «متوسط خطأ ≈ X ألف دينار» لا «0.87».\n- المعامل السالب للعمر منطقي اقتصاديًا؛ المعاملات هنا **ترابطية** لا سببية (الأسبوع 01).\n- دقة التصنيف بلا نسبة الفئات رقم أعمى — الدرس التالي يبني خط الأساس ومصفوفة التباس.",
    ))
    intuition("النموذجان يشتركان في **كل شيء** عدا سطر واحد (σ) والخسارة. عندما تكتب لاحقًا `Dense(1, activation='sigmoid')` فأنت تكتب الانحدار اللوجستي بلغة Keras.")
    research_note("في الأدبيات الاقتصادية، المعاملات هي المنتج (تفسير)؛ في التعلم الآلي، التنبؤ على بيانات جديدة هو المنتج. الشبكات تكسب الثاني وتخسر شفافية الأول — لذلك نبدأ دائمًا بالخطي كخط أساس **ومرجع تفسير**.")
    common_mistake("ترميز هدف فئوي كأرقام (1، 2، 3 للأقسام) ثم انحدار عليه: النموذج يفترض أن 3 > 2 > 1 وأن الفرق متساوٍ. الفئوي الاسمي → تصنيف بـ one-hot/sparse، لا انحدار.")
    quiz("w02.types", [
        Q("هدف «هل سيغادر العميل خلال 3 أشهر؟»", ["انحدار", "تصنيف ثنائي", "تجميع"], 1, "نعم/لا."),
        Q("الانحدار اللوجستي = الانحدار الخطي +", ["طبقة مخفية", "sigmoid وخسارة cross-entropy", "بيانات أكثر"], 1, "التنشيط والخسارة."),
        Q("RMSE = 28 لسعر بآلاف الدنانير يعني…", ["28%", "خطأ نموذجي ≈ 28 ألف دينار", "28 عقارًا"], 1, "وحدة الهدف."),
        Q("معامل سالب للعمر في الانحدار يثبت…", ["أن العمر يسبب انخفاض السعر", "ارتباطًا في هذه البيانات فقط", "خطأ في البيانات"], 1, "ترابط."),
    ])
    takeaway("الهدف يحدد المهمة، والمهمة تحدد المخرج والخسارة والمقياس. الخطي واللوجستي نموذجان بنفس الجزء الخطي — وهما خط الأساس ومرجع التفسير قبل أي شبكة.")
    lesson_footer(LESSON, ["جدول المهام بأمثلة اقتصادية.", "انحدار وتصنيف بالكود.", "المعاملات ترابطية."])
