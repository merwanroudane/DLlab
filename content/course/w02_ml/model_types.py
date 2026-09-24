import math

import numpy as np
import plotly.graph_objects as go_fig
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, interpretation_note, intuition, math_note, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from content.course.w02_ml._viz import logistic_frame, logistic_trace, simple_models, two_heads_svg
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="course.w02.model_types",
    title_ar="أنواع النماذج: الانحدار والتصنيف بمثال اقتصادي واحد",
    title_en="Model Types: Regression & Classification on One Economic Example",
    module="course.w02",
    order=2,
    prerequisites=["course.w02.overview", "foundations.ml.linear_regression", "foundations.ml.logistic_regression"],
    objectives_ar=[
        "تصنيف المهام: انحدار، تصنيف ثنائي/متعدد، وأنواع أخرى (تجميع، تسلسل) بأمثلة اقتصادية وإدارية.",
        "رؤية أن الانحدار الخطي واللوجستي **عصبون واحد برأسين مختلفين** (هوية أو sigmoid) عبر تحريك بأرقام حقيقية.",
        "قراءة معاملات اللوجستي كلوغاريتم نسبة أرجحية `log-odds` وتحويلها إلى نسبة أرجحية `odds ratio`.",
        "اشتقاق خسارة الإنتروبيا التقاطعية من الاحتمال الأعظم خطوة بخطوة، وفهم لماذا لا نستعمل MSE للتصنيف.",
        "مشاهدة انحدار لوجستي يتدرب: حد القرار يدور ويستقر بينما تنخفض الخسارة.",
    ],
    terms=["model", "target", "numerical", "categorical", "parameter", "regression", "classification", "sigmoid",
           "logit", "cross_entropy", "likelihood", "log_likelihood", "threshold"],
    labs=["labs.neuron_lab", "labs.threshold_lab", "labs.loss_lab"],
    difficulty="beginner",
    summary_ar="الهدف عددي → انحدار (هوية + MSE)؛ فئوي → تصنيف (sigmoid/softmax + cross-entropy). الانحدار الخطي واللوجستي نفس العصبون برأسين. معامل اللوجستي = تغير لوغاريتم الأرجحية؛ e^β = نسبة الأرجحية.",
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


def _bce_steps() -> list[tuple[str, str, str]]:
    l1, l2, l3 = -math.log(0.8), -math.log(0.2), -math.log(0.01)
    return [
        ("**عميل واحد، نتيجة ثنائية**: النموذج يعطي احتمال التعثر `p`. إن تعثر فعلًا (`y = 1`) فاحتمال ما حدث هو `p`، وإن سدد (`y = 0`) فهو `1 − p`. صيغة واحدة تجمع الحالتين (توزيع برنولي).",
         "P(y | p) = p^y · (1 − p)^(1 − y)", r"P(y\mid p) = p^{\,y}\,(1-p)^{\,1-y}"),
        ("**كل العملاء**: بافتراض الاستقلال، احتمال رؤية كل البيانات = حاصل ضرب الاحتمالات. هذا هو **الإمكان** `likelihood`، ونريد معلمات تجعله أكبر ما يمكن.",
         "𝓛 = ∏ᵢ pᵢ^yᵢ (1 − pᵢ)^(1 − yᵢ)", r"\mathcal{L}(\beta) = \prod_{i=1}^{n} p_i^{\,y_i}(1-p_i)^{\,1-y_i}"),
        ("**اللوغاريتم**: حاصل ضرب مئات الاحتمالات الصغيرة يقترب من الصفر عدديًا. اللوغاريتم يحوّل الضرب إلى جمع ولا يغيّر موضع القمة.",
         "log 𝓛 = Σᵢ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ]", r"\log\mathcal{L} = \sum_i \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]"),
        ("**من التعظيم إلى التصغير**: نضرب في −1 ونقسم على n فنحصل على **الإنتروبيا التقاطعية الثنائية** `binary cross-entropy` = log-loss. تصغيرها ⇔ تعظيم الإمكان.",
         "L = −(1/n) Σᵢ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ]", r"L_{\text{BCE}} = -\frac{1}{n}\sum_i\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]"),
        (f"**بالأرقام (y = 1)**: ثقة صحيحة p = 0.8 ← خسارة {l1:.3f}. ثقة خاطئة p = 0.2 ← {l2:.3f}. ثقة خاطئة جدًا p = 0.01 ← **{l3:.2f}**. الخسارة تعاقب **الثقة الخاطئة** بقسوة.",
         f"−log 0.8 = {l1:.3f} ;  −log 0.2 = {l2:.3f} ;  −log 0.01 = {l3:.2f}", rf"-\log 0.8 = {l1:.3f},\quad -\log 0.2 = {l2:.3f},\quad -\log 0.01 = {l3:.2f}"),
        ("**التدرج الأنيق**: مع `p = σ(z)` تختصر قاعدة السلسلة كل شيء إلى «الخطأ»: `∂L/∂z = p − y`. نفس شكل تدرج MSE للانحدار الخطي (الأسبوع 01)!",
         "∂L/∂z = p − y   ⇒   ∂L/∂β = (1/n) Σ (pᵢ − yᵢ) xᵢ", r"\frac{\partial L}{\partial z_i} = p_i - y_i \;\Rightarrow\; \frac{\partial L}{\partial \beta} = \frac{1}{n}\sum_i (p_i-y_i)\,x_i"),
    ]


def render() -> None:
    lesson_header(LESSON)
    h2("أنواع المهام", "Task types")
    compare_table(["المهمة", "الهدف", "المخرج", "الخسارة النموذجية", "مثال اقتصادي/إداري"],
                  [("انحدار", "عددي متصل", "رقم", "MSE / MAE / Huber", "سعر عقار، مبيعات الشهر القادم، تكلفة مشروع"), ("تصنيف ثنائي", "فئتان", "احتمال واحد", "Binary cross-entropy", "تعثر/سداد، احتيال/سليم، مغادرة عميل"),
                   ("تصنيف متعدد", "K فئات متنافية", "K احتمالات", "Categorical cross-entropy", "تصنيف شكاوى إلى أقسام، تصنيف قطاع الشركة"), ("انحدار متعدد المخرجات", "عدة أرقام", "متجه", "MSE", "توقع الطلب لعدة منتجات معًا"),
                   ("تسلسل → قيمة", "قيمة تالية", "رقم/احتمال", "بحسب الهدف", "التضخم الشهري القادم (الأسبوع 10)"), ("تجميع (غير مُشرف)", "لا هدف", "مجموعات", "—", "تقسيم العملاء إلى شرائح — خارج نطاق المقرر")],
                  ["rtl", "rtl", "rtl", "ltr", "rtl"])
    definition("**نوع المهمة يحدده الهدف** `y`: عددي متصل → انحدار؛ فئوي → تصنيف (ثنائي أو متعدد). ثم يحدد نوعُ المهمة **مخرج النموذج وخسارته ومقياسه** (الأسس 13 و18). كل ما بعد ذلك — خطي أو شبكة عميقة — تفصيل في شكل الدالة.")
    compare_table(["سؤال", "انحدار أم تصنيف؟", "لماذا"],
                  [("كم سيبلغ رقم مبيعات الربع القادم؟", "انحدار", "رقم متصل بوحدة (دينار)"),
                   ("هل سيتجاوز رقم المبيعات الهدف؟", "تصنيف ثنائي", "نفس البيانات لكن السؤال نعم/لا"),
                   ("تقييم رضا العميل من 1 إلى 5", "حالة حدّية: ترتيبي `ordinal`", "فئات مرتبة؛ يُعامل كتصنيف أو انحدار بحسب الهدف"),
                   ("عدد الزيارات إلى الفرع", "انحدار عدّي `count`", "أعداد صحيحة غير سالبة؛ خسارة Poisson أنسب من MSE")],
                  ["rtl", "rtl", "rtl"])
    intuition("نفس البيانات قد تخدم مهمتين؛ **السؤال** هو ما يحدد نوع المسألة، لا الجدول.")

    # ------------------------------------------------------------------ two heads
    h2("عصبون واحد، رأسان: الانحدار الخطي واللوجستي", "One neuron, two heads")
    m = simple_models()
    house = [("area_m2", 120.0), ("rooms", 3.0), ("age_years", 10.0)]
    cust_raw = [("income", 4000.0), ("debt_ratio", 0.55), ("late", 3.0)]
    cust = [(n, round((v - m["mu"][i]) / m["sd"][i], 2)) for i, (n, v) in enumerate(cust_raw)]
    z_house = m["lin_b"] + sum(v * c for (_, v), c in zip(house, m["lin_coef"]))
    z_cust = m["log_b"] + sum(v * c for (_, v), c in zip(cust, m["log_coef"]))
    p_cust = 1 / (1 + math.exp(-z_cust))
    caps = [
        ("**انحدار — المدخلات**: عقار مساحته 120 م² بثلاث غرف وعمره 10 سنوات. ثلاث خصائص عددية.", "inputs"),
        ("**الأوزان**: كل خاصية تُضرب في وزنها المتعلَّم من 225 عقارًا (بآلاف الدنانير لكل وحدة). الانحياز يضاف في العقدة Σ.", "× weights"),
        (f"**المجموع الموزون**: `z = {z_house:.1f}` ألف دينار. هذا الجزء **مشترك** بين النموذجين.", "z = Σ wx + b"),
        ("**رأس الانحدار = الهوية**: لا تحويل؛ `ŷ = z` مباشرة لأن الهدف رقم بلا حدود.", "identity head"),
        ("**الخسارة والمقياس**: نقيس الخطأ بالمربع (MSE) أثناء التدريب، ونبلّغ RMSE بوحدة الهدف (ألف دينار).", "MSE / RMSE"),
        ("**تصنيف — المدخلات**: عميل دخله 4000 ونسبة دينه 55% وله 3 تأخرات. نحجّم الخصائص بإحصاءات التدريب (الأرقام بوحدات الانحراف المعياري).", "inputs (standardized)"),
        ("**الأوزان**: معاملات لوجستية متعلَّمة. إشارتها تقول أي اتجاه يرفع خطر التعثر.", "× weights"),
        (f"**المجموع الموزون**: `z = {z_cust:.2f}` — رقم على مقياس **لوغاريتم الأرجحية**، قد يكون سالبًا أو موجبًا بلا حدود.", "z = log-odds"),
        (f"**رأس التصنيف = sigmoid**: `σ(z)` تضغط أي رقم إلى (0، 1): احتمال التعثر `p = {p_cust:.3f}`.", "sigmoid head"),
        (f"**العتبة والخسارة**: الفئة = 1 إذا p ≥ 0.5 (ويمكن تغيير العتبة — الدرس التالي). التدريب يصغّر log-loss لا MSE.", "threshold + log-loss"),
    ]
    frames = []
    for i, (c, act) in enumerate(caps):
        if i < 5:
            svg = two_heads_svg(i, house, m["lin_coef"], m["lin_b"], "regression")
        else:
            svg = two_heads_svg(i - 5, cust, m["log_coef"], m["log_b"], "classification")
        frames.append(Frame(svg, caption(c), action=act, highlight=0 if i < 5 else 1))
    animation_player("w02_heads", frames, title_ar="نفس المجموع الموزون، رأسان مختلفان", stages=["Regression head", "Classification head"], interval_ms=2200)
    equation(r"\text{Regression: } \hat y = \mathbf{x}^\top \boldsymbol\beta + \beta_0 \qquad\qquad \text{Classification: } \hat p = \sigma(\mathbf{x}^\top \boldsymbol\beta + \beta_0)",
             [(r"\mathbf{x}^\top\boldsymbol\beta + \beta_0", "نفس المجموع الموزون في الحالتين — هو الخلية العصبية (الأسس 9)."), (r"\sigma", "sigmoid تحوّل الرقم إلى احتمال للتصنيف."), (r"\boldsymbol\beta", "المعلمات: تُقدَّر بتقليل الخسارة (MSE أو cross-entropy).")],
             meaning_ar="الانحدار الخطي واللوجستي نموذجان بنفس الجزء الخطي؛ الفرق في التنشيط الأخير والخسارة. الشبكة العصبية تضيف طبقات مخفية قبل هذا الجزء.",
             example_ar=f"العقار أعلاه: ŷ ≈ {z_house:.0f} ألف دينار. العميل أعلاه: p ≈ {p_cust:.2f}.", dl_link_ar="`Dense(1)` = انحدار خطي؛ `Dense(1, sigmoid)` = لوجستي؛ أضف `Dense(16, relu)` قبلها = شبكة.", title_ar="النموذجان الأبسط")

    # ------------------------------------------------------------------ sigmoid + odds
    h2("sigmoid ولوغاريتم الأرجحية: كيف نقرأ معاملات اللوجستي", "Sigmoid, log-odds and odds ratios")
    z = st.slider("اسحب z (المجموع الموزون)", -6.0, 6.0, float(round(z_cust, 1)), 0.1, key="w02_z")
    p = 1 / (1 + math.exp(-z))
    zs = np.linspace(-6, 6, 200)
    fig = go_fig.Figure()
    fig.add_scatter(x=zs, y=1 / (1 + np.exp(-zs)), mode="lines", name="σ(z)", line=dict(color="#7C3AED", width=3))
    fig.add_scatter(x=[z], y=[p], mode="markers+text", text=[f"p = {p:.3f}"], textposition="top left", name="you", marker=dict(size=14, color="#DB2777"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#D97706")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="z = log-odds", yaxis_title="p = σ(z)", showlegend=False)
    st.plotly_chart(fig, width="stretch", key="w02_sig_fig")
    c1, c2, c3 = st.columns(3)
    c1.metric("الاحتمال p", f"{p:.3f}")
    c2.metric("الأرجحية p/(1−p)", f"{p / (1 - p):.2f}")
    c3.metric("لوغاريتم الأرجحية z", f"{z:.2f}")
    equation(r"\log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k \quad\Longrightarrow\quad \text{OR}_j = e^{\beta_j}",
             [(r"\frac{p}{1-p}", "الأرجحية `odds`: 3 تعني «احتمال التعثر ثلاثة أضعاف احتمال السداد»."), (r"\log\frac{p}{1-p}", "لوغاريتم الأرجحية `logit`: خطي في الخصائص — لهذا يسمى النموذج «انحدارًا»."),
              (r"e^{\beta_j}", "نسبة الأرجحية `odds ratio`: كم تُضرب الأرجحية عند زيادة الخاصية j وحدة واحدة (هنا: انحرافًا معياريًا واحدًا).")],
             meaning_ar="اللوجستي خطي على مقياس لوغاريتم الأرجحية، وغير خطي على مقياس الاحتمال. لذلك أثر الخاصية على الاحتمال يعتمد على موقعك من المنحنى.",
             example_ar="، ".join(f"{n}: β = {b:+.2f} ⇒ e^β = {math.exp(b):.2f}" for n, b in zip(["income", "debt_ratio", "late"], m["log_coef"])),
             dl_link_ar="آخر طبقة في شبكة تصنيف ثنائي `Dense(1)` تُخرج z (تسمى logit)، ثم sigmoid. الخسارة في Keras يمكن أن تأخذ z مباشرة: `from_logits=True`.", title_ar="من المعامل إلى نسبة الأرجحية")
    interpretation_note(f"نسبة الأرجحية لنسبة الدين {math.exp(m['log_coef'][1]):.2f}: زيادة نسبة الدين بانحراف معياري واحد تضرب أرجحية التعثر في {math.exp(m['log_coef'][1]):.2f} "
                        "مع ثبات الخصائص الأخرى. لاحظ «مع ثبات الأخرى» و«ترابطيًا لا سببيًا» — تذكّر محاكاة الأسبوع 01.")
    common_mistake("قراءة β = 0.8 على أنها «الاحتمال يزيد 0.8». β يزيد **لوغاريتم الأرجحية**؛ أثره على الاحتمال يختلف بحسب نقطة البداية (أكبر قرب p = 0.5، شبه معدوم قرب 0 أو 1). اسحب z أعلاه لترى ذلك.")

    # ------------------------------------------------------------------ derivation of BCE
    h2("لماذا الإنتروبيا التقاطعية؟ اشتقاق من الإمكان الأعظم", "Why cross-entropy? A maximum-likelihood derivation")
    steps = _bce_steps()
    cur = animation_player("w02_bce", [Frame("", caption(c), action=f"step {i + 1}", equation=u) for i, (c, u, _) in enumerate(steps)],
                           title_ar="من توزيع برنولي إلى log-loss", interval_ms=2800)
    with st.container(border=True):
        st.markdown(f"**الخطوة {cur + 1} من {len(steps)}**")
        st.latex(steps[cur][2])
    h3("لماذا لا نستعمل MSE للتصنيف؟", "Why not MSE?")
    ps = np.linspace(0.005, 0.995, 200)
    f2 = go_fig.Figure()
    f2.add_scatter(x=ps, y=-np.log(ps), name="log-loss  −log p", line=dict(color="#DB2777", width=3))
    f2.add_scatter(x=ps, y=(1 - ps) ** 2, name="MSE  (1 − p)²", line=dict(color="#2563EB", width=3, dash="dash"))
    f2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="predicted p when the truth is y = 1", yaxis_title="loss",
                     yaxis_range=[0, 5], legend=dict(orientation="h", x=0, y=1.15))
    st.plotly_chart(f2, width="stretch", key="w02_bce_fig")
    math_note("عندما يكون النموذج **مخطئًا بثقة** (p ≈ 0 والحقيقة 1): MSE لا تتجاوز 1 وتدرجها عبر sigmoid يكاد ينعدم (`σ'(z)` صغيرة جدًا) فيتعلم النموذج ببطء شديد. "
              "log-loss تنفجر نحو اللانهاية وتدرجها بالنسبة لـ z يبقى `p − y` ≈ −1: إشارة تصحيح قوية بالضبط حين نحتاجها. لهذا كل شبكات التصنيف تستعمل الإنتروبيا التقاطعية.")

    # ------------------------------------------------------------------ live logistic training
    h2("شاهد انحدارًا لوجستيًا يتدرب", "Watch a logistic regression train")
    st.markdown("خاصيتان فقط (نسبة الدين وعدد التأخرات، محجّمتان) لنرى كل شيء في مستوى. الخلفية = احتمال التعثر الذي يتنبأ به النموذج (أزرق = سداد، وردي = تعثر)، "
                "والخط الأسود = **حد القرار** حيث p = 0.5 أي `w₁x₁ + w₂x₂ + b = 0`.")
    tr = logistic_trace()
    shown = [0, 1, 2, 3, 5, 8, 12, 20, 35, 50, 80, 120, 150]
    lframes = []
    for idx, k in enumerate(shown):
        prev = shown[idx - 1] if idx else 0
        if k == 0:
            cap = "**البداية**: كل الأوزان صفر ⇒ `z = 0` ⇒ `p = 0.5` لكل العملاء. الخلفية بيضاء بالكامل: النموذج لا يعرف شيئًا، والخسارة = log 2 ≈ 0.693."
        elif k <= 3:
            cap = f"**الحقبة {k}**: التدرج `(p − y)·x` يدفع الوزنين نحو الموجب: العملاء المتعثرون لديهم نسبة دين وتأخرات أعلى. يظهر حد القرار ويبدأ بالدوران."
        else:
            cap = f"**الحقبة {k}**: الحد يستقر؛ الخسارة {tr['loss'][prev]:.3f} ← {tr['loss'][k]:.3f}. التحسن يتباطأ لأننا نقترب من أفضل خط **مستقيم** ممكن — لا يستطيع اللوجستي الانحناء."
        lframes.append(Frame(logistic_frame(tr, k), caption(cap), action=f"epoch {k}",
                             values=[("w₁ (debt)", f"{tr['w1'][prev]:.3f}", f"{tr['w1'][k]:.3f}"), ("w₂ (late)", f"{tr['w2'][prev]:.3f}", f"{tr['w2'][k]:.3f}"),
                                     ("b", f"{tr['b'][prev]:.3f}", f"{tr['b'][k]:.3f}"), ("log-loss", f"{tr['loss'][prev]:.3f}", f"{tr['loss'][k]:.3f}")],
                             equation=f"p = σ({tr['w1'][k]:.2f}·debt + {tr['w2'][k]:.2f}·late + {tr['b'][k]:.2f})", highlight=3))
    animation_player("w02_logtrain", lframes, title_ar="انحدار لوجستي بالنزول بالتدرج (η = 0.1)", stages=["p = σ(z)", "log-loss", "∂L/∂z = p − y", "update w, b"], interval_ms=1100)
    interpretation_note(f"الدقة النهائية على التدريب {tr['acc'][-1]:.3f} بينما نسبة المتعثرين {tr['base_rate']:.1%} — أي أن خط الأساس «الفئة الغالبة» يعطي نحو {max(tr['base_rate'], 1 - tr['base_rate']):.1%}. "
                        "الخطان ليسا بعيدين: بخاصيتين فقط وحد مستقيم، اللوجستي يلتقط جزءًا محدودًا من القصة. الدرس التالي يقيس ذلك بإنصاف على بيانات تحقق.")

    # ------------------------------------------------------------------ code lab
    h2("الآن بالكود: scikit-learn على بيانات أكبر", "Now in code")
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
    compare_table(["", "الانحدار الخطي", "الانحدار اللوجستي", "في Keras"],
                  [("المخرج", "ŷ ∈ ℝ", "p ∈ (0, 1)", "`Dense(1)` / `Dense(1, 'sigmoid')`"), ("الخسارة", "MSE", "binary cross-entropy", "`'mse'` / `'binary_crossentropy'`"),
                   ("التدرج بالنسبة لـ z", "2(ŷ − y)", "p − y", "يُحسب آليًا"), ("حل مغلق؟", "نعم (المربعات الصغرى)", "لا؛ تحسين تكراري", "دائمًا تكراري"),
                   ("قراءة المعامل", "وحدات الهدف لكل وحدة خاصية", "تغير log-odds؛ e^β نسبة أرجحية", "—")],
                  ["rtl", "rtl", "rtl", "ltr"])
    research_note("في الأدبيات الاقتصادية، المعاملات هي المنتج (تفسير)؛ في التعلم الآلي، التنبؤ على بيانات جديدة هو المنتج. الشبكات تكسب الثاني وتخسر شفافية الأول — لذلك نبدأ دائمًا بالخطي كخط أساس **ومرجع تفسير**.")
    common_mistake("ترميز هدف فئوي كأرقام (1، 2، 3 للأقسام) ثم انحدار عليه: النموذج يفترض أن 3 > 2 > 1 وأن الفرق متساوٍ. الفئوي الاسمي → تصنيف بـ one-hot/sparse، لا انحدار.")
    with st.container(horizontal=True):
        st.button("معمل العصبون", icon=":material/science:", on_click=go, args=("labs.neuron_lab",), key="w02_lab_neuron")
        st.button("معمل العتبة", icon=":material/science:", on_click=go, args=("labs.threshold_lab",), key="w02_lab_thr")
        st.button("معمل الخسارة", icon=":material/science:", on_click=go, args=("labs.loss_lab",), key="w02_lab_loss")
    quiz("w02.types", [
        Q("هدف «هل سيغادر العميل خلال 3 أشهر؟»", ["انحدار", "تصنيف ثنائي", "تجميع"], 1, "نعم/لا."),
        Q("الانحدار اللوجستي = الانحدار الخطي +", ["طبقة مخفية", "sigmoid وخسارة cross-entropy", "بيانات أكثر"], 1, "التنشيط والخسارة."),
        Q("RMSE = 28 لسعر بآلاف الدنانير يعني…", ["28%", "خطأ نموذجي ≈ 28 ألف دينار", "28 عقارًا"], 1, "وحدة الهدف."),
        Q("معامل سالب للعمر في الانحدار يثبت…", ["أن العمر يسبب انخفاض السعر", "ارتباطًا في هذه البيانات فقط", "خطأ في البيانات"], 1, "ترابط."),
        Q("معامل لوجستي β = 0.69 لخاصية محجّمة. نسبة الأرجحية ≈", ["0.69", "2.0", "0.5"], 1, "e^0.69 ≈ 2: الأرجحية تتضاعف لكل انحراف معياري."),
        Q("z = 0 في اللوجستي يعني p =", ["0", "0.5", "1"], 1, "σ(0) = 0.5."),
        Q("النموذج قال p = 0.01 والحقيقة y = 1. log-loss لهذه الملاحظة ≈", ["0.01", "0.99", "4.6"], 2, "−log 0.01 ≈ 4.6: ثقة خاطئة مكلفة."),
        Q("تدرج log-loss بالنسبة لـ z يساوي", ["p − y", "(p − y)²", "p(1 − p)"], 0, "الخطأ نفسه."),
    ])
    takeaway("الهدف يحدد المهمة، والمهمة تحدد الرأس والخسارة والمقياس. الخطي واللوجستي نفس العصبون برأسين. اللوجستي خطي في log-odds؛ e^β نسبة أرجحية. log-loss = الإمكان الأعظم وتعاقب الثقة الخاطئة.")
    lesson_footer(LESSON, ["جدول المهام بأمثلة اقتصادية وحالات حدّية.", "عصبون واحد برأسين (تحريك بأرقام حقيقية).", "قراءة المعاملات: log-odds ونسبة الأرجحية.",
                           "اشتقاق log-loss ولماذا لا MSE للتصنيف.", "لوجستي يتدرب أمامك.", "انحدار وتصنيف بالكود؛ المعاملات ترابطية."])
