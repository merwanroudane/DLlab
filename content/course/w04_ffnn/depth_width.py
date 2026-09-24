import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, interpretation_note, intuition, research_note, takeaway, warning_note, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q
from components.week import post_test
from content.course.w04_ffnn._viz import capacity_runs, capacity_svg
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.fw import keras_mlp_run

LESSON = Lesson(
    id="course.w04.depth_width",
    title_ar="بناء نماذج متعددة الطبقات: العمق والعرض بأدلة + الاختبار البعدي",
    title_en="Building Multi-Layer Models: Depth & Width with Evidence — & Post-test",
    module="course.w04",
    order=3,
    prerequisites=["course.w04.forward_flow", "foundations.architecture.depth_width_dense", "foundations.generalization.complexity_data_size"],
    objectives_ar=[
        "رؤية أثر السعة `capacity` على حد القرار: ست بنى من اللوجستي إلى 64-64 على نفس البيانات (تحريك).",
        "حساب نمو المعلمات مع العمق والعرض، وفهم لماذا يكلّف العرض تربيعيًا.",
        "بناء شبكات بأعماق وعروض مختلفة في Keras وقراءة معلماتها وأدائها على التحقق.",
        "اختيار البنية بقاعدة: أبسط ما يصل إلى الأداء المطلوب. والاختبار البعدي.",
    ],
    terms=["hyperparameter", "parameter", "overfitting", "underfitting", "generalization", "layer", "dense_layer"],
    labs=["labs.network_builder", "labs.overfitting_lab", "labs.parameter_counter"],
    difficulty="intermediate",
    summary_ar="العمق والعرض معلمتان فائقتان تُختاران على التحقق. السعة الأكبر تنحني أكثر — وتحفظ أكثر. المعلمات تنمو تربيعيًا مع العرض. ابدأ صغيرًا ووسّع عند القصور فقط.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, keras
from keras import layers, callbacks
from labs.datasets import loan_default
keras.utils.set_random_seed(0)
df = loan_default(n=400, seed=7)
num = df[["income", "age", "num_late_payments", "debt_ratio"]].to_numpy("float32"); cat = pd.get_dummies(df[["city", "employment"]]).to_numpy("float32"); y = df["defaulted"].to_numpy("float32")
idx = np.random.default_rng(0).permutation(400); tr, va = idx[:280], idx[280:]
num = np.where(np.isnan(num), np.nanmedian(num[tr], 0), num); X = np.concatenate([(num - num[tr].mean(0)) / (num[tr].std(0) + 1e-8), cat], 1)

def build(hidden):                                                   # hidden = قائمة عروض الطبقات المخفية
    m = keras.Sequential([layers.Input(shape=(11,))] + [layers.Dense(h, activation="relu") for h in hidden] + [layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(3e-3), loss="binary_crossentropy", metrics=["accuracy"]); return m

print(f"{'architecture':<18} {'params':>7} {'epochs':>7} {'train_acc':>10} {'val_acc':>8} {'gap':>7}")
for hidden in ([], [4], [16], [64], [16, 16], [64, 64], [64, 64, 64]):
    keras.utils.set_random_seed(0); m = build(hidden)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    h = m.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=150, batch_size=32, verbose=0, callbacks=[es])
    tr_acc = m.evaluate(X[tr], y[tr], verbose=0)[1]; va_acc = m.evaluate(X[va], y[va], verbose=0)[1]
    name = "11->" + "->".join(map(str, hidden)) + "->1" if hidden else "logistic (11->1)"
    print(f"{name:<18} {m.count_params():>7} {len(h.history['loss']):>7} {tr_acc:>10.3f} {va_acc:>8.3f} {tr_acc - va_acc:>+7.3f}")
print("baseline (majority):", round(max(y[va].mean(), 1 - y[va].mean()), 3))'''


def render() -> None:
    lesson_header(LESSON)
    why("«كم طبقة؟ كم وحدة؟» لا جواب نظريًا مغلقًا؛ الجواب تجريبي على التحقق. هذا الدرس يُظهر السعة بصريًا أولًا، ثم يجري التجربة على تعثر القروض بسبع بنى ويقرأ المعلمات والحقب والفجوة.")

    # ------------------------------------------------------------------ capacity animation
    h2("السعة بصريًا: ست بنى على نفس البيانات", "Capacity, visually: six architectures, same data")
    cr = capacity_runs()
    ccaps = []
    for r in cr["runs"]:
        a = r["arch"]
        if not a:
            ccaps.append(f"**بلا طبقة مخفية (لوجستي)**: {r['params']} معلمات، حد مستقيم. تحقق {r['val']:.3f}. يلتقط الاتجاه العام فقط — **قصور** بنيوي.")
        elif a == [2] or a == [4]:
            ccaps.append(f"**طبقة من {a[0]} وحدات**: {r['params']} معلمة. وحدات قليلة = مفاصل قليلة؛ الحد يكاد لا ينحني (وقد تموت وحدات ReLU). تحقق {r['val']:.3f}.")
        elif a == [16]:
            ccaps.append(f"**طبقة من 16**: {r['params']} معلمة. الحد ينحني حول الهلالين. تحقق **{r['val']:.3f}** — قفزة واضحة.")
        elif a == [16, 16]:
            ccaps.append(f"**16-16**: {r['params']} معلمة. تدريب {r['train']:.3f} وتحقق {r['val']:.3f}: العمق الإضافي لا يضيف على هذه البيانات.")
        else:
            ccaps.append(f"**64-64**: {r['params']} معلمة (أكثر من 20 ضعف عدد الملاحظات!). الحد يتلوّى حول نقاط فردية: تدريب {r['train']:.3f} وتحقق {r['val']:.3f} — **فجوة** = حفظ للضجيج.")
    animation_player("w04_capacity", [Frame(capacity_svg(cr, i), caption(c), action=("logistic" if not r["arch"] else "-".join(map(str, r["arch"]))),
                                            values=[("params", "", str(r["params"])), ("val acc", "", f"{r['val']:.3f}")])
                                      for i, (c, r) in enumerate(zip(ccaps, cr["runs"]))],
                     title_ar=f"هلالان بضجيج، {cr['n_train']} ملاحظة تدريب، 200 حقبة Adam", interval_ms=2400)
    fig = go.Figure()
    labels = ["logistic" if not r["arch"] else "-".join(map(str, r["arch"])) for r in cr["runs"]]
    fig.add_scatter(x=labels, y=[r["train"] for r in cr["runs"]], mode="lines+markers", name="train", line=dict(color="#7C3AED", width=3))
    fig.add_scatter(x=labels, y=[r["val"] for r in cr["runs"]], mode="lines+markers", name="validation", line=dict(color="#D97706", width=3))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="architecture (growing capacity →)", yaxis_title="accuracy", legend=dict(orientation="h", x=0, y=1.15))
    st.plotly_chart(fig, width="stretch", key="w04_cap_fig")
    best = max(cr["runs"], key=lambda r: r["val"])
    interpretation_note(f"أفضل تحقق عند البنية {'-'.join(map(str, best['arch'])) or 'logistic'} ({best['params']} معلمة). بعدها التدريب يستمر في الصعود والتحقق يتوقف أو يهبط: "
                        "نفس منحنى التعقيد الذي رأيناه مع عمق الشجرة في الأسبوع 02، والسعة هنا = عدد المعلمات وشكل البنية.")

    # ------------------------------------------------------------------ parameter growth
    h2("كم معلمة تضيف طبقة؟", "How many parameters does a layer add?")
    equation(r"P = \sum_{\ell=1}^{L} \big(n_{\ell-1}\, n_\ell + n_\ell\big) \;\approx\; (L-1)\,w^2 \quad\text{(L طبقة بعرض } w\text{)}",
             [(r"n_{\ell}", "عدد وحدات الطبقة ℓ (n₀ = عدد الخصائص)."), (r"w", "عرض ثابت للطبقات المخفية."), (r"L", "عدد الطبقات.")],
             meaning_ar="كل طبقة مخفية داخلية تضيف w² معلمة تقريبًا: مضاعفة العرض تضاعف المعلمات **أربع مرات**، وإضافة طبقة تضيف w² فقط.",
             example_ar="11 خاصية: [16] = 209؛ [64] = 833؛ [64, 64] = 4993؛ [64, 64, 64] = 9153.",
             dl_link_ar="`model.count_params()` يعطي P مباشرة؛ في PyTorch `sum(p.numel() for p in model.parameters())`.", title_ar="نمو المعلمات")
    c1, c2 = st.columns(2)
    with c1:
        width = st.select_slider("العرض w", [4, 8, 16, 32, 64, 128, 256], value=64, key="w04_pw")
    with c2:
        depth = st.slider("عدد الطبقات المخفية", 1, 5, 2, key="w04_pd")
    sizes = [11] + [width] * depth + [1]
    per = [(sizes[i] * sizes[i + 1] + sizes[i + 1]) for i in range(len(sizes) - 1)]
    table(["الطبقة", "الشكل", "المعلمات"], [(f"Dense {i + 1}", f"({sizes[i]} → {sizes[i + 1]})", f"{p:,}") for i, p in enumerate(per)] + [("المجموع", "", f"{sum(per):,}")], ["ltr", "code", "num"])
    warning_note(f"{sum(per):,} معلمة مقابل 280 ملاحظة تدريب في بيانات القروض = {sum(per) / 280:.0f} معلمة لكل ملاحظة. بلا تنظيم وإيقاف مبكر ستحفظ الشبكة البيانات تقريبًا بالكامل.")

    # ------------------------------------------------------------------ the experiment
    h2("التجربة على تعثر القروض: سبع بنى", "The loan-default experiment: seven architectures")
    code_lab(CodeLab(
        key="w04_dw", title_ar="سبع بنى على نفس البيانات: المعلمات، الحقب، دقة التدريب/التحقق، الفجوة", code=CODE, level="C",
        before=Before(goal_ar="مسح منهجي للعمق والعرض مع إيقاف مبكر موحّد، لاختيار البنية بأدلة لا بحدس.", stage_ar="الأسبوع 04: اختيار البنية.",
                      inputs_ar="loan_default (280 تدريب / 120 تحقق)، 11 خاصية.", expected_ar="جدول بسبعة صفوف: المعلمات تتضاعف مع العرض والعمق؛ دقة التحقق متقاربة ضمن ±0.04 (ضوضاء 120 ملاحظة)؛ الفجوة تميل للزيادة مع الحجم.", prerequisites_ar="الأسس 10 و19."),
        explain=[("6-9", "التحضير المعتاد (تقسيم ثم تحجيم بالتدريب)."), ("11-13", "`build(hidden)`: قائمة عروض → Sequential. `[]` = لا طبقات مخفية = انحدار لوجستي."), ("16-22", "لكل بنية: نفس البذرة، نفس الإيقاف المبكر، ثم دقة التدريب والتحقق والفجوة. الحقب الفعلية تخبرك متى توقف."), ("23", "خط الأساس للمقارنة.")],
        run=run_printed(CODE),
        after_ar="- كل البنى بين 0.61 و0.66 تقريبًا على التحقق: مع 120 ملاحظة تحقق فقط هامش الضوضاء ≈ ±0.04 — **لا دليل على ربح من العمق أو العرض** هنا. اللوجستي (12 معلمة) يعادل شبكة بـ 9000 معلمة.\n- الشبكات الكبيرة تتوقف أبكر بكثير (اللوجستي ≈ 70 حقبة، والكبيرة ≈ 10–15): الإيقاف المبكر يكبح الحفظ. الفجوة نفسها **لا تتبع نمطًا واضحًا** هنا — ضوضاء 120 ملاحظة تحقق تغطي على الفروق.\n- **القاعدة**: عند تساوي val_acc ضمن الضوضاء اختر الأبسط: هنا اللوجستي أو [4]. وأبلغ الضوضاء (كرر ببذور مختلفة قبل أي ادعاء).\n- على بيانات أكبر وغير خطية (الهلالان أدناه) ينقلب المشهد — لذلك التجربة لا الحفظ.",
    ))
    h2("مسح تفاعلي على بيانات غير خطية", "Interactive sweep on non-linear data")
    st.markdown("على moons (بيانات غير خطية، 450 تدريب): غيّر العمق والعرض وراقب val_accuracy والمعلمات.")
    c1, c2, c3 = st.columns(3)
    with c1:
        depth = st.slider("عدد الطبقات المخفية", 0, 3, 1, key="w04_depth")
    with c2:
        width = st.select_slider("العرض", options=[2, 4, 8, 16, 32, 64], value=8, key="w04_width")
    with c3:
        epochs = st.slider("epochs", 5, 40, 20, 5, key="w04_epochs")
    try:
        r = keras_mlp_run(tuple([int(width)] * depth), "relu", "adam", 0.01, int(epochs), 32, 0)
    except Exception as exc:  # noqa: BLE001 - TensorFlow missing on this runtime
        r = None
        warning_note(f"TensorFlow غير متاح هنا ({type(exc).__name__}). التحريك أعلاه يعرض نفس الفكرة بمحرك NumPy.")
    if r:
        table(["البنية", "المعلمات", "loss", "val_loss", "val_accuracy"], [("2 → " + " → ".join([str(width)] * depth) + " → 1" if depth else "2 → 1 (لوجستي)", str(r["n_params"]), f"{r['history']['loss'][-1]:.3f}", f"{r['history']['val_loss'][-1]:.3f}", f"{r['history']['val_accuracy'][-1]:.3f}")], ["code", "num", "num", "num", "num"])
        fig = go.Figure(); e = np.arange(1, len(r["history"]["loss"]) + 1)
        fig.add_trace(go.Scatter(x=e, y=r["history"]["loss"], name="loss", line=dict(color="#7C3AED", width=2)))
        fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name="val_loss", line=dict(color="#DB2777", width=3)))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="epoch", legend=dict(orientation="h", x=0, y=1.15))
        st.plotly_chart(fig, width="stretch", key="w04_fig")
    intuition("جرّب العمق 0: اللوجستي يعلق قرب 0.83 على الهلالين مهما زادت الحقب — قصور بنيوي. طبقة واحدة من 8 تكفي للقفز فوق 0.9. طبقتان من 64 لا تضيفان كثيرًا هنا لكنهما تحتاجان بيانات أكثر لتبقيا مستقرتين.")
    h3("العمق أم العرض؟", "Depth or width?")
    compare_table(["", "زيادة العرض", "زيادة العمق"],
                  [("ما يضيفه", "مفاصل/خصائص أكثر في نفس المستوى", "تركيب الخصائص: خصائص من خصائص (هرمية)"), ("كلفة المعلمات", "تربيعية في العرض", "w² لكل طبقة"),
                   ("سهولة التدريب", "سهل", "أصعب: تلاشي التدرج (الأسبوع 06) بلا ReLU/تهيئة جيدة"), ("متى يفيد", "جداول بعلاقات غير خطية بسيطة", "صور ونصوص وأصوات: بنية هرمية طبيعية")],
                  ["rtl", "rtl", "rtl"])
    compare_table(["الملاحظة على التحقق", "التشخيص", "التصرف"],
                  [("قصور: دقة تدريب وتحقق كلاهما منخفض وقريب", "البنية أصغر/أبسط من المسألة (أو η/الحقب غير كافية)", "زد العرض ثم العمق تدريجيًا؛ تحقق من التحجيم"), ("فرط تخصيص: تدريب عالٍ وفجوة كبيرة", "البنية أكبر من البيانات", "صغّر، Dropout/L2، إيقاف مبكر، بيانات أكثر"), ("ثبات: بنى مختلفة بنفس val_acc", "المسألة/الخصائص هي الحد", "لا تزد البنية؛ حسّن الخصائص أو اقبل الحد")],
                  ["rtl", "rtl", "rtl"])
    research_note("قاعدة عملية للجداول الصغيرة: ابدأ بطبقة واحدة بعرض بين عدد الخصائص و4 أضعافها، ثم أضف طبقة إذا بقي قصور واضح. الأعماق الكبيرة تُبرَّر في الصور والنصوص حيث التمثيل الهرمي مفيد (الأسابيع 08–12).")
    common_mistake("مقارنة بنى بعدد حقب ثابت بلا إيقاف مبكر: الشبكة الكبيرة تحفظ خلال نفس الحقب فتبدو أسوأ مما هي (أو أفضل على التدريب). قارن بإيقاف مبكر موحّد على التحقق.")
    common_mistake("إعلان تفوق بنية بفرق 0.01 على تحقق من 120 ملاحظة. كرر التجربة ببذور مختلفة (أو K-fold) وأبلغ المتوسط والانحراف قبل أي ادعاء.")
    with st.container(horizontal=True):
        st.button("معمل بناء الشبكة", icon=":material/science:", on_click=goto, args=("labs.network_builder",), key="w04_lab_nb")
        st.button("معمل فرط التخصيص", icon=":material/science:", on_click=goto, args=("labs.overfitting_lab",), key="w04_lab_of")
    post_test("course.w04", "04", [
        Q("في شبكة أمامية، المعلومات تتدفق…", ["في الاتجاهين", "من المدخل إلى المخرج فقط", "عشوائيًا"], 1, ""),
        Q("Dense(32) على مدخل بـ 10 خصائص: المعلمات", ["320", "352", "42"], 1, "10×32 + 32."),
        Q("طبقة مخفية بلا تنشيط…", ["تزيد القدرة", "تُطوى مع التي بعدها", "تمنع التدريب"], 1, ""),
        Q("شكل مخرج Dense(8) لدفعة من 64:", ["(8,)", "(64, 8)", "(8, 64)"], 1, ""),
        Q("على 280 صفًا، شبكة 64-64-64 مقابل 16:", ["أفضل دائمًا", "غالبًا تحفظ: فجوة أكبر بلا ربح", "لا تُدرَّب"], 1, ""),
        Q("قاعدة اختيار البنية:", ["الأعمق", "أبسط ما يصل إلى الأداء المطلوب على التحقق", "الأكثر معلمات"], 1, ""),
        Q("اللوجستي على الهلالين يعلق قرب 0.83 لأنه…", ["يحتاج حقبًا أكثر", "قصور بنيوي: حد قرار خطي", "η صغير"], 1, ""),
        Q("مضاعفة عرض طبقتين مخفيتين من 64 إلى 128 تضاعف معلمات الطبقة الوسطى…", ["مرتين", "أربع مرات", "لا تغيّرها"], 1, "w²."),
        Q("في تحريك السعة، 64-64: تدريب 0.945 وتحقق 0.895. التشخيص:", ["قصور", "فرط تخصيص", "مثالي"], 1, ""),
    ])
    takeaway("العمق والعرض معلمتان فائقتان: السعة تنحني أكثر وتحفظ أكثر. المعلمات تنمو تربيعيًا مع العرض. مسح موحّد على التحقق، ثم أبسط بنية كافية.")
    lesson_footer(LESSON, ["السعة بصريًا: ست بنى (تحريك) ومنحنى التعقيد.", "نمو المعلمات (حاسبة).", "مسح سبع بنى بالكود.", "مسح تفاعلي على بيانات غير خطية.", "العمق أم العرض، التشخيص، والاختبار البعدي."])
