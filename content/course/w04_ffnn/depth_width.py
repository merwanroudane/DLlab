import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from core.rtl import table
from labs.fw import keras_mlp_run

LESSON = Lesson(
    id="course.w04.depth_width",
    title_ar="بناء نماذج متعددة الطبقات: العمق والعرض بأدلة + الاختبار البعدي",
    title_en="Building Multi-Layer Models: Depth & Width with Evidence — & Post-test",
    module="course.w04",
    order=3,
    prerequisites=["course.w04.forward_flow", "foundations.architecture.depth_width_dense", "foundations.generalization.complexity_data_size"],
    objectives_ar=["بناء شبكات بأعماق وعروض مختلفة في Keras وقراءة معلماتها وأدائها على التحقق.", "اختيار البنية بقاعدة: أبسط ما يصل إلى الأداء المطلوب.", "الاختبار البعدي."],
    terms=["hyperparameter", "parameter"],
    labs=["labs.network_builder", "labs.overfitting_lab"],
    difficulty="intermediate",
    summary_ar="العمق والعرض معلمتان فائقتان تُختاران على التحقق. أكثر ≠ أفضل: على بيانات صغيرة، الشبكة الأكبر تحفظ. ابدأ صغيرًا ووسّع عند القصور فقط.",
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
    why("«كم طبقة؟ كم وحدة؟» لا جواب نظريًا مغلقًا؛ الجواب تجريبي على التحقق. هذا الدرس يجري التجربة على تعثر القروض بسبع بنى — من الانحدار اللوجستي إلى ثلاث طبقات من 64 — ويقرأ المعلمات والحقب والفجوة.")
    code_lab(CodeLab(
        key="w04_dw", title_ar="سبع بنى على نفس البيانات: المعلمات، الحقب، دقة التدريب/التحقق، الفجوة", code=CODE, level="C",
        before=Before(goal_ar="مسح منهجي للعمق والعرض مع إيقاف مبكر موحّد، لاختيار البنية بأدلة لا بحدس.", stage_ar="الأسبوع 04: اختيار البنية.",
                      inputs_ar="loan_default (280 تدريب / 120 تحقق)، 11 خاصية.", expected_ar="جدول بسبعة صفوف: المعلمات تتضاعف مع العرض والعمق؛ دقة التحقق متقاربة ضمن ±0.04 (ضوضاء 120 ملاحظة)؛ الفجوة تميل للزيادة مع الحجم.", prerequisites_ar="الأسس 10 و19."),
        explain=[("6-9", "التحضير المعتاد (تقسيم ثم تحجيم بالتدريب)."), ("11-13", "`build(hidden)`: قائمة عروض → Sequential. `[]` = لا طبقات مخفية = انحدار لوجستي."), ("16-22", "لكل بنية: نفس البذرة، نفس الإيقاف المبكر، ثم دقة التدريب والتحقق والفجوة. الحقب الفعلية تخبرك متى توقف."), ("23", "خط الأساس للمقارنة.")],
        run=run_printed(CODE),
        after_ar="- كل البنى بين 0.61 و0.66 على التحقق: مع 120 ملاحظة تحقق فقط هامش الضوضاء ≈ ±0.04 — **لا دليل على ربح من العمق أو العرض** هنا. اللوجستي (12 معلمة) يعادل شبكة بـ 9000 معلمة.\n- الفجوة (تدريب − تحقق) تميل إلى الزيادة مع المعلمات، والشبكات الكبيرة تتوقف أبكر (الإيقاف المبكر يعمل).\n- **القاعدة**: عند تساوي val_acc ضمن الضوضاء اختر الأبسط: هنا اللوجستي أو [4]. وأبلغ الضوضاء (كرر ببذور مختلفة قبل أي ادعاء).\n- على بيانات أكبر وغير خطية (الهلالان أدناه) ينقلب المشهد — لذلك التجربة لا الحفظ.",
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
    r = keras_mlp_run(tuple([int(width)] * depth), "relu", "adam", 0.01, int(epochs), 32, 0)
    table(["البنية", "المعلمات", "loss", "val_loss", "val_accuracy"], [("2 → " + " → ".join([str(width)] * depth) + " → 1" if depth else "2 → 1 (لوجستي)", str(r["n_params"]), f"{r['history']['loss'][-1]:.3f}", f"{r['history']['val_loss'][-1]:.3f}", f"{r['history']['val_accuracy'][-1]:.3f}")], ["code", "num", "num", "num", "num"])
    fig = go.Figure(); e = np.arange(1, len(r["history"]["loss"]) + 1)
    fig.add_trace(go.Scatter(x=e, y=r["history"]["loss"], name="loss", line=dict(color="#2F6FB5", width=2)))
    fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name="val_loss", line=dict(color="#C8473A", width=3)))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="epoch", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="w04_fig")
    intuition("جرّب العمق 0: اللوجستي يعلق قرب 0.83 على الهلالين مهما زادت الحقب — قصور بنيوي. طبقة واحدة من 8 تكفي للقفز فوق 0.9. طبقتان من 64 لا تضيفان كثيرًا هنا لكنهما تحتاجان بيانات أكثر لتبقيا مستقرتين.")
    compare_table(["الملاحظة على التحقق", "التشخيص", "التصرف"],
                  [("قصور: دقة تدريب وتحقق كلاهما منخفض وقريب", "البنية أصغر/أبسط من المسألة (أو η/الحقب غير كافية)", "زد العرض ثم العمق تدريجيًا؛ تحقق من التحجيم"), ("فرط تخصيص: تدريب عالٍ وفجوة كبيرة", "البنية أكبر من البيانات", "صغّر، Dropout/L2، إيقاف مبكر، بيانات أكثر"), ("ثبات: بنى مختلفة بنفس val_acc", "المسألة/الخصائص هي الحد", "لا تزد البنية؛ حسّن الخصائص أو اقبل الحد")],
                  ["rtl", "rtl", "rtl"])
    research_note("قاعدة عملية للجداول الصغيرة: ابدأ بطبقة واحدة بعرض بين عدد الخصائص و4 أضعافها، ثم أضف طبقة إذا بقي قصور واضح. الأعماق الكبيرة تُبرَّر في الصور والنصوص حيث التمثيل الهرمي مفيد (الأسابيع 08–12).")
    common_mistake("مقارنة بنى بعدد حقب ثابت بلا إيقاف مبكر: الشبكة الكبيرة تحفظ خلال نفس الحقب فتبدو أسوأ مما هي (أو أفضل على التدريب). قارن بإيقاف مبكر موحّد على التحقق.")
    post_test("course.w04", "04", [
        Q("في شبكة أمامية، المعلومات تتدفق…", ["في الاتجاهين", "من المدخل إلى المخرج فقط", "عشوائيًا"], 1, ""),
        Q("Dense(32) على مدخل بـ 10 خصائص: المعلمات", ["320", "352", "42"], 1, ""),
        Q("طبقة مخفية بلا تنشيط…", ["تزيد القدرة", "تُطوى مع التي بعدها", "تمنع التدريب"], 1, ""),
        Q("شكل مخرج Dense(8) لدفعة من 64:", ["(8,)", "(64, 8)", "(8, 64)"], 1, ""),
        Q("على 280 صفًا، شبكة 64-64-64 مقابل 16:", ["أفضل دائمًا", "غالبًا تحفظ: فجوة أكبر بلا ربح", "لا تُدرَّب"], 1, ""),
        Q("قاعدة اختيار البنية:", ["الأعمق", "أبسط ما يصل إلى الأداء المطلوب على التحقق", "الأكثر معلمات"], 1, ""),
        Q("اللوجستي على الهلالين يعلق قرب 0.83 لأنه…", ["يحتاج حقبًا أكثر", "قصور بنيوي: حد قرار خطي", "η صغير"], 1, ""),
    ])
    takeaway("العمق والعرض معلمتان فائقتان: مسح موحّد على التحقق، ثم أبسط بنية كافية. الجدول الصغير يبدأ بطبقة واحدة؛ اللاخطية تحتاج طبقة على الأقل؛ الحجم الزائد يحفظ.")
    lesson_footer(LESSON, ["مسح سبع بنى بالكود.", "مسح تفاعلي على بيانات غير خطية.", "جدول التشخيص والاختبار البعدي."])
