import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, research_note, takeaway, warning_note, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from components.animation_player import Frame, animation_player, caption
from content.course.w10_rnn._viz import forecast_svg, split_svg
from core.models import Lesson
from core.rtl import table
from labs.fw import keras_seq_run

LESSON = Lesson(
    id="course.w10.time_series_application",
    title_ar="تطبيق RNN على سلسلة زمنية في Keras + الاختبار البعدي",
    title_en="Applying an RNN to a Time Series in Keras — & Post-test",
    module="course.w10",
    order=4,
    prerequisites=["course.w10.rnn_bptt_vanishing", "course.w07.build_pipeline"],
    objectives_ar=["خط أنابيب زمني كامل: توحيد بإحصاءات الماضي، نوافذ، تقسيم زمني، SimpleRNN، تقييم بوحدة الهدف مقابل خط الأساس الساذج.", "قراءة النتيجة الصادقة: متى لا يهزم RNN «القيمة الأخيرة» ولماذا.", "الاختبار البعدي."],
    terms=["time_series", "sequence", "standardization"],
    labs=["labs.rnn_unrolling_lab"],
    difficulty="intermediate",
    summary_ar="SimpleRNN(16) + Dense(1) على نوافذ 12 شهرًا. على سلسلة قصيرة شبه عشوائية (التضخم) خط الأساس الساذج يصعب هزمه؛ على سلسلة ذات موسمية واضحة يتفوق RNN بوضوح. الحكم دائمًا بالمقارنة على اختبار زمني.",
)

CODE = '''# الكود الذي تشغّله هذه الصفحة (labs/fw.py: keras_seq_run)
s = monthly_inflation()["inflation"].to_numpy("float32")       # 180 شهرًا
mu, sd = s[:120].mean(), s[:120].std(); z = (s - mu) / sd         # توحيد بإحصاءات أول 120 شهرًا فقط (الماضي)
X, y = make_windows(z, window=12)                                 # (168, 12, 1), (168,)
X_tr, y_tr = X[:108], y[:108]                                     # زمنيًا: نوافذ تنتهي قبل الشهر 120
X_va, y_va = X[108:132], y[108:132]                               # 24 شهرًا تحقق
X_te, y_te = X[132:], y[132:]                                     # آخر 36 شهرًا اختبار
model = keras.Sequential([layers.Input(shape=(12, 1)), layers.SimpleRNN(16), layers.Dense(1)])   # (12,1)->(16,)->(1,)
model.compile(optimizer=keras.optimizers.Adam(5e-3), loss="mse")
model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=40, batch_size=16, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
pred = model.predict(X_te).ravel() * sd + mu                     # عكس التوحيد
naive = X_te[:, -1, 0] * sd + mu                                  # خط الأساس: القيمة الأخيرة في النافذة
rmse_model, rmse_naive = rmse(pred, y_te * sd + mu), rmse(naive, y_te * sd + mu)'''


def render() -> None:
    lesson_header(LESSON)
    why("كل ما سبق نظري حتى تُدرَّب RNN على سلسلة حقيقية-الشكل وتُقارَن بأبسط تنبؤ ممكن: «الشهر القادم = هذا الشهر». النتيجة تعلّم أكثر من أي معادلة.")
    scaps = ["**التدريب أولًا**: أول 120 شهرًا. منها فقط نحسب μ وσ للتوحيد، ومنها نبني نوافذ التدريب.",
             "**ثم التحقق**: 24 شهرًا تالية لاختيار الإيقاف المبكر.",
             "**ثم الاختبار**: آخر الأشهر، يُلمس مرة واحدة. الترتيب الزمني مقدس: لا خلط.",
             "**الحدود**: أول نافذة اختبار تحتاج 12 شهرًا سابقة — تقع في فترة التحقق. هذا مسموح (قيم ماضية معروفة وقت التنبؤ) لكن الهدف نفسه دائمًا في المستقبل."]
    animation_player("w10_split", [Frame(split_svg(stage=i), caption(c), action=["train", "validation", "test", "boundary"][i]) for i, c in enumerate(scaps)],
                     title_ar="التقسيم الزمني للتضخم (180 شهرًا)", interval_ms=2200)
    st.code(CODE, language="python")
    h2("التجربة", "The experiment")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        series = st.radio("السلسلة", ["inflation", "seasonal"], format_func=lambda s: "التضخم الشهري (180 شهرًا)" if s == "inflation" else "صناعية بموسمية مركّبة (400 شهر)", key="w10_series")
    with c2:
        window = st.select_slider("طول النافذة", options=[6, 12, 24], value=12, key="w10_win")
    with c3:
        units = st.select_slider("units", options=[4, 16, 32], value=16, key="w10_units")
    with c4:
        epochs = st.slider("epochs", 10, 60, 40, 10, key="w10_ep")
    r = keras_seq_run("rnn", int(window), int(units), int(epochs), series=series)
    st.markdown(f"**الأشكال**: X_train {tuple(r['shapes']['X_train'])} · X_test {tuple(r['shapes']['X_test'])} · المعلمات {r['n_params']} · الحقب الفعلية {r['epochs_run']} · الزمن {r['seconds']} ث")
    table(["النموذج", "RMSE على الاختبار (وحدة السلسلة)", "مقارنة بالساذج"],
          [("المتوسط التاريخي", f"{r['rmse_mean']:.3f}", f"{100 * (r['rmse_mean'] / r['rmse_naive'] - 1):+.0f}%"), ("الساذج (القيمة الأخيرة)", f"{r['rmse_naive']:.3f}", "—"), ("SimpleRNN", f"{r['rmse']:.3f}", f"{100 * (r['rmse'] / r['rmse_naive'] - 1):+.0f}%")], ["rtl", "num", "num"])
    fig = go.Figure(); t = np.arange(len(r["true"]))
    fig.add_trace(go.Scatter(x=t, y=r["true"], name="true", line=dict(color="#1E1B4B", width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=r["naive"], name="naive (last value)", line=dict(color="#94A3B8", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=r["pred"], name="SimpleRNN", line=dict(color="#DB2777", width=2)))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="test month", yaxis_title="value", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="w10_pred_fig")
    n_te = len(r["true"])
    fcaps = []
    for k in range(n_te):
        d_r, d_n = abs(r["pred"][k] - r["true"][k]), abs(r["naive"][k] - r["true"][k])
        fcaps.append(f"**شهر الاختبار {k + 1}**: الحقيقة {r['true'][k]:.2f}، RNN {r['pred'][k]:.2f} (خطأ {d_r:.2f})، الساذج {r['naive'][k]:.2f} (خطأ {d_n:.2f}) — " + ("RNN أقرب هذا الشهر." if d_r < d_n else "الساذج أقرب هذا الشهر."))
    animation_player(f"w10_fc_{series}_{window}_{units}_{epochs}", [Frame(forecast_svg(r, k), caption(c), action=f"month {k + 1}") for k, c in enumerate(fcaps)],
                     title_ar="التنبؤ شهرًا شهرًا على فترة الاختبار", interval_ms=650)
    verdict = r["rmse"] < r["rmse_naive"]
    with st.expander("كيف أقرأ النتيجة؟", expanded=True, icon=":material/visibility:"):
        if series == "inflation":
            st.markdown(f"""
- **الصادق أولًا**: على التضخم (108 نوافذ تدريب، سلسلة قريبة من AR(1) مع ضوضاء) {"يتفوق RNN قليلًا" if verdict else "**لا يهزم** RNN الخط الساذج"} — RMSE {r['rmse']:.3f} مقابل {r['rmse_naive']:.3f}. هذا شائع جدًا في السلاسل الاقتصادية القصيرة: الشهر الحالي هو أفضل تنبؤ للشهر القادم تقريبًا، والضوضاء أكبر من النمط.
- الرسم: التنبؤ يتبع الحقيقة **بتأخر** (يشبه الساذج) لأن هذا ما تعلّمه من 108 مثال.
- ما لا يجوز: الادعاء بأن «الشبكة تتنبأ بالتضخم» من هذا الرقم. ما يجوز: «RNN بهذه البيانات لا يضيف على الساذج؛ نحتاج بيانات أطول أو متغيرات إضافية أو نموذجًا أبسط».
- جرّب السلسلة الصناعية لترى ما يحدث عندما يوجد نمط حقيقي يمكن تعلّمه.
""")
        else:
            st.markdown(f"""
- سلسلة بموسميتين متراكبتين (12 و5 أشهر) وضوضاء قليلة و288 نافذة تدريب: RNN **يتفوق بوضوح** — RMSE {r['rmse']:.3f} مقابل {r['rmse_naive']:.3f} للساذج ({100 * (1 - r['rmse'] / r['rmse_naive']):.0f}% أقل).
- الرسم: التنبؤ يسبق التحولات بدل أن يتبعها — النموذج تعلّم النمط من النافذة.
- الفرق عن التضخم ليس في النموذج بل في **البيانات**: نمط قابل للتعلم + بيانات كافية. هذا هو الشرط، لا «الشبكة أقوى».
- جرّب نافذة 6: أقصر من الدورة الموسمية (12) فيضعف الأداء — النافذة يجب أن تغطي النمط.
""")
    intuition("خط الأساس الساذج هو أهم رقم في مسائل السلاسل الزمنية. كثير من الأوراق «تهزم» نموذجًا معقدًا بينما لا تهزم القيمة الأخيرة. أبلغه دائمًا — ومعه المتوسط الموسمي إن وُجدت موسمية.")
    h2("قائمة الفحص للسلاسل الزمنية", "Time-series checklist")
    compare_table(["البند", "الصحيح", "الخطأ الشائع"],
                  [("التقسيم", "زمني: ماضٍ → تحقق → اختبار", "خلط عشوائي للنوافذ (تسريب)"), ("التوحيد", "إحصاءات فترة التدريب فقط", "على السلسلة كاملة"), ("النافذة", "تغطي الدورة/النمط المتوقع", "قصيرة جدًا أو طويلة بلا بيانات"),
                   ("خط الأساس", "الساذج + المتوسط الموسمي", "لا خط أساس"), ("المقياس", "RMSE/MAE بوحدة السلسلة، مقارنة نسبية بالساذج", "MSE موحَّد"), ("الادعاء", "«يتفوق على الساذج بنسبة X% على اختبار زمني»", "«يتنبأ بالمستقبل»")],
                  ["rtl", "rtl", "rtl"])
    research_note("**تمهيد**: LSTM (الأسبوع 11) وGRU (الأسبوع 12) بنفس خط الأنابيب هذا، وسنقارن الثلاثة على السلسلتين. توقّع: على التضخم القصير الفرق ضمن الضوضاء؛ على الموسمية الطويلة قد تتقدم البوابات.")
    warning_note("على هذه الآلة (CPU) التدريب ثوانٍ لأن البيانات صغيرة. مع آلاف السلاسل أو نوافذ طويلة انتقل إلى GPU (الأسبوع 13) — لكن **المنهج لا يتغير**.")
    common_mistake("توحيد السلسلة بمتوسطها الكلي ثم تقسيمها: الاختبار «يعرف» متوسطه ومقياسه. المتوسط والانحراف من فترة التدريب فقط، ويُطبَّقان على ما بعدها.")
    post_test("course.w10", "10", [
        Q("شكل دفعة تسلسلات في Keras:", ["(samples, features)", "(samples, timesteps, features)", "(timesteps, samples)"], 1, ""),
        Q("h_t تعتمد على…", ["x_t فقط", "x_t وh_{t−1}", "y_t"], 1, ""),
        Q("أوزان RNN عبر الخطوات…", ["مختلفة", "نفسها", "عشوائية"], 1, ""),
        Q("تقسيم سلسلة زمنية:", ["عشوائي", "زمني", "بالتساوي"], 1, ""),
        Q("ρ(Wh) = 0.5 وT = 20 → التدرج إلى الخطوة الأولى ≈", ["0.5", "10⁻⁶", "1"], 1, ""),
        Q("RNN لا يهزم «القيمة الأخيرة» على سلسلة قصيرة ضوضائية:", ["خطأ في الكود حتمًا", "نتيجة شائعة وصادقة: لا نمط كافٍ", "يجب زيادة الحقب"], 1, ""),
        Q("خط الأساس الإلزامي في السلاسل الزمنية:", ["MLP", "القيمة الأخيرة (الساذج)", "شجرة"], 1, ""),
        Q("LSTM تعالج التلاشي بـ…", ["أوزان أكبر", "مسار جمع لحالة الخلية وبوابات", "حقب أكثر"], 1, ""),
    ])
    takeaway("خط أنابيب زمني: توحيد بالماضي، نوافذ، تقسيم زمني، RNN، تقييم بوحدة الهدف مقابل الساذج. النتيجة الصادقة أهم من النتيجة الجميلة: RNN يربح حيث يوجد نمط وبيانات.")
    lesson_footer(LESSON, ["الكود والتجربة الحية على سلسلتين.", "قراءة صادقة للنتيجة.", "قائمة الفحص والاختبار البعدي."])
