import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, debugging_note, intuition, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.fw import keras_seq_run

LESSON = Lesson(
    id="course.w11.train_eval_diagnose",
    title_ar="بناء LSTM وتدريبها وتقييمها وتشخيصها + الاختبار البعدي",
    title_en="Building, Training, Evaluating & Diagnosing an LSTM — & Post-test",
    module="course.w11",
    order=4,
    prerequisites=["course.w11.cell_gates_equations", "course.w10.time_series_application"],
    objectives_ar=["بناء LSTM في Keras على نفس خط أنابيب الأسبوع 10 ومقارنتها بـ SimpleRNN وبالساذج على سلسلتين.", "قراءة summary وHistory والتنبؤات، وتشخيص: فرط تخصيص، تلاشٍ/انفجار (القصّ)، نافذة قصيرة.", "الاختبار البعدي."],
    terms=["sequence", "gradient"],
    labs=["labs.lstm_gates_lab"],
    difficulty="intermediate",
    summary_ar="LSTM(16) + Dense(1): ×4 معلمات SimpleRNN. على سلسلة قصيرة ضوضائية الفرق ضمن الضوضاء ولا أحد يهزم الساذج؛ على سلسلة ذات نمط طويل تتفوق البوابات. التشخيص: الفجوة، منحنى val، clipnorm عند nan، النافذة ≥ الدورة.",
)

CODE = '''# نفس خط أنابيب الأسبوع 10 مع تبديل الطبقة (labs/fw.py: keras_seq_run)
model = keras.Sequential([layers.Input(shape=(window, 1)), layers.LSTM(units), layers.Dense(1)])
#                                                          ^^^^ بدل SimpleRNN: 4 × ((1 + units) × units + units) معلمة
model.compile(optimizer=keras.optimizers.Adam(5e-3, clipnorm=1.0), loss="mse")     # clipnorm: قصّ التدرج ضد الانفجار
model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=40, batch_size=16, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])'''


def render() -> None:
    lesson_header(LESSON)
    why("الأسبوع 10 انتهى بنتيجة صادقة: SimpleRNN لا يهزم الساذج على التضخم ويتفوق على السلسلة الموسمية. هل تغيّر LSTM ذلك؟ الجواب تجريبي — على السلسلتين، بنفس التقسيم والمعايير.")
    st.code(CODE, language="python")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        series = st.radio("السلسلة", ["inflation", "seasonal"], format_func=lambda s: "التضخم (180 شهرًا)" if s == "inflation" else "موسمية صناعية (400)", key="w11_series")
    with c2:
        window = st.select_slider("النافذة", options=[6, 12, 24], value=12, key="w11_win")
    with c3:
        units = st.select_slider("units", options=[8, 16, 32], value=16, key="w11_units")
    with c4:
        clip = st.toggle("clipnorm=1.0", value=False, key="w11_clip")
    r_rnn = keras_seq_run("rnn", int(window), int(units), 40, series=series, clipnorm=1.0 if clip else None)
    r_lstm = keras_seq_run("lstm", int(window), int(units), 40, series=series, clipnorm=1.0 if clip else None)
    with st.expander("model.summary() — LSTM", icon=":material/table_view:"):
        st.code(r_lstm["summary"].strip(), language="text")
    table(["النموذج", "المعلمات", "الحقب", "RMSE اختبار", "مقابل الساذج"],
          [("الساذج (القيمة الأخيرة)", "0", "—", f"{r_lstm['rmse_naive']:.3f}", "—"), ("SimpleRNN", str(r_rnn["n_params"]), str(r_rnn["epochs_run"]), f"{r_rnn['rmse']:.3f}", f"{100 * (r_rnn['rmse'] / r_rnn['rmse_naive'] - 1):+.0f}%"), ("LSTM", str(r_lstm["n_params"]), str(r_lstm["epochs_run"]), f"{r_lstm['rmse']:.3f}", f"{100 * (r_lstm['rmse'] / r_lstm['rmse_naive'] - 1):+.0f}%")],
          ["rtl", "num", "num", "num", "num"])
    c5, c6 = st.columns(2)
    with c5:
        fig = go.Figure()
        for name, r, color in (("SimpleRNN", r_rnn, "#7C5CBF"), ("LSTM", r_lstm, "#C8473A")):
            e = np.arange(1, len(r["history"]["loss"]) + 1)
            fig.add_trace(go.Scatter(x=e, y=r["history"]["loss"], name=f"{name} loss", line=dict(color=color, width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name=f"{name} val_loss", line=dict(color=color, width=2.5)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="mse (standardized)", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch", key="w11_hist")
    with c6:
        fig2 = go.Figure(); t = np.arange(len(r_lstm["true"]))
        fig2.add_trace(go.Scatter(x=t, y=r_lstm["true"], name="true", line=dict(color="#2B2A28", width=2.5)))
        fig2.add_trace(go.Scatter(x=t, y=r_lstm["naive"], name="naive", line=dict(color="#B9B2A6", width=1.5, dash="dot")))
        fig2.add_trace(go.Scatter(x=t, y=r_lstm["pred"], name="LSTM", line=dict(color="#C8473A", width=2)))
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="test month", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(fig2, width="stretch", key="w11_pred")
    with st.expander("كيف أقرأ؟", expanded=True, icon=":material/visibility:"):
        st.markdown(f"""
- **المعلمات**: LSTM {r_lstm['n_params']} مقابل {r_rnn['n_params']} لـ SimpleRNN — أربعة أضعاف تقريبًا (الذاكرة لها ثمن).
- **{'التضخم' if series == 'inflation' else 'الموسمية'}**: {'لا LSTM ولا RNN يهزم الساذج بوضوح؛ الفروق ضمن ضوضاء 36 شهرًا. الاستنتاج: البيانات لا تحوي نمطًا يتجاوز «الشهر الماضي» بما يكفي — لا تبحث عن بنية أعقد.' if series == 'inflation' else f'كلاهما يهزم الساذج بفارق كبير؛ LSTM {"أفضل" if r_lstm["rmse"] < r_rnn["rmse"] else "ليست أفضل"} من SimpleRNN هنا — مع نافذة 12 وبيانات كافية يكفي RNN البسيط أحيانًا؛ فارق LSTM يظهر مع اعتماديات أطول من ذلك.'}
- **المنحنى**: إن كان val_loss يهبط ثم يصعد بينما loss يهبط → فرط تخصيص (110 نوافذ تدريب فقط للتضخم): صغّر units أو زد patience.
- **clipnorm**: فعّله ولاحظ أنه لا يغيّر شيئًا هنا (لا انفجار)؛ يصبح ضروريًا مع نوافذ أطول وتعلم أعلى (`loss: nan`).
""")
    h2("التشخيص", "Diagnostics")
    compare_table(["العرض", "التشخيص", "العلاج"],
                  [("loss: nan بعد حقب قليلة", "انفجار تدرج عبر الزمن", "`clipnorm=1.0`، η أصغر، نافذة أقصر"), ("val_loss لا يقل عن الساذج", "لا نمط قابل للتعلم / بيانات قليلة", "أبلغ ذلك؛ أضف متغيرات؛ نموذج أبسط"), ("val_loss يصعد مبكرًا وloss يهبط", "فرط تخصيص", "units أقل، Dropout (`recurrent_dropout`)، إيقاف مبكر، بيانات أطول"),
                   ("الأداء يسوء مع نافذة أطول", "بيانات لا تكفي للنافذة / ضوضاء", "نافذة ≈ الدورة الطبيعية (12 للشهري)"), ("التنبؤ يتبع الحقيقة بتأخر", "النموذج تعلّم «القيمة الأخيرة»", "طبيعي عند غياب النمط؛ قارن بالساذج"), ("Keras يحذّر من Masking", "قناع مع طبقة لا تدعمه", "ضع Masking قبل LSTM مباشرة")],
                  ["rtl", "rtl", "rtl"])
    intuition("قاعدة القرار للسلاسل الاقتصادية القصيرة: الساذج → خطي/ARIMA → SimpleRNN → LSTM/GRU. انتقل خطوة فقط عندما تهزم السابقة **على اختبار زمني**. القفز إلى LSTM على 180 شهرًا ليس تعلمًا عميقًا بل تفاؤلًا عميقًا.")
    debugging_note("قبل الحكم على LSTM: (1) الأشكال (n, T, k)؛ (2) التوحيد بإحصاءات الماضي؛ (3) التقسيم زمني؛ (4) الساذج محسوب؛ (5) بذور متعددة. الفرق بين بذرتين قد يفوق الفرق بين LSTM وRNN على بيانات صغيرة.")
    with st.container(horizontal=True):
        st.button("معمل بوابات LSTM", icon=":material/science:", on_click=goto, args=("labs.lstm_gates_lab",), key="w11_lab_gates2")
        st.button("الأسبوع 10 — التطبيق", icon=":material/menu_book:", on_click=goto, args=("course.w10.time_series_application",), key="w11_go_w10")
    common_mistake("«LSTM أعطت RMSE أقل بـ 0.01 فهي أفضل». على 36 شهر اختبار هذا ضوضاء. أعد ببذور مختلفة أو بنوافذ اختبار متحركة (rolling origin) قبل أي ادعاء.")
    post_test("course.w11", "11", [
        Q("بوابة النسيان تُخرج قيمًا في…", ["(−1, 1)", "(0, 1)", "أي مدى"], 1, ""),
        Q("c_t تتحدث بـ…", ["ضرب متكرر في Wh", "جمع: f⊙c + i⊙g", "tanh فقط"], 1, ""),
        Q("LSTM(32) مقابل SimpleRNN(32) في المعلمات", ["نفسها", "×4", "×2"], 1, ""),
        Q("تسلسلات بأطوال مختلفة تحتاج…", ["حذف الطويلة", "حشوًا + قناعًا", "لا شيء"], 1, ""),
        Q("f ≈ 1 وi ≈ 0:", ["استبدال الذاكرة", "الاحتفاظ بها", "مسحها"], 1, ""),
        Q("loss: nan في LSTM على نوافذ طويلة:", ["زد η", "قصّ التدرج / η أصغر", "أزل Dense"], 1, ""),
        Q("LSTM لا تهزم الساذج على سلسلة قصيرة:", ["خطأ حتمًا", "نتيجة صادقة محتملة: لا نمط كافٍ", "زد الطبقات"], 1, ""),
        Q("المسار الذي يحفظ التدرج في LSTM:", ["h عبر tanh", "c عبر الجمع وf", "المخرج"], 1, ""),
    ])
    takeaway("LSTM = نفس خط الأنابيب مع طبقة أغنى (×4). تربح حيث توجد اعتماديات طويلة وبيانات كافية؛ على سلسلة قصيرة الفروق ضوضاء. شخّص بالفجوة والمنحنى والساذج والبذور.")
    lesson_footer(LESSON, ["مقارنة حية RNN/LSTM/الساذج على سلسلتين.", "جدول التشخيص.", "قاعدة القرار والاختبار البعدي."])
