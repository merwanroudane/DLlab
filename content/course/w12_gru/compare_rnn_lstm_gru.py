import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, research_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table
from labs.fw import keras_seq_run

LESSON = Lesson(
    id="course.w12.compare_rnn_lstm_gru",
    title_ar="بناء GRU وتدريبها، والمقارنة الثلاثية: RNN مقابل LSTM مقابل GRU",
    title_en="Building & Training a GRU, and the Three-way Comparison: RNN vs LSTM vs GRU",
    module="course.w12",
    order=3,
    prerequisites=["course.w12.update_reset_gates", "course.w11.train_eval_diagnose"],
    objectives_ar=["تدريب GRU على نفس خط الأنابيب ومقارنتها بـ RNN وLSTM والساذج: المعلمات، الزمن، RMSE، الحقب.", "قراءة المقارنة بأمانة إحصائية: ضوضاء الاختبار والبذور.", "قاعدة اختيار البنية التكرارية."],
    terms=["sequence"],
    labs=["labs.gru_gates_lab"],
    difficulty="intermediate",
    summary_ar="نفس البيانات والتقسيم والمعايير: RNN (أقل معلمات، أسرع، يتلاشى مع الطول)، LSTM (×4، أثقل)، GRU (×3، بين الاثنين). على سلاسل قصيرة الفروق ضوضاء؛ اختر بالتحقق وبالبساطة.",
)


def render() -> None:
    lesson_header(LESSON)
    why("ثلاث بنى تكرارية، ونفس السؤال: أيها؟ الجواب الوحيد المقبول علميًا هو **نفس البيانات، نفس التقسيم، نفس المعايير، عدة بذور** — وهذا ما تفعله هذه الصفحة على سلسلتين.")
    c1, c2, c3 = st.columns(3)
    with c1:
        series = st.radio("السلسلة", ["inflation", "seasonal"], format_func=lambda s: "التضخم (180 شهرًا)" if s == "inflation" else "موسمية صناعية (400)", key="w12_series")
    with c2:
        window = st.select_slider("النافذة", options=[6, 12, 24], value=12, key="w12_win")
    with c3:
        seeds = st.select_slider("عدد البذور", options=[1, 2, 3], value=1, key="w12_seeds")
    rows = []; curves = {}
    for kind, color in (("rnn", "#7C5CBF"), ("lstm", "#C8473A"), ("gru", "#1F7A78")):
        rmses, secs, params, eps = [], [], 0, []
        for sd in range(int(seeds)):
            r = keras_seq_run(kind, int(window), 16, 25, seed=sd, series=series)
            rmses.append(r["rmse"]); secs.append(r["seconds"]); params = r["n_params"]; eps.append(r["epochs_run"])
            if sd == 0:
                curves[kind] = (r, color)
        naive = r["rmse_naive"]
        rows.append((kind.upper() if kind != "rnn" else "SimpleRNN", str(params), f"{np.mean(secs):.1f}", f"{np.mean(eps):.0f}", f"{np.mean(rmses):.3f}" + (f" ± {np.std(rmses):.3f}" if seeds > 1 else ""), f"{100 * (np.mean(rmses) / naive - 1):+.0f}%"))
    rows.insert(0, ("الساذج", "0", "0", "—", f"{naive:.3f}", "—"))
    table(["النموذج", "المعلمات", "الزمن (ث)", "الحقب", "RMSE اختبار", "مقابل الساذج"], rows, ["rtl", "num", "num", "num", "num", "num"])
    c4, c5 = st.columns(2)
    with c4:
        fig = go.Figure()
        for kind, (r, color) in curves.items():
            e = np.arange(1, len(r["history"]["val_loss"]) + 1)
            fig.add_trace(go.Scatter(x=e, y=r["history"]["val_loss"], name=kind, line=dict(color=color, width=2.5)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="val_loss", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch", key="w12_val")
    with c5:
        fig2 = go.Figure(); r0 = curves["gru"][0]; t = np.arange(len(r0["true"]))
        fig2.add_trace(go.Scatter(x=t, y=r0["true"], name="true", line=dict(color="#2B2A28", width=2.5)))
        for kind, (r, color) in curves.items():
            fig2.add_trace(go.Scatter(x=t, y=r["pred"], name=kind, line=dict(color=color, width=1.5)))
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="test month", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(fig2, width="stretch", key="w12_pred")
    with st.expander("كيف أقرأ؟", expanded=True, icon=":material/visibility:"):
        st.markdown("""
- **المعلمات**: RNN < GRU (×3) < LSTM (×4) — بنفس units. **الزمن**: بنفس الترتيب تقريبًا على CPU.
- **التضخم**: الثلاثة قريبة من الساذج أو أسوأ؛ الفروق بينها أصغر من ضوضاء 36 شهرًا. زد البذور إلى 3 وانظر إلى ±: إن تداخلت الفترات فلا فائز.
- **الموسمية**: الثلاثة تهزم الساذج بوضوح؛ الترتيب بينها قد يتغير مع البذرة — GRU وLSTM غالبًا متقاربتان، وRNN قد يكفي عند نافذة 12.
- **القاعدة**: ابدأ بالأبسط الذي يهزم الساذج؛ انتقل إلى GRU ثم LSTM فقط إذا تحسّن التحقق **بفارق يتجاوز ±** البذور.
""")
    h2("قاعدة الاختيار", "The selection rule")
    compare_table(["الحالة", "ابدأ بـ", "لماذا"],
                  [("سلسلة قصيرة (< بضع مئات نافذة)", "الساذج ← خطي ← SimpleRNN صغير", "البيانات لا تدعم البوابات"), ("اعتماديات طويلة (> 20 خطوة) وبيانات معقولة", "GRU", "أقل معلمات من LSTM بأداء مماثل"), ("تسلسلات طويلة جدًا وبيانات كثيرة", "LSTM (أو Transformers خارج المقرر)", "الذاكرة المنفصلة c تساعد"),
                   ("نص → فئة", "Embedding + GRU/LSTM (bidirectional)", "قياسي"), ("زمن تدريب محدود", "GRU", "أسرع"), ("النتائج متقاربة", "الأبسط", "الاستنساخ والتفسير")],
                  ["rtl", "rtl", "rtl"])
    intuition("في الأدبيات: GRU ≈ LSTM على معظم المهام متوسطة الحجم؛ LSTM تتقدم قليلًا مع التسلسلات الطويلة جدًا؛ RNN البسيط يكفي عندما يهم الماضي القريب فقط. لا توجد بنية «أفضل» — توجد بنية أبسط تكفي.")
    research_note("للمقارنة العادلة في مشروعك: نفس النافذة، نفس units (أو نفس عدد المعلمات تقريبًا)، نفس الإيقاف المبكر، ≥ 3 بذور، اختبار زمني واحد أو rolling origin. أبلغ المتوسط ± الانحراف وزمن التدريب.")
    common_mistake("مقارنة LSTM(32) بـ RNN(32) واستنتاج تفوق LSTM: أربعة أضعاف المعلمات. قارن بنفس المعلمات (RNN(64) تقريبًا) أو أبلغ الفرق صراحةً.")
    quiz("w12.cmp", [
        Q("بنفس units، ترتيب المعلمات:", ["RNN < GRU < LSTM", "LSTM < GRU < RNN", "متساوية"], 0, ""),
        Q("الفروق بين الثلاثة على 36 شهر اختبار ببذرة واحدة…", ["حاسمة", "ضمن الضوضاء غالبًا", "لا تُقاس"], 1, ""),
        Q("قاعدة الاختيار:", ["الأشهر", "الأبسط الذي يهزم الساذج وينجح على التحقق", "LSTM دائمًا"], 1, ""),
        Q("المقارنة العادلة تحتاج…", ["حقبًا أكثر لـ LSTM", "نفس البيانات/التقسيم/المعايير وعدة بذور", "نافذة أطول لـ GRU"], 1, ""),
    ])
    takeaway("RNN/GRU/LSTM بنفس الخط: معلمات ×1/×3/×4. على سلاسل قصيرة الفروق ضوضاء؛ على أنماط طويلة البوابات تربح. اختر الأبسط الكافي بأدلة على التحقق وبذور متعددة.")
    lesson_footer(LESSON, ["مقارنة حية بثلاث بنى وبذور.", "قاعدة الاختيار.", "المقارنة العادلة."])
