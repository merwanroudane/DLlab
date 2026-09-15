import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table
from labs.fw import before_during_after

LESSON = Lesson(
    id="foundations.gallery.before_after",
    title_ar="قبل / أثناء / بعد التدريب: حالات النموذج مربوطة بالرياضيات والمنحنيات",
    title_en="Before / During / After Training: Model States Linked to the Math and the Curves",
    module="foundations.gallery",
    order=5,
    prerequisites=["foundations.gallery.errors", "foundations.training_loop.the_loop", "foundations.generalization.curves"],
    objectives_ar=["رؤية نفس النموذج في ثلاث حالات: تنبؤات عشوائية قبل التدريب، حد قرار يتشكل أثناءه، تنبؤات جيدة بعده.", "ربط كل حالة بموضعها على منحنى الخسارة وبقيمة θ في معادلة التحديث.", "قراءة الاحتمالات الفردية لنفس الملاحظات عبر الحالات."],
    terms=["loss", "epoch", "gradient", "probability"],
    labs=["labs.training_loop_simulator"],
    difficulty="beginner",
    summary_ar="قبل: θ₀ عشوائية، احتمالات ≈ 0.5، خسارة ≈ 0.7. أثناء: بعد k تحديثات الحد يتشكل والخسارة تهبط. بعد: θ_T قرب حد أدنى، احتمالات واثقة، خسارة صغيرة. الصور الثلاث نقاط على منحنى واحد.",
)


def _surface(g, P, X, y, title):
    fig = go.Figure(go.Contour(x=g, y=g, z=P, colorscale=[[0, "#E6F1FB"], [0.5, "#FFFDF9"], [1, "#FBE6E2"]], contours=dict(start=0, end=1, size=0.1), showscale=False, opacity=0.9))
    X = np.array(X); y = np.array(y)
    fig.add_trace(go.Scatter(x=X[y == 1, 0], y=X[y == 1, 1], mode="markers", marker=dict(color="#C8473A", size=5), name="y=1"))
    fig.add_trace(go.Scatter(x=X[y == 0, 0], y=X[y == 0, 1], mode="markers", marker=dict(color="#2F6FB5", size=5), name="y=0"))
    fig.update_layout(height=300, margin=dict(l=5, r=5, t=30, b=5), title=dict(text=title, x=0.5, font=dict(size=13)), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    return fig


def render() -> None:
    lesson_header(LESSON)
    why("اللقطات السابقة أرقام ونصوص. هنا نرى **ما تعنيه**: النموذج نفسه قبل أي تحديث، بعد بضع حقب، وفي النهاية — على البيانات نفسها، مع الاحتمالات التي يعطيها لنفس الملاحظات، وموضع كل حالة على منحنى الخسارة.")
    epochs = st.slider("عدد الحقب الكلي", 12, 60, 30, 6, key="ba_epochs")
    r = before_during_after(int(epochs), 0)
    s = r["snaps"]; g = r["grid"]
    h2("ثلاث حالات لنفس النموذج", "Three states of the same model")
    c1, c2, c3 = st.columns(3)
    for col, key, title in zip((c1, c2, c3), ("before", "during", "after"), ("Before training (epoch 0)", f"During (epoch {r['mid_epoch']})", f"After (epoch {epochs})")):
        with col:
            st.plotly_chart(_surface(g, s[key]["P"], r["X"], r["y"], title), width="stretch", key=f"ba_{key}")
            st.markdown(f"**val_loss = {s[key]['val_loss']:.3f}** · val_acc = {s[key]['val_acc']:.3f}")
    with st.expander("كيف أقرأ الصور الثلاث؟", expanded=True, icon=":material/visibility:"):
        st.markdown(f"""
- **الخلفية** = احتمال الفئة 1 الذي يعطيه النموذج لكل نقطة في المستوى (أحمر ≈ 1، أزرق ≈ 0، أبيض ≈ 0.5). **النقاط** = بيانات التدريب بلونها الحقيقي.
- **قبل**: θ₀ عشوائية (تهيئة Glorot) → الخلفية شبه موحدة قرب 0.5: النموذج لا يميّز شيئًا؛ الخسارة ≈ {s['before']['val_loss']:.2f} (≈ ln 2 = 0.69 لتخمين 0.5)، الدقة ≈ {s['before']['val_acc']:.2f} (قريب من العشوائي).
- **أثناء** (الحقبة {r['mid_epoch']}): بعد {r['mid_epoch']} × ⌈375/32⌉ = {r['mid_epoch'] * 12} تحديثًا للمعلمات، ظهر حد قرار خشن يفصل معظم النقاط؛ الخسارة {s['during']['val_loss']:.2f}.
- **بعد** (الحقبة {epochs}): الحد يتبع شكل الهلالين؛ الخسارة {s['after']['val_loss']:.2f}، الدقة {s['after']['val_acc']:.2f}. المناطق البيضاء (عدم يقين) انحصرت عند الحدود.
""")
    h2("نفس الملاحظات الست عبر الحالات", "The same six observations across states")
    rows = []
    for i in range(6):
        rows.append((str(i + 1), str(s["before"]["true"][i]), f"{s['before']['pred'][i]:.3f}", f"{s['during']['pred'][i]:.3f}", f"{s['after']['pred'][i]:.3f}"))
    table(["#", "y الحقيقية", "p̂ قبل", "p̂ أثناء", "p̂ بعد"], rows, ["num"] * 5)
    intuition("اقرأ صفًا: ملاحظة بـ y = 1 تبدأ عند p̂ ≈ 0.5 (لا رأي)، ثم تتجه نحو 1. وملاحظة بـ y = 0 تتجه نحو 0. الخسارة الثنائية −log p̂(y) لكل صف تنخفض مع كل عمود — ومتوسطها هو ما تراه في السجل.")
    h2("الحالات الثلاث على منحنى الخسارة", "The three states on the loss curve")
    hist = r["history"]; e = np.arange(0, len(hist["loss"]) + 1)
    loss_curve = [s["before"]["val_loss"]] + hist["val_loss"]; tr_curve = [np.nan] + hist["loss"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e, y=tr_curve, name="loss (train)", line=dict(color="#2F6FB5", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=e, y=loss_curve, name="val_loss", line=dict(color="#C8473A", width=3)))
    for ep, key, lbl in ((0, "before", "before"), (r["mid_epoch"], "during", "during"), (epochs, "after", "after")):
        fig.add_trace(go.Scatter(x=[ep], y=[s[key]["val_loss"]], mode="markers+text", text=[lbl], textposition="top center", marker=dict(size=13, color="#1F7A78"), showlegend=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="loss", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="ba_curve")
    equation(r"\theta_{t+1} = \theta_t - \eta\,\nabla_\theta L(\theta_t), \qquad \theta_0 \sim \text{init}, \quad \theta_T \approx \arg\min L",
             [(r"\theta_0", "الصورة الأولى: معلمات التهيئة العشوائية — الخلفية البيضاء."), (r"\theta_k", f"الصورة الثانية بعد k = {r['mid_epoch'] * 12} خطوة — الحد الخشن."), (r"\theta_T", "الصورة الثالثة: قرب حد أدنى — الحد الناعم.")],
             meaning_ar="كل صورة هي «قيمة θ» في لحظة؛ منحنى الخسارة هو L(θ_t) عبر الزمن؛ التحديث هو ما يحرّكنا من صورة إلى التي بعدها.",
             example_ar=f"من {s['before']['val_loss']:.2f} إلى {s['after']['val_loss']:.2f} خلال {epochs * 12} تحديثًا بـ Adam (η = 0.005).", dl_link_ar="`fit` = الانتقال من الصورة الأولى إلى الثالثة؛ EarlyStopping يختار عند أي حقبة تتوقف.", title_ar="الصور الثلاث كنقاط على مسار θ")
    common_mistake("«النموذج قبل التدريب يتنبأ بـ 0.5 بالضبط». لا — يتنبأ بقيم قريبة من 0.5 وبأنماط عشوائية صغيرة تحددها التهيئة؛ بذرة أخرى تعطي صورة أولى مختلفة (وأحيانًا مسار تدريب مختلف). لهذا نثبّت البذرة ونبلّغ عنها.")
    quiz("gallery.ba", [
        Q("خسارة ≈ 0.69 قبل التدريب في تصنيف ثنائي متوازن تعني…", ["خطأ في الكود", "تخمين ≈ 0.5: −ln 0.5", "فرط تخصيص"], 1, "ln 2."),
        Q("الصورة «أثناء» تقابل…", ["θ₀", "θ_k بعد k تحديثًا", "أفضل θ"], 1, "منتصف المسار."),
        Q("المناطق البيضاء في صورة «بعد»…", ["خطأ", "عدم يقين قرب الحد (p̂ ≈ 0.5)", "بيانات مفقودة"], 1, "الحد."),
        Q("ما الذي يحرّك النموذج من صورة إلى التالية؟", ["البيانات وحدها", "تحديثات θ بالتدرج", "الرسم"], 1, "قاعدة التحديث."),
    ])
    takeaway("قبل = θ₀ عشوائية وخسارة ln 2؛ أثناء = حد يتشكل؛ بعد = θ_T قرب حد أدنى. الصور نقاط على منحنى الخسارة، والاحتمالات الفردية تتحرك نحو الحقيقة صفًا صفًا.")
    lesson_footer(LESSON, ["ثلاث حالات حقيقية لنموذج واحد.", "نفس الملاحظات عبر الحالات.", "الربط بمعادلة التحديث والمنحنى."])
