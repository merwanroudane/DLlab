import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, debugging_note, interpretation_note, intuition, math_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="course.w05.loss_surface_convergence",
    title_ar="دالة الخسارة وسطحها، التقارب، ومشاكل معدل التعلم وجداوله + الاختبار البعدي",
    title_en="The Loss Function & Its Surface, Convergence, Learning-Rate Problems & Schedules — & Post-test",
    module="course.w05",
    order=4,
    prerequisites=["course.w05.momentum_rmsprop_adam", "foundations.optim.loss_surface", "foundations.optim.schedules_convergence"],
    objectives_ar=[
        "رؤية أثر اختيار الخسارة على سطح التحسين: MSE مقابل cross-entropy لنفس التصنيف، بمنحنى وتدرج.",
        "قراءة أشكال منحنى الخسارة: تقارب، هضبة، تذبذب، انفجار — وربط كل شكل بعلاج.",
        "مقارنة جداول معدل التعلم بصريًا (ثابت، خطوي، أسي، جيب تمام، إحماء) وفهم ReduceLROnPlateau.",
        "الاختبار البعدي للأسبوع.",
    ],
    terms=["loss", "learning_rate", "cross_entropy", "mse", "learning_rate_schedule", "callback"],
    labs=["labs.learning_rate_lab", "labs.loss_lab"],
    difficulty="intermediate",
    summary_ar="الخسارة تحدد السطح؛ cross-entropy مع sigmoid/softmax تعطي تدرجات صحية حيث MSE تتشبع. المنحنى يشخّص η: هضبة (صغير/تشبع)، تذبذب (كبير)، nan (كبير جدًا). الجداول تخفض η مع الوقت.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers, callbacks
from labs.tinynet import make_moons
X, y = make_moons(600, noise=0.25, seed=0); Xtr, ytr, Xva, yva = X[:450], y[:450], X[450:], y[450:]

def run(loss, lr=0.1, epochs=25, sched=None):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([layers.Input(shape=(2,)), layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9), loss=loss, metrics=["accuracy"])
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0, callbacks=sched or [])
    return h.history

# 1) الخسارة تغيّر السطح: MSE مقابل BCE لنفس التصنيف
for loss in ("mse", "binary_crossentropy"):
    h = run(loss)
    print(f"{loss:<22} val_acc by epoch 5/15/25: {h['val_accuracy'][4]:.3f} / {h['val_accuracy'][14]:.3f} / {h['val_accuracy'][-1]:.3f}")

# 2) جدول معدل التعلم: يبدأ كبيرًا ثم ينخفض عند الثبات
lr_log = []
class LR(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None): lr_log.append(round(float(self.model.optimizer.learning_rate), 4))
h = run("binary_crossentropy", lr=0.5, epochs=25, sched=[callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-3), LR()])
print("\\nReduceLROnPlateau: lr per epoch:", lr_log)
print("val_loss first/last:", round(h["val_loss"][0], 3), "/", round(h["val_loss"][-1], 3), "| val_acc last:", round(h["val_accuracy"][-1], 3))
h2 = run("binary_crossentropy", lr=0.5, epochs=25)
print("same lr=0.5 without schedule: val_loss last:", round(h2["val_loss"][-1], 3), "| val_acc last:", round(h2["val_accuracy"][-1], 3))'''


def _shape_fig(kind: str):
    e = np.arange(1, 41); rng = np.random.default_rng(1)
    if kind == "converge":
        y = 0.7 * np.exp(-e / 8) + 0.25 + rng.normal(0, 0.01, 40)
    elif kind == "plateau":
        y = np.where(e < 25, 0.69 + rng.normal(0, 0.003, 40), 0.69 - 0.02 * (e - 25) + rng.normal(0, 0.005, 40))
    elif kind == "oscillate":
        y = 0.5 + 0.2 * np.abs(np.sin(e * 1.3)) + rng.normal(0, 0.03, 40)
    else:
        y = np.where(e < 6, 0.7 - 0.05 * e, np.minimum(0.7 * 1.6 ** (e - 6), 60))
    colors = {"converge": "#059669", "plateau": "#D97706", "oscillate": "#7C3AED", "explode": "#DC2626"}
    f = go.Figure(go.Scatter(x=e, y=y, line=dict(color=colors[kind], width=2.5)))
    f.update_layout(height=180, margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f


def _schedules(epochs: int = 60, lr0: float = 0.1) -> dict[str, np.ndarray]:
    e = np.arange(epochs)
    warm = 5
    cosine = 0.5 * lr0 * (1 + np.cos(np.pi * e / (epochs - 1)))
    warm_cos = np.where(e < warm, lr0 * (e + 1) / warm, 0.5 * lr0 * (1 + np.cos(np.pi * (e - warm) / (epochs - 1 - warm))))
    return {"constant": np.full(epochs, lr0), "step (÷10 every 20)": lr0 * 0.1 ** (e // 20), "exponential (×0.95)": lr0 * 0.95 ** e,
            "cosine decay": cosine, "warmup + cosine": warm_cos}


def render() -> None:
    lesson_header(LESSON)
    why("المحسّن يمشي على **سطح** تحدده الخسارة والبيانات والبنية. الخسارة الخاطئة تصنع سطحًا بهضاب مسطحة (تشبع) فيتجمد المحسّن مهما كان ذكيًا. ومنحنى الخسارة عبر الحقب هو «تخطيط القلب» الذي يقرأ منه الباحث كل ذلك.")

    # ------------------------------------------------------------------ MSE vs BCE surface
    h2("نفس النموذج، خسارتان، سطحان", "Same model, two losses, two surfaces")
    st.markdown("عصبون sigmoid واحد `p = σ(w·x)` وملاحظة واحدة `x = 1` بفئة حقيقية `y = 1`. نرسم الخسارة وتدرجها بدلالة الوزن w:")
    w = np.linspace(-8, 6, 300)
    p = 1 / (1 + np.exp(-w))
    mse, bce = (1 - p) ** 2, -np.log(p)
    g_mse, g_bce = -2 * (1 - p) * p * (1 - p), -(1 - p)
    c1, c2 = st.columns(2)
    with c1:
        f1 = go.Figure()
        f1.add_scatter(x=w, y=bce, name="BCE = −log p", line=dict(color="#DB2777", width=3))
        f1.add_scatter(x=w, y=mse, name="MSE = (1 − p)²", line=dict(color="#2563EB", width=3, dash="dash"))
        f1.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="weight w", yaxis_title="loss", yaxis_range=[0, 5], legend=dict(orientation="h", x=0, y=1.2))
        st.plotly_chart(f1, width="stretch", key="w05_surf_loss")
    with c2:
        f2 = go.Figure()
        f2.add_scatter(x=w, y=np.abs(g_bce), name="|∂BCE/∂w|", line=dict(color="#DB2777", width=3))
        f2.add_scatter(x=w, y=np.abs(g_mse), name="|∂MSE/∂w|", line=dict(color="#2563EB", width=3, dash="dash"))
        f2.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="weight w", yaxis_title="gradient size", legend=dict(orientation="h", x=0, y=1.2))
        st.plotly_chart(f2, width="stretch", key="w05_surf_grad")
    i8 = 0
    interpretation_note(f"عند w = −8 (النموذج مخطئ بثقة: p = {p[i8]:.4f}) تدرج BCE = {abs(g_bce[i8]):.3f} (إشارة تصحيح قوية) بينما تدرج MSE = {abs(g_mse[i8]):.5f} — أصغر بنحو {abs(g_bce[i8]) / abs(g_mse[i8]):.0f} مرة. "
                        "سطح MSE مع sigmoid **مسطح** في منطقة الخطأ الواثق: هضبة يتجمد عليها أي محسّن.")
    equation(r"\frac{\partial L_{\text{BCE}}}{\partial w} = (p - y)\,x \qquad\text{vs}\qquad \frac{\partial L_{\text{MSE}}}{\partial w} = 2(p-y)\,\underbrace{p(1-p)}_{\to 0 \text{ when saturated}}\,x",
             [(r"p(1-p)", "مشتقة sigmoid: تقترب من الصفر عندما p قريب من 0 أو 1 (التشبع)."), (r"p - y", "الخطأ نفسه.")],
             meaning_ar="BCE «يلغي» مشتقة sigmoid فيبقى التدرج بحجم الخطأ؛ MSE يضربه فيها فيختفي حيث نحتاجه أكثر.",
             example_ar="p = 0.001، y = 1: تدرج BCE ≈ −1، تدرج MSE ≈ −0.002.",
             dl_link_ar="لهذا: sigmoid ⇄ binary_crossentropy، softmax ⇄ categorical_crossentropy، بلا تنشيط ⇄ mse.", title_ar="لماذا تتشبع MSE")
    code_lab(CodeLab(
        key="w05_surface", title_ar="MSE مقابل cross-entropy لنفس التصنيف، ثم جدول معدل التعلم", code=CODE, level="C",
        before=Before(goal_ar="إظهار أن اختيار الخسارة يغيّر سرعة التعلم على نفس النموذج والبيانات، ثم أثر ReduceLROnPlateau على تدريب بدأ بمعدل كبير.", stage_ar="الأسبوع 05: السطح والتقارب.",
                      inputs_ar="moons (450/150)، MLP صغير، SGD بزخم.", expected_ar="BCE تصل إلى دقة أعلى أبكر من MSE؛ الجدول يخفض η عند الثبات ويعطي val_loss نهائية أفضل من η الثابت الكبير."),
        explain=[("7-12", "دالة تشغيل موحّدة: نفس البذرة، SGD+زخم، خسارة كمعامل."), ("15-17", "**MSE للتصنيف** يعمل لكن ببطء: تدرجه يتشبع مع sigmoid (الأسس 13). BCE يعطي تدرجًا (p−y) مباشرًا."), ("20-25", "استدعاء يسجّل η؛ `ReduceLROnPlateau` يضرب η في 0.3 بعد 3 حقب بلا تحسن — راقب القائمة."), ("26-27", "المقارنة العادلة: نفس η البدائي بلا جدول.")],
        run=run_printed(CODE),
        after_ar="- اختلاف الخسارة = اختلاف السطح: نفس البيانات والنموذج والمحسّن، ونتيجة مختلفة.\n- قائمة η تُظهر الهبوطات: كل هبوط = «الثبات» الذي اكتشفه الاستدعاء.\n- η كبير ثم مخفَّض يجمع بين السرعة في البداية والدقة في النهاية — هذا جوهر الجداول (الأسس 15).",
    ))

    # ------------------------------------------------------------------ curve shapes
    h2("قراءة منحنى الخسارة", "Reading the loss curve")
    st.caption("رسوم توضيحية تخطيطية للأشكال الأربعة (ليست تشغيلًا حقيقيًا) — رأيت أمثلة حقيقية لكل منها في دروس الأسبوع.")
    cols = st.columns(4)
    for col, kind, title, diag, fix in zip(cols, ("converge", "plateau", "oscillate", "explode"), ("تقارب سليم", "هضبة", "تذبذب", "انفجار"),
                                             ("انخفاض سريع ثم بطيء ثم ثبات عند قيمة منخفضة", "ثبات عند خسارة عالية (≈ ln 2 للثنائي): η صغير جدًا، أو تشبع (MSE/sigmoid، تهيئة سيئة)، أو مدخل غير محجّم", "قفزات عنيفة حول قيمة: η كبير، دفعة صغيرة جدًا", "نمو أسي ثم nan: η كبير جدًا، انفجار تدرج، مدخل غير محجّم"),
                                             ("لا شيء؛ راقب val_loss للتعميم", "زد η ×10، غيّر الخسارة، حجّم المدخل، تحقق من التهيئة", "قلّل η ×3، كبّر الدفعة، أضف زخمًا/جدولًا", "قلّل η ×10، قصّ التدرج، TerminateOnNaN، حجّم")):
        with col:
            st.markdown(f"**{title}**")
            st.plotly_chart(_shape_fig(kind), width="stretch", key=f"w05_shape_{kind}")
            st.caption(f"**التشخيص**: {diag}")
            st.caption(f"**العلاج**: {fix}")
    intuition("هضبة عند 0.69 في تصنيف ثنائي = النموذج يخمّن 0.5 لكل شيء: −ln 0.5 = 0.693. هذا الرقم بعينه يخبرك أن لا تعلم يحدث — ابحث في η والتحجيم والخسارة قبل البنية.")
    compare_table(["عدد الفئات K", "خسارة «التخمين المنتظم»", "المعنى"],
                  [("2", "ln 2 ≈ 0.693", "p = 0.5 لكل ملاحظة"), ("3", "ln 3 ≈ 1.099", "p = 1/3 لكل فئة"), ("10", "ln 10 ≈ 2.303", "p = 0.1 لكل فئة")],
                  ["ltr", "ltr", "rtl"])

    # ------------------------------------------------------------------ schedules
    h2("جداول معدل التعلم", "Learning-rate schedules")
    sch = _schedules()
    fs = go.Figure()
    for (name, vals), col in zip(sch.items(), ("#94A3B8", "#2563EB", "#059669", "#7C3AED", "#DB2777")):
        fs.add_scatter(y=vals, mode="lines", name=name, line=dict(color=col, width=3))
    fs.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="epoch", yaxis_title="learning rate η", legend=dict(orientation="h", x=0, y=1.25))
    st.plotly_chart(fs, width="stretch", key="w05_sched_fig")
    math_note("الفكرة المشتركة: η كبير في البداية للتقدم السريع عبر السطح، ثم صغير في النهاية ليستقر المحسّن في القاع بدل الارتجاف حوله (تذكّر ضوضاء SGD). "
              "الإحماء `warmup` يبدأ صغيرًا لبضع حقب لأن التدرجات الأولى من أوزان عشوائية قد تكون كبيرة وغير موثوقة — شائع مع Adam والنماذج الكبيرة.")
    compare_table(["الجدول", "الكود", "متى"],
                  [("خفض عند الثبات", "`callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)`", "الافتراضي العملي: لا تحتاج معرفة عدد الحقب مسبقًا"), ("تناقص أسي", "`callbacks.LearningRateScheduler(lambda e, lr: lr * 0.95)`", "تدريب بعدد حقب معروف"),
                   ("خطوي", "`LearningRateScheduler(lambda e, lr: 0.1 * 0.1 ** (e // 30))`", "الرؤية الحاسوبية التقليدية"), ("جيب التمام / إحماء", "`keras.optimizers.schedules.CosineDecay(...)` داخل المحسّن", "نماذج كبيرة (تعميق)")],
                  ["rtl", "code", "rtl"])
    with st.container(horizontal=True):
        st.button("معمل معدل التعلم", icon=":material/science:", on_click=goto, args=("labs.learning_rate_lab",), key="w05_lab_lr2")
        st.button("معمل الخسارة", icon=":material/science:", on_click=goto, args=("labs.loss_lab",), key="w05_lab_loss")
        st.button("الأسس 15 — الجداول والتقارب", icon=":material/menu_book:", on_click=goto, args=("foundations.optim.schedules_convergence",), key="w05_go_sched")
    debugging_note("ترتيب التشخيص عند «لا يتعلم»: (1) الخسارة تطابق التنشيط والهدف؟ (2) المدخل محجّم؟ (3) η ×10 و÷10؟ (4) هل تتعلم على 10 ملاحظات فقط (يجب أن تحفظها)؟ (5) ثم البنية. معظم الحالات تنتهي عند 1–3.")
    common_mistake("تغيير البنية عند أول هضبة. الهضبة عند 0.69 مشكلة تحسين/خسارة/تحجيم في 90% من الحالات، لا مشكلة بنية.")
    post_test("course.w05", "05", [
        Q("خطوة الانحدار التدرجي:", ["θ ← θ + η∇L", "θ ← θ − η∇L", "θ ← ∇L"], 1, ""),
        Q("η كبير جدًا يسبب…", ["تقاربًا بطيئًا", "تذبذبًا أو انفجارًا", "لا شيء"], 1, ""),
        Q("الزخم يساعد على…", ["تسريع الاتجاهات المتسقة وتخفيف التذبذب", "زيادة الضوضاء", "تقليل المعلمات"], 0, ""),
        Q("Adam يجمع بين…", ["الزخم وتكييف η لكل معلمة", "L2 وDropout", "batch وepoch"], 0, ""),
        Q("هضبة عند 0.69 في تصنيف ثنائي تعني…", ["فرط تخصيص", "النموذج يخمّن 0.5: لا تعلم", "تقارب"], 1, ""),
        Q("MSE مع sigmoid للتصنيف…", ["أفضل من BCE", "تتشبع تدرجاتها: أبطأ", "لا تعمل"], 1, "p(1−p) → 0."),
        Q("`ReduceLROnPlateau(factor=0.3, patience=3)`:", ["يوقف التدريب", "يضرب η في 0.3 بعد 3 حقب بلا تحسن", "يزيد η"], 1, ""),
        Q("أول ما تفحصه عند «لا يتعلم»:", ["عدد الطبقات", "الخسارة/التحجيم/η", "batch_size"], 1, ""),
        Q("تصنيف 10 فئات والخسارة عالقة عند 2.30:", ["تقارب ممتاز", "تخمين منتظم: لا تعلم", "فرط تخصيص"], 1, "ln 10."),
        Q("لماذا نخفض η في آخر التدريب؟", ["لتسريع البداية", "ليستقر في القاع بدل الارتجاف حوله", "لتقليل الذاكرة"], 1, ""),
    ])
    takeaway("الخسارة تصنع السطح: BCE تبقي التدرج بحجم الخطأ، MSE مع sigmoid تتشبع. المنحنى يشخّص: تقارب/هضبة/تذبذب/انفجار ولكل علاج. الجداول: كبير ثم صغير. الترتيب: خسارة، تحجيم، η، ثم البنية.")
    lesson_footer(LESSON, ["سطحا MSE وBCE وتدرجاهما.", "MSE مقابل BCE وجدول η بالكود.", "أربعة أشكال للمنحنى بعلاجها وجدول خسائر التخمين.", "معرض الجداول والاختبار البعدي."])
