import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, practical_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import pipeline, table
from labs.datasets import house_prices
from labs.fw import house_project

LESSON = Lesson(
    id="course.w07.build_pipeline",
    title_ar="بناء خط الأنابيب: تعريف المسألة، إعداد البيانات، خط الأساس، البنية، التدريب",
    title_en="Building the Pipeline: Problem, Data Prep, Baseline, Architecture, Training",
    module="course.w07",
    order=2,
    prerequisites=["course.w07.overview", "foundations.prep.scaling", "foundations.prep.encoding", "course.w04.depth_width"],
    objectives_ar=["تعريف مسألة انحدار على أسعار العقارات بسؤال وهدف ومقياس وخط أساس.", "إعداد البيانات بلا تسريب: تقسيم، توحيد العددي بإحصاءات التدريب، ترميز الفئوي، تحجيم الهدف.", "خطا أساس (المتوسط، الخطي) ثم MLP بإيقاف مبكر، وقراءة منحنى التدريب."],
    terms=["standardization", "categorical", "target"],
    labs=["labs.scaling_lab", "labs.data_leakage_lab"],
    difficulty="intermediate",
    summary_ar="السؤال → الهدف (السعر بآلاف الدنانير) → المقياس (RMSE/MAE بوحدة الهدف) → تقسيم 60/20/20 → توحيد + one-hot → خط أساس المتوسط والخطي → MLP (7→32→16→1) بإيقاف مبكر → منحنى التدريب.",
)

CODE = '''# الكود الكامل الذي تشغّله هذه الصفحة (labs/fw.py: house_project)
df = house_prices(n=300, seed=11)                                  # 300 عقار: area_m2, rooms, age_years, district, price
num = df[["area_m2", "rooms", "age_years"]].to_numpy("float32")
cat = pd.get_dummies(df["district"]).to_numpy("float32")          # 4 مناطق -> 4 أعمدة one-hot
y = df["price"].to_numpy("float32") / 1000                         # الهدف بآلاف الدنانير
idx = rng.permutation(300); tr, va, te = idx[:180], idx[180:240], idx[240:]   # 60 / 20 / 20
mu, sd = num[tr].mean(0), num[tr].std(0)                           # إحصاءات التدريب فقط
X = np.concatenate([(num - mu) / sd, cat], 1)                      # (300, 7)
ymu, ysd = y[tr].mean(), y[tr].std(); yz = (y - ymu) / ysd         # توحيد الهدف للانحدار (يُعكس عند التقييم)

base_mean = y[tr].mean()                                           # خط أساس 1: المتوسط
lin = LinearRegression().fit(X[tr], y[tr])                         # خط أساس 2: انحدار خطي
model = keras.Sequential([layers.Input(shape=(7,)), layers.Dense(32, activation="relu"), layers.Dense(16, activation="relu"), layers.Dense(1)])   # بلا تنشيط للمخرج
model.compile(optimizer=keras.optimizers.Adam(3e-3), loss="mse", metrics=["mae"])
es = callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
history = model.fit(X[tr], yz[tr], validation_data=(X[va], yz[va]), epochs=300, batch_size=16, callbacks=[es], verbose=0)
pred_te = model.predict(X[te]).ravel() * ysd + ymu                # عكس التوحيد -> آلاف الدنانير'''


def render() -> None:
    lesson_header(LESSON)
    pipeline(["Problem", "Data prep", "Baselines", "Architecture", "Training", "→ next lesson: evaluation · diagnostics · interpretation · errors"], active=0)
    h2("1) تعريف المسألة", "1) Problem definition")
    compare_table(["البند", "القرار", "التبرير"],
                  [("السؤال", "ما السعر المتوقع لعقار من مساحته وغرفه وعمره ومنطقته؟", "قرار تسعير/تقييم؛ الجواب رقم"), ("الملاحظة", "عقار واحد", "صف في الجدول"), ("الهدف", "`price` بآلاف الدنانير (عددي متصل)", "→ انحدار"), ("المقياس", "RMSE وMAE بآلاف الدنانير", "قابل للتفسير للجهة المستخدمة؛ MAE أقل تأثرًا بالمتطرف"),
                   ("خط الأساس", "المتوسط؛ ثم انحدار خطي", "لا قيمة لشبكة لا تتفوق على الخطي هنا"), ("حدود الادعاء", "تنبؤ لا سببية", "الأسبوع 01")],
                  ["rtl", "rtl", "rtl"])
    h2("2) البيانات", "2) Data")
    df = house_prices(n=300, seed=11)
    st.dataframe(df.head(8), width="stretch", hide_index=True)
    r = house_project()
    st.markdown(f"**الأحجام**: تدريب {r['n'][0]} / تحقق {r['n'][1]} / اختبار {r['n'][2]} — **الخصائص بعد الإعداد**: {len(r['names'])} ({', '.join(r['names'])}).")
    practical_note("إعداد بلا تسريب: التقسيم أولًا، ثم `mu, sd` من التدريب فقط، ثم one-hot على الكل (الفئات معروفة)، ثم **توحيد الهدف** للانحدار (يُسرّع التدريب ويثبّت η) مع عكسه قبل أي رقم تُبلّغه.")
    st.code(CODE, language="python")
    h2("3) خطوط الأساس", "3) Baselines")
    t = r["test"]
    table(["النموذج", "RMSE (ألف دينار)", "MAE (ألف دينار)"], [("المتوسط", f"{t['mean_rmse']:.1f}", f"{t['mean_mae']:.1f}"), ("انحدار خطي", f"{t['lin_rmse']:.1f}", f"{t['lin_mae']:.1f}")], ["rtl", "num", "num"])
    st.markdown("**معاملات الخطي** (ألف دينار لكل وحدة موحَّدة / لكل منطقة): " + "، ".join(f"`{k}`: {v:+.1f}" for k, v in r["lin_coef"].items()))
    intuition(f"المتوسط يخطئ بـ ≈ {t['mean_rmse']:.0f} ألفًا؛ الخطي يقلّصها إلى ≈ {t['lin_rmse']:.0f}. هذا هو الرقم الذي يجب على الشبكة التفوق عليه — وإلا فالخطي أفضل: أبسط وأشفّ.")
    h2("4) البنية و5) التدريب", "4) Architecture & 5) Training")
    compare_table(["السؤال", "الجواب"],
                  [("لماذا Keras؟", "جدولي قياسي؛ fit + إيقاف مبكر"), ("المدخل", "7 خصائص (3 موحَّدة + 4 one-hot)"), ("الهدف", "السعر الموحَّد (يُعكس لاحقًا)"), ("البنية", f"7 → 32 (ReLU) → 16 (ReLU) → 1 (بلا تنشيط) — {r['n_params']} معلمة لـ 180 صفًا: صغيرة نسبيًا لكنها تحتاج إيقافًا مبكرًا"),
                   ("الخسارة", "MSE على الهدف الموحَّد (الأسس 13)"), ("المقياس أثناء التدريب", "MAE موحَّد للمراقبة؛ الأرقام النهائية بوحدة الهدف"), ("المحسّن وη", "Adam 3e-3"), ("الدفعة والحقب", "16 (12 تحديثًا/حقبة)؛ 300 حد أعلى، patience=25")],
                  ["rtl", "rtl"])
    h_ = r["history"]; e = np.arange(1, len(h_["loss"]) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e, y=h_["loss"], name="loss (standardized MSE)", line=dict(color="#2F6FB5", width=2)))
    fig.add_trace(go.Scatter(x=e, y=h_["val_loss"], name="val_loss", line=dict(color="#C8473A", width=3)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="w07_train_fig")
    st.markdown(f"**الحقب الفعلية**: {r['epochs_run']} (توقف مبكر) · أفضل val_loss عند الحقبة {int(np.argmin(h_['val_loss'])) + 1}.")
    with st.container(horizontal=True):
        st.button("معمل التحجيم", icon=":material/science:", on_click=goto, args=("labs.scaling_lab",), key="w07_lab_scale")
        st.button("معمل التسريب", icon=":material/science:", on_click=goto, args=("labs.data_leakage_lab",), key="w07_lab_leak")
        st.button("التالي: التقييم والتشخيص والتفسير", icon=":material/arrow_back:", type="primary", on_click=goto, args=("course.w07.diagnose_interpret",), key="w07_go_next")
    common_mistake("توحيد الهدف ثم الإبلاغ عن «MSE = 0.21». الرقم بلا معنى للجهة المستخدمة. اعكس التوحيد وأبلغ RMSE/MAE بآلاف الدنانير (الدرس التالي).")
    quiz("w07.build", [
        Q("لماذا نوحّد الهدف في الانحدار؟", ["لتحسين الدقة النهائية", "لتسريع/تثبيت التدريب؛ يُعكس قبل الإبلاغ", "إلزامي في Keras"], 1, ""),
        Q("مخرج شبكة الانحدار:", ["sigmoid", "بلا تنشيط", "softmax"], 1, ""),
        Q("الشبكة يجب أن تتفوق على…", ["المتوسط فقط", "الخطي (وإلا فهو أفضل)", "لا شيء"], 1, ""),
        Q("`mu, sd` تُحسب على…", ["كل البيانات", "التدريب", "الاختبار"], 1, ""),
    ])
    takeaway("سؤال → هدف → مقياس بوحدته → تقسيم → إعداد بلا تسريب → خطا أساس → MLP صغير بإيقاف مبكر. الأرقام النهائية في الدرس التالي بوحدة الهدف.")
    lesson_footer(LESSON, ["تعريف المسألة كجدول قرارات.", "الإعداد والكود الكامل.", "خطوط الأساس والتدريب."])
