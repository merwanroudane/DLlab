import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, debugging_note, interpretation_note, intuition, takeaway, warning_note, why
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from content.course.w09_cnn_apps._viz import occl_svg, occlusion
from core.routing import go as goto
from labs.fw import cnn_shapes_run

LESSON = Lesson(
    id="course.w09.shape_debugging",
    title_ar="خرائط الخصائص والنوى المتعلَّمة، تحليل الأخطاء، وتصحيح أخطاء الأشكال + الاختبار البعدي",
    title_en="Feature Maps & Learned Kernels, Error Analysis, and Shape Debugging — & Post-test",
    module="course.w09",
    order=4,
    prerequisites=["course.w09.overfit_augment", "course.w08.convolution", "foundations.frameworks.keras.errors"],
    objectives_ar=["استخراج خرائط خصائص الطبقة الأولى لصورة ورؤية النوى المتعلَّمة.", "تحليل الصور المصنَّفة خطأً، وحساسية الإخفاء `occlusion`: أين ينظر النموذج؟ (تحريك)", "أخطاء الأشكال الخمسة في CNN برسائلها وحلولها، والاختبار البعدي."],
    terms=["shape", "channel_dimension", "feature_map", "kernel", "cnn"],
    labs=["labs.cnn_shape_calculator"],
    difficulty="intermediate",
    summary_ar="keras.Model(model.inputs, layer.output) يعطي خرائط الخصائص؛ conv.kernel النوى (k, k, C_in, filters). أخطاء الأشكال: ndim=3 (قناة/دفعة مفقودة)، قنوات لا تطابق، Flatten ضخم، صورة أصغر من الشبكة، ترتيب القنوات.",
)

CODE = '''# استخراج خرائط الخصائص والنوى (ما تشغّله هذه الصفحة)
fm_model = keras.Model(model.inputs, model.get_layer("conv1").output)   # نموذج فرعي: المدخل -> مخرج الطبقة الأولى
fmaps = fm_model.predict(img[None, ...])[0]                              # (16, 16, 8): 8 خرائط
kernels = model.get_layer("conv1").kernel.numpy()                         # (3, 3, 1, 8): 8 نوى 3×3 على قناة واحدة'''


def _heat(z, title, key, cs="Greys", h=130):
    fig = go.Figure(go.Heatmap(z=np.array(z)[::-1], colorscale=cs, showscale=False, zmid=0 if cs != "Greys" else None))
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=16, b=0), title=dict(text=title, font=dict(size=10), x=0.5), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    st.plotly_chart(fig, width="stretch", key=key)


def render() -> None:
    lesson_header(LESSON)
    why("CNN ليست صندوقًا أسود كاملًا: الطبقة الأولى تتعلم نوى يمكن رؤيتها، وخرائط خصائصها تُظهر ما «انتبهت» إليه في صورة. وعند الفشل، أخطاء الأشكال في CNN لها خمسة أنماط تتكرر — كلها تُحل بحاسبة الأشكال وسطر طباعة واحد.")
    r = cnn_shapes_run(epochs=8)
    h2("1) النوى المتعلَّمة وخرائط الخصائص", "1) Learned kernels & feature maps")
    st.code(CODE, language="python")
    st.markdown(f"**الصورة** (فئة: {r['classes'][r['image_class']]}) ثم **8 نوى** من الطبقة الأولى (أحمر موجب، أزرق سالب) و**8 خرائط خصائص** لها:")
    c0, *cks = st.columns(9)
    with c0:
        _heat(r["image"], "input", "w09_fm_in")
    for i, (c, k) in enumerate(zip(cks, r["kernels"])):
        with c:
            _heat(k, f"kernel {i}", f"w09_k{i}", cs=[[0, "#2563EB"], [0.5, "#FFFDF9"], [1, "#DB2777"]], h=90)
    cols = st.columns(9)
    with cols[0]:
        st.caption("خرائط الخصائص →")
    for i, (c, fm) in enumerate(zip(cols[1:], r["fmaps"])):
        with c:
            _heat(fm, f"map {i}", f"w09_fm{i}", cs="Viridis")
    intuition("بعض النوى تشبه كواشف الحواف من الأسبوع 08 (صف موجب فوق صف سالب) — الشبكة **اكتشفتها** بالتدرج. الخرائط المضيئة على الخطوط الأفقية أو العمودية هي «الخصائص» التي يقرؤها الرأس الكثيف ليقرر الفئة. مع ReLU، الخرائط السوداء تمامًا = وحدات لم تستجب لهذه الصورة (وقد تكون ميتة إن كانت سوداء لكل الصور).")
    h2("2) تحليل الأخطاء", "2) Error analysis")
    r_hard = cnn_shapes_run(epochs=8, n_train=60)
    if r_hard["wrong"]:
        st.markdown(f"نموذج الـ60 صورة (دقة اختبار {r_hard['test_acc']:.2f}): أمثلة مصنَّفة خطأً — اقرأ النمط:")
        cols = st.columns(min(6, len(r_hard["wrong"])))
        for c, w in zip(cols, r_hard["wrong"]):
            with c:
                _heat(w["img"], f"true {r['classes'][w['true']][:5]} → pred {r['classes'][w['pred']][:5]} ({w['p']:.2f})", f"w09_wrong{w['true']}{w['pred']}{cols.index(c)}")
        st.markdown("**أنماط شائعة**: الصلبان الرفيعة تُقرأ شريطًا واحدًا (الخط الثاني ضاع في الضوضاء/التجميع)؛ الأشرطة قرب الحافة تعاني من الحشو؛ الثقة العالية في خطأ = النموذج واثق وخاطئ (مادة للزيادة أو لبيانات من تلك الحالة).")
    else:
        st.success("نموذج الـ60 صورة صنّف كل الاختبار صحيحًا في هذا التشغيل؛ زد الضوضاء أو قلّل الحقب لرؤية أخطاء.", icon="✅")
    h3("أين ينظر النموذج؟ حساسية الإخفاء", "Where does the model look? Occlusion sensitivity")
    oc = occlusion(0)
    if oc and "heat" in oc:
        n = len(oc["pos"])
        ocaps = []
        for k in range(n):
            r, q = oc["pos"][k]
            drop = oc["base"] - oc["p"][k]
            ocaps.append(f"**رقعة فارغة 4×4 عند ({r}, {q})**: احتمال الفئة الصحيحة {oc['p'][k]:.3f} (بدونها {oc['base']:.3f}) — انخفاض {drop:+.3f}."
                         + (" **انخفاض كبير**: هذه المنطقة حاسمة لقرار النموذج." if drop > 0.1 else ""))
        ocaps[-1] += " **الخريطة الكاملة**: المناطق الوردية/البنفسجية هي التي يعتمد عليها النموذج."
        animation_player("w09_occl", [Frame(occl_svg(oc, k), caption(c), action=f"patch {k + 1}/{n}") for k, c in enumerate(ocaps)],
                         title_ar="نغطي جزءًا من الصورة ونقيس كم يتغير القرار", interval_ms=380)
        interpretation_note("في هذه الصورة (صليب) الاحتمال لا يهبط إلا عند تغطية **منطقة التقاطع**: إخفاء جزء من شريط واحد لا يغيّر القرار لأن الشريطين ما زالا مرئيين، أما إخفاء التقاطع فيجعل الصورة تبدو كشريطين منفصلين. "
                            "هذا تفسير **لنموذج مدرَّب** على هذه البيانات، لا حقيقة عن الصلبان.")
        warning_note("حساسية الإخفاء مكلفة (تنبؤ لكل موضع) وتعتمد على حجم الرقعة ولون التعبئة. استعملها للتشخيص («هل ينظر النموذج إلى الشيء الصحيح أم إلى علامة مائية في الزاوية؟») لا كدليل سببي.")
    h2("3) أخطاء الأشكال في CNN", "3) CNN shape errors")
    compare_table(["الرسالة", "السبب", "الحل"],
                  [("`Input 0 of layer 'conv2d' is incompatible: expected min_ndim=4, found ndim=3`", "غاب بُعد القناة أو الدفعة: (600, 16, 16) أو (16, 16, 1)", "`X[..., None]` للقناة؛ `x[None, ...]` لصورة واحدة"), ("`expected axis -1 of input shape to have value 3, but received input with shape (None, 16, 16, 1)`", "النموذج بُني لـ 3 قنوات والصور رمادية (أو العكس)", "`Input(shape=(16, 16, X.shape[-1]))`"),
                   ("`Negative dimension size caused by subtracting 3 from 2`", "الصورة صارت أصغر من النواة بعد تجميعات كثيرة", "احسب بالحاسبة؛ قلّل Pool أو استخدم same"), ("`Total params: 52,428,810`", "Flatten على خريطة كبيرة ثم Dense", "Pool/stride قبل Flatten أو GlobalAveragePooling"),
                   ("دقة عشوائية مع صور PyTorch/OpenCV", "ترتيب القنوات (C, H, W) بدل (H, W, C)، أو BGR بدل RGB", "`np.transpose(x, (0, 2, 3, 1))`; تحقق من نطاق 0–1"), ("`ValueError: Shapes (32, 3) and (32, 1)`", "هدف one-hot مع sparse أو العكس", "الوحدة 21: ترميز الهدف")],
                  ["code", "rtl", "rtl"])
    debugging_note("سطر واحد قبل fit يمنع 4 من الأخطاء الخمسة: `print(X.shape, X.dtype, X.min(), X.max(), y.shape, np.unique(y))` → يجب أن ترى `(n, H, W, C) float32 0.0 1.0 (n,) [0 1 2]`. ثم `model.summary()` للخامس.")
    with st.container(horizontal=True):
        st.button("حاسبة أشكال CNN", icon=":material/science:", on_click=goto, args=("labs.cnn_shape_calculator",), key="w09_lab_calc2")
        st.button("معرض الأخطاء (الوحدة 22)", icon=":material/bug_report:", on_click=goto, args=("foundations.gallery.errors",), key="w09_go_errors")
    common_mistake("«الدقة 0.33 على 3 فئات بعد 20 حقبة» مع صور صحيحة الشكل: غالبًا القيم 0–255 بلا قسمة على 255 (تدرجات مشبعة)، أو y ليس 0..2. الشكل الصحيح لا يعني القيم الصحيحة.")
    post_test("course.w09", "09", [
        Q("قبل إدخال صور 0–255 إلى CNN:", ["لا شيء", "قسمة على 255 وإضافة بُعد القناة", "one-hot"], 1, ""),
        Q("60 صورة وشبكة بـ 10 آلاف معلمة:", ["تعميم ممتاز", "فرط تخصيص محتمل → زيادة/تنظيم", "قصور"], 1, ""),
        Q("الزيادة تُطبَّق على…", ["التدريب فقط", "الاختبار", "الكل"], 0, ""),
        Q("`expected min_ndim=4, found ndim=3`:", ["الدفعة كبيرة", "غابت قناة أو دفعة", "الخسارة خاطئة"], 1, ""),
        Q("خرائط خصائص الطبقة الأولى تُستخرج بـ…", ["model.predict", "keras.Model(model.inputs, layer.output)", "model.summary"], 1, ""),
        Q("`Negative dimension size`:", ["η كبير", "الصورة أصغر من النواة بعد تقليصات كثيرة", "dtype"], 1, ""),
        Q("قلب عمودي لرسم بياني لسعر:", ["زيادة مناسبة", "يكسر الفئة", "لا أثر"], 1, ""),
        Q("خريطة خصائص سوداء تمامًا لكل الصور:", ["نواة ممتازة", "وحدة ميتة محتملة", "الصورة فارغة"], 1, ""),
    ])
    takeaway("النوى والخرائط مرئية ومفسِّرة. الأخطاء تُقرأ من الصور الخاطئة. أخطاء الأشكال الخمسة تُمنع بسطر طباعة وsummary وحاسبة الأشكال.")
    lesson_footer(LESSON, ["النوى والخرائط الحية.", "تحليل الأخطاء وحساسية الإخفاء (تحريك).", "جدول أخطاء الأشكال والاختبار البعدي."])
