import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, intuition, math_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from content.course.w08_cnn._viz import channels_data, channels_svg, slide_data, slide_svg
from core.routing import go as goto
from labs.cnn import KERNELS, conv2d, shapes_dataset

LESSON = Lesson(
    id="course.w08.convolution",
    title_ar="الصورة كموتر، النواة، الالتفاف، وخريطة الخصائص",
    title_en="Image as Tensor, Kernel, Convolution & Feature Map",
    module="course.w08",
    order=2,
    prerequisites=["course.w08.overview", "foundations.linalg.tensors", "foundations.python.vectorization_broadcasting"],
    objectives_ar=["الصورة كموتر (H, W, C): بكسل، قناة، دفعة (B, H, W, C).", "الالتفاف: نواة صغيرة تنزلق، ضرب عنصري وجمع، خريطة خصائص — بالمعادلة وباليد وبتحريك النواة على صورة حقيقية موضعًا موضعًا.", "الالتفاف على عدة قنوات: شريحة نواة لكل قناة ثم جمع (تحريك).", "لماذا الالتفاف بدل Dense على الصور: المحلية ومشاركة الأوزان."],
    terms=["tensor", "channel_dimension", "hadamard_product", "weight", "convolution", "kernel", "feature_map", "cnn"],
    labs=["labs.cnn_convolution_lab"],
    difficulty="intermediate",
    summary_ar="صورة رمادية (H, W, 1)، ملونة (H, W, 3)؛ دفعة (B, H, W, C). الالتفاف: لكل موضع Σ(نافذة ⊙ نواة) + b. النواة كاشف نمط محلي بـ k²·C_in + 1 معلمة مهما كان حجم الصورة.",
)

CODE = '''import numpy as np
from labs.cnn import conv2d, KERNELS, shapes_dataset

# 1) الصورة كموتر
X, y = shapes_dataset(n=4, size=8, seed=1, noise=0.0)               # 4 صور 8×8 بقناة واحدة
print("batch tensor shape (B, H, W, C):", X.shape, "| dtype:", X.dtype, "| pixel range:", X.min(), "->", X.max())
img = X[0, :, :, 0]                                                # صورة واحدة كمصفوفة H×W
print("image #0 is class", y[0], "(0 horizontal bar, 1 vertical bar, 2 cross):")
print(np.where(img > 0.5, "█", "·").tolist().__str__().replace("'", "").replace(", ", "").replace("][", "]\\n["))

# 2) الالتفاف يدويًا: نواة 3×3 تنزلق، ضرب عنصري وجمع
k = KERNELS["horizontal edge"]
print("\\nkernel 'horizontal edge':\\n", k)
fmap = conv2d(img, k)                                              # بلا حشو، خطوة 1 -> (8-3+1) = 6×6
print("feature map shape:", fmap.shape, "(= (H − k + 1, W − k + 1))")
print(np.round(fmap, 1))
print("largest responses at rows:", np.unique(np.argwhere(np.abs(fmap) == np.abs(fmap).max())[:, 0]).tolist(), "<- where the bar's top/bottom edges are")

# 3) الخطوة الواحدة صراحةً: موضع (0, 3)
win = img[0:3, 3:6]
print("\\nwindow at (0,3):\\n", win, "\\nwindow ⊙ kernel summed =", float((win * k).sum()), "== fmap[0,3] =", round(float(fmap[0, 3]), 1))
# 4) المعلمات: نواة واحدة = 9 أوزان + انحياز، مهما كان حجم الصورة
for H in (8, 28, 224):
    print(f"image {H}×{H}: Dense(1) on flattened image needs {H*H + 1:>6} params | one 3×3 kernel needs {9 + 1} params")'''


def render() -> None:
    lesson_header(LESSON)
    h2("الصورة كموتر", "The image as a tensor")
    definition("**البكسل** رقم (شدة الضوء 0–255 أو 0–1). **الصورة الرمادية** مصفوفة (H, W)؛ **الملونة** موتر (H, W, 3) بثلاث **قنوات** (R, G, B). في Keras الشكل (H, W, C) بقناة أخيرة (channels-last)، و**الدفعة** (B, H, W, C) — رتبة 4. PyTorch يستخدم (B, C, H, W).")
    X, y = shapes_dataset(6, size=16, seed=2, noise=0.05)
    cols = st.columns(6)
    for c, i in zip(cols, range(6)):
        with c:
            fig = go.Figure(go.Heatmap(z=X[i, ::-1, :, 0], colorscale=[[0, "#FFFFFF"], [0.5, "#C4B5FD"], [1, "#4C1D95"]], showscale=False))
            fig.update_layout(height=120, margin=dict(l=0, r=0, t=18, b=0), title=dict(text=f"#{i}: {['h-bar', 'v-bar', 'cross'][y[i]]}", font=dict(size=10), x=0.5), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), paper_bgcolor="#FFFDF9")
            st.plotly_chart(fig, width="stretch", key=f"w08_img{i}")
    st.code(f"X.shape = {X.shape}   # (B=6, H=16, W=16, C=1)  ← rank 4\nX[0].shape = {X[0].shape}   # one image\nX[0, 5, 7, 0] = {X[0, 5, 7, 0]:.2f}   # one pixel", language="text")
    why("MLP على صورة 28×28 يسطّحها إلى 784 رقمًا مستقلًا: البكسل (5, 7) لا علاقة له عنده بجاره (5, 8)، ونقل الشكل بكسلًا واحدًا يعطيه مدخلًا «جديدًا» تمامًا. الصور محلية (الجوار مهم) ومتكررة (نفس الحافة تظهر في أماكن كثيرة). الالتفاف يبني هاتين الحقيقتين في البنية.")
    h2("الالتفاف", "Convolution")
    equation(r"F[i, j] = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} X[i+u,\, j+v]\; K[u, v] \;+\; b",
             [("K", "النواة (المرشّح): مصفوفة صغيرة k×k من الأوزان **المتعلَّمة**."), ("X[i+u, j+v]", "نافذة من الصورة بحجم النواة عند الموضع (i, j)."), ("F", "خريطة الخصائص: قيمة لكل موضع تنزلق إليه النواة."), ("b", "انحياز واحد لكل نواة.")],
             meaning_ar="نفس المجموع الموزون للخلية (الأسس 9) لكن على نافذة محلية، ونفس الأوزان تُستخدم في كل موضع.", example_ar="نواة «حافة أفقية» تعطي قيمة كبيرة حيث تنتقل الصورة من داكن إلى فاتح عموديًا.", dl_link_ar="`Conv2D(filters=16, kernel_size=3)` = 16 نواة 3×3 → 16 خريطة خصائص.", title_ar="الالتفاف ثنائي البعد")
    h2("شاهد النواة تنزلق", "Watch the kernel slide")
    sd = slide_data()
    scaps = []
    for k, stp in enumerate(sd["steps"]):
        v = stp["value"]
        kind = "حافة: أعلى النافذة داكن وأسفلها فاتح ⇒ استجابة **موجبة**" if v > 0 else ("حافة معاكسة: أعلى فاتح وأسفل داكن ⇒ استجابة **سالبة**" if v < 0 else "لا تغيّر عمودي في النافذة ⇒ **صفر**")
        scaps.append(f"**الموضع ({stp['i']}, {stp['j']})**: النافذة البرتقالية 3×3 تُضرب في النواة عنصرًا بعنصر (الشبكة البنفسجية) ثم تُجمع: Σ = **{v:+.0f}**. {kind}. القيمة تُكتب في الخلية ({stp['i']}, {stp['j']}) من خريطة الخصائص.")
    scaps[-1] += " **اكتملت الخريطة**: 36 موضعًا = (8 − 3 + 1)². نفس النواة (9 أوزان) استُعملت في كل الـ 36 — هذه مشاركة الأوزان."
    animation_player("w08_slide", [Frame(slide_svg(sd, k), caption(c), action=f"({s_['i']},{s_['j']})") for k, (c, s_) in enumerate(zip(scaps, sd["steps"]))],
                     title_ar="نواة «حافة أفقية» على صورة صليب 8×8 من بيانات المنصة", interval_ms=700)
    intuition("لاحظ الخريطة الناتجة: صف موجب حيث يبدأ الشكل (من الأعلى)، وصف سالب حيث ينتهي. النواة لم «ترَ» الصليب؛ رأت فقط أين يتغير السطوع عموديًا. الطبقات التالية تجمع هذه الإشارات البسيطة في أشكال.")
    h3("ماذا لو كانت الصورة ملونة؟ عدة قنوات", "What about colour? Multiple channels")
    cd = channels_data()
    ccaps = ["**ثلاث قنوات** (R، G، B): نفس النافذة 3×3 لكن من ثلاث طبقات لونية. المدخل هنا 3×3×3 = 27 رقمًا (أرقام صغيرة للتوضيح).",
             "**النواة أيضًا ثلاثية الأبعاد**: 3×3×**3** — شريحة لكل قناة. نواة واحدة لا تعمل على قناة واحدة بل على **كل** القنوات معًا.",
             f"**ضرب وجمع لكل قناة**: R يعطي {cd['per'][0]:+.0f}، G يعطي {cd['per'][1]:+.0f}، B يعطي {cd['per'][2]:+.0f}.",
             f"**جمع القنوات + انحياز واحد**: ({cd['per'][0]:+.0f}) + ({cd['per'][1]:+.0f}) + ({cd['per'][2]:+.0f}) + {cd['b']:.0f}.",
             f"**خلية واحدة** في خريطة خصائص **واحدة** = {cd['out']:+.0f}. النواة الواحدة تعطي خريطة واحدة مهما كان عدد القنوات؛ `filters=16` = 16 نواة من هذا النوع = 16 خريطة."]
    animation_player("w08_channels", [Frame(channels_svg(cd, i), caption(c), action=["channels", "3-D kernel", "per channel", "sum + b", "one cell"][i]) for i, c in enumerate(ccaps)],
                     title_ar="نواة 3×3×3 على صورة بثلاث قنوات", interval_ms=2200)
    math_note("معلمات نواة واحدة = k × k × C_in + 1 (انحياز). طبقة بـ C_out نواة: `k²·C_in·C_out + C_out`. على صورة RGB بنواة 3×3 و16 مرشحًا: 3·3·3·16 + 16 = 448 معلمة — لأي حجم صورة.")
    code_lab(CodeLab(
        key="w08_conv", title_ar="الصورة كموتر، الالتفاف يدويًا، خطوة واحدة صراحةً، ومعلمات النواة", code=CODE, level="A",
        before=Before(goal_ar="قراءة شكل دفعة صور، تنفيذ الالتفاف بنواة كاشفة للحواف الأفقية، التحقق من موضع واحد يدويًا، ومقارنة معلمات النواة بمعلمات Dense.", stage_ar="الأسبوع 08: الالتفاف.",
                      inputs_ar="4 صور 8×8 تركيبية.", expected_ar="شكل (4, 8, 8, 1)؛ خريطة 6×6 بقيم كبيرة عند حافتي الشريط؛ الموضع (0,3) يطابق؛ 10 معلمات للنواة مهما كبرت الصورة."),
        explain=[("4-9", "الدفعة رتبة 4؛ الصورة الواحدة مصفوفة؛ نطبعها كرموز لنرى الشكل."), ("12-17", "نواة الحافة الأفقية (−1 فوق، +1 تحت): الاستجابة كبيرة حيث يتغير السطوع عموديًا. الحجم 8−3+1 = 6."),
                 ("20-21", "الموضع الواحد: نافذة 3×3 ⊙ نواة ثم جمع = قيمة خريطة الخصائص عند (0,3) — لا سحر."), ("23-25", "**مشاركة الأوزان**: نواة واحدة = 10 معلمات لأي حجم صورة؛ Dense على 224×224 تحتاج 50 ألف معلمة لوحدة واحدة.")],
        run=run_printed(CODE),
        after_ar="- الاستجابات الكبيرة عند صفوف حافتي الشريط: النواة **كاشف** لنمط محلي.\n- في CNN لا نكتب النواة؛ الشبكة تتعلم قيمها بالتدرج (الأسس 14) — مثل أوزان Dense لكنها صغيرة ومشتركة.\n- 16 نواة = 16 كاشفًا مختلفًا = 16 خريطة خصائص: هذا `filters=16`.",
    ))
    st.button("افتح معمل الالتفاف: شاهد النواة تتحرك", icon=":material/science:", type="primary", on_click=goto, args=("labs.cnn_convolution_lab",), key="w08_lab_conv")
    h2("نوى جاهزة للتحسس", "Classic kernels to build intuition")
    img = X[2, :, :, 0]
    cols = st.columns(4)
    for c, name in zip(cols, ["identity", "vertical edge", "horizontal edge", "blur (box)"]):
        with c:
            f = conv2d(img, KERNELS[name], padding=1)
            fig = go.Figure(go.Heatmap(z=f[::-1], colorscale=[[0, "#2563EB"], [0.5, "#FFFFFF"], [1, "#DB2777"]], zmid=0, showscale=False))
            fig.update_layout(height=170, margin=dict(l=0, r=0, t=20, b=0), title=dict(text=name, font=dict(size=11), x=0.5), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), paper_bgcolor="#FFFDF9")
            st.plotly_chart(fig, width="stretch", key=f"w08_k_{name}")
    intuition("«حافة عمودية» ترى الخطوط العمودية للصليب وتتجاهل الأفقية؛ «حافة أفقية» العكس. الطبقة الأولى في CNN مدرَّبة تتعلم نوى تشبه هذه بالضبط؛ الطبقات التالية تركّب منها أشكالًا (زوايا، دوائر) ثم كائنات.")
    compare_table(["", "Dense على صورة مسطّحة", "Conv2D"],
                  [("ما تراه الوحدة", "كل البكسلات", "نافذة k×k محلية"), ("المعلمات", "H·W·C لكل وحدة", "k²·C_in لكل نواة (مشتركة)"), ("الإزاحة", "شكل منقول = مدخل مختلف", "نفس النواة تكشفه في أي موضع"), ("المخرج", "رقم لكل وحدة", "خريطة خصائص لكل نواة"), ("الأسبوع", "04", "08")],
                  ["rtl", "rtl", "rtl"])
    common_mistake("تمرير صورة بشكل (28, 28) إلى Conv2D: تتوقع (H, W, C) — أضف بُعد القناة `x[..., None]` → (28, 28, 1). وللدفعة (B, 28, 28, 1). خطأ الشكل الأول في كل مشروع CNN.")
    quiz("w08.conv", [
        Q("دفعة من 64 صورة ملونة 32×32 في Keras:", ["(64, 3, 32, 32)", "(64, 32, 32, 3)", "(32, 32, 3)"], 1, "channels-last."),
        Q("نواة 5×5 على 3 قنوات: معلماتها", ["25", "75", "76"], 2, "5·5·3 + 1."),
        Q("`Conv2D(32, 3)` ينتج…", ["32 رقمًا", "32 خريطة خصائص", "3 خرائط"], 1, ""),
        Q("مشاركة الأوزان تعني…", ["نفس النواة في كل موضع", "أوزان Dense مشتركة", "لا انحياز"], 0, ""),
        Q("صورة 10×10 ونواة 3×3 بلا حشو: خريطة الخصائص", ["10×10", "8×8", "7×7"], 1, "10 − 3 + 1."),
        Q("نواة واحدة على صورة RGB تنتج…", ["3 خرائط", "خريطة واحدة (تجمع القنوات)", "9 خرائط"], 1, ""),
        Q("نواة «حافة أفقية» تعطي صفرًا في منطقة…", ["فيها حافة", "متجانسة السطوع عموديًا", "داكنة"], 1, ""),
    ])
    takeaway("صورة = موتر (H, W, C)؛ دفعة رتبة 4. الالتفاف = نواة صغيرة متعلَّمة تنزلق: Σ(نافذة ⊙ نواة) + b → خريطة خصائص. محلية + مشاركة أوزان = معلمات قليلة وكشف مستقل عن الموضع.")
    lesson_footer(LESSON, ["الصورة والدفعة كموترات.", "النواة تنزلق موضعًا موضعًا (تحريك).", "عدة قنوات (تحريك) وعدّ المعلمات.", "الالتفاف بالمعادلة واليد.", "لماذا Conv بدل Dense."])
