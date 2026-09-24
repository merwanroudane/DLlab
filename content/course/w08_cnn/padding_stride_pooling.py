import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, interpretation_note, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from content.course.w08_cnn._viz import geometry_svg, pool_data, pool_svg
from core.routing import go as goto
from labs.cnn import out_size

LESSON = Lesson(
    id="course.w08.padding_stride_pooling",
    title_ar="الحشو والخطوة والتجميع",
    title_en="Padding, Stride & Pooling",
    module="course.w08",
    order=3,
    prerequisites=["course.w08.convolution"],
    objectives_ar=["صيغة حجم المخرج floor((n + 2p − k)/s) + 1 وتطبيقها، ورؤية مواضع النواة مع الحشو والخطوة (تحريك).", "الحشو (same/valid) والخطوة وأثرهما على الحجم والحواف والحساب.", "التجميع الأقصى/المتوسط: لماذا، وماذا يفعل بالأشكال، وبلا معلمات."],
    terms=["shape", "dimension", "padding", "stride", "pooling", "kernel", "feature_map"],
    labs=["labs.padding_stride_lab"],
    difficulty="intermediate",
    summary_ar="out = ⌊(n + 2p − k)/s⌋ + 1. same: p = (k−1)/2 يحفظ الحجم؛ valid: p = 0 ينقص k−1. الخطوة s تقسم الحجم. MaxPool(2) يقسم على 2 بلا معلمات ويضيف ثباتًا للإزاحة.",
)

CODE = '''import numpy as np
from labs.cnn import conv2d, pool2d, out_size, KERNELS
img = np.zeros((8, 8)); img[3:5, 1:7] = 1.0                        # شريط أفقي في صورة 8×8
k = KERNELS["horizontal edge"]

print("n=8, k=3:")
for p, s in ((0, 1), (1, 1), (0, 2), (1, 2)):
    f = conv2d(img, k, padding=p, stride=s)
    print(f"  padding={p} stride={s} -> formula floor((8+2*{p}-3)/{s})+1 = {out_size(8, 3, p, s)} | actual {f.shape}")

# الحشو يحفظ معلومات الحافة
f_valid = conv2d(img, k); f_same = conv2d(img, k, padding=1)
print("\\nvalid: rows", f_valid.shape[0], "(edge pixels seen by fewer windows) | same: rows", f_same.shape[0], "(every pixel is a window centre)")

# التجميع
f = conv2d(img, k, padding=1)                                     # 8×8
print("\\nfeature map (same) row sums:", np.abs(f).sum(1).round(1))
pm = pool2d(f, 2, mode="max"); pa = pool2d(f, 2, mode="avg")
print("max pool 2 ->", pm.shape, "| avg pool 2 ->", pa.shape, "| parameters added: 0")
# ثبات للإزاحة: نقل الصورة بكسلًا واحدًا
img2 = np.roll(img, 1, axis=1)
f2 = conv2d(img2, k, padding=1); pm2 = pool2d(f2, 2, mode="max")
print("shift by 1 px: conv map changed cells =", int((f != f2).sum()), "| max-pooled map changed cells =", int((pm != pm2).sum()), "<- pooling absorbs small shifts")'''


def render() -> None:
    lesson_header(LESSON)
    equation(r"n_{out} = \left\lfloor \frac{n_{in} + 2p - k}{s} \right\rfloor + 1",
             [("n_{in}", "طول الصورة (أو خريطة الخصائص) على محور."), ("p", "الحشو: أصفار تُضاف على كل جانب."), ("k", "حجم النواة."), ("s", "الخطوة: كم بكسلًا تقفز النواة.")],
             meaning_ar="صيغة واحدة تحدد حجم كل خريطة خصائص في أي CNN. احسبها بيدك قبل summary.", example_ar="28، k=3، p=1، s=1 → 28 (same). 28، k=3، p=0، s=1 → 26 (valid). 28، k=3، p=1، s=2 → 14.", dl_link_ar="`padding='same'` تحسب p تلقائيًا؛ `strides=2` تقلّص.", title_ar="صيغة حجم المخرج")
    definition("**الحشو** `padding`: إضافة أصفار حول الصورة كي تصل النواة إلى الحواف (`same` يحفظ الحجم، `valid` بلا حشو). **الخطوة** `stride`: قفزة النواة بين موضعين؛ s = 2 يقلّص الحجم إلى النصف تقريبًا. **التجميع** `pooling`: نافذة (2×2 عادةً) تُستبدل بأقصى قيمة فيها (max) أو متوسطها (avg) — يقلّص الحجم، بلا معلمات، ويجعل الكشف أقل حساسية للإزاحات الصغيرة.")
    why("بلا حشو، كل طبقة تقضم k−1 بكسلًا وتُهمل الحواف؛ 10 طبقات 3×3 تُلغي 20 بكسلًا. الخطوة والتجميع يقلّصان الصورة **عمدًا** لتوسيع «مجال الرؤية» لكل نواة لاحقة ولتقليل الحساب — من 28×28 بكسلات إلى 7×7 أنماط عالية المستوى.")
    h2("الحشو والخطوة على الشبكة", "Padding and stride on the grid")
    configs = [(5, 3, 0, 1, "valid, stride 1: النواة لا تخرج عن الصورة؛ 5 − 3 + 1 = 3 مواضع لكل محور، وبكسلات الحواف تُمسح أقل من غيرها."),
               (5, 3, 1, 1, "same (p = 1): حلقة أصفار تسمح للنواة بأن تتمركز على كل بكسل، بما فيها الحواف ⇒ المخرج 5×5 بنفس حجم المدخل."),
               (5, 3, 1, 2, "same + stride 2: النواة تقفز بكسلين ⇒ 3×3 فقط. التقليص يحدث داخل الالتفاف نفسه بأوزان متعلَّمة."),
               (7, 3, 0, 2, "صورة 7×7 بلا حشو وخطوة 2: ⌊(7 − 3)/2⌋ + 1 = 3.")]
    gframes = []
    for n, k, p, s, txt in configs:
        o = out_size(n, k, p, s)
        for q in range(o * o):
            note = f" {txt}" if (q == 0 or q == o * o - 1) else ""
            gframes.append(Frame(geometry_svg(n, k, p, s, q), caption(f"**n = {n}، p = {p}، s = {s}** — الموضع {q + 1} من {o * o}.{note}"),
                                 action=f"p={p} s={s}", values=[("output", "", f"{o}×{o}")]))
    animation_player("w08_geom", gframes, title_ar="أين تقف النواة؟ أربعة إعدادات", interval_ms=450)
    code_lab(CodeLab(
        key="w08_psp", title_ar="الصيغة مقابل الفعلي، حشو الحواف، التجميع، وثبات الإزاحة", code=CODE, level="A",
        before=Before(goal_ar="التحقق من صيغة الحجم لأربع تراكيب (p, s)، رؤية ما يحفظه الحشو، وتطبيق التجميع ثم إثبات أنه يمتص إزاحة بكسل واحد.", stage_ar="الأسبوع 08: الحشو والخطوة والتجميع.",
                      inputs_ar="صورة 8×8 بشريط أفقي.", expected_ar="6/8/3/4 للتراكيب الأربعة مطابقة للصيغة؛ same يعطي 8 صفوف؛ التجميع يقسم على 2 بلا معلمات؛ الإزاحة تغيّر خلايا كثيرة في خريطة الالتفاف وقليلة بعد التجميع."),
        explain=[("6-9", "الصيغة والشكل الفعلي جنبًا إلى جنب: مطابقة دائمًا."), ("12-13", "valid يفقد صفًا من كل جانب؛ same يجعل كل بكسل مركز نافذة."), ("16-19", "التجميع: نافذة 2×2 → قيمة واحدة. الشكل ينصف؛ لا معلمات تُضاف."), ("21-24", "`np.roll` يزيح الصورة بكسلًا: خريطة الالتفاف تتغير كثيرًا، خريطة التجميع الأقصى أقل بكثير — الثبات للإزاحة.")],
        run=run_printed(CODE),
        after_ar="- الصيغة تعمل على كل محور على حدة (H وW قد يختلفان).\n- التجميع «يعمّم» الموقع: «توجد حافة أفقية في هذا الربع» بدل «في البكسل 3».\n- خطوة 2 في الالتفاف تعطي تقليصًا مشابهًا مع معلمات متعلَّمة — بديل حديث للتجميع.",
    ))
    st.button("افتح معمل الحشو والخطوة والتجميع", icon=":material/science:", type="primary", on_click=goto, args=("labs.padding_stride_lab",), key="w08_lab_psp")
    h3("التجميع الأقصى نافذةً نافذة", "Max-pooling window by window")
    pd_ = pool_data()
    pcaps = []
    for q in range(4):
        r, c = divmod(q, 2)
        win = pd_["f"][2 * r:2 * r + 2, 2 * c:2 * c + 2]
        pcaps.append(f"**النافذة ({r}, {c})**: القيم {win.ravel().tolist()} ⇒ الأقصى **{win.max():.0f}** (والمتوسط {win.mean():.2f}). أقوى استجابة للكاشف في هذا الربع تبقى، والباقي يُهمل.")
    pcaps[-1] += " النتيجة 2×2: نصف الطول، ربع المساحة، **صفر معلمات**."
    animation_player("w08_pool", [Frame(pool_svg(pd_, q), caption(c), action=f"window {q + 1}") for q, c in enumerate(pcaps)],
                     title_ar="خريطة خصائص حقيقية (حافة عمودية + ReLU) → max-pool 2×2", interval_ms=1800)
    interpretation_note("التجميع يجيب عن سؤال «هل رأى الكاشف نمطه **في مكان ما** من هذه المنطقة؟» بدل «أين بالضبط؟». هذا الفقد المقصود للموقع الدقيق هو ما يمنح الثبات للإزاحات الصغيرة.")
    compare_table(["الأداة", "الأثر على الحجم", "معلمات", "لماذا تستخدمه", "في Keras"],
                  [("padding='same'", "يحفظ H, W", "0", "طبقات كثيرة بلا فقدان الحواف", "`Conv2D(f, 3, padding='same')`"), ("padding='valid'", "−(k−1)", "0", "الافتراضي؛ عندما لا تهم الحواف", "`Conv2D(f, 3)`"), ("stride=2", "÷2 تقريبًا", "0 إضافية", "تقليص مع تعلّم (بدل التجميع)", "`Conv2D(f, 3, strides=2)`"),
                   ("MaxPooling2D(2)", "÷2", "0", "تقليص + ثبات للإزاحة + الأقوى يفوز", "`MaxPooling2D(2)`"), ("AveragePooling2D(2)", "÷2", "0", "تنعيم؛ أقل شيوعًا", "`AveragePooling2D(2)`"), ("GlobalAveragePooling2D", "(H, W, C) → (C,)", "0", "بديل Flatten يقلّل معلمات Dense", "`GlobalAveragePooling2D()`")],
                  ["code", "rtl", "num", "rtl", "code"])
    intuition("فكّر في CNN كقمع: الأبعاد المكانية (H, W) تنكمش عبر الطبقات بينما القنوات C (عدد الكواشف) تزداد: 28×28×1 → 14×14×32 → 7×7×64. الحشو والخطوة والتجميع هي أدوات الانكماش؛ filters أداة الاتساع.")
    common_mistake("MaxPooling(2) على خريطة 7×7 يعطي 3×3 (يهمل صفًا وعمودًا) لا 3.5. وتجميع 4 مرات على 28 يعطي 1×1: تحقق من الحجم بالصيغة قبل إضافة طبقة.")
    quiz("w08.psp", [
        Q("32، k=5، p=0، s=1 →", ["32", "28", "27"], 1, "32−5+1."),
        Q("64، k=3، p=1، s=2 →", ["64", "32", "31"], 1, "⌊(64+2−3)/2⌋+1 = 32."),
        Q("MaxPooling2D(2) تضيف معلمات:", ["0", "4", "بحسب القنوات"], 0, ""),
        Q("الحشو same مهم عندما…", ["الصورة صغيرة أو الطبقات كثيرة", "دائمًا ممنوع", "الدفعة كبيرة"], 0, ""),
    ])
    takeaway("out = ⌊(n+2p−k)/s⌋+1. same يحفظ، valid يقضم، stride يقسم، pooling يقسم بلا معلمات ويمتص الإزاحة. القمع: (H,W) تنكمش وC تتسع.")
    lesson_footer(LESSON, ["الصيغة الواحدة.", "الحشو والخطوة والتجميع بالكود.", "جدول الأدوات في Keras."])
