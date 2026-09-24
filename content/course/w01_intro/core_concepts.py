import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, intuition, math_note, research_note, takeaway, warning_note, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from content.course.w01_intro._viz import fmt, frame_epochs, gd_trace, inference_svg, nested_svg, pipelines_svg, training_svg
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline

LESSON = Lesson(
    id="course.w01.core_concepts",
    title_ar="المفاهيم الأساسية: نموذج، بنية، تدريب، استدلال",
    title_en="Core Concepts: Model, Architecture, Training, Inference",
    module="course.w01",
    order=2,
    prerequisites=["course.w01.overview", "foundations.start.model_training_prediction"],
    objectives_ar=[
        "التمييز بين الذكاء الاصطناعي والتعلم الآلي والتعلم العميق، وشرح ما يضيفه «العمق».",
        "وصف الشبكة العصبية كسلسلة تحويلات من المدخل إلى المخرج.",
        "التمييز بين البنية (التصميم) والنموذج (البنية + المعلمات المدرَّبة).",
        "مشاهدة التدريب وهو يحدث: كيف تتحرك w و b وتنخفض الخسارة حقبة بعد حقبة، ومتى يفشل.",
        "اشتقاق قاعدة التحديث من الخسارة خطوة بخطوة، ثم استعمال النموذج المجمّد في الاستدلال.",
        "وصف دورة حياة النموذج: تصميم ← تدريب ← تقييم ← تشخيص ← استدلال.",
    ],
    terms=["artificial_intelligence", "machine_learning", "deep_learning", "model", "parameter", "weight", "bias",
           "loss", "mse", "gradient", "learning_rate", "epoch", "training", "inference"],
    related=["foundations.start.ai_ml_dl"],
    labs=["labs.gradient_descent_lab", "labs.learning_rate_lab"],
    difficulty="beginner",
    summary_ar="AI ⊃ ML ⊃ DL. الشبكة سلسلة طبقات؛ البنية تصميم، والنموذج بنية + معلمات مدرَّبة. التدريب = تكرار: تنبؤ ← خسارة ← تدرج ← تحديث. الاستدلال = استعمال المعلمات المجمّدة.",
)

GD_CODE = '''import numpy as np
rng = np.random.default_rng(5)
hours = rng.uniform(0.5, 10, size=40).round(1)                          # x: ساعات المذاكرة (الخاصية)
score = (45 + 4.6 * hours + rng.normal(0, 3.5, size=40)).clip(0, 100).round(1)  # y: الدرجة (الهدف)

w, b = 0.0, 0.0                    # المعلمات: نبدأ من الصفر
eta = {eta}                        # معدل التعلم η (متغير فائق يختاره الباحث)
for epoch in range(1, {epochs} + 1):
    y_hat = w * hours + b          # 1) تنبؤ لكل الطلاب
    err = y_hat - score            # 2) الخطأ لكل طالب
    loss = np.mean(err ** 2)       # 3) الخسارة MSE (قبل التحديث)
    grad_w = 2 * np.mean(err * hours)   # 4) ∂L/∂w
    grad_b = 2 * np.mean(err)           #    ∂L/∂b
    w = w - eta * grad_w           # 5) خطوة عكس اتجاه التدرج
    b = b - eta * grad_b
    if epoch in (1, 2, 10, 100) or epoch == {epochs}:
        print(f"epoch {{epoch:4d}} | loss {{loss:10.2f}} | w {{w:8.3f}} | b {{b:8.3f}}")

print("least-squares answer (np.polyfit):", np.polyfit(hours, score, 1).round(3))
print("inference for a new student with 7 hours:", round(w * 7 + b, 1))'''


def _layers_svg() -> str:
    s = '<svg viewBox="0 0 640 230" width="100%" style="max-width:640px">' + svg_defs()
    boxes = [("Input", "#E6F1FB", "#2F6FB5"), ("Layer 1", "#EFE9FA", "#7C5CBF"), ("Layer 2", "#EFE9FA", "#7C5CBF"),
             ("Output", "#DDF5EA", "#2E8B57")]
    x = 30
    for label, fill, stroke in boxes:
        s += svg_box(x, 80, 120, 60, label, fill, stroke=stroke, font=15, bold=True)
        x += 160
    for i in range(3):
        s += svg_arrow(30 + 120 + i * 160, 110, 30 + 160 * (i + 1) - 2, 110)
    s += svg_text(90, 165, "x  (features)", size=12, color="#2F6FB5")
    s += svg_text(250, 165, "a₁ = f(x W₁ + b₁)", size=12, color="#5B4A88")
    s += svg_text(410, 165, "a₂ = f(a₁ W₂ + b₂)", size=12, color="#5B4A88")
    s += svg_text(570, 165, "ŷ  (prediction)", size=12, color="#2E8B57")
    s += svg_text(320, 40, "Architecture = how many layers, how wide, which activations", size=13, bold=True)
    s += svg_text(320, 205, "Model = architecture + trained parameters (W₁, b₁, W₂, b₂, …)", size=13, color="#6B675F")
    return s + "</svg>"


def _lifecycle_frames() -> list[Frame]:
    stages = [
        ("Define problem", "**تعريف المسألة**: ما الذي نتنبأ به؟ رقم (انحدار) أم فئة (تصنيف)؟ ما الملاحظة الواحدة؟ مثال: «درجة الطالب من ساعات مذاكرته» — انحدار، والملاحظة طالب.",
         ("problem", "?", "regression")),
        ("Prepare data", "**إعداد البيانات**: تنظيف القيم المفقودة، ترميز الفئات، تحجيم الأرقام، ثم **تقسيم** البيانات إلى تدريب/تحقق/اختبار **قبل** أي تعلم حتى لا تتسرب معلومات الاختبار.",
         ("data", "raw", "train / val / test")),
        ("Design architecture", "**تصميم البنية**: قرار الباحث قبل التدريب — عدد الطبقات والوحدات ودوال التنشيط. هنا المعلمات موجودة لكنها **عشوائية** بلا معنى بعد.",
         ("W, b", "—", "random")),
        ("Train", "**التدريب**: حلقة تتكرر: تنبؤ ← خسارة ← تدرج ← تحديث. هذه هي المرحلة **الوحيدة** التي تتغير فيها المعلمات.",
         ("W, b", "random", "learned")),
        ("Evaluate", "**التقييم**: قياس الخسارة/الدقة على بيانات **لم يرها** النموذج أثناء التدريب. المعلمات مجمّدة هنا.",
         ("test loss", "?", "measured")),
        ("Diagnose & fix", "**التشخيص والإصلاح**: قراءة منحنيات التدريب والتحقق: فرط تخصيص؟ قصور تعلم؟ معدل تعلم كبير؟ ثم نعود لمرحلة سابقة ونعدّل. الدورة ليست خطًا مستقيمًا.",
         ("decision", "?", "loop back / accept")),
        ("Inference", "**الاستدلال**: استعمال النموذج المجمّد على بيانات جديدة لإنتاج تنبؤات. لا خسارة ولا تدرج ولا تحديث — فقط تمرير أمامي.",
         ("W, b", "learned", "frozen")),
    ]
    frames = []
    for i, (name, text, val) in enumerate(stages):
        svg = '<svg viewBox="0 0 660 90" width="100%" style="max-width:660px">' + svg_defs()
        for j, (nm, _, _) in enumerate(stages):
            on = j == i
            fill = "#1F7A78" if on else ("#E0F3F1" if j < i else "#FFFDF8")
            s_box = svg_box(4 + j * 94, 20, 84, 44, nm.split()[0], fill, stroke="#1F7A78" if j <= i else "#D9D3C7",
                            font=11, bold=on, text_color="#fff" if on else "#2B2A28")
            svg += s_box
            if j < len(stages) - 1:
                svg += svg_arrow(88 + j * 94, 42, 97 + j * 94, 42)
        if i == 5:
            svg += '<path d="M 520 66 Q 400 95 280 66" fill="none" stroke="#C8473A" stroke-dasharray="4 3" stroke-width="1.6" marker-end="url(#arrowhead)"/>'
        svg += "</svg>"
        frames.append(Frame(svg, caption(text), action=name, values=[val], highlight=i))
    return frames


def _derivation_steps(tr: dict, eta: float) -> list[tuple[str, str, str]]:
    """(caption, unicode equation for the player, full LaTeX shown under the player)."""
    gw, gb = tr["gw"][0], tr["gb"][0]
    w1, b1 = tr["w"][1], tr["b"][1]
    return [
        ("**النموذج**: تنبؤ الطالب `i` هو خط مستقيم. `w` (الوزن) = كم نقطة تضيفها كل ساعة، و`b` (الانحياز) = الدرجة المتوقعة عند صفر ساعات.",
         "ŷᵢ = w·xᵢ + b", r"\hat y_i = w\,x_i + b"),
        ("**الخطأ**: الفرق بين التنبؤ والحقيقة لكل طالب. موجب = بالغنا في التقدير، سالب = قلّلنا.",
         "eᵢ = ŷᵢ − yᵢ = w·xᵢ + b − yᵢ", r"e_i = \hat y_i - y_i = w\,x_i + b - y_i"),
        ("**الخسارة MSE**: متوسط مربعات الأخطاء. التربيع يجعل كل خطأ موجبًا ويعاقب الأخطاء الكبيرة أكثر. هدف التدريب: أصغر L ممكنة.",
         "L(w, b) = (1/n) Σ eᵢ²", r"L(w,b) = \frac{1}{n}\sum_{i=1}^{n} e_i^{2}"),
        ("**قاعدة السلسلة**: L تعتمد على w عبر كل eᵢ. مشتقة `e²` بالنسبة لـ e هي `2e`، ثم نضرب في مشتقة e بالنسبة لـ w.",
         "∂L/∂w = (1/n) Σ 2eᵢ · ∂eᵢ/∂w", r"\frac{\partial L}{\partial w} = \frac{1}{n}\sum_i 2e_i\,\frac{\partial e_i}{\partial w}"),
        ("**مشتقة الخطأ**: في `eᵢ = w·xᵢ + b − yᵢ` معامل w هو `xᵢ`، ومعامل b هو 1. لذلك الطالب الذي ذاكر أكثر «يدفع» w أكثر.",
         "∂eᵢ/∂w = xᵢ ,  ∂eᵢ/∂b = 1", r"\frac{\partial e_i}{\partial w} = x_i,\qquad \frac{\partial e_i}{\partial b} = 1"),
        ("**التدرجان**: نعوّض فنحصل على صيغتين قابلتين للحساب مباشرة بمتوسطين.",
         "∂L/∂w = (2/n) Σ eᵢ·xᵢ ,  ∂L/∂b = (2/n) Σ eᵢ", r"\frac{\partial L}{\partial w} = \frac{2}{n}\sum_i e_i x_i,\qquad \frac{\partial L}{\partial b} = \frac{2}{n}\sum_i e_i"),
        ("**قاعدة التحديث (النزول بالتدرج)**: التدرج يشير إلى اتجاه **صعود** الخسارة، فنمشي عكسه بخطوة طولها η (معدل التعلم).",
         "w ← w − η·∂L/∂w ,  b ← b − η·∂L/∂b", r"w \leftarrow w - \eta\,\frac{\partial L}{\partial w},\qquad b \leftarrow b - \eta\,\frac{\partial L}{\partial b}"),
        (f"**بالأرقام (الحقبة الأولى، w = b = 0، η = {eta:g})**: كل التنبؤات 0 فكل الأخطاء سالبة كبيرة، فالتدرجان سالبان، فتزيد w و b. هذه الأرقام محسوبة فعلًا على بيانات الأربعين طالبًا.",
         f"∂L/∂w = {gw:.2f} → w = 0 − {eta:g}·({gw:.2f}) = {w1:.3f} ;  ∂L/∂b = {gb:.2f} → b = {b1:.3f}",
         rf"\frac{{\partial L}}{{\partial w}} = {gw:.2f}\;\Rightarrow\; w_1 = 0 - {eta:g}\times({gw:.2f}) = {w1:.3f},\qquad \frac{{\partial L}}{{\partial b}} = {gb:.2f}\;\Rightarrow\; b_1 = {b1:.3f}"),
    ]


def render() -> None:
    lesson_header(LESSON)

    # ------------------------------------------------------------------ AI ⊃ ML ⊃ DL
    h2("الذكاء الاصطناعي ⊃ التعلم الآلي ⊃ التعلم العميق", "AI ⊃ ML ⊃ DL")
    definition(
        "**الذكاء الاصطناعي** `AI`: كل نظام يؤدي مهمة تحتاج «ذكاءً» — قد يكون قواعد مكتوبة يدويًا (إذا كان الدخل < X فارفض القرض). "
        "**التعلم الآلي** `ML`: فرع من AI لا نكتب فيه القواعد، بل **نتعلم دالة f من البيانات**: نعطيه أمثلة (مدخل، إجابة صحيحة) فيجد f تقرّب بينهما. "
        "**التعلم العميق** `DL`: فرع من ML تكون فيه f **شبكة عصبية من طبقات كثيفة متعددة**، وكل طبقة تتعلم تمثيلًا جديدًا للمدخل."
    )
    diagram("دوائر متداخلة: كل تعلم عميق تعلم آلي، وليس العكس", nested_svg(),
            what_ar="ثلاث دوائر متداخلة: AI الأوسع، داخله ML، وداخله DL.",
            how_ar="كل نقطة داخل دائرة DL هي أيضًا داخل ML وداخل AI. الانحدار الخطي مثلًا داخل ML لكنه خارج DL.",
            takeaway_ar="DL ليس بديلًا عن ML بل حالة خاصة منه: نفس منطق «التعلم من البيانات» مع دالة f أعمق وأكثر مرونة.",
            title_en="AI ⊃ ML ⊃ DL")
    h3("ما الذي يضيفه «العمق» فعلًا؟", "What depth adds")
    diagram("خط التعلم الآلي الكلاسيكي مقابل التعلم العميق", pipelines_svg(),
            what_ar="صفّان: في الأعلى الباحث يصنع الخصائص يدويًا ثم نموذج بسيط؛ في الأسفل الطبقات تتعلم الخصائص والنموذج معًا.",
            how_ar="الفرق في الصندوق الأوسط: من يقرر «ما المهم في البيانات»؟ الباحث (ML كلاسيكي) أم الشبكة نفسها (DL)؟",
            takeaway_ar="قوة DL = تعلّم التمثيل `representation learning`. ثمنها: بيانات أكثر، حساب أكثر، وتفسير أصعب.",
            title_en="Hand-crafted features vs learned features")
    compare_table(
        ["", "تعلم آلي كلاسيكي", "تعلم عميق"],
        [
            ("الخصائص", "يصنعها الباحث (نسبة الدين، متوسطات متحركة…)", "تتعلمها الطبقات من البيانات الخام"),
            ("حجم البيانات المناسب", "مئات إلى آلاف الصفوف تكفي غالبًا", "آلاف إلى ملايين؛ يلمع مع البيانات الكبيرة"),
            ("نوع البيانات الذي يتفوق فيه", "جداول منظمة", "صور، نصوص، أصوات، سلاسل طويلة"),
            ("الحساب", "ثوانٍ على معالج عادي", "قد يحتاج `GPU` وساعات"),
            ("القابلية للتفسير", "أعلى (معاملات، أهمية خصائص)", "أصعب؛ تحتاج أدوات تفسير خاصة"),
            ("أمثلة", "انحدار خطي/لوجستي، أشجار، `Random Forest`", "`MLP`، `CNN`، `RNN`/`LSTM`/`GRU`، `Transformer`"),
        ],
        ["rtl", "rtl", "rtl"],
    )
    intuition("في جدول قروض، الباحث الخبير يعرف أن «نسبة الدين إلى الدخل» مهمة فيحسبها بنفسه — التعلم الآلي الكلاسيكي ممتاز هنا. "
              "لكن في صورة فاتورة لا أحد يعرف كيف يكتب «خاصية التزوير» بيده؛ هنا تتعلم طبقات CNN الحواف ثم الأشكال ثم الأنماط المشبوهة بنفسها.")

    # ------------------------------------------------------------------ network in one sentence
    h2("الشبكة العصبية في جملة واحدة", "A neural network in one sentence")
    definition(
        "**الشبكة العصبية** سلسلة من **الطبقات**؛ كل طبقة تأخذ متجهًا، تضربه في مصفوفة **أوزان** `W`، تضيف **انحيازًا** `b`، "
        "وتمرر النتيجة عبر **دالة تنشيط** `f` غير خطية. تراكب هذه الطبقات هو ما يجعلها «عميقة»."
    )
    intuition("كل طبقة تعيد وصف المدخل بلغة جديدة أكثر تجريدًا: من بكسلات إلى حواف، إلى أشكال، إلى «هذه فاتورة».")
    diagram(
        "من المدخل إلى التنبؤ عبر الطبقات",
        _layers_svg(),
        what_ar="أربعة صناديق: المدخل، طبقتان مخفيتان، المخرج. كل سهم = تحويل خطي (ضرب في W وإضافة b) متبوع بتنشيط f.",
        how_ar="اقرأ من اليسار إلى اليمين. نكتب المدخل كصف `x` ونضربه من اليمين في W (`x W`) — نفس الاصطلاح الذي تستعمله Keras والأسبوع 04.",
        takeaway_ar="البنية تصف الصناديق والأسهم؛ النموذج هو البنية بعد أن تعلمت قيم W و b.",
        title_en="Input → Layers → Output",
    )

    h2("بنية أم نموذج؟", "Architecture vs model")
    compare_table(
        ["", "البنية", "النموذج"],
        [
            ("English", "Architecture", "Model"),
            ("ما هي؟", "تصميم: عدد الطبقات، عرضها، دوال التنشيط، الروابط", "البنية + قيم المعلمات بعد التدريب"),
            ("متى تُحدد؟", "قبل التدريب (قرار الباحث)", "بعد التدريب"),
            ("ما الذي يُحفظ في ملف؟", "وصف الطبقات فقط (config)", "الوصف + كل الأوزان (weights)"),
            ("مثال", "MLP بطبقتين مخفيتين من 64 وحدة", "نفس الـMLP بأوزان محددة تتنبأ بالتعثر بدقة 87%"),
        ],
        ["rtl", "rtl", "rtl"],
    )
    why("عندما تُبلّغ عن نتيجة بحثية يجب أن تذكر **البنية** (ليستطيع غيرك إعادة البناء) و**إجراء التدريب** (البيانات، معدل التعلم، عدد الحقب، البذرة العشوائية) ليستطيع الوصول إلى نموذج مماثل.")

    # ------------------------------------------------------------------ smallest model
    h2("أصغر نموذج ممكن: عصبون واحد يتنبأ بالدرجة", "The smallest model: one neuron")
    st.markdown(
        "لنرَ التدريب بعينك قبل أي شبكة كبيرة. البيانات: **40 طالبًا**، الخاصية `x` = ساعات المذاكرة، والهدف `y` = درجة الامتحان. "
        "النموذج عصبون واحد بلا تنشيط (انحدار خطي) بمعلمتين فقط: `w` و `b`. **كل** شبكة عميقة تتدرب بنفس الحلقة التي ستراها الآن — فقط بملايين المعلمات بدل اثنتين."
    )
    equation(r"\hat y = w\,x + b",
             [(r"x", "المدخل: ساعات المذاكرة (خاصية واحدة)."), (r"w", "الوزن: كم نقطة تضيفها كل ساعة إضافية — معلمة تُتعلَّم."),
              (r"b", "الانحياز: الدرجة المتوقعة عند 0 ساعة — معلمة تُتعلَّم."), (r"\hat y", "التنبؤ (يُقرأ «y قبعة»).")],
             meaning_ar="خط مستقيم؛ التدريب يبحث عن الميل والتقاطع اللذين يجعلان الخط أقرب ما يمكن إلى النقاط.",
             example_ar="لو w = 4.6 و b = 45 فطالب ذاكر 6 ساعات: ŷ = 4.6×6 + 45 = 72.6.",
             dl_link_ar="هذه هي معادلة الطبقة `x W + b` نفسها بخاصية واحدة ووحدة واحدة وبلا تنشيط.", title_ar="نموذج العصبون الواحد")

    # ------------------------------------------------------------------ live training
    h2("شاهد التدريب وهو يحدث", "Watch training happen (live)")
    st.markdown("اختر **معدل التعلم** `η` وعدد **الحقب** (الحقبة = مرور كامل على الطلاب الأربعين)، ثم اضغط ▶. "
                "كل إطار محسوب فعليًا: الخط الأحمر هو النموذج الحالي، والمتقطع الرمادي هو أفضل خط ممكن (المربعات الصغرى).")
    c1, c2 = st.columns(2)
    with c1:
        eta = st.select_slider("معدل التعلم η", options=[0.001, 0.01, 0.02, 0.03], value=0.02, key="w01_eta")
    with c2:
        epochs = st.select_slider("عدد الحقب", options=[50, 200, 600], value=600, key="w01_epochs")
    tr = gd_trace(float(eta), int(epochs))
    n_done = len(tr["loss"]) - 1
    diverged = n_done < epochs or tr["loss"][-1] > tr["loss"][0]
    frames = []
    shown = frame_epochs(n_done)
    for idx, k in enumerate(shown):
        prev = shown[idx - 1] if idx else 0          # compare with the previous *frame*
        if k == 0:
            cap = "**الحقبة 0**: البداية `w = 0` و `b = 0`: الخط أفقي على الصفر، فالنموذج يتنبأ بدرجة 0 لكل الطلاب والخسارة ضخمة."
        elif diverged and k == n_done:
            cap = f"**انفجار!** مع η = {eta:g} كل خطوة تتجاوز القاع وتقفز إلى الجهة الأخرى أبعد مما كانت، فتتضاعف الخسارة حتى تصبح لا نهائية. هذا **فشل معدل التعلم الكبير**."
        elif tr["loss"][k] > tr["loss"][prev]:
            cap = f"**الحقبة {k}**: منذ الحقبة {prev} الخسارة **ارتفعت** ({fmt(tr['loss'][prev])} ← {fmt(tr['loss'][k])}): الخطوة أكبر من اللازم فتجاوزت القاع. علامة تحذير من η."
        else:
            cap = (f"**الحقبة {k}**: التحديثات منذ الحقبة {prev} حرّكت الخط نحو النقاط. الخسارة {fmt(tr['loss'][prev])} ← **{fmt(tr['loss'][k])}**. "
                   f"`w` تتعلم الميل و`b` التقاطع؛ لاحظ أن `b` أبطأ بكثير لأن تدرجها أصغر.")
        frames.append(Frame(training_svg(tr, k), caption(cap), action=f"epoch {k}",
                            equation=f"ŷ = {fmt(tr['w'][k], 3)}·x + {fmt(tr['b'][k], 3)}     L = {fmt(tr['loss'][k])}",
                            values=[("w", fmt(tr["w"][prev], 3), fmt(tr["w"][k], 3)), ("b", fmt(tr["b"][prev], 3), fmt(tr["b"][k], 3)),
                                    ("loss", fmt(tr["loss"][prev]), fmt(tr["loss"][k]))],
                            highlight=3))
    animation_player(f"w01_train_{eta}_{epochs}", frames, title_ar=f"التدريب بالنزول بالتدرج — η = {eta:g}",
                     stages=["Predict ŷ", "Loss", "Gradient", "Update w, b"], interval_ms=900)
    ow, ob = tr["ols"]
    if diverged:
        warning_note(f"مع η = {eta:g} انفجرت الخسارة بعد {n_done} حقبة. الحل: معدل تعلم أصغر. (سنرى في الأسبوع 02 أن **تحجيم** الخاصية يسمح بمعدل أكبر دون انفجار.)")
    else:
        gap = tr["loss"][-1] - tr["ols_loss"]
        st.markdown(f"بعد {n_done} حقبة: `w = {tr['w'][-1]:.3f}` و `b = {tr['b'][-1]:.3f}` والخسارة **{tr['loss'][-1]:.2f}**. "
                    f"أفضل خط ممكن: `w = {ow:.3f}` و `b = {ob:.3f}` بخسارة {tr['ols_loss']:.2f}. "
                    + ("**وصلنا عمليًا إلى القاع.**" if gap < 0.5 else f"ما زال الفرق {gap:.2f}: التدريب **لم يكتمل** — زد الحقب أو η."))
    compare_table(
        ["η", "ماذا ترى؟", "التشخيص"],
        [("0.001", "الخسارة تنخفض ببطء شديد و b بالكاد يتحرك", "بطيء جدًا: هدر حقب"),
         ("0.01", "تنخفض بثبات لكن لا تبلغ القاع في 600 حقبة", "آمن لكن بطيء"),
         ("0.02", "تبلغ أفضل خط تقريبًا", "مناسب لهذه البيانات"),
         ("0.03", "ترتفع وتتذبذب ثم تنفجر", "كبير جدًا: تباعد `divergence`")],
        ["ltr", "rtl", "rtl"],
    )
    intuition("تخيل كرة في وادٍ. التدرج يخبرك بميل الأرض تحت قدميك؛ η طول خطوتك. خطوة قصيرة جدًا = تصل بعد دهر. خطوة طويلة جدًا = تقفز فوق القاع إلى الجدار المقابل أعلى مما كنت.")

    # ------------------------------------------------------------------ derivation
    h2("من أين جاءت قاعدة التحديث؟ اشتقاق خطوة بخطوة", "Deriving the update rule, step by step")
    st.markdown("اضغط ⏭ لتكشف خطوة واحدة في كل مرة؛ عند التوقف تظهر المعادلة بصيغتها الرياضية الكاملة أسفل المشغّل.")
    tr_d = gd_trace(0.02, 1)
    steps = _derivation_steps(tr_d, 0.02)
    dframes = [Frame("", caption(c), action=f"step {i + 1}", equation=u) for i, (c, u, _) in enumerate(steps)]
    cur = animation_player("w01_derive", dframes, title_ar="اشتقاق النزول بالتدرج لخسارة MSE", interval_ms=2600)
    with st.container(border=True):
        st.markdown(f"**الخطوة {cur + 1} من {len(steps)}**")
        st.latex(steps[cur][2])
    math_note("الخطوتان 4 و5 هما **قاعدة السلسلة** `chain rule` — نفس القاعدة التي يطبقها **الانتشار العكسي** `backpropagation` عبر مئات الطبقات. "
              "الأطر (Keras، PyTorch) تحسب هذه المشتقات آليًا، لكن فهمها على معلمتين يجعلها غير سحرية.")

    code_lab(CodeLab(
        key="w01_gd", title_ar="حلقة التدريب كاملة في NumPy (نفس الأرقام التي في التحريك)", code=GD_CODE, level="A",
        before=Before(goal_ar="كتابة حلقة التدريب بخطواتها الخمس صراحة، والتحقق أن النزول بالتدرج يصل إلى حل المربعات الصغرى.",
                      stage_ar="الأسبوع 01: التدريب ثم الاستدلال.",
                      inputs_ar="40 طالبًا: `hours` (40,) و `score` (40,)، ومعدل التعلم η وعدد الحقب من أدوات التحكم.",
                      expected_ar="خسارة تنخفض من ≈ 4897 إلى ≈ 11.2 عند η = 0.02، و w ≈ 4.67، b ≈ 44.5؛ تنبؤ لطالب 7 ساعات ≈ 77.",
                      math_ar="ŷ = w x + b ؛ L = mean(e²) ؛ ∂L/∂w = 2·mean(e·x) ؛ ∂L/∂b = 2·mean(e)."),
        explain=[("2-4", "نفس بيانات الدرس: 40 طالبًا، الدرجة الحقيقية = 45 + 4.6×الساعات + ضجيج."),
                 ("6-7", "المعلمات تبدأ من الصفر؛ η متغير فائق `hyperparameter` نختاره نحن ولا يُتعلَّم."),
                 ("9-11", "تنبؤ، خطأ، خسارة: ثلاث عمليات متجهة على كل الطلاب دفعة واحدة."),
                 ("12-13", "التدرجان بالصيغتين اللتين اشتققناهما."),
                 ("14-15", "التحديث عكس التدرج."),
                 ("19-20", "`np.polyfit` يعطي الحل الدقيق للمقارنة، ثم استدلال بالمعلمات النهائية المجمّدة.")],
        controls=lambda: {"eta": st.select_slider("η", [0.001, 0.01, 0.02, 0.03], value=0.02, key="ctrl_w01_gd_eta"),
                          "epochs": st.select_slider("epochs", [10, 100, 600], value=600, key="ctrl_w01_gd_epochs")},
        defaults={"eta": 0.02, "epochs": 600},
        run=run_printed(GD_CODE, template=True), template=True,
        after_ar="- الخسارة المطبوعة في كل سطر هي خسارة **ما قبل** تحديث تلك الحقبة.\n- عند η = 0.03 سترى `inf`/`nan`: هذا ما يبدو عليه التباعد في مخرجات حقيقية.\n- تطابق w و b مع `polyfit` يثبت أن التدريب مجرد **بحث عددي** عن نفس الحل.",
    ))

    # ------------------------------------------------------------------ inference
    h2("الاستدلال: النموذج المجمّد يتنبأ لطالب جديد", "Inference with frozen parameters")
    tr_ok = gd_trace(0.02, 600)
    wf, bf = tr_ok["w"][-1], tr_ok["b"][-1]
    x_new = st.slider("ساعات مذاكرة طالب جديد", 0.5, 10.0, 7.0, 0.5, key="w01_xnew")
    iframes = [
        Frame(inference_svg(0, x_new, wf, bf), caption(f"طالب جديد **لم يكن** في بيانات التدريب ذاكر `{x_new:g}` ساعة. هذا هو المدخل الوحيد."), action="input"),
        Frame(inference_svg(1, x_new, wf, bf), caption(f"نضرب في الوزن المتعلَّم `w = {wf:.2f}` (مجمّد، لن يتغير): `{x_new:g} × {wf:.2f} = {wf * x_new:.2f}`."), action="× w"),
        Frame(inference_svg(2, x_new, wf, bf), caption(f"نضيف الانحياز المتعلَّم `b = {bf:.2f}`."), action="+ b"),
        Frame(inference_svg(3, x_new, wf, bf), caption(f"التنبؤ `ŷ = {wf * x_new + bf:.1f}`. لا خسارة (لا نعرف الدرجة الحقيقية بعد)، ولا تدرج، ولا تحديث."), action="ŷ",
              values=[("w", f"{wf:.3f}", f"{wf:.3f}"), ("b", f"{bf:.3f}", f"{bf:.3f}")]),
    ]
    animation_player(f"w01_infer_{x_new}", iframes, title_ar="تمرير أمامي واحد بمعلمات مجمّدة", interval_ms=1300)
    compare_table(
        ["", "التدريب", "الاستدلال"],
        [("المعلمات", "تتغير كل خطوة", "مجمّدة"), ("يحتاج الإجابة الصحيحة y؟", "نعم (لحساب الخسارة)", "لا"),
         ("العمليات", "تمرير أمامي + خسارة + تدرج + تحديث", "تمرير أمامي فقط"), ("التكلفة", "عالية، مرة واحدة", "منخفضة، ملايين المرات"),
         ("في Keras", "`model.fit(...)`", "`model.predict(...)`")],
        ["rtl", "rtl", "rtl"],
    )
    warning_note(f"الطالب الذي «ذاكر 20 ساعة» سيحصل على تنبؤ ≈ {wf * 20 + bf:.0f} — فوق 100! النموذج لم يرَ إلا 0.5–10 ساعات، والخط لا يعرف أن الدرجة محدودة. "
                 "التنبؤ خارج مدى بيانات التدريب `extrapolation` غير موثوق.")

    # ------------------------------------------------------------------ lifecycle
    h2("دورة حياة النموذج", "Model lifecycle")
    animation_player("w01_lifecycle", _lifecycle_frames(), title_ar="من المسألة إلى الاستدلال — ومتى تتغير المعلمات",
                     stages=["Define", "Data", "Design", "Train", "Evaluate", "Diagnose", "Inference"], interval_ms=2400)
    pipeline(["Define problem", "Prepare data", "Design architecture", "Train", "Evaluate", "Diagnose & fix", "Inference"], active=3)
    common_mistake("تقييم النموذج على بيانات التدريب نفسها وإعلان دقة 99%. هذا يقيس الحفظ لا التعلم. التقييم يكون على بيانات محجوزة.")
    common_mistake("الظن أن النموذج «يتعلم» أثناء الاستعمال. في الاستدلال المعلمات مجمّدة؛ ليتعلم من بيانات جديدة يجب إعادة التدريب صراحة.")
    research_note("في ورقة بحثية: صف البنية، وإجراء التدريب (المُحسِّن، η، الحقب، حجم الدفعة، البذرة)، وطريقة التقسيم، ومقياس التقييم على بيانات الاختبار، وخط الأساس الذي قارنت به.")
    with st.container(horizontal=True):
        st.button("معمل النزول بالتدرج", icon=":material/science:", on_click=go, args=("labs.gradient_descent_lab",), key="w01_lab_gd")
        st.button("معمل معدل التعلم", icon=":material/science:", on_click=go, args=("labs.learning_rate_lab",), key="w01_lab_lr")

    quiz(
        "w01.core",
        [
            Q("«MLP بثلاث طبقات مخفية من 128 وحدة مع ReLU» يصف…", ["نموذجًا مدرَّبًا", "بنية", "خوارزمية تحسين"], 1, "هذا تصميم بلا معلمات مدرَّبة."),
            Q("ما الذي يميز النموذج عن البنية؟", ["عدد الطبقات", "قيم المعلمات المدرَّبة", "نوع البيانات"], 1, "النموذج = بنية + معلمات."),
            Q("الانحدار الخطي يقع…", ["داخل DL", "داخل ML وخارج DL", "خارج AI"], 1, "يتعلم من البيانات (ML) لكنه ليس شبكة متعددة الطبقات."),
            Q("ما الذي يميز DL عن ML الكلاسيكي أساسًا؟", ["استعمال بيانات", "تعلّم الخصائص/التمثيل بالطبقات بدل صنعها يدويًا", "استعمال Python"], 1, "تعلّم التمثيل."),
            Q("الخسارة ارتفعت من حقبة إلى التالية ثم انفجرت. أرجح سبب؟", ["η صغير جدًا", "η كبير جدًا", "بيانات قليلة"], 1, "خطوة تتجاوز القاع."),
            Q("إشارة ∂L/∂w سالبة. قاعدة التحديث w ← w − η·∂L/∂w تجعل w…", ["تنقص", "تزيد", "لا تتغير"], 1, "طرح عدد سالب = زيادة."),
            Q("في أي مرحلة تتغير المعلمات؟", ["التقييم", "التدريب", "الاستدلال"], 1, "التدريب وحده."),
            Q("في أي مرحلة تُقاس القدرة على التعميم؟", ["التدريب", "التقييم على بيانات محجوزة", "تصميم البنية"], 1, "بيانات لم يرها النموذج."),
        ],
    )
    takeaway("AI ⊃ ML ⊃ DL. شبكة = طبقات متراكبة؛ بنية = تصميم؛ نموذج = بنية + معلمات مدرَّبة. التدريب = تنبؤ ← خسارة ← تدرج ← تحديث، مكررة. قيّم على بيانات محجوزة، واستدلّ بمعلمات مجمّدة.")
    lesson_footer(LESSON, [
        "DL فرع من ML يتعلم الخصائص بالطبقات بدل صنعها يدويًا.",
        "الطبقة: ضرب في أوزان + انحياز + تنشيط غير خطي.",
        "البنية قرار الباحث قبل التدريب؛ النموذج نتيجة التدريب.",
        "التدريب يحرّك المعلمات عكس التدرج بخطوة η؛ η الكبير يفجّر الخسارة.",
        "الاستدلال تمرير أمامي فقط بمعلمات مجمّدة؛ الاستقراء خارج مدى البيانات خطر.",
    ])
