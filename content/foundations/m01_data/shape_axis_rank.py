import numpy as np
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab
from components.comparison import compare_table
from components.diagnostics import Problem, problem_card
from components.diagram import diagram, svg_box, svg_text
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.data.shape_axis_rank",
    title_ar="الشكل والبُعد والمحور والرتبة",
    title_en="Shape, Dimension, Axis & Rank",
    module="foundations.data",
    order=9,
    prerequisites=["foundations.data.dtype", "foundations.data.observation"],
    objectives_ar=[
        "قراءة `shape` كقائمة بأطوال المحاور.",
        "التمييز بين الرتبة (عدد المحاور) والبُعد (طول محور) — ولماذا كلمة Dimension ملتبسة.",
        "فهم المحور 0 (الملاحظات/الدفعة) والمحور 1 (الخصائص) وعمليات التجميع على محور.",
        "قراءة خطأ `Shapes are incompatible` وإصلاحه.",
    ],
    terms=["shape", "axis", "rank", "dimension", "tensor", "batch_dimension"],
    labs=["labs.dataset_anatomy"],
    difficulty="beginner",
    summary_ar="shape = أطوال المحاور، rank = عددها. المحور 0 للملاحظات. معظم أخطاء التشغيل الأولى أخطاء شكل.",
)

CODE = '''import numpy as np

X = np.array([[4200, 0, 34],
              [6100, 2, 45],
              [2900, 5, 29],
              [8800, 0, 52]], dtype=np.float32)

print("shape:", X.shape)        # (4, 3)
print("rank :", X.ndim)         # 2  ← عدد المحاور
print("size :", X.size)         # 12 ← عدد العناصر الكلي

print("mean over axis 0:", X.mean(axis=0))   # متوسط كل خاصية عبر الملاحظات → (3,)
print("mean over axis 1:", X.mean(axis=1))   # متوسط كل ملاحظة عبر خصائصها  → (4,)

one = X[0]                      # (3,)
one_batch = X[0:1]              # (1, 3)
flat = X.reshape(-1)            # (12,)
back = flat.reshape(4, 3)       # (4, 3)
print(one.shape, one_batch.shape, flat.shape, back.shape)'''


def _shape_svg() -> str:
    s = '<svg viewBox="0 0 560 220" width="100%" style="max-width:560px">'
    cells = [[4200, 0, 34], [6100, 2, 45], [2900, 5, 29], [8800, 0, 52]]
    x0, y0, w, h = 150, 40, 90, 34
    for i, row in enumerate(cells):
        for j, v in enumerate(row):
            fill = "#E6F1FB" if j == 0 else ("#DDF5EA" if j == 1 else "#EFE9FA")
            s += svg_box(x0 + j * w, y0 + i * h, w, h, str(v), fill, font=13, rx=4)
    s += svg_text(x0 + 1.5 * w, 28, "axis 1 → features (d = 3)", size=13, bold=True, color="#2E8B57")
    s += svg_text(95, y0 + 2 * h, "axis 0 ↓", size=13, bold=True, color="#2F6FB5")
    s += svg_text(95, y0 + 2 * h + 18, "observations", size=12, color="#2F6FB5")
    s += svg_text(95, y0 + 2 * h + 34, "(n = 4)", size=12, color="#2F6FB5")
    s += svg_text(x0 + 1.5 * w, 205, "X.shape = (4, 3)   X.ndim = 2   X.size = 12", size=13, mono=True)
    return s + "</svg>"


def _run(p: dict) -> None:
    X = np.array([[4200, 0, 34], [6100, 2, 45], [2900, 5, 29], [8800, 0, 52]], dtype=np.float32)
    st.code(
        f"shape: {X.shape}\nrank : {X.ndim}\nsize : {X.size}\n"
        f"mean over axis 0: {X.mean(axis=0)}\nmean over axis 1: {X.mean(axis=1)}\n"
        f"{X[0].shape} {X[0:1].shape} {X.reshape(-1).shape} {X.reshape(-1).reshape(4, 3).shape}",
        language="text",
    )


def render() -> None:
    lesson_header(LESSON)
    h2("ما هي؟", "What are they?")
    definition(
        "**الشكل** `Shape` هو قائمة بطول كل **محور** `Axis` في المصفوفة: `(4, 3)` تعني محورين، الأول طوله 4 والثاني 3. "
        "**الرتبة** `Rank` هي عدد المحاور (`ndim`). **البُعد** `Dimension` كلمة ملتبسة: تُستخدم أحيانًا بمعنى الرتبة "
        "(«مصفوفة ثنائية البعد») وأحيانًا بمعنى طول محور («بُعد المدخل 3»). في المنصة نقول **رتبة** لعدد المحاور و**طول المحور** للأطوال."
    )
    why("خطأ الشكل هو أول خطأ يقابله كل باحث. رسالة مثل `Shapes (32, 1) and (32,) are incompatible` تصبح مفهومة تمامًا بعد هذا الدرس.")
    diagram(
        "محورا مصفوفة المدخلات",
        _shape_svg(),
        what_ar="مصفوفة X بأربع ملاحظات (صفوف) وثلاث خصائص (أعمدة). المحور 0 يمتد عبر الصفوف، والمحور 1 عبر الأعمدة.",
        how_ar="الرقم الأول في shape يعدّ على المحور 0 (كم صفًا)، والثاني على المحور 1 (كم عمودًا). الرتبة = عدد الأرقام في shape.",
        takeaway_ar="المحور 0 في التعلم العميق هو دائمًا محور الملاحظات (أو الدفعة). إذا اختلط عليك الأمر فابدأ منه.",
        legend=[("#E6F1FB", "income"), ("#DDF5EA", "late payments"), ("#EFE9FA", "age")],
        title_en="Axes of X",
    )
    h2("الرتب الشائعة", "Common ranks")
    compare_table(
        ["الرتبة", "الاسم", "مثال shape", "مثال بيانات"],
        [
            ("0", "Scalar (عدد)", "()", "الخسارة loss = 0.37"),
            ("1", "Vector (متجه)", "(3,)", "ملاحظة واحدة، أو الهدف y"),
            ("2", "Matrix (مصفوفة)", "(4, 3)", "بيانات جدولية X"),
            ("3", "Tensor رتبة 3", "(n, T, f)", "سلاسل زمنية / نص"),
            ("4", "Tensor رتبة 4", "(n, H, W, C)", "صور"),
        ],
        ["num", "ltr", "code", "rtl"],
    )
    intuition(
        "الشكل يجيب عن سؤال «كم؟» لكل محور. الموتر رتبة 4 للصور: كم صورة؟ كم صف بكسل؟ كم عمود بكسل؟ كم قناة لون؟ "
        "أربعة أسئلة = أربعة أرقام."
    )
    code_lab(CodeLab(
        key="shape_lab",
        title_ar="shape و ndim و axis و reshape",
        level="A",
        code=CODE,
        before=Before(
            goal_ar="قراءة الشكل والرتبة، والتجميع على محور، وإعادة التشكيل دون فقدان بيانات.",
            stage_ar="البيانات ← الفحص والتحضير.",
            inputs_ar="مصفوفة `X` بالشكل `(4, 3)`.",
            expected_ar="أشكال متوقعة: `(3,)` للمتوسط على المحور 0، `(4,)` على المحور 1، `(1, 3)` لدفعة من ملاحظة واحدة.",
            math_ar="المتوسط على المحور 0: $\\bar{x}_j = \\frac{1}{n}\\sum_i X_{ij}$ لكل خاصية $j$.",
        ),
        explain=[
            ("3-6", "أربعة صفوف (ملاحظات) × ثلاثة أعمدة (خصائص). `float32` منذ البداية."),
            ("8-10", "`shape` أطوال المحاور، `ndim` عددها، `size` حاصل ضربها = عدد العناصر."),
            ("12", "`axis=0` يعني «اجمع/اطوِ المحور 0»: يختفي محور الملاحظات ويبقى `(3,)` — متوسط كل خاصية. هذا ما يفعله التحجيم."),
            ("13", "`axis=1` يطوي محور الخصائص ويبقى `(4,)` — رقم واحد لكل ملاحظة."),
            ("15-16", "`X[0]` يُسقط المحور 0 ← `(3,)`. `X[0:1]` يبقيه ← `(1, 3)`. الشبكة تريد الثاني."),
            ("17-18", "`reshape(-1)` يفرد كل العناصر في متجه؛ `-1` تعني «احسب هذا الطول بنفسك». العودة إلى `(4, 3)` ممكنة لأن 12 = 4 × 3."),
        ],
        run=_run,
        after_ar=(
            "- قاعدة `axis`: المحور الذي تذكره هو المحور الذي **يختفي** من النتيجة.\n"
            "- `reshape` لا يغيّر عدد العناصر أبدًا؛ `reshape(5, 3)` على 12 عنصرًا يعطي خطأ."
        ),
    ))
    h3("قراءة خطأ شكل حقيقي", "Reading a real shape error")
    problem_card(Problem(
        key="shape_mismatch",
        name_ar="عدم توافق الأشكال",
        name_en="Shape mismatch",
        description_ar="الخطأ الأول الذي يقابله كل باحث: شكل مخرج النموذج لا يطابق شكل الهدف، أو شكل المدخل لا يطابق ما تتوقعه أول طبقة.",
        symptoms_ar=["استثناء فوري عند أول خطوة تدريب أو عند `compile/fit`.", "أو — أسوأ — تدريب «ينجح» بخسارة غريبة الحجم بسبب بث صامت."],
        sees_ar=["رسالة `ValueError: Shapes (...) are incompatible` في `Keras`.",
                 "رسالة `RuntimeError: ... size mismatch` أو `The size of tensor a (32) must match the size of tensor b (1)` في `PyTorch`.",
                 "خسارة أكبر بكثير من المتوقع رغم عدم وجود استثناء."],
        error_text="ValueError: Shapes (32, 1) and (32,) are incompatible",
        possible_causes_ar=["الهدف `y` متجه `(n,)` بينما مخرج النموذج `(n, 1)`.", "ملاحظة واحدة مُرِّرت بلا محور دفعة `(d,)` بدل `(1, d)`.",
                            "ترتيب محاور صورة خاطئ `(C, H, W)` مقابل `(H, W, C)`.", "هدف بأرقام فئات `(n,)` مع خسارة تتوقع one-hot `(n, k)`."],
        root_causes_ar=["عدم طباعة الأشكال بعد كل تحويل.", "الخلط بين الرتبة وطول المحور."],
        diagnosis_ar=["اطبع `X.shape` و`y.shape` قبل النموذج.", "اطبع شكل مخرج النموذج على دفعة واحدة: `model(X[:2]).shape`.",
                      "قارن رتبة المخرج برتبة الهدف، ثم أطوال المحاور واحدًا واحدًا.", "حدد التحويل الذي غيّر الشكل بصورة غير متوقعة."],
        evidence_ar=["رتبتان مختلفتان (2 مقابل 1) أو طولا محور مختلفان في الرسالة نفسها.", "إصلاح شكل واحد يزيل الخطأ ويجعل قيمة الخسارة معقولة."],
        fixes_ar=["وحّد شكل الهدف مع المخرج: `y = y.reshape(-1, 1)` أو اعكس ذلك.", "أبقِ محور الدفعة: `X[0:1]` بدل `X[0]`.",
                  "اختر الخسارة المناسبة لشكل الهدف (`sparse_categorical_crossentropy` لأرقام الفئات)."],
        tradeoffs_ar=["إعادة التشكيل الصامتة في كل مكان تخفي الأخطاء؛ الأفضل تثبيت اتفاق واحد للأشكال في المشروع."],
        misdiagnosis_ar=["«النموذج لا يتعلم» بينما السبب بث صامت `(32, 1) − (32,)` إلى `(32, 32)`."],
        related_ar=["خطأ `dtype`", "ترتيب محاور الصور بين `Keras` و`PyTorch`"],
        checklist_ar=["طبعت `X.shape` و`y.shape`؟", "رتبة المخرج = رتبة الهدف؟", "أطوال المحاور متطابقة؟", "الخسارة تناسب شكل الهدف؟"],
        math_ar="مع البث، `(32, 1) − (32,)` يُفسَّر كمصفوفة عمود ناقص متجه صف فينتج `(32, 32)`: كل تنبؤ يُقارَن بكل هدف. الخسارة الناتجة رقم بلا معنى لكنه قابل للاشتقاق، فيستمر التدريب بصمت.",
        wrong_code=(
            "# (32, 1) - (32,) → broadcasting to (32, 32)!\n"
            "loss = ((y_pred - y) ** 2).mean()"
        ),
        correct_code=(
            "y = y.reshape(-1, 1)   # (32,) → (32, 1)\n"
            "assert y_pred.shape == y.shape\n"
            "loss = ((y_pred - y) ** 2).mean()"
        ),
        challenge=[
            Q("مخرج `(64, 3)` وهدف `(64,)` بأرقام فئات 0..2 مع `categorical_crossentropy`. ما المشكلة؟",
              ["عدد الفئات خاطئ", "الخسارة تتوقع one-hot (64, 3) بينما الهدف أرقام", "حجم الدفعة صغير"], 1,
              "استخدم sparse_categorical_crossentropy أو حوّل الهدف إلى one-hot.", kind="error"),
            Q("`model.predict(X[7])` يعطي خطأ شكل بينما `model.predict(X[7:8])` يعمل. لماذا؟",
              ["الملاحظة 7 تالفة", "X[7] بلا محور دفعة", "predict لا يقبل الفهارس"], 1, "الشبكة تتوقع (1, d).", kind="error"),
        ],
    ))
    debugging_note(
        "أسلوب التشخيص: اطبع `X.shape` بعد كل تحويل. عندما يتغير الشكل بصورة غير متوقعة تكون قد وجدت السطر المذنب. "
        "الطباعة أرخص بكثير من التخمين."
    )
    common_mistake("«البُعد» بمعنيين. عندما يقول أحدهم «بُعد المدخل 784» فالمقصود طول المحور، لا أن الموتر له 784 محورًا.")
    quiz(
        "data.shape",
        [
            Q("`X.shape == (16, 28, 28, 3)`. ما الرتبة؟", ["3", "4", "16"], 1, "أربعة أرقام = أربعة محاور.", kind="shape"),
            Q("`X.mean(axis=0)` لمصفوفة `(100, 5)` يعطي شكل…", ["(100,)", "(5,)", "()"], 1, "المحور 0 يختفي.", kind="shape"),
            Q("`np.zeros((3, 4)).reshape(6, 2)` — هل ينجح؟", ["نعم، 12 = 12", "لا، الشكل مختلف", "نعم لكن يفقد بيانات"], 0,
              "reshape يحافظ على عدد العناصر.", kind="code"),
            Q("لماذا `(32, 1)` و`(32,)` غير متوافقين؟", ["أعداد مختلفة", "رتب مختلفة", "نوع مختلف"], 1, "رتبة 2 مقابل 1.", kind="error"),
        ],
    )
    takeaway("shape = أطوال المحاور، rank = عددها، المحور 0 للملاحظات. axis المذكور يختفي عند التجميع. اطبع الأشكال دائمًا.")
    lesson_footer(LESSON, [
        "shape قائمة أطوال المحاور؛ ndim عددها؛ size حاصل ضربها.",
        "المحور 0 = الملاحظات/الدفعة. المحور 1 = الخصائص في الجداول.",
        "التجميع على axis يُسقط ذلك المحور.",
        "(32, 1) ≠ (32,)؛ وحّد الشكل قبل الخسارة.",
    ])
