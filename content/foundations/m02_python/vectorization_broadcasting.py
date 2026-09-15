import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.python.vectorization_broadcasting",
    title_ar="التوجيه المتجهي والبث",
    title_en="Vectorization & Broadcasting",
    module="foundations.python",
    order=7,
    prerequisites=["foundations.python.numpy_ndarray"],
    objectives_ar=[
        "استبدال حلقات بايثون بعمليات على المصفوفة كاملة (التوجيه المتجهي).",
        "تطبيق قواعد البث الثلاث للتنبؤ بشكل النتيجة أو بالخطأ قبل التشغيل.",
        "رؤية أن `z = Wx + b` و`X - mean` كلها بث.",
    ],
    terms=["shape", "axis", "batch_dimension"],
    difficulty="intermediate",
    summary_ar="التوجيه المتجهي يلغي الحلقات؛ البث يمدّد الأشكال المتوافقة تلقائيًا وفق قواعد محددة.",
)

CODE = '''import numpy as np, time

X = np.random.default_rng(0).normal(size=(200_000, 4))

t0 = time.perf_counter()
means_loop = [sum(X[i, j] for i in range(X.shape[0])) / X.shape[0] for j in range(4)]
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
means_vec = X.mean(axis=0)
t_vec = time.perf_counter() - t0
print(f"loop: {t_loop*1000:.1f} ms   vectorized: {t_vec*1000:.2f} ms   speedup ≈ {t_loop / max(t_vec, 1e-9):.0f}x")

# --- البث ---
A = np.array([[1, 2, 3],
              [4, 5, 6]])            # (2, 3)
b = np.array([10, 20, 30])           # (3,)   → يُمدَّد إلى (2, 3)
print(A + b)

col = np.array([[100], [200]])       # (2, 1) → يُمدَّد عبر الأعمدة
print(A + col)

print((A - A.mean(axis=0)).round(2)) # توحيد الخصائص: (2,3) - (3,)

W = np.ones((3, 2)); x = np.array([1.0, 2.0, 3.0]); bias = np.array([0.5, -0.5])
print(x @ W + bias)                  # z = xW + b : (3,)@(3,2) → (2,) + (2,)

try:
    print(A + np.array([1, 2]))      # (2,3) + (2,) → لا تتوافق
except ValueError as e:
    print("ValueError:", e)'''


def render() -> None:
    lesson_header(LESSON)
    h2("التوجيه المتجهي", "Vectorization")
    definition("**التوجيه المتجهي** كتابة العملية على المصفوفة كلها بدل حلقة على عناصرها. الحساب ينتقل من بايثون البطيء إلى كود مُجمَّع سريع؛ وعلى `GPU` يصبح متوازيًا فعلًا.")
    why("الشبكة العصبية تحسب `z = XW + b` لدفعة كاملة في عملية واحدة. لو كُتبت بحلقات لاستغرقت أيامًا. فهم التوجيه المتجهي هو فهم لماذا أطر العمل تفكر بالموترات لا بالأرقام.")
    h2("البث", "Broadcasting")
    definition("**البث** آلية تسمح بعملية بين مصفوفتين مختلفتي الشكل بتمديد المحور الذي طوله 1 (أو المفقود) ليطابق الآخر — دون نسخ فعلي للبيانات.")
    intuition("إضافة انحياز `b` بالشكل `(3,)` إلى دفعة `(2, 3)`: بدل أن تكرر `b` صفين يدويًا، يعامل البث `b` كأنه مكرر لكل صف.")
    h3("القواعد الثلاث", "The three rules")
    st.markdown("""
1. **محاذاة من اليمين**: قارن الأشكال من آخر محور إلى أوله. `(2, 3)` مع `(3,)` → قارن 3 مع 3.
2. **التوافق**: محوران متوافقان إذا تساويا أو كان أحدهما 1. المحور المفقود يُعامل كـ 1.
3. **النتيجة**: كل محور يأخذ الطول الأكبر. `(2, 3)` مع `(2, 1)` → `(2, 3)`.
""")
    table(
        ["الشكل A", "الشكل B", "النتيجة", "لماذا"],
        [("(2, 3)", "(3,)", "(2, 3)", "3=3، وB بلا محور أول → 1 يُمدَّد إلى 2"),
         ("(2, 3)", "(2, 1)", "(2, 3)", "1 يُمدَّد إلى 3"),
         ("(4, 1)", "(1, 5)", "(4, 5)", "كلاهما يُمدَّد: جدول ضرب خارجي"),
         ("(32, 1)", "(32,)", "(32, 32)", "خطير: يُمدَّد الاثنان بصمت (خطأ الشكل الشهير)"),
         ("(2, 3)", "(2,)", "خطأ", "3 ≠ 2 ولا أحدهما 1")],
        ["code", "code", "code", "rtl"],
    )
    code_lab(CodeLab(
        key="py_broadcast", title_ar="قياس السرعة + أمثلة البث", code=CODE,
        before=Before(goal_ar="قياس فرق السرعة بين حلقة وعملية متجهية، ثم رؤية البث ينجح ويفشل.", stage_ar="أساسيات NumPy.",
                      inputs_ar="مصفوفة عشوائية `(200000, 4)` ومصفوفات صغيرة.", expected_ar="تسريع بعشرات إلى مئات المرات، نتائج بث، ورسالة `ValueError` واحدة متعمدة.",
                      math_ar="$z = xW + b$ مع $x \\in \\mathbb{R}^{3}$، $W \\in \\mathbb{R}^{3\\times 2}$، $b \\in \\mathbb{R}^{2}$."),
        explain=[("3-11", "نفس المتوسط بطريقتين. `perf_counter` ساعة دقيقة. النسبة تختلف بين الأجهزة لكنها دائمًا كبيرة."),
                 ("14-17", "القاعدة 1 و2: `(3,)` يُمدَّد إلى `(2, 3)` فتُضاف `b` إلى كل صف."),
                 ("19-20", "`(2, 1)` يُمدَّد عبر الأعمدة: كل صف يضاف إليه رقمه."),
                 ("22", "توحيد الخصائص: طرح متوسط كل عمود من كل الصفوف — عملية التحجيم الأساسية."),
                 ("24-25", "طبقة كثيفة لملاحظة واحدة: `@` ضرب المصفوفات ثم بث الانحياز."),
                 ("27-30", "`(2,3)` مع `(2,)`: من اليمين 3 مقابل 2 — لا تساوي ولا 1 → خطأ واضح.")],
        run=run_printed(CODE),
        after_ar="- الرسالة `operands could not be broadcast together with shapes (2,3) (2,)` تقول بالضبط الشكلين المتنازعين.\n- لإصلاحها اجعل المتجه `(2, 1)` بـ `reshape(-1, 1)` إن كان المقصود صفًا لكل ملاحظة.",
    ))
    common_mistake("الاعتماد على البث «ليُصلح» الأشكال: `(32, 1)` مع `(32,)` **ينجح** ويعطي `(32, 32)` — لا خطأ، لكن الخسارة خاطئة. تحقق من الأشكال قبل الطرح.")
    quiz("py.broadcast", [
        Q("`(5, 4) + (4,)` يعطي…", ["(5, 4)", "(4,)", "خطأ"], 0, "يُمدَّد المتجه عبر الصفوف.", kind="shape"),
        Q("`(5, 4) + (5,)` يعطي…", ["(5, 4)", "(5, 5)", "خطأ"], 2, "4 مقابل 5 من اليمين.", kind="shape"),
        Q("`(32, 1) - (32,)` يعطي…", ["(32,)", "(32, 32)", "خطأ"], 1, "الأخطر: بث صامت.", kind="shape"),
    ])
    takeaway("اكتب على المصفوفة كلها. البث: محاذاة من اليمين، تساوٍ أو 1، النتيجة الأكبر. (32,1) مع (32,) فخ.")
    lesson_footer(LESSON, ["الحلقات بطيئة؛ العمليات المتجهية سريعة ومتوازية.", "قواعد البث الثلاث تتنبأ بالشكل أو بالخطأ.", "z = xW + b بث."])
