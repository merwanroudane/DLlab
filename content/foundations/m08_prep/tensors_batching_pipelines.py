import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.prep.tensors_batching_pipelines",
    title_ar="إلى موترات ودفعات وخطوط أنابيب؛ هندسة الخصائص وتعلّم التمثيل",
    title_en="To Tensors, Batches & Pipelines; Feature Engineering vs Representation Learning",
    module="foundations.prep",
    order=7,
    prerequisites=["foundations.prep.leakage", "foundations.prep.imbalance_augmentation", "foundations.linalg.tensors"],
    objectives_ar=[
        "تجميع كل خطوات الإعداد في خط أنابيب واحد يُدرَّب على التدريب ويُطبَّق على الباقي.",
        "تحويل الناتج إلى موترات float32 بالشكل الصحيح وتقسيمه إلى دفعات.",
        "التمييز بين هندسة الخصائص اليدوية وتعلّم التمثيل داخل الشبكة.",
    ],
    terms=["tensor", "batch", "batch_size", "dtype", "feature"],
    labs=["labs.epoch_batch_simulator", "labs.dataset_anatomy"],
    difficulty="intermediate",
    summary_ar="خط أنابيب: تنظيف → ترميز → تحجيم → موترات float32 → دفعات. fit على التدريب فقط. الشبكة تتعلم التمثيلات بعد ذلك.",
)

CODE = '''import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n = 12
df = pd.DataFrame({"income": rng.normal(5000, 1500, n).round(0), "city": rng.choice(["A", "B", "C"], n),
                   "late": rng.poisson(1, n), "defaulted": rng.integers(0, 2, n)})
df.loc[3, "income"] = np.nan

class Pipeline:
    """تنظيف → ترميز → تحجيم، كل معلمة تُتعلم في fit وتُستخدم في transform."""
    def fit(self, df):
        self.median_income = df["income"].median()
        self.cities = sorted(df["city"].unique())
        num = self._numeric(df)
        self.mu, self.sigma = num.mean(axis=0), num.std(axis=0) + 1e-9
        return self
    def _numeric(self, df):
        inc = df["income"].fillna(self.median_income)
        return np.column_stack([inc, df["late"]]).astype(np.float32)
    def transform(self, df):
        num = (self._numeric(df) - self.mu) / self.sigma
        onehot = np.column_stack([(df["city"] == c).astype(np.float32) for c in self.cities])
        return np.column_stack([num, onehot]).astype(np.float32)

tr, te = df.iloc[:9], df.iloc[9:]
pipe = Pipeline().fit(tr)
X_tr, X_te = pipe.transform(tr), pipe.transform(te)
y_tr = tr["defaulted"].to_numpy(np.float32); y_te = te["defaulted"].to_numpy(np.float32)
print("X_tr", X_tr.shape, X_tr.dtype, " X_te", X_te.shape, " y_tr", y_tr.shape)
print("first row:", X_tr[0].round(3), " (2 scaled numeric + 3 one-hot)")

batch_size = 4
for i, s in enumerate(range(0, len(X_tr), batch_size), start=1):
    xb, yb = X_tr[s:s+batch_size], y_tr[s:s+batch_size]
    print(f"batch {i}: X{xb.shape} y{yb.shape}")'''


def render() -> None:
    lesson_header(LESSON)
    h2("الصورة الكاملة", "The full pipeline")
    pipeline(["Raw table", "Clean", "Encode", "Scale", "float32 tensors", "Batches", "Model"], active=4)
    definition("**خط الأنابيب** `Pipeline` كائن واحد يجمع كل خطوات الإعداد: يُدرَّب (`fit`) على التدريب فقط ويُطبَّق (`transform`) على التحقق والاختبار والإنتاج بنفس المعلمات. يمنع التسريب ويجعل الإنتاج مطابقًا للتدريب.")
    code_lab(CodeLab(
        key="prep_pipe", title_ar="خط أنابيب من الصفر ينتج موترات ودفعات", code=CODE,
        before=Before(goal_ar="بناء خط أنابيب بـ fit/transform يعالج المفقودات ويرمّز ويحجّم، ثم إنتاج `X` و`y` بـ float32 وتقسيمهما إلى دفعات.", stage_ar="إعداد البيانات ← التمثيل النهائي.",
                      inputs_ar="جدول 12 صفًا بعمود مفقود وعمود نصي.", expected_ar="`X_tr (9, 5) float32`، `X_te (3, 5)`، ثم 3 دفعات: 4، 4، 1 (جزئية)."),
        explain=[("8-16", "`fit` يتعلم أربع معلمات: الوسيط للتعويض، قائمة المدن، μ وσ — كلها من التدريب."),
                 ("17-22", "`transform` يطبقها بلا تعلم جديد. نفس الدالة للتدريب والاختبار والإنتاج."),
                 ("24-28", "الشكل النهائي: 2 عددي موحد + 3 one-hot = 5 مدخلات. `float32` جاهز للإطار."),
                 ("31-34", "الدفعات بالتقطيع؛ الأخيرة جزئية. هذا ما يفعله `DataLoader` أو `tf.data` تلقائيًا.")],
        run=run_printed(CODE),
        after_ar="- `input_shape=(5,)` لأول طبقة يُقرأ مباشرة من `X_tr.shape[1]`.\n- كل ما في `fit` معلومة من التدريب؛ لو حسبنا الوسيط على `df` كاملًا لكان تسريبًا صغيرًا.",
    ))
    h2("هندسة الخصائص أم تعلّم التمثيل؟", "Feature engineering vs representation learning")
    st.markdown("""
- **هندسة الخصائص** `Feature engineering`: الباحث يصنع خصائص مشتقة (نسبة الدين إلى الدخل، فروق زمنية، لوغاريتمات). ضرورية في ML التقليدي، ومفيدة في التعلم العميق للجداول الصغيرة.
- **تعلّم التمثيل** `Representation learning`: الطبقات المخفية تتعلم تركيبات لاخطية من الخصائص الخام بنفسها. قوة التعلم العميق في الصور والنص والتسلسلات.
- عمليًا للجداول: جهّز الخصائص الخام الصحيحة (تنظيف، ترميز، تحجيم) وأضف ما تعرفه من المجال (نسب ذات معنى اقتصادي)؛ اترك التفاعلات المعقدة للشبكة.
""")
    practical_note("قاعدة: **ما تعرفه من المجال ضعه كخاصية؛ ما لا تعرفه اتركه للشبكة.** نسبة الدين إلى الدخل خاصية بديهية للاقتصادي؛ الشبكة قد تتعلمها لكن ببيانات أكثر.")
    common_mistake("بناء خط الأنابيب في دفتر تفاعلي بخلايا متفرقة ثم نسيان خطوة عند الانتقال إلى الإنتاج: الشبكة تستقبل خصائص بمقياس مختلف وتتنبأ هراءً بلا خطأ. الكائن الواحد يمنع ذلك.")
    quiz("prep.pipe", [
        Q("`Pipeline.fit` يُستدعى على…", ["كل البيانات", "التدريب فقط", "كل مجموعة على حدة"], 1, "مبدأ واحد للكل."),
        Q("2 عددي + 4 one-hot + 1 ترتيبي: `X.shape[1]`…", ["3", "7", "6"], 1, "المجموع.", kind="shape"),
        Q("تعلّم التمثيل يعني…", ["الباحث يصنع الخصائص", "الطبقات تتعلم تركيبات من الخام", "التحجيم"], 1, "قوة التعلم العميق."),
    ])
    takeaway("خط أنابيب واحد: fit على التدريب، transform على الجميع، مخرج float32 بشكل معروف، ثم دفعات. الشبكة تتولى التمثيلات.")
    lesson_footer(LESSON, ["تنظيف → ترميز → تحجيم → موتر → دفعة.", "input_shape = X.shape[1].", "المجال في الخصائص، التفاعلات للشبكة."])
