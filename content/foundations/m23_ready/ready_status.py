import streamlit as st

from components.callouts import intuition, practical_note, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from content.foundations.m23_ready._scoring import scores
from core.models import Lesson
from core.progress import module_progress, section_progress
from core.registry import get_registry
from core.routing import go
from core.rtl import table

LESSON = Lesson(
    id="foundations.ready.ready_status",
    title_ar="حالة الجاهزية",
    title_en="Ready Status",
    module="foundations.ready",
    order=4,
    prerequisites=["foundations.ready.challenges"],
    objectives_ar=["حساب حالة الجاهزية من ثلاثة مصادر: تغطية الوحدات، الاختبار التشخيصي، التحديات.", "قائمة عودة محددة بالوحدات والدروس.", "الانتقال إلى المقرر الرسمي (الأسبوع 01) عند الجاهزية."],
    terms=[],
    difficulty="beginner",
    summary_ar="الجاهزية = تغطية ≥ 80% من دروس الأسس + تشخيصي ≥ 75% + متوسط التحديات ≥ 70%. ما دون ذلك: قائمة عودة بالوحدات الأضعف.",
)

CHALLENGES = [("shape", "الأشكال", "foundations.architecture"), ("batch", "الحقب/الدفعات", "foundations.batch_epoch"), ("loss_optim", "الخسارة/المحسّن", "foundations.loss"), ("keras_order", "ترتيب Keras", "foundations.frameworks"), ("keras_fill", "فراغات Keras", "foundations.frameworks"), ("tf", "TensorFlow", "foundations.frameworks"), ("torch_loop", "حلقة PyTorch", "foundations.frameworks")]
CORE_MODULES = ["foundations.data", "foundations.python", "foundations.math", "foundations.linalg", "foundations.calculus", "foundations.prob", "foundations.ml", "foundations.prep", "foundations.neuron", "foundations.architecture", "foundations.activations", "foundations.forward", "foundations.loss", "foundations.backprop", "foundations.optim", "foundations.training_loop", "foundations.batch_epoch", "foundations.eval", "foundations.generalization", "foundations.regularization", "foundations.frameworks"]


def render() -> None:
    lesson_header(LESSON)
    why("«هل أنا جاهز؟» سؤال يُجاب بالأدلة: ماذا قرأت، ماذا أجبت، ماذا أنتجت. الصفحة تجمع الثلاثة وتعطي حكمًا وقائمة عودة — في هذه الجلسة (التقدم غير محفوظ بين الجلسات بحسب تصميم المنصة).")
    reg = get_registry(); sc = scores()
    # 1) coverage
    per_mod = [(mid, module_progress(mid)) for mid in CORE_MODULES]
    done = sum(p.done for _, p in per_mod); total = sum(p.total for _, p in per_mod)
    coverage = 100 * done / total if total else 0
    # 2) diagnostic
    diag = sc.get("diagnostic"); diag_pct = 100 * diag["correct"] / diag["total"] if diag else None
    # 3) challenges
    solved = [(k, ar, mid, sc[k]) for k, ar, mid in CHALLENGES if k in sc]
    ch_pct = (100 * sum(s["correct"] for *_, s in solved) / sum(s["total"] for *_, s in solved)) if solved else None
    h2("المؤشرات الثلاثة", "The three indicators")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("تغطية دروس الأسس", f"{coverage:.0f}%", help="نسبة دروس الوحدات 1–21 التي فتحتها في هذه الجلسة")
        st.progress(coverage / 100)
    with c2:
        st.metric("الاختبار التشخيصي", f"{diag_pct:.0f}%" if diag_pct is not None else "—", help="20 سؤالًا في صفحة فحص المتطلبات")
        st.progress((diag_pct or 0) / 100)
    with c3:
        st.metric("التحديات", f"{ch_pct:.0f}%" if ch_pct is not None else "—", help=f"{len(solved)} من {len(CHALLENGES)} تحديات محلولة")
        st.progress((ch_pct or 0) / 100)
    ok_cov, ok_diag, ok_ch = coverage >= 80, (diag_pct or 0) >= 75, (ch_pct or 0) >= 70 and len(solved) >= 5
    ready = ok_cov and ok_diag and ok_ch
    h2("الحكم", "Verdict")
    if ready:
        st.success("**جاهز للتعلم العميق.** غطّيت الأسس، اجتزت التشخيص، وأنتجت الحلول في التحديات. المقرر الرسمي يبدأ من الأسبوع 01.", icon="✅")
        st.button("ابدأ المقرر الرسمي — الأسبوع 01", type="primary", icon=":material/school:", key="ready_go_course", on_click=go, args=("course.w01",))
    else:
        missing = []
        if not ok_cov:
            missing.append(f"التغطية {coverage:.0f}% < 80%")
        if not ok_diag:
            missing.append("التشخيصي " + (f"{diag_pct:.0f}% < 75%" if diag_pct is not None else "لم يُحل"))
        if not ok_ch:
            missing.append("التحديات " + (f"{ch_pct:.0f}% < 70% أو أقل من 5 تحديات محلولة" if ch_pct is not None else "لم تُحل"))
        st.warning("**ليس بعد** — " + "؛ ".join(missing) + ".", icon="🟡")
    h2("قائمة العودة", "Return list")
    rows = []
    for mid, p in per_mod:
        weak = [ar for k, ar, cmid in CHALLENGES if cmid == mid and k in sc and sc[k]["correct"] < sc[k]["total"]]
        status = "✅" if p.percent == 100 and not weak else ("🟡" if p.done else "⬜")
        rows.append((status, reg.modules[mid].title_ar, f"{p.done}/{p.total}", "، ".join(weak) if weak else "—"))
    table(["", "الوحدة", "الدروس المفتوحة", "تحديات غير مكتملة"], rows, ["rtl", "rtl", "num", "rtl"])
    todo = [mid for mid, p in per_mod if p.percent < 100 or any(cmid == mid and k in sc and sc[k]["correct"] < sc[k]["total"] for k, _, cmid in CHALLENGES)]
    if todo:
        st.markdown("**ابدأ من هنا:**")
        cols = st.columns(3)
        for i, mid in enumerate(todo[:9]):
            with cols[i % 3]:
                st.button(reg.modules[mid].title_ar, key=f"ready_go_{mid}", type="tertiary", on_click=go, args=(mid,))
    if diag is None:
        st.button("حلّ الاختبار التشخيصي", key="ready_go_diag", on_click=go, args=("foundations.ready.prerequisite_check",), icon=":material/quiz:")
    if len(solved) < len(CHALLENGES):
        st.button(f"أكمل التحديات ({len(solved)}/{len(CHALLENGES)})", key="ready_go_ch", on_click=go, args=("foundations.ready.challenges",), icon=":material/fitness_center:")
    intuition("العتبات (80/75/70) ليست مقدسة؛ هي حدود «لا أخدع نفسي». من يجتازها يفهم المقرر الرسمي من أسبوعه الأول؛ من دونها سيعود إلى الأسس على أي حال — والعودة الآن أرخص.")
    practical_note("التقدم في هذه الجلسة فقط (لا حسابات بحسب تصميم المنصة). إن أردت أثرًا دائمًا: صوّر هذه الصفحة أو دوّن قائمة العودة.")
    labs = section_progress("labs")
    st.caption(f"المعامل المفتوحة: {labs.done}/{labs.total} — ليست جزءًا من الحكم لكنها أفضل استثمار قبل الأسبوع 03.")
    takeaway("جاهز = تغطية ≥ 80% + تشخيصي ≥ 75% + تحديات ≥ 70%. الحكم يأتي مع قائمة عودة محددة وزر البداية.")
    lesson_footer(LESSON, ["ثلاثة مؤشرات بأدلة.", "الحكم وقائمة العودة.", "الانتقال إلى الأسبوع 01."])
