import streamlit as st

from core.progress import course_week_position, section_progress
from core.registry import get_registry
from core.routing import go


def render() -> None:
    reg = get_registry()
    st.html(
        '<div class="dlia-hero">'
        "<h1>أكاديمية التعلم العميق التفاعلية</h1>"
        '<p class="subtitle"><span class="en">Deep Learning Interactive Academy — From Foundations to Deep Neural Networks</span></p>'
        "<p class=\"subtitle\">منصة تعليمية تفاعلية لتعلّم التعلم العميق من الأسس إلى الشبكات العصبية العميقة: "
        "بالعربية، بالرياضيات، بالكود، وبالتشخيص.</p>"
        '<p class="author">إعداد وتصميم أكاديمي: الدكتور مروان رودان · <span class="en">Dr. Marwan Roudane</span></p>'
        "</div>"
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        with st.container(border=True):
            st.markdown("**🎯 هدف المقرر**")
            st.markdown(
                "تمكين الطالب والباحث من فهم وتطبيق خوارزميات التعلم العميق باستخدام `Keras` (مع `TensorFlow`، "
                "وتوجيه مقابل في `PyTorch`)، وبناء نماذج ذكية تدعم اتخاذ القرار في المجالات الاقتصادية والإدارية "
                "أو مجال تخصصه من خلال بيانات تخصصية."
            )
            st.markdown("**👥 الجمهور**: باحثون وأساتذة وطلبة قد لا يملكون خلفية كافية في الرياضيات أو بايثون أو التعلم الآلي.")
    with c2:
        with st.container(border=True):
            st.markdown("**📈 تقدّمك في هذه الجلسة**")
            f = section_progress("foundations"); c = section_progress("course"); l = section_progress("labs")
            wk, wtot = course_week_position()
            st.progress(f.percent / 100, text=f"أكاديمية الأسس: {f.percent}% ({f.done}/{f.total} درسًا)")
            st.progress(c.percent / 100, text=f"المقرر الرسمي: الأسبوع {wk:02d}/{wtot:02d} · {c.done}/{c.total} درسًا")
            st.progress((l.done / l.total) if l.total else 0, text=f"المعامل: {l.done}/{l.total}")
            last = st.session_state.get("last_lesson")
            if last and last in reg.lessons:
                st.button(f"متابعة التعلم: {reg.lessons[last].title_ar}", type="primary", icon=":material/resume:",
                          on_click=go, args=(last,), key="home_resume", width="stretch")
            else:
                st.button("ابدأ من الأسس", type="primary", icon=":material/flag:", on_click=go,
                          args=("foundations.start.how_to_use",), key="home_start", width="stretch")

    st.markdown("### مسار التعلم")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True, height="stretch"):
            st.markdown("**1️⃣ أكاديمية الأسس**")
            st.markdown("بيانات، بايثون، رياضيات، جبر خطي، تفاضل، احتمال، تعلم آلي، الخلية والشبكة، التدريب والتشخيص، أطر العمل.")
            st.button("افتح الأسس", icon=":material/foundation:", on_click=go, args=("foundations",), key="home_f")
    with c2:
        with st.container(border=True, height="stretch"):
            st.markdown("**2️⃣ المقرر الرسمي — 15 أسبوعًا**")
            st.markdown("من مقدمة التعلم العميق إلى CNN وRNN وLSTM وGRU وGPU/Colab والمشروع التطبيقي.")
            st.button("افتح المقرر", icon=":material/school:", on_click=go, args=("course",), key="home_c")
    with c3:
        with st.container(border=True, height="stretch"):
            st.markdown("**3️⃣ المعامل التفاعلية**")
            st.markdown("محاكي الحقبة والدفعة، تشريح البيانات، وسلسلة معامل الرياضيات والتدريب وCNN وRNN.")
            st.button("افتح المعامل", icon=":material/science:", on_click=go, args=("labs",), key="home_l")

    st.markdown("### وصول سريع")
    with st.container(horizontal=True, gap="small"):
        for lab in sorted(reg.labs.values(), key=lambda x: x.order)[:6]:
            st.button(lab.title_ar, icon=":material/science:", on_click=go, args=(lab.id,), key=f"home_lab_{lab.id}")
        st.button("المصطلحات", icon=":material/dictionary:", on_click=go, args=("glossary",), key="home_g")
        st.button("خريطة المقرر", icon=":material/account_tree:", on_click=go, args=("map",), key="home_m")

    st.markdown("### الفلسفة")
    st.markdown(
        "> **Learn → See → Manipulate → Calculate → Code → Run → Break → Diagnose → Fix → Compare → Understand → Practice**\n\n"
        "لا يظهر مصطلح قبل شرحه. كل مفهوم يمر بالحدس والصورة والمعادلة والمثال الرقمي والكود والتجربة والفشل والتشخيص والإصلاح."
    )
