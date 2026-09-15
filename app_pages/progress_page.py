import streamlit as st

from core.progress import course_week_position, module_progress, section_progress
from core.registry import get_registry
from core.routing import go


def render() -> None:
    reg = get_registry()
    st.markdown("# تقدّمي")
    st.markdown("<div class='en' style='color:#6B675F'>Progress</div>", unsafe_allow_html=True)
    st.caption("التقدم محفوظ في هذه الجلسة فقط (لا تسجيل دخول بحسب تصميم المنصة).")
    f = section_progress("foundations"); c = section_progress("course"); l = section_progress("labs")
    wk, wtot = course_week_position()
    st.progress(f.percent / 100, text=f"أكاديمية الأسس: {f.percent}%")
    st.progress(c.percent / 100, text=f"المقرر الرسمي: الأسبوع {wk:02d}/{wtot:02d}")
    st.progress((l.done / l.total) if l.total else 0, text=f"المعامل: {l.done}/{l.total}")
    for sid in ("foundations", "course"):
        st.markdown(f"## {reg.sections[sid].title_ar}")
        for m in reg.modules_of(sid):
            p = module_progress(m.id)
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            with c1:
                st.button(m.title_ar, key=f"prog_{m.id}", type="tertiary", on_click=go, args=(m.id,))
            with c2:
                st.progress(p.percent / 100, text=f"{p.done}/{p.total}")
    if st.button("إعادة ضبط التقدم", icon=":material/restart_alt:", key="prog_reset"):
        st.session_state.visited = set()
        st.session_state.last_lesson = None
        st.rerun()
