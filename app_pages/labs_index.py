import streamlit as st

from core.progress import is_visited
from core.registry import get_registry
from core.routing import go

CATEGORY_AR = {
    "data": "بيانات", "math": "رياضيات", "training": "تدريب", "cnn": "الشبكات الالتفافية",
    "sequence": "التسلسل: RNN / LSTM / GRU", "frameworks": "أطر العمل", "evaluation": "تقييم",
}


def render() -> None:
    reg = get_registry()
    sec = reg.sections["labs"]
    st.markdown(f"# {sec.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{sec.title_en}</div>", unsafe_allow_html=True)
    st.markdown(sec.description_ar)
    for cat, labs in reg.labs_by_category().items():
        st.markdown(f"## {CATEGORY_AR.get(cat, cat)}")
        cols = st.columns(2)
        for i, lab in enumerate(labs):
            with cols[i % 2]:
                with st.container(border=True, height="stretch"):
                    mark = "✅ " if is_visited(lab.id) else ""
                    st.markdown(f"**{mark}{lab.title_ar}**")
                    st.caption(lab.title_en)
                    st.markdown(lab.description_ar)
                    st.button("افتح المعمل", key=f"labs_open_{lab.id}", icon=":material/science:", type="primary",
                              on_click=go, args=(lab.id,))
                    if lab.related_lessons:
                        with st.container(horizontal=True, gap="small"):
                            for r in lab.related_lessons:
                                st.button(reg.lessons[r].title_ar, key=f"labs_rel_{lab.id}_{r}", type="tertiary",
                                          on_click=go, args=(r,))
