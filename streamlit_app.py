"""Deep Learning Interactive Academy — entry point.

Layout: [ main content | right navigation ]  (spec §6: navigation on the right)
Routing: one query param `p` (see core/routing.py).
"""

import streamlit as st

st.set_page_config(
    page_title="أكاديمية التعلم العميق التفاعلية — Dr. Marwan Roudane",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Deep Learning Interactive Academy — From Foundations to Deep Neural Networks. "
                 "Designed for Teaching and Research by Dr. Marwan Roudane (الدكتور مروان رودان).",
    },
)

from app_pages import about, course_map, glossary, home, labs_index, progress_page  # noqa: E402
from components.breadcrumb import breadcrumb  # noqa: E402
from components.lesson_layout import module_landing, section_landing  # noqa: E402
from components.right_navigation import mobile_drawer, right_navigation  # noqa: E402
from core.registry import get_registry  # noqa: E402
from core.routing import go, resolve  # noqa: E402
from core.state import init_state  # noqa: E402
from core.theme import inject_css  # noqa: E402

init_state()
inject_css()
registry = get_registry()
resolved = resolve()

STATIC_PAGES = {
    "home": home.render,
    "about": about.render,
    "glossary": glossary.render,
    "map": course_map.render,
    "progress": progress_page.render,
    "search": glossary.render,
}

# First column is the RIGHTMOST one (see base.css: horizontal blocks are row-reverse).
nav_col, content_col = st.columns([1, 3.4], gap="large")

with nav_col:
    with st.container(key="right_nav_panel"):
        right_navigation(resolved)

with content_col:
    crumb_col, drawer_col = st.columns([6, 1], vertical_alignment="center")
    with crumb_col:
        breadcrumb(resolved)
    with drawer_col:
        mobile_drawer(resolved)

    if resolved.kind == "static":
        STATIC_PAGES[resolved.route]()
    elif resolved.kind == "section":
        if resolved.route == "labs":
            labs_index.render()
        else:
            section_landing(resolved.obj)
    elif resolved.kind == "module":
        module_landing(resolved.obj)
    elif resolved.kind == "lesson":
        resolved.obj.render()
    elif resolved.kind == "lab":
        resolved.obj.render()
    else:
        st.error(f"الصفحة غير موجودة: `{resolved.route}`", icon="🔎")
        st.button("العودة إلى الرئيسية", on_click=go, args=("home",), key="notfound_home")

    st.html(
        '<div class="dlia-footer">أكاديمية التعلم العميق التفاعلية · '
        '<span class="en">Deep Learning Interactive Academy</span> · '
        'إعداد وتصميم أكاديمي: الدكتور مروان رودان · <span class="en">Dr. Marwan Roudane</span></div>'
    )
