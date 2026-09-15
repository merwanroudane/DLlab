"""Custom right-side navigation panel (spec §6).

Hierarchical and collapsible: Section → Module → Lesson → Sub-page.
Only the current module is expanded by default, so the panel never shows
hundreds of links at once. The same tree is reused inside a top "drawer"
popover for narrow screens.
"""

from __future__ import annotations

import streamlit as st

from core.progress import module_progress, section_progress
from core.registry import get_registry
from core.routing import Resolved, go

SECTION_ICONS = {
    "home": ":material/home:",
    "foundations": ":material/foundation:",
    "course": ":material/school:",
    "labs": ":material/science:",
    "glossary": ":material/dictionary:",
    "map": ":material/account_tree:",
    "about": ":material/info:",
}
SECTION_LABELS = {
    "home": "الرئيسية",
    "foundations": "أكاديمية الأسس",
    "course": "المقرر الرسمي — 15 أسبوعًا",
    "labs": "المعامل التفاعلية",
    "glossary": "المصطلحات والبحث",
    "map": "خريطة المقرر",
    "about": "حول المنصة",
}
SECTION_ORDER = ["home", "foundations", "course", "labs", "glossary", "map", "about"]


def _active_section(resolved: Resolved) -> str:
    if resolved.kind == "static":
        return resolved.route
    if resolved.kind == "section":
        return resolved.route
    if resolved.kind == "module":
        return resolved.obj.section
    if resolved.kind == "lesson":
        return resolved.obj.section
    if resolved.kind == "lab":
        return "labs"
    return "home"


def _btn(label: str, route: str, *, active: bool, key_prefix: str, icon: str | None = None,
         level: int = 0) -> None:
    prefix = "" if level == 0 else ("　" * level)   # ideographic space = clean indent in RTL
    st.button(
        prefix + label,
        key=f"{key_prefix}_{route}",
        icon=icon,
        type="primary" if active else "tertiary",
        width="stretch",
        on_click=go,
        args=(route,),
    )


def _module_tree(section_id: str, resolved: Resolved, key_prefix: str) -> None:
    reg = get_registry()
    current_lesson = resolved.obj if resolved.kind == "lesson" else None
    current_module = (
        resolved.route if resolved.kind == "module"
        else current_lesson.module if current_lesson else None
    )
    for module in reg.modules_of(section_id):
        prog = module_progress(module.id)
        label = f"{module.title_ar}  ·  {prog.done}/{prog.total}"
        expanded = module.id == current_module or module.id in st.session_state.nav_open_modules
        with st.expander(label, expanded=expanded, icon=module.icon):
            _btn("صفحة الوحدة", module.id, active=resolved.route == module.id,
                 key_prefix=key_prefix, icon=":material/dashboard:")
            for lesson in reg.top_lessons_of(module.id):
                is_here = current_lesson is not None and (
                    current_lesson.id == lesson.id or current_lesson.parent == lesson.id
                )
                _btn(lesson.title_ar, lesson.id, active=current_lesson is not None and current_lesson.id == lesson.id,
                     key_prefix=key_prefix, level=0)
                if is_here and lesson.children:
                    for child_id in lesson.children:
                        child = reg.lessons[child_id]
                        _btn(child.title_ar, child.id, active=current_lesson.id == child.id,
                             key_prefix=key_prefix, level=1)


def _labs_tree(resolved: Resolved, key_prefix: str) -> None:
    reg = get_registry()
    cat_labels = {
        "data": "بيانات", "math": "رياضيات", "training": "تدريب", "cnn": "CNN",
        "sequence": "تسلسل / RNN", "frameworks": "أطر العمل", "evaluation": "تقييم",
    }
    current_cat = resolved.obj.category if resolved.kind == "lab" else None
    for cat, labs in reg.labs_by_category().items():
        with st.expander(cat_labels.get(cat, cat), expanded=(cat == current_cat)):
            for lab in labs:
                _btn(lab.title_ar, lab.id, active=resolved.route == lab.id, key_prefix=key_prefix)


def nav_tree(resolved: Resolved, key_prefix: str = "nav") -> None:
    """The navigation tree body (shared by the right panel and the mobile drawer)."""
    reg = get_registry()
    active = _active_section(resolved)
    for sid in SECTION_ORDER:
        _btn(SECTION_LABELS[sid], sid, active=(sid == active and sid not in ("foundations", "course", "labs")),
             key_prefix=key_prefix, icon=SECTION_ICONS[sid])
        if sid == active and sid in ("foundations", "course"):
            _module_tree(sid, resolved, key_prefix)
        elif sid == active and sid == "labs":
            _labs_tree(resolved, key_prefix)


def brand_block() -> None:
    st.html(
        '<div class="dlia-brand">'
        '<div class="t">أكاديمية التعلم العميق التفاعلية</div>'
        '<div class="s"><span class="en">Deep Learning Interactive Academy</span></div>'
        '<div class="s">الدكتور مروان رودان · <span class="en">Dr. Marwan Roudane</span></div>'
        "</div>"
    )


def progress_block() -> None:
    f = section_progress("foundations")
    c = section_progress("course")
    l = section_progress("labs")
    st.html(
        '<div class="dlia-card" style="padding:.6rem .8rem;margin-top:.4rem">'
        f'<div class="muted">أكاديمية الأسس: <b>{f.percent}%</b> ({f.done}/{f.total})</div>'
        f'<div class="muted">المقرر الرسمي: <b>{c.percent}%</b> ({c.done}/{c.total})</div>'
        f'<div class="muted">المعامل: <b>{l.done}/{l.total}</b></div>'
        "</div>"
    )


def right_navigation(resolved: Resolved) -> None:
    """Render the full right-hand panel inside the current container."""
    with st.container(key="right_nav"):
        brand_block()
        nav_tree(resolved, key_prefix="nav")
        progress_block()


def mobile_drawer(resolved: Resolved) -> None:
    """Compact menu shown above the content for narrow screens (and as a
    quick jump on desktop)."""
    with st.popover("القائمة", icon=":material/menu:", width="content"):
        nav_tree(resolved, key_prefix="drawer")
