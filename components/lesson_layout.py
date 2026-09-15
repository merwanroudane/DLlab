"""Shared lesson chrome: header (objectives / prerequisites / terms),
footer (summary / related / labs / previous-next), and landing pages for
modules and sections (spec §17, §59, §60)."""

from __future__ import annotations

import streamlit as st

from core.models import Lesson, Module, Section
from core.progress import is_visited, mark_visited, module_progress
from core.registry import get_registry
from core.routing import go
from core.rtl import esc

DIFFICULTY_AR = {"beginner": "مبتدئ", "intermediate": "متوسط", "advanced": "متقدم"}


def h2(title_ar: str, title_en: str | None = None) -> None:
    """Section heading inside a lesson: Arabic + isolated English."""
    if title_en:
        st.markdown(f"## {title_ar} <span class='en' style='font-size:.8em'>{esc(title_en)}</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"## {title_ar}")


def h3(title_ar: str, title_en: str | None = None) -> None:
    if title_en:
        st.markdown(f"### {title_ar} <span class='en' style='font-size:.8em'>{esc(title_en)}</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"### {title_ar}")


def lesson_header(lesson: Lesson) -> None:
    reg = get_registry()
    mark_visited(lesson.id)
    module = reg.modules[lesson.module]
    st.markdown(f"# {lesson.title_ar}")
    st.markdown(f"<div class='en' style='font-size:1.05rem;color:#6B675F'>{esc(lesson.title_en)}</div>",
                unsafe_allow_html=True)
    tags = [
        f"<span class='tag'>{esc(module.title_ar)}</span>",
        f"<span class='tag'>المستوى: {DIFFICULTY_AR[lesson.difficulty]}</span>",
    ]
    if lesson.children:
        tags.append(f"<span class='tag'>{len(lesson.children)} صفحات فرعية</span>")
    st.html(f"<div class='dlia-meta'>{''.join(tags)}</div>")

    cols = st.columns(2) if (lesson.objectives_ar and lesson.prerequisites) else [st.container(), st.container()]
    with cols[0]:
        if lesson.objectives_ar:
            with st.container(border=True):
                st.markdown("**🎯 أهداف التعلم**")
                st.markdown("\n".join(f"- {o}" for o in lesson.objectives_ar))
    with cols[1]:
        if lesson.prerequisites:
            with st.container(border=True):
                st.markdown("**📚 المتطلبات القبلية**")
                st.caption("يفضّل إنهاء هذه الدروس أولًا. العلامة ✅ تعني أنك فتحته سابقًا.")
                for pid in lesson.prerequisites:
                    pre = reg.lessons[pid]
                    mark = "✅" if is_visited(pid) else "⬜"
                    st.button(f"{mark} {pre.title_ar}", key=f"pre_{lesson.id}_{pid}", type="tertiary",
                              on_click=go, args=(pid,))
    if lesson.terms:
        chips = []
        for tid in lesson.terms:
            t = reg.terms[tid]
            chips.append(f"<span class='chip sem-data' title='{esc(t.definition_ar)}'>{esc(t.en)} · {esc(t.ar)}</span>")
        st.html("<div class='dlia-rtl' style='margin:.2rem 0 .6rem'><b>🔑 مصطلحات هذا الدرس:</b> "
                + " ".join(chips) + "</div>")
    if lesson.children:
        with st.container(border=True):
            st.markdown("**🗂️ الصفحات الفرعية لهذا الدرس**")
            with st.container(horizontal=True, gap="small"):
                for cid in lesson.children:
                    c = reg.lessons[cid]
                    st.button(c.title_ar, key=f"child_{cid}", on_click=go, args=(cid,))


def lesson_footer(lesson: Lesson, summary_points: list[str] | None = None) -> None:
    reg = get_registry()
    st.divider()
    if summary_points:
        with st.container(border=True):
            st.markdown("**🎯 الخلاصة والنقاط الأساسية**")
            st.markdown("\n".join(f"- {p}" for p in summary_points))
    if lesson.related or lesson.labs:
        c1, c2 = st.columns(2)
        with c1:
            if lesson.related:
                st.markdown("**🔗 مفاهيم مرتبطة**")
                for rid in lesson.related:
                    st.button(reg.lessons[rid].title_ar, key=f"rel_{lesson.id}_{rid}", type="tertiary",
                              on_click=go, args=(rid,), icon=":material/link:")
        with c2:
            if lesson.labs:
                st.markdown("**🧪 معامل مرتبطة**")
                for lid in lesson.labs:
                    st.button(reg.labs[lid].title_ar, key=f"lab_{lesson.id}_{lid}", type="tertiary",
                              on_click=go, args=(lid,), icon=":material/science:")
    prev_next(lesson)


def prev_next(lesson: Lesson) -> None:
    reg = get_registry()
    # First column is on the right: previous on the right, next on the left (RTL convention).
    c_prev, c_next = st.columns(2)
    with c_next:
        if lesson.next:
            nxt = reg.lessons[lesson.next]
            st.button(f"التالي: {nxt.title_ar}", key=f"next_{lesson.id}", type="primary",
                      icon=":material/arrow_back:", width="stretch", on_click=go, args=(nxt.id,))
    with c_prev:
        if lesson.previous:
            prv = reg.lessons[lesson.previous]
            st.button(f"السابق: {prv.title_ar}", key=f"prev_{lesson.id}", type="secondary",
                      icon=":material/arrow_forward:", width="stretch", on_click=go, args=(prv.id,))


def module_landing(module: Module) -> None:
    reg = get_registry()
    prog = module_progress(module.id)
    st.markdown(f"# {module.title_ar}")
    st.markdown(f"<div class='en' style='font-size:1.05rem;color:#6B675F'>{esc(module.title_en)}</div>",
                unsafe_allow_html=True)
    st.progress(prog.percent / 100, text=f"التقدم في الوحدة: {prog.done}/{prog.total} درسًا")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**🎯 الغرض من الوحدة**")
            st.markdown(module.purpose_ar or "—")
        if module.objectives_ar:
            with st.container(border=True):
                st.markdown("**📌 أهداف التعلم**")
                st.markdown("\n".join(f"- {o}" for o in module.objectives_ar))
    with c2:
        with st.container(border=True):
            st.markdown("**❓ لماذا تهم هذه الوحدة؟**")
            st.markdown(module.why_ar or "—")
        if module.prerequisites:
            with st.container(border=True):
                st.markdown("**📚 وحدات يُفضّل إنهاؤها أولًا**")
                for mid in module.prerequisites:
                    m = reg.modules[mid]
                    st.button(m.title_ar, key=f"mpre_{module.id}_{mid}", type="tertiary", on_click=go, args=(mid,))
        if module.challenges_ar:
            with st.container(border=True):
                st.markdown("**🧗 تحديات شائعة في هذه الوحدة**")
                st.markdown("\n".join(f"- {c}" for c in module.challenges_ar))

    st.markdown("## الدروس")
    for lesson in reg.top_lessons_of(module.id):
        mark = "✅" if is_visited(lesson.id) else "⬜"
        with st.container(border=True):
            st.button(f"{mark} {lesson.title_ar}", key=f"ml_{lesson.id}", type="tertiary",
                      on_click=go, args=(lesson.id,))
            if lesson.summary_ar:
                st.caption(lesson.summary_ar)
            if lesson.children:
                with st.container(horizontal=True, gap="small"):
                    for cid in lesson.children:
                        c = reg.lessons[cid]
                        st.button(c.title_ar, key=f"mlc_{cid}", type="tertiary", on_click=go, args=(cid,))


def section_landing(section: Section) -> None:
    reg = get_registry()
    st.markdown(f"# {section.title_ar}")
    st.markdown(f"<div class='en' style='font-size:1.05rem;color:#6B675F'>{esc(section.title_en)}</div>",
                unsafe_allow_html=True)
    if section.description_ar:
        st.markdown(section.description_ar)
    for module in reg.modules_of(section.id):
        prog = module_progress(module.id)
        with st.container(border=True):
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            with c1:
                st.button(f"{module.title_ar}", key=f"sl_{module.id}", type="tertiary",
                          icon=module.icon, on_click=go, args=(module.id,))
                if module.purpose_ar:
                    st.caption(module.purpose_ar)
            with c2:
                st.progress(prog.percent / 100, text=f"{prog.done}/{prog.total}")
