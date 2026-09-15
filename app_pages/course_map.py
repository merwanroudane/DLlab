"""Course map + knowledge graph (spec §50): a dependency graph built from
lesson prerequisites, plus a concept explorer (prerequisites / you are
here / related / next)."""

import streamlit as st

from core.progress import is_visited
from core.registry import get_registry
from core.routing import go
from core.rtl import esc

AXES = [
    ("أساسيات التعلم الآلي", ["course.w01", "course.w02"]),
    ("الشبكات العصبية الأمامية", ["course.w03", "course.w04", "course.w07"]),
    ("خوارزميات التحسين", ["course.w05", "course.w06"]),
    ("الشبكات العصبية الالتفافية CNN", ["course.w08", "course.w09"]),
    ("الشبكات العصبية التكرارية RNN", ["course.w10"]),
    ("شبكات LSTM", ["course.w11"]),
    ("شبكات GRU", ["course.w12", "course.w13", "course.w14", "course.w15"]),
]


def _dot_for_module(module_id: str) -> str:
    reg = get_registry()
    lessons = [reg.lessons[l] for l in reg.modules[module_id].lessons]
    ids = {l.id for l in lessons}
    lines = ["digraph G {", 'rankdir="LR"; bgcolor="transparent"; node [shape=box, style="rounded,filled", '
             'fontname="Inter", fontsize=11, color="#EADFCD"]; edge [color="#B9B2A6"];']
    for l in lessons:
        fill = "#DDF5EA" if is_visited(l.id) else "#FFFFFF"
        lines.append(f'"{l.id}" [label="{l.title_en}", fillcolor="{fill}"];')
    for l in lessons:
        for p in l.prerequisites:
            if p in ids:
                lines.append(f'"{p}" -> "{l.id}";')
            else:
                ext = reg.lessons[p]
                lines.append(f'"{p}" [label="{ext.title_en}", fillcolor="#FBF4E8", style="rounded,filled,dashed"];')
                lines.append(f'"{p}" -> "{l.id}";')
    lines.append("}")
    return "\n".join(lines)


def _ancestors(lesson_id: str, depth: int = 2) -> list[list[str]]:
    reg = get_registry()
    levels, frontier, seen = [], [lesson_id], {lesson_id}
    for _ in range(depth):
        nxt = []
        for lid in frontier:
            for p in reg.lessons[lid].prerequisites:
                if p not in seen:
                    seen.add(p); nxt.append(p)
        if not nxt:
            break
        levels.append(nxt); frontier = nxt
    return levels


def render() -> None:
    reg = get_registry()
    st.markdown("# خريطة المقرر وشبكة المعرفة")
    st.markdown("<div class='en' style='color:#6B675F'>Course Map & Knowledge Graph</div>", unsafe_allow_html=True)

    st.markdown("## المحاور الرسمية السبعة ← الأسابيع")
    for name, weeks in AXES:
        with st.container(border=True):
            c1, c2 = st.columns([1, 2], vertical_alignment="center")
            c1.markdown(f"**{name}**")
            with c2:
                with st.container(horizontal=True, gap="small"):
                    for w in weeks:
                        if w in reg.modules:
                            st.button(reg.modules[w].title_ar, key=f"map_{w}", type="tertiary", on_click=go, args=(w,))
                        else:
                            st.markdown(f"<span class='tag' style='color:#B9B2A6'>{esc(w.split('.')[1].upper())} — قيد البناء</span>",
                                        unsafe_allow_html=True)

    st.markdown("## مستكشف المفاهيم")
    st.markdown("اختر مفهومًا لترى ما يجب معرفته قبله (المتطلبات)، وأين أنت، وما يرتبط به، وما يليه.")
    options = reg.ordered_lessons
    default = st.session_state.get("last_lesson") or options[0]
    pick = st.selectbox("المفهوم", options, index=options.index(default) if default in options else 0,
                        format_func=lambda i: f"{reg.lessons[i].title_ar} — {reg.lessons[i].title_en}", key="map_pick")
    lesson = reg.lessons[pick]
    levels = _ancestors(pick)
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True, height="stretch"):
            st.markdown("**📚 المتطلبات (الأقرب أولًا)**")
            if not levels:
                st.caption("لا متطلبات — نقطة بداية.")
            for depth, lvl in enumerate(levels, start=1):
                for p in lvl:
                    mark = "✅" if is_visited(p) else "⬜"
                    st.button(f"{mark} {reg.lessons[p].title_ar}", key=f"anc_{depth}_{p}", type="tertiary", on_click=go, args=(p,))
    with c2:
        with st.container(border=True, height="stretch"):
            st.markdown("**📍 أنت هنا**")
            st.markdown(f"**{lesson.title_ar}**")
            st.caption(lesson.title_en)
            if lesson.summary_ar:
                st.markdown(lesson.summary_ar)
            st.button("افتح الدرس", type="primary", key="map_open", on_click=go, args=(pick,))
            if lesson.related:
                st.markdown("**🔗 مرتبط:**")
                for r in lesson.related:
                    st.button(reg.lessons[r].title_ar, key=f"rel_{r}", type="tertiary", on_click=go, args=(r,))
    with c3:
        with st.container(border=True, height="stretch"):
            st.markdown("**➡️ ما يليه**")
            dependents = [l for l in reg.lessons.values() if pick in l.prerequisites]
            if lesson.next:
                st.button(f"التالي في المسار: {reg.lessons[lesson.next].title_ar}", key="map_next", type="secondary",
                          on_click=go, args=(lesson.next,))
            for d in dependents:
                st.button(f"يعتمد عليه: {d.title_ar}", key=f"dep_{d.id}", type="tertiary", on_click=go, args=(d.id,))
            if not dependents and not lesson.next:
                st.caption("نهاية المسار الحالي.")

    st.markdown("## شبكة الاعتماديات داخل وحدة")
    mods = [m for m in reg.modules.values()]
    mod = st.selectbox("الوحدة", [m.id for m in sorted(mods, key=lambda m: (m.section, m.order))],
                       format_func=lambda i: reg.modules[i].title_ar, key="map_module",
                       index=[m.id for m in sorted(mods, key=lambda m: (m.section, m.order))].index(lesson.module))
    st.graphviz_chart(_dot_for_module(mod), width="stretch")
    st.caption("الأخضر = زرته في هذه الجلسة. المتقطع = متطلب من وحدة أخرى. الأسهم تقرأ: «يلزم قبل».")
