"""Progress tracking (session-scoped, no authentication by design — see spec §52)."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from core.registry import Registry, get_registry


@dataclass
class ProgressStat:
    done: int
    total: int

    @property
    def percent(self) -> int:
        return 0 if self.total == 0 else round(100 * self.done / self.total)


def mark_visited(item_id: str) -> None:
    st.session_state.visited.add(item_id)
    if item_id in get_registry().lessons:
        st.session_state.last_lesson = item_id


def is_visited(item_id: str) -> bool:
    return item_id in st.session_state.get("visited", set())


def module_progress(module_id: str, registry: Registry | None = None) -> ProgressStat:
    reg = registry or get_registry()
    lessons = reg.modules[module_id].lessons
    visited = st.session_state.get("visited", set())
    return ProgressStat(sum(1 for l in lessons if l in visited), len(lessons))


def section_progress(section_id: str, registry: Registry | None = None) -> ProgressStat:
    reg = registry or get_registry()
    if section_id == "labs":
        ids = list(reg.labs)
    else:
        ids = [l.id for l in reg.lessons.values() if l.section == section_id]
    visited = st.session_state.get("visited", set())
    return ProgressStat(sum(1 for i in ids if i in visited), len(ids))


def course_week_position(registry: Registry | None = None) -> tuple[int, int]:
    """(current week number reached, total weeks) based on visited course lessons."""
    reg = registry or get_registry()
    weeks = reg.modules_of("course")
    visited = st.session_state.get("visited", set())
    reached = 0
    for i, week in enumerate(weeks, start=1):
        if any(l in visited for l in week.lessons):
            reached = i
    return reached, len(weeks)
