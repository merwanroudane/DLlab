"""Session-state initialisation — the single place where per-user state keys
are declared. Pages and components read/write only through these helpers.

Keys
----
visited            set[str]   lesson & lab ids the user has opened
quiz               dict       {quiz_key: {"answers": {...}, "submitted": bool}}
anim               dict       {anim_key: {"step": int, "playing": bool}}
experiments        dict       {lab_key: {...settings...}}
last_lesson        str|None   most recent lesson id (for "Resume learning")
nav_open_modules   set[str]   modules the user expanded manually in the nav
"""

from __future__ import annotations

import streamlit as st


def init_state() -> None:
    st.session_state.setdefault("visited", set())
    st.session_state.setdefault("quiz", {})
    st.session_state.setdefault("anim", {})
    st.session_state.setdefault("experiments", {})
    st.session_state.setdefault("last_lesson", None)
    st.session_state.setdefault("nav_open_modules", set())


def experiment(key: str, defaults: dict) -> dict:
    """Return the mutable settings dict for a lab/experiment, creating it
    with `defaults` on first use."""
    store = st.session_state.experiments
    if key not in store:
        store[key] = dict(defaults)
    return store[key]
