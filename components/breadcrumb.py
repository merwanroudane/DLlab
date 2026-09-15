"""Breadcrumb trail (spec §8): always shows the current path, clickable."""

from __future__ import annotations

import streamlit as st

from core.registry import get_registry
from core.routing import Resolved, go


def _crumbs(resolved: Resolved) -> list[tuple[str, str]]:
    """Return [(label, route), ...] from root to current."""
    reg = get_registry()
    crumbs: list[tuple[str, str]] = [("الرئيسية", "home")]
    if resolved.kind == "static":
        labels = {
            "about": "حول المنصة", "glossary": "المصطلحات", "map": "خريطة المقرر",
            "search": "بحث", "progress": "تقدّمي",
        }
        if resolved.route != "home":
            crumbs.append((labels.get(resolved.route, resolved.route), resolved.route))
        return crumbs
    if resolved.kind == "section":
        crumbs.append((resolved.obj.title_ar, resolved.route))
        return crumbs
    if resolved.kind == "module":
        crumbs.append((reg.sections[resolved.obj.section].title_ar, resolved.obj.section))
        crumbs.append((resolved.obj.title_ar, resolved.route))
        return crumbs
    if resolved.kind == "lab":
        crumbs.append((reg.sections["labs"].title_ar, "labs"))
        crumbs.append((resolved.obj.title_ar, resolved.route))
        return crumbs
    if resolved.kind == "lesson":
        for pid in reg.lesson_path(resolved.route):
            crumbs.append((reg.title_of(pid), pid))
        return crumbs
    return crumbs


def breadcrumb(resolved: Resolved) -> None:
    crumbs = _crumbs(resolved)
    # Horizontal blocks are row-reverse (base.css): the first item sits at the far right,
    # so the trail reads home → ... → current from right to left.
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="center", key="breadcrumb_row"):
        for i, (label, route) in enumerate(crumbs):
            is_current = i == len(crumbs) - 1
            if is_current:
                st.markdown(f"**{label}**")
            else:
                st.button(label, key=f"bc_{route}", type="tertiary", on_click=go, args=(route,))
            if not is_current:
                st.markdown("<span style='color:#B9B2A6'>‹</span>", unsafe_allow_html=True)
