"""Routing: one query parameter (`p`) is the single source of truth for the
current page, so every page is linkable and the browser back button works.

Route kinds
-----------
"home" | "about" | "glossary" | "map" | "search" | "progress"   static pages
"<section>"                                                    section landing
"<section>.<module>"                                           module landing
"<section>.<module>.<lesson>[.<sub>]"                          lesson page
"labs.<lab>"                                                   lab page
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import streamlit as st

from core.registry import Registry, get_registry

PARAM = "p"
STATIC_ROUTES = {"home", "about", "glossary", "map", "search", "progress"}

RouteKind = Literal["static", "section", "module", "lesson", "lab", "unknown"]


@dataclass
class Resolved:
    kind: RouteKind
    route: str
    obj: Any = None


def current_route() -> str:
    value = st.query_params.get(PARAM, "home")
    return value or "home"


def resolve(route: str | None = None, registry: Registry | None = None) -> Resolved:
    reg = registry or get_registry()
    route = route or current_route()
    if route in STATIC_ROUTES:
        return Resolved("static", route)
    if route in reg.lessons:
        return Resolved("lesson", route, reg.lessons[route])
    if route in reg.labs:
        return Resolved("lab", route, reg.labs[route])
    if route in reg.modules:
        return Resolved("module", route, reg.modules[route])
    if route in reg.sections:
        return Resolved("section", route, reg.sections[route])
    return Resolved("unknown", route)


def go(route: str) -> None:
    """Navigate. Safe to use as an `on_click` callback or inline."""
    st.query_params[PARAM] = route


def nav_button(label: str, route: str, *, key: str, active: bool = False,
               icon: str | None = None, width: str = "stretch", help: str | None = None) -> None:
    """A navigation button that routes on click."""
    st.button(
        label,
        key=key,
        icon=icon,
        type="primary" if active else "tertiary",
        width=width,
        on_click=go,
        args=(route,),
        help=help,
    )


def link_button(label: str, route: str, *, key: str, icon: str | None = None,
                kind: str = "secondary") -> None:
    st.button(label, key=key, icon=icon, type=kind, on_click=go, args=(route,))
