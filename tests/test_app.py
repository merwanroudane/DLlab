"""Headless smoke tests of the real app through every route (spec §72:
"the app opens" is not enough — every page must render without exception)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "streamlit_app.py")


def run_route(route: str, timeout: float = 120) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.query_params["p"] = route
    at.run()
    assert not at.exception, f"{route}: {at.exception}"
    return at


def all_routes():
    from core.registry import get_registry

    reg = get_registry()
    routes = ["home", "about", "glossary", "map", "progress"]
    routes += list(reg.sections)
    routes += list(reg.modules)
    routes += list(reg.lessons)
    routes += list(reg.labs)
    return routes


@pytest.mark.parametrize("route", all_routes())
def test_every_route_renders(route):
    at = run_route(route)
    # Identity must be present on every page (footer / brand block) — spec §4
    html_blobs = " ".join(str(getattr(e, "value", "")) for e in at.get("html"))
    assert "Marwan Roudane" in html_blobs
    # No placeholder content anywhere (spec §75)
    all_text = html_blobs + " ".join(m.value for m in at.markdown)
    for banned in ("TODO", "Coming soon", "Lorem ipsum", "Placeholder"):
        assert banned not in all_text, f"{route} contains placeholder text: {banned}"


def test_unknown_route_shows_error_not_crash():
    at = run_route("does.not.exist")
    assert at.error, "unknown route should show an error box"


def test_right_navigation_highlights_current_lesson():
    from core.registry import get_registry

    lid = get_registry().ordered_lessons[0]
    at = run_route(lid)
    active = [b for b in at.button if b.key == f"nav_{lid}"]
    assert active and active[0].proto.type == "primary"


def test_navigation_button_changes_route():
    at = run_route("home")
    at.button(key="home_f").click().run()
    assert not at.exception
    p = at.query_params.get("p")
    assert (p[0] if isinstance(p, list) else p) == "foundations"


def test_visiting_a_lesson_marks_progress():
    from core.registry import get_registry

    lid = get_registry().ordered_lessons[1]
    at = run_route(lid)
    assert lid in at.session_state["visited"]
    assert at.session_state["last_lesson"] == lid


def test_quiz_scores_answers():
    at = run_route("foundations.start.how_to_use")
    key = "start.how_to_use"
    radios = [r for r in at.radio if r.key and r.key.startswith(f"quiz_{key}_")]
    assert len(radios) == 3
    for r in radios:
        r.set_value(1)  # correct answer index for all three questions in this quiz
    submit = [b for b in at.button if b.key and "quiz_form" in str(getattr(b, "proto", "")) or b.label == "تحقق من الإجابات"]
    assert submit
    submit[0].click().run()
    assert not at.exception
    assert any("3 / 3" in m.value for m in at.markdown)


def test_epoch_simulator_next_button_advances():
    at = run_route("labs.epoch_batch_simulator")
    assert at.session_state["ebs_slider"] == 1
    at.button(key="ebs_next").click().run()
    assert not at.exception
    assert at.session_state["ebs_slider"] == 2


def test_every_code_lab_runs_without_traceback(monkeypatch):
    """Force every Code Lab into its Run view and execute the authored code."""
    from core.registry import get_registry

    monkeypatch.setenv("DLIA_FORCE_RUN_CODELABS", "1")
    reg = get_registry()
    failures = []
    for lid in reg.lessons:
        at = AppTest.from_file(APP, default_timeout=150)
        at.query_params["p"] = lid
        try:
            at.run()
        except RuntimeError as exc:  # AppTest timeout: report it with the route instead of aborting the loop
            failures.append((lid, f"timeout: {exc}"[:200]))
            continue
        if at.exception:
            failures.append((lid, str(at.exception)[:200]))
            continue
        for block in at.code:
            if "Traceback (most recent call last)" in block.value:
                failures.append((lid, block.value[-300:]))
    assert not failures, "\n".join(f"{l}: {e}" for l, e in failures)
