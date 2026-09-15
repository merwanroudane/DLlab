"""Content registry, routing registry and prerequisite-graph integrity (spec §72, §73)."""

import re

import pytest

from core.registry import Registry

ID_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")


def test_registry_validates_clean(registry: Registry):
    issues = registry.validate()
    assert issues == [], "\n".join(str(i) for i in issues)


def test_every_section_has_modules_or_labs(registry: Registry):
    for sid, section in registry.sections.items():
        if sid == "labs":
            assert registry.labs, "labs section without labs"
        else:
            assert registry.modules_of(sid), f"section {sid} has no modules"


def test_ids_follow_convention(registry: Registry):
    for lid, lesson in registry.lessons.items():
        assert ID_RE.match(lid), lid
        assert lid.startswith(lesson.module + "."), lid
        assert lesson.section in registry.sections
    for mid, module in registry.modules.items():
        assert mid.startswith(module.section + "."), mid
    for lab_id in registry.labs:
        assert lab_id.startswith("labs."), lab_id


def test_reading_order_covers_every_lesson_once(registry: Registry):
    assert sorted(registry.ordered_lessons) == sorted(registry.lessons)
    assert len(set(registry.ordered_lessons)) == len(registry.ordered_lessons)


def test_prev_next_chain_is_consistent(registry: Registry):
    order = registry.ordered_lessons
    for i, lid in enumerate(order):
        lesson = registry.lessons[lid]
        assert lesson.previous == (order[i - 1] if i > 0 else None)
        assert lesson.next == (order[i + 1] if i + 1 < len(order) else None)
    # walking `next` from the first lesson visits everything
    seen, cur = [], order[0]
    while cur:
        seen.append(cur)
        cur = registry.lessons[cur].next
    assert seen == order


def test_sub_pages_follow_parent(registry: Registry):
    order = registry.ordered_lessons
    for lid, lesson in registry.lessons.items():
        if lesson.parent:
            assert order.index(lesson.parent) < order.index(lid)


def test_prerequisites_point_backwards_or_across_sections(registry: Registry):
    """A lesson's prerequisite must come earlier in the reading order unless it
    lives in another section (course weeks legitimately depend on foundations)."""
    order = registry.ordered_lessons
    for lid, lesson in registry.lessons.items():
        for pre in lesson.prerequisites:
            if registry.lessons[pre].section == lesson.section:
                assert order.index(pre) < order.index(lid), f"{lid} depends on later lesson {pre}"


def test_prerequisite_graph_is_acyclic(registry: Registry):
    assert registry._find_prereq_cycle() is None


def test_glossary_terms_unique_and_linked(registry: Registry):
    en = [t.en.strip().lower() for t in registry.terms.values()]
    ar = [t.ar.strip() for t in registry.terms.values()]
    assert len(en) == len(set(en)), "duplicate English glossary terms"
    assert len(ar) == len(set(ar)), "duplicate Arabic glossary terms"
    for t in registry.terms.values():
        assert t.definition_ar and t.category
        if t.lesson:
            assert t.lesson in registry.lessons


def test_lesson_metadata_complete(registry: Registry):
    for lesson in registry.lessons.values():
        assert lesson.title_ar and lesson.title_en, lesson.id
        assert callable(lesson.render), lesson.id
        assert lesson.difficulty in ("beginner", "intermediate", "advanced")
        assert lesson.objectives_ar, f"{lesson.id} has no learning objectives"
        assert lesson.id not in lesson.related


def test_search_is_bilingual(registry: Registry):
    assert any(k == "term" for k, _, _ in registry.search("epoch"))
    assert any(k == "term" for k, _, _ in registry.search("الحقبة"))
    assert registry.search("") == []


def test_lesson_path_breadcrumb(registry: Registry):
    lid = registry.ordered_lessons[0]
    path = registry.lesson_path(lid)
    assert path[0] == registry.lessons[lid].section
    assert path[-1] == lid


@pytest.mark.parametrize("route", ["home", "about", "glossary", "map", "progress"])
def test_static_routes_resolve(route):
    from core.routing import STATIC_ROUTES

    assert route in STATIC_ROUTES
