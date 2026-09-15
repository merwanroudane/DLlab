"""Content registry: discovers, links and validates all sections, modules,
lessons, labs and glossary terms.

Discovery is explicit (no magic globbing): every content package lists what
it contains, in order. This keeps ordering deterministic and testable.

    content/<section>/__init__.py        -> SECTION, MODULE_PACKAGES
    content/<section>/<module>/__init__.py -> MODULE, LESSON_MODULES
    content/<section>/<module>/<lesson>.py -> LESSON, render()
    labs/__init__.py                      -> LAB_MODULES
    labs/<lab>.py                         -> LAB, render()
    content/glossary/terms.py             -> TERMS
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

from core.models import Lab, Lesson, Module, Section, Term

CONTENT_SECTIONS = ["foundations", "course"]


@dataclass
class ValidationIssue:
    level: str          # "error" | "warning"
    where: str          # id of the offending object
    message: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.level}] {self.where}: {self.message}"


@dataclass
class Registry:
    sections: dict[str, Section] = field(default_factory=dict)
    modules: dict[str, Module] = field(default_factory=dict)
    lessons: dict[str, Lesson] = field(default_factory=dict)
    labs: dict[str, Lab] = field(default_factory=dict)
    terms: dict[str, Term] = field(default_factory=dict)
    ordered_lessons: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls) -> "Registry":
        reg = cls()
        for section_name in CONTENT_SECTIONS:
            reg._load_section(section_name)
        reg._load_labs()
        reg._load_terms()
        reg._link()
        return reg

    def _load_section(self, section_name: str) -> None:
        pkg = importlib.import_module(f"content.{section_name}")
        section: Section = pkg.SECTION
        self._add_section(section)
        for module_pkg_name in pkg.MODULE_PACKAGES:
            mod_pkg = importlib.import_module(f"content.{section_name}.{module_pkg_name}")
            module: Module = mod_pkg.MODULE
            self._add_module(module)
            for lesson_mod_name in mod_pkg.LESSON_MODULES:
                lesson_mod = importlib.import_module(
                    f"content.{section_name}.{module_pkg_name}.{lesson_mod_name}"
                )
                lesson: Lesson = lesson_mod.LESSON
                if lesson.render is None:
                    lesson.render = getattr(lesson_mod, "render", None)
                self._add_lesson(lesson)

    def _load_labs(self) -> None:
        labs_pkg = importlib.import_module("labs")
        self._add_section(labs_pkg.SECTION)
        for order, lab_mod_name in enumerate(labs_pkg.LAB_MODULES, start=1):
            lab_mod = importlib.import_module(f"labs.{lab_mod_name}")
            lab: Lab = lab_mod.LAB
            if lab.render is None:
                lab.render = getattr(lab_mod, "render", None)
            if lab.order == 0:
                lab.order = order
            self._add_lab(lab)

    def _load_terms(self) -> None:
        terms_mod = importlib.import_module("content.glossary.terms")
        for term in terms_mod.TERMS:
            if term.id in self.terms:
                raise ValueError(f"Duplicate glossary term id: {term.id}")
            self.terms[term.id] = term

    def _add_section(self, section: Section) -> None:
        if section.id in self.sections:
            raise ValueError(f"Duplicate section id: {section.id}")
        self.sections[section.id] = section

    def _add_module(self, module: Module) -> None:
        if module.id in self.modules:
            raise ValueError(f"Duplicate module id: {module.id}")
        self.modules[module.id] = module

    def _add_lesson(self, lesson: Lesson) -> None:
        if lesson.id in self.lessons:
            raise ValueError(f"Duplicate lesson id: {lesson.id}")
        self.lessons[lesson.id] = lesson

    def _add_lab(self, lab: Lab) -> None:
        if lab.id in self.labs:
            raise ValueError(f"Duplicate lab id: {lab.id}")
        self.labs[lab.id] = lab

    def _link(self) -> None:
        """Compute module.lessons, lesson.children, previous / next."""
        # Attach lessons to modules, in declared order.
        for module in self.modules.values():
            module.lessons = []
        for lesson in self.lessons.values():
            if lesson.module in self.modules:
                self.modules[lesson.module].lessons.append(lesson.id)
        for module in self.modules.values():
            module.lessons.sort(key=lambda lid: self.lessons[lid].order)

        # Children of parent lessons (sub-pages).
        for lesson in self.lessons.values():
            lesson.children = []
        for lesson in self.lessons.values():
            if lesson.parent and lesson.parent in self.lessons:
                self.lessons[lesson.parent].children.append(lesson.id)
        for lesson in self.lessons.values():
            lesson.children.sort(key=lambda lid: self.lessons[lid].order)

        # Global reading order: section order -> module order -> lesson order,
        # with sub-pages directly after their parent.
        self.ordered_lessons = []
        for section in sorted(self.sections.values(), key=lambda s: s.order):
            mods = [m for m in self.modules.values() if m.section == section.id]
            for module in sorted(mods, key=lambda m: m.order):
                for lid in module.lessons:
                    lesson = self.lessons[lid]
                    if lesson.parent is None:
                        self.ordered_lessons.append(lid)
                        self.ordered_lessons.extend(lesson.children)

        for i, lid in enumerate(self.ordered_lessons):
            lesson = self.lessons[lid]
            lesson.previous = self.ordered_lessons[i - 1] if i > 0 else None
            lesson.next = (
                self.ordered_lessons[i + 1] if i + 1 < len(self.ordered_lessons) else None
            )

    # --------------------------------------------------------------- queries
    def modules_of(self, section_id: str) -> list[Module]:
        mods = [m for m in self.modules.values() if m.section == section_id]
        return sorted(mods, key=lambda m: m.order)

    def top_lessons_of(self, module_id: str) -> list[Lesson]:
        module = self.modules[module_id]
        return [self.lessons[lid] for lid in module.lessons if self.lessons[lid].parent is None]

    def lesson_path(self, lesson_id: str) -> list[str]:
        """Breadcrumb ids: [section, module, parent?, lesson]."""
        lesson = self.lessons[lesson_id]
        path = [lesson.section, lesson.module]
        if lesson.parent:
            path.append(lesson.parent)
        path.append(lesson.id)
        return path

    def labs_by_category(self) -> dict[str, list[Lab]]:
        out: dict[str, list[Lab]] = {}
        for lab in sorted(self.labs.values(), key=lambda l: l.order):
            out.setdefault(lab.category, []).append(lab)
        return out

    def title_of(self, any_id: str) -> str:
        if any_id in self.lessons:
            return self.lessons[any_id].title_ar
        if any_id in self.modules:
            return self.modules[any_id].title_ar
        if any_id in self.sections:
            return self.sections[any_id].title_ar
        if any_id in self.labs:
            return self.labs[any_id].title_ar
        return any_id

    def search(self, query: str, limit: int = 30) -> list[tuple[str, str, str]]:
        """Search lessons, labs and terms by Arabic or English text.

        Returns (kind, id, label) tuples.
        """
        q = query.strip().lower()
        if not q:
            return []
        hits: list[tuple[int, str, str, str]] = []
        for term in self.terms.values():
            hay = " ".join([term.ar, term.en, term.definition_ar, *term.aliases_ar]).lower()
            if q in hay:
                score = 0 if q in term.en.lower() or q in term.ar.lower() else 1
                hits.append((score, "term", term.id, f"{term.ar} — {term.en}"))
        for lesson in self.lessons.values():
            hay = f"{lesson.title_ar} {lesson.title_en} {lesson.summary_ar}".lower()
            if q in hay:
                score = 0 if q in lesson.title_en.lower() or q in lesson.title_ar.lower() else 1
                hits.append((score, "lesson", lesson.id, f"{lesson.title_ar} — {lesson.title_en}"))
        for lab in self.labs.values():
            hay = f"{lab.title_ar} {lab.title_en} {lab.description_ar}".lower()
            if q in hay:
                hits.append((0, "lab", lab.id, f"{lab.title_ar} — {lab.title_en}"))
        hits.sort(key=lambda h: (h[0], h[3]))
        return [(k, i, l) for _, k, i, l in hits[:limit]]

    # ------------------------------------------------------------ validation
    def validate(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        err = lambda where, msg: issues.append(ValidationIssue("error", where, msg))
        warn = lambda where, msg: issues.append(ValidationIssue("warning", where, msg))

        for module in self.modules.values():
            if module.section not in self.sections:
                err(module.id, f"unknown section '{module.section}'")
            if not module.title_ar or not module.title_en:
                err(module.id, "missing title")
            for pre in module.prerequisites:
                if pre not in self.modules:
                    err(module.id, f"broken module prerequisite '{pre}'")
            if not module.lessons:
                err(module.id, "module has no lessons (empty page)")

        for lesson in self.lessons.values():
            if not lesson.title_ar or not lesson.title_en:
                err(lesson.id, "missing title")
            if lesson.module not in self.modules:
                err(lesson.id, f"orphan lesson: unknown module '{lesson.module}'")
            if not lesson.id.startswith(lesson.module + "."):
                err(lesson.id, f"id must start with module id '{lesson.module}.'")
            if lesson.parent is not None and lesson.parent not in self.lessons:
                err(lesson.id, f"broken parent '{lesson.parent}'")
            if lesson.parent is not None and not lesson.id.startswith(lesson.parent + "."):
                err(lesson.id, f"sub-page id must start with parent id '{lesson.parent}.'")
            for pre in lesson.prerequisites:
                if pre not in self.lessons:
                    err(lesson.id, f"broken prerequisite '{pre}'")
                elif pre == lesson.id:
                    err(lesson.id, "lesson lists itself as prerequisite")
            for rel in lesson.related:
                if rel not in self.lessons:
                    err(lesson.id, f"broken related link '{rel}'")
            for lab in lesson.labs:
                if lab not in self.labs:
                    err(lesson.id, f"broken lab link '{lab}'")
            for term in lesson.terms:
                if term not in self.terms:
                    err(lesson.id, f"unknown glossary term '{term}'")
            if lesson.render is None:
                err(lesson.id, "lesson has no render() function")
            if lesson.id in self.ordered_lessons:
                idx = self.ordered_lessons.index(lesson.id)
                if idx > 0 and lesson.previous is None:
                    err(lesson.id, "missing previous link")
                if idx < len(self.ordered_lessons) - 1 and lesson.next is None:
                    err(lesson.id, "missing next link")
            else:
                err(lesson.id, "lesson not reachable in reading order")

        # Prerequisite graph must be acyclic.
        cycle = self._find_prereq_cycle()
        if cycle:
            err(" -> ".join(cycle), "prerequisite cycle")

        for lab in self.labs.values():
            if lab.render is None:
                err(lab.id, "lab has no render() function")
            for rel in lab.related_lessons:
                if rel not in self.lessons:
                    err(lab.id, f"broken related lesson '{rel}'")

        seen_en: dict[str, str] = {}
        for term in self.terms.values():
            if term.lesson is not None and term.lesson not in self.lessons:
                err(f"term:{term.id}", f"glossary links to missing lesson '{term.lesson}'")
            for rel in term.related:
                if rel not in self.terms:
                    err(f"term:{term.id}", f"broken related term '{rel}'")
            key = term.en.strip().lower()
            if key in seen_en:
                err(f"term:{term.id}", f"duplicate English term '{term.en}' (also {seen_en[key]})")
            seen_en[key] = term.id
            if not term.ar or not term.definition_ar:
                err(f"term:{term.id}", "missing Arabic term or definition")

        return issues

    def _find_prereq_cycle(self) -> Optional[list[str]]:
        WHITE, GREY, BLACK = 0, 1, 2
        color = {lid: WHITE for lid in self.lessons}
        stack: list[str] = []

        def visit(lid: str) -> Optional[list[str]]:
            color[lid] = GREY
            stack.append(lid)
            for pre in self.lessons[lid].prerequisites:
                if pre not in self.lessons:
                    continue
                if color[pre] == GREY:
                    return stack[stack.index(pre):] + [pre]
                if color[pre] == WHITE:
                    found = visit(pre)
                    if found:
                        return found
            stack.pop()
            color[lid] = BLACK
            return None

        for lid in self.lessons:
            if color[lid] == WHITE:
                found = visit(lid)
                if found:
                    return found
        return None


_REGISTRY: Optional[Registry] = None


def get_registry() -> Registry:
    """Return the process-wide registry, building it on first use."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry.build()
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
