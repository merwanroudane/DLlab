"""Data model for the content registry.

Everything the platform knows about a lesson, a module, a section, a lab or
a glossary term lives in these dataclasses. Content files declare instances;
`core.registry` collects them, links them and validates them.

Identifier conventions
----------------------
- Section ids:  "home", "foundations", "course", "labs", "glossary", "map", "about"
- Module ids:   "<section>.<module>"            e.g. "foundations.data"
- Lesson ids:   "<section>.<module>.<lesson>"   e.g. "foundations.data.what_is_data"
- Sub-lesson:   "<lesson-id>.<sub>"             e.g. "foundations.data.what_is_data.examples"
- Lab ids:      "labs.<lab>"                    e.g. "labs.epoch_batch_simulator"
- Term ids:     snake_case English              e.g. "learning_rate"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

Difficulty = Literal["beginner", "intermediate", "advanced"]

RenderFn = Callable[[], None]


@dataclass
class Section:
    """A top-level area of the platform (Foundations, Course, Labs...)."""

    id: str
    title_ar: str
    title_en: str
    icon: str
    order: int
    description_ar: str = ""


@dataclass
class Module:
    """A module groups lessons inside a section (e.g. 'Data Foundations')."""

    id: str
    section: str
    title_ar: str
    title_en: str
    order: int
    purpose_ar: str = ""
    why_ar: str = ""
    prerequisites: list[str] = field(default_factory=list)   # module ids
    objectives_ar: list[str] = field(default_factory=list)
    challenges_ar: list[str] = field(default_factory=list)
    icon: str = ":material/menu_book:"
    # Filled by the registry
    lessons: list[str] = field(default_factory=list)          # lesson ids (ordered)

    @property
    def number(self) -> int:
        return self.order


@dataclass
class Lesson:
    """One navigable page of content (concept lesson, sub-page or week page)."""

    id: str
    title_ar: str
    title_en: str
    module: str
    order: int
    parent: Optional[str] = None                 # lesson id for sub-pages
    prerequisites: list[str] = field(default_factory=list)   # lesson ids
    objectives_ar: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)           # glossary term ids
    related: list[str] = field(default_factory=list)         # lesson ids
    labs: list[str] = field(default_factory=list)            # lab ids
    diagnostics: list[str] = field(default_factory=list)     # problem names
    difficulty: Difficulty = "beginner"
    summary_ar: str = ""
    render: Optional[RenderFn] = None
    # Filled by the registry
    previous: Optional[str] = None
    next: Optional[str] = None
    children: list[str] = field(default_factory=list)

    @property
    def section(self) -> str:
        return self.id.split(".")[0]

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else 1


@dataclass
class Lab:
    """An interactive lab. Labs are also lessons in the 'labs' section."""

    id: str
    title_ar: str
    title_en: str
    category: str                          # data | math | training | cnn | sequence | frameworks
    description_ar: str
    render: Optional[RenderFn] = None
    related_lessons: list[str] = field(default_factory=list)
    icon: str = ":material/science:"
    order: int = 0


@dataclass
class Term:
    """A glossary entry. One canonical Arabic translation per English term."""

    id: str
    ar: str
    en: str
    definition_ar: str
    category: str
    lesson: Optional[str] = None           # full lesson id
    related: list[str] = field(default_factory=list)   # term ids
    notation: str = ""                     # LaTeX without $ delimiters
    aliases_ar: list[str] = field(default_factory=list)
