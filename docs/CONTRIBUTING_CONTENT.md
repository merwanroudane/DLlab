# Contributing Content

## Add a lesson

1. Create `content/<section>/<module_pkg>/<lesson>.py`:

```python
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.data.my_lesson",     # must start with the module id
    title_ar="…", title_en="…",
    module="foundations.data",
    order=11,                            # position inside the module
    prerequisites=["foundations.data.dataset"],
    objectives_ar=["…"],
    terms=["dataset"],                   # glossary ids
    related=[], labs=[], difficulty="beginner",
    summary_ar="one line shown on the module page",
)

def render() -> None:
    lesson_header(LESSON)
    h2("ما هو؟", "What is it?")
    ...
    quiz("data.my_lesson", [Q("…", ["a", "b"], 1, "why")])
    lesson_footer(LESSON, ["takeaway 1", "takeaway 2"])
```

2. Append the module name to `LESSON_MODULES` in the module package `__init__.py` (order matters
   only for readability; `Lesson.order` decides).
3. Run `python validate_content.py` and `pytest tests -q`.

Sub-pages: same file shape with `parent="<lesson id>"` and `id="<parent id>.<sub>"`.

## Add a module

Create `content/<section>/<pkg>/__init__.py` with `MODULE = Module(...)` and `LESSON_MODULES`,
add at least one real lesson, then append the package name to `MODULE_PACKAGES` in
`content/<section>/__init__.py`. Modules without lessons fail validation on purpose.

## Add a lab

`labs/<lab>.py` with `LAB = Lab(id="labs.<lab>", …, category=…)` and `render()`; append to
`LAB_MODULES` in `labs/__init__.py`. Call `mark_visited(LAB.id)` at the top of `render()`.
Widget keys must be unique: prefix them with the lab name.

## Add a glossary term

Append a `Term(...)` to `content/glossary/terms.py`. `lesson=` must be an existing lesson id;
`related=` must be existing term ids. English terms must be unique (case-insensitive).

## Components you should use

| Need | Use |
|---|---|
| Section heading with English | `h2(ar, en)`, `h3(ar, en)` |
| Notes | `components.callouts.*` (research, practical, interpretation, mistake, warning, math, coding, debugging, intuition, why, definition, takeaway) |
| Equation with breakdown | `components.math_explainer.equation(...)`, `worked_steps(...)` |
| Mixed-direction table | `core.rtl.table(headers, rows, col_dirs)`; terminology table `core.rtl.term_table(rows)` |
| Comparison / good vs bad | `components.comparison.compare_table`, `good_vs_bad`, `concept_card` |
| Diagram | `components.diagram.diagram(title, svg, what_ar, how_ar, takeaway_ar, legend)` with `svg_box/svg_arrow/svg_text` |
| Pipeline strip | `core.rtl.pipeline(steps, active)` |
| Runnable code | `components.code_lab.CodeLab` + `code_lab(lab)` |
| Quiz | `components.quiz.quiz(key, [Q(...)])` |
| Navigation button | `core.routing.go` as `on_click` |

## Datasets

Use `labs/datasets.py` (offline, deterministic). Add new generators there with `@st.cache_data`
and a seed. Do not fetch from the internet inside lessons (spec §55).

## Style

Match the surrounding code. Arabic prose in Markdown; every code token in back-ticks; no
placeholders. Keep heavy imports (TensorFlow, PyTorch) inside functions.
