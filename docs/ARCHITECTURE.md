# Architecture

## Principles

1. **Content is data, not UI code.** Every lesson declares a `Lesson` dataclass (metadata) and a
   `render()` function. The registry links, orders and validates everything; the UI only asks the
   registry what to show.
2. **One router, one URL parameter.** `?p=<route>` is the single source of truth for navigation, so
   every page is linkable and the browser back button works. No `st.navigation` left-sidebar pages.
3. **Reusable components, no monolith.** Callouts, equations, tables, code labs, quizzes, diagrams,
   breadcrumbs and navigation are components; lessons compose them.
4. **Arabic-first, bidi-correct.** RTL for prose, LTR isolation for code, equations, paths and
   English terms — implemented once in `assets/styles/base.css` (see `docs/RTL_LTR_GUIDE.md`).
5. **Real code only.** Code labs execute predefined Python with parameter controls — never
   `exec` on user text.

## Directory layout

```
streamlit_app.py            entry point: theme, state, registry, router, [content | right nav] layout
.streamlit/config.toml      native theme (warm/bright palette, Arabic + code fonts)
core/
  models.py                 Section / Module / Lesson / Lab / Term dataclasses
  registry.py               discovery, linking (prev/next/children), validation, search
  routing.py                current_route(), resolve(), go(), nav_button()
  state.py                  session-state keys (visited, quiz, anim, experiments, last_lesson)
  progress.py               per-module / per-section progress
  theme.py                  CSS injection, semantic colour legend
  rtl.py                    esc/en/code/chip helpers, bidi table(), pipeline()
components/
  right_navigation.py       hierarchical collapsible right panel + mobile drawer
  breadcrumb.py             clickable trail
  lesson_layout.py          lesson_header / lesson_footer / prev_next / module & section landings
  callouts.py               researcher notes (spec §53) + semantic callouts (spec §15)
  math_explainer.py         equation() with symbol breakdown, worked_steps()
  code_lab.py               Code Understanding Contract: before / line-by-line / run / after
  comparison.py             compare_table(), good_vs_bad(), concept_card()
  diagram.py                diagram() wrapper (title / what / legend / how / takeaway) + SVG helpers
  quiz.py                   form-based quiz with explanations
content/
  foundations/<module>/     MODULE + LESSON_MODULES; one file per lesson
  course/<week>/            same shape, one package per week
  glossary/terms.py         TERMS list
labs/
  __init__.py               SECTION + LAB_MODULES
  datasets.py               packaged synthetic datasets (offline, deterministic)
  <lab>.py                  LAB + render()
app_pages/                  static pages: home, about, glossary, course_map, labs_index, progress
assets/styles/base.css      design tokens + bidi system + cards/callouts/tables
tests/                      registry, helpers, headless app tests
docs/                       this documentation
```

## Identifiers and routes

| Kind | Id / route | Example |
|---|---|---|
| section | `<section>` | `foundations`, `course`, `labs` |
| module | `<section>.<module>` | `foundations.data`, `course.w01` |
| lesson | `<module>.<lesson>` | `foundations.data.shape_axis_rank` |
| sub-page | `<lesson>.<sub>` (with `parent=`) | `foundations.backprop.chain_rule.examples` |
| lab | `labs.<lab>` | `labs.epoch_batch_simulator` |
| static | `home`, `about`, `glossary`, `map`, `progress`, `search` | |

## Registry lifecycle

`core.registry.get_registry()` builds a process-wide `Registry` on first use by importing the
content packages in declared order (`MODULE_PACKAGES`, `LESSON_MODULES`, `LAB_MODULES`). Linking
computes `module.lessons`, `lesson.children`, and the global reading order (section → module →
lesson, sub-pages right after their parent) which defines `previous` / `next`.
`Registry.validate()` returns a list of issues; `validate_content.py` exposes it as a CLI and
`tests/test_registry.py` asserts it is empty.

## Layout and navigation

`streamlit_app.py` renders `st.columns([1, 3.4])`. Because `base.css` sets every
`stHorizontalBlock` to `flex-direction: row-reverse`, the **first column is the rightmost**:
the navigation panel sits on the right, content on the left, and every `st.columns` /
`st.container(horizontal=True)` in the app reads right-to-left like Arabic. On narrow screens the
columns stack (content, then navigation); a "القائمة" popover above the content provides the drawer.

The right panel shows sections; inside the active section, modules as expanders (current module
open) with lesson buttons; the current lesson's sub-pages are indented under it. The current page
is highlighted with `type="primary"`.

## State

All per-user state is in `st.session_state` and initialised in `core/state.py`:
`visited` (progress), `last_lesson` (resume), `quiz`, `anim`, `experiments`, `nav_open_modules`.
Widget keys use prefixes (`nav_`, `bc_`, `quiz_`, `ctrl_<lab>_`, `ebs_`) to avoid collisions.

## Performance

- The registry is built once per process; content modules are plain Python (fast imports).
- Heavy libraries (TensorFlow, PyTorch) are imported **inside** the lessons that need them, never at
  module import time.
- Datasets and simulations are `st.cache_data`-cached with bounded `max_entries`.
- Stylesheets are read once and cached by file mtime.

## Extending

See `docs/CONTRIBUTING_CONTENT.md`.
