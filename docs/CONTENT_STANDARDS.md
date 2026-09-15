# Content Standards

## The non-negotiable rule (spec §2)

**No technical term is used before it is explained**, or linked to the lesson that explains it.
Small terms count: `Data`, `Shape`, `dtype`, `Tensor`, `Weight`, `Bias`, `Loss`, `Gradient`,
`Epoch`, `Batch`, `Step`, `Padding`, `Stride`, `Kernel`, `Hidden State`, `Gate` …
If a concept matters it gets a Concept Lesson or sub-page, not a one-line tooltip.

## Terminology (spec §13)

- One canonical Arabic translation per English term, declared once in `content/glossary/terms.py`.
- English terms always appear inside back-ticks (LTR-isolated) next to the Arabic: **الحقبة** `Epoch`.
- A lesson lists the terms it introduces in `Lesson.terms`; the validator rejects unknown ids.

## Anatomy of a concept lesson (spec §17)

Cover what fits the concept, in this order; depth where needed, not 46 tabs:

overview · objectives · prerequisites · key terminology · what is it? · why do we need it? ·
intuition · conceptual explanation · visual explanation · diagram · mathematical foundation ·
equation breakdown · numerical example · step-by-step calculation · animation · interactive
explorer · Python example · run · explain code · run step-by-step · what happens internally? ·
practical example · realistic dataset example · variations · parameters involved · hyperparameters
involved · what happens if…? · problems & failure modes · symptoms · root causes · diagnosis ·
fixes · failure animation · good vs bad · common mistakes · edge cases · researcher notes · mini
experiment · exercise · diagnostic challenge · quiz · summary · key takeaways · related · previous ·
next.

Implementation: `lesson_header(LESSON)` (objectives, prerequisites, terms, sub-pages) … body …
`lesson_footer(LESSON, summary_points)` (summary, related, labs, previous/next).

## Progressive disclosure (spec §18)

Core explanation first. Deep Dive / Mathematical Deep Dive / Implementation Details / Research
Notes go in `st.expander` or dedicated sub-pages. Do not hide everything in expanders (spec §60).

## Mathematics (spec §11, §12)

- Use `components.math_explainer.equation(latex, symbols, meaning_ar, example_ar, dl_link_ar)`.
  Every equation is followed by: each symbol explained, meaning, a numerical example, the link to DL.
- A math concept passes through: intuition → graph → numerical example → definition → derivation
  (when useful) → connection to DL → Python implementation → interactive experiment.

## Code (spec §22, §22A, §22B, §23)

- Use `components.code_lab.CodeLab`: **Before** (goal, pipeline stage, prerequisites, inputs with
  shape/dtype, expected outputs, mathematical counterpart, framework objects) → code → line-by-line
  explanation → **Run** with real output → **After** (how to read the output).
- Three levels: A concept code · B framework code · C applied pipeline. Never jump to C first.
- Never show `model.fit()`, `optimizer.step()`, `loss.backward()`, `DataLoader`, `Conv2D`, `LSTM`
  before the concept behind them has its lesson.
- No unexplained abbreviations (`X`, `y`, `lr`, `bs`, `wd`) on first appearance.
- Predefined, safe code + parameter controls. No `exec`/`eval` on user text.
- Code that is labelled runnable must run; pseudocode must be labelled pseudocode.

## Diagnostics inside the lesson (spec §32, §33)

Problems live in the lesson where they occur, using the 18-point framework in
`docs/DIAGNOSTICS_STANDARDS.md`. No separate "troubleshooting center" as the only place.

## Colour (spec §15)

Semantic colours are fixed (data, parameter, hyperparameter, loss, gradient, correct, warning,
failure) and always paired with an icon and a label. Use `components.callouts` and `core.rtl.chip`.

## Progressive example (spec §27)

Use **Study Hours → Exam Score** to introduce observation, feature, target, function, weight,
bias, prediction, error, loss, gradient, update, epoch, batch. Introduce other datasets only when
needed (`labs/datasets.py`: loan default, house prices, monthly inflation).

## Placeholders are forbidden (spec §75)

No TODO, "coming soon", placeholder lessons, lorem ipsum, fake buttons or fake sliders. A module is
registered only when it has real lessons. The app test asserts none of these strings appear.

## Quality gate per lesson (spec §77)

Before marking a lesson done, answer: unexplained term? prerequisites clear? intuition? visual?
math correct? symbols explained? numerical example? interaction (if meaningful)? Python (if
meaningful)? code runnable? process explained, not only output? problems? diagnosis? fixes? common
mistakes? summary? previous/next?
