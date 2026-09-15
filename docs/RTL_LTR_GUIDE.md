# RTL / LTR Guide — the bidirectional typography system (spec §9, §10, §74)

Arabic is the document language, but every page mixes Arabic, English, code and math.
`direction: rtl` on the whole page is **not enough**; the rules below are implemented once in
`assets/styles/base.css` and exposed to authors through `core/rtl.py`.

## Directions by content type

| Content | Direction | Mechanism |
|---|---|---|
| Arabic prose (markdown, captions, headings, alerts, expander titles, widget labels, buttons, tabs) | RTL, right-aligned | CSS on Streamlit containers (`[data-testid="stMarkdownContainer"]`, …) with `unicode-bidi: plaintext` |
| English technical term inside Arabic | LTR, isolated | back-ticks → `<code>` (`display:inline-block; direction:ltr; unicode-bidi:isolate`) or `en("Learning Rate")` |
| Python code / console output | LTR, left-aligned | `st.code`, `pre`, `code` |
| Equations | LTR, isolated | `.katex`, `.katex-display` |
| URLs / paths / variable names / shape tuples | LTR, isolated | back-ticks |
| Tables | per column | `core.rtl.table(headers, rows, col_dirs)` with `rtl` / `ltr` / `code` / `num` |
| Plotly / Vega charts, dataframes, JSON | LTR | testid selectors |

## Layout direction

All horizontal groups (`st.columns`, `st.container(horizontal=True)`) are `flex-direction:
row-reverse`, so **the first declared column is the rightmost**. Consequences:

- Main layout: `nav_col, content_col = st.columns([1, 3.4])` → navigation on the right.
- Breadcrumb: declare crumbs home → current; they read right-to-left.
- Previous/next: declare `c_prev, c_next` → previous on the right, next on the left (RTL convention).
- Good-vs-bad: declare good first → it sits on the right.
- On narrow screens columns stack top-to-bottom in declaration order (unchanged).

## Author rules

1. Write Arabic as plain Markdown.
2. Put every identifier, code token, path, shape tuple, file name or CLI flag in back-ticks:
   `batch_size = 32`, `X_train.shape`, `(batch, features)`, `model.fit()`.
3. For an English phrase that is not code, use `en("…")` inside HTML or write it in back-ticks.
4. Equations: `st.latex` for display, `$…$` inline. Do not put Arabic inside `\text{}`.
5. Tables with mixed content: use `core.rtl.table` and give each column a direction. Streamlit
   Markdown tables are acceptable only when every cell is Arabic or back-ticked.
6. Never rely on Unicode control characters (LRM/RLM) in content; fix the container instead.
7. Numbers inside Arabic prose are fine (they are weak-directional); numbers inside a `num` column
   are centred with tabular figures.

## Validation checklist (spec §74)

Verify visually after any CSS change: Arabic paragraph · Arabic + English term · Arabic + inline
math · Arabic + inline code · mixed table · breadcrumb · right navigation · buttons with icons ·
segmented control · sliders (LTR track, RTL label) · expander summaries · quiz radios.
Known-good reference page: `?p=foundations.start.how_to_use` (section "العربية والإنجليزية والكود معًا").
