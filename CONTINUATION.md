# CONTINUATION — build state (spec §84)

Last updated: 2026-09-15 (Foundations 0–23 and course weeks 01–15 complete; 208 lessons, 35 labs, 146 terms)

## Completed

- **Phase 1 — Architecture + UI shell + RTL/LTR + right navigation**
  - `streamlit_app.py` router with `?p=` routes; `[nav | content]` layout, nav on the right
    (`base.css` makes all horizontal blocks row-reverse → first column is rightmost).
  - Native theme (`.streamlit/config.toml`, warm/bright palette, IBM Plex Sans Arabic + JetBrains Mono).
  - Bidi typography system (`assets/styles/base.css`, `core/rtl.py`): RTL prose, LTR-isolated
    code/math/English, per-column table directions. Verified in the browser.
  - Content registry (`core/registry.py`) with linking + validation; `validate_content.py` CLI.
  - Right navigation (hierarchical, collapsible, current module open, current page highlighted),
    mobile drawer popover, clickable breadcrumb, previous/next, progress (session-scoped).
- **Phase 2 — Reusable components**
  - callouts (12 note kinds + semantic), math_explainer (equation breakdown, worked steps),
    comparison (table, good-vs-bad, concept card), diagram wrapper + SVG helpers, quiz (form,
    scoring, explanations, retry), code_lab (before / line-by-line / run / after / step-by-step),
    lesson_layout (header, footer, module & section landings).
  - **animation_player** (CCv2): Play/Pause/Prev/Next/Restart/speed, step counter, pipeline
    highlight, synchronized caption, equation, before/after values; reduced-motion aware; verified
    in the browser (autoplay + pause + state report). Used in the Epoch/Batch Simulator.
  - **diagnostics.problem_card**: the 18-point Problem framework; first use = "Shape mismatch" in
    `foundations.data.shape_axis_rank`.
- **Phase 3 — Content registry**: done (dataclasses, explicit discovery, ordering, validation).
- **Content built (208 lessons, 35 labs, 146 terms)**
  - Foundations Modules 0–20: Start Here, Data, Python, Math, Linear Algebra, Calculus,
    Probability, ML foundations, Data Prep, Neuron, Architecture, Activations, Forward Pass, Loss,
    Backpropagation (parent + 7 sub-pages), Optimization, Training Loop, Batch/Epoch, Evaluation,
    Generalization, Regularization.
  - **Module 21 — Frameworks Academy (30 lessons, `content/foundations/m21_frameworks/`)**:
    from_python_to_framework, ecosystem_map; Keras parent + 9 sub-pages (layers/models, compile,
    fit + real-trace fit() visualizer, evaluate/predict, callbacks/saving, summary deconstruction,
    output explorer with "how to read this output", common errors, code-lab standard);
    TensorFlow parent + 5 sub-pages (tensors, GradientTape, tf.data, custom loop/saving,
    TensorBoard recreation + errors); PyTorch parent + 8 sub-pages (tensors, Autograd visualizer,
    nn.Module, loss/optim/data, training-loop visualizer with line highlight, zero_grad experiment,
    train/eval experiment, saving + errors); concept mapping (filterable 37-row table), same model
    three views (runs both frameworks), framework choice guide + ecosystem orientation.
  - `labs/fw.py`: lazy TF/Keras/torch imports, captured `summary()`/fit logs, cached runs and
    traces (`keras_mlp_run`, `keras_fit_trace`, `keras_summary_info`, `keras_tensorboard_run`,
    `torch_mlp_run`, `torch_loop_trace`, `torch_zero_grad_experiment`). Pinned:
    tensorflow 2.20.0, keras 3.15.1, torch 2.14.0 (CPU).
  - New labs: TensorFlow Tensor Explorer, GradientTape Lab, PyTorch Tensor Explorer.
  - `core.rtl.inline`: table cells now render `code` spans and **bold**.
  - **Module 22 — Visual Gallery (5 lessons, `m22_gallery/`)**: `components/gallery.py`
    (notebook_cell / console / TensorBoard-panel recreations, all labelled "Educational
    recreation", each with the six reading questions; `svg_line_chart` for recreated figures).
    Outputs are real (`labs.fw.gallery_outputs`, `before_during_after`): TF tensor, summary, fit
    log, evaluate/predict, History plot, TensorBoard scalars, PyTorch tensor/model/log, CPU vs
    GPU, shape/dtype/device errors, warning-vs-error, before/during/after decision surfaces.
  - **Module 23 — Ready for DL (4 lessons, `m23_ready/`)**: concept map (graphviz, coloured by
    progress) + framework map; prerequisite "I can…" checklist + 20-question diagnostic;
    six auto-checked challenges (shapes/params, epoch/batch, loss/optimizer, Keras workflow
    order + fill-in, TF shape + GradientTape value, PyTorch loop bug hunt) recorded in
    `st.session_state["ready_scores"]` (`_scoring.py`); ready status = coverage ≥ 80% +
    diagnostic ≥ 75% + challenges ≥ 70% with a return list and a "start week 01" button.
  - `labs/tinynet.py`: shared NumPy MLP engine (forward/backward/optimizers/dropout/L2/early
    stopping) used by the training/optimization/regularization labs.
  - **Official course, weeks 01–15 (53 lessons)** — `components/week.py` (`week_overview` with
    pipeline/links/pre-test, `post_test`). W02 ML basics (sklearn baselines, ML→DL experiment);
    W03 Keras/TF + PyTorch orientation (full first network, under fit(), PyTorch mini-lab, five
    errors); W04 FFNN (flow animation, depth/width sweep); W05 optimization (η on Keras, optimizer
    race, curve reading, schedules); W06 activations (vanishing over 8 layers, dead ReLU, choice
    rules); W07 applied project (house prices end-to-end: `labs.fw.house_project`); W08 CNN
    concepts (`labs/cnn.py` engine + 3 labs); W09 CNN applications (`labs.fw.cnn_shapes_run` on
    synthetic bar/cross images: overfit/augment/feature maps/shape errors); W10 RNN (`labs/rnn.py`
    engine, unrolling lab, BPTT experiment, `labs.fw.keras_seq_run` on inflation + seasonal series
    — honest naive-baseline comparison); W11 LSTM (windows/padding/Masking, gates lab, RNN vs
    LSTM); W12 GRU (hand GRU == Keras GRU, three-way comparison with seeds, applications map);
    W13 GPU/Colab (memory formula, GPU/batch memory lab, nvidia-smi/OOM recreations); W14 project
    (14-stage guide with checklist, full notebook template); W15 presentation + final rubric with
    self-assessment and course post-test.
  - Glossary extended to 146 terms (ML/prep/neuron/activation/loss/optim/eval/frameworks/CNN/RNN).
  - Labs (25): Dataset Anatomy, Epoch/Batch Simulator, Tensor Shape Explorer, Vector/Matrix,
    Derivative, Gradient, Chain Rule, Scaling, Data Split, Data Leakage, Neuron, Network Builder,
    Parameter Counter, Activation, Loss, Gradient Descent, Learning Rate, Optimizer Race, Training
    Loop Simulator, Confusion Matrix, Threshold, Overfitting, Curves Diagnostic, Regularization, Dropout.
  - `code_lab.exec_source/run_printed/fill_template`: lessons ship authored code that runs live;
    `DLIA_FORCE_RUN_CODELABS=1` makes every code lab execute (used by tests).
  - Glossary/search page, Course map + knowledge graph (graphviz), About, Progress, Home.
- **Tests**: 268 passing (`pytest tests -q`): registry integrity, helpers (steps/epoch, SGD
  simulation, gradient vs finite difference, datasets), headless render of every route, quiz
  scoring, navigation, simulator stepping.
- **Docs**: README, ARCHITECTURE, CURRICULUM_MAP, CONTENT_STANDARDS, RTL_LTR_GUIDE,
  ANIMATION_STANDARDS, DIAGNOSTICS_STANDARDS, CONTRIBUTING_CONTENT.

## Partially completed

- Nothing structural. Possible polish: more diagnostics `problem_card`s inside course weeks,
  extra glossary aliases, and per-lesson knowledge-graph edges for course lessons.

## Not started

- Nothing from the spec's mandatory list.

## Exact next file / task

1. Final polish pass: run `pytest tests -q` (all routes + every code lab), browse a sample of
   pages at desktop and mobile widths, and read the spec §84 checklist once more.
2. Optional depth: add `problem_card` diagnostics to weeks 09–12, and a rolling-origin
   evaluation helper for time series (mentioned in W11/W12 text).

## Known issues

- `st.html(Path("*.css"))` (style-only HTML) did not reach the DOM in Streamlit 1.63; CSS is
  injected through `st.markdown(unsafe_allow_html=True)` instead (`core/theme.py`).
- Widget `value=` + `key=` conflict pattern: step controls must mutate the widget key in
  `on_click` callbacks (see `labs/epoch_batch_simulator.py`), not pass `value=`.
- CCv2 registration is per runtime: `animation_player._mount` re-registers on "not registered"
  so headless `AppTest` runs (fresh runtime per test) work like `streamlit run`.
- The Windows console is cp1252: run tools with `PYTHONIOENCODING=utf-8`.
- Global Python has a locked `streamlit.exe` from another running app; this project uses `.venv`.

## Test status

`339 passed` on 2026-09-15 with `.venv` (Streamlit 1.63.0, NumPy 2.4.6, pandas 3.0.5, Plotly 7.0.0,
TensorFlow 2.20.0, Keras 3.15.1, torch 2.14.0 CPU). The full suite renders every route and runs every
code lab with real frameworks, so it takes ~14 minutes; `tests/test_registry.py tests/test_frameworks.py`
alone take ~15 s and are the quick smoke check.

## How to resume

```bash
.venv/Scripts/python -m streamlit run streamlit_app.py --server.port 8511
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python validate_content.py
```
