"""Math / data helpers: steps per epoch, the real SGD simulation, bidi table
guards and packaged datasets (spec §72)."""

import math

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------ epoch / batch
@pytest.mark.parametrize(
    "n,bs,drop,expected",
    [(100, 20, False, 5), (100, 20, True, 5), (100, 30, False, 4), (100, 30, True, 3),
     (7, 8, False, 1), (7, 8, True, 0), (1, 1, False, 1)],
)
def test_steps_per_epoch(n, bs, drop, expected):
    from labs.epoch_batch_simulator import batches_for_epoch

    assert batches_for_epoch(n, bs, drop) == expected


def test_simulation_step_count_and_partial_batch():
    from labs.epoch_batch_simulator import simulate

    steps = simulate(n=100, batch_size=30, epochs=2, shuffle=False, lr=0.1, drop_remainder=False)
    assert len(steps) == 8
    assert [s.is_partial for s in steps[:4]] == [False, False, False, True]
    assert steps[3].epoch_end and steps[7].epoch_end
    assert len(steps[3].indices) == 10
    # every observation seen exactly once per epoch
    seen = np.concatenate([s.indices for s in steps[:4]])
    assert sorted(seen.tolist()) == list(range(100))


def test_simulation_parameters_chain_and_loss_decreases():
    from labs.epoch_batch_simulator import simulate

    steps = simulate(n=100, batch_size=20, epochs=3, shuffle=True, lr=0.1, drop_remainder=False)
    for a, b in zip(steps, steps[1:]):
        assert math.isclose(a.w_after, b.w_before) and math.isclose(a.b_after, b.b_before)
    first_epoch = np.mean([s.loss for s in steps[:5]])
    last_epoch = np.mean([s.loss for s in steps[-5:]])
    assert last_epoch < first_epoch


def test_simulation_diverges_with_huge_learning_rate():
    from labs.epoch_batch_simulator import simulate

    steps = simulate(n=100, batch_size=20, epochs=3, shuffle=False, lr=1.5, drop_remainder=False)
    assert steps[-1].loss > steps[0].loss


def test_gradient_matches_finite_difference():
    """The hand-written MSE gradient equals a numerical derivative."""
    rng = np.random.default_rng(0)
    x, y = rng.uniform(size=20), rng.uniform(size=20)
    w, b = 0.3, -0.1

    def loss(w_, b_):
        return float(np.mean((w_ * x + b_ - y) ** 2))

    dw = float(np.mean(2 * (w * x + b - y) * x))
    db = float(np.mean(2 * (w * x + b - y)))
    eps = 1e-6
    assert math.isclose(dw, (loss(w + eps, b) - loss(w - eps, b)) / (2 * eps), rel_tol=1e-5)
    assert math.isclose(db, (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps), rel_tol=1e-5)


# ---------------------------------------------------------------- datasets
def test_packaged_datasets_are_deterministic_and_shaped():
    from labs.datasets import house_prices, loan_default, monthly_inflation, study_hours

    a, b = loan_default(), loan_default()
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (400, 8) and set(a["defaulted"].unique()) <= {0, 1}
    assert a["debt_ratio"].isna().sum() > 0  # deliberate missing values for teaching
    assert house_prices().shape == (300, 5)
    assert monthly_inflation().shape == (180, 4)
    assert study_hours().shape == (40, 2)


def test_variable_kind_heuristic():
    from labs.datasets import variable_kind

    assert variable_kind(pd.Series(["a", "b", "a"])) == "فئوي ثنائي"
    assert variable_kind(pd.Series(["a", "b", "c"])) == "فئوي اسمي"
    assert variable_kind(pd.Series([0, 1, 1, 0])) == "فئوي ثنائي (0/1)"
    assert variable_kind(pd.Series([1.5, 2.25, 3.0])) == "عددي متصل"


# -------------------------------------------------------------- rtl helpers
def test_bidi_table_rejects_bad_rows():
    from core.rtl import table
    from streamlit.testing.v1 import AppTest

    def app():
        from core.rtl import table
        table(["a", "b"], [("1", "2", "3")])

    at = AppTest.from_function(app).run()
    assert at.exception, "row with wrong cell count must raise"


def test_esc_and_inline_helpers():
    from core.rtl import chip, code, en, esc

    assert esc("<b>") == "&lt;b&gt;"
    assert 'class="en"' in en("Learning Rate")
    assert code("x<1") == '<code class="ltr">x&lt;1</code>'
    assert "sem-param" in chip("w", "param")


def test_callout_body_markdown_subset():
    from components.callouts import _paragraphs

    html = _paragraphs("**bold** and `code`\n\nsecond")
    assert "<strong>bold</strong>" in html and '<code class="ltr">code</code>' in html
    assert html.count("<p>") == 2
