"""Frameworks Academy (module 21), Gallery (22) and Ready (23) checks."""

import numpy as np
import pytest

from labs import fw


def test_versions_are_the_pinned_ones():
    v = fw.versions()
    assert v["tensorflow"].startswith("2.20") and v["keras"].startswith("3.") and v["torch"].startswith("2.14")


def test_keras_run_returns_plain_data_with_expected_shapes():
    r = fw.keras_mlp_run((8,), "relu", "adam", 0.01, 2, 64, 0)
    assert r["n_params"] == 2 * 8 + 8 + 8 + 1
    assert r["predict_shape"] == [5, 1] and len(r["history"]["loss"]) == 2
    assert r["steps_per_epoch"] == int(np.ceil(450 / 64))
    assert "Total params" in r["summary"] and "Epoch 1/2" in r["log"]


def test_keras_fit_trace_has_one_batch_event_per_update():
    tr = fw.keras_fit_trace(8, 2, 40, 0.5, 0)
    batches = [e for e in tr["events"] if e["kind"] == "batch"]
    assert len(batches) == 2 * tr["steps_per_epoch"] == 6
    assert all(e["w_before"] != e["w_after"] for e in batches)


def test_summary_info_counts_batchnorm_non_trainable():
    info = fw.keras_summary_info((("dense", 16, "relu"), ("bn",), ("dropout", 0.2), ("dense", 3, "softmax")), (4,))
    assert info["total"] == 80 + 64 + 0 + 51 and info["non_trainable"] == 32
    assert [r["type"] for r in info["rows"]] == ["Dense", "BatchNormalization", "Dropout", "Dense"]


def test_torch_loop_trace_shapes_and_weight_updates():
    tr = fw.torch_loop_trace(8, 1, 40, 0.5, 0)
    b = [e for e in tr["events"] if e["kind"] == "batch"]
    assert len(b) == tr["steps_per_epoch"] == 3 and b[0]["x_shape"] == [40, 2] and b[0]["out_shape"] == [40, 1]
    assert b[0]["g_before"] is None and b[1]["g_before"] is not None


def test_zero_grad_experiment_shows_accumulation():
    z = fw.torch_zero_grad_experiment(5, 0.1, 0)
    with_, without = z["with"], z["without"]
    assert with_[0]["g00"] == pytest.approx(without[0]["g00"])
    assert without[-1]["gnorm"] > 2 * with_[-1]["gnorm"]
    assert without[1]["g00"] == pytest.approx(without[0]["g00"] + without[1]["fresh"], abs=1e-6)


def test_gallery_outputs_are_real_library_text():
    o = fw.gallery_outputs()
    assert "tf.Tensor: shape=(2, 3), dtype=float32" in o["tf_tensor"]
    assert "Total params: 193" in o["keras_summary"] and "Epoch 1/5" in o["keras_fit_log"]
    assert "Sequential(" in o["torch_model"] and "mat1 and mat2 shapes cannot be multiplied" in o["torch_shape_error"]
    assert "expected axis -1 of input shape to have value 5" in o["keras_shape_error"]


def test_inline_table_markup():
    from core.rtl import inline
    assert inline("a `x<y` **b**") == 'a <code class="ltr">x&lt;y</code> <b>b</b>'


def test_ready_challenges_record_scores():
    from tests.test_app import run_route

    at = run_route("foundations.ready.challenges", timeout=120)
    at.multiselect(key="ch_torch_pick").set_value(["A", "D", "F", "G"]).run()
    at.button(key="ch_torch_check").click().run()
    assert not at.exception
    assert at.session_state["ready_scores"]["torch_loop"] == {"correct": 4, "total": 4, "module": "foundations.frameworks"}
    at.query_params["p"] = "foundations.ready.ready_status"
    at.run()
    assert not at.exception
    assert any("ليس بعد" in w.value for w in at.warning)
