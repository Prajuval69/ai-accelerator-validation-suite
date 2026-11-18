import json
from copy import deepcopy

import pytest
from test_unit import record

from aivs.diagnostics import error_message, error_trace
from aivs.regression import compare_runs
from aivs.storage import load_run


def test_diagnostics_strip_absolute_paths():
    exception = RuntimeError(
        'Unable to load "C:\\Users\\private\\model.onnx" or /tmp/private/model'
    )
    message = error_message(exception)
    assert "private" not in message
    assert "<path>" in message
    try:
        raise RuntimeError("controlled error")
    except RuntimeError as exc:
        trace = error_trace(exc)
    assert trace[-1]["file"] == "test_diagnostics.py"
    assert "/" not in trace[-1]["file"] and "\\" not in trace[-1]["file"]


@pytest.mark.parametrize("field", ["model_sha256", "input_sha256", "session_settings"])
def test_missing_case_provenance_is_incomparable(field):
    before, after = record(), record()
    for item in (before, after):
        del item["cases"][0][field]
    row = compare_runs(before, after, allow_environment_drift=True)["comparisons"][0]
    assert row["verdict"] == "INCOMPARABLE"
    assert "missing provenance" in row["reason"]


def test_unknown_cpu_and_missing_measurement_origin_incomparable():
    before, after = record(), record()
    before["environment"]["cpu"] = after["environment"]["cpu"] = "unknown"
    assert compare_runs(before, after)["comparisons"][0]["verdict"] == "INCOMPARABLE"
    before, after = record(), record()
    del before["measurement_origin"]
    del after["measurement_origin"]
    assert compare_runs(before, after)["comparisons"][0]["verdict"] == "INCOMPARABLE"


def test_boolean_p95_invalid():
    assert compare_runs(record(True), record())["comparisons"][0]["verdict"] == "INVALID"


@pytest.mark.parametrize("change", ["root", "missing", "case", "duplicate"])
def test_malformed_saved_record_rejected(tmp_path, change):
    run = record()
    if change == "root":
        run = []
    elif change == "missing":
        del run["environment"]
    elif change == "case":
        del run["cases"][0]["accuracy"]
    else:
        run["cases"].append(deepcopy(run["cases"][0]))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        load_run(path)


def test_duplicate_comparison_case_rejected():
    before, after = record(), record()
    before["cases"].append(deepcopy(before["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        compare_runs(before, after)
