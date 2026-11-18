from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aivs.accuracy import check_outputs
from aivs.benchmark import summarize
from aivs.config import ConfigError, parse_config
from aivs.faults import classify, timeout_probe
from aivs.models import build_model, make_input
from aivs.providers import ProviderUnavailable, create_session, finish_profile
from aivs.regression import ENV_FIELDS, compare_runs
from aivs.reporting import link, render_report
from aivs.storage import finite_json, prepare_output


def record(p95=1.0):
    case = {
        "case_id": "mlp_gemm_b1",
        "model_id": "mlp_gemm",
        "model_sha256": "abc",
        "input_sha256": "def",
        "input_shape": [1, 128],
        "batch": 1,
        "precision": "float32",
        "requested_provider": "CPUExecutionProvider",
        "selected_providers": ["CPUExecutionProvider"],
        "session_settings": parse_config()["backend"],
        "status": "PASS",
        "accuracy": {"verdict": "PASS"},
        "timing": {"p95_ms": p95},
    }
    return {
        "schema_version": 1,
        "run_id": "test",
        "started_at_utc": "2026-01-01T00:00:00Z",
        "kind": "benchmark",
        "synthetic_models": True,
        "measurement_origin": "simulated_fixture",
        "environment": {
            **dict.fromkeys(ENV_FIELDS, "simulated"),
            "platform": "unit-test",
            "cpu": "simulated",
            "python": "3.11",
            "onnxruntime": "test",
            "available_providers": ["CPUExecutionProvider"],
        },
        "config": parse_config(),
        "cases": [case],
        "summary": {},
    }


def test_statistics():
    samples = [1.0, 2.0, 3.0, 4.0]
    stats = summarize(samples, 0.5)
    assert stats["mean_ms"] == 2.5
    assert stats["stddev_ms"] == np.std(samples)
    for p in (50, 95, 99):
        assert stats[f"p{p}_ms"] == np.percentile(samples, p)
    assert stats["throughput_inferences_per_second"] == 8
    assert summarize([], 0) is None


@pytest.mark.parametrize(
    "samples,elapsed", [([0], 1), ([float("nan")], 1), ([1], 0), ([1], float("inf")), ([-1], 1)]
)
def test_bad_statistics(samples, elapsed):
    with pytest.raises(ValueError):
        summarize(samples, elapsed)


@pytest.mark.parametrize(
    "data",
    [
        {"unknown": 1},
        {"measurement": {"typo": 1}},
        {"measurement": {"timed_iterations": 0}},
        {"stress": {"duration_seconds": -1}},
        {"measurement": {"concurrency": 0}},
        {"measurement": {"concurrency": 5}},
        {"measurement": {"timed_iterations": True}},
        {"input": {"dtype": "float16"}},
        {"input": {"batch_sizes": [10000]}},
        {"input": {"batch_sizes": [1, 1]}},
        {"models": ["unknown"]},
        {"models": ["mlp_gemm", "mlp_gemm"]},
        {"backend": {"required": "yes"}},
        {"backend": {"intra_op_num_threads": 0}},
        {"accuracy": {"atol": float("nan")}},
        {"report": {"json": False}},
        {"schema_version": True},
        {"input": None},
        {"regression": {"warning_percent": 20, "fail_percent": 15}},
        {"stress": {"max_duration_seconds": 121}},
        {"stress": {"max_samples_per_worker": 0}},
    ],
)
def test_invalid_config(data):
    with pytest.raises(ConfigError):
        parse_config(data)


def test_stress_config_and_defaults_independent():
    assert parse_config(kind="stress")["measurement"]["concurrency"] == 2
    one = parse_config()
    one["models"].clear()
    assert parse_config()["models"]
    with pytest.raises(ConfigError, match="require concurrency"):
        parse_config({"measurement": {"concurrency": 2}})


@pytest.mark.parametrize("family", ["mlp_gemm", "conv2d", "attention_matmul"])
def test_deterministic_model_input(family):
    before, after = build_model(family, 12), build_model(family, 12)
    for key, value in before.state_dict().items():
        assert np.array_equal(value.numpy(), after.state_dict()[key].numpy())
    assert np.array_equal(make_input(family, 1, 12), make_input(family, 1, 12))
    assert not np.array_equal(make_input(family, 1, 12), make_input(family, 1, 13))
    with pytest.raises(ValueError, match="memory budget"):
        make_input(family, 1, 12, 0.0001)


def test_accuracy_shape_nan_and_tolerance():
    ref = np.array([1.0], dtype=np.float32)
    assert check_outputs(ref, ref)["verdict"] == "PASS"
    assert check_outputs(ref, ref + 0.2)["verdict"] == "FAIL"
    assert check_outputs(ref, ref.reshape(1, 1))["verdict"] == "FAIL"
    assert check_outputs(ref, np.array([np.nan]))["verdict"] == "FAIL"


@pytest.mark.parametrize("provider", ["CUDAExecutionProvider", "MIGraphXExecutionProvider"])
def test_missing_provider_never_constructs_cpu_session(monkeypatch, tmp_path, provider):
    monkeypatch.setattr(
        "aivs.providers.ort.get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    construct = MagicMock()
    monkeypatch.setattr("aivs.providers.ort.InferenceSession", construct)
    cfg = parse_config({"backend": {"provider": provider}})
    with pytest.raises(ProviderUnavailable):
        create_session(Path("unused.onnx"), cfg, tmp_path, "test")
    construct.assert_not_called()


def test_provider_failed_load_detected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "aivs.providers.ort.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    monkeypatch.setattr("aivs.providers.ort.InferenceSession", lambda *a, **k: session)
    cfg = parse_config({"backend": {"provider": "CUDAExecutionProvider"}})
    with pytest.raises(ProviderUnavailable, match="failed to load"):
        create_session(Path("unused.onnx"), cfg, tmp_path, "test")


def test_profile_missing_evidence_is_unverified(tmp_path):
    result = finish_profile(MagicMock(), {"profiling_enabled": False}, tmp_path)
    assert result["status"] == "placement unverified"
    assert result["cpu_fallback_observed"] is None


@pytest.mark.parametrize(
    "delta,expected",
    [(0, "PASS"), (5, "WARNING"), (15, "REGRESSION"), (25, "REGRESSION"), (-10, "PASS")],
)
def test_regression_percent_and_boundary(delta, expected):
    result = compare_runs(record(100), record(100 + delta))
    assert result["comparisons"][0]["verdict"] == expected
    assert result["comparisons"][0]["delta_percent"] == pytest.approx(delta)


@pytest.mark.parametrize(
    "field",
    [
        "model_sha256",
        "input_sha256",
        "batch",
        "input_shape",
        "precision",
        "requested_provider",
        "selected_providers",
        "session_settings",
    ],
)
def test_incompatible_case(field):
    before, after = record(), record()
    after["cases"][0][field] = "changed"
    result = compare_runs(before, after, allow_environment_drift=True)
    assert result["comparisons"][0]["verdict"] == "INCOMPARABLE"
    assert result["comparisons"][0]["changed_fields"][0]["field"] == field


@pytest.mark.parametrize(
    "field", ["cpu", "python", "onnxruntime", "onnx", "torch", "architecture", "total_ram_bytes"]
)
def test_incompatible_environment_and_override(field):
    before, after = record(), record(1.25)
    after["environment"][field] = "changed"
    assert compare_runs(before, after)["comparisons"][0]["verdict"] == "INCOMPARABLE"
    row = compare_runs(before, after, allow_environment_drift=True)["comparisons"][0]
    assert row["comparison_confidence"] == "low"
    assert row["verdict"] == "REGRESSION"
    assert "SEVERE" in row["warning"]


def test_incompatible_methodology_and_case_missing():
    before, after = record(), record()
    after["config"]["measurement"]["enable_ort_profiling"] = True
    assert compare_runs(before, after)["comparisons"][0]["verdict"] == "INCOMPARABLE"
    after = deepcopy(before)
    after["cases"] = []
    assert compare_runs(before, after)["verdict"] == "FAIL"
    assert compare_runs({**before, "cases": []}, after)["verdict"] == "FAIL"


@pytest.mark.parametrize("invalid", [0, -1, None, float("nan"), float("inf")])
def test_invalid_baseline(invalid):
    assert compare_runs(record(invalid), record())["comparisons"][0]["verdict"] == "INVALID"


def test_failed_or_unvalidated_cannot_compare():
    before, after = record(), record()
    after["cases"][0]["status"] = "FAIL"
    assert compare_runs(before, after)["comparisons"][0]["verdict"] == "INCOMPARABLE"
    after["cases"][0]["status"] = "PASS"
    after["cases"][0]["accuracy"]["verdict"] = "SKIP"
    assert compare_runs(before, after)["comparisons"][0]["verdict"] == "INCOMPARABLE"


def test_fault_unexpected_success_and_wrong_exception():
    assert classify("test", "EXPECTED_REJECTION", lambda: None, (ValueError,))["outcome"] == "FAIL"

    def reject():
        raise RuntimeError("unexpected")

    assert (
        classify("test", "EXPECTED_REJECTION", reject, (ValueError,))["observed_error_category"]
        == "UNEXPECTED_ERROR"
    )


def test_timeout_reaped():
    result = timeout_probe()
    assert result["outcome"] == "PASS"
    assert result["cleanup_complete"]


def test_finite_json_and_report_escaping(tmp_path):
    assert finite_json({"values": [float("nan"), float("inf")]}) == {"values": [None, None]}
    run = record()
    run["cases"][0]["timing"] = None
    run["cases"][0]["model_id"] = "<script>alert(1)</script>"
    run["cases"][0]["case_id"] = "<script>alert(1)</script>"
    path = render_report(run, tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "SIMULATED" in text
    assert "https://" not in text
    assert "—" in text
    assert "href" not in link("../../secret", "unsafe")
    assert "href" not in link("javascript:alert(1)", "unsafe")


def test_no_overwrite(tmp_path):
    root = tmp_path / "out"
    prepare_output(root)
    with pytest.raises(ValueError, match="must be empty"):
        prepare_output(root)
