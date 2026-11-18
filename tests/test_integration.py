import csv
import inspect
import json
from copy import deepcopy

import jsonschema
import numpy as np
import onnx
import pytest

from aivs.config import parse_config
from aivs.exporter import export_model
from aivs.faults import run_faults
from aivs.fixtures import WorkloadSpec, file_hash
from aivs.models import build_model, make_input
from aivs.regression import simulated_regression_demo
from aivs.runner import execute
from aivs.storage import prepare_output
from aivs.stress import measure_stress

pytestmark = pytest.mark.integration


def workload_from_case(root, case, cfg):
    return WorkloadSpec(
        case,
        build_model(case["model_id"], cfg["seed"]),
        make_input(case["model_id"], case["batch"], cfg["seed"]),
        root / case["model_path"],
    )


def test_real_three_models_samples_schemas_and_profiles(cpu_run, schemas):
    root, run = cpu_run
    assert run["summary"]["verdict"] == "PASS"
    assert len(run["cases"]) == 3
    jsonschema.validate(run, schemas["run.schema"], format_checker=jsonschema.FormatChecker())
    with (root / "latency_samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for case in run["cases"]:
        jsonschema.validate(case, schemas["benchmark_case.schema"])
        onnx.checker.check_model(str(root / case["model_path"]))
        assert case["accuracy"]["verdict"] == "PASS"
        assert case["selected_providers"] == ["CPUExecutionProvider"]
        samples = [float(r["latency_ms"]) for r in rows if r["case_id"] == case["case_id"]]
        assert len(samples) == 10  # Three warmups are excluded.
        assert samples == case["samples_ms"]
        assert all(np.isfinite(s) and s > 0 for s in samples)
        for p in (50, 95, 99):
            assert case["timing"][f"p{p}_ms"] == pytest.approx(np.percentile(samples, p))
        assert case["timing"]["throughput_inferences_per_second"] == pytest.approx(
            10 / case["timing"]["timed_wall_seconds"]
        )
        evidence = case["profile_evidence"]
        assert evidence["status"] == "target execution verified"
        assert evidence["provider_event_counts"]["CPUExecutionProvider"] > 0
        assert (root / evidence["profile_path"]).exists()
    assert (root / "report.html").is_file()
    text = (root / "report.html").read_text(encoding="utf-8")
    assert "Synthetic models / locally generated inputs" in text
    assert "<svg" in text
    assert "https://" not in text


def test_export_reproducible(cpu_run, tmp_path):
    root, run = cpu_run
    case = run["cases"][0]
    workload = workload_from_case(root, case, run["config"])
    path = tmp_path / "repeat.onnx"
    export_model(workload.model, workload.inputs, path)
    assert file_hash(path) == case["model_sha256"]


def test_safe_fault_matrix(cpu_run, tmp_path):
    root, run = cpu_run
    workload = workload_from_case(root, run["cases"][0], run["config"])
    before = file_hash(workload.path)
    faults = run_faults(workload, run["config"], tmp_path)
    assert len(faults) == 6
    assert all(f["outcome"] == "PASS" and f["cleanup_complete"] for f in faults)
    assert (
        next(f for f in faults if f["fault_id"] == "unavailable_provider")[
            "observed_error_category"
        ]
        == "UNAVAILABLE"
    )
    assert file_hash(workload.path) == before


def test_real_stress_five_seconds_two_workers(tmp_path, schemas):
    root = tmp_path / "stress"
    prepare_output(root)
    cfg = parse_config(
        {
            "models": ["mlp_gemm"],
            "measurement": {"concurrency": 2, "warmup_iterations": 3},
            "stress": {"duration_seconds": 5, "max_samples_per_worker": 100},
        },
        kind="stress",
    )
    run = execute(cfg, root, "stress")
    jsonschema.validate(run, schemas["run.schema"])
    case = run["cases"][0]
    assert case["status"] == "PASS"
    assert case["cleanup_complete"]
    assert case["stability"]["workers"] == 2
    assert case["stability"]["elapsed_seconds"] >= 5
    assert case["stability"]["requests"] > 0
    assert case["stability"]["errors"] == []
    assert 0 < len(case["samples_ms"]) <= 200
    assert case["resources"]["samples"]
    assert case["resources"]["rss_peak_sampled_bytes"] > 0
    assert case["timing"]["throughput_inferences_per_second"] == pytest.approx(
        case["stability"]["requests"] / case["stability"]["elapsed_seconds"]
    )


def test_stress_hard_timeout_cleanup(cpu_run, tmp_path):
    root, run = cpu_run
    cfg = parse_config({"stress": {"timeout_seconds": 0.01}}, kind="stress")
    workload = workload_from_case(root, run["cases"][0], cfg)
    result = measure_stress(workload, np.zeros((1, 64), dtype=np.float32), cfg, tmp_path)
    assert result["status"] == "ERROR"
    assert result["error_category"] == "TIMEOUT"
    assert result["cleanup_complete"]


def test_stress_periodic_correctness_failure(cpu_run, tmp_path):
    root, run = cpu_run
    cfg = parse_config({"stress": {"duration_seconds": 0.2}}, kind="stress")
    workload = workload_from_case(root, run["cases"][0], cfg)
    result = measure_stress(workload, np.full((1, 64), 100, dtype=np.float32), cfg, tmp_path)
    assert result["status"] == "FAIL"
    assert result["stability"]["errors"][0]["category"] == "CORRECTNESS"
    assert result["cleanup_complete"]


@pytest.mark.parametrize("required,status", [(True, "UNAVAILABLE"), (False, "SKIP")])
def test_unavailable_records_no_benchmark(cpu_run, tmp_path, monkeypatch, required, status):
    root, run = cpu_run
    cfg = parse_config({"backend": {"provider": "CUDAExecutionProvider", "required": required}})
    workload = workload_from_case(root, run["cases"][0], cfg)
    monkeypatch.setattr("aivs.runner.generate", lambda *_: [workload])
    monkeypatch.setattr(
        "aivs.providers.ort.get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    prepare_output(tmp_path / "out")
    result = execute(cfg, tmp_path / "out")
    case = result["cases"][0]
    assert case["status"] == status
    assert case["timing"] is None
    assert case["samples_ms"] == []
    assert case["selected_providers"] == []
    assert result["summary"]["verdict"] == ("ERROR" if required else "PASS")


def test_correctness_failure_blocks_timing(cpu_run, tmp_path, monkeypatch):
    root, run = cpu_run
    cfg = parse_config()
    workload = workload_from_case(root, run["cases"][0], cfg)
    monkeypatch.setattr("aivs.runner.generate", lambda *_: [workload])
    monkeypatch.setattr("aivs.runner.validate", lambda *_: {"verdict": "FAIL"})

    def unexpected_timing(*_):
        raise AssertionError("correctness gate must stop timing")

    monkeypatch.setattr("aivs.runner.measure", unexpected_timing)
    prepare_output(tmp_path / "out")
    case = execute(cfg, tmp_path / "out")["cases"][0]
    assert case["status"] == "FAIL"
    assert case["timing"] is None


def test_simulated_comparison_separated_from_measurements(cpu_run, tmp_path, schemas):
    _, measured = cpu_run
    before = deepcopy(measured)
    report_link = simulated_regression_demo(measured, tmp_path)
    assert measured == before
    run = json.loads((tmp_path / "simulated_comparison/run.json").read_text())
    jsonschema.validate(run, schemas["run.schema"])
    row = run["comparison"]["comparisons"][0]
    assert row["delta_percent"] == 25
    assert row["verdict"] == "REGRESSION"
    assert run["simulated_values"]
    assert "SIMULATED COMPARISON FIXTURE VALUES" in (tmp_path / report_link).read_text(
        encoding="utf-8"
    )


def test_export_legacy_fallback_is_checked(monkeypatch, tmp_path):
    import aivs.exporter as exporter

    original = exporter.torch.onnx.export

    def export(*args, **kwargs):
        if kwargs.get("dynamo"):
            raise RuntimeError("controlled unsupported modern exporter")
        return original(*args, **kwargs)

    export.__signature__ = inspect.signature(original)
    monkeypatch.setattr(exporter.torch.onnx, "export", export)
    metadata = export_model(
        build_model("mlp_gemm", 1), make_input("mlp_gemm", 1, 1), tmp_path / "fallback.onnx"
    )
    assert metadata["exporter"] == "legacy"
    assert metadata["graph_checked"]
    assert "RuntimeError" in metadata["fallback_reason"]
