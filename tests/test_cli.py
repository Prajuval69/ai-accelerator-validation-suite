import json
import subprocess
import sys
from pathlib import Path

import pytest

from aivs.cli import main


@pytest.mark.parametrize(
    "command",
    [None, "doctor", "fixtures", "demo", "bench", "stress", "faults", "compare", "report"],
)
def test_help(command):
    args = [sys.executable, "-m", "aivs"] + ([command] if command else []) + ["--help"]
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_report_does_not_rerun_inference(cpu_run, monkeypatch):
    root, _ = cpu_run

    def forbid(*args, **kwargs):
        raise AssertionError("report must use saved records only")

    monkeypatch.setattr("aivs.runner.execute", forbid)
    monkeypatch.setattr("aivs.runner.generate", forbid)
    assert main(["report", "--run", str(root)]) == 0


def test_invalid_config_exit_code(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("measurement:\n  timed_iterations: -1\n")
    assert main(["bench", "--config", str(cfg), "--out", str(tmp_path / "out")]) == 2
    cfg.write_text("[invalid, config]")
    assert main(["bench", "--config", str(cfg), "--out", str(tmp_path / "out")]) == 2


def test_actual_demo_subprocess(tmp_path):
    out = tmp_path / "demo"
    result = subprocess.run(
        [sys.executable, "-m", "aivs", "demo", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run = json.loads((out / "run.json").read_text())
    assert len(run["cases"]) == 3
    assert run["summary"]["correctness_passes"] == 3
    assert run["summary"]["expected_fault_passes"] == 6
    assert run["measurement_origin"] == "actual_local_execution"
    assert (out / "simulated_comparison/report.html").exists()


def test_compare_exit_and_drift(cpu_run, tmp_path):
    root, _ = cpu_run
    assert (
        main(
            [
                "compare",
                "--baseline",
                str(root),
                "--current",
                str(root),
                "--out",
                str(tmp_path / "same"),
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "same/run.json").read_text())
    assert result["summary"]["verdict"] == "PASS"
    assert (
        main(
            [
                "compare",
                "--baseline",
                str(root),
                "--current",
                str(root),
                "--out",
                str(tmp_path / "bad"),
                "--fail-percent",
                "nan",
            ]
        )
        == 2
    )


def test_config_examples_parse():
    from aivs.config import load_config

    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "config/benchmark.cpu.yaml")["measurement"]["timed_iterations"] == 30
    assert (
        load_config(root / "config/stress.cpu.yaml", kind="stress")["stress"]["duration_seconds"]
        == 10
    )
