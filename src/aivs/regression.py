"""Compatibility precedes percent change; deterministic fixtures are labeled simulated."""

import math
from copy import deepcopy
from pathlib import Path

from aivs.fixtures import file_hash
from aivs.storage import load_run, new_run, save_run, write_json

ENV_FIELDS = (
    "platform",
    "os_release",
    "architecture",
    "cpu",
    "logical_cores",
    "physical_cores",
    "total_ram_bytes",
    "python",
    "torch",
    "onnx",
    "onnxruntime",
    "numpy",
    "onnxscript",
)
CASE_FIELDS = (
    "model_id",
    "model_sha256",
    "input_sha256",
    "input_shape",
    "batch",
    "precision",
    "requested_provider",
    "selected_providers",
    "session_settings",
)


def compare_runs(
    baseline, current, fail_percent=15, warning_percent=5, allow_environment_drift=False
):
    rows = []
    baseline_cases = {c["case_id"]: c for c in baseline["cases"]}
    current_cases = {c["case_id"]: c for c in current["cases"]}
    if len(baseline_cases) != len(baseline["cases"]) or len(current_cases) != len(current["cases"]):
        raise ValueError("duplicate case IDs cannot be compared")
    env_diff = [
        {
            "field": f"environment.{key}",
            "baseline": baseline["environment"].get(key),
            "current": current["environment"].get(key),
        }
        for key in ENV_FIELDS
        if baseline["environment"].get(key) != current["environment"].get(key)
    ]
    missing_env = [
        key
        for key in ENV_FIELDS
        if key not in ("onnxscript", "physical_cores")
        and (
            baseline["environment"].get(key) in (None, "", "unknown")
            or current["environment"].get(key) in (None, "", "unknown")
        )
    ]
    # Timing/profiling and stress policies change measurement methodology.
    context_diff = []
    for key in ("kind", "measurement_origin"):
        if baseline.get(key) != current.get(key):
            context_diff.append(
                {"field": key, "baseline": baseline.get(key), "current": current.get(key)}
            )
    for key in ("measurement", "stress", "accuracy"):
        if baseline["config"].get(key) != current["config"].get(key):
            context_diff.append(
                {
                    "field": f"config.{key}",
                    "baseline": baseline["config"].get(key),
                    "current": current["config"].get(key),
                }
            )
    for case_id in sorted(set(baseline_cases) | set(current_cases)):
        before, after = baseline_cases.get(case_id), current_cases.get(case_id)
        row = {
            "case_id": case_id,
            "verdict": "INCOMPARABLE",
            "changed_fields": [],
            "delta_percent": None,
            "baseline_p95_ms": None,
            "current_p95_ms": None,
            "fail_percent": fail_percent,
            "warning_percent": warning_percent,
            "comparison_confidence": "strict",
            "warning": None,
        }
        if before is None or after is None:
            row["reason"] = "case missing from baseline or current"
            rows.append(row)
            continue
        differences = [
            {"field": key, "baseline": before.get(key), "current": after.get(key)}
            for key in CASE_FIELDS
            if before.get(key) != after.get(key)
        ]
        differences.extend(context_diff)
        row["changed_fields"] = differences + env_diff
        missing_case = [key for key in CASE_FIELDS if not before.get(key) or not after.get(key)]
        missing_case += [
            f"config.{key}"
            for key in ("measurement", "stress", "accuracy")
            if not baseline["config"].get(key) or not current["config"].get(key)
        ]
        if not baseline.get("measurement_origin") or not current.get("measurement_origin"):
            missing_case.append("measurement_origin")
        if missing_case or missing_env:
            row["reason"] = "missing provenance: " + ", ".join(missing_case + missing_env)
        elif differences or (env_diff and not allow_environment_drift):
            row["reason"] = "model, execution methodology or environment mismatch"
        elif before["status"] != "PASS" or after["status"] != "PASS":
            row["reason"] = "both cases must pass inference correctness and execution"
        elif before.get("accuracy", {}).get("verdict") != "PASS" or (
            after.get("accuracy", {}).get("verdict") != "PASS"
        ):
            row["reason"] = "both runs require validated outputs"
        else:
            previous = (before.get("timing") or {}).get("p95_ms")
            now = (after.get("timing") or {}).get("p95_ms")
            row.update(baseline_p95_ms=previous, current_p95_ms=now)
            if (
                type(previous) not in (int, float)
                or type(now) not in (int, float)
                or not math.isfinite(previous)
                or not math.isfinite(now)
                or previous <= 0
                or now <= 0
            ):
                row["verdict"], row["reason"] = "INVALID", "p95 values must be finite and positive"
            else:
                delta = (now - previous) / previous * 100
                row.update(
                    delta_percent=delta,
                    verdict="REGRESSION"
                    if delta >= fail_percent
                    else "WARNING"
                    if delta >= warning_percent
                    else "PASS",
                )
                if env_diff:
                    row.update(
                        comparison_confidence="low",
                        warning="SEVERE: environment drift "
                        "override; timing change is confounded by changed hardware/versions",
                    )
        rows.append(row)
    verdict = (
        "FAIL"
        if any(r["verdict"] in ("REGRESSION", "INVALID", "INCOMPARABLE") for r in rows) or not rows
        else "WARNING"
        if any(r["verdict"] == "WARNING" for r in rows)
        else "PASS"
    )
    return {
        "verdict": verdict,
        "comparisons": rows,
        "note": "p95 change is descriptive; no statistical significance inferred",
    }


def comparison_artifact(
    baseline_path: Path,
    current_path: Path,
    out: Path,
    allow_environment_drift=False,
    fail_percent=15,
    warning_percent=5,
):
    baseline, bfile = load_run(baseline_path)
    current, cfile = load_run(current_path)
    run = new_run("comparison", current["config"])
    result = compare_runs(baseline, current, fail_percent, warning_percent, allow_environment_drift)
    run.update(
        comparison=result,
        sources={
            "baseline_sha256": file_hash(bfile),
            "current_sha256": file_hash(cfile),
            "baseline_run_id": baseline["run_id"],
            "current_run_id": current["run_id"],
        },
        measurement_origin="saved_record_comparison",
        environment=current["environment"],
    )
    run["summary"]["verdict"] = result["verdict"]
    run["simulated_values"] = (
        baseline.get("measurement_origin") == "simulated_fixture"
        or current.get("measurement_origin") == "simulated_fixture"
    )
    return save_run(out, run)


def simulated_regression_demo(measured, root: Path):
    successful = [c for c in measured["cases"] if c["status"] == "PASS" and c.get("timing")]
    if not successful:
        return None
    folder = root / "simulated_comparison"
    folder.mkdir()
    baseline = deepcopy(measured)
    baseline.update(
        kind="benchmark",
        measurement_origin="simulated_fixture",
        simulated_values=True,
        cases=[deepcopy(successful[0])],
        faults=[],
        methodology="SIMULATED comparator fixture; values are not measurements",
    )
    baseline["cases"][0]["timing"]["p95_ms"] = 1.0
    baseline["cases"][0]["samples_ms"] = []
    current = deepcopy(baseline)
    current["cases"][0]["timing"]["p95_ms"] = 1.25
    # Only the intentionally set p95 participates in this comparator fixture.
    for case in (baseline["cases"][0], current["cases"][0]):
        case["timing"] = {"p95_ms": case["timing"]["p95_ms"], "simulated": True}
    write_json(folder / "baseline.json", baseline)
    write_json(folder / "current.json", current)
    (folder / "logs").mkdir()
    comparison_artifact(folder / "baseline.json", folder / "current.json", folder)
    return "simulated_comparison/report.html"
