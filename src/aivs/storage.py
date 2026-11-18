"""Finite JSON, consistent CSV and atomic run record writes."""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from aivs.system_info import environment


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(val) for val in value]
    return value


def write_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def new_run(kind: str, cfg: dict) -> dict:
    return {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "kind": kind,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_models": True,
        "measurement_origin": "actual_local_execution",
        "environment": environment(),
        "config": cfg,
        "cases": [],
        "faults": [],
        "summary": {},
        "methodology": "Synthetic seeded models/inputs; synchronous session.run; "
        "perf_counter_ns; warmup excluded. Shared systems are noisy. "
        "No production task accuracy or physical board validation.",
        "artifacts": {
            "json": "run.json",
            "cases_csv": "cases.csv",
            "latency_csv": "latency_samples.csv",
            "html": "report.html",
            "log": "logs/run.log",
        },
    }


def prepare_output(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"output directory must be empty: {root.name}")
    (root / "logs").mkdir()


def save_run(root: Path, run: dict):
    counts = {
        status: sum(c["status"] == status for c in run["cases"])
        for status in ("PASS", "FAIL", "ERROR", "SKIP", "UNAVAILABLE")
    }
    prior_status = run["summary"].get("verdict")
    verdict = (
        "ERROR"
        if counts["ERROR"] or counts["UNAVAILABLE"]
        else "FAIL"
        if counts["FAIL"] or any(f["outcome"] != "PASS" for f in run["faults"])
        else "PASS"
    )
    run["summary"].update(
        case_status_counts=counts,
        verdict=prior_status or verdict,
        executed_cases=sum(c.get("timing") is not None for c in run["cases"]),
        correctness_passes=sum(c["accuracy"].get("verdict") == "PASS" for c in run["cases"]),
        expected_fault_passes=sum(f["outcome"] == "PASS" for f in run["faults"]),
    )
    run = finite_json(run)
    write_json(root / "run.json", run)
    fields = [
        "case_id",
        "model_id",
        "status",
        "precision",
        "requested_provider",
        "input_shape",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "stddev_ms",
        "throughput_inferences_per_second",
        "sample_count",
    ]
    with (root / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in run["cases"]:
            row = {**case, **(case.get("timing") or {})}
            writer.writerow({key: row.get(key) for key in fields})
    with (root / "latency_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "sample_index", "latency_ms"])
        for case in run["cases"]:
            writer.writerows(
                (case["case_id"], i, sample) for i, sample in enumerate(case.get("samples_ms", []))
            )
    log = root / "logs" / "run.log"
    log.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": c["case_id"],
                    "status": c["status"],
                    "error_category": c.get("error_category"),
                    "error": c.get("error"),
                    "sanitized_trace": c.get("sanitized_trace"),
                }
            )
            for c in run["cases"]
        )
        + "\n"
        + "\n".join(json.dumps(f) for f in run["faults"]),
        encoding="utf-8",
    )
    from aivs.reporting import render_report

    render_report(run, root / "report.html")
    return run


def load_run(path: Path):
    source = path / "run.json" if path.is_dir() else path
    run = json.loads(
        source.read_text(encoding="utf-8"),
        parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)),
    )
    required = {
        "run_id": str,
        "kind": str,
        "started_at_utc": str,
        "environment": dict,
        "config": dict,
        "cases": list,
        "summary": dict,
    }
    if (
        not isinstance(run, dict)
        or run.get("schema_version") != 1
        or run.get("synthetic_models") is not True
        or any(not isinstance(run.get(key), typ) for key, typ in required.items())
    ):
        raise ValueError("unsupported or invalid run record")
    ids = []
    for case in run["cases"]:
        fields = {
            "case_id": str,
            "model_id": str,
            "model_sha256": str,
            "input_sha256": str,
            "input_shape": list,
            "precision": str,
            "requested_provider": str,
            "selected_providers": list,
            "status": str,
            "accuracy": dict,
        }
        if (
            not isinstance(case, dict)
            or "timing" not in case
            or any(not isinstance(case.get(key), typ) for key, typ in fields.items())
            or case["timing"] is not None
            and not isinstance(case["timing"], dict)
        ):
            raise ValueError("invalid benchmark case record")
        ids.append(case["case_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs in run record")
    return run, source
