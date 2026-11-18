"""CLI-to-artifacts orchestration with explicit per-case failures."""

from pathlib import Path

import psutil
import torch

from aivs.accuracy import validate
from aivs.benchmark import measure
from aivs.diagnostics import error_message, error_trace
from aivs.faults import run_faults
from aivs.fixtures import generate
from aivs.providers import ProviderUnavailable, create_session, finish_profile
from aivs.regression import simulated_regression_demo
from aivs.storage import new_run, save_run
from aivs.stress import measure_stress


def execute(cfg: dict, root: Path, kind="benchmark") -> dict:
    torch.set_num_threads(cfg["backend"]["intra_op_num_threads"])
    run = new_run(kind, cfg)
    workloads = generate(cfg, root)
    for workload in workloads:
        case = {
            **workload.metadata,
            "requested_provider": cfg["backend"]["provider"],
            "selected_providers": [],
            "status": "ERROR",
            "timing": None,
            "accuracy": {"verdict": "NOT_RUN"},
            "profile_evidence": None,
            "samples_ms": [],
        }
        session = observation = None
        try:
            session, observation = create_session(workload.path, cfg, root, case["case_id"])
            case.update(observation)
            if cfg["accuracy"]["enabled"]:
                case["accuracy"] = validate(workload, session, cfg)
            else:
                case["accuracy"] = {"verdict": "SKIP", "reason": "disabled by configuration"}
            if case["accuracy"]["verdict"] == "FAIL" and cfg["accuracy"]["block_on_failure"]:
                case["status"] = "FAIL"
            elif kind == "stress":
                with torch.inference_mode():
                    reference = workload.model(torch.from_numpy(workload.inputs)).numpy()
                # Close parent profiling before child takes over execution evidence.
                case["profile_evidence"] = finish_profile(session, observation, root)
                observation = None
                case.update(measure_stress(workload, reference, cfg, root))
                if case["accuracy"]["verdict"] == "FAIL" and case["status"] == "PASS":
                    case["status"] = "FAIL"
            elif kind == "faults":
                case["status"] = "PASS" if case["accuracy"]["verdict"] != "FAIL" else "FAIL"
            else:
                rss_start = psutil.Process().memory_info().rss
                case["samples_ms"], case["timing"] = measure(session, workload.inputs, cfg)
                case["status"] = "PASS" if case["accuracy"]["verdict"] != "FAIL" else "FAIL"
                case["resources"] = {
                    "rss_start_bytes": rss_start,
                    "rss_end_bytes": psutil.Process().memory_info().rss,
                    "rss_peak_sampled_bytes": None,
                    "note": "boundary snapshots only; no peak sampler in benchmark",
                }
        except ProviderUnavailable as exc:
            case.update(
                status="UNAVAILABLE" if cfg["backend"]["required"] else "SKIP",
                error_category="UNAVAILABLE",
                error=str(exc),
            )
        except Exception as exc:
            case.update(
                status="ERROR",
                error_category=type(exc).__name__,
                error=error_message(exc),
                sanitized_trace=error_trace(exc),
            )
        finally:
            if session is not None and observation is not None:
                case["profile_evidence"] = finish_profile(session, observation, root)
        run["cases"].append(case)
    if kind in ("demo", "faults"):
        cpu_cfg = {**cfg, "backend": {**cfg["backend"], "provider": "CPUExecutionProvider"}}
        run["faults"] = run_faults(workloads[0], cpu_cfg, root)
    if kind == "demo":
        link = simulated_regression_demo(run, root)
        if link:
            run["artifacts"]["simulated_comparison"] = link
    return save_run(root, run)
