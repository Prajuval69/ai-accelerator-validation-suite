"""Deliberate safe failures; rejection is successful fault validation, not inference."""

import subprocess
import sys
import tempfile
from pathlib import Path

from aivs.diagnostics import error_message
from aivs.providers import ProviderUnavailable, create_session


def timeout_probe(timeout_seconds=0.15):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
    return {
        "fault_id": "timeout",
        "expected": "EXPECTED_TIMEOUT",
        "observed_error_category": "EXPECTED_TIMEOUT" if timed_out else "NO_ERROR",
        "outcome": "PASS" if timed_out else "FAIL",
        "observed_error": "child deadline exceeded" if timed_out else None,
        "cleanup_complete": process.poll() is not None,
    }


def classify(fault_id, expected, operation, accepted_exceptions):
    category, message = "NO_ERROR", None
    try:
        operation()
    except accepted_exceptions as exc:
        category = expected
        message = f"{type(exc).__name__}: {error_message(exc)}"
    except Exception as exc:
        category, message = "UNEXPECTED_ERROR", type(exc).__name__
    return {
        "fault_id": fault_id,
        "expected": expected,
        "observed_error_category": category,
        "observed_error": message,
        "outcome": "PASS" if category == expected else "FAIL",
        "cleanup_complete": True,
    }


def run_faults(workload, cfg, root):
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        Fail,
        InvalidArgument,
        InvalidProtobuf,
        NoSuchFile,
    )

    session, observation = create_session(workload.path, cfg, root, "fault_driver")
    try:
        records = [
            classify(
                "bad_dtype",
                "EXPECTED_REJECTION",
                lambda: session.run(None, {"input": workload.inputs.astype("float64")}),
                (InvalidArgument,),
            ),
            classify(
                "bad_shape",
                "EXPECTED_REJECTION",
                lambda: session.run(None, {"input": workload.inputs.reshape(-1)}),
                (InvalidArgument,),
            ),
        ]
        with tempfile.TemporaryDirectory(prefix="aivs_fault_") as folder:
            temp = Path(folder)
            corrupt = temp / "corrupt.onnx"
            corrupt.write_bytes(b"deliberately corrupt ONNX")

            def load(path):
                model_session, info = create_session(path, cfg, root, "fault_model")
                if info["profiling_enabled"]:
                    model_session.end_profiling()

            records.extend(
                [
                    classify(
                        "missing_file",
                        "EXPECTED_REJECTION",
                        lambda: load(temp / "missing.onnx"),
                        (NoSuchFile,),
                    ),
                    classify(
                        "corrupt_file",
                        "EXPECTED_REJECTION",
                        lambda: load(corrupt),
                        (InvalidProtobuf, Fail),
                    ),
                ]
            )
        sentinel_cfg = {
            **cfg,
            "backend": {**cfg["backend"], "provider": "AIVS_Sentinel_UnavailableProvider"},
        }
        records.append(
            classify(
                "unavailable_provider",
                "UNAVAILABLE",
                lambda: create_session(workload.path, sentinel_cfg, root, "absent"),
                (ProviderUnavailable,),
            )
        )
        records.append(timeout_probe())
        return records
    finally:
        if observation["profiling_enabled"]:
            session.end_profiling()
