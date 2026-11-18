"""Capability-adaptive ONNX export, always followed by graph checking."""

import importlib.util
import inspect
import warnings
from pathlib import Path

import onnx
import torch


def export_model(model, example, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    modern = (
        "dynamo" in inspect.signature(torch.onnx.export).parameters
        and importlib.util.find_spec("onnxscript") is not None
    )
    fallback_reason = None
    if modern:
        try:
            program = torch.onnx.export(
                model,
                (torch.from_numpy(example),),
                dynamo=True,
                input_names=["input"],
                output_names=["output"],
                opset_version=18,
                verbose=False,
            )
            program.save(str(path), external_data=False)
        except Exception as exc:
            fallback_reason = f"modern exporter unavailable: {type(exc).__name__}"
            modern = False
    if not modern:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                model,
                torch.from_numpy(example),
                str(path),
                dynamo=False,
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
            )
    onnx.checker.check_model(str(path), full_check=True)
    return {
        "exporter": "dynamo" if modern else "legacy",
        "fallback_reason": fallback_reason,
        "graph_checked": True,
        "dynamic_batch": False,
    }
