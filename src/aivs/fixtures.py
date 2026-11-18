from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from aivs.exporter import export_model
from aivs.models import build_model, make_input


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass
class WorkloadSpec:
    metadata: dict
    model: object
    inputs: np.ndarray
    path: Path


def generate(cfg: dict, root: Path) -> list[WorkloadSpec]:
    result = []
    for model_id in cfg["models"]:
        model = build_model(model_id, cfg["seed"])
        for batch in cfg["input"]["batch_sizes"]:
            inputs = make_input(
                model_id, batch, cfg["seed"], cfg["measurement"]["max_input_megabytes"]
            )
            name = f"{model_id}_b{batch}"
            path = root / "fixtures" / f"{name}.onnx"
            export_info = export_model(model, inputs, path)
            metadata = {
                "case_id": name,
                "model_id": model_id,
                "batch": batch,
                "seed": cfg["seed"],
                "precision": "float32",
                "model_path": path.relative_to(root).as_posix(),
                "model_sha256": file_hash(path),
                "input_sha256": sha256(inputs.tobytes()).hexdigest(),
                "input_shape": list(inputs.shape),
                "export": export_info,
            }
            result.append(WorkloadSpec(metadata, model, inputs, path))
    return result
