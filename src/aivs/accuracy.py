import numpy as np
import torch


def check_outputs(reference, actual, atol=1e-4, rtol=1e-3) -> dict:
    same_shape = reference.shape == actual.shape
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    passed = same_shape and finite and bool(np.allclose(reference, actual, atol=atol, rtol=rtol))
    return {
        "verdict": "PASS" if passed else "FAIL",
        "atol": atol,
        "rtol": rtol,
        "reference": "PyTorch CPU float32",
        "finite": finite,
        "max_absolute_error": float(np.max(np.abs(reference - actual)))
        if same_shape and finite and reference.size
        else None,
        "reason": None if passed else "nonfinite output, shape or tolerance mismatch",
    }


def validate(workload, session, cfg):
    with torch.inference_mode():
        reference = workload.model(torch.from_numpy(workload.inputs)).numpy()
    actual = session.run(None, {"input": workload.inputs})[0]
    return check_outputs(reference, actual, cfg["accuracy"]["atol"], cfg["accuracy"]["rtol"])
