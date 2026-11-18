"""Strict, bounded configuration shared by CLI and worker processes."""

from copy import deepcopy
from pathlib import Path

import yaml

MODELS = ("mlp_gemm", "conv2d", "attention_matmul")
PROVIDERS = ("CPUExecutionProvider", "CUDAExecutionProvider", "MIGraphXExecutionProvider")
DEFAULTS = {
    "schema_version": 1,
    "seed": 2026,
    "models": list(MODELS),
    "input": {"batch_sizes": [1], "dtype": "float32"},
    "backend": {
        "provider": PROVIDERS[0],
        "required": True,
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
    },
    "measurement": {
        "warmup_iterations": 5,
        "timed_iterations": 30,
        "concurrency": 1,
        "max_input_megabytes": 128,
        "enable_ort_profiling": False,
    },
    "stress": {
        "duration_seconds": 10,
        "max_duration_seconds": 120,
        "max_concurrency": 4,
        "memory_sample_interval_seconds": 0.25,
        "timeout_seconds": 15,
        "max_samples_per_worker": 10000,
    },
    "accuracy": {"enabled": True, "block_on_failure": True, "atol": 1e-4, "rtol": 1e-3},
    "regression": {"fail_percent": 15, "warning_percent": 5},
    "report": {"html": True, "csv": True, "json": True},
}


class ConfigError(ValueError):
    pass


def parse_config(data: dict | None = None, *, kind: str = "benchmark") -> dict:
    cfg = deepcopy(DEFAULTS)
    if kind == "stress":
        cfg["measurement"]["concurrency"] = 2
    if data is not None:
        if not isinstance(data, dict):
            raise ConfigError("configuration must be a YAML mapping")
        for key, value in data.items():
            if key not in cfg:
                raise ConfigError(f"unknown configuration key: {key}")
            if isinstance(cfg[key], dict):
                if not isinstance(value, dict) or set(value) - set(cfg[key]):
                    raise ConfigError(f"unknown keys or invalid mapping in {key}")
                cfg[key].update(value)
            else:
                cfg[key] = value

    def number(section, key, low, high, integer=False):
        val = cfg[section][key] if section else cfg[key]
        typ = (int,) if integer else (int, float)
        if isinstance(val, bool) or not isinstance(val, typ) or not low <= val <= high:
            raise ConfigError(f"{section or 'config'}.{key} must be in [{low}, {high}]")

    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    number(None, "seed", 0, 2**32 - 1, True)
    if (
        not isinstance(cfg["models"], list)
        or not cfg["models"]
        or any(m not in MODELS for m in cfg["models"])
        or len(set(cfg["models"])) != len(cfg["models"])
    ):
        raise ConfigError(f"models must be a unique nonempty subset of {MODELS}")
    batches = cfg["input"]["batch_sizes"]
    if (
        not isinstance(batches, list)
        or not batches
        or len(batches) > 16
        or any(type(b) is not int or not 1 <= b <= 64 for b in batches)
        or len(set(batches)) != len(batches)
    ):
        raise ConfigError("batch_sizes must contain unique integers from 1 to 64")
    if cfg["input"]["dtype"] != "float32":
        raise ConfigError("only verified float32 precision is supported")
    if cfg["backend"]["provider"] not in PROVIDERS:
        raise ConfigError(f"provider must be one of {PROVIDERS}")
    for section, key in [
        ("backend", "required"),
        ("measurement", "enable_ort_profiling"),
        ("accuracy", "enabled"),
        ("accuracy", "block_on_failure"),
    ]:
        if type(cfg[section][key]) is not bool:
            raise ConfigError(f"{section}.{key} must be boolean")
    for key in cfg["report"]:
        if cfg["report"][key] is not True:
            raise ConfigError("P0 reports require html, csv and json enabled")
    for key in ("intra_op_num_threads", "inter_op_num_threads"):
        number("backend", key, 1, 16, True)
    number("measurement", "warmup_iterations", 0, 1000, True)
    number("measurement", "timed_iterations", 1, 100000, True)
    number("measurement", "concurrency", 1, 4, True)
    number("measurement", "max_input_megabytes", 0.001, 128)
    number("stress", "max_duration_seconds", 0.01, 120)
    number("stress", "duration_seconds", 0.01, cfg["stress"]["max_duration_seconds"])
    number("stress", "max_concurrency", 1, 4, True)
    number("stress", "memory_sample_interval_seconds", 0.01, 10)
    number("stress", "timeout_seconds", 0.01, 180)
    number("stress", "max_samples_per_worker", 1, 100000, True)
    if cfg["measurement"]["concurrency"] > cfg["stress"]["max_concurrency"]:
        raise ConfigError("concurrency exceeds stress.max_concurrency")
    if kind != "stress" and cfg["measurement"]["concurrency"] != 1:
        raise ConfigError("latency benchmarks require concurrency=1; use stress for workers")
    for key in ("atol", "rtol"):
        number("accuracy", key, 0, 1)
    for key in ("fail_percent", "warning_percent"):
        number("regression", key, 0, 1000)
    if cfg["regression"]["warning_percent"] > cfg["regression"]["fail_percent"]:
        raise ConfigError("warning_percent must not exceed fail_percent")
    return cfg


def load_config(path: Path | None, *, kind="benchmark") -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path else None
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config: {exc}") from exc
    return parse_config(data, kind=kind)
