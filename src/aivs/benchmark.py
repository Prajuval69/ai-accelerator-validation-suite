"""Synchronous wall time measurement; warmup and validation are excluded."""

import time

import numpy as np


def summarize(samples_ms: list[float], elapsed_seconds: float) -> dict | None:
    if not samples_ms:
        return None
    if elapsed_seconds <= 0 or not np.isfinite(elapsed_seconds):
        raise ValueError("elapsed_seconds must be finite and positive")
    samples = np.asarray(samples_ms)
    if not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("samples must be finite and positive")
    return {
        "sample_count": len(samples_ms),
        "mean_ms": float(samples.mean()),
        "stddev_ms": float(samples.std()),
        **{f"p{p}_ms": float(np.percentile(samples, p)) for p in (50, 95, 99)},
        "timed_wall_seconds": elapsed_seconds,
        "throughput_inferences_per_second": len(samples_ms) / elapsed_seconds,
    }


def measure(session, inputs, cfg):
    feed = {"input": inputs}
    measurement = cfg["measurement"]
    for _ in range(measurement["warmup_iterations"]):
        session.run(None, feed)
    samples = []
    start = time.perf_counter_ns()
    for _ in range(measurement["timed_iterations"]):
        before = time.perf_counter_ns()
        session.run(None, feed)
        samples.append((time.perf_counter_ns() - before) / 1e6)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return samples, {
        **summarize(samples, elapsed),
        "warmup_iterations": measurement["warmup_iterations"],
        "method": "sequential synchronous session.run; perf_counter_ns",
        "concurrency": 1,
    }
