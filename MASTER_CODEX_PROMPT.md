# Master Codex prompt — AI Accelerator Inference Performance & Reliability Validation Suite

Act as a senior AI/ML runtime validation engineer building an end-to-end Python project called `ai-accelerator-validation-suite`. Implement the application, not merely a plan. You may write code, set up environments, run real CPU benchmarks/tests, debug errors and improve the repo autonomously.

## Mission
Create a Linux-first **AI inference benchmarking, stress-testing and reliability validation** platform built around locally generated tiny PyTorch->ONNX models, ONNX Runtime execution providers, performance measurements, fault injection, reproducible results and actionable HTML reports. It must work on a normal CPU-only laptop; NVIDIA CUDA and AMD MIGraphX are optional enhancements only if compatible installed hardware/wheels exist.

## Non-negotiable P0 features
1. Automatically synthesize small MLP, convolution, and matmul/attention workloads with fixed reproducible weights, export to ONNX and validate graphs (no downloading datasets/models).
2. Discover runtime providers; always support CPU, optionally support installed CUDA/MIGraphX. Do not silently claim GPU execution or treat unavailability as a CPU success for a requested GPU run.
3. Warmup + timed runs using high-resolution monotonic clock. Compute per-inference latency p50/p95/p99, arithmetic mean, stddev, and sequential throughput; record methodology and parameters.
4. Repeated/longer stress test, bounded worker concurrency option, timeout isolation, memory sampling and error classification.
5. Numerical output agreement against PyTorch reference or ONNX CPU baseline with correct tolerances; always store precision & actual provider metadata.
6. Robust fault-injection tests for bad shape/dtype, missing/corrupt model file, unavailable provider and deterministic subprocess timeout; never intentionally exhaust RAM or risk machine stability.
7. Baseline-aware regression comparison with version/hardware/model compatibility checks, percentage change calculation, warning vs fail and explicit environmental mismatch classification.
8. Static HTML, machine-readable JSON/CSV reports; reproducible configuration and system provenance.
9. `aivs doctor`, `aivs fixtures`, `aivs demo`, `aivs bench`, `aivs stress`, `aivs compare`, `aivs report`, `aivs faults` CLI per docs.
10. Real tests, GitHub Actions CPU CI, modern `pyproject.toml` package, usable README. Optional INT8 experiment with explicit correctness checks only after P0.

## Hard guardrails
- This is **software-runtime and hardware-oriented validation only**: no physical development board validation, silicon fault diagnosis or GPU kernel implementation without hardware.
- ONNX Runtime removed the legacy ROCm EP in v1.23. For AMD acceleration probe `MIGraphXExecutionProvider` and document platform compatibility; do not depend on legacy `ROCMExecutionProvider` or invent HIP/CUDA development credentials.
- CPU baseline required. Optional GPU behavior must report unavailable/no test, not substitute CPU while claiming GPU success. Use ORT profiling to expose provider placement where possible; list actual selected providers and potential fallback.
- Measure actual durations. Never inject canned results or fabricated speedups. Timings from shared CI are informational only, not quality gates.
- Avoid flaky hardcoded throughput assertions. Regression detection uses controlled saved fixtures in tests, not live CI performance thresholds.
- Quantization may not be supported or improve latency in every model/EP; treat as an experiment and compare numerical fidelity.

## Work sequence
Read all specs. Implement M0 packaging and CLI; M1 fixtures and provider checks; M2 validation+benchmark; M3 stress+faults; M4 compare+report; M5 tests/CI/doc; optional M6 quantization/profile. Keep every stage runnable. Do not stop after planning. Do not ask for keys, external resources, Docker or GPU hardware.

## Required end-of-work verification
Attempt `python -m pip install -e '.[dev]'`, `ruff check .`, `pytest -q`, `aivs doctor`, `aivs demo --out artifacts/demo`, `aivs bench --config config/benchmark.cpu.yaml --out artifacts/cpu_run`, `aivs report --run artifacts/cpu_run`, and deterministic fault tests. Ensure reports reflect actually measured values. Summarize exact commands executed, pass/fail results, artifact paths and unsupported optional items.
