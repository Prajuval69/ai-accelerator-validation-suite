# Product brief — inference runtime reliability

## Problem
Before releasing an AI-enabled software/hardware product, teams need repeatable evidence that model inference is accurate, performant, stable and failure-resilient on the targeted backend. Comparing deployments requires preserving model hashes and system context.

## User story
As a junior AI/ML validation engineer, I can produce known-good tiny ONNX workloads, run measurable CPU inference, stress the backend safely, inject deliberate invalid conditions, compare runs and share a clear evidence report.

## P0 goals
- Works without GPU or internet once dependencies are installed.
- Understands and distinguishes target backend, loaded provider list, actual node placement evidence and fallback.
- Captures precision, model/input hashes, batch shape, environment, versions and execution context.
- Supports single-run benchmarks, bounded stress tests, fault scenarios, baseline comparisons and static HTML reports.

## Out of scope
- Board-level signal/instrumentation tests, BIOS/PCIe validation, low-level CUDA/HIP kernels, ROCm driver installation, actual hardware diagnostics.
- Public cloud, large LLMs, large downloads, continuous long soak tests by default.
- Synthetic data does not establish task accuracy or real-world business performance.

## Showcase
One `aivs demo` generates real measured CPU results for multiple synthetic ONNX workload types, records a fault outcome and produces a self-contained HTML report. A comparison demo shows an injected regression detected using **synthetic fixture values** clearly separated from real measurements.
