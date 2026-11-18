# Career presentation and demo

## 30-second elevator pitch
'I built a CPU-first platform to validate ONNX inference workloads before release. It exports small PyTorch models, verifies numerical correctness, collects real warm-up-separated latency measurements, stresses runtime execution safely, classifies injected faults and flags comparable performance regressions. It can optionally use a supported GPU execution provider when hardware exists.'

## Demo sequence
1. `aivs doctor`: inspect provider presence and CPU environment.
2. `aivs demo`: open report with actual real CPU samples and methodology.
3. Show stored per-iteration samples and reproducible synthetic model checksums.
4. Show one controlled timeout/invalid-input fault and its expected outcome.
5. Demonstrate simulated regression comparison; clearly say numbers are fixture-based, not measured real speedup.
6. Explain extension path: cross-compile test agent, collect board telemetry, run identical ONNX fixture matrix on reference board; not implemented.

## Interview questions
- What's the difference between a provider being available vs a node actually running on it?
- How do warm-up and thread count bias benchmarks?
- What is p95 latency, and why use percentile distributions rather than mean only?
- How would you isolate a hanging inference test on Linux?
- Why can't you compare two benchmark runs from different machines without caveats?
- Why MIGraphX rather than legacy ROCm EP in recent ONNX Runtime builds?

## Safe resume bullets (use only after work completes)
- Engineered a Linux-first ONNX inference validation suite with PyTorch-generated workloads, correctness checks, measured latency/throughput, and structured benchmark reporting.
- Implemented deterministic runtime fault injection, bounded stress tests, and environment-aware baseline regression checks with `pytest`/GitHub Actions automation.
