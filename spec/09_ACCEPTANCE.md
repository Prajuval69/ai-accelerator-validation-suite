# Definition of done

## Required P0
- [ ] CPU-only install and `aivs --help` work from package.
- [ ] 3 deterministic synthetic ONNX workloads exported and graph-checked.
- [ ] Provider discovery and mandatory provider behavior are accurate; absent GPU not falsely shown as tested.
- [ ] Real PyTorch-vs-ORT float32 inference correctness gate.
- [ ] Warmup excluded; p50/p95/p99/mean/stddev/throughput produced from actual saved samples.
- [ ] Repeated stress test runs within configured bounds; reports memory/CPU stats if collected.
- [ ] All safe expected fault scenarios recorded without destroying system resources.
- [ ] Baseline comparator detects a seeded simulated +25% regression and rejects incomparable environments.
- [ ] `doctor`, `fixtures`, `demo`, `bench`, `stress`, `faults`, `compare`, `report` commands work.
- [ ] Self-contained HTML + reproducible machine-readable JSON/CSV with version and model provenance.
- [ ] Robust unit/integration/CLI tests, `ruff`, CPU GitHub Actions, README, MIT license.
- [ ] Claims limited to runtime/software validation; no fabricated benchmarks or hardware validation.

## Optional P1 (only after P0)
- [ ] CPU quantization experiment with correctness report and honest speed comparison.
- [ ] ONNX Runtime operator profiler and actual execution-provider placement visibility.
- [ ] Optional CUDA or MIGraphX runs when hardware available, with honest skip behavior otherwise.
- [ ] Result explorer sorting/filtering charts using static inline SVG/JS.

## Handoff requirements
Include exact test commands and observed results, generated report location, which optional provider features were skipped, limits of methodology, and next steps for real accelerator-board extension.
