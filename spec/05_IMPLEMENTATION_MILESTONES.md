# Milestones (do not stop after writing these)

**M0: scaffolding**: package, CLI/help, config parser, pytest/ruff/CI. Gate CLI command names and parser unit tests.

**M1: models/fixtures**: deterministic 3-model registry, ONNX exports/checker, CLI fixtures command, checksum metadata. Gate 3 valid ONNX files generated in one command on CPU.

**M2: provider/accuracy/benchmark**: capability probes, ORT CPU, PyTorch correctness gate, monotonic samples, summary stats, CSV/JSON. Gate 3 workloads with measured metrics and no claimed GPU.

**M3: stress and faults**: bounded 1/2/4 concurrency, safe repeated workloads, subprocess timeout control, invalid input/model/provider cases, outcome semantics. Gate expected faults classified; worker cleaned after timeout; no machine-killing tests.

**M4: comparison and reporting**: baseline loader, like-for-like compatibility check, delta%, regression threshold; static HTML with environment, case table and individual samples. Gate baseline compare tests, report regeneration and clear negative demo.

**M5: reliability engineering**: tests, docs, pipeline, portable packaging, resource limit checks, reproducible sample screenshots instructions. Gate CPU CI and CLI/demo run.

**M6: optional enhancements**: quantization experiments (with numeric fidelity), actual ORT profiler placement, optional compatible CUDA/MIGraphX. Do not block P0 on M6.
