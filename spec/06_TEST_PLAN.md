# Deterministic QA plan

## Unit tests
- Percentile p50/p95/p99 from known arrays using `numpy.percentile`; ms conversion; sequential throughput = runs / elapsed seconds; correct zero/empty sample handling.
- Comparison: same env/hashes -> comparable, different model hash/batch/provider/version/hardware -> incompatible, positive percentage delta and threshold boundary -> PASS/REGRESSION, zero baseline -> explicitly invalid.
- Provider selection: missing requested CUDA/MIGraphX => UNAVAILABLE and no CPU benchmark mislabeled. Stub mocked provider list/session for unit testing.
- Config rejects bad iterations, negative duration, unbounded concurrency, unsupported precision, unknown config keys.
- Fault classification expected vs unexpected; timeout uses isolated subprocess and reliable cleanup. Never simulate memory exhaustion.
- Reports/JSON handle null metrics and suppress NaN/Inf.

## Real CPU integration
- 3 small models export to ONNX and run through CPU ORT; correctness matches PyTorch with documented tolerance.
- Warmup excluded from latency sample count. Real timed samples saved; percentiles recompute from saved CSV.
- Smoke `bench` with 3 warmups/10 iterations/1 concurrency completes on CPU.
- Smoke `stress` with 5-second duration /2 workers and bounded sample count exits cleanly.
- Faults: invalid dtype, wrong shape, missing model path, invalid provider and deterministic timeout expected/observed statuses.
- Reporting works offline and can regenerate from `run.json` without rerunning model.

## Avoid flaky testing
- No assertions like 'CPU must achieve >N fps' or 'GPU must be faster'.
- Bench precision assertions use known synthetic sample arrays; real CPU tests check positive finite times, sample counts, artifact schemas.
- CI does not require a GPU, nor require a given throughput. Optional GPU tests skipped with explicit reason.

## CI
GitHub Actions Ubuntu Python 3.11; dependency install, Ruff, pytest, CPU demo and schema validation. Bound run durations and resource usage so CI does not exceed reasonable minutes.
