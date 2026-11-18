# CLI and configuration

Executable: `aivs` (`python -m aivs` encouraged). Support commands with predictable help:

```bash
aivs --help
aivs doctor
aivs fixtures --out artifacts/fixtures
aivs demo --out artifacts/demo
aivs bench --config config/benchmark.cpu.yaml --out artifacts/run1
aivs stress --config config/stress.cpu.yaml --out artifacts/stress1
aivs faults --out artifacts/faults1
aivs compare --baseline artifacts/run1 --current artifacts/run2 --out artifacts/compare12
aivs report --run artifacts/run1
```

`fixtures` exports/checks three ONNX models. `demo` performs a quick bounded CPU benchmark and fault demo and renders a report. `bench` warms up, validates correctness, records timings. `stress` tests repeated concurrent runs safely. `faults` runs controlled invalid cases and catches them as expected. `compare` loads saved run records, checks environmental comparability and gives regression verdict. `report` re-renders without re-running inference.

Configs in `config/*.yaml` define actual behavior; implement validated schema with useful errors. Never use default unbounded concurrency/time.

Run artifact layout:
```
artifacts/run1/
  run.json
  cases.csv
  latency_samples.csv
  report.html
  fixtures/*.onnx
  logs/run.log
  profiles/*.json  # when profiling enabled and supported
```

Keep output consistent with `spec/schemas/run.schema.json`, `spec/schemas/benchmark_case.schema.json`; schema_version=1, valid JSON (no NaN/Inf), relative paths. Actual run values should be measured; example fixture for comparison must be clearly tagged simulated.

Exit codes: 0 run completed with expected/optional skips; 1 correctness/regression FAIL; 2 invalid CLI/config; 3 unexpected runtime/system failures. For absent explicitly requested mandatory provider, exit 3 with `UNAVAILABLE` record and helpful diagnostics.
