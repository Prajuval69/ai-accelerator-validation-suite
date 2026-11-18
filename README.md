# AI Accelerator Validation Suite

A runnable CPU-first Python suite for synthetic ONNX inference performance and reliability
validation. It generates tiny seeded PyTorch models locally, exports and checks their ONNX
graphs, verifies numerical agreement, measures latency, exercises bounded concurrent stress,
classifies safe injected failures, and creates offline HTML/JSON/CSV evidence.

This validates **software runtime behavior**. Synthetic inputs do not establish production task
accuracy. No physical development board testing, silicon diagnosis, CUDA/HIP kernel engineering,
or accelerator speedup is claimed.

## Install

Python 3.11+; no Docker, credentials, pretrained models, datasets, or GPU required. Dependencies
need network access only for installation. All commands then run locally and offline.

```bash
python -m venv .venv
source .venv/bin/activate                     # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
aivs --help
aivs doctor
aivs demo --out artifacts/demo
```

If activation is unavailable, use `.venv/bin/python -m aivs` on Linux or
`.venv/Scripts/python -m aivs` on Windows. The CPU-wheel installation is recommended to avoid
unneeded GPU dependencies. The ordinary editable install also supports CPU execution.

Use a new or empty output directory for each run; existing reports and models are never
overwritten by inference commands. `report` intentionally regenerates the HTML beside saved JSON.

## Commands

```bash
aivs fixtures --out artifacts/fixtures
aivs bench --config config/benchmark.cpu.yaml --out artifacts/cpu_run
aivs stress --config config/stress.cpu.yaml --out artifacts/stress_run
aivs faults --out artifacts/faults
aivs compare --baseline artifacts/cpu_run --current artifacts/cpu_run --out artifacts/same
aivs report --run artifacts/cpu_run
# Optional CPU node-placement evidence (profiling changes measurement overhead):
aivs bench --config config/benchmark.profile.cpu.yaml --out artifacts/profile_run
```

Each command has `--help`. `doctor` performs a real tiny CPU inference probe, reports installed
provider names and public environment metadata, and qualifies optional accelerator requirements.
`fixtures` writes ONNX files and a `fixtures.json` manifest. The three workload types are:

| Model | Input | Operations |
|---|---|---|
| `mlp_gemm` | `[B,128]` | Linear(128,256), ReLU, Linear(256,64) |
| `conv2d` | `[B,3,64,64]` | Two 3×3 convolutions, ReLU, global average pooling |
| `attention_matmul` | `[B,16,32]` | Seeded Q/K/V projections, matmul, softmax, matmul |

Batch is exported as a fixed shape per fixture. The modern `dynamo=True` exporter is attempted
when supported; a checked legacy export is used on capability failure and recorded in metadata.
Only float32 is implemented. No downloads or training occur. Model files and generated input
bytes have SHA256 fingerprints; inputs can be regenerated from model ID, shape and seed.

`demo` records three actual CPU cases with three warmups and ten timed inferences each, plus
six fault outcomes. It also creates `simulated_comparison/`: a **separate, explicitly simulated**
1.0 → 1.25 ms p95 comparator fixture that demonstrates the +25% regression gate. These values
are never substituted into the measured demo cases. Its expected FAIL verdict does not make
the successful demo fail.

## Configuration and measurement

The YAML examples under `config/` are executable inputs. Unknown keys, invalid numeric ranges,
unsupported precision and inconsistent resource limits are rejected before creating output.
Missing keys receive defaults stored in `run.json`.

- Benchmark: single worker, five warmups, thirty timed synchronous `session.run()` calls by
  default. `perf_counter_ns()` measures each inference in milliseconds and the complete timed
  loop separately. Throughput = completed inferences / timed wall seconds, including call and
  loop overhead. Batch throughput is inferences/s, not individual images/s.
- Numerical agreement: PyTorch CPU float32 reference, `atol=1e-4`, `rtol=1e-3`, finite outputs
  and identical shapes required. A failure blocks timing by default. Disabling validation is
  explicitly reported and prevents strict regression comparison.
- Stress: each case runs for ten seconds by default in a spawned child process, using two
  threads sharing one ORT session. Hard limits: 120 seconds per case, four workers, 128 MiB
  prepared input, sixteen session threads. The hard subprocess deadline is fifteen seconds
  by default, **including child startup, session initialization and warmup**. Set a larger
  `stress.timeout_seconds` (maximum 180) when increasing duration. Cases run sequentially;
  two ten-second model cases take at least twenty seconds plus setup.
- Stress checks output on each worker's first completed request and every 100 thereafter.
  All successful calls contribute to throughput; only the first bounded samples per worker
  are retained (10,000 each by default). Percentiles of this capped sample set are labeled.
  RSS and approximate CPU percent are sampled from the child process every 0.25 seconds.
  Sampled peak RSS can miss transients; CPU percent may exceed 100 across cores. Benchmark
  RSS uses boundary snapshots only and leaves peak null.
- Thread timeouts are never used to claim inference stopped. A hung stress child is terminated,
  reaped and marked ERROR/TIMEOUT. Benchmark calls are synchronous; only stress uses hard
  process isolation. Export/reference setup occurs outside the stress child's deadline.

Latency statistics use NumPy percentiles, arithmetic mean and population standard deviation.
Warmup, fixture export and correctness checks are excluded. CPU frequency scaling, thermal
conditions, profiling overhead and other processes affect results. Shared CI timing is
informational; no live speed threshold is a quality gate.

## Provider behavior

CPU is mandatory. CUDA and MIGraphX are optional compatible installed-runtime paths. The suite
does not install drivers or GPU wheels automatically. Request an accelerator using
`backend.provider` and choose `backend.required: true` or `false`.

Installed provider names are presence hints. Actual loaded providers are read from the created
session; absence of a requested mandatory provider yields UNAVAILABLE and exit 3, with no CPU
benchmark relabeled as an accelerator result. An optional unavailable provider yields SKIP.
Runtime EP fallback after session construction is disabled. CPU node fallback within a loaded
accelerator session remains possible and is reported separately.

ORT profiling is opt-in for CPU (`measurement.enable_ort_profiling: true`) and attempted
automatically for accelerator requests. Node-event counts qualify actual target execution,
CPU fallback or unverified placement. A successful correctness result is not itself proof
that GPU nodes executed. Profiles contain upstream runtime event detail; inspect them before
sharing. Default public provenance excludes username, hostname and absolute source paths.

AMD acceleration targets `MIGraphXExecutionProvider`, requiring a compatible platform and
ORT build. The removed legacy ROCm EP is not used.

## Faults and regression

The fault command runs invalid float64 dtype, wrong rank, missing file, corrupt temporary
file, definitely absent sentinel provider, and a subprocess sleeping past a deadline. Models
are preserved. Each record distinguishes the expected category, observed category, PASS/FAIL
fault-test outcome and cleanup. Expected provider unavailability retains the UNAVAILABLE
category; it is not successful inference. No RAM exhaustion or destabilizing faults occur.

Comparisons require matching model/input hashes, shapes/batch, precision, requested/selected
provider, threading/optimization settings, measurement policy, source kind, relevant library
versions and public hardware identity. Missing cases, failed correctness, disabled validation,
nonpositive metrics or incompatible environments cannot pass. Public hardware identity does
not include a serial number; indistinguishable same-spec machines cannot be distinguished.

Latency delta = `(current_p95 - baseline_p95) / baseline_p95 * 100`. Default warning is +5%,
failure +15%; equality triggers the threshold. `--warning-percent` and `--fail-percent` set
explicit thresholds. `--allow-environment-drift` overrides **only** hardware/version differences
with a SEVERE warning and low confidence; it cannot override model/provider/methodology mismatch.
Reports include changed fields and SHA256 fingerprints of both source records. No statistical
significance is inferred from the percentage.

Exit codes: 0 complete/PASS/WARNING or optional skips; 1 correctness, fault-validation or
regression failure (including invalid/incomparable comparison); 2 invalid CLI/config/input;
3 unexpected runtime/system error or mandatory provider unavailable. Report rendering returns
0 when generation succeeds; the saved result verdict remains visible.

## Artifacts and tests

```text
artifacts/cpu_run/
  run.json                     # schema_version=1, public environment and resolved config
  cases.csv                    # one row per case
  latency_samples.csv          # individual measured milliseconds
  report.html                  # UTF-8, responsive tables and inline SVG, opens offline
  fixtures/*.onnx              # checked graph files
  logs/run.log                 # structured, sanitized case/fault outcomes
  profiles/*.json              # when supported and enabled
```

JSON does not serialize NaN/Infinity; unavailable metrics are null. Evidence links are relative.
`report` reads saved records without inference. Reports scope timing metrics to each case.

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python scripts/validate_artifacts.py artifacts/demo artifacts/cpu_run
```

Tests cover deterministic weights/inputs and model export, statistics/threshold boundaries,
strict config errors, unavailable/failing providers, numerical failures, real CPU execution,
warmup sample counts and CSV recomputation, schemas, actual CPU profile placement, five-second
two-worker stress, periodic output failure, hard timeout cleanup, fault semantics, offline
escaped HTML, CLI help/exit behavior and saved-record regeneration. No tests require GPU speed
or real hardware beyond the CPU. GitHub Actions defines Ubuntu Python 3.11 CPU installation,
lint/format checks, pytest, doctor, demo and artifact validation; it does not publish artifacts.
The local verification record is in `VERIFICATION.md`. A resolved platform-specific dependency
snapshot is supplied for the tested Windows environment; regenerate constraints per platform
with `python -m pip freeze --exclude-editable` rather than assuming interchangeable wheel builds.

## Scope and extension

Optional ORT profiling is implemented. INT8 quantization and FP16 are not implemented and
cannot be selected. Real CUDA/MIGraphX execution requires compatible hardware/runtime and must
be tested there. Linux behavior is covered by the supplied CI workflow, but the local execution
host may be Windows; do not infer a successful hosted CI run from its presence.

For board validation, deploy an agent on the actual target, verify provider node placement,
collect reliable vendor telemetry, preserve firmware/driver provenance and run the same
fixture matrix. Board access, instrumentation and cross-compilation are future extensions.

## Upstream references and attribution

This project is MIT licensed. Dependencies retain their upstream licenses (including PyTorch's
BSD-style license, ONNX's Apache 2.0 license and ONNX Runtime's MIT license).

- [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html)
- [ONNX graph checking](https://onnx.ai/onnx/api/checker.html)
- [ORT Python sessions and provider ordering](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [ORT profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
- [CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [MIGraphX provider requirements](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

The original specifications are retained under `spec/` for traceability.
