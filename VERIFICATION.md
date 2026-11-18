# Local implementation verification

Publication update (18 November 2026): the project was initialized as a Git repository and
pushed to the public `Prajuval69/ai-accelerator-validation-suite` repository. A copy of the
previously measured `artifacts/demo_final` evidence is intentionally tracked under `docs/demo`
for GitHub Pages. The original execution record below describes the October validation session.

Executed on 13 October 2026 in `U:\projects\ai_accelerator_validation`, Windows 11 AMD64,
Python 3.12.4. All benchmark workloads and inputs were locally generated and **synthetic**.
Actual environment: Intel64 Family 6 Model 154 Stepping 3, 16 logical / 10 physical cores,
16,782,176,256 bytes total system RAM. No physical board or GPU inference testing occurred.

Installed runtime versions: PyTorch 2.14.1+cpu, ONNX 1.23.2, ONNX Runtime 1.31.0,
NumPy 2.5.4, ONNXScript 0.7.2, psutil 7.2.2. ORT reported AzureExecutionProvider and
CPUExecutionProvider; all measured cases explicitly selected CPUExecutionProvider.
CUDAExecutionProvider and MIGraphXExecutionProvider were unavailable. Git commit is null because
the supplied directory is not a Git repository. No repository was fabricated or committed.

Commands below were executed from the repository root. `.venv/Scripts/python` and
`.venv/Scripts/aivs` are the isolated environment's executables; no global activation was needed.

| Actual command | Observed result |
|---|---|
| `python -m venv .venv` | Completed |
| `.venv/Scripts/python -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu` | Installed CPU PyTorch successfully |
| `.venv/Scripts/python -m pip install -e '.[dev]'` | Editable package and all test dependencies installed |
| `.venv/Scripts/python -m pip check` | No broken requirements found |
| `.venv/Scripts/python -m pip install -e '.[dev]' -c constraints-windows-py312.txt` | Reinstalled successfully using the actual resolved Windows dependency snapshot |
| `.venv/Scripts/python -m ruff check .` | All checks passed |
| `.venv/Scripts/python -m ruff format --check .` | All checked files formatted |
| `.venv/Scripts/python -m pytest -q` | 103 tests passed; includes actual CPU integration and fresh temporary demo |
| `.venv/Scripts/aivs --help` | All eight required commands listed; each command's help also tested in subprocesses |
| `.venv/Scripts/aivs doctor` | CPU identity-model inference PASS; optional accelerator execution NOT_TESTED |
| `.venv/Scripts/python -m aivs demo --out artifacts/demo` | Three correctness passes, 30 measured latency samples, six expected fault passes |
| `.venv/Scripts/aivs demo --out artifacts/demo_final` | Fresh final-code run: three correctness passes, 30 measured latency samples, six expected fault passes |
| `.venv/Scripts/aivs bench --config config/benchmark.cpu.yaml --out artifacts/cpu_run` | Six PASS cases: three model types × batches 1 and 4; 180 measured latency samples |
| `.venv/Scripts/aivs report --run artifacts/cpu_run` | Report regenerated successfully from stored JSON |
| `.venv/Scripts/aivs fixtures --out artifacts/fixtures` | Three ONNX graphs exported and checked |
| `.venv/Scripts/aivs stress --config config/stress.cpu.yaml --out artifacts/stress_run` | Two PASS cases, ten seconds each, two workers; no captured inference/correctness errors; both children cleaned |
| `.venv/Scripts/aivs faults --out artifacts/faults_run` | All six safe injected faults passed with cleanup complete |
| `.venv/Scripts/aivs compare --baseline artifacts/cpu_run --current artifacts/cpu_run --out artifacts/comparison_same` | PASS: six compatible cases with zero p95 delta |
| `.venv/Scripts/aivs compare --baseline artifacts/demo_final/simulated_comparison/baseline.json --current artifacts/demo_final/simulated_comparison/current.json --out artifacts/comparison_regression_final` | Expected FAIL/REGRESSION for the **simulated** +25% p95 fixture; CLI exit 1 |
| `.venv/Scripts/aivs bench --config config/benchmark.profile.cpu.yaml --out artifacts/profile_run` | Three PASS measured CPU cases with profiles retained |
| `.venv/Scripts/python -m aivs bench --config artifacts/required_cuda.yaml --out artifacts/unavailable_cuda_verified` | Actual subprocess returned exit 3; one UNAVAILABLE case, zero selected providers, zero inference samples |
| `.venv/Scripts/python scripts/validate_artifacts.py artifacts/demo artifacts/demo_final artifacts/cpu_run artifacts/stress_run artifacts/faults_run artifacts/comparison_same artifacts/comparison_regression_final artifacts/profile_run artifacts/unavailable_cuda_verified` | Run/case schemas, finite JSON, saved CSV values, benchmark percentile recomputation, offline HTML and evidence links checked |
| `.venv/Scripts/python -m pip freeze --exclude-editable` | Resolved dependency snapshot retained in `constraints-windows-py312.txt` |

Artifact validation was run against demo, final demo, their nested simulated comparisons,
six-case benchmark, configured stress, standalone faults, same-run comparison and simulated
regression reports. The validator accepts an expected regression verdict as a valid report;
it does not reinterpret FAIL as successful inference.

Actual configured stress counts:

| Synthetic case | Timed wall seconds | Completed requests | Retained latency samples | Resource samples | Sampled peak child RSS bytes |
|---|---:|---:|---:|---:|---:|
| `mlp_gemm_b1` | 10.000194000080228 | 625,007 | 20,000 | 39 | 229,097,472 |
| `conv2d_b1` | 10.000310699921101 | 219,206 | 20,000 | 40 | 229,781,504 |

Both stress sample sets were capped. Throughput uses all completed requests divided by actual
wall time; percentiles describe retained first samples per worker. These are local synthetic
observations, not repeatable production performance guarantees. Benchmark and stress runs may
experience background system interference. No performance quality threshold was asserted.

Tests also exercised deterministic +25% regression, threshold equality, environment/model/provider
mismatch, forced environment drift warnings, unavailable provider behavior, numerical mismatch
blocking, modern-export failure with checked legacy fallback, five-second two-worker stress,
periodic output mismatch, deterministic subprocess timeout cleanup, malformed saved records,
path-redacted diagnostics, finite JSON, offline escaping and report regeneration without inference.

Reports were inspected through schema/CSV/link validation and in the local browser. The final demo
displayed measured per-case metrics, synthetic caption, numerical results, memory boundary snapshots,
provider qualifications and all six fault outcomes. The comparison report visibly labels its
fixture values SIMULATED and displays the detected regression. JSON and CSV retain the underlying
evidence; ONNX files and optional profiles use relative links.

CPU node-placement profiling was exercised by integration tests and the permanent profile run.
Non-profiled CPU reports accurately say placement unverified; they do not invent profile evidence.
Modern export succeeded on this installation. PyTorch emitted warnings about absent optional
torchvision operators; none of these three generated models require torchvision.

Limitations: Ubuntu CPU GitHub Actions is supplied but was not run on a hosted runner in this
session. Linux installation/execution is not claimed as locally tested. CUDA/MIGraphX inference,
FP16 and optional INT8 quantization were not performed; float32 and optional ORT profiling are
implemented. GPU telemetry remains null. Real board validation requires target hardware,
instrumentation and separate firmware/driver provenance.

All generated artifacts are ignored by Git and remain local. The README, package sources, tests,
CI workflow, license, configurations and this verification record are handoff files.
