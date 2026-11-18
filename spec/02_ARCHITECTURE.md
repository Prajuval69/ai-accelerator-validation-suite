# Architecture

## Dataflow

`FixtureGenerator -> ONNXExport/Checker -> BackendDiscovery -> CorrectnessGate -> BenchmarkEngine/StressEngine -> ResultStore -> RegressionComparator -> Offline HTML Reporter`

Fault cases run in isolated test drivers/subprocesses where needed.

## Proposed package layout
```
src/aivs/{__init__,cli,config,system_info,providers,models,fixtures,exporter,accuracy,benchmark,stress,faults,regression,storage,reporting}.py
src/aivs/__main__.py
tests/{unit,integration,cli}/...
.github/workflows/ci.yml
```

## Core interfaces
- `WorkloadSpec`: model family, initializer seed, input tensor spec, model hash, batch, precision, ONNX file path.
- `BackendRequest`: requested provider (CPU/CUDA/MIGraphX), mandatory vs optional, threading/session options.
- `BackendObservation`: installed providers, chosen providers, provider event counts if profiler available, CPU fallback observed/unknown.
- `RunConfig`: iterations, warmup, concurrency, duration, timeout, memory sampling, tolerance, thresholds.
- `BenchmarkRecord`: raw durations ms and their statistics, throughput, correctness verdict, provider metadata and environment.
- `FaultRecord`: injected fault type, expected failure class, observed error, expected vs unexpected outcome, cleanup status.
- `ComparisonRecord`: compatible/incompatible, baseline/current values, percentage change, regression threshold, verdict.

## Provenance & determinism
Capture OS + architecture, CPU info where available, total RAM, Python/PyTorch/ONNX/ORT versions, config, Git commit if accessible, timestamp and hardware detection. Avoid leaking username/hostname/private paths in default report. Seed both PyTorch weights and NumPy inputs. Use SHA256 for ONNX files and inputs.

## Execution provider safety
- `onnxruntime.get_available_providers()` is only a presence hint; construct actual session with requested providers and record `session.get_providers()`.
- To claim accelerator execution, attempt ORT profiling and review per-node provider events. If unavailable, state **provider selected; placement unverified**.
- If requested GPU missing, return `UNAVAILABLE` (or SKIP for optional); never relabel CPU fallback as accelerator result.
- `CPUExecutionProvider` always supported path; optional CUDA and MIGraphX must be capability probed; no requirements to compile driver/kernel code.
