# Quest Global JD traceability

| Requirement | Project feature | Proof |
|---|---|---|
| Python engineering | packaged CLI, reproducible artifacts, tests | pytest and CI |
| PyTorch/ONNX | three PyTorch workload definitions and ONNX exporter | graph checks + accuracy gate |
| Data generation | deterministic workload inputs | fixture fingerprints |
| Design validation | software-level functional correctness | parity and malformed input checks |
| Board-level / architectures | hardware-oriented runtime scenarios only | accurate limitation disclosure |
| Test analysis | latency samples, aggregates, regression reports | report + CSV |
| Debugging | isolated timeout, fault matrix, provider diagnostics | expected fault test logs |
| Linux | Linux-first CLI, psutil metrics and CI | developer guide and workflow |
| CUDA/ROCm | optional CUDA/MIGraphX provider; no kernel coding | provider detection & truthful skips |
