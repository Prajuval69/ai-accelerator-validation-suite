# HTML dashboard specification

The output should feel like a professional hardware/software validation report with crisp typography, responsive tables, and clear status.

## Report sections
1. **Header**: run ID, actual timestamp, explicit `Synthetic models / locally generated inputs` caption, verdict.
2. **Metric ribbon**: workloads, executed cases, correctness passes, p95 ms, sequential throughput, expected fault test results. Never aggregate latency across incomparable models without clarifying.
3. **Configuration + environment**: actual hardware/OS/library versions, requested/loaded provider, precision, threading, warmup, repetitions.
4. **Performance table**: model, input shape, backend, p50/p95/p99/avg/stddev, throughput, memory, correctness verdict.
5. **Latency distribution**: small inline SVG histogram or table of raw sample summaries; label units.
6. **Stability section**: stress duration, worker count, requests, errors, throughput and memory sampling caveats.
7. **Fault injection matrix**: injected cause, expected vs observed, outcome, error category.
8. **Comparison panel** if compare report: baseline vs current, compatibility check, percent delta, threshold and caveats.
9. **Methodology + honesty note**: timing environment noisy, synthetic models not production workloads, no physical board validation, GPU claims qualified.

HTML must open offline with no external requests, support UTF-8/accessibility, and handle missing optional sections gracefully. Raw JSON, CSV and ONNX links should be relative. JSON `schema_version=1`.
