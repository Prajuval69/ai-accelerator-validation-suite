# Benchmarking and stress methodology

## Timing measurements
- Use `time.perf_counter_ns()` for durations; warm up before collecting timed samples (defaults: 5 warmups, 30 timed iterations).
- For per-inference latency, use synchronous `session.run()` calls in same process, same prepared input. Record individual milliseconds in CSV, then mean/median/p95/p99/stddev; record total timed wall time and sequential throughput.
- Clearly distinguish latency benchmark (single worker) from multiworker throughput stress test. Do not compute throughput as `1/mean_latency` if wall-time measurement is available.
- Note CPU frequency scaling, interference, thermal effects and optimized graph caching; never use CI as stable baseline.

## Resource instrumentation
- Via `psutil`: process RSS start/end/peak if a sampler is actually running; CPU percent is approximate; use sampling interval to avoid measurement noise.
- System info includes CPU architecture/logical cores, RAM, OS, library versions and flags.
- GPU memory/energy/temperature only if reliable vendor tools are actually present and parsed; otherwise null/unavailable.

## Stress
- Defaults: 10 sec runtime, threads <=2; max config duration 120 sec and concurrency <=4 for local guardrail (overridable only by explicit safe flag if desired).
- Use a bounded thread pool to exercise one validated ORT session if safe; capture counts, throughput, exceptions, summary latency. Compare stress outputs periodically with baseline if deterministic.
- Subprocess isolation for tests that must be hard-timed-out; never use a thread timeout to pretend the running code stopped.
- Cleanup child processes, temp files and pooled resources on error/interrupt.

## Backend claims
- Report backend requested, session providers, provider availability, and profile evidence. A CPU fallback run is not proof of GPU acceleration.
- GPU results not comparable to CPU baselines by default; compare like backend and machine only.
