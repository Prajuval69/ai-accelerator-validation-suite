"""Bounded thread concurrency inside a killable spawned process."""

import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

from aivs.accuracy import check_outputs
from aivs.benchmark import summarize
from aivs.diagnostics import error_message, error_trace
from aivs.providers import create_session, finish_profile


def _stress_child(connection, path, inputs, reference, cfg, root, case_id):
    try:
        session, observation = create_session(Path(path), cfg, Path(root), case_id)
        feed = {"input": inputs}
        for _ in range(cfg["measurement"]["warmup_iterations"]):
            session.run(None, feed)
        process = psutil.Process()
        rss_start = process.memory_info().rss
        resource_samples = []
        stopped = threading.Event()
        process.cpu_percent(None)

        def sample_resources():
            while not stopped.is_set():
                resource_samples.append(
                    {
                        "elapsed_seconds": time.perf_counter() - start,
                        "rss_bytes": process.memory_info().rss,
                        "cpu_percent": process.cpu_percent(None),
                    }
                )
                stopped.wait(cfg["stress"]["memory_sample_interval_seconds"])

        start = time.perf_counter()
        deadline = start + cfg["stress"]["duration_seconds"]
        sampler = threading.Thread(target=sample_resources, daemon=True)
        sampler.start()

        def worker():
            count, errors, samples = 0, [], []
            while time.perf_counter() < deadline and not stopped.is_set():
                before = time.perf_counter_ns()
                try:
                    output = session.run(None, feed)[0]
                    elapsed_ms = (time.perf_counter_ns() - before) / 1e6
                    count += 1
                    if len(samples) < cfg["stress"]["max_samples_per_worker"]:
                        samples.append(elapsed_ms)
                    if cfg["accuracy"]["enabled"] and count % 100 == 1:
                        accuracy = check_outputs(
                            reference, output, cfg["accuracy"]["atol"], cfg["accuracy"]["rtol"]
                        )
                        if accuracy["verdict"] != "PASS":
                            errors.append(
                                {"category": "CORRECTNESS", "message": accuracy["reason"]}
                            )
                            stopped.set()
                except Exception as exc:
                    errors.append(
                        {
                            "category": "RUNTIME",
                            "type": type(exc).__name__,
                            "message": error_message(exc),
                            "trace": error_trace(exc),
                        }
                    )
                    stopped.set()
            return count, errors, samples

        try:
            with ThreadPoolExecutor(max_workers=cfg["measurement"]["concurrency"]) as pool:
                results = list(
                    pool.map(lambda _: worker(), range(cfg["measurement"]["concurrency"]))
                )
        finally:
            stopped.set()
            sampler.join()
        elapsed = time.perf_counter() - start
        count = sum(r[0] for r in results)
        errors = [err for r in results for err in r[1]]
        samples = [sample for r in results for sample in r[2]]
        rss_end = process.memory_info().rss
        timing = summarize(samples, elapsed)
        if timing:
            timing.update(
                throughput_inferences_per_second=count / elapsed,
                warmup_iterations=cfg["measurement"]["warmup_iterations"],
                concurrency=cfg["measurement"]["concurrency"],
                method="bounded concurrent stress; capped first samples per worker",
            )
        result = {
            "status": "FAIL" if errors else "PASS",
            "samples_ms": samples,
            "timing": timing,
            "profile_evidence": finish_profile(session, observation, Path(root)),
            "backend_observation": observation,
            "stability": {
                "requests": count,
                "errors": errors,
                "elapsed_seconds": elapsed,
                "duration_requested_seconds": cfg["stress"]["duration_seconds"],
                "workers": cfg["measurement"]["concurrency"],
                "sample_count": len(samples),
                "samples_capped": count > len(samples),
                "output_check_period_requests": 100,
                "output_checks_enabled": cfg["accuracy"]["enabled"],
            },
            "resources": {
                "rss_start_bytes": rss_start,
                "rss_end_bytes": rss_end,
                "rss_peak_sampled_bytes": max(
                    [rss_start, rss_end] + [r["rss_bytes"] for r in resource_samples]
                ),
                "sampling_interval_seconds": cfg["stress"]["memory_sample_interval_seconds"],
                "samples": resource_samples,
                "note": "child process RSS; sampled peak may miss transients; "
                "CPU percent approximate and may exceed 100%",
            },
        }
        connection.send(result)
    except BaseException as exc:
        connection.send(
            {
                "status": "ERROR",
                "error_category": type(exc).__name__,
                "error": error_message(exc),
                "sanitized_trace": error_trace(exc),
                "timing": None,
            }
        )
    finally:
        connection.close()


def measure_stress(workload, reference, cfg, root):
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_stress_child,
        args=(
            send,
            str(workload.path),
            workload.inputs,
            reference,
            cfg,
            str(root),
            workload.metadata["case_id"],
        ),
    )
    process.start()
    send.close()
    try:
        if receive.poll(cfg["stress"]["timeout_seconds"]):
            try:
                result = receive.recv()
            except EOFError:
                result = {"status": "ERROR", "error_category": "WORKER_CRASH", "timing": None}
        else:
            result = {
                "status": "ERROR",
                "error_category": "TIMEOUT",
                "timing": None,
                "error": "stress process exceeded hard timeout",
            }
    finally:
        if process.is_alive():
            process.join(timeout=0.5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()
        result_cleanup = not process.is_alive()
        receive.close()
        process.close()
    result["cleanup_complete"] = result_cleanup
    return result
