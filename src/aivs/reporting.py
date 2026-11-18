"""Offline, escaped HTML with scoped metrics and relative evidence links."""

import html
import json
from pathlib import Path, PurePosixPath

import numpy as np


def esc(value):
    return html.escape(str(value), quote=True)


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5g}" if np.isfinite(value) else "—"
    return esc(value)


def table(headers, rows, caption):
    return (
        '<div class="scroll"><table><caption>'
        + esc(caption)
        + "</caption><thead><tr>"
        + "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def link(path, label):
    if not isinstance(path, str):
        return ""
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or ":" in path or "\\" in path:
        return esc(label)
    return f'<a href="{esc(path)}">{esc(label)}</a>'


def histogram(case):
    values = case.get("samples_ms", [])
    if not values:
        return ""
    values = np.asarray([v for v in values if isinstance(v, (int, float)) and np.isfinite(v)])
    if not len(values):
        return ""
    counts, edges = np.histogram(values, bins=16)
    bars = "".join(
        f'<rect x="{10 + i * 24}" y="{110 - count / max(counts) * 90:.2f}" '
        f'width="20" height="{count / max(counts) * 90:.2f}"/>'
        for i, count in enumerate(counts)
    )
    title = f"{case['case_id']} latency distribution (ms), {len(values)} samples"
    return (
        f"<figure><figcaption>{esc(title)}</figcaption>"
        f'<svg viewBox="0 0 410 145" role="img" aria-label="{esc(title)}">'
        f'<title>{esc(title)}</title>{bars}<text x="10" y="137">{edges[0]:.4g} ms</text>'
        f'<text x="300" y="137">{edges[-1]:.4g} ms</text></svg></figure>'
    )


def render_report(run: dict, path: Path):
    simulated = run.get("simulated_values") or run.get("measurement_origin") == "simulated_fixture"
    status = run.get("summary", {}).get("verdict", "UNKNOWN")
    case_rows, provider_rows, stress_rows, diagnostics = [], [], [], []
    for case in run["cases"]:
        t = case.get("timing") or {}
        resource = case.get("resources") or {}
        case_rows.append(
            [
                esc(case["case_id"]),
                esc(case["input_shape"]),
                esc(case["precision"]),
                esc(case["status"]),
                esc(case["accuracy"].get("verdict")),
                *[
                    display(t.get(key))
                    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "stddev_ms")
                ],
                display(t.get("throughput_inferences_per_second")),
                display(resource.get("rss_end_bytes")),
                display(resource.get("rss_peak_sampled_bytes")),
            ]
        )
        detail = {
            key: case.get(key)
            for key in (
                "seed",
                "model_sha256",
                "input_sha256",
                "export",
                "session_settings",
                "accuracy",
                "error_category",
                "error",
                "sanitized_trace",
            )
        }
        detail["memory"] = {key: value for key, value in resource.items() if key != "samples"}
        diagnostics.append(
            "<details><summary>"
            + esc(case["case_id"])
            + "</summary><pre>"
            + esc(json.dumps(detail, indent=2))
            + "</pre></details>"
        )
        evidence = case.get("profile_evidence") or {}
        provider_rows.append(
            [
                esc(case["case_id"]),
                esc(case["requested_provider"]),
                esc(", ".join(case["selected_providers"]) or "none"),
                esc(evidence.get("status", "not executed")),
                esc(evidence.get("provider_event_counts", {})),
                display(evidence.get("cpu_fallback_observed")),
                link(evidence.get("profile_path"), "ORT profile"),
            ]
        )
        if case.get("stability"):
            s = case["stability"]
            stress_rows.append(
                [
                    esc(case["case_id"]),
                    display(s["elapsed_seconds"]),
                    display(s["workers"]),
                    display(s["requests"]),
                    display(len(s["errors"])),
                    display(s["samples_capped"]),
                    display(case.get("cleanup_complete")),
                    esc(resource.get("note")),
                ]
            )
    fault_rows = [
        [
            esc(f["fault_id"]),
            esc(f["expected"]),
            esc(f["observed_error_category"]),
            esc(f["outcome"]),
            esc(f.get("observed_error") or "—"),
            display(f.get("cleanup_complete")),
        ]
        for f in run.get("faults", [])
    ]
    summary = run.get("summary", {})
    metrics = [
        ("Workload types", len({c["model_id"] for c in run["cases"]})),
        ("Executed cases", summary.get("executed_cases", 0)),
        ("Correctness passes", summary.get("correctness_passes", 0)),
        ("Expected fault passes", summary.get("expected_fault_passes", 0)),
    ]
    sections = [
        "<h2>Performance</h2>"
        + table(
            [
                "Case / model",
                "Shape",
                "Precision",
                "Status",
                "Correctness",
                "p50 ms",
                "p95 ms",
                "p99 ms",
                "Mean ms",
                "Stddev ms",
                "Inferences/s",
                "End RSS bytes",
                "Sampled peak RSS bytes",
            ],
            case_rows,
            "Performance by case; latency and throughput are never averaged across models",
        ),
        "<h2>Execution provider evidence</h2>"
        + table(
            [
                "Case",
                "Requested",
                "Loaded",
                "Placement",
                "Node event counts",
                "CPU fallback",
                "Evidence",
            ],
            provider_rows,
            "Presence, selection and placement",
        ),
        '<h2>Latency distributions</h2><div class="charts">'
        + "".join(histogram(c) for c in run["cases"])
        + "</div>",
        "<h2>Case provenance and diagnostics</h2>" + "".join(diagnostics),
        "<h2>Stability</h2>"
        + (
            table(
                [
                    "Case",
                    "Wall seconds",
                    "Workers",
                    "Requests",
                    "Errors",
                    "Samples capped",
                    "Child cleaned",
                    "Resource caveat",
                ],
                stress_rows,
                "Concurrent stress throughput uses all completed requests / timed wall seconds",
            )
            if stress_rows
            else "<p>No concurrent stress measurements in this run.</p>"
        ),
        "<h2>Fault injection</h2>"
        + (
            table(
                ["Injection", "Expected", "Observed category", "Outcome", "Error", "Cleanup"],
                fault_rows,
                "PASS means the injected failure was handled as expected",
            )
            if fault_rows
            else "<p>No faults injected in this run.</p>"
        ),
    ]
    if "comparison" in run:
        compared = run["comparison"]["comparisons"]
        metrics = [
            ("Compared cases", len(compared)),
            ("Regressions", sum(r["verdict"] == "REGRESSION" for r in compared)),
            ("Incomparable", sum(r["verdict"] == "INCOMPARABLE" for r in compared)),
            ("Warnings", sum(r["verdict"] == "WARNING" for r in compared)),
        ]
        sections = []
        rows = [
            [
                esc(r["case_id"]),
                esc(r["verdict"]),
                display(r["baseline_p95_ms"]),
                display(r["current_p95_ms"]),
                display(r["delta_percent"]),
                display(r["warning_percent"]),
                display(r["fail_percent"]),
                esc(r["comparison_confidence"]),
                esc(r.get("warning") or r.get("reason") or "—"),
                "<pre>" + esc(json.dumps(r["changed_fields"], indent=2)) + "</pre>",
            ]
            for r in run["comparison"]["comparisons"]
        ]
        sections.append(
            "<h2>Baseline comparison</h2>"
            + table(
                [
                    "Case",
                    "Verdict",
                    "Baseline p95 ms",
                    "Current p95 ms",
                    "Change %",
                    "Warn %",
                    "Fail %",
                    "Confidence",
                    "Caveat",
                    "Changed fields",
                ],
                rows,
                "Compatibility precedes regression; threshold equality increases severity",
            )
            + "<pre>"
            + esc(json.dumps(run.get("sources"), indent=2))
            + "</pre>"
        )
    links = " · ".join(link(v, k) for k, v in run.get("artifacts", {}).items())
    models = " · ".join(link(c.get("model_path"), c["case_id"] + " ONNX") for c in run["cases"])
    notice = (
        '<p class="notice">SIMULATED COMPARISON FIXTURE VALUES — not measured performance.</p>'
        if simulated
        else ""
    )
    ribbon = "".join(
        f'<div class="metric"><strong>{display(val)}</strong>{esc(label)}</div>'
        for label, val in metrics
    )
    payload = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIVS {esc(run["kind"])} validation report</title>
<style>
:root{{color-scheme:light;--ink:#182a38;--accent:#066969}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f6;color:var(--ink);
font:15px/1.6 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:36px}}
header{{border-top:5px solid var(--accent);padding:24px 0}}h1{{font-size:32px;margin:0}}
h2{{font-size:22px;margin-top:40px}}.eyebrow{{text-transform:uppercase;letter-spacing:.14em}}
.status{{font-weight:700;border:1px solid;padding:4px 12px;display:inline-block}}
.ribbon{{display:flex;flex-wrap:wrap;border-top:1px solid #b8c7cb;border-bottom:1px solid #b8c7cb}}
.metric{{padding:20px 28px 20px 0;flex:1;min-width:180px}}
.metric strong{{display:block;font-size:28px}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white}}
caption{{text-align:left;padding:12px 0;color:#415664}}th,td{{padding:12px;text-align:left;
border-bottom:1px solid #d5dfe1;vertical-align:top}}th{{background:#e7eeee;white-space:nowrap}}
td{{font-variant-numeric:tabular-nums}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;
font-size:12px;background:#e8eeee;padding:18px}}details{{margin:14px 0}}summary{{cursor:pointer}}
a{{color:#005f63}}.charts{{display:flex;flex-wrap:wrap;gap:20px}}figure{{margin:0;max-width:440px;
width:100%;background:white;padding:16px}}svg{{width:100%;fill:var(--accent)}}
svg text{{fill:var(--ink);
font-size:11px}}.notice{{border-left:4px solid #a26400;padding:14px;background:#fff0d2}}
@media(max-width:650px){{main{{padding:18px}}h1{{font-size:26px}}}}
</style></head><body><main>
<header><div class="eyebrow">AIVS / Runtime validation</div>
<h1>{esc(run["kind"].title())} report</h1>
<p>Synthetic models / locally generated inputs</p>
<p>Run {esc(run["run_id"])} · {esc(run["started_at_utc"])}</p>
<span class="status">{esc(status)}</span>{notice}
</header><div class="ribbon">
{ribbon}
</div><p>{links}</p><p>{models}</p>
<h2>Configuration and environment</h2>
<details><summary>Actual runtime / hardware provenance</summary>
<pre>{esc(json.dumps(run["environment"], indent=2))}</pre></details>
<details><summary>Reproducible configuration, timing and tolerances</summary>
<pre>{esc(json.dumps(run["config"], indent=2))}</pre></details>
{"".join(sections)}
<h2>Methodology and limits</h2><p>{esc(run.get("methodology", ""))}</p>
<p>Each inference uses the same prepared input. CPU frequency, thermal effects, graph caching and
other activity can affect timings. Stress retains bounded first samples per worker; their
percentiles may not represent the entire run. Sequential benchmarks include Python call overhead;
stress throughput is concurrent. GPU selection alone does not prove accelerator node execution.
Boundary RSS snapshots do not establish peak memory. Synthetic numerical agreement establishes
runtime fidelity, not task accuracy. No physical development board testing was performed.</p>
</main></body></html>"""
    path.write_text(payload, encoding="utf-8")
    return path
