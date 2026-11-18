"""Inspect saved schemas, latency calculations, offline links and finite JSON."""

import argparse
import csv
import json
from html.parser import HTMLParser
from pathlib import Path

import jsonschema
import numpy as np


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for name, value in attrs if name in ("href", "src"))


def inspect_run(folder: Path, schemas: dict):
    text = (folder / "run.json").read_text(encoding="utf-8")
    run = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    jsonschema.validate(run, schemas["run.schema"], format_checker=jsonschema.FormatChecker())
    for case in run["cases"]:
        jsonschema.validate(case, schemas["benchmark_case.schema"])
        if run["kind"] in ("benchmark", "demo") and case.get("timing"):
            samples = case["samples_ms"]
            assert len(samples) == run["config"]["measurement"]["timed_iterations"]
            for p in (50, 95, 99):
                assert np.isclose(case["timing"][f"p{p}_ms"], np.percentile(samples, p))
            assert all(np.isfinite(s) and s > 0 for s in samples)
    with (folder / "latency_samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == sum(len(c.get("samples_ms", [])) for c in run["cases"])
    for case in run["cases"]:
        assert [float(r["latency_ms"]) for r in rows if r["case_id"] == case["case_id"]] == (
            case.get("samples_ms", [])
        )
    html = (folder / "report.html").read_text(encoding="utf-8")
    assert "Synthetic models / locally generated inputs" in html
    parsed = Links()
    parsed.feed(html)
    for target in parsed.targets:
        assert ":" not in target and not target.startswith("//"), f"external link: {target}"
        assert (folder / target).exists(), f"broken link: {target}"
    print(
        f"Validated {folder}: {run['summary']['verdict']}, {len(run['cases'])} cases, "
        f"{len(rows)} saved latency samples"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    schema_root = Path(__file__).resolve().parents[1] / "spec/schemas"
    schemas = {p.stem: json.loads(p.read_text()) for p in schema_root.glob("*.json")}
    for root in args.folders:
        for file in sorted(root.rglob("run.json")):
            inspect_run(file.parent, schemas)


if __name__ == "__main__":
    main()
