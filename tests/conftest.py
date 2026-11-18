from pathlib import Path

import pytest

from aivs.config import parse_config
from aivs.runner import execute
from aivs.storage import prepare_output


@pytest.fixture(scope="session")
def cpu_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("cpu_integration") / "run"
    prepare_output(root)
    cfg = parse_config(
        {
            "measurement": {
                "warmup_iterations": 3,
                "timed_iterations": 10,
                "enable_ort_profiling": True,
            }
        }
    )
    run = execute(cfg, root)
    return root, run


@pytest.fixture
def schemas():
    import json

    directory = Path(__file__).resolve().parents[1] / "spec" / "schemas"
    return {p.stem: json.loads(p.read_text()) for p in directory.glob("*.json")}
