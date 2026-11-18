"""Predictable commands, lazy runtime imports, explicit exit code semantics."""

import argparse
import json
import sys
from pathlib import Path

from aivs import __version__
from aivs.config import ConfigError, load_config


def parser():
    root = argparse.ArgumentParser(description="CPU-first synthetic ONNX runtime validation")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="show environment and provider diagnostics")
    for name in ("fixtures", "demo", "bench", "stress", "faults"):
        command = commands.add_parser(name, help=f"run {name} with local synthetic models")
        command.add_argument(
            "--out",
            type=Path,
            default=Path("artifacts") / name,
            help="new or empty artifact directory",
        )
        if name in ("bench", "stress", "fixtures"):
            command.add_argument("--config", type=Path)
    command = commands.add_parser("compare", help="compare compatible saved runs")
    command.add_argument("--baseline", required=True, type=Path)
    command.add_argument("--current", required=True, type=Path)
    command.add_argument("--out", type=Path, default=Path("artifacts/comparison"))
    command.add_argument(
        "--allow-environment-drift",
        action="store_true",
        help="override hardware/version mismatch with SEVERE low-confidence warning",
    )
    command.add_argument("--fail-percent", type=float, default=15)
    command.add_argument("--warning-percent", type=float, default=5)
    command = commands.add_parser("report", help="regenerate offline HTML without inference")
    command.add_argument("--run", required=True, type=Path)
    return root


def doctor():
    import tempfile

    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    from aivs.system_info import environment

    info = environment()
    with tempfile.TemporaryDirectory(prefix="aivs_doctor_") as directory:
        path = Path(directory) / "identity.onnx"
        tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])], "cpu_probe", [tensor], [output]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        onnx.save(model, path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        import numpy as np

        value = session.run(None, {"input": np.array([1], dtype=np.float32)})[0]
        info["cpu_probe"] = {
            "selected_providers": session.get_providers(),
            "verdict": "PASS" if float(value[0]) == 1 else "FAIL",
        }
    info["optional_accelerators"] = {
        provider: {
            "installed": provider in info["available_providers"],
            "execution": "NOT_TESTED",
            "requirements": requirements,
        }
        for provider, requirements in (
            (
                "CUDAExecutionProvider",
                "Compatible ORT GPU wheel, NVIDIA hardware and CUDA libraries",
            ),
            (
                "MIGraphXExecutionProvider",
                "Compatible ORT/MIGraphX build, AMD hardware and drivers",
            ),
        )
    }
    print(json.dumps(info, indent=2))
    return 0 if info["cpu_probe"]["verdict"] == "PASS" else 3


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "report":
            from aivs.reporting import render_report
            from aivs.storage import load_run

            run, source = load_run(args.run)
            result = render_report(run, source.parent / "report.html")
            print(f"Report: {result}")
            return 0
        from aivs.storage import prepare_output

        if args.command == "compare":
            from aivs.regression import comparison_artifact

            if not 0 <= args.warning_percent <= args.fail_percent <= 1000:
                raise ConfigError("require 0 <= warning-percent <= fail-percent <= 1000")
            prepare_output(args.out)
            result = comparison_artifact(
                args.baseline,
                args.current,
                args.out,
                args.allow_environment_drift,
                args.fail_percent,
                args.warning_percent,
            )
        else:
            kind = "benchmark" if args.command == "bench" else args.command
            cfg = load_config(getattr(args, "config", None), kind=kind)
            if kind == "demo":
                cfg["measurement"].update(warmup_iterations=3, timed_iterations=10)
            if kind == "faults":
                cfg["models"] = ["mlp_gemm"]
            prepare_output(args.out)
            if kind == "fixtures":
                from aivs.fixtures import generate
                from aivs.storage import write_json
                from aivs.system_info import environment

                workloads = generate(cfg, args.out)
                write_json(
                    args.out / "fixtures.json",
                    {
                        "schema_version": 1,
                        "synthetic_models": True,
                        "environment": environment(),
                        "config": cfg,
                        "fixtures": [w.metadata for w in workloads],
                    },
                )
                print(f"Exported and graph-checked {len(workloads)} fixtures: {args.out}")
                return 0
            from aivs.runner import execute

            result = execute(cfg, args.out, kind)
        verdict = result["summary"]["verdict"]
        print(f"{verdict}: {args.out / 'report.html'}")
        print(json.dumps(result["summary"], indent=2))
        return 3 if verdict == "ERROR" else 1 if verdict == "FAIL" else 0
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; isolated workers cleaned up", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Runtime failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
