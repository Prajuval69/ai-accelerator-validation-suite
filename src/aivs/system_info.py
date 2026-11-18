"""Public environment provenance without hostname, username or absolute paths."""

import importlib.metadata
import platform
import subprocess
from pathlib import Path

import onnxruntime as ort
import psutil


def environment() -> dict:
    cpu = platform.processor()
    if not cpu and platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    versions = {}
    for package in ("torch", "onnx", "onnxruntime", "numpy", "onnxscript", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=False
        )
        commit = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {
        "platform": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": cpu or "unknown",
        "logical_cores": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(),
        **versions,
        "available_providers": ort.get_available_providers(),
        "git_commit": commit,
        "gpu_telemetry": None,
    }
