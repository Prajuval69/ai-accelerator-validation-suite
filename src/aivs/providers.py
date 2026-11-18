"""Provider presence is a hint; sessions and profiler supply execution evidence."""

import json
from collections import Counter
from pathlib import Path

import onnxruntime as ort


class ProviderUnavailable(RuntimeError):
    pass


def create_session(path: Path, cfg: dict, root: Path, case_id: str):
    provider = cfg["backend"]["provider"]
    available = ort.get_available_providers()
    if provider not in available:
        raise ProviderUnavailable(f"{provider} UNAVAILABLE; installed providers: {available}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = cfg["backend"]["intra_op_num_threads"]
    options.inter_op_num_threads = cfg["backend"]["inter_op_num_threads"]
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    profiling = cfg["measurement"]["enable_ort_profiling"] or provider != "CPUExecutionProvider"
    options.enable_profiling = profiling
    if profiling:
        (root / "profiles").mkdir(parents=True, exist_ok=True)
        options.profile_file_prefix = str(root / "profiles" / case_id)
    requested = (
        [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(path), sess_options=options, providers=requested)
    session.disable_fallback()
    selected = session.get_providers()
    if provider not in selected:
        if profiling:
            session.end_profiling()
        raise ProviderUnavailable(f"{provider} failed to load; selected providers: {selected}")
    return session, {
        "requested_provider": provider,
        "selected_providers": selected,
        "available_providers": available,
        "profiling_enabled": profiling,
        "session_settings": {
            **cfg["backend"],
            "execution_mode": "sequential",
            "graph_optimization": "all",
        },
    }


def finish_profile(session, observation: dict, root: Path) -> dict:
    evidence = {
        "status": "placement unverified",
        "provider_event_counts": {},
        "cpu_fallback_observed": None,
        "profile_path": None,
    }
    if observation["profiling_enabled"]:
        try:
            path = Path(session.end_profiling())
            events = json.loads(path.read_text(encoding="utf-8"))
            counts = Counter(
                event.get("args", {}).get("provider")
                for event in events
                if event.get("cat") == "Node" and event.get("args", {}).get("provider")
            )
            evidence.update(
                provider_event_counts=dict(counts), profile_path=path.relative_to(root).as_posix()
            )
            if counts:
                target = observation["requested_provider"]
                evidence["status"] = (
                    "target execution verified"
                    if counts[target]
                    else "target execution not observed"
                )
                evidence["cpu_fallback_observed"] = target != "CPUExecutionProvider" and bool(
                    counts["CPUExecutionProvider"]
                )
        except (OSError, ValueError, RuntimeError) as exc:
            evidence["error"] = type(exc).__name__
    return evidence
