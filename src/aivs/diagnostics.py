"""Useful exception evidence with paths removed from public records."""

import re
import traceback
from pathlib import Path


def error_message(exc: BaseException) -> str:
    message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s'\"<>]*", "<path>", message)[:600]


def error_trace(exc: BaseException) -> list[dict]:
    return [
        {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
        for frame in traceback.extract_tb(exc.__traceback__)[-8:]
    ]
