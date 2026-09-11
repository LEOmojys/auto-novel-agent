"""Zero-dependency .env loader.

Reads a ``.env`` file from the project root (or an explicit path) and injects
its ``KEY=VALUE`` lines into ``os.environ`` **only when the variable is not
already set** (real environment variables win, so secrets can be supplied by
the shell / CI without a checked-in file).  Parses simple quotes, comments and
blank lines; no external dependency (keeps requirements.txt at its pinned set).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | os.PathLike[str] | None = None, *, force: bool = False) -> Path | None:
    """Load ``KEY=VALUE`` pairs from an env file into ``os.environ``.

    Returns the resolved file path when loaded, or ``None`` if the file does
    not exist.  ``force=True`` overwrites existing environment variables
    (off by default so a real env var always wins).
    """
    env_path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"").strip()
        if not key:
            continue
        if force or not os.environ.get(key):
            os.environ[key] = value
    return env_path


__all__ = ("load_env",)
