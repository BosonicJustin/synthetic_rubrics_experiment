"""Use the standard TOML reader with a Python 3.10 compatibility fallback."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

__all__ = ["tomllib"]
