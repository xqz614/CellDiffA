"""Configuration helpers for the benchmark manifests."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_environment(value: str) -> str:
    """Expand ``${NAME}`` and ``${NAME:-default}`` without invoking a shell."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Required environment variable {name} is not set.")

    return _ENV.sub(replace, value)


def _expand_tree(value):
    if isinstance(value, str):
        return expand_environment(value)
    if isinstance(value, list):
        return [_expand_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_tree(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return _expand_tree(value)
