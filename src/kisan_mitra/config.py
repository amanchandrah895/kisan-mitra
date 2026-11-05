"""Configuration loading. Paths in the YAML are resolved relative to the repo root."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict with attribute access for top-level sections."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(item) from e

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else REPO_ROOT / p


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _deep_update(copy.deepcopy(cfg), overrides or {})
    return Config(cfg)
