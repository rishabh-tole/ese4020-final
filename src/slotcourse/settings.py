"""Load configs/default.yaml with optional `section.key=value` overrides from the CLI."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from slotcourse.config import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def load_settings(path: Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        base = yaml.safe_load(f)
    cfg = copy.deepcopy(base)
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        section, name = key.split(".", 1)
        if section not in cfg or name not in cfg[section]:
            raise KeyError(f"unknown setting {key!r}")
        cfg = {**cfg, section: {**cfg[section], name: _parse_value(raw)}}
    return cfg
