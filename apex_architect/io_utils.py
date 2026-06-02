from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import N_VAR
from .paths import CFG_PATH, LIMITS_PATH


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict | list) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=float)


def load_limits() -> dict:
    return load_json(LIMITS_PATH) if LIMITS_PATH.exists() else {}


@dataclass(frozen=True)
class DesignContext:
    original_cfg: dict
    limits: dict
    stern_base: list[list[float]]
    bowint_base: list[list[float]]
    xl: np.ndarray
    xu: np.ndarray

    @property
    def n_var(self) -> int:
        return N_VAR


def load_design_context(xl: np.ndarray, xu: np.ndarray) -> DesignContext:
    cfg = load_json(CFG_PATH)
    limits = load_limits()
    stern_base = [list(p) for p in cfg.get("Stern_Bezier_Points", [])]
    bowint_base = [list(p) for p in cfg.get("Bow_Intermediate_Points", [])]
    return DesignContext(
        original_cfg=cfg,
        limits=limits,
        stern_base=stern_base,
        bowint_base=bowint_base,
        xl=xl,
        xu=xu,
    )
