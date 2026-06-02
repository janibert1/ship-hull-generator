from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CFG_PATH = HERE / "config.json"
LIMITS_PATH = HERE / "limits.json"
EVAL_CFG_PATH = HERE / "_optim_eval_config.json"
RESULTS_FILE = HERE / "optim_results.json"
BEST_CFG_FILE = HERE / "config_best.json"
REPORT_FILE = HERE / "optim_rapport_rank1.txt"
