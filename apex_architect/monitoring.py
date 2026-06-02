from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV

from .constants import (
    HV_EARLYSTOP_MIN_DELTA,
    HV_EARLYSTOP_PATIENCE_GEN,
    HV_REF_F1,
    HV_REF_F2,
    HV_REF_F3,
)
from .design import decode_design
from .io_utils import DesignContext, save_json
from .paths import HERE


class HypervolumeEarlyStopCallback(Callback):
    def __init__(
        self,
        ctx: DesignContext | None = None,
        ref_point: np.ndarray | None = None,
        patience_gen: int = HV_EARLYSTOP_PATIENCE_GEN,
        min_delta: float = HV_EARLYSTOP_MIN_DELTA,
        live_plot: bool = True,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.ref_point = np.array([HV_REF_F1, HV_REF_F2, HV_REF_F3], dtype=float) if ref_point is None else ref_point
        self.patience_gen = max(5, int(patience_gen))
        self.min_delta = float(min_delta)
        self.hv_indicator = HV(ref_point=self.ref_point, norm_ref_point=False)
        self.hv_history: list[dict] = []
        self.best_hv = -np.inf
        self.last_improve_gen = 0
        self.live_plot = bool(live_plot)
        self._plot_proc: subprocess.Popen | None = None
        self._plot_cfg_path: Path = HERE / "_optim_live_plot_config.json"
        self._last_plotted_gen = -1

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")

        hv_val = float("nan")
        n_feas = 0
        if F is not None and G is not None and len(F) > 0:
            feas_mask = np.all(G <= 0.0, axis=1)
            n_feas = int(np.sum(feas_mask))
            if n_feas > 0:
                F_feas = F[feas_mask]
                hv_val = float(self.hv_indicator.do(F_feas))

        self.hv_history.append({"gen": int(algorithm.n_gen), "hv": hv_val, "n_feasible": n_feas})

        if np.isfinite(hv_val):
            if hv_val > self.best_hv + self.min_delta:
                self.best_hv = hv_val
                self.last_improve_gen = int(algorithm.n_gen)
            elif int(algorithm.n_gen) - self.last_improve_gen >= self.patience_gen:
                algorithm.termination.force_termination = True

        if self.live_plot and self.ctx is not None:
            self._plot_generation_best(algorithm, F, G)

    def _pick_best_idx(self, F: np.ndarray, G: np.ndarray) -> int | None:
        if F is None or G is None or len(F) == 0 or len(G) == 0:
            return None
        feas_mask = np.all(G <= 0.0, axis=1)
        if np.any(feas_mask):
            feas_idx = np.where(feas_mask)[0]
            F_feas = F[feas_idx]
            score = F_feas[:, 0] + F_feas[:, 1] + F_feas[:, 2]
            return int(feas_idx[int(np.argmin(score))])
        cv = np.sum(np.maximum(G, 0.0), axis=1)
        return int(np.argmin(cv))

    def _terminate_plot_proc(self) -> None:
        if self._plot_proc is None:
            return
        if self._plot_proc.poll() is None:
            try:
                self._plot_proc.terminate()
            except Exception:
                pass
            # Never block optimizer loop waiting for UI shutdown.
            if self._plot_proc.poll() is None:
                try:
                    self._plot_proc.kill()
                except Exception:
                    pass
        self._plot_proc = None

    def _plot_generation_best(self, algorithm, F: np.ndarray | None, G: np.ndarray | None) -> None:
        gen = int(getattr(algorithm, "n_gen", -1))
        if gen <= self._last_plotted_gen:
            return
        X = algorithm.pop.get("X")
        CFG = algorithm.pop.get("CFG")
        METRICS = algorithm.pop.get("METRICS")
        if X is None or len(X) == 0:
            return
        idx = self._pick_best_idx(F, G)
        if idx is None:
            return
        try:
            cfg = None
            if CFG is not None and len(CFG) > idx:
                maybe_cfg = CFG[idx]
                if isinstance(maybe_cfg, dict):
                    cfg = dict(maybe_cfg)
            if cfg is None:
                cfg = decode_design(np.asarray(X[idx], dtype=float), self.ctx)
            cfg["Optimizer_Generation"] = gen
            if METRICS is not None and len(METRICS) > idx and isinstance(METRICS[idx], dict):
                cfg["_optimizer_metrics"] = dict(METRICS[idx])
            save_json(self._plot_cfg_path, cfg)
            self._terminate_plot_proc()
            self._plot_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(HERE / "plot_full_surface.py"),
                    "--config",
                    str(self._plot_cfg_path),
                    "--crane-mode",
                    "stowaway"
                ],
                cwd=str(HERE),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._last_plotted_gen = gen
        except Exception:
            # Never break optimizer flow due to plotting errors.
            return

    def close(self) -> None:
        self._terminate_plot_proc()
