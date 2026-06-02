from __future__ import annotations

import select
import subprocess
import sys
import threading
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
        plot_mode: str = "auto",  # "auto" | "ondemand" | "none"
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
        self.plot_mode = plot_mode if live_plot else "none"
        self._plot_proc: subprocess.Popen | None = None
        self._plot_cfg_path: Path = HERE / "_optim_live_plot_config.json"
        self._last_plotted_gen = -1

        # on-demand state
        self._pending_cfg: dict | None = None
        self._key_stop = threading.Event()
        self._key_thread: threading.Thread | None = None

        if self.plot_mode == "ondemand":
            self._start_key_listener()

    # ------------------------------------------------------------------ HV

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

        if self.plot_mode == "auto" and self.ctx is not None:
            self._plot_generation_best(algorithm, F, G)
        elif self.plot_mode == "ondemand" and self.ctx is not None:
            self._store_generation_best(algorithm, F, G)

    # ------------------------------------------------------------------ helpers

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

    def _build_cfg_for_idx(self, algorithm, F, G, idx: int) -> dict | None:
        X = algorithm.pop.get("X")
        CFG = algorithm.pop.get("CFG")
        METRICS = algorithm.pop.get("METRICS")
        if X is None or len(X) == 0:
            return None
        try:
            cfg = None
            if CFG is not None and len(CFG) > idx:
                maybe = CFG[idx]
                if isinstance(maybe, dict):
                    cfg = dict(maybe)
            if cfg is None:
                cfg = decode_design(np.asarray(X[idx], dtype=float), self.ctx)
            cfg["Optimizer_Generation"] = int(getattr(algorithm, "n_gen", -1))
            if METRICS is not None and len(METRICS) > idx and isinstance(METRICS[idx], dict):
                cfg["_optimizer_metrics"] = dict(METRICS[idx])
            return cfg
        except Exception:
            return None

    def _terminate_plot_proc(self) -> None:
        if self._plot_proc is None:
            return
        if self._plot_proc.poll() is None:
            try:
                self._plot_proc.terminate()
            except Exception:
                pass
            if self._plot_proc.poll() is None:
                try:
                    self._plot_proc.kill()
                except Exception:
                    pass
        self._plot_proc = None

    def _spawn_plot(self, cfg: dict) -> None:
        save_json(self._plot_cfg_path, cfg)
        self._terminate_plot_proc()
        self._plot_proc = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "plot_full_surface.py"),
                "--config",
                str(self._plot_cfg_path),
                "--crane-mode",
                "stowaway",
            ],
            cwd=str(HERE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # ------------------------------------------------------------------ auto mode

    def _plot_generation_best(self, algorithm, F: np.ndarray | None, G: np.ndarray | None) -> None:
        gen = int(getattr(algorithm, "n_gen", -1))
        if gen <= self._last_plotted_gen:
            return
        idx = self._pick_best_idx(F, G)
        if idx is None:
            return
        try:
            cfg = self._build_cfg_for_idx(algorithm, F, G, idx)
            if cfg is None:
                return
            self._spawn_plot(cfg)
            self._last_plotted_gen = gen
        except Exception:
            return

    # ------------------------------------------------------------------ ondemand mode

    def _store_generation_best(self, algorithm, F: np.ndarray | None, G: np.ndarray | None) -> None:
        idx = self._pick_best_idx(F, G)
        if idx is None:
            return
        try:
            cfg = self._build_cfg_for_idx(algorithm, F, G, idx)
            if cfg is not None:
                self._pending_cfg = cfg
        except Exception:
            return

    def _start_key_listener(self) -> None:
        self._key_stop.clear()
        self._key_thread = threading.Thread(target=self._key_loop, daemon=True)
        self._key_thread.start()
        print("  [Druk Enter om de hull van het beste ontwerp te tonen]", flush=True)

    def _key_loop(self) -> None:
        while not self._key_stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.3)
                if ready and not self._key_stop.is_set():
                    sys.stdin.readline()
                    if self._pending_cfg is not None:
                        try:
                            self._spawn_plot(dict(self._pending_cfg))
                        except Exception:
                            pass
                        print("  [Hull getoond — druk Enter om te vernieuwen]", flush=True)
                    else:
                        print("  [Nog geen ontwerp beschikbaar]", flush=True)
            except Exception:
                break

    # ------------------------------------------------------------------ cleanup

    def close(self) -> None:
        self._key_stop.set()
        self._terminate_plot_proc()
