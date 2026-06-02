from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
from pymoo.core.problem import Problem

from .constants import DEFAULT_EVAL_THREADS, N_VAR
from .evaluation import evaluate_one_unit
from .io_utils import DesignContext


_WORKER_CTX: DesignContext | None = None
_WORKER_MIN_TPS: int = 0
_WORKER_SHIP_MODE: str = "both"


def _init_worker(ctx: DesignContext, min_tps: int, ship_mode: str) -> None:
    global _WORKER_CTX, _WORKER_MIN_TPS, _WORKER_SHIP_MODE
    _WORKER_CTX = ctx
    _WORKER_MIN_TPS = max(0, int(min_tps))
    _WORKER_SHIP_MODE = str(ship_mode)


def _worker_eval(u: np.ndarray) -> dict:
    if _WORKER_CTX is None:
        raise RuntimeError("Worker context is not initialized.")
    return evaluate_one_unit(u, _WORKER_CTX, min_tps=_WORKER_MIN_TPS, ship_mode=_WORKER_SHIP_MODE)


class ApexArchitectProblem(Problem):
    def __init__(
        self,
        ctx: DesignContext,
        n_threads: int = DEFAULT_EVAL_THREADS,
        min_tps: int = 0,
        ship_mode: str = "both",
    ) -> None:
        self.ctx = ctx
        self.n_threads = max(1, int(n_threads))
        self.min_tps = max(0, int(min_tps))
        self.ship_mode = str(ship_mode)
        self._executor: ProcessPoolExecutor | None = None
        if self.n_threads > 1:
            self._executor = ProcessPoolExecutor(
                max_workers=min(self.n_threads, max(1, os.cpu_count() or 1)),
                initializer=_init_worker,
                initargs=(ctx, self.min_tps, self.ship_mode),
            )
        super().__init__(
            n_var=N_VAR,
            n_obj=3,
            n_ieq_constr=16,
            xl=np.zeros(N_VAR, dtype=float),
            xu=np.ones(N_VAR, dtype=float),
            elementwise=False,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        if self._executor is not None:
            results = list(self._executor.map(_worker_eval, list(X), chunksize=2))
        else:
            results = [
                evaluate_one_unit(X[i], self.ctx, min_tps=self.min_tps, ship_mode=self.ship_mode)
                for i in range(X.shape[0])
            ]

        F = np.vstack([r["F"] for r in results]).astype(float, copy=False)
        G = np.vstack([r["G"] for r in results]).astype(float, copy=False)
        CFG = np.array([r.get("cfg", None) for r in results], dtype=object)
        METRICS = np.array([r.get("metrics", None) for r in results], dtype=object)
        out["F"] = F
        out["G"] = G
        out["CFG"] = CFG
        out["METRICS"] = METRICS

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
