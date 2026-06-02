from __future__ import annotations

import numpy as np
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

from .constants import (
    BENCHMARK_CRANE_LOAD_MN,
    BENCHMARK_PAYLOAD_T,
    BENCHMARK_RESISTANCE_KN,
    BENCHMARK_SHIP_WEIGHT_T,
    LIMIT_LSW_T,
    LIMIT_PAYLOAD_TPS,
    LIMIT_RESISTANCE_KN,
    TARGET_LSW_T,
    TARGET_PAYLOAD_TPS,
    TARGET_RESISTANCE_KN,
)


def score_min(v: float, target: float, limit: float) -> float:
    if v <= target:
        return 10.0
    if v <= limit:
        return 10.0 - 4.0 * (v - target) / max(limit - target, 1e-9)
    over = (v - limit) / max(limit - target, 1e-9)
    return max(0.0, 6.0 - 6.0 * over)


def score_max(v: float, target: float, limit: float) -> float:
    if v >= target:
        return 10.0
    if v >= limit:
        return 10.0 - 4.0 * (target - v) / max(target - limit, 1e-9)
    under = (limit - v) / max(target - limit, 1e-9)
    return max(0.0, 6.0 - 6.0 * under)


def score_benchmark_min(v: float, benchmark: float) -> float:
    if v <= benchmark:
        return 10.0
    return float(np.clip(10.0 * benchmark / max(v, 1e-9), 0.0, 10.0))


def score_benchmark_max(v: float, benchmark: float) -> float:
    if v >= benchmark:
        return 10.0
    return float(np.clip(10.0 * v / max(benchmark, 1e-9), 0.0, 10.0))


def goal_asf_value(metrics: dict) -> float:
    r = float(metrics["Rtot_kN"])
    p = float(metrics["N_TPs"])
    ship_w = float(metrics["ShipWeight_t"])

    goal_vec = np.array([r, -p, ship_w], dtype=float).reshape(1, -1)
    target_vec = np.array([TARGET_RESISTANCE_KN, -TARGET_PAYLOAD_TPS, TARGET_LSW_T], dtype=float)
    span = np.array(
        [
            LIMIT_RESISTANCE_KN - TARGET_RESISTANCE_KN,
            TARGET_PAYLOAD_TPS - LIMIT_PAYLOAD_TPS,
            LIMIT_LSW_T - TARGET_LSW_T,
        ],
        dtype=float,
    )
    n_goal = (goal_vec - target_vec) / np.maximum(span, 1e-9)
    asf_raw = ASF().do(n_goal, weights=np.array([1.0, 1.0, 1.0]))
    base_asf = float(np.ravel(asf_raw)[0])

    lim_violation = 0.0
    lim_violation += max(0.0, (r - LIMIT_RESISTANCE_KN) / max(span[0], 1e-9))
    lim_violation += max(0.0, (LIMIT_PAYLOAD_TPS - p) / max(span[1], 1e-9))
    lim_violation += max(0.0, (ship_w - LIMIT_LSW_T) / max(span[2], 1e-9))
    return base_asf + 100.0 * lim_violation


def harrington_desirability(metrics: dict) -> float:
    d1 = score_min(float(metrics["Rtot_kN"]), TARGET_RESISTANCE_KN, LIMIT_RESISTANCE_KN) / 10.0
    d2 = score_max(float(metrics["N_TPs"]), TARGET_PAYLOAD_TPS, LIMIT_PAYLOAD_TPS) / 10.0
    d3 = score_min(float(metrics["ShipWeight_t"]), TARGET_LSW_T, LIMIT_LSW_T) / 10.0
    d = (max(d1, 0.0) * max(d2, 0.0) * max(d3, 0.0)) ** (1.0 / 3.0)
    return float(np.clip(d, 0.0, 1.0))


def enrich_scores(metrics: dict) -> dict:
    m = dict(metrics)
    m["asf_value"] = goal_asf_value(m)
    m["score_R"] = score_min(m["Rtot_kN"], TARGET_RESISTANCE_KN, LIMIT_RESISTANCE_KN)
    m["score_P"] = score_max(m["N_TPs"], TARGET_PAYLOAD_TPS, LIMIT_PAYLOAD_TPS)
    m["score_LSW"] = score_min(m["ShipWeight_t"], TARGET_LSW_T, LIMIT_LSW_T)
    m["score_total_30"] = m["score_R"] + m["score_P"] + m["score_LSW"]
    m["desirability_D"] = harrington_desirability(m)

    # Competition-focused scoring against current best-known student values.
    r_kn = float(m.get("Rtot_kN", 1.0e9))
    payload_t = float(m.get("Payload_t", 0.0))
    crane_load_mn = float(m.get("Crane_Load_MN", 0.0))
    ship_weight_t = float(m.get("ShipWeight_t", 1.0e9))

    m["score_comp_resistance"] = score_benchmark_min(r_kn, BENCHMARK_RESISTANCE_KN)
    m["score_comp_payload"] = score_benchmark_max(payload_t, BENCHMARK_PAYLOAD_T)
    m["score_comp_crane"] = score_benchmark_max(crane_load_mn, BENCHMARK_CRANE_LOAD_MN)
    m["score_comp_shipweight"] = score_benchmark_min(ship_weight_t, BENCHMARK_SHIP_WEIGHT_T)
    m["score_comp_total_40"] = (
        m["score_comp_resistance"]
        + m["score_comp_payload"]
        + m["score_comp_crane"]
        + m["score_comp_shipweight"]
    )
    m["beats_resistance"] = bool(r_kn < BENCHMARK_RESISTANCE_KN)
    m["beats_payload"] = bool(payload_t > BENCHMARK_PAYLOAD_T)
    m["beats_crane"] = bool(crane_load_mn > BENCHMARK_CRANE_LOAD_MN)
    m["beats_shipweight"] = bool(ship_weight_t < BENCHMARK_SHIP_WEIGHT_T)
    m["beats_all_4"] = bool(
        m["beats_resistance"] and m["beats_payload"] and m["beats_crane"] and m["beats_shipweight"]
    )
    m["beats_count_4"] = int(m["beats_resistance"]) + int(m["beats_payload"]) + int(m["beats_crane"]) + int(
        m["beats_shipweight"]
    )
    return m


def select_best(candidates: list[dict]) -> tuple[dict, int | None]:
    if not candidates:
        raise RuntimeError("Geen haalbare kandidaten om te selecteren.")

    F = np.array([c["metrics"]["F"] for c in candidates], dtype=float)
    try:
        idxs = HighTradeoffPoints().do(F)
        ht_idx = int(idxs[0]) if len(idxs) > 0 else None
    except Exception:
        ht_idx = None

    # Primary: beat as many benchmark KPIs as possible (0..4),
    # then maximize benchmark score, then desirability, then minimize ASF.
    beats_count = np.array([c["metrics"].get("beats_count_4", 0) for c in candidates], dtype=float)
    comp_total = np.array([c["metrics"].get("score_comp_total_40", 0.0) for c in candidates], dtype=float)
    desirability = np.array([c["metrics"].get("desirability_D", 0.0) for c in candidates], dtype=float)
    asf_values = np.array([c["metrics"]["asf_value"] for c in candidates], dtype=float)
    best_idx = int(np.lexsort((asf_values, -desirability, -comp_total, -beats_count))[0])
    return candidates[best_idx], ht_idx
