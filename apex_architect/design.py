from __future__ import annotations

import numpy as np

from .constants import (
    CRANE_MIN_PIVOT_HEIGHT_M,
    CRANE_PIVOT_EDGE_INSET_M,
    CRANE_SWL_FIXED_T,
    DESIGN_SPEED_KN,
    MIN_FREEBOARD_M,
    N_VAR,
)
from .io_utils import DesignContext


def bound_from_limits(limits: dict, name: str, fallback: tuple[float, float]) -> tuple[float, float]:
    vals = limits.get(name)
    if isinstance(vals, list) and len(vals) == 2:
        return float(vals[0]), float(vals[1])
    return fallback


def build_physical_bounds(limits: dict) -> tuple[np.ndarray, np.ndarray]:
    lo = []
    hi = []

    fixed = [
        ("Length_Loa_m", (40.0, 120.0)),
        ("Breadth_Boa_m", (12.0, 24.0)),
        ("Depth_Doa_m", (5.0, 14.0)),
        ("Lpp_Loa_ratio", (0.87, 0.97)),
        ("MidshipLength_pct_Lpp", (20.0, 55.0)),
        ("Location_midship_pct_Lpp", (40.0, 60.0)),
        ("Bilge_Radius_m", (0.3, 4.5)),
        ("Aft_Shoulder_pct", (60.0, 98.0)),
        ("Fwd_Shoulder_pct", (20.0, 65.0)),
        ("Bow_Rounding_deg", (15.0, 75.0)),
        ("Location_bow_intermediate_curve_pct", (40.0, 80.0)),
        ("Hull_Thickness_mm", (8.0, 75.0)),
        ("Draft_Fraction", (0.0, 1.0)),
        ("Tank1_Width_Fraction", (0.0, 1.0)),
        ("Tank3_Width_Fraction", (0.0, 1.0)),
        ("Tank2_Length_pct_Loa", (20.0, 70.0)),
        ("Tank1_Fill_pct", (0.0, 95.0)),
        ("Parallel_Midship_Combinations", (0.0, 2.99)),
        ("Side_Flare_deg", (0.0, 20.0)),
        ("Side_Flare_Rotation_Point", (0.0, 1.0)),
        ("Crane_Pivot_X_Fraction_Lpp", (0.05, 0.95)),
        ("Payload_Control_Fraction", (0.0, 1.0)),
        ("Crane_Boom_Length_m", (28.0, 80.0)),
        ("Crane_Pivot_Y_Fraction", (-1.0, 1.0)),
        ("Crane_Jib_Angle_deg", (60.0, 80.0)),
        ("Crane_Slew_Angle_deg", (0.0, 360.0)),
    ]
    for key, fb in fixed:
        a, b = bound_from_limits(limits, key, fb)
        lo.append(a)
        hi.append(b)

    lo.extend([-0.12, -0.12])
    hi.extend([0.12, 0.12])
    lo.extend([-0.10] * 16)
    hi.extend([0.10] * 16)

    xl = np.array(lo, dtype=float)
    xu = np.array(hi, dtype=float)
    if len(xl) != N_VAR or len(xu) != N_VAR:
        raise ValueError(f"Bounds mismatch: verwacht {N_VAR}, kreeg {len(xl)}")
    return xl, xu


def unit_to_physical(u: np.ndarray, ctx: DesignContext) -> np.ndarray:
    return ctx.xl + np.clip(u, 0.0, 1.0) * (ctx.xu - ctx.xl)


def decode_design(u: np.ndarray, ctx: DesignContext) -> dict:
    x = unit_to_physical(u, ctx)

    loa = float(x[0])
    boa = float(x[1])
    doa = float(x[2])
    lpp_ratio = float(x[3])
    mid_len = float(x[4])
    mid_loc = float(x[5])
    bilge = float(x[6])
    aft_sh = float(x[7])
    fwd_sh = float(x[8])
    bow_round = float(x[9])
    bow_int_loc = float(x[10])
    hull_t_mm = float(x[11])
    draft_frac = float(x[12])
    t1_frac = float(x[13])
    t3_frac = float(x[14])
    t2_len_pct = float(x[15])
    t1_fill_pct = float(x[16])
    pmc_raw = float(x[17])
    side_flare_deg = float(x[18])
    side_flare_rot = float(x[19])

    crane_pivot_x_frac = float(x[20])
    payload_control_frac = float(x[21])
    crane_boom_length_m = float(x[22])  # physical: 25–80 m
    crane_y_frac = float(x[23])         # physical: −1..+1
    jib_angle_deg = float(x[24])        # physical: 60–80 deg
    slew_angle_deg = float(x[25])       # physical: 0–360 deg

    stern_global = float(x[26])
    bow_global = float(x[27])
    stern_deltas = x[28:36]
    bowint_deltas = x[36:44]
    pmc = int(np.floor(np.clip(pmc_raw, 0.0, 2.99)))

    lpp = loa * lpp_ratio
    t_min = max(0.5, doa * 0.20)
    t_max = doa - MIN_FREEBOARD_M
    # Bias random search toward deeper drafts, which are more likely feasible
    # for heavy TP + crane combinations.
    draft = t_min + (draft_frac ** 0.35) * max(0.0, t_max - t_min)

    b_half = boa / 2.0
    t_w_max = max(1.0, b_half - 0.5)
    t1_w = 1.0 + t1_frac * max(0.0, t_w_max - 1.0)
    t3_w = 1.0 + t3_frac * max(0.0, t_w_max - 1.0)

    bilge = float(np.clip(bilge, 0.3, min(4.5, b_half * 0.35, max(draft, 0.2) * 0.5)))

    # Tank 2 length is entered as %LOA, but we enforce geometric limits on LPP.
    t2_len_m_raw = loa * t2_len_pct / 100.0
    t2_len_m = float(np.clip(t2_len_m_raw, 0.15 * lpp, 0.85 * lpp))
    t2_len_pct_loa = 100.0 * t2_len_m / max(loa, 1e-9)
    t2_len_pct_lpp = 100.0 * t2_len_m / max(lpp, 1e-9)
    t2_center_default = float(ctx.original_cfg.get("Tank2_Center_from_AP_m", 0.5 * lpp))
    t2_half = 0.5 * t2_len_m
    t2_center = float(np.clip(t2_center_default, t2_half, lpp - t2_half))
    crane_pivot_x_m = float(np.clip(crane_pivot_x_frac * lpp, 0.0, lpp))
    crane_pivot_y_m = float(max(0.0, b_half - CRANE_PIVOT_EDGE_INSET_M))

    stern = [list(p) for p in ctx.stern_base]
    for i, k in enumerate(range(1, min(5, len(stern) - 1))):
        dx = float(stern_deltas[i * 2])
        dy = float(stern_deltas[i * 2 + 1])
        stern[k][0] = float(np.clip(stern[k][0] * (1.0 + 0.20 * stern_global) + dx, 0.01, 0.99))
        stern[k][1] = float(np.clip(stern[k][1] * (1.0 + 0.20 * stern_global) + dy, 0.01, 0.99))

    bowint = [list(p) for p in ctx.bowint_base]
    for i, k in enumerate(range(1, min(5, len(bowint) - 1))):
        dx = float(bowint_deltas[i * 2])
        dy = float(bowint_deltas[i * 2 + 1])
        bowint[k][0] = float(np.clip(bowint[k][0] * (1.0 + 0.20 * bow_global) + dx, 0.0, 0.99))
        bowint[k][1] = float(np.clip(bowint[k][1] * (1.0 + 0.20 * bow_global) + dy, -0.1, 0.99))

    cd = dict(ctx.original_cfg)
    cd.update(
        {
            "Length_Loa_m": loa,
            "Breadth_Boa_m": boa,
            "Depth_Doa_m": doa,
            "Lpp_Loa_ratio": lpp_ratio,
            "MidshipLength_pct_Lpp": mid_len,
            "Location_midship_pct_Lpp": mid_loc,
            "Bilge_Radius_m": bilge,
            "Aft_Shoulder_pct": aft_sh,
            "Fwd_Shoulder_pct": fwd_sh,
            "Bow_Rounding_deg": bow_round,
            "Location_bow_intermediate_curve_pct": bow_int_loc,
            "Parallel_Midship_Combinations": pmc,
            "Hull_Thickness_mm": hull_t_mm,
            "Target_Draft_m": draft,
            "Tank1_Width_m": t1_w,
            "Tank1_Fill_pct": t1_fill_pct,
            "Side_Flare_deg": side_flare_deg,
            "Side_Flare_Rotation_Point": side_flare_rot,
            "Tank2_Length_pct_Loa": t2_len_pct_loa,
            "Tank2_Length_pct_Lpp": t2_len_pct_lpp,
            "Tank2_Center_from_AP_m": t2_center,
            "Tank3_Width_m": t3_w,
            "Crane": {
                "swl_max_t": CRANE_SWL_FIXED_T,
                "pivot_x_m": crane_pivot_x_m,
                "pivot_y_frac": crane_y_frac,
                "pivot_y_m": crane_y_frac * crane_pivot_y_m,  # signed; recomputed from actual hull in eval
                "pivot_height_m": doa + CRANE_MIN_PIVOT_HEIGHT_M,  # 1 m above deck
                "boom_length_m": crane_boom_length_m,
                "jib_angle_deg": jib_angle_deg,
                "slewing_angle_deg": slew_angle_deg,
            },
            "Payload_Control_Fraction": float(np.clip(payload_control_frac, 0.0, 1.0)),
            "Transition_Pieces": [],
            "Stern_Bezier_Points": stern,
            "Bow_Intermediate_Points": bowint,
            "Design_Speed_kn": DESIGN_SPEED_KN,
            "Method_S_wet": 0,
            "Method_IE": 0,
        }
    )
    return cd
