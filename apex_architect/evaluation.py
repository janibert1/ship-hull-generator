from __future__ import annotations

import os
import sys
import traceback

import numpy as np

import plot_full_surface as pfs
from engineering.geometry import analyze_stern_shape, hull_hw_at_z
from engineering.hydrostatics import _displacement_at_draft, center_of_buoyancy, waterplane_data
from engineering.run import _build_hull_area_entries, compute_equilibrium
from engineering.resistance import (
    compute_at,
    compute_hull_coefficients,
    compute_ie_regression,
    compute_s_wet_hm,
    holtrop_mennen,
)
from engineering.tanks import build_tank_fill_diagram

from .constants import (
    CRANE_BOOM_MASS_FRAC,
    CRANE_CLEARANCE_M,
    CRANE_HOUSE_MASS_FRAC,
    CRANE_JIB_ANGLE_FIXED_DEG,
    CRANE_LOAD_HEIGHT_M,
    CRANE_MAX_HEEL_DEG,
    CRANE_MIN_PIVOT_HEIGHT_M,
    CRANE_PIVOT_EDGE_INSET_M,
    CRANE_SWL_FULL_ANGLE_DEG,
    CRANE_SWL_ZERO_ANGLE_FACTOR,
    CRANE_RIGGING_HEIGHT_M,
    CRANE_RIGGING_MASS_FRAC,
    DESIGN_SPEED_KN,
    HULL_MASS_FACTOR,
    LOFT_N_T_FAST,
    LOFT_N_U_FAST,
    LSW_B_EXP,
    LSW_B_REF_M,
    LSW_D_EXP,
    LSW_D_REF_M,
    LSW_STRENGTH_GAIN,
    LSW_PER_PAYLOAD_CAP,
    MAX_BALLAST_FRAC,
    MAX_DRAFT_M,
    MAX_FN,
    MAX_ROLL_PERIOD_S,
    MIN_BALLAST_FRAC,
    MIN_DRAFT_M,
    MIN_FREEBOARD_M,
    MIN_GM_M,
    MIN_ROLL_PERIOD_S,
    STEEL_RHO,
    STRENGTH_SIGMA_ALLOW_MPA,
    STRICT_EQ_TANK2_HEADROOM_BUFFER_M3,
    STRICT_EQ_TANK3_BUFFER_M3,
    STRUCTURAL_AREA_FACTOR,
    TP_CG_HEIGHT_M,
    TP_RADIUS_M,
    TP_WEIGHT_T,
)
from .design import decode_design
from .io_utils import DesignContext, save_json
from .packing import pack_transition_pieces_hex
from .paths import EVAL_CFG_PATH

def write_eval_config(cfg_dict: dict, cfg_path) -> None:
    save_json(cfg_path, cfg_dict)
    pfs.CONFIG_FILE = str(cfg_path)
    pfs.reload_config_module(str(cfg_path))


def estimate_lsw_kg(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, hull_t_mm: float, boa_m: float, doa_m: float) -> float:
    hull_half_area = pfs._compute_surface_area(X, Y, Z)
    x_mid = X.mean(axis=1)
    y_deck = Y[:, -1]
    deck_area = 2.0 * float(np.trapezoid(y_deck, x_mid))
    total_shell = (2.0 * hull_half_area + deck_area) * STRUCTURAL_AREA_FACTOR
    base_kg = total_shell * (hull_t_mm / 1000.0) * STEEL_RHO * HULL_MASS_FACTOR

    b_ratio = max(0.5, boa_m / LSW_B_REF_M)
    d_ratio = max(0.5, doa_m / LSW_D_REF_M)
    section_factor = 1.0 + LSW_STRENGTH_GAIN * (b_ratio ** LSW_B_EXP) * (d_ratio ** LSW_D_EXP)
    return base_kg * section_factor


def estimate_side_tank_fill_state(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    y_inner: float,
    fill_pct: float,
    h_max: float,
    n_z: int = 28,
) -> tuple[float, float]:
    """Return (volume_m3, vcg_m) for one side tank over full LPP."""
    fill_pct = float(np.clip(fill_pct, 0.0, 100.0))
    h_fill = h_max * fill_pct / 100.0
    if h_fill <= 1e-9:
        return 0.0, 0.0

    x_mid = X.mean(axis=1)
    if len(x_mid) < 2:
        return 0.0, 0.0

    csas = []
    z_moms = []
    for t in range(len(x_mid)):
        y_cs = Y[t]
        z_cs = Z[t]
        z_arr = np.linspace(0.0, h_fill, n_z)
        widths = np.array(
            [max(0.0, hull_hw_at_z(y_cs, z_cs, z) - y_inner) for z in z_arr],
            dtype=float,
        )
        csa = float(np.trapezoid(widths, z_arr))
        z_mom = float(np.trapezoid(z_arr * widths, z_arr))
        csas.append(csa)
        z_moms.append(z_mom)

    csa_arr = np.array(csas, dtype=float)
    z_mom_arr = np.array(z_moms, dtype=float)

    volume_m3 = float(np.trapezoid(csa_arr, x_mid))
    if volume_m3 <= 1e-12:
        return 0.0, 0.0
    z_first_moment = float(np.trapezoid(z_mom_arr, x_mid))
    vcg_m = z_first_moment / volume_m3
    return volume_m3, vcg_m


def strict_equilibrium_violation(
    cd: dict,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    LPP: float,
    D: float,
) -> tuple[float, str]:
    """Run strict option-7-style equilibrium checks; return (violation, reason)."""
    b_half = 0.5 * float(cd["Breadth_Boa_m"])
    y_inner1 = b_half - float(cd.get("Tank1_Width_m", 0.0))
    y_inner3 = b_half - float(cd.get("Tank3_Width_m", 0.0))
    if y_inner1 <= 0.0 or y_inner3 <= 0.0:
        return 1.0e3, "tank width exceeds half-beam"

    loa = float(cd["Length_Loa_m"])
    t2_len_pct = float(cd.get("Tank2_Length_pct_Loa", 30.0))
    t2_len = loa * t2_len_pct / 100.0

    geo = {
        "X_surf": X,
        "Y_surf": Y,
        "Z_surf": Z,
        "LPP": LPP,
        "B_half": b_half,
        "D": D,
    }

    diag1 = build_tank_fill_diagram(
        X,
        Y,
        Z,
        tank_type="side_sb",
        h_max=D,
        x_min=0.0,
        x_max=LPP,
        y_inner_sb=y_inner1,
        y_inner_port=y_inner3,
        fill_steps=101,
    )
    diag3 = build_tank_fill_diagram(
        X,
        Y,
        Z,
        tank_type="side_port",
        h_max=D,
        x_min=0.0,
        x_max=LPP,
        y_inner_sb=y_inner1,
        y_inner_port=y_inner3,
        fill_steps=101,
    )
    hull_area_entries = _build_hull_area_entries(geo, cd)

    try:
        eq = compute_equilibrium(
            geo,
            cd,
            diag1,
            diag3,
            hull_area_entries,
            t2_len,
            y_inner1,
            y_inner3,
        )
    except ValueError as exc:
        return 1.0e3, str(exc)

    diag2 = build_tank_fill_diagram(
        X,
        Y,
        Z,
        tank_type="center",
        h_max=D,
        x_min=float(eq["t2_x0"]),
        x_max=float(eq["t2_x1"]),
        y_inner_sb=y_inner1,
        y_inner_port=y_inner3,
        fill_steps=101,
    )
    v2_max = float(diag2["volume"][-1]) if len(diag2["volume"]) else 0.0
    v2_req = float(eq["v2"])
    if v2_req > v2_max + 1e-9:
        return (v2_req - v2_max), "tank2 volume exceeds capacity"

    # Robustness margins: keep away from edge-of-feasibility solutions so
    # engineering reruns don't flip to tiny negative/overflow tank volumes.
    v3_req = float(eq.get("v3", 0.0))
    v3_max = float(diag3["volume"][-1]) if len(diag3["volume"]) else 0.0
    g_t3_low = max(0.0, STRICT_EQ_TANK3_BUFFER_M3 - v3_req)
    g_t3_high = max(0.0, v3_req - max(0.0, v3_max - STRICT_EQ_TANK3_BUFFER_M3))
    g_t2_headroom = max(0.0, STRICT_EQ_TANK2_HEADROOM_BUFFER_M3 - max(0.0, v2_max - v2_req))
    g_margin = g_t3_low + g_t3_high + g_t2_headroom
    if g_margin > 0.0:
        return g_margin, "strict-equilibrium robustness buffer violated"

    return 0.0, "ok"


def evaluate_crane_constraints(
    cd: dict,
    local_half_beam_m: float,
    m_disp_kg: float,
    gm_m: float,
    compensation_moment_nm: float = 0.0,
) -> dict:
    crane = dict(cd.get("Crane", {}))
    swl_max_t = float(crane.get("swl_max_t", TP_WEIGHT_T))
    pivot_h = float(crane.get("pivot_height_m", 1.0))
    pivot_y = float(crane.get("pivot_y_m", 0.0))
    boom_len = float(crane.get("boom_length_m", 30.0))
    pivot_x = float(crane.get("pivot_x_m", 0.0))
    slew_deg = float(crane.get("slewing_angle_deg", 90.0))

    # Required hook position while placing TP outside ship side.
    required_y_abs = local_half_beam_m + TP_RADIUS_M + CRANE_CLEARANCE_M
    required_transverse_reach = max(0.0, required_y_abs - pivot_y)
    # Required absolute hook height: deck (pivot_h − 1m above deck) + TP + rigging + clearance.
    # Equivalently: boom × sin(jib) must exceed CRANE_LOAD_HEIGHT_M + CRANE_RIGGING_HEIGHT_M
    # + CRANE_CLEARANCE_M − CRANE_MIN_PIVOT_HEIGHT_M = 27.5 m.
    required_hook_z = (
        pivot_h
        + CRANE_LOAD_HEIGHT_M
        + CRANE_RIGGING_HEIGHT_M
        + CRANE_CLEARANCE_M
        - CRANE_MIN_PIVOT_HEIGHT_M
    )

    jib_deg = float(crane.get("jib_angle_deg", CRANE_JIB_ANGLE_FIXED_DEG))
    outreach_h = boom_len * np.cos(np.radians(jib_deg))
    transverse_reach = outreach_h * abs(np.sin(np.radians(slew_deg)))

    if boom_len <= 1e-9:
        return {
            "g10": 1.0e3,
            "g11": 90.0,
            "angle_deg": jib_deg,
            "outreach_m": 0.0,
            "required_outreach_m": required_transverse_reach,
            "hook_z_m": pivot_h,
            "heel_deg": 90.0,
            "heeling_moment_mnm": 0.0,
            "compensation_moment_mnm": 0.0,
            "residual_moment_mnm": 0.0,
            "righting_moment_mnm": 0.0,
            "crane_ok": False,
            "swl_eff_t": 0.0,
            "swl_eff_pickup_t": 0.0,
            "swl_max_t": swl_max_t,
            "jib_pickup_deg": 0.0,
            "trim_pickup_deg": 90.0,
            "pivot_x_m": pivot_x,
            "pivot_y_m": pivot_y,
            "slewing_angle_deg": slew_deg,
        }

    angle_deg = jib_deg
    hook_z = pivot_h + boom_len * np.sin(np.radians(angle_deg))

    # SWL derating: linear from CRANE_SWL_ZERO_ANGLE_FACTOR×SWL at 0° to full SWL at CRANE_SWL_FULL_ANGLE_DEG.
    def _derated_swl(jib: float) -> float:
        if jib >= CRANE_SWL_FULL_ANGLE_DEG:
            return swl_max_t
        return swl_max_t * (CRANE_SWL_ZERO_ANGLE_FACTOR + (1.0 - CRANE_SWL_ZERO_ANGLE_FACTOR) * jib / CRANE_SWL_FULL_ANGLE_DEG)

    swl_eff_t = _derated_swl(jib_deg)  # deployment jib (60-80° design bounds → 100% SWL)

    # SWL includes hook/rigging; required lifted mass = TP + rigging mass.
    m_rigging_t = CRANE_RIGGING_MASS_FRAC * swl_max_t
    required_lift_t = TP_WEIGHT_T + m_rigging_t
    g_swl = required_lift_t - swl_eff_t
    g_swl_min = 550.0 - swl_max_t  # enforce minimum crane SWL capacity
    g_reach = required_transverse_reach - transverse_reach
    g_hook = required_hook_z - hook_z

    # Farthest TP pickup check: crane must reach every TP on deck, and derated SWL must suffice.
    tp_list = cd.get("Transition_Pieces", [])
    g_pickup = 0.0
    jib_pickup_deg = jib_deg
    swl_eff_pickup_t = swl_eff_t
    trim_pickup_deg = 0.0
    if tp_list:
        dists = [
            float(np.sqrt((float(tp["x"]) - pivot_x) ** 2 + (float(tp.get("y", 0.0)) - pivot_y) ** 2))
            for tp in tp_list
        ]
        max_dist = float(max(dists))
        i_far = int(np.argmax(dists))
        tp_far = tp_list[i_far]

        if max_dist > boom_len:
            g_pickup = max_dist - boom_len
        else:
            cos_arg = float(np.clip(max_dist / max(boom_len, 1e-9), 0.0, 1.0))
            jib_pickup_deg = float(np.degrees(np.arccos(cos_arg)))
            jib_pickup_deg = float(np.clip(jib_pickup_deg, 0.0, 80.0))
            swl_eff_pickup_t = _derated_swl(jib_pickup_deg)
            g_pickup = required_lift_t - swl_eff_pickup_t

        # Trim and heel when crane swings farthest TP from its deck position to the overboard deployment position.
        x_far = float(tp_far["x"])
        y_far = float(tp_far.get("y", 0.0))
        loa_est = float(cd.get("Length_Loa_m", 100.0))
        lpp_est = loa_est * float(cd.get("Lpp_Loa_ratio", 0.97))
        draft_est = float(cd.get("Target_Draft_m", 4.0))
        bml_est = lpp_est ** 2 / max(12.0 * draft_est, 1e-3)
        delta_lcg_m = required_lift_t * 1000.0 * abs(x_far - pivot_x) / max(m_disp_kg, 1.0)
        trim_pickup_deg = float(np.degrees(np.arctan(delta_lcg_m / max(bml_est, 1e-9))))
        # Heel while hook is above the TP on deck (arm = absolute y from centerline).
        m_heel_pickup_nm = required_lift_t * 1000.0 * 9.81 * abs(y_far)
        m_comp_nm_pickup = max(0.0, float(compensation_moment_nm))
        m_res_pickup_nm = max(0.0, m_heel_pickup_nm - m_comp_nm_pickup)
        ratio_pickup = np.clip(m_res_pickup_nm / max(m_disp_kg * 9.81 * max(gm_m, 1e-9), 1e-9), 0.0, 1.0)
        heel_pickup_deg = float(np.degrees(np.arcsin(ratio_pickup)))
    else:
        heel_pickup_deg = 0.0

    g10 = max(g_swl, g_hook, g_reach, g_pickup, g_swl_min)

    heeling_mass_t = required_lift_t
    m_heeling_nm = heeling_mass_t * 1000.0 * 9.81 * required_y_abs
    m_righting_nm = m_disp_kg * 9.81 * max(gm_m, 1e-9) * np.sin(np.radians(CRANE_MAX_HEEL_DEG))
    m_comp_nm = max(0.0, float(compensation_moment_nm))
    m_residual_nm = max(0.0, m_heeling_nm - m_comp_nm)
    ratio_heel = np.clip(m_residual_nm / max(m_disp_kg * 9.81 * max(gm_m, 1e-9), 1e-9), 0.0, 1.0)
    heel_deg = float(np.degrees(np.arcsin(ratio_heel)))
    g11 = max(heel_deg - CRANE_MAX_HEEL_DEG, trim_pickup_deg - CRANE_MAX_HEEL_DEG, heel_pickup_deg - CRANE_MAX_HEEL_DEG)

    return {
        "g10": float(g10),
        "g11": float(g11),
        "angle_deg": angle_deg,
        "outreach_m": float(transverse_reach),
        "required_outreach_m": float(required_transverse_reach),
        "hook_z_m": float(hook_z),
        "heel_deg": heel_deg,
        "heeling_moment_mnm": float(m_heeling_nm / 1e6),
        "compensation_moment_mnm": float(m_comp_nm / 1e6),
        "residual_moment_mnm": float(m_residual_nm / 1e6),
        "righting_moment_mnm": float(m_righting_nm / 1e6),
        "crane_ok": bool(g10 <= 0.0 and g11 <= 0.0),
        "swl_eff_t": float(swl_eff_t),
        "swl_eff_pickup_t": float(swl_eff_pickup_t),
        "swl_max_t": float(swl_max_t),
        "jib_pickup_deg": float(jib_pickup_deg),
        "trim_pickup_deg": float(trim_pickup_deg),
        "heel_pickup_deg": float(heel_pickup_deg),
        "pivot_x_m": float(pivot_x),
        "pivot_y_m": float(pivot_y),
        "slewing_angle_deg": float(slew_deg),
    }


def evaluate_one_unit(u: np.ndarray, ctx: DesignContext, min_tps: int = 0, ship_mode: str = "both") -> dict:
    big = 1.0e6
    min_tps = max(0, int(min_tps))
    ship_mode = str(ship_mode)
    _no_crane = ship_mode == "tps-only"
    _no_tps   = ship_mode == "crane-only"
    # g1..g15 fast hydro/geometry/crane/strength + g16 fast transverse moment
    out = {
        "F": np.array([big, big, big], dtype=float),
        "G": np.array(
            [50.0]*16,
            dtype=float,
        ),
        "cfg": None,
        "metrics": {},
    }
    cfg_path = EVAL_CFG_PATH.with_name(f"{EVAL_CFG_PATH.stem}_{os.getpid()}{EVAL_CFG_PATH.suffix}")
    try:
        cd = decode_design(u, ctx)
        draft = float(cd["Target_Draft_m"])
        doa = float(cd["Depth_Doa_m"])
        freeboard = doa - draft

        g_min_draft = MIN_DRAFT_M - draft

        write_eval_config(cd, cfg_path=cfg_path)
        geo = pfs.build_hull_loft(N_u=LOFT_N_U_FAST, N_t=LOFT_N_T_FAST)
        X, Y, Z = geo["X_surf"], geo["Y_surf"], geo["Z_surf"]
        LPP = float(geo["LPP"])
        D = float(geo["D"])
        x_mid = X.mean(axis=1)

        tp_all, deck_width_violation = pack_transition_pieces_hex(X, Y, LPP)
        n_tps = len(tp_all)
        cd["Transition_Pieces"] = tp_all

        V_disp = _displacement_at_draft(X, Y, Z, draft, 0.0, LPP, n_z=40)
        if V_disp <= 1e-9:
            out["G"] = np.array(
                [10.0, MIN_FREEBOARD_M - freeboard, deck_width_violation, 100.0,
                 g_min_draft, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 draft - MAX_DRAFT_M, float(min_tps - n_tps), 1.0, 1.0, 1.0],
                dtype=float,
            )
            return out

        m_disp_kg = V_disp * 1025.0
        m_steel_kg = estimate_lsw_kg(
            X=X,
            Y=Y,
            Z=Z,
            hull_t_mm=float(cd["Hull_Thickness_mm"]),
            boa_m=float(cd["Breadth_Boa_m"]),
            doa_m=doa,
        )
        t1_fill_pct = float(np.clip(cd.get("Tank1_Fill_pct", 0.0), 0.0, 95.0))
        b_half = 0.5 * float(cd["Breadth_Boa_m"])
        t1_w = float(cd["Tank1_Width_m"])
        y_inner1 = max(0.0, b_half - t1_w)
        v_t1_m3, vcg_t1_m = estimate_side_tank_fill_state(
            X=X,
            Y=Y,
            Z=Z,
            y_inner=y_inner1,
            fill_pct=t1_fill_pct,
            h_max=doa,
        )
        m_t1_kg = v_t1_m3 * 1025.0

        crane = dict(cd.get("Crane", {}))
        crane_swl_t = float(crane.get("swl_max_t", TP_WEIGHT_T))
        crane_pivot_x = float(crane.get("pivot_x_m", 0.0))
        i_pivot = int(np.argmin(np.abs(x_mid - crane_pivot_x)))
        local_half_beam = float(np.max(Y[i_pivot]))
        crane_y_frac = float(crane.get("pivot_y_frac", 1.0))
        crane_pivot_y = crane_y_frac * max(0.0, local_half_beam - CRANE_PIVOT_EDGE_INSET_M)
        crane["pivot_y_m"] = float(crane_pivot_y)
        cd["Crane"] = crane
        crane_pivot_h = float(crane.get("pivot_height_m", 1.0))
        crane_jib_deg = float(crane.get("jib_angle_deg", CRANE_JIB_ANGLE_FIXED_DEG))

        # Remove TPs whose footprint overlaps the crane house (TP_RADIUS clearance).
        _house_l = float(np.clip(3.0 + 0.004 * crane_swl_t, 3.0, 8.0))
        _house_w = float(np.clip(3.0 + 0.003 * crane_swl_t, 3.0, 7.0))
        _excl_dx = 0.5 * _house_l + TP_RADIUS_M + CRANE_CLEARANCE_M
        _excl_dy = 0.5 * _house_w + TP_RADIUS_M + CRANE_CLEARANCE_M
        tp_all = [
            tp for tp in tp_all
            if not (
                abs(tp["x"] - crane_pivot_x) < _excl_dx
                and abs(tp["y"] - crane_pivot_y) < _excl_dy
            )
        ]
        crane_boom_len = float(crane.get("boom_length_m", 30.0))
        m_crane_house_kg = CRANE_HOUSE_MASS_FRAC * crane_swl_t * 1000.0
        m_crane_boom_kg = CRANE_BOOM_MASS_FRAC * crane_swl_t * 1000.0
        m_crane_rig_kg = CRANE_RIGGING_MASS_FRAC * crane_swl_t * 1000.0
        m_crane_kg = m_crane_house_kg + m_crane_boom_kg + m_crane_rig_kg
        boom_cg_z = crane_pivot_h + 0.5 * crane_boom_len * np.sin(np.radians(crane_jib_deg))
        vcg_crane = (
            m_crane_house_kg * crane_pivot_h
            + m_crane_boom_kg * boom_cg_z
            + m_crane_rig_kg * crane_pivot_h
        ) / max(m_crane_kg, 1e-9)
        # Boom tip height: suspended load acts here for GM purposes (marine stability convention).
        hook_z = crane_pivot_h + crane_boom_len * np.sin(np.radians(crane_jib_deg))
        m_lifted_tp_kg = 0.0  # No extra TP; we will shift one from the deck during lift checks.

        # tps-only mode: remove crane mass, VCG, and suppress crane in config so
        # the 3D renderer (which checks swl_max_t > 0) does not draw a crane.
        if _no_crane:
            m_crane_kg = 0.0
            vcg_crane = 0.0
            crane["swl_max_t"] = 0.0
            cd["Crane"] = crane

        # Sort TPs by distance from LCB: partial selections cluster near buoyancy centre,
        # minimising trim disturbance and making equilibrium easier to satisfy.
        lcb_x, _, _ = center_of_buoyancy(X, Y, Z, draft, 0.0, LPP, n_z=20)
        tp_all.sort(key=lambda tp: abs(tp["x"] - lcb_x))

        payload_control_frac = float(np.clip(cd.get("Payload_Control_Fraction", 1.0), 0.0, 1.0))
        n_tps_geo_max = int(len(tp_all))
        n_tps = int(np.rint(payload_control_frac * n_tps_geo_max))
        n_tps = int(np.clip(n_tps, 0, n_tps_geo_max))
        tp_list = tp_all[:n_tps]
        cd["Transition_Pieces"] = tp_list
        m_tps_kg = n_tps * TP_WEIGHT_T * 1000.0

        # crane-only mode: strip all TPs from the design.
        if _no_tps:
            n_tps = 0
            tp_list = []
            cd["Transition_Pieces"] = []
            m_tps_kg = 0.0
        # Payload budget: displacement minus crane, tank1 and minimum ballast.
        tp_mass_budget_kg = max(0.0, m_disp_kg - m_steel_kg - m_t1_kg - m_crane_kg - MIN_BALLAST_FRAC * m_disp_kg)
        n_tps_mass_cap = int(max(0.0, np.floor(tp_mass_budget_kg / (TP_WEIGHT_T * 1000.0))))

        # Tank 2 geometric validation to avoid out-of-hull/overrun designs.
        t2_len_pct_lpp = float(cd.get("Tank2_Length_pct_Lpp", 0.0))
        t2_len_m = LPP * t2_len_pct_lpp / 100.0
        t2_center = float(cd.get("Tank2_Center_from_AP_m", 0.5 * LPP))
        t2_x0 = t2_center - 0.5 * t2_len_m
        t2_x1 = t2_center + 0.5 * t2_len_m
        g9_x = max(0.0, -t2_x0) + max(0.0, t2_x1 - LPP)
        i0 = int(np.argmin(np.abs(X.mean(axis=1) - t2_x0)))
        i1 = int(np.argmin(np.abs(X.mean(axis=1) - t2_x1)))
        hw0 = float(np.max(Y[i0]))
        hw1 = float(np.max(Y[i1]))
        y_inner3 = max(0.0, b_half - float(cd["Tank3_Width_m"]))
        v_t3_max_m3, _ = estimate_side_tank_fill_state(
            X=X,
            Y=Y,
            Z=Z,
            y_inner=y_inner3,
            fill_pct=100.0,
            h_max=doa,
        )
        m_t3_max_kg = v_t3_max_m3 * 1025.0
        tcg_t3_eff = max(y_inner3, 0.5)
        m_compensate_nm = m_t3_max_kg * 9.81 * tcg_t3_eff
        y_req = max(y_inner1, y_inner3)
        g9_w = max(0.0, y_req - hw0) + max(0.0, y_req - hw1)
        g9_geom = g9_x + g9_w

        mass_required_kg = m_steel_kg + m_tps_kg + m_t1_kg + m_crane_kg + MIN_BALLAST_FRAC * m_disp_kg
        g_mass = (mass_required_kg - m_disp_kg) / 1000.0

        wp = waterplane_data(X, Y, Z, draft, x_min=0.0, x_max=LPP)
        Ix_wp = float(wp["Ix"])
        _, _, vcb = center_of_buoyancy(X, Y, Z, draft, 0.0, LPP, n_z=40)
        BM = Ix_wp / max(V_disp, 1e-9)

        m_ballast = max(0.0, m_disp_kg - m_steel_kg - m_tps_kg - m_t1_kg - m_crane_kg)
        kg_empty = 0.60 * D
        kg_tps = D + TP_CG_HEIGHT_M
        kg_ballast = 0.30 * D
        
        # Calculate base transit KG
        KG_transit = (
            m_steel_kg * kg_empty
            + m_tps_kg * kg_tps
            + m_t1_kg * vcg_t1_m
            + m_ballast * kg_ballast
            + m_crane_kg * vcg_crane
        ) / max(m_disp_kg, 1e-9)
        gm_transit = vcb + BM - KG_transit - 0.10
        
        # Calculate lift condition GM (shift one TP from deck to hook if available)
        lift_mass_kg = TP_WEIGHT_T * 1000.0 if n_tps > 0 else 0.0
        KG_lift = KG_transit + lift_mass_kg * (hook_z - kg_tps) / max(m_disp_kg, 1e-9)
        gm_lift = vcb + BM - KG_lift - 0.10
        
        # For general constraints (g1), use transit GM
        gm = gm_transit

        crane_eval = evaluate_crane_constraints(
            cd=cd,
            local_half_beam_m=local_half_beam,
            m_disp_kg=m_disp_kg,
            gm_m=gm_lift,
            compensation_moment_nm=m_compensate_nm,
        )

        lsw_t = (m_steel_kg + m_ballast) / 1000.0
        # Empty ship weight objective: steel-only mass (no crane, no TPs, no ballast).
        ship_weight_t = m_steel_kg / 1000.0
        payload_t = n_tps * TP_WEIGHT_T
        crane_load_mn = float(crane_eval["swl_eff_t"]) * 9.81 / 1000.0
        if payload_t <= 1e-9:
            lsw_per_payload = LSW_PER_PAYLOAD_CAP
        else:
            lsw_per_payload = min(lsw_t / payload_t, LSW_PER_PAYLOAD_CAP)

        v_ms = DESIGN_SPEED_KN * 0.514444
        fn = v_ms / np.sqrt(9.81 * max(LPP, 1e-9))

        f2 = ship_weight_t
        f3 = -float(n_tps)

        g1 = MIN_GM_M - gm
        g2 = MIN_FREEBOARD_M - freeboard
        g3 = deck_width_violation
        g4 = g_mass
        g5 = g_min_draft
        g6 = fn - MAX_FN
        g7 = (m_ballast / max(m_disp_kg, 1e-9)) - MAX_BALLAST_FRAC
        t_roll_s = 0.7 * float(cd["Breadth_Boa_m"]) / np.sqrt(max(gm, 1e-6))
        g8 = MIN_ROLL_PERIOD_S - t_roll_s
        g14 = t_roll_s - MAX_ROLL_PERIOD_S

        # Fast transverse moment check (om X-as):
        # crane + lifted TP create an asymmetric moment; tank 3 (BB) must compensate.
        slew_rad_ev = np.radians(float(crane.get("slewing_angle_deg", 90.0)))
        outreach_h_ev = crane_boom_len * np.cos(np.radians(crane_jib_deg))
        hook_y = crane_pivot_y + outreach_h_ev * np.sin(slew_rad_ev)

        t1_w_approx = b_half - y_inner1
        tcg_t1_approx = -(y_inner1 + 0.5 * t1_w_approx)  # SB centroid (negative)
        M_crane_trans_Nm  = m_crane_kg * crane_pivot_y * 9.81
        M_lifted_trans_Nm = lift_mass_kg * hook_y * 9.81
        M_t1_trans_Nm     = m_t1_kg * tcg_t1_approx * 9.81
        # Total moment requiring T3 compensation (must be positive = port-ward)
        M_t3_needed_Nm = -(M_crane_trans_Nm + M_lifted_trans_Nm + M_t1_trans_Nm)
        M_t3_max_Nm    = m_t3_max_kg * 9.81 * tcg_t3_eff
        g16_dir = -M_t3_needed_Nm / 1e6         # > 0 if net port moment (T3 can't help from same side)
        g16_cap = (M_t3_needed_Nm - M_t3_max_Nm) / 1e6  # > 0 if T3 capacity exceeded
        g16 = max(g16_dir, g16_cap)

        # Fast approximate strength (box-beam section modulus vs simplified BM).
        hull_t = float(cd["Hull_Thickness_mm"]) / 1000.0
        I_box = hull_t * (float(cd["Breadth_Boa_m"]) * doa ** 2 / 2.0 + doa ** 3 / 6.0)
        Z_box = I_box / max(doa / 2.0, 1e-3)
        # Concentrated load bending moment: simply-supported beam, each load at its x position.
        M_fast = sum(
            TP_WEIGHT_T * 1000.0 * 9.81 * float(tp["x"]) * max(0.0, LPP - float(tp["x"])) / max(LPP, 1.0)
            for tp in tp_list
        )
        M_fast += (m_crane_kg + lift_mass_kg) * 9.81 * crane_pivot_x * max(0.0, LPP - crane_pivot_x) / max(LPP, 1.0)
        sigma_fast = M_fast / max(Z_box, 1e-9) / 1e6 * 0.25  # 0.25 = calibration (distributed buoyancy relief)
        g15 = sigma_fast - STRENGTH_SIGMA_ALLOW_MPA

        g10 = crane_eval["g10"]
        g11 = crane_eval["g11"]
        g12 = draft - MAX_DRAFT_M
        g13 = float(min_tps - n_tps)

        # Mode overrides: bypass constraints that don't apply.
        if _no_crane:
            # tps-only: no crane on board, crane constraints don't apply.
            g10 = -1.0
            g11 = -1.0
        if _no_tps:
            # crane-only: no TPs, min-TPs constraint doesn't apply.
            g13 = -1.0

        # Stage A prefilter: reject clearly infeasible designs before expensive
        # strict-equilibrium and resistance calculations.
        stage_a_checks = [
            ("g1_gm", g1),
            ("g2_freeboard", g2),
            ("g3_tp_deck_width", g3),
            ("g4_mass_reserve", g4),
            ("g5_min_draft", g5),
            ("g6_froude_limit", g6),
            ("g7_ballast_fraction", g7),
            ("g8_roll_period", g8),
            ("g10_crane_swl_and_geometry", g10),
            ("g11_crane_heel_limit", g11),
            ("g12_max_draft", g12),
            ("g13_min_transition_pieces", g13),
            ("g14_roll_period_max", g14),
            ("g15_fast_strength", g15),
            ("g16_transverse_moment", g16),
        ]
        stage_a_failed = [name for name, val in stage_a_checks if float(val) > 0.0]
        stage_a_passed = len(stage_a_failed) == 0
        if not stage_a_passed:
            out["G"] = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9_geom, g10, g11, g12, g13, g14, g15, g16], dtype=float)
            out["cfg"] = cd
            out["metrics"] = {
                "Rtot_kN": big,
                "LSW_t": (m_steel_kg + m_ballast) / 1000.0,
                "ShipWeight_t": m_steel_kg / 1000.0,
                "LSW_per_payload": LSW_PER_PAYLOAD_CAP,
                "N_TPs": int(n_tps),
                "N_TPs_geo_max": int(n_tps_geo_max),
                "N_TPs_mass_cap": int(n_tps_mass_cap),
                "Payload_t": float(n_tps * TP_WEIGHT_T),
                "Payload_Control_Fraction": float(payload_control_frac),
                "Crane_Load_MN": float(crane_eval["swl_eff_t"]) * 9.81 / 1000.0,
                "GM_m": gm,
                "Freeboard_m": freeboard,
                "Draft_m": draft,
                "LOA_m": float(cd["Length_Loa_m"]),
                "BOA_m": float(cd["Breadth_Boa_m"]),
                "DOA_m": doa,
                "Fn": float(fn),
                "MassReserve_t": (m_disp_kg - mass_required_kg) / 1000.0,
                "Ballast_t": m_ballast / 1000.0,
                "Ballast_frac_disp": m_ballast / max(m_disp_kg, 1e-9),
                "Crane_mass_t": m_crane_kg / 1000.0,
                "Crane_vcg_m": float(vcg_crane),
                "Tank1_Fill_pct": t1_fill_pct,
                "Tank1_Liquid_t": m_t1_kg / 1000.0,
                "Tank3_MaxLiquid_t": m_t3_max_kg / 1000.0,
                "Tank2_x0_m": float(t2_x0),
                "Tank2_x1_m": float(t2_x1),
                "Tank2_geom_violation": float(g9_geom),
                "StrictEq_violation": float("nan"),
                "StrictEq_reason": "skipped_stage_a",
                "StrictEq_ok": False,
                "T_roll_s": float(t_roll_s),
                "RollPeriodMax_violation_s": float(g14),
                "DraftMax_violation_m": float(g12),
                "MinTP_violation": float(g13),
                "MinTP_required": int(min_tps),
                "FastStrength_sigma_MPa": float(sigma_fast),
                "FastStrength_violation": float(g15),
                "Transverse_M_crane_MNm": float(M_crane_trans_Nm / 1e6),
                "Transverse_M_lifted_MNm": float(M_lifted_trans_Nm / 1e6),
                "Transverse_M_t3_needed_MNm": float(M_t3_needed_Nm / 1e6),
                "Transverse_M_t3_max_MNm": float(M_t3_max_Nm / 1e6),
                "Transverse_violation": float(g16),
                "Crane_ok": bool(crane_eval["crane_ok"]),
                "Crane_Swl_max_t": float(crane_eval["swl_max_t"]),
                "Crane_Swl_effective_t": float(crane_eval["swl_eff_t"]),
                "Crane_pivot_x_m": float(crane_eval["pivot_x_m"]),
                "Crane_pivot_y_m": float(crane_eval["pivot_y_m"]),
                "Crane_slewing_angle_deg": float(crane_eval["slewing_angle_deg"]),
                "Crane_required_outreach_m": float(crane_eval["required_outreach_m"]),
                "Crane_actual_outreach_m": float(crane_eval["outreach_m"]),
                "Crane_required_angle_deg": float(crane_eval["angle_deg"]),
                "Crane_hook_height_m": float(crane_eval["hook_z_m"]),
                "Crane_heel_deg": float(crane_eval["heel_deg"]),
                "Crane_heeling_moment_MNm": float(crane_eval["heeling_moment_mnm"]),
                "Crane_compensation_moment_MNm": float(crane_eval["compensation_moment_mnm"]),
                "Crane_residual_moment_MNm": float(crane_eval["residual_moment_mnm"]),
                "Crane_righting_moment_MNm": float(crane_eval["righting_moment_mnm"]),
                "StageA_passed": False,
                "StageA_failed_constraints": stage_a_failed,
            }
            return out

        g9_eq, eq_reason = strict_equilibrium_violation(
            cd=cd,
            X=X,
            Y=Y,
            Z=Z,
            LPP=LPP,
            D=D,
        )
        g9 = max(g9_geom, g9_eq)

        lcb, _, _ = center_of_buoyancy(X, Y, Z, draft, 0.0, LPP, n_z=40)
        hm = compute_hull_coefficients(X, Y, Z, draft, LPP, V_disp, lcb)
        BWL = float(hm["BWL"])
        CP = float(hm["CP"])
        lcb_pct = float(hm["lcb_pct"])
        x_stern_sample = 0.05 * LPP
        i_stern = int(np.argmin(np.abs(x_mid - x_stern_sample)))
        y_stern = Y[i_stern]
        z_stern = Z[i_stern]
        beam_stern = 2.0 * hull_hw_at_z(y_stern, z_stern, draft)
        if beam_stern <= 1e-6:
            beam_stern = max(1e-6, 2.0 * float(np.max(y_stern)))
        stern_shape = analyze_stern_shape(
            y_cs=y_stern,
            z_cs=z_stern,
            beam=beam_stern,
            draft=draft,
        )
        cstern = float(stern_shape["cstern"])
        c_beta_stern = float(stern_shape["c_beta"])
        cd["Cstern"] = cstern
        LR = LPP * (1.0 - CP + 0.06 * CP * lcb_pct / (4.0 * CP - 1.0 + 1e-12))
        iE = compute_ie_regression(LPP, BWL, CP, hm["CWP"], lcb_pct, V_disp, LR)
        S = compute_s_wet_hm(LPP, BWL, draft, hm["CB"], hm["CM"], hm["CWP"])
        AT = compute_at(Y, Z, draft)
        res = holtrop_mennen(
            LPP,
            BWL,
            draft,
            V_disp,
            hm["CB"],
            CP,
            hm["CM"],
            hm["CWP"],
            lcb_pct,
            S,
            AT,
            iE,
            cstern,
            np.array([DESIGN_SPEED_KN]),
            method=1,
        )
        rtot_kn = float(res[0]["Rtot_kN"]) if res else big
        f1 = rtot_kn

        out["F"] = np.array([f1, f2, f3], dtype=float)
        out["G"] = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16], dtype=float)
        out["cfg"] = cd
        out["metrics"] = {
            "Rtot_kN": rtot_kn,
            "LSW_t": lsw_t,
            "ShipWeight_t": ship_weight_t,
            "LSW_per_payload": lsw_per_payload,
            "N_TPs": int(n_tps),
            "N_TPs_geo_max": int(n_tps_geo_max),
            "N_TPs_mass_cap": int(n_tps_mass_cap),
            "Payload_t": float(payload_t),
            "Payload_Control_Fraction": float(payload_control_frac),
            "Crane_Load_MN": crane_load_mn,
            "GM_m": gm,
            "Freeboard_m": freeboard,
            "Draft_m": draft,
            "LOA_m": float(cd["Length_Loa_m"]),
            "BOA_m": float(cd["Breadth_Boa_m"]),
            "DOA_m": doa,
            "CB": float(hm["CB"]),
            "CP": CP,
            "Cbeta_stern": c_beta_stern,
            "Cstern": cstern,
            "Fn": float(fn),
            "iE_deg": float(iE),
            "S_wet_m2": float(S),
            "MassReserve_t": (m_disp_kg - mass_required_kg) / 1000.0,
            "Ballast_t": m_ballast / 1000.0,
            "Ballast_frac_disp": m_ballast / max(m_disp_kg, 1e-9),
            "Crane_mass_t": m_crane_kg / 1000.0,
            "Crane_vcg_m": float(vcg_crane),
            "Tank1_Fill_pct": t1_fill_pct,
            "Side_Flare_deg": float(cd.get("Side_Flare_deg", 0.0)),
            "Side_Flare_Rotation_Point": float(cd.get("Side_Flare_Rotation_Point", 0.0)),
            "Tank1_Liquid_t": m_t1_kg / 1000.0,
            "Tank3_MaxLiquid_t": m_t3_max_kg / 1000.0,
            "Tank2_x0_m": float(t2_x0),
            "Tank2_x1_m": float(t2_x1),
            "Tank2_geom_violation": float(g9_geom),
            "StrictEq_violation": float(g9_eq),
            "StrictEq_reason": eq_reason,
            "StrictEq_ok": bool(g9_eq <= 0.0),
            "T_roll_s": float(t_roll_s),
            "RollPeriodMax_violation_s": float(g14),
            "DraftMax_violation_m": float(g12),
            "MinTP_violation": float(g13),
            "MinTP_required": int(min_tps),
            "FastStrength_sigma_MPa": float(sigma_fast),
            "FastStrength_violation": float(g15),
            "Transverse_M_crane_MNm": float(M_crane_trans_Nm / 1e6),
            "Transverse_M_lifted_MNm": float(M_lifted_trans_Nm / 1e6),
            "Transverse_M_t3_needed_MNm": float(M_t3_needed_Nm / 1e6),
            "Transverse_M_t3_max_MNm": float(M_t3_max_Nm / 1e6),
            "Transverse_violation": float(g16),
            "Crane_ok": bool(crane_eval["crane_ok"]),
            "Crane_Swl_max_t": float(crane_eval["swl_max_t"]),
            "Crane_Swl_effective_t": float(crane_eval["swl_eff_t"]),
            "Crane_pivot_x_m": float(crane_eval["pivot_x_m"]),
            "Crane_pivot_y_m": float(crane_eval["pivot_y_m"]),
            "Crane_slewing_angle_deg": float(crane_eval["slewing_angle_deg"]),
            "Crane_required_outreach_m": float(crane_eval["required_outreach_m"]),
            "Crane_actual_outreach_m": float(crane_eval["outreach_m"]),
            "Crane_required_angle_deg": float(crane_eval["angle_deg"]),
            "Crane_hook_height_m": float(crane_eval["hook_z_m"]),
            "Crane_heel_deg": float(crane_eval["heel_deg"]),
            "Crane_heeling_moment_MNm": float(crane_eval["heeling_moment_mnm"]),
            "Crane_compensation_moment_MNm": float(crane_eval["compensation_moment_mnm"]),
            "Crane_residual_moment_MNm": float(crane_eval["residual_moment_mnm"]),
            "Crane_righting_moment_MNm": float(crane_eval["righting_moment_mnm"]),
            "StageA_passed": True,
            "StageA_failed_constraints": [],
        }
        return out
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        print(
            f"DEBUG EVAL FOUT [pid={os.getpid()} cfg={cfg_path.name}]: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return out
    finally:
        try:
            if cfg_path.exists():
                cfg_path.unlink()
        except Exception:
            pass
