# -*- coding: utf-8 -*-
"""
strength.py — Longitudinal strength calculation driven by parametric hull geometry.

Mirrors langsscheepse_sterkte.py from the reference code, but all input data is
computed directly from the loft (build_hull_loft) and config.json.

Entry point: run_strength_calculation(geo, cfg_dict) → result dict
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, interp1d

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_STEEL   = 7850.0    # kg/m³
RHO_WATER   = 1025.0    # kg/m³
G           = 9.81      # m/s²
FAC         = 2.1       # construction factor (mass × FAC for weight)
E           = 205e9     # Young's modulus steel [Pa]
SIGMA_ALLOW = 190.0     # allowable stress [MPa] for S235 design criterion
WALL_T      = 0.010     # bulkhead / wall thickness [m]


# ---------------------------------------------------------------------------
# Helper: interp with zero outside range
# ---------------------------------------------------------------------------

def _interp0(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Interpolate; return 0 outside the source range."""
    f = interp1d(x_old, y_old, bounds_error=False, fill_value=0.0)
    return f(x_new)


# ---------------------------------------------------------------------------
# Semi-circle load
# ---------------------------------------------------------------------------

def _add_semicircle_load(load: np.ndarray, x_fijn: np.ndarray,
                          x_center: float, radius: float, weight_n: float) -> None:
    """Add a semi-circle distributed load centred at x_center.

    weight_n : total weight [N]
    """
    i0 = int(np.searchsorted(x_fijn, x_center - radius))
    i1 = int(np.searchsorted(x_fijn, x_center + radius))
    n  = i1 - i0
    if n < 2:
        # fallback: point load at nearest index
        idx = int(np.searchsorted(x_fijn, x_center))
        idx = min(idx, len(x_fijn) - 1)
        dx  = (x_fijn[1] - x_fijn[0]) if len(x_fijn) > 1 else 1.0
        load[idx] += weight_n / dx
        return
    height = weight_n / radius / np.pi * 2.0
    boog   = np.linspace(-1.0, 1.0, n)
    load[i0:i1] += height * np.sin(np.arccos(boog))


# ---------------------------------------------------------------------------
# Tank equilibrium (mirrors bepaalTankEvenwicht)
# ---------------------------------------------------------------------------

def _bepaal_tank_evenwicht(q_evenwicht: float,
                            diag1: dict, diag2: dict, diag3: dict,
                            v1_init: float, v2_init: float, v3_init: float) -> dict:
    """Adjust tank volumes to balance the net load.

    diag1/2/3 : fill-diagram dicts (keys 'volume', 'fill_pct')
    """
    delta_v = -q_evenwicht / (G * RHO_WATER)

    pct1   = diag1['fill_pct']
    pct2   = diag2['fill_pct']
    pct3   = diag3['fill_pct']
    vol1   = diag1['volume']
    vol2   = diag2['volume']
    vol3   = diag3['volume']

    # Min volume tank2 at 20 % fill
    cs2 = CubicSpline(pct2, vol2)
    v2_min = float(cs2(20.0))
    v2_max = float(vol2[-1])

    dv2 = float(np.clip(delta_v, v2_min - v2_init, v2_max - v2_init))
    dv_rest = delta_v - dv2

    dv13_max = min(float(vol1[-1]) - v1_init, float(vol3[-1]) - v3_init)
    dv13_min = max(-v1_init, -v3_init)
    dv13 = float(np.clip(dv_rest / 2.0, dv13_min, dv13_max))

    v1_new = v1_init + dv13
    v2_new = v2_init + dv2
    v3_new = v3_init + dv13

    # Convert volumes back to percentages using inverse splines
    # Filter out duplicate volume values (0-fill rows have identical zero volumes)
    def _unique_cs(v_arr, p_arr):
        """Build CubicSpline(volume → pct) skipping duplicate x values."""
        v_arr = np.asarray(v_arr, dtype=float)
        p_arr = np.asarray(p_arr, dtype=float)
        _, idx = np.unique(v_arr, return_index=True)
        v_u, p_u = v_arr[idx], p_arr[idx]
        if len(v_u) < 2:
            return None, v_u, p_u
        return CubicSpline(v_u, p_u), v_u, p_u

    cs1_inv, v1u, p1u = _unique_cs(vol1, pct1)
    cs2_inv, v2u, p2u = _unique_cs(vol2, pct2)
    cs3_inv, v3u, p3u = _unique_cs(vol3, pct3)

    def _pct_from_vol(cs, v_u, p_u, v_target):
        v_clamp = float(np.clip(v_target, v_u[0], v_u[-1]))
        if cs is None:
            return float(p_u[0]) if len(p_u) else 0.0
        return float(cs(v_clamp))

    pct1_new = _pct_from_vol(cs1_inv, v1u, p1u, v1_new)
    pct2_new = _pct_from_vol(cs2_inv, v2u, p2u, v2_new)
    pct3_new = _pct_from_vol(cs3_inv, v3u, p3u, v3_new)

    return dict(
        v1=v1_new, v2=v2_new, v3=v3_new,
        pct1=pct1_new, pct2=pct2_new, pct3=pct3_new,
        dv_rest=float(dv_rest - 2.0 * dv13),
    )


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def run_strength_calculation(geo: dict, cfg_dict: dict,
                              shell_data: dict,
                              buoyant_x: np.ndarray, buoyant_csa: np.ndarray,
                              tank1_x: np.ndarray, tank1_csa: np.ndarray,
                              tank2_x: np.ndarray, tank2_csa: np.ndarray,
                              tank3_x: np.ndarray, tank3_csa: np.ndarray,
                              diag1: dict, diag2: dict, diag3: dict,
                              stern_area: float,
                              bhd_data: list,
                              hull_mass_data: dict,
                              stability_result: dict,
                              ) -> dict:
    """Run the full longitudinal strength calculation.

    Parameters
    ----------
    geo          : dict from build_hull_loft()
    cfg_dict     : raw config.json dict
    shell_data   : dict from compute_shell_csa()
    buoyant_x    : x-positions of buoyant CSA [m]
    buoyant_csa  : buoyant cross-sectional area at each x [m²]
    tank1/2/3_x  : x-positions of tank CSA (at 100 % fill) [m]
    tank1/2/3_csa: tank CSA at 100 % fill [m²]
    diag1/2/3    : fill-diagram dicts from build_tank_fill_diagram()
    stern_area   : area of transom face [m²]
    bhd_data     : list of dicts {area, x_min, x_max, lcg, vcg}
    hull_mass_data: dict with area and lcg of hull panels for weight
    stability_result: dict from run_stability_calculation()

    Returns
    -------
    Comprehensive result dict (see module docstring).
    """
    LPP      = geo['LPP']
    B_half   = geo['B_half']
    D        = geo['D']

    hull_t_mm = float(cfg_dict.get('Hull_Thickness_mm', 8.0))
    hull_t    = hull_t_mm / 1000.0

    TPs = cfg_dict.get('Transition_Pieces', [])
    crane = cfg_dict.get('Crane', {}) if isinstance(cfg_dict.get('Crane', {}), dict) else {}
    crane_boom_len = float(crane.get('boom_length_m', 0.0))

    crane_pivot_x = float(crane.get('pivot_x_m', 0.5 * LPP))
    crane_pivot_h = float(crane.get('pivot_height_m', 1.0))

    # Stowaway crane: structure mass only (house + boom, no hook).
    # SWL derived from TP mass; boom stowed horizontal, pointing forward.
    if crane_boom_len > 0.0 and TPs:
        one_tp_kg = float(TPs[0]['weight_t']) * 1000.0
        kraan_swl_kg = one_tp_kg / 0.94
    elif crane_boom_len > 0.0:
        kraan_swl_kg = float(crane.get('swl_max_t', 0.0)) * 1000.0
    else:
        kraan_swl_kg = 0.0
    crane_m_house_kg = 0.34 * kraan_swl_kg
    crane_m_boom_kg  = 0.17 * kraan_swl_kg
    crane_boom_cg_x  = crane_pivot_x + 0.5 * crane_boom_len  # stowed horizontal, pointing forward

    # --- Integration grid -------------------------------------------------
    x_fijn = np.linspace(0.0, LPP, 10000)

    # --- Shell (hull plating) distributed weight --------------------------
    # q_huid = CSA_1mm(x) * t_mm * rho_steel * FAC * g  [N/m]
    huid_csa = _interp0(shell_data['x'], shell_data['shell_csa_1mm'], x_fijn)
    q_huid   = huid_csa * hull_t_mm * RHO_STEEL * FAC * G   # N/m

    # --- Tank distributed loads at initial fill ---------------------------
    # CSA at 100 % fill; scale by actual fill fraction
    t1_fill = float(cfg_dict.get('Tank1_Fill_pct', 50.0)) / 100.0
    t2_fill = float(cfg_dict.get('Tank2_Fill_pct', 75.0)) / 100.0
    t3_fill = float(cfg_dict.get('Tank3_Fill_pct', 50.0)) / 100.0

    # Volume at initial fill (from diagrams)
    cs1 = CubicSpline(diag1['fill_pct'], diag1['volume'])
    cs2 = CubicSpline(diag2['fill_pct'], diag2['volume'])
    cs3 = CubicSpline(diag3['fill_pct'], diag3['volume'])

    v1_init = float(cs1(t1_fill * 100.0))
    v2_init = float(cs2(t2_fill * 100.0))
    v3_init = float(cs3(t3_fill * 100.0))

    # Volume at 100 % fill (CSA integrated)
    v1_100 = float(np.trapezoid(tank1_csa, tank1_x)) if len(tank1_x) > 1 else 1.0
    v2_100 = float(np.trapezoid(tank2_csa, tank2_x)) if len(tank2_x) > 1 else 1.0
    v3_100 = float(np.trapezoid(tank3_csa, tank3_x)) if len(tank3_x) > 1 else 1.0

    f1 = v1_init / v1_100 if v1_100 > 1e-9 else t1_fill
    f2 = v2_init / v2_100 if v2_100 > 1e-9 else t2_fill
    f3 = v3_init / v3_100 if v3_100 > 1e-9 else t3_fill

    q_tank1 = _interp0(tank1_x, tank1_csa, x_fijn) * f1 * RHO_WATER * G
    q_tank2 = _interp0(tank2_x, tank2_csa, x_fijn) * f2 * RHO_WATER * G
    q_tank3 = _interp0(tank3_x, tank3_csa, x_fijn) * f3 * RHO_WATER * G

    # --- Downward load assembly -------------------------------------------
    q_neer = q_huid + q_tank1 + q_tank2 + q_tank3

    # Stern (transom) weight: distribute over small region near AP
    w_stern = stern_area * WALL_T * RHO_STEEL * FAC * G   # N
    i_stern = max(1, int(np.searchsorted(x_fijn, 0.5)))
    dx_stern = 0.0
    if i_stern > 1:
        dx_stern = x_fijn[i_stern - 1] - x_fijn[1]
        if dx_stern > 1e-6:
            q_neer[1:i_stern] += w_stern / dx_stern

    # Bulkhead loads
    for bhd in bhd_data:
        x0 = float(bhd['x_min'])
        x1 = float(bhd['x_max'])
        w  = float(bhd['area']) * WALL_T * RHO_STEEL * FAC * G
        i0 = int(np.searchsorted(x_fijn, x0))
        i1 = int(np.searchsorted(x_fijn, x1))
        dx = x1 - x0
        if i1 > i0 and dx > 1e-6:
            q_neer[i0:i1] += w / dx

    # Transition Piece (TP) semi-circle loads
    tp_info = []
    for tp in TPs:
        x_tp = float(tp['x'])
        w_tp = float(tp['weight_t']) * 1000.0 * G   # kg → N
        _add_semicircle_load(q_neer, x_fijn, x_tp, radius=4.0, weight_n=w_tp)
        tp_info.append(dict(x=x_tp, weight_n=w_tp))

    # Stowaway crane: house at pivot, boom at stowed position (horizontal, forward).
    if crane_m_house_kg > 0.0:
        _add_semicircle_load(q_neer, x_fijn, crane_pivot_x,   radius=1.5, weight_n=crane_m_house_kg * G)
    if crane_m_boom_kg > 0.0:
        _add_semicircle_load(q_neer, x_fijn, crane_boom_cg_x, radius=1.5, weight_n=crane_m_boom_kg * G)
    # --- Upward (buoyancy) load -------------------------------------------
    q_op = -_interp0(buoyant_x, buoyant_csa, x_fijn) * RHO_WATER * G

    # --- Match Dry Weight and LCG to run.py ---
    if 'm_dry' in hull_mass_data and 'lcg_dry' in hull_mass_data:
        target_m_dry = float(hull_mass_data['m_dry']) * G
        target_lcg_dry = float(hull_mass_data['lcg_dry'])
        target_moment = target_m_dry * target_lcg_dry
        
        q_dry = q_neer - q_tank1 - q_tank2 - q_tank3
        current_m_dry = float(np.trapezoid(q_dry, x_fijn))
        current_moment = float(np.trapezoid(q_dry * x_fijn, x_fijn))
        
        dm = target_m_dry - current_m_dry
        dMom = target_moment - current_moment
        
        A = dm / LPP
        B = (dMom - A * LPP**2 / 2.0) / (LPP**3 / 12.0)
        
        dq = A + B * (x_fijn - LPP/2.0)
        q_neer += dq
        q_dry_final = q_dry + dq

    # --- Net imbalance (before equilibrium adjustment) -------------------
    q_total_init = q_neer + q_op
    q_evenwicht  = float(np.trapezoid(q_total_init, x_fijn))
    
    _I = lambda q: float(np.trapezoid(q, x_fijn)) / 9.81 / 1000
    _M = lambda q: float(np.trapezoid(q * x_fijn, x_fijn)) / 9.81 / 1000
    print(f"DEBUG STRENGTH COMPONENTS:")
    print(f"  q_huid={_I(q_huid):.1f}t, q_tank1={_I(q_tank1):.1f}t, q_tank2={_I(q_tank2):.1f}t, q_tank3={_I(q_tank3):.1f}t")
    print(f"  q_neer_total={_I(q_neer):.1f}t, q_op_total={_I(-q_op):.1f}t")
    print(f"  Moments: M_DRY_FINAL={_M(q_dry_final):.1f} tm, M_T1={_M(q_tank1):.1f} tm, M_T2={_M(q_tank2):.1f} tm, M_T3={_M(q_tank3):.1f} tm")
    print(f"  Moments: M_NEER={_M(q_neer):.1f} tm, M_OP={_M(-q_op):.1f} tm, DIFF={_M(q_neer) - _M(-q_op):.1f} tm")

    # --- Tank equilibrium -------------------------------------------------
    evw = _bepaal_tank_evenwicht(q_evenwicht, diag1, diag2, diag3,
                                  v1_init, v2_init, v3_init)

    # Recompute tank loads with balanced factors
    v1_bal = evw['v1']
    v2_bal = evw['v2']
    v3_bal = evw['v3']

    f1b = v1_bal / v1_100 if v1_100 > 1e-9 else evw['pct1'] / 100.0
    f2b = v2_bal / v2_100 if v2_100 > 1e-9 else evw['pct2'] / 100.0
    f3b = v3_bal / v3_100 if v3_100 > 1e-9 else evw['pct3'] / 100.0

    q_tank1b = _interp0(tank1_x, tank1_csa, x_fijn) * f1b * RHO_WATER * G
    q_tank2b = _interp0(tank2_x, tank2_csa, x_fijn) * f2b * RHO_WATER * G
    q_tank3b = _interp0(tank3_x, tank3_csa, x_fijn) * f3b * RHO_WATER * G

    # Rebuild balanced downward load
    q_neer_b = q_huid + q_tank1b + q_tank2b + q_tank3b

    # Add structural loads again
    if i_stern > 1 and dx_stern > 1e-6:
        q_neer_b[1:i_stern] += w_stern / dx_stern

    for bhd in bhd_data:
        x0 = float(bhd['x_min'])
        x1 = float(bhd['x_max'])
        w  = float(bhd['area']) * WALL_T * RHO_STEEL * FAC * G
        i0 = int(np.searchsorted(x_fijn, x0))
        i1 = int(np.searchsorted(x_fijn, x1))
        dx = x1 - x0
        if i1 > i0 and dx > 1e-6:
            q_neer_b[i0:i1] += w / dx

    for tp in TPs:
        x_tp = float(tp['x'])
        w_tp = float(tp['weight_t']) * 1000.0 * G
        _add_semicircle_load(q_neer_b, x_fijn, x_tp, radius=4.0, weight_n=w_tp)

    if crane_m_house_kg > 0.0:
        _add_semicircle_load(q_neer_b, x_fijn, crane_pivot_x,   radius=1.5, weight_n=crane_m_house_kg * G)
    if crane_m_boom_kg > 0.0:
        _add_semicircle_load(q_neer_b, x_fijn, crane_boom_cg_x, radius=1.5, weight_n=crane_m_boom_kg * G)
        
    if 'dq' in locals():
        q_neer_b += dq
        
    # --- Balanced combined load -------------------------------------------
    q_gebal = q_neer_b + q_op

    # --- Shear force & bending moment -------------------------------------
    V = cumulative_trapezoid(q_gebal, x_fijn, initial=0.0)
    M = cumulative_trapezoid(V,       x_fijn, initial=0.0)

    # Linear correction: M(0)=0 already; enforce M(LPP)=0
    lin_corr = M[-1] * (x_fijn - x_fijn[0]) / (x_fijn[-1] - x_fijn[0])
    M        = M - lin_corr

    # --- Section modulus (second moment of area) --------------------------
    I_1mm = _interp0(shell_data['x'], shell_data['inertia_y_1mm'], x_fijn)
    I     = np.maximum(I_1mm * hull_t_mm, 0.01)   # [m⁴]
    EI    = E * I

    # Curvature κ = M / EI
    kappa = np.where(I > 0.01, M / EI, 0.0)

    # --- Deflection -------------------------------------------------------
    theta_raw  = cumulative_trapezoid(kappa, x_fijn, initial=0.0)
    # Correct so that average slope is zero (pinned-pinned boundary)
    corr_theta = float(np.trapezoid(theta_raw, x_fijn)) / (x_fijn[-1] - x_fijn[0])
    theta      = theta_raw - corr_theta
    w_defl     = cumulative_trapezoid(theta, x_fijn, initial=0.0)

    # --- Stresses ---------------------------------------------------------
    z_NA   = _interp0(shell_data['x'], shell_data['centroid_z'],  x_fijn)
    z_keel = _interp0(shell_data['x'], shell_data['z_keel'],      x_fijn)
    z_deck = _interp0(shell_data['x'], shell_data['z_deck'],      x_fijn)

    valid = I > 0.01
    sigma_bodem = np.where(valid, M * (z_keel - z_NA) / I, 0.0) / 1e6   # MPa
    sigma_dek   = np.where(valid, M * (z_deck - z_NA) / I, 0.0) / 1e6   # MPa

    # Zero out edges
    valid_x = (x_fijn >= shell_data['x'][0]) & (x_fijn <= shell_data['x'][-2])
    sigma_bodem[~valid_x] = 0.0
    sigma_dek[~valid_x]   = 0.0
    kappa[~valid_x]        = 0.0

    # --- Summary scalars --------------------------------------------------
    L        = x_fijn[-1] - x_fijn[0]
    srch     = (x_fijn >= x_fijn[0] + 0.1 * L) & (x_fijn <= x_fijn[0] + 0.9 * L)

    max_M_idx  = int(np.argmax(np.abs(M)))
    max_w_idx  = int(np.argmax(np.abs(w_defl)))

    krachtrestant_kn  = float(np.trapezoid(q_gebal,          x_fijn)) / 1e3
    momentrestant_mnm = float(np.trapezoid(q_gebal * x_fijn, x_fijn)) / 1e6

    # Tank2 LCG from fill diagram
    cs2_lcg = CubicSpline(diag2['fill_pct'], diag2['lcg'])
    tank2_lcg = float(cs2_lcg(evw['pct2']))

    # LCG, TCG, VCG of total system
    total_mass = stability_result.get('total_mass_kg', 0.0)
    lcg_total  = stability_result.get('lcg_total', 0.5 * LPP)
    tcg_total  = stability_result.get('tcg_total', 0.0)
    vcg_total  = stability_result.get('vcg_total', 0.5 * D)

    return dict(
        # Arrays
        x_fijn               = x_fijn,
        verdeelde_belasting  = q_gebal,                  # [N/m]
        dwarskrachtlijn      = V,                        # [N]
        momentlijn           = M,                        # [Nm]
        traagheidsmoment     = I,                        # [m⁴]
        buigstijfheid        = EI,                       # [Nm²]
        gereduceerd_moment   = kappa,                    # [1/m]
        sigma_bodem_mpa      = sigma_bodem,              # [MPa]
        sigma_dek_mpa        = sigma_dek,                # [MPa]
        hoekverdraaiing      = theta,                    # [rad]
        doorbuiging          = w_defl,                   # [m]
        # Scalars
        q_evenwicht          = q_evenwicht,              # [N]
        tank1_pct            = evw['pct1'],
        tank2_pct            = evw['pct2'],
        tank3_pct            = evw['pct3'],
        tank2_lcg            = tank2_lcg,                # [m]
        gm                   = stability_result.get('gm', 0.0),
        max_sigma_bodem      = float(np.max(np.abs(sigma_bodem[srch]))) if np.any(srch) else 0.0,
        max_sigma_dek        = float(np.max(np.abs(sigma_dek[srch])))   if np.any(srch) else 0.0,
        max_doorbuiging_mm   = float(np.max(np.abs(w_defl[srch])) * 1000) if np.any(srch) else 0.0,
        max_moment_nm        = float(M[max_M_idx]),
        locatie_max_moment   = float(x_fijn[max_M_idx]),
        locatie_max_doorbuiging = float(x_fijn[max_w_idx]),
        krachtrestant_kn     = krachtrestant_kn,
        momentrestant_mnm    = momentrestant_mnm,
        displacement_kg      = stability_result.get('total_mass_kg', 0.0),
        lcg_total            = lcg_total,
        tcg_total            = tcg_total,
        vcg_total            = vcg_total,
        lcb                  = stability_result.get('lcb', 0.0),
        tcb                  = stability_result.get('tcb', 0.0),
        vcb                  = stability_result.get('vcb', 0.0),
        tp_info              = tp_info,
    )
