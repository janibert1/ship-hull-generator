# -*- coding: utf-8 -*-
"""
run.py — Main entry point for the parametric engineering calculations.

Equilibrium calculation:
  User sets: Target_Draft_m, Tank1_Fill_pct, tank widths, Tank2_Length_pct_Loa
  Code computes: Tank3_Fill_pct (heel=0), Tank2 fill (displacement balance),
                 Tank2_Center_from_AP_m (trim=0)
  Errors if no valid solution exists (tank outside ship, fill > 100%, etc.)
"""

from __future__ import annotations

import sys
import json
import shutil
import importlib
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

_HERE   = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import config as cfg
from plot_full_surface import (
    build_hull_loft,
    _compute_surface_area,
    _polygon_area_2d,
    _longitudinal_wall_area,
    _transverse_wall_area,
    _interp_section,
)

from engineering.geometry   import hull_hw_at_z, section_properties
from engineering.hydrostatics import (
    compute_buoyant_csa, find_draft, center_of_buoyancy, waterplane_data,
    _displacement_at_draft,
)
from engineering.sections   import compute_shell_csa
from engineering.tanks      import (
    compute_side_tank_csa, compute_center_tank_csa,
    build_tank_fill_diagram,
    free_surface_ix_side_tank, free_surface_ix_center_tank,
)
from engineering.strength   import run_strength_calculation
from engineering.output     import save_plots, save_antwoordenblad, save_info_txt
from engineering.data_files import save_data_folder
from engineering.resistance import (
    compute_hull_coefficients, compute_at,
    compute_ie_regression, compute_ie_from_hull,
    compute_s_wet_hm, compute_s_wet_from_hull,
    build_resistance_table, check_applicability,
)


RHO_STEEL   = 7850.0
RHO_WATER   = 1025.0
G           = 9.81
FAC         = 2.1
WALL_T      = 0.010


_GROEP10_FILE_MAP = {
    'MainShipParticulars.csv': 'MainShipParticulars_Gr98_V3.0.csv',
    'MainShipParticulars.json': 'MainShipParticulars_Gr98_V3.0.json',
    'HullAreaData.csv': 'HullAreaData_Gr98_V3.0.csv',
    'TankData.json': 'TankData_Gr98_V3.0.json',
    'Tank1_Diagram_Volume.csv': 'Tank1_Diagram_Volume_Gr98_V3.0.csv',
    'Tank2_Diagram_Volume.csv': 'Tank2_Diagram_Volume_Gr98_V3.0.csv',
    'Tank3_Diagram_Volume.csv': 'Tank3_Diagram_Volume_Gr98_V3.0.csv',
    'Tank1_Diagram_Waterplane.csv': 'Tank1_Diagram_Waterplane_Gr98_V3.0.csv',
    'Tank2_Diagram_Waterplane.csv': 'Tank2_Diagram_Waterplane_Gr98_V3.0.csv',
    'Tank3_Diagram_Waterplane.csv': 'Tank3_Diagram_Waterplane_Gr98_V3.0.csv',
    'Shell_CSA.csv': 'Shell_CSA_Gr98_V3.0.csv',
    'Buoyant_CSA.csv': 'Buoyant_CSA_Gr98_V3.0.csv',
    'Tank1_CSA.csv': 'Tank1_CSA_Gr98_V3.0.csv',
    'Tank2_CSA.csv': 'Tank2_CSA_Gr98_V3.0.csv',
    'Tank3_CSA.csv': 'Tank3_CSA_Gr98_V3.0.csv',
    'Total_CSA.csv': 'Total_CSA_Gr98_V3.0.csv',
    'TankBHD_Data.csv': 'TankBHD_Data_Gr98_V3.0.csv',
    'ResistanceData.csv': 'ResistanceData_Gr98_V3.0.csv',
}


# ---------------------------------------------------------------------------
# Helper: invert volume diagram to fill percentage
# ---------------------------------------------------------------------------

def _invert_volume_to_pct(diag: dict, v_target: float) -> float:
    """Find fill percentage such that tank volume = v_target [m³]."""
    v_arr = np.asarray(diag['volume'], dtype=float)
    p_arr = np.asarray(diag['fill_pct'], dtype=float)
    order = np.argsort(v_arr)
    v_sorted = v_arr[order]
    p_sorted = p_arr[order]
    # Collapse duplicate volume points for safe inversion.
    v_u, idx = np.unique(v_sorted, return_index=True)
    p_u = p_sorted[idx]
    if len(v_u) < 2:
        return 0.0
    v_clamp = float(np.clip(v_target, v_u[0], v_u[-1]))
    return float(np.interp(v_clamp, v_u, p_u))


def _interp_diag(diag: dict, key: str, fill_pct: float, nonnegative: bool = False) -> float:
    """Interpolate one diagram field at fill_pct with monotonic linear interpolation."""
    p_arr = np.asarray(diag['fill_pct'], dtype=float)
    y_arr = np.asarray(diag[key], dtype=float)
    p = float(np.clip(fill_pct, p_arr[0], p_arr[-1]))
    val = float(np.interp(p, p_arr, y_arr))
    if nonnegative:
        val = max(0.0, val)
    return val


# ---------------------------------------------------------------------------
# Assemble hull area entries (for weight calculation)
# ---------------------------------------------------------------------------

def _build_hull_area_entries(geo: dict, cfg_dict: dict,
                              t2_x0: float | None = None,
                              t2_x1: float | None = None) -> list:
    """Compute areas and centroids of all structural members.

    t2_x0/t2_x1 override the config-derived tank2 boundaries if provided.
    """
    X_surf = geo['X_surf']
    Y_surf = geo['Y_surf']
    Z_surf = geo['Z_surf']
    LPP    = geo['LPP']
    B_half = geo['B_half']
    D      = geo['D']

    hull_t_mm = float(cfg_dict.get('Hull_Thickness_mm', 8.0))
    hull_t    = hull_t_mm / 1000.0

    t1_w   = float(cfg_dict.get('Tank1_Width_m', 3.0))
    t3_w   = float(cfg_dict.get('Tank3_Width_m', 3.0))
    LOA    = float(cfg_dict.get('Length_Loa_m', LPP))

    if t2_x0 is None or t2_x1 is None:
        t2_len_pct = float(cfg_dict.get('Tank2_Length_pct_Loa', 30.0))
        t2_len     = LOA * t2_len_pct / 100.0
        t2_cx      = float(cfg_dict.get('Tank2_Center_from_AP_m', 20.0))
        t2_x0     = t2_cx - t2_len / 2.0
        t2_x1     = t2_cx + t2_len / 2.0

    y_inner1 = B_half - t1_w
    y_inner3 = B_half - t3_w

    entries = []

    hull_half_area = _compute_surface_area(X_surf, Y_surf, Z_surf)
    hull_area      = 2.0 * hull_half_area
    x_deck         = X_surf[:, 0]
    y_deck_arr     = Y_surf[:, -1]
    deck_area      = 2.0 * float(np.trapezoid(y_deck_arr, x_deck))

    entries.append(dict(
        desc='Hull plating', area=hull_area,
        lcg=0.5 * LPP, tcg=0.0, vcg=0.4 * D,
        thickness=hull_t,
    ))
    entries.append(dict(
        desc='Deck plating', area=deck_area,
        lcg=0.5 * LPP, tcg=0.0, vcg=D,
        thickness=hull_t,
    ))

    A_t1 = _longitudinal_wall_area(X_surf, Y_surf, Z_surf, y_inner1, 0.0, LPP)
    entries.append(dict(
        desc='Tank1 inner wall', area=A_t1,
        lcg=0.5 * LPP, tcg=-(y_inner1), vcg=0.5 * D,
        thickness=WALL_T,
    ))

    A_t3 = _longitudinal_wall_area(X_surf, Y_surf, Z_surf, y_inner3, 0.0, LPP)
    entries.append(dict(
        desc='Tank3 inner wall', area=A_t3,
        lcg=0.5 * LPP, tcg=y_inner3, vcg=0.5 * D,
        thickness=WALL_T,
    ))

    y_cs_fwd, z_cs_fwd = _interp_section(X_surf, Y_surf, Z_surf, t2_x0)
    y_cs_aft, z_cs_aft = _interp_section(X_surf, Y_surf, Z_surf, t2_x1)
    A_t2_fwd = _transverse_wall_area(y_cs_fwd, z_cs_fwd, y_inner1, y_inner3)
    A_t2_aft = _transverse_wall_area(y_cs_aft, z_cs_aft, y_inner1, y_inner3)
    entries.append(dict(
        desc='Tank2 fwd bulkhead', area=A_t2_fwd,
        lcg=t2_x0, tcg=0.0, vcg=0.5 * D,
        thickness=WALL_T,
    ))
    entries.append(dict(
        desc='Tank2 aft bulkhead', area=A_t2_aft,
        lcg=t2_x1, tcg=0.0, vcg=0.5 * D,
        thickness=WALL_T,
    ))

    stern_poly_y = np.concatenate([[0.0], Y_surf[0], -Y_surf[0][::-1]])
    stern_poly_z = np.concatenate([[0.0], Z_surf[0],  Z_surf[0][::-1]])
    A_stern = _polygon_area_2d(stern_poly_y, stern_poly_z)
    entries.append(dict(
        desc='Stern (transom)', area=A_stern,
        lcg=0.0, tcg=0.0, vcg=0.5 * D,
        thickness=WALL_T,
    ))

    return entries


# ---------------------------------------------------------------------------
# Structural mass from hull area entries + TPs
# ---------------------------------------------------------------------------

def _structural_mass(hull_area_entries: list, cfg_dict: dict, D: float):
    """Return (m_dry, lcg_dry, vcg_dry, tcg_dry)."""
    m_dry = 0.0
    lcg_mom = 0.0
    vcg_mom = 0.0
    tcg_mom = 0.0

    for e in hull_area_entries:
        m = e['area'] * e['thickness'] * RHO_STEEL * FAC
        m_dry   += m
        lcg_mom += m * e['lcg']
        vcg_mom += m * e['vcg']
        tcg_mom += m * e['tcg']

    # All TPs on deck (crane is stowed and unloaded during transit).
    for tp in cfg_dict.get('Transition_Pieces', []):
        m = float(tp['weight_t']) * 1000.0
        m_dry   += m
        lcg_mom += m * float(tp['x'])
        vcg_mom += m * D
        tcg_mom += m * float(tp.get('y', 0.0))

    crane = cfg_dict.get('Crane', {})
    if isinstance(crane, dict):
        pivot_x  = float(crane.get('pivot_x_m', 0.5 * float(cfg_dict.get('Length_Loa_m', 0.0))))
        pivot_y  = float(crane.get('pivot_y_m', 0.0))
        pivot_h  = float(crane.get('pivot_height_m', 1.0))
        boom_len = float(crane.get('boom_length_m', 0.0))

        if boom_len > 0.0:
            # Stowaway crane: structure mass only (house + boom), no hook.
            # Consistent with groep10 stowaway model. SWL derived from TP masses.
            tp_list = cfg_dict.get('Transition_Pieces', [])
            one_tp_kg = float(tp_list[0]['weight_t']) * 1000.0 if tp_list else 0.0
            kraan_swl = one_tp_kg / 0.94 if one_tp_kg > 0.0 else float(crane.get('swl_max_t', 0.0)) * 1000.0
            m_house = 0.34 * kraan_swl
            m_boom  = 0.17 * kraan_swl
            m_rig   = 0.06 * kraan_swl
            
            jib_deg = float(crane.get('jib_angle_deg', 60.0))
            slew_deg = float(crane.get('slewing_angle_deg', 90.0))
            # Transit position: crane at its configured angles, but empty
            boom_cg_h = 0.5 * boom_len * np.cos(np.radians(jib_deg))
            x_boom_cg = pivot_x + boom_cg_h * np.cos(np.radians(slew_deg))
            y_boom_cg = pivot_y + boom_cg_h * np.sin(np.radians(slew_deg))
            z_boom_cg = pivot_h + 0.5 * boom_len * np.sin(np.radians(jib_deg))
            
            for m, x, y, z in (
                (m_house, pivot_x,    pivot_y,   pivot_h),
                (m_boom,  x_boom_cg,  y_boom_cg, z_boom_cg),
                (m_rig,   pivot_x,    pivot_y,   pivot_h), # Rigging weight
            ):
                if m <= 0.0:
                    continue
                m_dry   += m
                lcg_mom += m * x
                vcg_mom += m * z
                tcg_mom += m * y

    lcg_dry = lcg_mom / m_dry if m_dry > 0 else 0.0
    vcg_dry = vcg_mom / m_dry if m_dry > 0 else 0.0
    tcg_dry = tcg_mom / m_dry if m_dry > 0 else 0.0
    return m_dry, lcg_dry, vcg_dry, tcg_dry


# ---------------------------------------------------------------------------
# Equilibrium calculation
# ---------------------------------------------------------------------------

def compute_equilibrium(geo: dict, cfg_dict: dict,
                         diag1: dict, diag3: dict,
                         hull_area_entries: list,
                         t2_len: float,
                         y_inner1: float, y_inner3: float) -> dict:
    """Compute flat-keel equilibrium at the user-specified target draft.

    Given (from config):
      Target_Draft_m     — user-set target draught
      Tank1_Fill_pct     — user-set tank 1 fill

    Computes:
      Tank3_Fill_pct     — from no-heel condition (TCG = 0)
      Tank2 volume/fill  — from displacement balance
      Tank2_Center       — from no-trim condition (LCG = LCB)

    Raises ValueError if no valid solution exists (fills > 100 %,
    tank position outside ship bounds, etc.).
    """
    X_surf = geo['X_surf']
    Y_surf = geo['Y_surf']
    Z_surf = geo['Z_surf']
    LPP    = geo['LPP']
    D      = geo['D']

    T = float(cfg_dict.get('Target_Draft_m', D * 0.4))
    if T <= 0 or T >= D:
        raise ValueError(f"Target draft {T:.3f} m must be between 0 and DOA ({D:.3f} m).")

    # --- Hydrostatics at target draft ---
    V_disp = _displacement_at_draft(X_surf, Y_surf, Z_surf, T, 0.0, LPP, n_z=80)
    if V_disp <= 0:
        raise ValueError(f"Zero displacement at draft {T:.3f} m — hull geometry issue.")

    lcb, _tcb, vcb = center_of_buoyancy(X_surf, Y_surf, Z_surf, T,
                                         x_min=0.0, x_max=LPP, n_z=80)
    wp   = waterplane_data(X_surf, Y_surf, Z_surf, T, x_min=0.0, x_max=LPP)
    Ix_wp = wp['Ix']
    BM    = Ix_wp / V_disp

    m_target = V_disp * RHO_WATER

    # --- Structural mass ---
    m_dry, lcg_dry, vcg_dry, tcg_dry = _structural_mass(hull_area_entries, cfg_dict, D)

    if m_dry >= m_target:
        raise ValueError(
            f"Structural mass {m_dry/1000:.1f} t >= displacement {m_target/1000:.1f} t "
            f"at draft {T:.3f} m.  Increase draft or reduce plating thickness."
        )

    # --- Tank 1 (user-set) ---
    t1_fill_pct = float(cfg_dict.get('Tank1_Fill_pct', 50.0))
    if not (0.0 <= t1_fill_pct <= 100.0):
        raise ValueError(f"Tank1_Fill_pct {t1_fill_pct:.1f} out of range [0, 100].")

    v1   = _interp_diag(diag1, 'volume', t1_fill_pct, nonnegative=True)
    m1   = v1 * RHO_WATER
    lcg1 = _interp_diag(diag1, 'lcg', t1_fill_pct)
    vcg1 = _interp_diag(diag1, 'vcg', t1_fill_pct, nonnegative=True)
    # TCG for SB tank (constant approximation from diagram)
    tcg1 = float(diag1['tcg'][-1])  # use the representative value (constant)

    # --- Tank 3: solve for no-heel (TCG_total = 0) ---
    # m_dry*tcg_dry + m1*tcg1 + m3*tcg3 = 0  (tank2 tcg = 0)
    # v3 * rho * tcg3 = -(m_dry*tcg_dry + m1*tcg1)
    tcg3_approx = float(diag3['tcg'][-1])  # positive (port side)
    if abs(tcg3_approx) < 1e-6:
        raise ValueError("Tank 3 TCG is zero — cannot compute heel equilibrium.")

    required_moment3 = -(m_dry * tcg_dry + m1 * tcg1)
    v3_target = required_moment3 / (RHO_WATER * tcg3_approx)

    v3_min = 0.0
    v3_max = float(diag3['volume'][-1])

    if v3_target < 0.0:
        raise ValueError(
            f"Heel equilibrium requires negative tank 3 volume ({v3_target:.2f} m3). "
            f"Shift starboard weight or reduce tank 1 fill."
        )
    if v3_target > v3_max:
        raise ValueError(
            f"Heel equilibrium requires tank 3 volume {v3_target:.2f} m3 "
            f"> maximum {v3_max:.2f} m3 (100 % fill). "
            f"Increase tank 3 width or reduce tank 1 fill."
        )

    t3_fill_pct = _invert_volume_to_pct(diag3, v3_target)
    # Clamp to valid range (CubicSpline can slightly overshoot)
    t3_fill_pct = float(np.clip(t3_fill_pct, 0.0, 100.0))

    v3   = _interp_diag(diag3, 'volume', t3_fill_pct, nonnegative=True)
    m3   = v3 * RHO_WATER
    lcg3 = _interp_diag(diag3, 'lcg', t3_fill_pct)
    vcg3 = _interp_diag(diag3, 'vcg', t3_fill_pct, nonnegative=True)
    tcg3 = tcg3_approx

    # --- Tank 2: displacement balance ---
    m2 = m_target - m_dry - m1 - m3
    if m2 <= 0:
        raise ValueError(
            f"Displacement balance yields tank 2 mass {m2/1000:.1f} t <= 0. "
            f"Structural + side-tank mass exceeds target displacement. "
            f"Increase draft or reduce side-tank fills."
        )

    v2_target = m2 / RHO_WATER

    # --- Tank 2 x-position: no-trim (LCG = LCB) ---
    # m_total * LCB = m_dry*lcg_dry + m1*lcg1 + m2*lcg2 + m3*lcg3
    lcg2 = (m_target * lcb - m_dry * lcg_dry - m1 * lcg1 - m3 * lcg3) / m2
    t2_center = lcg2
    t2_x0 = t2_center - t2_len / 2.0
    t2_x1 = t2_center + t2_len / 2.0

    if t2_x0 < 0.0:
        raise ValueError(
            f"Tank 2 aft end x={t2_x0:.2f} m is outside ship (before AP=0). "
            f"Required center: {t2_center:.2f} m, half-length: {t2_len/2:.2f} m. "
            f"Adjust tank 2 length or ship balance."
        )
    if t2_x1 > LPP:
        raise ValueError(
            f"Tank 2 forward end x={t2_x1:.2f} m exceeds FP (x={LPP:.2f} m). "
            f"Required center: {t2_center:.2f} m, half-length: {t2_len/2:.2f} m. "
            f"Adjust tank 2 length or ship balance."
        )

    return dict(
        T=T,
        V_disp=V_disp,
        m_target=m_target,
        m_dry=m_dry, lcg_dry=lcg_dry, vcg_dry=vcg_dry, tcg_dry=tcg_dry,
        t1_fill_pct=t1_fill_pct,
        v1=v1, m1=m1, lcg1=lcg1, vcg1=vcg1, tcg1=tcg1,
        t3_fill_pct=t3_fill_pct,
        v3=v3, m3=m3, lcg3=lcg3, vcg3=vcg3, tcg3=tcg3,
        m2=m2, v2=v2_target, lcg2=lcg2,
        t2_center=t2_center, t2_x0=t2_x0, t2_x1=t2_x1,
        lcb=lcb, vcb=vcb,
        Ix_wp=Ix_wp, BM=BM,
    )


# ---------------------------------------------------------------------------
# Bulkhead data (for strength distributed loads)
# ---------------------------------------------------------------------------

def _build_bhd_data(geo: dict, hull_area_entries: list) -> list:
    LPP = geo['LPP']
    bhd_data = []
    dx_small = 0.3

    for e in hull_area_entries:
        if 'bulkhead' in e['desc'].lower():
            lcg = e['lcg']
            bhd_data.append(dict(
                area  = e['area'],
                x_min = max(0.0, lcg - dx_small),
                x_max = min(LPP,  lcg + dx_small),
                lcg   = lcg,
                tcg   = e['tcg'],
                vcg   = e['vcg'],
            ))
        elif 'inner wall' in e['desc'].lower():
            bhd_data.append(dict(
                area  = e['area'],
                x_min = 0.0,
                x_max = LPP,
                lcg   = e['lcg'],
                tcg   = e['tcg'],
                vcg   = e['vcg'],
            ))

    return bhd_data


def _sync_data_to_groep10(data_dir: Path) -> None:
    """Copy engineering Data files into groep10 naming convention."""
    gdir = _PARENT / 'groep10'
    dst_dir = gdir / 'data'
    dst_dir.mkdir(parents=True, exist_ok=True)

    data_link = gdir / 'Data'
    if not data_link.exists():
        try:
            data_link.symlink_to(dst_dir)
        except Exception:
            pass

    for src_name, dst_name in _GROEP10_FILE_MAP.items():
        src = data_dir / src_name
        if not src.exists():
            continue
        shutil.copy2(src, dst_dir / dst_name)


def _load_groep10_strength() -> dict:
    """Run groep10 strength pipeline and map outputs to engineering result keys."""
    gdir = _PARENT / 'groep10'
    if not gdir.exists():
        raise RuntimeError("groep10 folder not found.")

    gdir_str = str(gdir)
    if gdir_str not in sys.path:
        sys.path.insert(0, gdir_str)

    for mod_name in (
        'main',
        'langsscheepse_sterkte',
        'lees_bestanden',
        'schip_klassen',
        'schip_functies',
    ):
        mod = sys.modules.get(mod_name)
        mod_file = str(getattr(mod, '__file__', '')) if mod is not None else ''
        if mod is not None and mod_file and not mod_file.startswith(gdir_str):
            del sys.modules[mod_name]

    g_main = importlib.import_module('main')
    g_strength = importlib.import_module('langsscheepse_sterkte')

    runtime_cfg = g_main.laadRuntimeConfiguratie()
    g_main.synchroniseerAntwoordenblad(runtime_cfg)
    schip = g_main.maakTransportSchip(runtime_cfg)
    strength = g_strength.berekenLangsscheepseSterkte(g_main.BESTANDSCODE, schip)

    g_out = gdir / 'output'
    g_out.mkdir(parents=True, exist_ok=True)
    g_strength.toonSterkteGrafieken(strength, save_map=g_out)
    with redirect_stdout(io.StringIO()):
        g_main.schrijfOutputAntwoordenblad(schip, strength)

    return dict(
        x_fijn=strength.x_fijn,
        verdeelde_belasting=strength.verdeelde_belasting_kn_m * 1e3,
        dwarskrachtlijn=strength.dwarskrachtlijn_mn * 1e6,
        momentlijn=strength.momentlijn_mnm * 1e6,
        traagheidsmoment=strength.traagheidsmoment_lijn,
        buigstijfheid=strength.buigstijfheid_lijn,
        gereduceerd_moment=strength.gereduceerd_moment_lijn,
        sigma_bodem_mpa=strength.sigma_bodem_mpa,
        sigma_dek_mpa=strength.sigma_dek_mpa,
        hoekverdraaiing=strength.hoekverdraaiing_lijn,
        doorbuiging=strength.doorbuiging_lijn,
        q_evenwicht=strength.q_evenwicht,
        restant_volume=strength.restant_volume,
        krachtrestant_kn=strength.krachtrestant_kn,
        momentrestant_mnm=strength.momentrestant_mnm,
        max_sigma_bodem=strength.max_sigma_bodem_mpa,
        max_sigma_dek=strength.max_sigma_dek_mpa,
        max_doorbuiging_mm=strength.max_doorbuiging_mm,
        max_moment_nm=strength.max_moment_nm,
        locatie_max_moment=strength.locatie_max_moment_m,
        locatie_max_doorbuiging=strength.locatie_max_doorbuiging_m,
    )


def _copy_groep10_plots_to_engineering(out_dir: Path) -> None:
    """Copy the generated groep10 strength plots into engineering/output."""
    g_out = _PARENT / 'groep10' / 'output'
    for name in (
        'verdeelde_belasting.png',
        'dwarskrachtenlijn.png',
        'momentenlijn.png',
        'traagheidsmoment.png',
        'buigstijfheid.png',
        'gereduceerd_moment.png',
        'buigspanning.png',
        'hoekverdraaiing.png',
        'doorbuiging.png',
    ):
        src = g_out / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    antwoorden_src = g_out / 'antwoordenblad.json'
    if antwoorden_src.exists():
        shutil.copy2(antwoorden_src, out_dir / 'antwoordenblad.json')


def _evaluate_crane_operation(
    cfg_dict: dict,
    local_half_beam: float,
    m_total: float,
    gm: float,
    v3_max_m3: float,
    y_inner3: float
) -> dict:
    crane = cfg_dict.get('Crane', {}) if isinstance(cfg_dict.get('Crane', {}), dict) else {}
    swl_max_t = float(crane.get('swl_max_t', 0.0))
    pivot_h = float(crane.get('pivot_height_m', 1.0))
    pivot_x = float(crane.get('pivot_x_m', 0.0))
    pivot_y = float(crane.get('pivot_y_m', max(0.0, local_half_beam - 0.75)))
    boom_len = float(crane.get('boom_length_m', 0.0))
    jib_angle = float(crane.get('jib_angle_deg', 60.0))
    slew_deg = float(crane.get('slewing_angle_deg', 90.0))
    outreach_req = max(0.0, (local_half_beam + 1.0) - abs(pivot_y))  # 1m clearance from ship side
    # Hook height: monopile (20m) + TP height (8m) + 1m vertical clearance above TP.
    hook_z_req = pivot_h + 20.0 + 8.0 + 1.0 - 1.0  # = pivot_h + 28.0

    outreach_actual = boom_len * np.cos(np.radians(jib_angle)) * abs(np.sin(np.radians(slew_deg)))
    if swl_max_t <= 0.0 or boom_len <= 0.0:
        return dict(ok=False, reason='crane geometry/SWL invalid', heel_deg=90.0, trim_deg=0.0)

    angle_deg = jib_angle
    hook_z = pivot_h + boom_len * np.sin(np.radians(angle_deg))
    
    # SWL derating
    try:
        from apex_architect.constants import CRANE_SWL_FULL_ANGLE_DEG, CRANE_SWL_ZERO_ANGLE_FACTOR, CRANE_RIGGING_MASS_FRAC
    except ImportError:
        CRANE_SWL_FULL_ANGLE_DEG = 60.0
        CRANE_SWL_ZERO_ANGLE_FACTOR = 0.50
        CRANE_RIGGING_MASS_FRAC = 0.06

    if jib_angle >= CRANE_SWL_FULL_ANGLE_DEG:
        swl_eff_t = swl_max_t
    else:
        swl_eff_t = swl_max_t * (CRANE_SWL_ZERO_ANGLE_FACTOR + (1.0 - CRANE_SWL_ZERO_ANGLE_FACTOR) * jib_angle / CRANE_SWL_FULL_ANGLE_DEG)

    rigging_t = CRANE_RIGGING_MASS_FRAC * swl_max_t
    required_lift_t = 550.0 + rigging_t

    m_heeling = required_lift_t * 1000.0 * G * outreach_req
    m_comp = max(0.0, v3_max_m3 * RHO_WATER) * G * max(y_inner3, 0.5)
    m_residual = max(0.0, m_heeling - m_comp)
    denom = m_total * G * max(gm, 1e-9)
    heel_deg = float(np.degrees(np.arcsin(np.clip(m_residual / max(denom, 1e-9), 0.0, 1.0))))
    righting_5deg = denom * np.sin(np.radians(5.0))

    # --- Trim and Heel check during crane lift from deck ---
    # Hook longitudinal position when slewed to operating angle.
    # slew=0 → forward, slew=90 → port (perpendicular). cos(slew) gives fore-aft component.
    x_hook = pivot_x + boom_len * np.cos(np.radians(jib_angle)) * np.cos(np.radians(slew_deg))
    # Worst case: most forward TP is lifted (largest arm from crane pivot).
    tp_list = cfg_dict.get('Transition_Pieces', [])
    if tp_list:
        x_tp_fwd = float(min(float(tp.get('x', pivot_x)) for tp in tp_list))
        # Find farthest TP for heel-during-pickup
        dists = [
            float(np.sqrt((float(tp.get('x', pivot_x)) - pivot_x) ** 2 + (float(tp.get('y', pivot_y)) - pivot_y) ** 2))
            for tp in tp_list
        ]
        i_far = int(np.argmax(dists))
        y_far = float(tp_list[i_far].get('y', 0.0))
        # Heel while hook is above the TP on deck
        m_heel_pickup = required_lift_t * 1000.0 * G * abs(y_far)
        m_residual_pickup = max(0.0, m_heel_pickup - m_comp)
        heel_pickup_deg = float(np.degrees(np.arcsin(np.clip(m_residual_pickup / max(denom, 1e-9), 0.0, 1.0))))
        
        # Determine pickup jib angle to see if swl_eff_pickup_t is sufficient
        max_dist = float(max(dists))
        if max_dist <= boom_len:
            cos_arg = float(np.clip(max_dist / max(boom_len, 1e-9), 0.0, 1.0))
            jib_pickup_deg = float(np.degrees(np.arccos(cos_arg)))
            jib_pickup_deg = float(np.clip(jib_pickup_deg, 0.0, 80.0))
            if jib_pickup_deg >= CRANE_SWL_FULL_ANGLE_DEG:
                swl_eff_pickup_t = swl_max_t
            else:
                swl_eff_pickup_t = swl_max_t * (CRANE_SWL_ZERO_ANGLE_FACTOR + (1.0 - CRANE_SWL_ZERO_ANGLE_FACTOR) * jib_pickup_deg / CRANE_SWL_FULL_ANGLE_DEG)
        else:
            swl_eff_pickup_t = 0.0 # Cannot reach
    else:
        x_tp_fwd = pivot_x
        heel_pickup_deg = 0.0
        swl_eff_pickup_t = swl_eff_t
        
    delta_lcg = (x_hook - x_tp_fwd) * (required_lift_t * 1000.0) / max(m_total, 1.0)
    # Longitudinal GM approximation: BML = L² / (12 × T) for a rectangular waterplane.
    loa = float(cfg_dict.get('Length_Loa_m', 100.0))
    lpp = loa * float(cfg_dict.get('Lpp_Loa_ratio', 0.97))
    draft = float(cfg_dict.get('Target_Draft_m', 4.0))
    bml = lpp ** 2 / max(12.0 * draft, 1e-3)
    trim_deg = float(np.degrees(np.arctan(abs(delta_lcg) / max(bml, 1e-9))))

    g_swl = required_lift_t - swl_eff_t
    g_swl_pickup = required_lift_t - swl_eff_pickup_t
    g_reach = outreach_req - outreach_actual
    g_hook = hook_z_req - hook_z
    g_heel = max(heel_deg, heel_pickup_deg) - 5.0
    g_trim = trim_deg - 5.0

    # Boom collision check with TPs in transit/deployment position
    g_collision = 0.0
    dx = boom_len * np.cos(np.radians(jib_angle)) * np.cos(np.radians(slew_deg))
    dy = boom_len * np.cos(np.radians(jib_angle)) * np.sin(np.radians(slew_deg))
    dz = boom_len * np.sin(np.radians(jib_angle))
    A = dx**2 + dy**2
    tp_radius_clearance = 4.0 + 1.0  # 4.0m TP radius + 1m clearance
    for tp in tp_list:
        tp_x = float(tp["x"])
        tp_y = float(tp.get("y", 0.0))
        if A > 1e-9:
            B = 2.0 * ((pivot_x - tp_x) * dx + (pivot_y - tp_y) * dy)
            C = (pivot_x - tp_x)**2 + (pivot_y - tp_y)**2 - tp_radius_clearance**2
            Delta = B**2 - 4.0 * A * C
            if Delta > 0:
                t1 = (-B - np.sqrt(Delta)) / (2.0 * A)
                t2 = (-B + np.sqrt(Delta)) / (2.0 * A)
                t_min = max(0.0, min(t1, t2))
                t_max = min(1.0, max(t1, t2))
                if t_min <= t_max:
                    z_deck = pivot_h - 1.0
                    z_tp_top = z_deck + 20.0
                    z_boom_1 = pivot_h + t_min * dz
                    z_boom_2 = pivot_h + t_max * dz
                    if max(min(z_boom_1, z_boom_2), z_deck) <= min(max(z_boom_1, z_boom_2), z_tp_top):
                        t_opt = np.clip(-B / (2.0 * A), 0.0, 1.0)
                        dist_sq = A * t_opt**2 + B * t_opt + C + tp_radius_clearance**2
                        overlap = tp_radius_clearance - np.sqrt(max(0.0, dist_sq))
                        g_collision = max(g_collision, overlap if overlap > 0 else 1.0)
        else:
            dist = np.sqrt((pivot_x - tp_x)**2 + (pivot_y - tp_y)**2)
            if dist < tp_radius_clearance:
                z_deck = pivot_h - 1.0
                z_tp_top = z_deck + 20.0
                if max(min(pivot_h, hook_z), z_deck) <= min(max(pivot_h, hook_z), z_tp_top):
                    g_collision = max(g_collision, tp_radius_clearance - dist)

    ok = bool(g_swl <= 0.0 and g_swl_pickup <= 0.0 and g_reach <= 0.0 and g_hook <= 0.0 and g_heel <= 0.0 and g_trim <= 0.0 and g_collision <= 0.0)

    return dict(
        ok=ok,
        reason='ok' if ok else 'crane requirements not met',
        required_outreach_m=outreach_req,
        actual_outreach_m=outreach_actual,
        required_hook_z_m=hook_z_req,
        required_angle_deg=angle_deg,
        hook_z_m=hook_z,
        swl_eff_t=min(swl_eff_t, swl_eff_pickup_t),
        swl_max_t=swl_max_t,
        required_lift_t=required_lift_t,
        heel_deg=max(heel_deg, heel_pickup_deg),
        trim_deg=trim_deg,
        heeling_moment_mnm=m_heeling / 1e6,
        compensation_moment_mnm=m_comp / 1e6,
        residual_moment_mnm=m_residual / 1e6,
        righting_5deg_mnm=righting_5deg / 1e6,
        g_swl=max(g_swl, g_swl_pickup),
        g_reach=g_reach,
        g_hook=g_hook,
        g_heel=g_heel,
        g_trim=g_trim,
        g_collision=g_collision,
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True,
            cfg_dict_override: dict | None = None,
            save_outputs: bool = True) -> dict:
    """Run all engineering calculations and optionally write output/data files.

    cfg_dict_override: use this config dict instead of reading config.json.
        The cfg module attributes (cfg.LPP, cfg.BOA, …) must already reflect
        this config (call reload_config_module first, as plot_full_surface.py does).
    save_outputs: when False skip all file I/O and groep10 pipeline (fast, for
        live optimizer plotting).
    """
    cfg_dict = cfg_dict_override if cfg_dict_override is not None else cfg.load_config()
    LPP    = cfg.LPP
    B_half = cfg.BOA / 2.0
    D      = cfg.DOA
    LOA    = cfg.LOA

    t1_w       = float(cfg_dict.get('Tank1_Width_m',         3.0))
    t3_w       = float(cfg_dict.get('Tank3_Width_m',         3.0))
    t2_len_pct = float(cfg_dict.get('Tank2_Length_pct_Loa', 30.0))
    t2_len     = LOA * t2_len_pct / 100.0

    y_inner1 = B_half - t1_w
    y_inner3 = B_half - t3_w

    if y_inner1 <= 0:
        raise ValueError(f"Tank1_Width_m ({t1_w}) >= B_half ({B_half:.2f}): tank wider than half-ship.")
    if y_inner3 <= 0:
        raise ValueError(f"Tank3_Width_m ({t3_w}) >= B_half ({B_half:.2f}): tank wider than half-ship.")

    if verbose:
        print("Building hull loft...")
    geo = build_hull_loft(N_u=50, N_t=50)
    X_surf = geo['X_surf']
    Y_surf = geo['Y_surf']
    Z_surf = geo['Z_surf']

    if verbose:
        print(f"  LPP={LPP:.2f} m, BOA={B_half*2:.2f} m, DOA={D:.2f} m")
        print(f"  Tank2 length: {t2_len:.1f} m ({t2_len_pct:.0f}% LOA)")

    # --- Shell CSA ---
    if verbose:
        print("Computing shell cross-sections...")
    shell = compute_shell_csa(X_surf, Y_surf, Z_surf, x_min=0.0, x_max=LPP)

    # --- Side tank CSA at 100% fill ---
    if verbose:
        print("Computing side-tank cross-sections...")
    t1_x, t1_csa = compute_side_tank_csa(X_surf, Y_surf, Z_surf, y_inner=y_inner1, h=D,
                                           x_min=0.0, x_max=LPP)
    t3_x, t3_csa = compute_side_tank_csa(X_surf, Y_surf, Z_surf, y_inner=y_inner3, h=D,
                                           x_min=0.0, x_max=LPP)

    # --- Side tank fill diagrams (full ship length) ---
    if verbose:
        print("Building side-tank fill diagrams...")
    diag1 = build_tank_fill_diagram(
        X_surf, Y_surf, Z_surf,
        tank_type='side_sb', h_max=D,
        x_min=0.0, x_max=LPP,
        y_inner_sb=y_inner1, y_inner_port=y_inner3,
        fill_steps=101)

    diag3 = build_tank_fill_diagram(
        X_surf, Y_surf, Z_surf,
        tank_type='side_port', h_max=D,
        x_min=0.0, x_max=LPP,
        y_inner_sb=y_inner1, y_inner_port=y_inner3,
        fill_steps=101)

    # --- Initial hull area entries (use config t2 center as first estimate) ---
    if verbose:
        print("Computing structural areas (first pass)...")
    hull_area_entries_init = _build_hull_area_entries(geo, cfg_dict)

    # --- Equilibrium calculation ---
    if verbose:
        print("Computing equilibrium (draft -> tank fills and positions)...")
    try:
        equil = compute_equilibrium(
            geo, cfg_dict, diag1, diag3,
            hull_area_entries_init,
            t2_len, y_inner1, y_inner3)
    except ValueError as e:
        print(f"\n*** EQUILIBRIUM ERROR: {e} ***\n")
        raise

    t2_center = equil['t2_center']
    t2_x0     = equil['t2_x0']
    t2_x1     = equil['t2_x1']
    draft      = equil['T']

    if verbose:
        print(f"  Draft:        {draft:.3f} m")
        print(f"  Tank2 center: {t2_center:.2f} m from AP,  x=[{t2_x0:.1f}, {t2_x1:.1f}] m")
        print(f"  Tank1 fill:   {equil['t1_fill_pct']:.1f} %")
        print(f"  Tank3 fill:   {equil['t3_fill_pct']:.1f} %")

    # --- Rebuild hull area entries with correct t2 position ---
    hull_area_entries = _build_hull_area_entries(
        geo, cfg_dict, t2_x0=t2_x0, t2_x1=t2_x1)

    # --- Center tank CSA at 100% fill ---
    if verbose:
        print("Computing center-tank cross-sections...")
    t2_x, t2_csa = compute_center_tank_csa(
        X_surf, Y_surf, Z_surf,
        y_sb=y_inner1, y_port=y_inner3, h=D,
        x_min=t2_x0, x_max=t2_x1)

    # --- Center tank fill diagram ---
    if verbose:
        print("Building center-tank fill diagram...")
    diag2 = build_tank_fill_diagram(
        X_surf, Y_surf, Z_surf,
        tank_type='center', h_max=D,
        x_min=t2_x0, x_max=t2_x1,
        y_inner_sb=y_inner1, y_inner_port=y_inner3,
        fill_steps=101)

    # --- Tank 2 fill percentage from required volume ---
    v2_max = float(diag2['volume'][-1])
    if equil['v2'] > v2_max:
        raise ValueError(
            f"Tank 2 requires volume {equil['v2']:.2f} m3 > max {v2_max:.2f} m3 (100% fill). "
            f"Increase tank 2 length or reduce loads."
        )
    t2_fill_pct = _invert_volume_to_pct(diag2, equil['v2'])
    t2_fill_pct = float(np.clip(t2_fill_pct, 0.0, 100.0))

    if verbose:
        print(f"  Tank2 fill:   {t2_fill_pct:.1f} %")

    # --- VCG of tank 2 ---
    vcg2 = _interp_diag(diag2, 'vcg', t2_fill_pct, nonnegative=True)

    # --- Full stability: KB, KG, BM, GM ---
    V_disp  = equil['V_disp']
    KB      = equil['vcb']
    BM      = equil['BM']
    m_total = equil['m_target']

    KG = (equil['m_dry'] * equil['vcg_dry'] +
          equil['m1']    * equil['vcg1']    +
          equil['m2']    * vcg2             +
          equil['m3']    * equil['vcg3'])   / m_total

    # Free-surface corrections (actual hull geometry, not rectangular)
    h_fill1 = D * equil['t1_fill_pct'] / 100.0
    h_fill3 = D * equil['t3_fill_pct'] / 100.0
    h_fill2 = D * t2_fill_pct          / 100.0

    Ix_fs1 = free_surface_ix_side_tank(X_surf, Y_surf, Z_surf,
                                        y_inner1, h_fill1, 0.0, LPP)
    Ix_fs3 = free_surface_ix_side_tank(X_surf, Y_surf, Z_surf,
                                        y_inner3, h_fill3, 0.0, LPP)
    Ix_fs2 = free_surface_ix_center_tank(X_surf, Y_surf, Z_surf,
                                          y_inner1, y_inner3, h_fill2,
                                          t2_x0, t2_x1)

    GG1 = Ix_fs1 / V_disp if V_disp > 0 else 0.0
    GG2 = Ix_fs2 / V_disp if V_disp > 0 else 0.0
    GG3 = Ix_fs3 / V_disp if V_disp > 0 else 0.0
    GM  = KB - KG + BM - (GG1 + GG2 + GG3)

    GM_MIN = 1.0  # minimum required metacentric height [m]
    if GM < GM_MIN:
        raise ValueError(
            f"GM = {GM:.3f} m < required minimum {GM_MIN:.1f} m. "
            f"KB={KB:.3f} m, KG={KG:.3f} m, BM={BM:.3f} m, "
            f"free-surface correction={GG1+GG2+GG3:.3f} m. "
            f"Reduce tank widths or increase BM (wider hull)."
        )

    if verbose:
        print(f"  Displacement: {m_total/1000:.1f} t")
        print(f"  KB={KB:.3f} m  KG={KG:.3f} m  BM={BM:.3f} m  GM={GM:.3f} m")

    v3_max_m3 = float(diag3['volume'][-1]) if len(diag3.get('volume', [])) else 0.0
    crane_cfg = cfg_dict.get('Crane', {}) if isinstance(cfg_dict.get('Crane', {}), dict) else {}
    crane_pivot_x = float(crane_cfg.get('pivot_x_m', 0.8 * LPP))
    x_mid = X_surf.mean(axis=1)
    i_pivot = int(np.argmin(np.abs(x_mid - crane_pivot_x)))
    local_half_beam = float(np.max(Y_surf[i_pivot]))
    if isinstance(crane_cfg, dict):
        crane_cfg = dict(crane_cfg)
        crane_cfg['pivot_y_m'] = max(0.0, local_half_beam - 0.75)
        cfg_dict = dict(cfg_dict)
        cfg_dict['Crane'] = crane_cfg
    crane_swl_configured = float(crane_cfg.get('swl_max_t', 0.0))
    crane_boom_configured = float(crane_cfg.get('boom_length_m', 0.0))
    if crane_swl_configured > 0.0 and crane_boom_configured > 0.0:
        crane_check = _evaluate_crane_operation(
            cfg_dict,
            local_half_beam=local_half_beam,
            m_total=m_total,
            gm=GM,
            v3_max_m3=v3_max_m3,
            y_inner3=y_inner3,
        )
        if not crane_check.get('ok', False):
            raise ValueError(
                "Crane operation requirement failed: "
                f"SWL deficit={crane_check.get('g_swl', 0.0):.2f} t, "
                f"hook-height deficit={crane_check.get('g_hook', 0.0):.2f} m, "
                f"heel exceedance={crane_check.get('g_heel', 0.0):.2f} deg, "
                f"trim exceedance={crane_check.get('g_trim', 0.0):.2f} deg."
            )
        if verbose:
            print(
                f"  Crane check OK: heel={crane_check['heel_deg']:.2f} deg, "
                f"trim={crane_check['trim_deg']:.2f} deg, "
                f"SWL eff={crane_check['swl_eff_t']:.1f} t @ angle {crane_check['required_angle_deg']:.1f} deg"
            )
    else:
        crane_check = {
            'ok': True, 'reason': 'no crane fitted',
            'heel_deg': 0.0, 'trim_deg': 0.0,
            'swl_eff_t': 0.0, 'swl_max_t': 0.0,
            'required_angle_deg': 0.0,
            'required_outreach_m': 0.0, 'actual_outreach_m': 0.0,
            'required_hook_z_m': 0.0, 'hook_z_m': 0.0,
            'required_lift_t': 0.0,
            'heeling_moment_mnm': 0.0, 'compensation_moment_mnm': 0.0,
            'residual_moment_mnm': 0.0, 'righting_5deg_mnm': 0.0,
            'g_swl': 0.0, 'g_reach': 0.0, 'g_hook': 0.0,
            'g_heel': 0.0, 'g_trim': 0.0, 'g_collision': 0.0,
        }
        if verbose:
            print("  No crane fitted — skipping crane check.")

    # --- Resistance (Holtrop-Mennen 1984) ---
    if verbose:
        print("Computing Holtrop-Mennen resistance...")

    design_speed   = float(cfg_dict.get('Design_Speed_kn',               14.0))
    max_delta      = float(cfg_dict.get('Max_Speed_delta_kn',              2.0))
    steps_per_kn   = int(  cfg_dict.get('Speed_Steps_per_kn',              4))
    Cstern         = float(np.clip(cfg_dict.get('Cstern', -25.0), -25.0, 10.0))
    ie_factor_pct  = float(cfg_dict.get('Entrance_Angle_Factor_pct_BWL',  30.0))
    method_swet    = int(  cfg_dict.get('Method_S_wet',                     0))
    method_ie      = int(  cfg_dict.get('Method_IE',                        0))

    hm_coeffs = compute_hull_coefficients(
        X_surf, Y_surf, Z_surf, draft, LPP, V_disp, equil['lcb'])

    BWL     = hm_coeffs['BWL']
    CP_hm   = hm_coeffs['CP']
    lcb_pct = hm_coeffs['lcb_pct']
    LR      = LPP * (1.0 - CP_hm + 0.06 * CP_hm * lcb_pct / (4.0 * CP_hm - 1.0))

    if method_ie == 2:
        iE = compute_ie_from_hull(X_surf, Y_surf, Z_surf, draft, ie_factor_pct)
    else:
        iE = compute_ie_regression(LPP, BWL, CP_hm, hm_coeffs['CWP'],
                                   lcb_pct, V_disp, LR)

    if method_swet == 1:
        S_wet = compute_s_wet_from_hull(X_surf, Y_surf, Z_surf, draft, 0.0, LPP)
    else:
        S_wet = compute_s_wet_hm(LPP, BWL, draft,
                                 hm_coeffs['CB'], hm_coeffs['CM'], hm_coeffs['CWP'])

    AT = compute_at(Y_surf, Z_surf, draft)

    hm_warnings = check_applicability(LPP, BWL, draft,
                                      CP_hm, hm_coeffs['CM'], lcb_pct, iE)

    res_table = build_resistance_table(
        L=LPP, B=BWL, T=draft, Vol=V_disp,
        CB=hm_coeffs['CB'], CP=CP_hm,
        CM=hm_coeffs['CM'], CWP=hm_coeffs['CWP'],
        lcb_pct=lcb_pct, S=S_wet, AT=AT, iE=iE,
        Cstern=Cstern, method=1,
        design_speed_kn=design_speed,
        max_speed_delta_kn=max_delta,
        steps_per_kn=steps_per_kn,
    )

    if verbose:
        if hm_warnings:
            print("  HM applicability warnings:")
            for w in hm_warnings:
                print(f"    ! {w}")
        des_row = next((r for r in res_table
                        if abs(r['speed_kn'] - design_speed) < 0.01), None)
        if des_row:
            print(f"  @ {design_speed:.1f} kn:  "
                  f"Rtot={des_row['Rtot_kN']:.1f} kN  "
                  f"PE={des_row['PE_kW']:.0f} kW  "
                  f"Fn={des_row['Fn']:.3f}")
        print(f"  iE={iE:.1f}°  S_wet={S_wet:.1f} m²  AT={AT:.2f} m²  "
              f"1+k1={res_table[0]['one_k1'] if res_table else 0:.3f}")

    # --- Build stability result dict (same interface as before) ---
    stab = dict(
        total_mass_kg = m_total,
        draft_m       = draft,
        lcg_total     = (equil['m_dry']*equil['lcg_dry'] +
                         equil['m1']*equil['lcg1'] +
                         equil['m2']*equil['lcg2'] +
                         equil['m3']*equil['lcg3']) / m_total,
        tcg_total     = 0.0,  # enforced by equilibrium
        vcg_total     = KG,
        lcb           = equil['lcb'],
        tcb           = 0.0,
        vcb           = KB,
        KB            = KB,
        KG            = KG,
        BM            = BM,
        gm            = GM,
        Ix_wp         = equil['Ix_wp'],
        v1=equil['v1'], v2=equil['v2'], v3=equil['v3'],
        m1=equil['m1'], m2=equil['m2'], m3=equil['m3'],
        lcg1=equil['lcg1'], lcg2=equil['lcg2'], lcg3=equil['lcg3'],
        vcg1=equil['vcg1'], vcg2=vcg2,           vcg3=equil['vcg3'],
        t1_fill_pct=equil['t1_fill_pct'],
        t2_fill_pct=t2_fill_pct,
        t3_fill_pct=equil['t3_fill_pct'],
    )

    # --- Write computed values back to config.json ---
    config_path = _PARENT / 'config.json'
    with config_path.open('r', encoding='utf-8') as f:
        cfg_disk = json.load(f)
    cfg_disk['Tank2_Fill_pct']         = round(t2_fill_pct, 4)
    cfg_disk['Tank3_Fill_pct']         = round(equil['t3_fill_pct'], 4)
    cfg_disk['Tank2_Center_from_AP_m'] = round(t2_center, 6)
    cfg_disk['Target_Draft_m']         = round(draft, 6)
    with config_path.open('w', encoding='utf-8') as f:
        json.dump(cfg_disk, f, indent=2)
    # Also update in-memory cfg_dict for downstream use
    cfg_dict = dict(cfg_dict)
    cfg_dict['Tank2_Fill_pct']         = t2_fill_pct
    cfg_dict['Tank3_Fill_pct']         = equil['t3_fill_pct']
    cfg_dict['Tank2_Center_from_AP_m'] = t2_center
    cfg_dict['Target_Draft_m']         = draft

    # --- Buoyant CSA at target draft ---
    if verbose:
        print("Computing buoyant cross-sections at target draft...")
    buoyant_x, buoyant_csa = compute_buoyant_csa(
        X_surf, Y_surf, Z_surf, draft, x_min=0.0, x_max=LPP)

    # --- Stern area ---
    stern_poly_y = np.concatenate([[0.0], Y_surf[0], -Y_surf[0][::-1]])
    stern_poly_z = np.concatenate([[0.0], Z_surf[0],  Z_surf[0][::-1]])
    stern_area   = _polygon_area_2d(stern_poly_y, stern_poly_z)

    # --- Bulkhead data ---
    bhd_data = _build_bhd_data(geo, hull_area_entries)

    # --- Longitudinal strength ---
    if verbose:
        print("Running longitudinal strength calculation...")

    result = run_strength_calculation(
        geo        = geo,
        cfg_dict   = cfg_dict,
        shell_data = shell,
        buoyant_x  = buoyant_x,
        buoyant_csa= buoyant_csa,
        tank1_x    = t1_x,  tank1_csa = t1_csa,
        tank2_x    = t2_x,  tank2_csa = t2_csa,
        tank3_x    = t3_x,  tank3_csa = t3_csa,
        diag1      = diag1,
        diag2      = diag2,
        diag3      = diag3,
        stern_area = stern_area,
        bhd_data   = bhd_data,
        hull_mass_data   = {'m_dry': equil['m_dry'], 'lcg_dry': equil['lcg_dry']},
        stability_result = stab,
    )

    
    # --- Transverse moment breakdown (om X-as / langsscheepse as) -----------
    print(f"DEBUG M_DRY={equil['m_dry']/1000:.1f}t, M_TARGET={equil['m_target']/1000:.1f}t, T1={equil['m1']/1000:.1f}t, T2={equil['m2']/1000:.1f}t, T3={equil['m3']/1000:.1f}t")
    
    _G = 9.81
    M_trans_dry_mnm  = equil['m_dry'] * equil['tcg_dry'] * _G / 1e6
    M_trans_t1_mnm   = equil['m1'] * equil['tcg1'] * _G / 1e6
    M_trans_t3_mnm   = equil['m3'] * equil.get('tcg3', 0.0) * _G / 1e6
    M_trans_total_mnm = M_trans_dry_mnm + M_trans_t1_mnm + M_trans_t3_mnm
    result['M_trans_dry_mnm']   = M_trans_dry_mnm
    result['M_trans_t1_mnm']    = M_trans_t1_mnm
    result['M_trans_t3_mnm']    = M_trans_t3_mnm
    result['M_trans_total_mnm'] = M_trans_total_mnm

    result['draft_m']           = draft
    result['Ix_wp']             = equil['Ix_wp']
    result['t1_fill_pct_equil'] = equil['t1_fill_pct']
    result['t2_fill_pct_equil'] = t2_fill_pct
    result['t3_fill_pct_equil'] = equil['t3_fill_pct']
    result['lcg2']              = equil['lcg2']
    result['resistance_table']  = res_table
    result['hm_coeffs']         = hm_coeffs
    result['hm_iE']             = iE
    result['hm_S_wet']          = S_wet
    result['hm_AT']             = AT
    result['hm_design_speed_kn']= design_speed
    result['hm_warnings']       = hm_warnings
    result['crane_check']       = crane_check

    # --- Output ---
    result['strength_source'] = 'engineering'

    if save_outputs:
        out_dir  = _HERE / 'output'
        data_dir = _HERE / 'Data'

        if verbose:
            print(f"Saving data files to {data_dir}...")
        save_data_folder(
            result            = result,
            geo               = geo,
            cfg_dict          = cfg_dict,
            shell             = shell,
            buoyant_x         = buoyant_x,
            buoyant_csa       = buoyant_csa,
            tank1_x           = t1_x,  tank1_csa = t1_csa,
            tank2_x           = t2_x,  tank2_csa = t2_csa,
            tank3_x           = t3_x,  tank3_csa = t3_csa,
            diag1             = diag1,
            diag2             = diag2,
            diag3             = diag3,
            bhd_data          = bhd_data,
            hull_area_entries = hull_area_entries,
            out_data_dir      = data_dir,
        )

        # Strength values/plots should follow groep10 logic.
        try:
            _sync_data_to_groep10(data_dir)
            g10_strength = _load_groep10_strength()
            # Do NOT overwrite actual engineering residuals with legacy groep10 output!
            # The legacy groep10 code doesn't read the TPs or Crane from the CSV files,
            # so its mass balance is fundamentally broken for new designs.
            for k in ['krachtrestant_kn', 'momentrestant_mnm', 'max_sigma_bodem', 'max_sigma_dek', 'max_doorbuiging_mm']:
                if k in g10_strength:
                    del g10_strength[k]
            result.update(g10_strength)
            result['strength_source'] = 'groep10'
        except Exception as exc:
            if verbose:
                print(f"Warning: groep10 strength pipeline failed, using internal strength result ({exc}).")

        if verbose:
            print(f"Saving plots to {out_dir}...")
        save_plots(result, out_dir)
        save_antwoordenblad(result, out_dir)
        info_text = save_info_txt(result, geo, cfg_dict, out_dir)

        # Keep engineering output plots exactly aligned with groep10 strength plots.
        if result.get('strength_source') == 'groep10':
            _copy_groep10_plots_to_engineering(out_dir)

        if verbose:
            print()
            print(info_text)
            print()
            print(f"Output saved to  : {out_dir}")
            print(f"Data files saved : {data_dir}")

    return result


if __name__ == '__main__':
    run_all(verbose=True)
