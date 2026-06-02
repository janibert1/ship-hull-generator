# -*- coding: utf-8 -*-
"""
data_files.py — Write all Data/ folder files in the same format as the
reference Calculations/Data/ directory.

Main entry: save_data_folder(result, geo, cfg_dict, out_data_dir)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CSV helpers (no pandas)
# ---------------------------------------------------------------------------

def _write_csv(path: Path, header: list[str], rows) -> None:
    """Write a CSV file with a header row."""
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f'{v:.6g}' if isinstance(v, float) else v for v in row])


def _write_csv_with_title(path: Path, title: str,
                           header: list[str], rows) -> None:
    """Write a CSV with a title row, then the header."""
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([title])
        w.writerow(header)
        for row in rows:
            w.writerow([f'{v:.6g}' if isinstance(v, float) else v for v in row])


def _pick_float(mapping: dict, keys: list[str], default: float = 0.0) -> float:
    """Return first numeric key found in mapping, else default."""
    for k in keys:
        if k in mapping:
            try:
                return float(mapping[k])
            except Exception:
                continue
    return float(default)


# ---------------------------------------------------------------------------
# Individual file writers
# ---------------------------------------------------------------------------

def _save_main_ship_particulars(path: Path, result: dict, geo: dict,
                                 cfg_dict: dict) -> None:
    LPP    = geo['LPP']
    B_half = geo['B_half']
    D      = geo['D']
    LOA    = float(cfg_dict.get('Length_Loa_m', LPP))
    BOA    = B_half * 2.0
    hm = result.get('hm_coeffs', {}) if isinstance(result.get('hm_coeffs', {}), dict) else {}
    draft = _pick_float(result, ['draft_m'], default=0.0)
    disp_kg = _pick_float(result, ['displacement_kg', 'total_mass_kg'], default=0.0)
    buoy_vol = disp_kg / 1025.0 if disp_kg > 0.0 else 0.0
    bilge_r = float(cfg_dict.get('Bilge_Radius_m', cfg_dict.get('Bilge_Radius_m_max', 0.0)))
    wl_entry_pct = float(cfg_dict.get('Entrance_Angle_Factor_pct_BWL', 30.0))
    ie_deg = _pick_float(result, ['hm_iE'], default=0.0)
    lcb = _pick_float(result, ['lcb'], default=0.0)
    tcb = _pick_float(result, ['tcb'], default=0.0)
    vcb = _pick_float(result, ['vcb'], default=0.0)
    lcg = _pick_float(result, ['lcg_total'], default=0.0)
    tcg = _pick_float(result, ['tcg_total'], default=0.0)
    vcg = _pick_float(result, ['vcg_total'], default=0.0)
    ix_wp = _pick_float(result, ['Ix_wp'], default=0.0)

    doc = {
        "MAIN DIMENSIONS": {
            "Loa_m":  LOA,
            "B_m":    BOA,
            "H_m":    D,
            "Lpp_m":  LPP,
            "Lwl_m":  float(hm.get('LWL', LOA)),
            "X_midship_aft_m": 0.303 * LPP,
            "X_midship_fwd_m": 0.657 * LPP,
            "Bilge_Radius_m": bilge_r,
        },
        "DRAUGHT DATA": {
            "T_moulded_m": draft,
            "T_aft_m": draft,
            "T_fwd_m": draft,
            "heel_deg": 0.0,
        },
        "WATERLINE ENTRANCE ANGLE": {
            "Waterline_Entrace_angle_deg": ie_deg,
            "Location_of_WEA_%Bwl": wl_entry_pct,
            "XYZ_location_of_WEA_m": [0.92 * LOA, -0.15 * BOA, draft],
        },
        "VOLUME RELATED DATA (MOULDED)": {
            "Buoyant_Volume_m3": buoy_vol,
            "Total_Volume_m3": buoy_vol,
            "COB_m": [lcb, tcb, vcb],
            "COV_Total_m": [lcg, tcg, vcg],
            "Cb_pp": float(hm.get('CB', 0.0)),
            "Cb_wl": float(hm.get('CB', 0.0)),
        },
        "DATA OF UNDERWATER AREAS (MOULDED)": {
            "Water_Plane_Area_m2": float(hm.get('AWP', 0.0)),
            "COF_m": [lcb, 0.0, draft],
            "Inertia_WPA_around_COF_m4": [ix_wp, 0.0, ix_wp],
            "Wetted_Shell_Area_m2": _pick_float(result, ['hm_S_wet'], default=0.0),
            "Wetted_Transom_Area_m2": _pick_float(result, ['hm_AT'], default=0.0),
            "Am_m2": float(hm.get('AM', 0.0)),
        },
        "_export_info": {
            "created": "",
            "source_file": "generated",
            "group": 98.0,
            "version": 3,
            "subversion": 0.0,
        },
    }
    with path.with_suffix('.json').open('w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)

    # CSV layout compatible with reference files.
    csv_path = path.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["MAIN DIMENSIONS Group 98; Version 3.0"])
        w.writerow(["Loa[m]", LOA])
        w.writerow(["B[m]", BOA])
        w.writerow(["H[m]", D])
        w.writerow(["Lpp[m]", LPP])
        w.writerow(["Lwl[m]", doc["MAIN DIMENSIONS"]["Lwl_m"]])
        w.writerow(["X midship aft[m]", doc["MAIN DIMENSIONS"]["X_midship_aft_m"]])
        w.writerow(["X midship fwd[m]", doc["MAIN DIMENSIONS"]["X_midship_fwd_m"]])
        w.writerow(["Bilge Radius[m]", bilge_r])
        w.writerow(["DRAUGHT DATA"])
        w.writerow(["T moulded [m]", draft])
        w.writerow(["T aft [m]", draft])
        w.writerow(["T fwd [m]", draft])
        w.writerow(["heel [deg]", 0.0])
        w.writerow(["WATERLINE ENTRANCE ANGLE"])
        w.writerow(["Waterline Entrace angle [deg]", ie_deg])
        w.writerow(["Location of WEA [%Bwl]", wl_entry_pct])
        w.writerow(["XYZ location of WEA [m]", 0.92 * LOA, -0.15 * BOA, draft])
        w.writerow(["VOLUME RELATED DATA (MOULDED)"])
        w.writerow(["Buoyant Volume [m3]", buoy_vol])
        w.writerow(["Total Volume [m3]", buoy_vol])
        w.writerow(["COB [m]", lcb, tcb, vcb])
        w.writerow(["COV Total [m]", lcg, tcg, vcg])
        w.writerow(["Cb_pp [-]", doc["VOLUME RELATED DATA (MOULDED)"]["Cb_pp"]])
        w.writerow(["Cb_wl [-]", doc["VOLUME RELATED DATA (MOULDED)"]["Cb_wl"]])
        w.writerow(["DATA OF UNDERWATER AREAS (MOULDED)"])
        w.writerow(["Water Plane Area [m2]", float(hm.get('AWP', 0.0))])
        w.writerow(["COF [m]", lcb, 0.0, draft])
        w.writerow(["Inertia WPA around COF [m4]", ix_wp, 0.0, ix_wp])
        w.writerow(["Wetted Shell Area [m2]", _pick_float(result, ['hm_S_wet'], default=0.0)])
        w.writerow(["Wetted Transom Area [m2]", _pick_float(result, ['hm_AT'], default=0.0)])
        w.writerow(["Am [m2]", float(hm.get('AM', 0.0))])


def _save_hull_area_data(path: Path, result: dict, geo: dict,
                          cfg_dict: dict) -> None:
    """HullAreaData.csv — areas and centroids of main structural members."""
    entries = result.get('hull_area_entries', [])
    wanted = [
        ("Stern (transom)", "Transom Area"),
        ("Hull plating", "Shell Area"),
        ("Deck plating", "Deck Area"),
    ]
    mapped_rows = []
    for src_name, out_name in wanted:
        match = next((e for e in entries if str(e.get('desc', '')) == src_name), None)
        if match is None:
            continue
        mapped_rows.append((out_name, match['area'], match['lcg'], match['tcg'], match['vcg']))

    if not mapped_rows:
        # Fallback to original content if expected entries are missing.
        mapped_rows = [(e.get('desc', ''), e.get('area', 0.0), e.get('lcg', 0.0), e.get('tcg', 0.0), e.get('vcg', 0.0))
                       for e in entries]

    header = ["Name ", "Area [m2]", "lca [m]", "tca [m]", "vca [m]"]
    _write_csv_with_title(path, "HULL AREA DATA Group 98; Version 3.0", header, mapped_rows)


def _save_tank_data_json(path: Path, result: dict, geo: dict,
                          cfg_dict: dict, diag1: dict, diag2: dict, diag3: dict) -> None:
    def _diag_value(diag: dict, y_key: str, pct: float) -> float:
        x = np.asarray(diag.get('fill_pct', []), dtype=float)
        y = np.asarray(diag.get(y_key, []), dtype=float)
        if len(x) == 0 or len(y) == 0:
            return 0.0
        return float(np.interp(pct, x, y))

    t1_pct = _pick_float(result, ['tank1_pct', 't1_fill_pct_equil'], default=0.0)
    t2_pct = _pick_float(result, ['tank2_pct', 't2_fill_pct_equil'], default=0.0)
    t3_pct = _pick_float(result, ['tank3_pct', 't3_fill_pct_equil'], default=0.0)
    v1 = _diag_value(diag1, 'volume', t1_pct)
    v2 = _diag_value(diag2, 'volume', t2_pct)
    v3 = _diag_value(diag3, 'volume', t3_pct)
    lcg1 = _diag_value(diag1, 'lcg', t1_pct)
    lcg2 = _pick_float(result, ['tank2_lcg', 'lcg2'], default=_diag_value(diag2, 'lcg', t2_pct))
    lcg3 = _diag_value(diag3, 'lcg', t3_pct)
    tcg1 = _diag_value(diag1, 'tcg', t1_pct)
    tcg2 = _diag_value(diag2, 'tcg', t2_pct)
    tcg3 = _diag_value(diag3, 'tcg', t3_pct)
    vcg1 = _diag_value(diag1, 'vcg', t1_pct)
    vcg2 = _diag_value(diag2, 'vcg', t2_pct)
    vcg3 = _diag_value(diag3, 'vcg', t3_pct)

    LPP = float(geo.get('LPP', 0.0))
    BOA = float(geo.get('B_half', 0.0)) * 2.0
    D = float(geo.get('D', 0.0))
    LOA = float(cfg_dict.get('Length_Loa_m', LPP))
    t1_w = float(cfg_dict.get('Tank1_Width_m', 0.0))
    t3_w = float(cfg_dict.get('Tank3_Width_m', 0.0))
    t2_len = LOA * float(cfg_dict.get('Tank2_Length_pct_Loa', 0.0)) / 100.0
    t2_w = max(0.0, BOA - t1_w - t3_w)

    def _wb_entry(vol: float, lcg: float, tcg: float, vcg: float,
                  fill_pct: float, dims: list[float], diag: dict, use_alt_key: bool) -> dict:
        ix = _diag_value(diag, 'ix_fs', fill_pct)
        area = vol / max((D * fill_pct / 100.0), 1e-6) if fill_pct > 0 else 0.0
        entry = {
            "COV_WB_m": [lcg, tcg, vcg],
            "Height_of_WB_%_of_h_tank": fill_pct,
            "Tank_lxbxh_m": dims,
            "Area_WB_plane_m2": area,
            "COA_WB_plane_m": [lcg, tcg, D * fill_pct / 100.0],
            "Inertia_WB_plane_m4": [ix, 0.0, ix],
        }
        key = "Volume_WB_m3" if use_alt_key else "Volume_water_ballast_m3"
        entry[key] = vol
        return entry

    doc = {
        "WB TANK 1": _wb_entry(v1, lcg1, tcg1, vcg1, t1_pct, [LPP, t1_w, D], diag1, use_alt_key=False),
        "WB TANK 2": _wb_entry(v2, lcg2, tcg2, vcg2, t2_pct, [t2_len, t2_w, D], diag2, use_alt_key=True),
        "WB TANK 3": _wb_entry(v3, lcg3, tcg3, vcg3, t3_pct, [LPP, t3_w, D], diag3, use_alt_key=False),
        "_export_info": {
            "created": "",
            "source_file": "generated",
            "group": 98.0,
            "version": 3,
            "subversion": 0.0,
        },
    }
    with path.open('w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)


def _sample_diagram(diag: dict, sample_points: int = 7) -> dict:
    """Sample a dense fill diagram to a sparse, reference-like set of rows."""
    keys = ('fill_m', 'fill_pct', 'volume', 'lcg', 'tcg', 'vcg', 'ix_fs')
    arrs = {k: np.asarray(diag.get(k, []), dtype=float) for k in keys}
    n = len(arrs['fill_pct'])
    if n == 0:
        return {k: np.array([]) for k in keys}
    if n <= sample_points:
        return arrs
    start = 1 if n > 1 else 0
    idx = np.linspace(start, n - 1, sample_points).round().astype(int)
    idx = np.unique(np.clip(idx, start, n - 1))
    if idx[-1] != (n - 1):
        idx = np.append(idx, n - 1)
    return {k: v[idx] for k, v in arrs.items()}


def _save_tank_diagram_volume(path: Path, diag: dict, name: str) -> None:
    header = [
        "Tankfilling [m]",
        "Tankfilling [% of h_tank]",
        "Tankvolume [m3]",
        "lcg [m]",
        "tcg [m]",
        "vcg [m]",
    ]
    sampled = _sample_diagram(diag, sample_points=7)
    fill_m = sampled['fill_m']
    fill_pct = sampled['fill_pct']
    vol_vals = sampled['volume']
    lcg_vals = sampled['lcg']
    tcg_vals = sampled['tcg']
    vcg_vals = sampled['vcg']

    # Compatibility with legacy spline inversions: enforce strictly
    # increasing volume axis (x) used as CubicSpline input in groep10.
    vol_vals = np.maximum.accumulate(vol_vals)
    eps = 1e-6
    for i in range(1, len(vol_vals)):
        if vol_vals[i] <= vol_vals[i - 1]:
            vol_vals[i] = vol_vals[i - 1] + eps

    if name == "Tank1":
        tcg_vals = np.abs(tcg_vals)
    elif name == "Tank3":
        tcg_vals = -np.abs(tcg_vals)
    rows = list(zip(
        fill_m.tolist(),
        fill_pct.tolist(),
        vol_vals.tolist(),
        lcg_vals.tolist(),
        tcg_vals.tolist(),
        vcg_vals.tolist(),
    ))
    tank_idx = name[-1] if name and name[-1].isdigit() else "?"
    _write_csv_with_title(
        path,
        f"TANKDIAGRAM TNK {tank_idx} VOLUME DATA Group 98; Version 3.0",
        header,
        rows,
    )


def _save_tank_diagram_waterplane(path: Path, diag: dict, name: str) -> None:
    header = [
        "Tankfilling [m]",
        "Tankfilling [% of h_tank]",
        "Area [m2]",
        "lca [m]",
        "tca [m]",
        "vca [m]",
        "Inertia_x [m4]",
        "Inertia_y [m4]",
        "Inertia_z [m4]",
    ]
    sampled = _sample_diagram(diag, sample_points=7)
    fill_m = sampled['fill_m']
    fill_pct = sampled['fill_pct']
    vol = sampled['volume']
    lcg = sampled['lcg']
    tcg = sampled['tcg']
    ix = sampled['ix_fs']

    area = np.zeros_like(fill_m)
    if len(fill_m) > 1:
        area = np.gradient(vol, fill_m, edge_order=1)
        area = np.clip(area, 0.0, None)

    rows = list(zip(
        fill_m.tolist(),
        fill_pct.tolist(),
        area.tolist(),
        lcg.tolist(),
        tcg.tolist(),
        fill_m.tolist(),
        ix.tolist(),
        [0.0] * len(fill_m),
        ix.tolist(),
    ))
    tank_idx = name[-1] if name and name[-1].isdigit() else "?"
    _write_csv_with_title(
        path,
        f"TANKDIAGRAM TNK {tank_idx} WATERPLANE DATA Group 98; Version 3.0",
        header,
        rows,
    )


def _save_shell_csa(path: Path, shell: dict) -> None:
    """Shell_CSA.csv — one row per loft station."""
    header = [
        "X [m]",
        "OUTLINE LENGTH [m]",
        "CROSS SECTION AREA OF SHELL PLATING [m2]",
        "CENTROID_X[m]",
        "CENTROID_Y[m]",
        "CENTROID_Z[m]",
        "INERTIA_X[m4]",
        "INERTIA_Y[m4]",
        "INERTIA_Z[m4]",
        "Z_Keel[m]",
        "Z_DECK[m]",
    ]
    n = len(shell['x'])
    rows = []
    for i in range(n):
        x   = float(shell['x'][i])
        out = float(shell['outline_length'][i])
        csa = float(shell['shell_csa_1mm'][i])
        cy  = 0.0
        cz  = float(shell['centroid_z'][i])
        iy  = float(shell['inertia_y_1mm'][i])
        zk  = float(shell['z_keel'][i])
        zd  = float(shell['z_deck'][i])
        # Ix, Iz are small secondary values; approximate as Iy/10
        ix_val = iy / 10.0
        iz_val = iy / 10.0
        rows.append((x, out, csa, x, cy, cz, ix_val, iy, iz_val, zk, zd))
    _write_csv_with_title(path, "Shell Cross-Section Data (1mm convention)", header, rows)


def _save_buoyant_csa(path: Path, x_arr: np.ndarray, csa_arr: np.ndarray) -> None:
    header = ["x_in_m", "crossarea_in_m2"]
    rows = list(zip(x_arr.tolist(), csa_arr.tolist()))
    _write_csv_with_title(path, "Buoyant Cross-Section Area", header, rows)


def _save_tank_csa(path: Path, x_arr: np.ndarray, csa_arr: np.ndarray,
                   name: str) -> None:
    header = ["x_in_m", "crossarea_in_m2"]
    rows = list(zip(x_arr.tolist(), csa_arr.tolist()))
    _write_csv_with_title(path, f"{name} Cross-Section Area at 100% fill",
                          header, rows)


def _save_total_csa(path: Path,
                    buoyant_x: np.ndarray, buoyant_csa: np.ndarray,
                    t1_x: np.ndarray, t1_csa: np.ndarray,
                    t2_x: np.ndarray, t2_csa: np.ndarray,
                    t3_x: np.ndarray, t3_csa: np.ndarray) -> None:
    """Total_CSA.csv — buoyant + all tanks at 100% fill."""
    # Interpolate everything onto the buoyant x-grid
    def interp0(xo, yo, xn):
        from scipy.interpolate import interp1d
        f = interp1d(xo, yo, bounds_error=False, fill_value=0.0)
        return f(xn)

    x = buoyant_x
    total = (buoyant_csa
             + interp0(t1_x, t1_csa, x)
             + interp0(t2_x, t2_csa, x)
             + interp0(t3_x, t3_csa, x))

    header = ["x_in_m", "crossarea_m2"]
    rows = list(zip(x.tolist(), total.tolist()))
    _write_csv_with_title(path, "Total CSA (buoyant + all tanks 100%)", header, rows)


def _save_tank_bhd_data(path: Path, bhd_data: list) -> None:
    header = ["BHD Area [m2]", "lcg [m]", "tcg [m]", "vcg [m]", "x_min [m]", "x_max [m]"]
    rows = [(b['area'], b.get('lcg', 0.5 * (b['x_min'] + b['x_max'])),
             b.get('tcg', 0.0), b.get('vcg', 0.0), b['x_min'], b['x_max'])
            for b in bhd_data]
    _write_csv_with_title(path, "TANK BULKHEAD DATA Group 98; Version 3.0", header, rows)


def _save_resistance_data(path: Path, result: dict) -> None:
    """Write ResistanceData.csv compatible with original Gr98 format."""
    res_table = result.get('resistance_table', [])
    design_speed = _pick_float(result, ['hm_design_speed_kn'], default=14.0)
    design_row = None
    for r in res_table:
        if abs(float(r.get('speed_kn', 0.0)) - design_speed) < 1e-6:
            design_row = r
            break
    if design_row is None and res_table:
        design_row = min(res_table, key=lambda r: abs(float(r.get('speed_kn', 0.0)) - design_speed))

    header = [
        "V [kn]", "V [m/s]", "Fn", "Rtot [N]", "R_visc [N]", "R_app [N]",
        "R_w [N]", "R_TR [N]", "R_b [N]", "R_A [N]", "R_BTO [N]", "w[-]", "t[-]",
    ]

    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["RESISTANCE ESTIMATION BY METHOD HOLTROP & MENNEN 1982 & 1984"])
        w.writerow(["DESIGN SPEED DATA"])
        w.writerow(header)
        if design_row is not None:
            w.writerow([
                design_row.get('speed_kn', 0.0),
                design_row.get('V_ms', 0.0),
                design_row.get('Fn', 0.0),
                design_row.get('Rtot_N', 0.0),
                design_row.get('R_visc_N', 0.0),
                0.0,
                design_row.get('RW_N', 0.0),
                design_row.get('RTR_N', 0.0),
                design_row.get('RB_N', 0.0),
                design_row.get('RA_N', 0.0),
                0.0,
                0.0,
                0.0,
            ])
        else:
            w.writerow([design_speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        w.writerow(["RESISTANCE TABLE BY METHOD HOLTROP & MENNEN 1982 & 1984"])
        w.writerow(header)
        for r in res_table:
            w.writerow([
                r.get('speed_kn', 0.0),
                r.get('V_ms', 0.0),
                r.get('Fn', 0.0),
                r.get('Rtot_N', 0.0),
                r.get('R_visc_N', 0.0),
                0.0,
                r.get('RW_N', 0.0),
                r.get('RTR_N', 0.0),
                r.get('RB_N', 0.0),
                r.get('RA_N', 0.0),
                0.0,
                0.0,
                0.0,
            ])


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def save_data_folder(result: dict, geo: dict, cfg_dict: dict,
                     shell: dict,
                     buoyant_x: np.ndarray, buoyant_csa: np.ndarray,
                     tank1_x: np.ndarray, tank1_csa: np.ndarray,
                     tank2_x: np.ndarray, tank2_csa: np.ndarray,
                     tank3_x: np.ndarray, tank3_csa: np.ndarray,
                     diag1: dict, diag2: dict, diag3: dict,
                     bhd_data: list,
                     hull_area_entries: list,
                     out_data_dir: str | Path) -> None:
    """Write all Data/ files to out_data_dir."""
    out  = Path(out_data_dir)
    out.mkdir(parents=True, exist_ok=True)

    _save_main_ship_particulars(out / 'MainShipParticulars', result, geo, cfg_dict)
    _save_hull_area_data(out / 'HullAreaData.csv', result, geo, cfg_dict)
    _save_tank_data_json(out / 'TankData.json', result, geo, cfg_dict, diag1, diag2, diag3)

    _save_tank_diagram_volume(out / 'Tank1_Diagram_Volume.csv',  diag1, 'Tank1')
    _save_tank_diagram_volume(out / 'Tank2_Diagram_Volume.csv',  diag2, 'Tank2')
    _save_tank_diagram_volume(out / 'Tank3_Diagram_Volume.csv',  diag3, 'Tank3')

    _save_tank_diagram_waterplane(out / 'Tank1_Diagram_Waterplane.csv', diag1, 'Tank1')
    _save_tank_diagram_waterplane(out / 'Tank2_Diagram_Waterplane.csv', diag2, 'Tank2')
    _save_tank_diagram_waterplane(out / 'Tank3_Diagram_Waterplane.csv', diag3, 'Tank3')

    _save_shell_csa(out / 'Shell_CSA.csv', shell)
    _save_buoyant_csa(out / 'Buoyant_CSA.csv', buoyant_x, buoyant_csa)

    _save_tank_csa(out / 'Tank1_CSA.csv', tank1_x, tank1_csa, 'Tank1')
    _save_tank_csa(out / 'Tank2_CSA.csv', tank2_x, tank2_csa, 'Tank2')
    _save_tank_csa(out / 'Tank3_CSA.csv', tank3_x, tank3_csa, 'Tank3')

    _save_total_csa(out / 'Total_CSA.csv',
                    buoyant_x, buoyant_csa,
                    tank1_x, tank1_csa,
                    tank2_x, tank2_csa,
                    tank3_x, tank3_csa)

    _save_tank_bhd_data(out / 'TankBHD_Data.csv', bhd_data)

    _save_resistance_data(out / 'ResistanceData.csv', result)

    # HullAreaData: write with actual entries if provided
    if hull_area_entries:
        _save_hull_area_data(out / 'HullAreaData.csv',
                             {'hull_area_entries': hull_area_entries},
                             geo, cfg_dict)
