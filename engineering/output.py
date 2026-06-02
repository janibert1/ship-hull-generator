# -*- coding: utf-8 -*-
"""
output.py — Save plots, antwoordenblad.json, and info.txt.

All plots match the reference style: black line, grid where applicable,
same axis labels and titles as langsscheepse_sterkte.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for file saving
import matplotlib.pyplot as plt
import numpy as np


SIGMA_ALLOW = 190.0   # MPa (S235 allowable)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_line(x, y, xlabel: str, ylabel: str, title: str,
               grid: bool = False, save_path=None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, 'k', linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if grid:
        ax.grid(True)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_plots(result: dict, out_dir: str | Path) -> None:
    """Save 9 PNG plots to out_dir, matching reference layout.

    result : dict returned by run_strength_calculation()
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = result['x_fijn']

    # 1. Verdeelde belasting
    _plot_line(x, result['verdeelde_belasting'] / 1e3,
               'Lengte [m]', 'Verdeelde belasting q(x) [kN/m]',
               'Verdeelde belasting q(x)',
               grid=False,
               save_path=out_dir / 'verdeelde_belasting.png')

    # 2. Dwarskrachtlijn
    _plot_line(x, result['dwarskrachtlijn'] / 1e6,
               'Lengte [m]', 'Dwarskracht V(x) [MN]',
               'Dwarskrachtenlijn V(x)',
               grid=True,
               save_path=out_dir / 'dwarskrachtenlijn.png')

    # 3. Momentlijn
    _plot_line(x, result['momentlijn'] / 1e6,
               'Lengte [m]', 'Buigend moment M(x) [MNm]',
               'Momentenlijn M(x)',
               grid=True,
               save_path=out_dir / 'momentenlijn.png')

    # 4. Traagheidsmoment
    _plot_line(x, result['traagheidsmoment'],
               'Lengte [m]', 'Traagheidsmoment I(x) [m⁴]',
               'Traagheidsmoment I(x)',
               grid=False,
               save_path=out_dir / 'traagheidsmoment.png')

    # 5. Buigstijfheid
    _plot_line(x, result['buigstijfheid'],
               'Lengte [m]', 'Buigstijfheid EI(x) [Nm²]',
               'Buigstijfheid EI(x)',
               grid=False,
               save_path=out_dir / 'buigstijfheid.png')

    # 6. Gereduceerd moment (kappa)
    _plot_line(x, result['gereduceerd_moment'],
               'Lengte [m]', 'Gereduceerd moment κ(x) [1/m]',
               'Gereduceerd moment κ(x)',
               grid=False,
               save_path=out_dir / 'gereduceerd_moment.png')

    # 7. Buigspanning (dual line)
    sig_b = result['sigma_bodem_mpa']
    sig_d = result['sigma_dek_mpa']
    max_abs = max(float(np.max(np.abs(sig_b))), float(np.max(np.abs(sig_d))), 1.0)
    y_range = max(max_abs * 1.4, 20.0)   # auto-scale, minimum ±20 MPa
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, sig_b, 'b', label='Bodem', linewidth=1.0)
    ax.plot(x, sig_d, 'r', label='Dek',   linewidth=1.0)
    if SIGMA_ALLOW <= y_range:
        ax.axhline( SIGMA_ALLOW, color='k', linestyle='--',
                    label=f'σ_y = {SIGMA_ALLOW:.0f} MPa', linewidth=0.8)
        ax.axhline(-SIGMA_ALLOW, color='k', linestyle='--', linewidth=0.8)
    else:
        ax.annotate(f'σ_y = {SIGMA_ALLOW:.0f} MPa (buiten schaal)',
                    xy=(0.01, 0.97), xycoords='axes fraction',
                    fontsize=8, color='k', va='top')
    ax.set_ylim(-y_range, y_range)
    ax.set_xlabel('Lengte [m]')
    ax.set_ylabel('Buigspanning [MPa]')
    ax.set_title('Buigspanning dek en bodem over scheepslengte')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(out_dir / 'buigspanning.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 8. Hoekverdraaiing
    _plot_line(x, result['hoekverdraaiing'],
               'Lengte [m]', 'Hoekverdraaiing θ(x) [rad]',
               'Hoekverdraaiing over scheepslengte',
               grid=True,
               save_path=out_dir / 'hoekverdraaiing.png')

    # 9. Doorbuiging
    _plot_line(x, result['doorbuiging'],
               'Lengte [m]', 'Doorbuiging w(x) [m]',
               'Doorbuiging over scheepslengte',
               grid=True,
               save_path=out_dir / 'doorbuiging.png')

    # 10. Weerstand (Holtrop-Mennen) — only if resistance table present
    res_table = result.get('resistance_table')
    if res_table:
        v_kn   = [r['speed_kn'] for r in res_table]
        rtot   = [r['Rtot_kN']  for r in res_table]
        pe_kw  = [r['PE_kW']    for r in res_table]
        des_v  = result.get('hm_design_speed_kn', 0.0)

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.plot(v_kn, rtot,  'k',  linewidth=1.2, label='Rtot [kN]')
        ax2.plot(v_kn, pe_kw, 'b--', linewidth=1.0, label='PE [kW]')
        if des_v > 0:
            ax1.axvline(des_v, color='r', linestyle=':', linewidth=0.9,
                        label=f'Vs={des_v:.1f} kn')
        ax1.set_xlabel('Snelheid [kn]')
        ax1.set_ylabel('Totale weerstand Rtot [kN]')
        ax2.set_ylabel('Effectief vermogen PE [kW]')
        ax1.set_title('Holtrop-Mennen weerstand')
        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='upper left')
        ax1.grid(True)
        plt.tight_layout()
        fig.savefig(out_dir / 'weerstand.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def save_antwoordenblad(result: dict, out_dir: str | Path) -> None:
    """Save antwoordenblad.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = {
        "Displacement_kg":            result.get('displacement_kg', 0.0),
        "LCG_m":                      result.get('lcg_total', 0.0),
        "TCG_m":                      result.get('tcg_total', 0.0),
        "VCG_m":                      result.get('vcg_total', 0.0),
        "LCB_m":                      result.get('lcb', 0.0),
        "TCB_m":                      result.get('tcb', 0.0),
        "VCB_m":                      result.get('vcb', 0.0),
        "GM_m":                       result.get('gm', 0.0),
        "Draft_m":                    result.get('draft_m', 0.0),
        "Tank1_fill_pct":             result.get('tank1_pct', 0.0),
        "Tank2_fill_pct":             result.get('tank2_pct', 0.0),
        "Tank3_fill_pct":             result.get('tank3_pct', 0.0),
        "Tank2_LCG_m":                result.get('tank2_lcg', 0.0),
        "Max_sigma_bodem_MPa":        result.get('max_sigma_bodem', 0.0),
        "Max_sigma_dek_MPa":          result.get('max_sigma_dek', 0.0),
        "Allowable_stress_MPa":       SIGMA_ALLOW,
        "Max_deflection_mm":          result.get('max_doorbuiging_mm', 0.0),
        "Max_moment_MNm":             result.get('max_moment_nm', 0.0) / 1e6,
        "Location_max_moment_m":      result.get('locatie_max_moment', 0.0),
        "Location_max_deflection_m":  result.get('locatie_max_doorbuiging', 0.0),
        "Force_residual_kN":                    result.get('krachtrestant_kn', 0.0),
        "Moment_residual_trim_Y_axis_MNm":      result.get('momentrestant_mnm', 0.0),
        "Moment_heel_dry_loads_X_axis_MNm":     result.get('M_trans_dry_mnm', 0.0),
        "Moment_heel_tank1_X_axis_MNm":         result.get('M_trans_t1_mnm', 0.0),
        "Moment_heel_tank3_compensation_X_axis_MNm": result.get('M_trans_t3_mnm', 0.0),
        "Moment_heel_residual_X_axis_MNm":      result.get('M_trans_total_mnm', 0.0),
        "TP_info":                    result.get('tp_info', []),
    }

    path = out_dir / 'antwoordenblad.json'
    with path.open('w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def save_info_txt(result: dict, geo: dict, cfg_dict: dict,
                  out_dir: str | Path) -> str:
    """Save info.txt and return the formatted string."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LPP     = geo['LPP']
    B_half  = geo['B_half']
    D       = geo['D']
    LOA     = float(cfg_dict.get('Length_Loa_m', LPP))
    BOA     = B_half * 2.0
    hull_t  = float(cfg_dict.get('Hull_Thickness_mm', 8.0))

    res_table     = result.get('resistance_table', [])
    hm_coeffs     = result.get('hm_coeffs', {})
    hm_iE         = result.get('hm_iE', 0.0)
    hm_S_wet      = result.get('hm_S_wet', 0.0)
    hm_AT         = result.get('hm_AT', 0.0)
    hm_des_v      = result.get('hm_design_speed_kn', 0.0)
    hm_warnings   = result.get('hm_warnings', [])
    des_row = next((r for r in res_table
                    if abs(r['speed_kn'] - hm_des_v) < 0.01), None)

    lines = [
        "=" * 60,
        "  LANGSSCHEEPSE STERKTE — SAMENVATTING",
        "=" * 60,
        f"  LOA                : {LOA:.2f} m",
        f"  LPP                : {LPP:.2f} m",
        f"  BOA                : {BOA:.2f} m",
        f"  DOA                : {D:.2f} m",
        f"  Romplaat dikte     : {hull_t:.1f} mm",
        "-" * 60,
        "  GEWICHT & STABILITEIT",
        "-" * 60,
        f"  Displacement       : {result.get('displacement_kg', 0)/1000:.1f} t",
        f"  Draft              : {result.get('draft_m', 0):.3f} m",
        f"  LCG                : {result.get('lcg_total', 0):.2f} m",
        f"  TCG                : {result.get('tcg_total', 0):.3f} m",
        f"  VCG                : {result.get('vcg_total', 0):.3f} m",
        f"  LCB                : {result.get('lcb', 0):.2f} m",
        f"  VCB                : {result.get('vcb', 0):.3f} m",
        f"  GM                 : {result.get('gm', 0):.3f} m",
        "-" * 60,
        "  TANKVULLINGEN (na evenwicht)",
        "-" * 60,
        f"  Tank 1 (SB)        : {result.get('t1_fill_pct_equil', result.get('tank1_pct', 0)):.1f} %",
        f"  Tank 2 (midden)    : {result.get('t2_fill_pct_equil', result.get('tank2_pct', 0)):.1f} %  (LCG {result.get('lcg2', result.get('tank2_lcg', 0)):.2f} m)",
        f"  Tank 3 (BB)        : {result.get('t3_fill_pct_equil', result.get('tank3_pct', 0)):.1f} %",
        "-" * 60,
        "  STERKTE",
        "-" * 60,
        f"  Max sigma bodem    : {result.get('max_sigma_bodem', 0):.1f} MPa",
        f"  Max sigma dek      : {result.get('max_sigma_dek', 0):.1f} MPa",
        f"  Toelaatbaar        : {SIGMA_ALLOW:.0f} MPa",
        f"  Max moment         : {result.get('max_moment_nm', 0)/1e6:.2f} MNm  "
        f"  @ x = {result.get('locatie_max_moment', 0):.1f} m",
        f"  Max doorbuiging    : {result.get('max_doorbuiging_mm', 0):.1f} mm  "
        f"  @ x = {result.get('locatie_max_doorbuiging', 0):.1f} m",
        f"  Krachtrestant (vert.): {result.get('krachtrestant_kn', 0):.2f} kN",
        f"  Momentrestant trim (om Y-as): {result.get('momentrestant_mnm', 0):.4f} MNm",
        "-" * 60,
        "  TRANSVERSAAL MOMENT (om X-as)",
        "-" * 60,
        f"  Last droog + T1      : {result.get('M_trans_dry_mnm', 0) + result.get('M_trans_t1_mnm', 0):.3f} MNm",
        f"    w.v. last droog    : {result.get('M_trans_dry_mnm', 0):.3f} MNm  (+ = BB, − = SB)",
        f"    w.v. tank 1 (SB)   : {result.get('M_trans_t1_mnm', 0):.3f} MNm",
        f"  Tank 3 compensatie   : {result.get('M_trans_t3_mnm', 0):.3f} MNm  (+ = BB)",
        f"  Restmoment (om X-as) : {result.get('M_trans_total_mnm', 0):.4f} MNm  (≈ 0 na evenwicht)",
        "=" * 60,
    ]

    if res_table:
        lines += [
            "  HOLTROP-MENNEN WEERSTAND (1984)",
            "-" * 60,
            f"  Methode            : HM 1984",
            f"  iE (halve intr.h.) : {hm_iE:.1f}°",
            f"  S_wet              : {hm_S_wet:.1f} m²",
            f"  AT (transom)       : {hm_AT:.2f} m²",
            f"  CB                 : {hm_coeffs.get('CB', 0):.3f}",
            f"  CP                 : {hm_coeffs.get('CP', 0):.3f}",
            f"  CM                 : {hm_coeffs.get('CM', 0):.3f}",
            f"  CWP                : {hm_coeffs.get('CWP', 0):.3f}",
            f"  lcb                : {hm_coeffs.get('lcb_pct', 0):+.2f}% L",
            "-" * 60,
        ]
        if des_row:
            lines += [
                f"  @ Vs = {hm_des_v:.1f} kn  (Fn={des_row['Fn']:.3f})",
                f"    Rtot           : {des_row['Rtot_kN']:.1f} kN",
                f"    RF             : {des_row['RF_N']/1e3:.1f} kN",
                f"    R_visc (1+k1)  : {des_row['R_visc_N']/1e3:.1f} kN  "
                f"  (1+k1 = {des_row['one_k1']:.3f})",
                f"    R_wave         : {des_row['RW_N']/1e3:.1f} kN",
                f"    R_transom      : {des_row['RTR_N']/1e3:.1f} kN",
                f"    R_corr (CA)    : {des_row['RA_N']/1e3:.1f} kN",
                f"    PE             : {des_row['PE_kW']:.0f} kW",
            ]
        if hm_warnings:
            lines.append("  Toepasbaarheids­waarschuwingen:")
            for w in hm_warnings:
                lines.append(f"    ! {w}")
        lines.append("=" * 60)
    text = "\n".join(lines)

    path = out_dir / 'info.txt'
    with path.open('w', encoding='utf-8') as f:
        f.write(text)

    return text
