# -*- coding: utf-8 -*-
"""
resistance.py — Holtrop & Mennen ship resistance calculation.

Implements two variants:
  Method 0 ("1982"): Holtrop & Mennen, Int. Shipbuilding Progress 1982 (Vol.29 No.335)
  Method 1 ("1984"): Holtrop, Int. Shipbuilding Progress 1984 (Vol.31 No.363)

The two main differences are:
  - Form factor (1+k1): different regression formula
  - Wave resistance cosine coefficient: m2 (1982) vs m4 (1984)
  - 1984 adds a high-speed formula for Fn > 0.55 with interpolation at 0.40-0.55

iE regression: same formula in both 1982 and 1984 papers.
"""

from __future__ import annotations

import numpy as np
from .geometry import hull_hw_at_z, submerged_area_half

RHO  = 1025.0      # kg/m³  salt water at 15°C
NU   = 1.1395e-6   # m²/s   kinematic viscosity at 15°C
G    = 9.81        # m/s²
KNOT = 0.5144444   # m/s per knot


# ---------------------------------------------------------------------------
# Half entrance angle iE
# ---------------------------------------------------------------------------

def compute_ie_regression(L: float, B: float, CP: float, CWP: float,
                           lcb_pct: float, Vol: float, LR: float) -> float:
    """Half angle of waterplane entrance — Holtrop regression [degrees].

    Same formula appears in both the 1982 and 1984 papers.
    iE = 1 + 89·exp(-(L/B)^0.80856 · (1-CWP)^0.30484
                     · (1-CP-0.0225·lcb)^0.6367
                     · (LR/B)^0.34574 · (100∇/L³)^0.16302)

    lcb_pct : LCB from 0.5L, positive forward [% L]
    LR      : length of run L·(1-CP + 0.06·CP·lcb/(4CP-1))
    """
    inner = (-(L / B) ** 0.80856
             * (1.0 - CWP) ** 0.30484
             * (1.0 - CP - 0.0225 * lcb_pct) ** 0.6367
             * (LR / B) ** 0.34574
             * (100.0 * Vol / L ** 3) ** 0.16302)
    return float(1.0 + 89.0 * np.exp(inner))


def compute_ie_from_hull(X_surf: np.ndarray, Y_surf: np.ndarray,
                          Z_surf: np.ndarray, T: float,
                          entrance_factor_pct: float) -> float:
    """Half angle of waterplane entrance from hull geometry [degrees].

    Traces the waterplane half-breadth hw(x) at draft T, finds the x-position
    where hw = entrance_factor_pct% of the maximum half-breadth, and returns
    the chord angle from that reference point to the bow tip.

    entrance_factor_pct : % of max waterplane half-breadth (e.g. 30 → 30%)
    """
    x_mid = X_surf.mean(axis=1)
    sort_idx = np.argsort(x_mid)
    xs  = x_mid[sort_idx]
    hws = np.array([hull_hw_at_z(Y_surf[i], Z_surf[i], T) for i in sort_idx])

    hw_max = float(np.max(hws))
    if hw_max < 1e-6:
        return 0.0

    hw_ref = entrance_factor_pct / 100.0 * hw_max
    x_bow  = float(xs[-1])

    # Search in forward half (x > 0.5 * x_bow), from bow aftward
    fwd_mask = xs > 0.5 * x_bow
    if not np.any(fwd_mask):
        fwd_mask = np.ones(len(xs), dtype=bool)
    xs_fwd  = xs[fwd_mask]
    hws_fwd = hws[fwd_mask]

    # Find last downward crossing of hw_ref (closest to bow)
    diff_sign = np.sign(hws_fwd - hw_ref)
    crossings = np.where(np.diff(diff_sign) < 0)[0]  # h going below hw_ref

    if len(crossings) > 0:
        k = crossings[-1]
        h1, h2 = hws_fwd[k], hws_fwd[k + 1]
        x1, x2 = xs_fwd[k],  xs_fwd[k + 1]
        frac = (hw_ref - h1) / (h2 - h1) if abs(h2 - h1) > 1e-12 else 0.0
        x_ref  = x1 + frac * (x2 - x1)
        hw_at  = hw_ref
    else:
        # Fallback: use forward-most station
        x_ref  = xs_fwd[0]
        hw_at  = hws_fwd[0]

    dx = x_bow - x_ref
    if abs(dx) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(hw_at, dx)))


# ---------------------------------------------------------------------------
# Wetted surface area
# ---------------------------------------------------------------------------

def compute_s_wet_hm(L: float, B: float, T: float,
                     CB: float, CM: float, CWP: float,
                     ABT: float = 0.0) -> float:
    """Wetted hull surface area — Holtrop & Mennen regression [m²].

    S = L·(2T+B)·√CM·(0.453 + 0.4425·CB - 0.2862·CM
                        - 0.003467·B/T + 0.3696·CWP) + 2.38·ABT/CB
    """
    S = (L * (2.0 * T + B) * np.sqrt(CM)
         * (0.453 + 0.4425 * CB - 0.2862 * CM
            - 0.003467 * B / T + 0.3696 * CWP))
    if ABT > 0:
        S += 2.38 * ABT / CB
    return float(S)


def _wetted_perimeter_half(y_cs: np.ndarray, z_cs: np.ndarray,
                            T: float) -> float:
    """Arc length of one-side hull contour below waterline T [m]."""
    total = 0.0
    for k in range(len(z_cs) - 1):
        z1, z2 = z_cs[k], z_cs[k + 1]
        y1, y2 = y_cs[k], y_cs[k + 1]
        if z1 <= T and z2 <= T:
            total += np.hypot(y2 - y1, z2 - z1)
        elif z1 <= T < z2:
            frac = (T - z1) / (z2 - z1)
            total += np.hypot(frac * (y2 - y1), T - z1)
        elif z2 <= T < z1:
            frac = (T - z2) / (z1 - z2)
            total += np.hypot(frac * (y1 - y2), T - z2)
    return total


def compute_s_wet_from_hull(X_surf: np.ndarray, Y_surf: np.ndarray,
                             Z_surf: np.ndarray, T: float,
                             x_min: float = 0.0,
                             x_max: float | None = None) -> float:
    """Wetted hull surface area from geometry at draft T [m²].

    Integrates 2 × the half-hull wetted perimeter over x.
    """
    x_mid = X_surf.mean(axis=1)
    if x_max is None:
        x_max = float(x_mid.max())
    mask = (x_mid >= x_min - 1e-6) & (x_mid <= x_max + 1e-6)
    ts   = np.where(mask)[0]
    if len(ts) < 2:
        return 0.0

    xs_arr = [float(x_mid[t]) for t in ts]
    wp_arr = [_wetted_perimeter_half(Y_surf[t], Z_surf[t], T) for t in ts]
    return 2.0 * float(np.trapezoid(wp_arr, xs_arr))


# ---------------------------------------------------------------------------
# Submerged transom area AT
# ---------------------------------------------------------------------------

def compute_at(Y_surf: np.ndarray, Z_surf: np.ndarray, T: float) -> float:
    """Submerged transom area at draft T [m²] (stern cross-section, t=0)."""
    return float(submerged_area_half(Y_surf[0], Z_surf[0], T))


# ---------------------------------------------------------------------------
# Hull coefficients from geometry
# ---------------------------------------------------------------------------

def compute_hull_coefficients(X_surf: np.ndarray, Y_surf: np.ndarray,
                               Z_surf: np.ndarray, T: float,
                               LPP: float, Vol: float,
                               LCB_from_ap: float) -> dict:
    """Compute CB, CM, CWP, CP, lcb_pct for Holtrop-Mennen.

    lcb_pct: LCB position, positive forward of 0.5·LWL [% LWL]
    """
    from .hydrostatics import waterplane_data

    # BWL = max waterplane half-breadth × 2
    hw_arr = [hull_hw_at_z(Y_surf[t], Z_surf[t], T)
              for t in range(len(X_surf))]
    BWL = 2.0 * float(np.max(hw_arr))

    # Waterplane area
    wp   = waterplane_data(X_surf, Y_surf, Z_surf, T, x_min=0.0, x_max=LPP)
    AWP  = wp['Aw']
    CWP  = AWP / (LPP * BWL) if (LPP * BWL) > 0 else 0.75

    CB = Vol / (LPP * BWL * T)

    # Midship area (submerged) at x = 0.5·LPP
    x_mid_arr = X_surf.mean(axis=1)
    idx_mid   = int(np.argmin(np.abs(x_mid_arr - 0.5 * LPP)))
    AM = submerged_area_half(Y_surf[idx_mid], Z_surf[idx_mid], T)
    CM = AM / (BWL * T) if (BWL * T) > 0 else 0.98
    CP = Vol / (AM * LPP) if (AM * LPP) > 0 else CB / CM

    # lcb in HM convention: positive forward of 0.5·L [% L]
    lcb_pct = 100.0 * (LCB_from_ap - 0.5 * LPP) / LPP

    return dict(BWL=BWL, AWP=AWP, CWP=CWP, CB=CB,
                AM=AM, CM=CM, CP=CP, lcb_pct=lcb_pct)


# ---------------------------------------------------------------------------
# Applicability range check
# ---------------------------------------------------------------------------

def check_applicability(L: float, B: float, T: float, CP: float,
                         CM: float, lcb_pct: float,
                         iE: float) -> list[str]:
    """Return list of warning strings for parameters outside HM valid range."""
    warnings = []
    if not (0.5 <= CM <= 1.0):
        warnings.append(f"CM={CM:.3f} outside [0.50, 1.00]")
    if not (3.5 <= L / B <= 9.5):
        warnings.append(f"L/B={L/B:.2f} outside [3.5, 9.5]")
    if not (-5.0 <= lcb_pct <= 5.0):
        warnings.append(f"lcb={lcb_pct:.2f}% outside [-5, +5]%")
    if iE > 70.0:
        warnings.append(f"iE={iE:.1f}° > 70° (max)")
    if not (0.40 <= CP <= 0.93):
        warnings.append(f"CP={CP:.3f} outside [0.40, 0.93]")
    return warnings


# ---------------------------------------------------------------------------
# 1982 form factor
# ---------------------------------------------------------------------------

def _form_factor_1982(L: float, B: float, T: float, CP: float,
                       lcb_pct: float, Vol: float, Cstern: float) -> float:
    """(1+k1) by Holtrop & Mennen 1982."""
    LR = L * (1.0 - CP + 0.06 * CP * lcb_pct / (4.0 * CP - 1.0))

    TL = T / L
    if TL > 0.05:
        c12 = TL ** 0.2228446
    elif TL > 0.02:
        c12 = 48.20 * (TL - 0.02) ** 2.078 + 0.479948
    else:
        c12 = 0.479948

    c13 = 1.0 + 0.003 * Cstern

    return float(c13 * (0.93 + c12
                        * (B / LR) ** 0.92497
                        * (0.95 - CP) ** (-0.521448)
                        * (1.0 - CP + 0.0225 * lcb_pct) ** 0.6906))


# ---------------------------------------------------------------------------
# 1984 form factor
# ---------------------------------------------------------------------------

def _form_factor_1984(L: float, B: float, T: float, CP: float,
                       lcb_pct: float, Vol: float, Cstern: float) -> float:
    """(1+k1) by Holtrop 1984."""
    LR  = L * (1.0 - CP + 0.06 * CP * lcb_pct / (4.0 * CP - 1.0))
    c14 = 1.0 + 0.011 * Cstern
    return float(0.93 + 0.487118 * c14
                 * (B / L) ** 1.06806
                 * (T / L) ** 0.46106
                 * (L / LR) ** 0.121563
                 * (L ** 3 / Vol) ** 0.36486
                 * (1.0 - CP) ** (-0.604247))


# ---------------------------------------------------------------------------
# Shared wave-resistance coefficients
# ---------------------------------------------------------------------------

def _wave_coefficients(L: float, B: float, T: float, CP: float,
                        CM: float, CWP: float, lcb_pct: float, Vol: float,
                        AT: float, iE: float,
                        ABT: float, hB: float, TF: float) -> dict:
    """Pre-compute all speed-independent wave resistance coefficients."""
    LR = L * (1.0 - CP + 0.06 * CP * lcb_pct / (4.0 * CP - 1.0))

    # c7
    BL = B / L
    if BL < 0.11:
        c7 = 0.229577 * BL ** 0.33333
    elif BL <= 0.25:
        c7 = BL
    else:
        c7 = 0.5 - 0.0625 * L / B

    # c1 (wave resistance amplitude)
    c1 = 2223105.0 * c7 ** 3.78613 * (T / B) ** 1.07961 * (90.0 - iE) ** (-1.37565)

    # c2, c3 (bulbous bow)
    if ABT > 1e-6 and TF > hB:
        c3 = 0.56 * ABT ** 1.5 / (B * T * (0.31 * np.sqrt(ABT) + TF - hB))
    else:
        c3 = 0.0
    c2 = np.exp(-1.89 * np.sqrt(c3))

    # c5 (transom influence on wave resistance)
    denom = B * T * CM
    c5 = 1.0 - 0.8 * AT / denom if denom > 0 else 1.0

    # m1
    if CP < 0.8:
        c16 = 8.07981 * CP - 13.8673 * CP ** 2 + 6.984388 * CP ** 3
    else:
        c16 = 1.73014 - 0.7067 * CP
    m1 = (0.0140407 * L / T
          - 1.75254 * Vol ** (1.0 / 3.0) / L
          - 4.79323 * B / L
          - c16)

    # c15 (for m2/m4)
    L3V = L ** 3 / Vol
    if L3V < 512.0:
        c15 = -1.69385
    elif L3V <= 1726.91:
        c15 = -1.69385 + (L / Vol ** (1.0 / 3.0) - 8.0) / 2.36
    else:
        c15 = 0.0

    # lambda
    lam = 1.446 * CP - 0.03 * L / B if L / B < 12.0 else 1.446 * CP - 0.36

    # c17 and m3 for 1984 high-speed formula (Fn > 0.55)
    c17 = (6919.3 * CM ** (-1.3346)
           * (Vol / L ** 3) ** 2.00977
           * (L / B - 2.0) ** 1.40692)
    m3 = -7.2035 * (B / L) ** 0.326869 * (T / B) ** 0.605375

    return dict(c1=c1, c2=c2, c3=c3, c5=c5, m1=m1, c15=c15,
                lam=lam, c17=c17, m3=m3, LR=LR)


# ---------------------------------------------------------------------------
# Main Holtrop-Mennen calculation
# ---------------------------------------------------------------------------

def holtrop_mennen(
    L: float, B: float, T: float, Vol: float,
    CB: float, CP: float, CM: float, CWP: float,
    lcb_pct: float,        # LCB from 0.5L, positive forward [% L]
    S: float,              # wetted surface area [m²]
    AT: float,             # immersed transom area [m²]
    iE: float,             # half waterplane entrance angle [degrees]
    Cstern: float,         # afterbody form coefficient
    speeds_kn: np.ndarray,
    method: int = 1,       # 0 = 1982,  1 = 1984
    ABT: float = 0.0,      # transverse bulb cross-section area [m²]
    hB: float = 0.0,       # centroid height of bulb above keel [m]
    TF: float | None = None,
) -> list[dict]:
    """Full Holtrop-Mennen resistance calculation.

    method 0 = 1982 (uses 1982 form factor and m2 cos coefficient)
    method 1 = 1984 (uses 1984 form factor and m4 cos coefficient,
                      plus high-speed formula for Fn > 0.55)

    Returns list of dicts — one per speed — with resistance components.
    """
    if TF is None:
        TF = T

    # Pre-compute form factor
    if method == 0:
        one_k1 = _form_factor_1982(L, B, T, CP, lcb_pct, Vol, Cstern)
    else:
        one_k1 = _form_factor_1984(L, B, T, CP, lcb_pct, Vol, Cstern)

    # Pre-compute wave coefficients
    wc = _wave_coefficients(L, B, T, CP, CM, CWP, lcb_pct, Vol,
                            AT, iE, ABT, hB, TF)
    c1, c2, c5, m1, c15, lam = (wc['c1'], wc['c2'], wc['c5'],
                                  wc['m1'], wc['c15'], wc['lam'])
    c17, m3 = wc['c17'], wc['m3']

    # c4 and CA (model-ship correlation, speed-independent)
    c4 = min(TF / L, 0.04)
    CA = (0.006 * (L + 100.0) ** (-0.16) - 0.00205
          + 0.003 * np.sqrt(L / 7.5) * CB ** 4 * c2 * (0.04 - c4))

    results = []

    for V_kn in speeds_kn:
        V = float(V_kn) * KNOT
        if V < 1e-6:
            results.append(dict(
                speed_kn=float(V_kn), V_ms=0.0, Fn=0.0,
                Rtot_N=0.0, Rtot_kN=0.0, PE_kW=0.0,
                R_visc_N=0.0, RF_N=0.0, RW_N=0.0,
                RB_N=0.0, RTR_N=0.0, RA_N=0.0,
                CF=0.0, one_k1=float(one_k1),
            ))
            continue

        Fn = V / np.sqrt(G * L)
        Rn = V * L / NU

        # ---- Frictional resistance ----
        CF = 0.075 / (np.log10(Rn) - 2.0) ** 2
        RF     = 0.5 * RHO * V ** 2 * S * CF
        R_visc = RF * one_k1          # viscous resistance (friction + form)

        # ---- Wave resistance ----
        d_exp = -0.9

        if method == 0:
            # 1982: m2 = c15·CP²·exp(-0.1·Fn⁻²)
            m2  = c15 * CP ** 2 * np.exp(-0.1 * Fn ** (-2.0))
            RW  = (c1 * c2 * c5 * Vol * RHO * G
                   * np.exp(m1 * Fn ** d_exp + m2 * np.cos(lam * Fn ** (-2.0))))
            RW  = max(0.0, RW)

        else:
            # 1984: m4 = c15·0.4·exp(-0.034·Fn⁻³·²⁹)
            m4 = c15 * 0.4 * np.exp(-0.034 * Fn ** (-3.29))

            if Fn <= 0.40:
                RW = (c1 * c2 * c5 * Vol * RHO * G
                      * np.exp(m1 * Fn ** d_exp
                               + m4 * np.cos(lam * Fn ** (-2.0))))
            elif Fn >= 0.55:
                RW = (c17 * c2 * c5 * Vol * RHO * G
                      * np.exp(m3 * Fn ** d_exp
                               + m4 * np.cos(lam * Fn ** (-2.0))))
            else:
                # Interpolate between Fn=0.40 and Fn=0.55
                m4_04 = c15 * 0.4 * np.exp(-0.034 * 0.40 ** (-3.29))
                m4_55 = c15 * 0.4 * np.exp(-0.034 * 0.55 ** (-3.29))
                RW_A04 = (c1 * c2 * c5 * Vol * RHO * G
                          * np.exp(m1 * 0.40 ** d_exp
                                   + m4_04 * np.cos(lam * 0.40 ** (-2.0))))
                RW_B55 = (c17 * c2 * c5 * Vol * RHO * G
                          * np.exp(m3 * 0.55 ** d_exp
                                   + m4_55 * np.cos(lam * 0.55 ** (-2.0))))
                RW = RW_A04 + (10.0 * Fn - 4.0) * (RW_B55 - RW_A04) / 1.5

            RW = max(0.0, RW)

        # ---- Bulbous bow resistance ----
        RB = 0.0
        if ABT > 1e-6:
            PB  = 0.56 * np.sqrt(ABT) / (TF - 1.5 * hB)
            arg = G * (TF - hB - 0.25 * np.sqrt(ABT)) + 0.15 * V ** 2
            if arg > 0:
                Fni = V / np.sqrt(arg)
                RB  = (0.11 * np.exp(-3.0 * PB ** (-2.0))
                       * Fni ** 3 * ABT ** 1.5 * RHO * G / (1.0 + Fni ** 2))

        # ---- Transom stern resistance ----
        RTR = 0.0
        if AT > 1e-6:
            denom_tr = 2.0 * G * AT / (B + B * CWP)
            if denom_tr > 0:
                FnT = V / np.sqrt(denom_tr)
                if FnT < 5.0:
                    RTR = 0.5 * RHO * V ** 2 * AT * 0.2 * (1.0 - 0.2 * FnT)

        # ---- Model-ship correlation resistance ----
        RA = 0.5 * RHO * V ** 2 * S * CA

        Rtot = R_visc + RW + RB + RTR + RA
        PE   = Rtot * V

        results.append(dict(
            speed_kn  = float(V_kn),
            V_ms      = float(V),
            Fn        = float(Fn),
            Rtot_N    = float(Rtot),
            Rtot_kN   = float(Rtot / 1000.0),
            PE_kW     = float(PE / 1000.0),
            R_visc_N  = float(R_visc),
            RF_N      = float(RF),
            RW_N      = float(RW),
            RB_N      = float(RB),
            RTR_N     = float(RTR),
            RA_N      = float(RA),
            CF        = float(CF),
            one_k1    = float(one_k1),
        ))

    return results


# ---------------------------------------------------------------------------
# Speed table builder
# ---------------------------------------------------------------------------

def build_resistance_table(
    L: float, B: float, T: float, Vol: float,
    CB: float, CP: float, CM: float, CWP: float,
    lcb_pct: float, S: float, AT: float, iE: float,
    Cstern: float, method: int,
    design_speed_kn: float,
    max_speed_delta_kn: float,
    steps_per_kn: int,
    ABT: float = 0.0, hB: float = 0.0,
) -> list[dict]:
    """Compute resistance table from 0 to design_speed + delta at given resolution."""
    step  = 1.0 / max(1, int(steps_per_kn))
    v_max = design_speed_kn + max_speed_delta_kn
    speeds = np.arange(0.0, v_max + step / 2.0, step)
    # Ensure design speed endpoint is included
    if abs(speeds[-1] - v_max) > 1e-6:
        speeds = np.append(speeds, v_max)

    return holtrop_mennen(L, B, T, Vol, CB, CP, CM, CWP,
                          lcb_pct, S, AT, iE, Cstern, speeds,
                          method=method, ABT=ABT, hB=hB)
