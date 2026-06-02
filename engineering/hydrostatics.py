# -*- coding: utf-8 -*-
"""
hydrostatics.py — Buoyancy, draft-finding, and waterplane calculations.

All functions work directly on the loft arrays (X_surf, Y_surf, Z_surf)
returned by build_hull_loft().
"""

from __future__ import annotations

import numpy as np

from .geometry import hull_hw_at_z, submerged_area_half

RHO_WATER = 1025.0   # kg/m³
G         = 9.81     # m/s²


# ---------------------------------------------------------------------------
# Buoyant cross-sectional area along the ship
# ---------------------------------------------------------------------------

def compute_buoyant_csa(X_surf: np.ndarray, Y_surf: np.ndarray,
                        Z_surf: np.ndarray, T: float,
                        x_min: float = 0.0, x_max: float | None = None,
                        n_z: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Compute the submerged cross-sectional area at each loft station.

    Parameters
    ----------
    X_surf, Y_surf, Z_surf : loft arrays (N_t, N_u)
    T      : draught [m]
    x_min  : start of integration domain (AP = 0) [m]
    x_max  : end of integration domain (FP = LPP) [m]
    n_z    : integration points in z direction

    Returns
    -------
    x_arr  : x-positions of stations [m]
    csa    : submerged CSA at each station [m²]
    """
    x_mid = X_surf.mean(axis=1)
    if x_max is None:
        x_max = float(x_mid.max())

    xs, csas = [], []
    for t in range(len(x_mid)):
        xh = float(x_mid[t])
        if xh < x_min - 1e-6 or xh > x_max + 1e-6:
            continue
        area = submerged_area_half(Y_surf[t], Z_surf[t], T, n_z=n_z)
        xs.append(xh)
        csas.append(area)

    return np.array(xs), np.array(csas)


# ---------------------------------------------------------------------------
# Draft finding
# ---------------------------------------------------------------------------

def _displacement_at_draft(X_surf: np.ndarray, Y_surf: np.ndarray,
                            Z_surf: np.ndarray, T: float,
                            x_min: float, x_max: float, n_z: int = 60) -> float:
    """Integrate submerged volume over the full ship length."""
    x_arr, csa = compute_buoyant_csa(X_surf, Y_surf, Z_surf, T,
                                     x_min=x_min, x_max=x_max, n_z=n_z)
    if len(x_arr) < 2:
        return 0.0
    return float(np.trapezoid(csa, x_arr))


def find_draft(X_surf: np.ndarray, Y_surf: np.ndarray, Z_surf: np.ndarray,
               mass_kg: float, x_min: float = 0.0, x_max: float | None = None,
               D: float | None = None, tol: float = 1e-4,
               n_iter: int = 60) -> float:
    """Find the draught T such that displaced volume × RHO_WATER = mass_kg.

    Uses bisection between T=0 and T=D (full depth).
    """
    x_mid = X_surf.mean(axis=1)
    if x_max is None:
        x_max = float(x_mid.max())
    if D is None:
        D = float(Z_surf.max())

    target_vol = mass_kg / RHO_WATER

    T_lo, T_hi = 0.001, D * 0.999
    for _ in range(n_iter):
        T_mid = 0.5 * (T_lo + T_hi)
        vol = _displacement_at_draft(X_surf, Y_surf, Z_surf, T_mid,
                                     x_min, x_max, n_z=60)
        if vol < target_vol:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) < tol:
            break

    return 0.5 * (T_lo + T_hi)


# ---------------------------------------------------------------------------
# Centre of buoyancy
# ---------------------------------------------------------------------------

def center_of_buoyancy(X_surf: np.ndarray, Y_surf: np.ndarray,
                       Z_surf: np.ndarray, T: float,
                       x_min: float = 0.0, x_max: float | None = None,
                       n_z: int = 60) -> tuple[float, float, float]:
    """Return (LCB, TCB, VCB) — the centre of buoyancy at draught T.

    TCB is 0 by symmetry.  VCB computed as integral of z * dV / V.
    """
    x_mid = X_surf.mean(axis=1)
    if x_max is None:
        x_max = float(x_mid.max())

    mask = (x_mid >= x_min - 1e-6) & (x_mid <= x_max + 1e-6)
    ts   = np.where(mask)[0]

    xs, csas, z_mom = [], [], []
    for t in ts:
        xh = float(x_mid[t])
        y_cs = Y_surf[t]
        z_cs = Z_surf[t]
        # submerged CSA
        z_arr = np.linspace(0.0, T, n_z)
        widths = np.array([2.0 * hull_hw_at_z(y_cs, z_cs, z) for z in z_arr])
        area = float(np.trapezoid(widths, z_arr))
        # first moment of area about z=0
        z_mom_val = float(np.trapezoid(z_arr * widths, z_arr))
        xs.append(xh)
        csas.append(area)
        z_mom.append(z_mom_val)

    xs     = np.array(xs)
    csas   = np.array(csas)
    z_moms = np.array(z_mom)

    if len(xs) < 2:
        return (0.5 * (x_min + x_max), 0.0, 0.5 * T)

    V    = float(np.trapezoid(csas,   xs))
    lcb  = float(np.trapezoid(xs * csas, xs)) / V if V > 0 else 0.0
    vcb  = float(np.trapezoid(z_moms, xs))    / V if V > 0 else 0.0
    return lcb, 0.0, vcb


# ---------------------------------------------------------------------------
# Waterplane data  (second moment of area for BM calculation)
# ---------------------------------------------------------------------------

def waterplane_data(X_surf: np.ndarray, Y_surf: np.ndarray,
                    Z_surf: np.ndarray, T: float,
                    x_min: float = 0.0, x_max: float | None = None) -> dict:
    """Compute waterplane area and transverse second moment of area (Ix) about CL.

    Returns
    -------
    dict with:
      Aw    : waterplane area [m²]
      Ix    : second moment of waterplane area about ship centreline [m⁴]
      COF_x : centre of flotation (x) [m]
    """
    x_mid = X_surf.mean(axis=1)
    if x_max is None:
        x_max = float(x_mid.max())

    mask = (x_mid >= x_min - 1e-6) & (x_mid <= x_max + 1e-6)
    ts   = np.where(mask)[0]

    xs, bwl, bwl3 = [], [], []
    for t in ts:
        xh    = float(x_mid[t])
        half_b = hull_hw_at_z(Y_surf[t], Z_surf[t], T)
        xs.append(xh)
        bwl.append(2.0 * half_b)
        bwl3.append((2.0 / 3.0) * half_b**3)  # integral of y² dy from -b to +b = 2/3 * b³

    xs    = np.array(xs)
    bwl   = np.array(bwl)
    bwl3  = np.array(bwl3)

    if len(xs) < 2:
        return dict(Aw=0.0, Ix=0.0, COF_x=0.5 * (x_min + x_max))

    Aw    = float(np.trapezoid(bwl,  xs))
    Ix    = float(np.trapezoid(bwl3, xs))
    cof_x = float(np.trapezoid(xs * bwl, xs)) / Aw if Aw > 0 else 0.0
    return dict(Aw=Aw, Ix=Ix, COF_x=cof_x)
