# -*- coding: utf-8 -*-
"""
tanks.py — Tank geometry and fill-diagram calculations.

Three tank types:
  Tank 1  — starboard side tank  (negative Y side, inner boundary at y = B_half - t1_w)
  Tank 2  — centre tank          (bounded by y_sb = B_half - t1_w, y_port = B_half - t3_w,
                                   and x in [t2_x0, t2_x1])
  Tank 3  — port side tank       (positive Y side, inner boundary at y = B_half - t3_w)

The functions below compute:
  - Cross-sectional area (CSA) at full fill (h = DOA)
  - Fill diagrams (volume, LCG, TCG, VCG, free-surface Ix) as functions of fill %
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .geometry import hull_hw_at_z


# ---------------------------------------------------------------------------
# Width functions
# ---------------------------------------------------------------------------

def _side_tank_width_at_z(y_cs: np.ndarray, z_cs: np.ndarray,
                           y_inner: float, z: float) -> float:
    """Width of one side-tank strip at height z.

    The side tank extends from y_inner to the hull boundary.
    """
    hw = hull_hw_at_z(y_cs, z_cs, z)
    return max(0.0, hw - y_inner)


def _center_tank_width_at_z(y_cs: np.ndarray, z_cs: np.ndarray,
                              y_sb: float, y_port: float, z: float) -> float:
    """Width of the centre tank at height z (both sides summed).

    Starboard half limited to y_sb, port half limited to y_port.
    """
    hw = hull_hw_at_z(y_cs, z_cs, z)
    return min(hw, y_sb) + min(hw, y_port)


# ---------------------------------------------------------------------------
# CSA at full fill
# ---------------------------------------------------------------------------

def compute_side_tank_csa(X_surf: np.ndarray, Y_surf: np.ndarray,
                           Z_surf: np.ndarray,
                           y_inner: float, h: float,
                           x_min: float = 0.0, x_max: float | None = None,
                           n_z: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """CSA of one side tank at full fill height h, at each loft station.

    Returns (x_arr, csa_arr).
    """
    x_mid = X_surf.mean(axis=1)
    if x_max is None:
        x_max = float(x_mid.max())

    xs, csas = [], []
    for t in range(len(x_mid)):
        xh = float(x_mid[t])
        if xh < x_min - 1e-6 or xh > x_max + 1e-6:
            continue
        z_arr = np.linspace(0.0, h, n_z)
        widths = np.array([_side_tank_width_at_z(Y_surf[t], Z_surf[t], y_inner, z)
                           for z in z_arr])
        area = float(np.trapezoid(widths, z_arr))
        xs.append(xh)
        csas.append(area)

    return np.array(xs), np.array(csas)


def compute_center_tank_csa(X_surf: np.ndarray, Y_surf: np.ndarray,
                              Z_surf: np.ndarray,
                              y_sb: float, y_port: float, h: float,
                              x_min: float = 0.0, x_max: float | None = None,
                              n_z: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """CSA of the centre tank at full fill height h, at each loft station.

    Returns (x_arr, csa_arr).
    """
    x_mid = X_surf.mean(axis=1)
    if x_max is None:
        x_max = float(x_mid.max())

    xs, csas = [], []
    for t in range(len(x_mid)):
        xh = float(x_mid[t])
        if xh < x_min - 1e-6 or xh > x_max + 1e-6:
            continue
        z_arr = np.linspace(0.0, h, n_z)
        widths = np.array([_center_tank_width_at_z(Y_surf[t], Z_surf[t], y_sb, y_port, z)
                           for z in z_arr])
        area = float(np.trapezoid(widths, z_arr))
        xs.append(xh)
        csas.append(area)

    return np.array(xs), np.array(csas)


# ---------------------------------------------------------------------------
# Fill diagrams
# ---------------------------------------------------------------------------

def _side_tank_fill_at_h(X_surf: np.ndarray, Y_surf: np.ndarray,
                          Z_surf: np.ndarray,
                          y_inner: float, h: float,
                          x_min: float, x_max: float,
                          n_z: int = 40) -> dict:
    """Volume, LCG, VCG of a side tank at fill height h."""
    x_mid = X_surf.mean(axis=1)
    mask  = (x_mid >= x_min - 1e-6) & (x_mid <= x_max + 1e-6)
    ts    = np.where(mask)[0]

    xs, csas, vcg_moms = [], [], []
    for t in ts:
        xh    = float(x_mid[t])
        z_arr = np.linspace(0.0, h, n_z)
        widths = np.array([_side_tank_width_at_z(Y_surf[t], Z_surf[t], y_inner, z)
                           for z in z_arr])
        area    = float(np.trapezoid(widths, z_arr))
        vcg_mom = float(np.trapezoid(z_arr * widths, z_arr))
        xs.append(xh)
        csas.append(area)
        vcg_moms.append(vcg_mom)

    xs       = np.array(xs)
    csas     = np.array(csas)
    vcg_moms = np.array(vcg_moms)

    if len(xs) < 2 or np.sum(csas) < 1e-12:
        return dict(volume=0.0, lcg=0.5 * (x_min + x_max), vcg=0.5 * h)

    volume = float(np.trapezoid(csas, xs))
    lcg    = float(np.trapezoid(xs * csas, xs)) / volume if volume > 0 else 0.5 * (x_min + x_max)
    vcg    = float(np.trapezoid(vcg_moms, xs))  / volume if volume > 0 else 0.5 * h
    return dict(volume=volume, lcg=lcg, vcg=vcg)


def _center_tank_fill_at_h(X_surf: np.ndarray, Y_surf: np.ndarray,
                            Z_surf: np.ndarray,
                            y_sb: float, y_port: float, h: float,
                            x_min: float, x_max: float,
                            n_z: int = 40) -> dict:
    """Volume, LCG, VCG of the centre tank at fill height h."""
    x_mid = X_surf.mean(axis=1)
    mask  = (x_mid >= x_min - 1e-6) & (x_mid <= x_max + 1e-6)
    ts    = np.where(mask)[0]

    xs, csas, vcg_moms = [], [], []
    for t in ts:
        xh    = float(x_mid[t])
        z_arr = np.linspace(0.0, h, n_z)
        widths = np.array([_center_tank_width_at_z(Y_surf[t], Z_surf[t], y_sb, y_port, z)
                           for z in z_arr])
        area    = float(np.trapezoid(widths, z_arr))
        vcg_mom = float(np.trapezoid(z_arr * widths, z_arr))
        xs.append(xh)
        csas.append(area)
        vcg_moms.append(vcg_mom)

    xs       = np.array(xs)
    csas     = np.array(csas)
    vcg_moms = np.array(vcg_moms)

    if len(xs) < 2 or np.sum(csas) < 1e-12:
        return dict(volume=0.0, lcg=0.5 * (x_min + x_max), vcg=0.5 * h)

    volume = float(np.trapezoid(csas, xs))
    lcg    = float(np.trapezoid(xs * csas, xs)) / volume if volume > 0 else 0.5 * (x_min + x_max)
    vcg    = float(np.trapezoid(vcg_moms, xs))  / volume if volume > 0 else 0.5 * h
    return dict(volume=volume, lcg=lcg, vcg=vcg)


def build_tank_fill_diagram(
    X_surf: np.ndarray, Y_surf: np.ndarray, Z_surf: np.ndarray,
    tank_type: str,          # 'side_sb', 'side_port', or 'center'
    h_max: float,            # DOA or tank height limit [m]
    x_min: float,
    x_max: float,
    y_inner_sb: float = 0.0,  # inner boundary of SB side tank (also used for centre)
    y_inner_port: float = 0.0,
    fill_steps: int = 101,   # 0 % to 100 %
    n_z: int = 40,
) -> dict:
    """Build 0–100 % fill diagram for one tank.

    Returns
    -------
    dict with arrays (length = fill_steps):
      fill_pct    : fill percentage [%]
      fill_m      : fill height [m]
      volume      : tank volume [m³]
      lcg         : LCG from AP [m]
      tcg         : TCG from centreline [m]  (neg. for SB, pos. for port)
      vcg         : VCG from keel [m]
      ix_fs       : free-surface second moment (rectangular approx) [m⁴]
    """
    pcts  = np.linspace(0.0, 100.0, fill_steps)
    vols, lcgs, tcgs, vcgs, ixs = [], [], [], [], []

    # Approximate tank width for TCG and free-surface Ix
    if tank_type in ('side_sb', 'side_port'):
        t_width = h_max  # DOA as representative height — actual width = B_half - y_inner
        # Better approximation: use midship section
        x_mid_all = X_surf.mean(axis=1)
        t_mid_idx  = np.argmin(np.abs(x_mid_all - 0.5 * (x_min + x_max)))
        hw_at_doa  = hull_hw_at_z(Y_surf[t_mid_idx], Z_surf[t_mid_idx], h_max)
        t_width    = hw_at_doa - y_inner_sb if tank_type == 'side_sb' else hw_at_doa - y_inner_port
        t_width    = max(t_width, 0.01)
        tank_length = x_max - x_min
        # TCG: approximately at (y_inner + t_width/2) from CL, negative for SB
        tcg_sign = -1.0 if tank_type == 'side_sb' else 1.0
        y_inner   = y_inner_sb if tank_type == 'side_sb' else y_inner_port
        tcg_approx = tcg_sign * (y_inner + 0.5 * t_width)
    else:  # centre
        tank_length = x_max - x_min
        t_width     = y_inner_sb + y_inner_port  # full width of centre tank ≈ 2×half_width
        tcg_approx  = 0.0

    # Free-surface Ix (rectangular cross-section, constant over fill)
    # Ix about centreline = L × w³/12  for side tanks
    # For centre tank Ix = L × (2 * half_width)³ / 12
    ix_fs_const = tank_length * t_width**3 / 12.0

    for pct in pcts:
        h = h_max * pct / 100.0
        if h < 1e-9:
            vols.append(0.0)
            lcgs.append(0.5 * (x_min + x_max))
            tcgs.append(tcg_approx)
            vcgs.append(0.0)
            ixs.append(0.0)
            continue

        if tank_type == 'side_sb':
            res = _side_tank_fill_at_h(X_surf, Y_surf, Z_surf,
                                       y_inner_sb, h, x_min, x_max, n_z)
        elif tank_type == 'side_port':
            res = _side_tank_fill_at_h(X_surf, Y_surf, Z_surf,
                                       y_inner_port, h, x_min, x_max, n_z)
        else:
            res = _center_tank_fill_at_h(X_surf, Y_surf, Z_surf,
                                         y_inner_sb, y_inner_port,
                                         h, x_min, x_max, n_z)

        vols.append(res['volume'])
        lcgs.append(res['lcg'])
        tcgs.append(tcg_approx)
        vcgs.append(res['vcg'])
        ixs.append(ix_fs_const if pct > 0.0 else 0.0)

    return dict(
        fill_pct=pcts,
        fill_m=h_max * pcts / 100.0,
        volume=np.array(vols),
        lcg=np.array(lcgs),
        tcg=np.array(tcgs),
        vcg=np.array(vcgs),
        ix_fs=np.array(ixs),
    )


# ---------------------------------------------------------------------------
# Free-surface second moment (for GM correction)
# ---------------------------------------------------------------------------

def free_surface_ix_side_tank(X_surf: np.ndarray, Y_surf: np.ndarray,
                               Z_surf: np.ndarray,
                               y_inner: float, fill_height: float,
                               x_min: float, x_max: float) -> float:
    """Actual (hull-integrated) free-surface second moment of area for a side tank.

    Ix_fs = integral over length of  w(x)^3/12  where w(x) = hull_hw(x,h) - y_inner
    """
    x_mid = X_surf.mean(axis=1)
    mask = (x_mid >= x_min - 1e-6) & (x_mid <= x_max + 1e-6)
    ts = np.where(mask)[0]
    if len(ts) < 2:
        return 0.0
    xs_arr = [float(x_mid[t]) for t in ts]
    ix_arr = []
    for t in ts:
        hw = hull_hw_at_z(Y_surf[t], Z_surf[t], fill_height)
        w = max(0.0, hw - y_inner)
        ix_arr.append(w**3 / 12.0)
    return float(np.trapezoid(ix_arr, xs_arr))


def free_surface_ix_center_tank(X_surf: np.ndarray, Y_surf: np.ndarray,
                                  Z_surf: np.ndarray,
                                  y_sb: float, y_port: float, fill_height: float,
                                  x_min: float, x_max: float) -> float:
    """Actual free-surface Ix for the center tank.

    At each station: effective width = min(hw, y_sb) + min(hw, y_port)
    Ix_fs = integral of (total_width^3)/12
    """
    x_mid = X_surf.mean(axis=1)
    mask = (x_mid >= x_min - 1e-6) & (x_mid <= x_max + 1e-6)
    ts = np.where(mask)[0]
    if len(ts) < 2:
        return 0.0
    xs_arr = [float(x_mid[t]) for t in ts]
    ix_arr = []
    for t in ts:
        hw = hull_hw_at_z(Y_surf[t], Z_surf[t], fill_height)
        b = min(hw, y_sb) + min(hw, y_port)
        ix_arr.append(b**3 / 12.0)
    return float(np.trapezoid(ix_arr, xs_arr))


def side_tank_free_surface_ix(tank_length: float, y_inner: float,
                               B_half: float) -> float:
    """Approximate free-surface Ix for a side tank (rectangular approximation).

    Ix = L × w³ / 12  where w = B_half - y_inner
    """
    w = B_half - y_inner
    return tank_length * w**3 / 12.0


def center_tank_free_surface_ix(tank_length: float,
                                 y_sb: float, y_port: float) -> float:
    """Approximate free-surface Ix for the centre tank.

    Full width = y_sb + y_port (both halves),  Ix = L × (y_sb+y_port)³ / 12
    """
    w = y_sb + y_port
    return tank_length * w**3 / 12.0
