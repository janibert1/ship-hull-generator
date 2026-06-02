# -*- coding: utf-8 -*-
"""
geometry.py — Low-level geometric helpers for parametric hull cross-sections.

All functions operate on a single cross-section (y_cs, z_cs) representing the
port-side half-contour from keel (index 0, y~0, z~0) to deck edge (index -1,
y=B_half, z=D).  All dimensions in metres.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Half-width at a given height
# ---------------------------------------------------------------------------

def hull_hw_at_z(y_cs: np.ndarray, z_cs: np.ndarray, z: float) -> float:
    """Return the hull half-width at height z on one cross-section.

    Parameters
    ----------
    y_cs : 1-D array of y-coordinates (port half, ascending from keel)
    z_cs : 1-D array of z-coordinates (ascending)
    z    : target height [m]

    Returns
    -------
    half-width [m] — 0.0 if z is above the deck or the hull is not yet wide
    """
    mask = z_cs <= z + 1e-9
    if not np.any(mask):
        return 0.0
    return float(np.max(y_cs[mask]))


# ---------------------------------------------------------------------------
# Submerged area (one cross-section)
# ---------------------------------------------------------------------------

def submerged_area_half(y_cs: np.ndarray, z_cs: np.ndarray, T: float,
                        n_z: int = 80) -> float:
    """Return the full (both sides) submerged cross-sectional area at draught T.

    Uses trapezoid integration of 2 * hull_hw_at_z(z) from z=0 to z=T.
    """
    z_arr = np.linspace(0.0, T, n_z)
    widths = np.array([2.0 * hull_hw_at_z(y_cs, z_cs, z) for z in z_arr])
    return float(np.trapezoid(widths, z_arr))


def analyze_stern_shape(
    y_cs: np.ndarray,
    z_cs: np.ndarray,
    beam: float,
    draft: float,
    mapping_factor: float = 50.0,
) -> dict:
    """Estimate Holtrop stern-shape coefficient from section fullness.

    Returns
    -------
    dict with:
      c_beta  : sectional area coefficient at the sampled stern section
      cstern  : mapped/clamped Holtrop stern coefficient in [-25, +10]
    """
    beam_eff = max(float(beam), 1e-6)
    draft_eff = max(float(draft), 1e-6)
    t_lim = min(draft_eff, float(np.max(z_cs)))
    if t_lim <= 1e-9:
        return {"c_beta": 0.7, "cstern": -25.0}

    area = submerged_area_half(y_cs, z_cs, T=t_lim, n_z=80)
    c_beta = area / (beam_eff * draft_eff)
    cstern = (c_beta - 0.7) * float(mapping_factor)
    cstern = float(np.clip(cstern, -25.0, 10.0))
    return {"c_beta": float(c_beta), "cstern": cstern}


# ---------------------------------------------------------------------------
# Section properties (shell CSA, centroid, second moment of area)
# ---------------------------------------------------------------------------

def section_properties(y_cs: np.ndarray, z_cs: np.ndarray) -> dict:
    """Compute cross-sectional properties of the shell at one station.

    The 'shell' comprises:
      - Hull plating on both sides  (arc from keel to deck edge)
      - A flat deck plate           (y from 0 to y_deck at z=z_deck)

    All quantities are per unit thickness (1 mm = 0.001 m convention).

    Returns
    -------
    dict with keys:
      outline_length   : total arc length of hull perimeter (both sides) [m]
      shell_csa_1mm    : cross-sectional area of shell for 1 mm thickness [m²]
      centroid_z       : z-coordinate of neutral axis [m]
      inertia_y_1mm    : second moment of area about NA for 1 mm thickness [m⁴]
      z_keel           : z at keel (always ~0) [m]
      z_deck           : z at deck edge [m]
      y_deck           : y at deck edge (= B_half) [m]
    """
    # Arc-length elements along the hull half-contour
    dy = np.diff(y_cs)
    dz = np.diff(z_cs)
    ds = np.hypot(dy, dz)                  # segment lengths
    z_mid = 0.5 * (z_cs[:-1] + z_cs[1:])  # midpoint heights

    # Hull contribution (both sides)
    A_hull_both = 2.0 * float(np.sum(ds))   # [m] arc length both sides (× t gives m²)
    S_z_hull    = 2.0 * float(np.sum(z_mid * ds))  # first moment [m²]
    I_0_hull    = 2.0 * float(np.sum(z_mid**2 * ds))  # second moment [m³]

    # Deck plate contribution
    y_deck = float(y_cs[-1])
    z_deck = float(z_cs[-1])
    z_keel = float(z_cs[0])
    A_deck  = 2.0 * y_deck          # [m] (× t gives m²)
    S_z_deck = z_deck * A_deck      # [m²]
    I_0_deck = z_deck**2 * A_deck   # [m³]

    # Totals
    A_total   = A_hull_both + A_deck       # [m]  (per-unit-thickness area denominator)
    S_z_total = S_z_hull + S_z_deck        # [m²]
    I_0_total = I_0_hull + I_0_deck        # [m³]

    z_NA = S_z_total / A_total if A_total > 1e-12 else 0.0
    I_NA_per_unit = I_0_total - A_total * z_NA**2   # [m³]  (× t → [m⁴])

    # 1 mm convention
    t_ref = 0.001   # 1 mm in metres
    shell_csa_1mm = A_total * t_ref   # [m²]
    inertia_y_1mm = I_NA_per_unit * t_ref  # [m⁴]
    outline_length = A_hull_both + A_deck  # total outline perimeter [m]

    return dict(
        outline_length=outline_length,
        shell_csa_1mm=shell_csa_1mm,
        centroid_z=z_NA,
        inertia_y_1mm=inertia_y_1mm,
        z_keel=z_keel,
        z_deck=z_deck,
        y_deck=y_deck,
    )
