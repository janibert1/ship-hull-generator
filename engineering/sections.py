# -*- coding: utf-8 -*-
"""
sections.py — Shell cross-sectional properties along the hull.

Loops over all loft stations and returns arrays suitable for the
longitudinal-strength calculation.
"""

from __future__ import annotations

import numpy as np

from .geometry import section_properties


def compute_shell_csa(X_surf: np.ndarray, Y_surf: np.ndarray,
                      Z_surf: np.ndarray,
                      x_min: float = 0.0,
                      x_max: float | None = None) -> dict:
    """Compute shell cross-sectional properties at every loft station.

    Parameters
    ----------
    X_surf, Y_surf, Z_surf : loft arrays (N_t, N_u)
    x_min  : AP position [m]  (filter stations)
    x_max  : FP position [m]  (filter stations)

    Returns
    -------
    dict with 1-D arrays (one entry per filtered station):
      x               : station x-positions [m]
      outline_length  : perimeter (both sides + deck) [m]
      shell_csa_1mm   : cross-section area for 1 mm plate [m²]
      centroid_z      : z of neutral axis [m]
      inertia_y_1mm   : second moment of area for 1 mm plate [m⁴]
      z_keel          : z at keel [m]
      z_deck          : z at deck edge [m]
      y_deck          : y at deck edge [m]
    """
    x_mid = X_surf.mean(axis=1)
    if x_max is None:
        x_max = float(x_mid.max())

    xs             = []
    outline        = []
    csa_1mm        = []
    centroid_z_arr = []
    inertia_1mm    = []
    z_keel_arr     = []
    z_deck_arr     = []
    y_deck_arr     = []

    for t in range(len(x_mid)):
        xh = float(x_mid[t])
        if xh < x_min - 1e-6 or xh > x_max + 1e-6:
            continue
        props = section_properties(Y_surf[t], Z_surf[t])
        xs.append(xh)
        outline.append(props['outline_length'])
        csa_1mm.append(props['shell_csa_1mm'])
        centroid_z_arr.append(props['centroid_z'])
        inertia_1mm.append(props['inertia_y_1mm'])
        z_keel_arr.append(props['z_keel'])
        z_deck_arr.append(props['z_deck'])
        y_deck_arr.append(props['y_deck'])

    return dict(
        x=np.array(xs),
        outline_length=np.array(outline),
        shell_csa_1mm=np.array(csa_1mm),
        centroid_z=np.array(centroid_z_arr),
        inertia_y_1mm=np.array(inertia_1mm),
        z_keel=np.array(z_keel_arr),
        z_deck=np.array(z_deck_arr),
        y_deck=np.array(y_deck_arr),
    )
