from __future__ import annotations

import numpy as np

from .constants import TP_GAP_M, TP_MARGIN_M, TP_RADIUS_M, TP_WEIGHT_T


def pack_transition_pieces_hex(X: np.ndarray, Y: np.ndarray, LPP: float) -> tuple[list[dict], float]:
    x_line = X[:, 0]
    y_deck_half = Y[:, -1]

    x0 = max(5.0, LPP * 0.05)
    x1 = min(LPP - 5.0, LPP * 0.95)
    if x1 <= x0 + 1e-6:
        return [], TP_RADIUS_M + TP_MARGIN_M

    # Hex-like circle packing with effective radius (gap shared between neighbors).
    r_eff = TP_RADIUS_M + 0.5 * TP_GAP_M
    dx = np.sqrt(3.0) * r_eff
    dy = 2.0 * r_eff
    x_rows = np.arange(x0, x1 + 1e-9, dx)
    if x_rows.size == 0:
        x_rows = np.array([x0], dtype=float)

    pieces: list[dict] = []
    max_half_width = 0.0
    min_center_dist = 2.0 * TP_RADIUS_M + TP_GAP_M
    min_center_dist2 = min_center_dist * min_center_dist
    for row_idx, x_pos in enumerate(x_rows):
        hw = float(np.interp(x_pos, x_line, y_deck_half))
        max_half_width = max(max_half_width, hw)
        half_span = hw - TP_MARGIN_M - TP_RADIUS_M
        if half_span < 0.0:
            continue

        y_offset = 0.0 if (row_idx % 2 == 0) else dy * 0.5
        y_abs_vals = np.arange(y_offset, half_span + 1e-9, dy)
        if y_abs_vals.size == 0:
            y_abs_vals = np.array([0.0], dtype=float)

        for y_abs in y_abs_vals:
            candidates = [0.0] if abs(y_abs) <= 1e-9 else [float(y_abs), float(-y_abs)]
            if any(abs(y_c) > half_span + 1e-9 for y_c in candidates):
                continue
            overlap = False
            for y_c in candidates:
                x_new = float(x_pos)
                y_new = float(y_c)
                for p in pieces:
                    dxp = p["x"] - x_new
                    dyp = p["y"] - y_new
                    if dxp * dxp + dyp * dyp < (min_center_dist2 - 1e-9):
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                continue
            for y_c in candidates:
                pieces.append({"x": float(x_pos), "y": float(y_c), "weight_t": TP_WEIGHT_T})

    deck_width_violation = (TP_RADIUS_M + TP_MARGIN_M) - max_half_width
    return pieces, deck_width_violation
