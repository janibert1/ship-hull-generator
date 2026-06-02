import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import sys
import argparse
from pathlib import Path
import config as cfg
from bezier_math import get_rhino_style_spline

# Global config filename, can be overridden by CLI
CONFIG_FILE = "config.json"
OPTIM_RESULTS_FILE = "optim_results.json"
CRANE_MODE = "stowaway"  # "loading" or "stowaway"

def reload_config_module(config_file):
    """Update the 'cfg' module attributes from a specific JSON file."""
    pad = Path(__file__).parent / config_file
    if not pad.exists():
        print(f"Error: {config_file} not found.")
        sys.exit(1)
        
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cfg.LOA = data["Length_Loa_m"]
    cfg.BOA = data["Breadth_Boa_m"]
    cfg.DOA = data["Depth_Doa_m"]
    cfg.LPP_RATIO = data["Lpp_Loa_ratio"]
    cfg.MIDSHIP_LENGTH_PCT = data["MidshipLength_pct_Lpp"]
    cfg.MIDSHIP_LOC_PCT = data["Location_midship_pct_Lpp"]
    cfg.BILGE_RADIUS = data["Bilge_Radius_m"]
    cfg.AFT_SHLD_PCT = data.get("Aft_Shoulder_pct", 50.0)
    cfg.FWD_SHLD_PCT = data.get("Fwd_Shoulder_pct", 50.0)
    cfg.BOW_INT_PCT = data.get("Location_bow_intermediate_curve_pct", 50.0)
    cfg.BOW_ROUNDING_DEG = data.get("Bow_Rounding_deg", 50.0)
    cfg.SIDE_FLARE_DEG = data.get("Side_Flare_deg", 0.0)
    cfg.SIDE_FLARE_ROTATION_POINT = data.get("Side_Flare_Rotation_Point", 0.0)
    cfg.PARALLEL_MIDSHIP_COMB = int(data.get("Parallel_Midship_Combinations", 2))
    cfg.HULL_THICKNESS_MM = data.get("Hull_Thickness_mm", 8.0)
    cfg.TARGET_DRAFT = data.get("Target_Draft_m", 2.0)
    cfg.TANK1_WIDTH = data.get("Tank1_Width_m", 3.0)
    cfg.TANK1_FILL = data.get("Tank1_Fill_pct", 50.0)
    cfg.TANK2_LEN_PCT = data.get("Tank2_Length_pct_Loa", 30.0)
    cfg.TANK2_CENTER = data.get("Tank2_Center_from_AP_m", 20.0)
    cfg.TANK2_FILL = data.get("Tank2_Fill_pct", 75.0)
    cfg.TANK3_WIDTH = data.get("Tank3_Width_m", 3.0)
    cfg.TANK3_FILL = data.get("Tank3_Fill_pct", 50.0)
    cfg.LPP = cfg.LOA * cfg.LPP_RATIO


def _strip_runtime_cfg_keys(d):
    if not isinstance(d, dict):
        return {}
    return {
        k: v
        for k, v in d.items()
        if not (isinstance(k, str) and (k.startswith("Optimizer_") or k.startswith("_optimizer_")))
    }


def _cfg_value_equal(a, b, tol=1e-8):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        ka = sorted(a.keys())
        kb = sorted(b.keys())
        if ka != kb:
            return False
        return all(_cfg_value_equal(a[k], b[k], tol=tol) for k in ka)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_cfg_value_equal(x, y, tol=tol) for x, y in zip(a, b))
    if isinstance(a, (int, float, np.floating)) and isinstance(b, (int, float, np.floating)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _iter_optim_candidates(data):
    if not isinstance(data, dict):
        return
    selected = data.get("selected_best")
    if isinstance(selected, dict):
        yield "selected_best", selected
    extremes = data.get("extremes")
    if isinstance(extremes, dict):
        for key in ("min_resistance", "max_payload", "min_empty_ship_weight"):
            c = extremes.get(key)
            if isinstance(c, dict):
                yield f"extremes.{key}", c
    pareto = data.get("pareto_top")
    if isinstance(pareto, list):
        for i, c in enumerate(pareto):
            if isinstance(c, dict):
                yield f"pareto_top[{i}]", c


def _lookup_optimizer_metrics_for_cfg(cfg_data):
    results_path = Path(__file__).parent / OPTIM_RESULTS_FILE
    if not results_path.exists():
        return None, None
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    cfg_clean = _strip_runtime_cfg_keys(cfg_data)
    base_name = Path(CONFIG_FILE).name.lower()
    if base_name == "config_best.json":
        selected = data.get("selected_best")
        if isinstance(selected, dict) and isinstance(selected.get("metrics"), dict):
            return selected.get("metrics"), "optim_results:selected_best"

    for label, cand in _iter_optim_candidates(data):
        cand_cfg = cand.get("cfg")
        cand_metrics = cand.get("metrics")
        if not isinstance(cand_cfg, dict) or not isinstance(cand_metrics, dict):
            continue
        if _cfg_value_equal(_strip_runtime_cfg_keys(cand_cfg), cfg_clean):
            return cand_metrics, f"optim_results:{label}"

    return None, None


def _run_engineering_for_live_plot(data: dict):
    """Run the full engineering pipeline in the plot subprocess using a config dict.
    Returns the run_all result dict, or None on any failure."""
    try:
        import sys as _sys
        _parent = str(Path(__file__).parent)
        if _parent not in _sys.path:
            _sys.path.insert(0, _parent)
        from engineering.run import run_all
        return run_all(verbose=False, cfg_dict_override=data, save_outputs=False)
    except Exception:
        return None


def _render_full_engineering_text(eng: dict, geo: dict, cfg_data: dict, source_label: str) -> str:
    """Format an info.txt-style panel from the engineering run_all result."""
    W = 56

    def fv(key, nd=2, sfx="", d=eng):
        v = d.get(key)
        return f"{float(v):.{nd}f}{sfx}" if isinstance(v, (int, float, np.floating)) else "—"

    loa = float(cfg_data.get("Length_Loa_m", 0))
    lpp = float(geo.get("LPP", 0))
    boa = float(geo.get("B_half", 0)) * 2.0
    doa = float(geo.get("D", 0))
    t_mm = float(cfg_data.get("Hull_Thickness_mm", 0))

    disp_t = eng.get("displacement_kg", 0) / 1000.0
    draft  = eng.get("draft_m", eng.get("T", 0))
    gm     = eng.get("gm", 0)
    t_roll = 0.7 * boa / max(float(gm) ** 0.5, 1e-6) if isinstance(gm, (int, float, np.floating)) else float("nan")

    hm     = eng.get("hm_coeffs", {})
    speed  = float(eng.get("hm_design_speed_kn", 14.0))
    res_table = eng.get("resistance_table", [])
    des    = next((r for r in res_table if abs(r.get("speed_kn", 0) - speed) < 0.01), None)

    def fh(key, nd=3): return fv(key, nd, d=hm)

    lines = [
        "=" * W,
        "  ENGINEERING (live optimizer)",
        f"  {source_label}",
        "=" * W,
        f"  LOA: {loa:.2f} m   LPP: {lpp:.2f} m   dikte: {t_mm:.1f} mm",
        f"  BOA: {boa:.2f} m   DOA: {doa:.2f} m",
        "-" * W,
        "  GEWICHT & STABILITEIT",
        "-" * W,
        f"  Displacement : {disp_t:.1f} t      Draft: {float(draft):.3f} m",
        f"  LCG : {fv('lcg_total')} m   VCG: {fv('vcg_total')} m   TCG: {fv('tcg_total', 3)} m",
        f"  LCB : {fv('lcb')} m   VCB: {fv('vcb')} m",
        f"  GM  : {fv('gm', 3)} m      T_roll: {t_roll:.2f} s",
        "-" * W,
        "  TANKVULLINGEN",
        "-" * W,
        f"  Tank 1 (SB)  : {fv('t1_fill_pct_equil', 1)} %",
        f"  Tank 2 (mid) : {fv('t2_fill_pct_equil', 1)} %   LCG {fv('lcg2', 2)} m",
        f"  Tank 3 (BB)  : {fv('t3_fill_pct_equil', 1)} %",
        "-" * W,
        "  STERKTE",
        "-" * W,
        f"  Sigma bodem : {fv('max_sigma_bodem', 1)} MPa   Sigma dek: {fv('max_sigma_dek', 1)} MPa   (max 190)",
        f"  Max moment  : {eng.get('max_moment_nm', 0)/1e6:.2f} MNm  @ x={fv('locatie_max_moment', 1)} m",
        f"  Doorbuiging : {fv('max_doorbuiging_mm', 1)} mm  @ x={fv('locatie_max_doorbuiging', 1)} m",
        f"  Krachtrest. : {fv('krachtrestant_kn', 3)} kN   Momentrest.: {fv('momentrestant_mnm', 4)} MNm",
    ]

    if des:
        lines += [
            "-" * W,
            f"  WEERSTAND @ {speed:.1f} kn  (Fn={des.get('Fn', 0):.3f})",
            "-" * W,
            f"  Rtot        : {des.get('Rtot_kN', 0):.1f} kN   PE: {des.get('PE_kW', 0):.0f} kW",
            f"  RF          : {des.get('RF_N', 0)/1e3:.1f} kN",
            f"  R_visc(1+k1): {des.get('R_visc_N', 0)/1e3:.1f} kN  (k1={des.get('one_k1', 0):.3f})",
            f"  R_wave      : {des.get('RW_N', 0)/1e3:.1f} kN   R_transom: {des.get('RTR_N', 0)/1e3:.1f} kN",
            f"  R_corr (CA) : {des.get('RA_N', 0)/1e3:.1f} kN",
            f"  CB: {fh('CB')}  CP: {fh('CP')}  CM: {fh('CM')}  CWP: {fh('CWP')}",
            f"  iE: {eng.get('hm_iE', 0):.1f}°  S_wet: {eng.get('hm_S_wet', 0):.1f} m²  AT: {eng.get('hm_AT', 0):.2f} m²",
        ]
        warnings = eng.get("hm_warnings", [])
        for w in warnings:
            lines.append(f"  ! {w}")

    lines.append("=" * W)
    return "\n".join(lines)


def _load_engineering_strength():
    """Load strength data from engineering/output/antwoordenblad.json (our simple format)."""
    ab_path = Path(__file__).parent / "engineering" / "output" / "antwoordenblad.json"
    if not ab_path.exists():
        return None
    try:
        data = json.loads(ab_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "Max_sigma_bodem_MPa" not in data:
        return None
    return data


def _render_optimizer_metrics_text(metrics, source_label, eng_data=None):
    def fnum(key, nd=2, suffix=""):
        v = metrics.get(key, None)
        if not isinstance(v, (int, float, np.floating)):
            return "—"
        return f"{float(v):.{nd}f}{suffix}"

    def fint(key):
        v = metrics.get(key, None)
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            return f"{int(round(float(v)))}"
        return "—"

    speed_kn = metrics.get("Design_Speed_kn", 14.0)
    if not isinstance(speed_kn, (int, float, np.floating)):
        speed_kn = 14.0

    lines = [
        "=" * 52,
        "  OPTIMIZER METRICS (3D VIEW)",
        "=" * 52,
        f"  Bron: {source_label}",
        "-" * 52,
        f"  Rtot @ {float(speed_kn):.1f} kn : {fnum('Rtot_kN', 1, ' kN')}",
        f"  N_TPs                  : {fint('N_TPs')}",
        f"  Payload                : {fnum('Payload_t', 1, ' t')}",
        f"  Empty Ship Weight      : {fnum('ShipWeight_t', 1, ' t')}",
        f"  LSW (incl ballast)     : {fnum('LSW_t', 1, ' t')}",
        f"  GM                     : {fnum('GM_m', 3, ' m')}",
        f"  Vrijboord              : {fnum('Freeboard_m', 3, ' m')}",
        f"  Draft                  : {fnum('Draft_m', 3, ' m')}",
        f"  T_roll                 : {fnum('T_roll_s', 2, ' s')}",
        f"  Fn                     : {fnum('Fn', 3)}",
        "-" * 52,
        f"  Tank 1 (SB) vulling    : {fnum('Tank1_Fill_pct', 1, ' %')}",
        f"  Tank 2 (mid) vulling   : {cfg.TANK2_FILL:.1f} %",
        f"  Tank 3 (BB) vulling    : {cfg.TANK3_FILL:.1f} %",
    ]

    sigma_b = metrics.get("Strength_Max_sigma_bodem_MPa")
    sigma_d = metrics.get("Strength_Max_sigma_dek_MPa")
    kres = metrics.get("Strength_Force_residual_kN")
    mres = metrics.get("Strength_Moment_residual_MNm")
    has_strength_in_metrics = any(isinstance(v, (int, float, np.floating)) for v in (sigma_b, sigma_d, kres, mres))
    if has_strength_in_metrics:
        lines.extend(
            [
                "-" * 52,
                f"  Sigma bodem max        : {fnum('Strength_Max_sigma_bodem_MPa', 1, ' MPa')}",
                f"  Sigma dek max          : {fnum('Strength_Max_sigma_dek_MPa', 1, ' MPa')}",
                f"  Krachtrestant |F|      : {fnum('Strength_Force_residual_kN', 3, ' kN')}",
                f"  Momentrestant |M|      : {fnum('Strength_Moment_residual_MNm', 3, ' MNm')}",
            ]
        )
    elif isinstance(eng_data, dict):
        sig_b_e = eng_data.get("Max_sigma_bodem_MPa")
        sig_d_e = eng_data.get("Max_sigma_dek_MPa")
        kres_e = eng_data.get("Force_residual_kN")
        mres_e = eng_data.get("Moment_residual_MNm")
        if any(isinstance(v, (int, float)) for v in (sig_b_e, sig_d_e)):
            def _fe(v, nd=1, sfx=""): return f"{float(v):.{nd}f}{sfx}" if isinstance(v, (int, float)) else "—"
            lines.extend(
                [
                    "-" * 52,
                    "  Sterkte (laatste eng. run)",
                    f"  Sigma bodem max        : {_fe(sig_b_e, 1, ' MPa')}",
                    f"  Sigma dek max          : {_fe(sig_d_e, 1, ' MPa')}",
                    f"  Krachtrestant |F|      : {_fe(kres_e, 3, ' kN')}",
                    f"  Momentrestant |M|      : {_fe(mres_e, 3, ' MNm')}",
                ]
            )
    lines.append("=" * 52)
    return "\n".join(lines)

def _make_open_knot_vector(n_ctrl, degree):
    """Clamped (open) uniform knot vector voor n_ctrl controlpunten en graad degree."""
    n_internal = n_ctrl - degree - 1
    internal = np.linspace(0, 1, n_internal + 2)[1:-1] if n_internal > 0 else []
    return np.concatenate([[0] * (degree + 1), internal, [1] * (degree + 1)])


def _bspline_basis_matrix(knots, n_ctrl, degree, t_array):
    """Evalueer alle B-spline basisfuncties voor alle t-waarden (vectorised).
    Geeft (len(t_array), n_ctrl) terug."""
    n_t = len(t_array)
    n_total = n_ctrl + degree

    N = np.zeros((n_t, n_total))
    for i in range(n_total):
        if i + 1 < len(knots):
            N[:, i] = ((t_array >= knots[i]) & (t_array < knots[i + 1])).astype(float)

    # Eindpunt t=1: zoek laatste niet-lege interval
    mask_end = t_array >= 1.0
    if np.any(mask_end):
        for i in range(n_total - 1, -1, -1):
            if i + 1 < len(knots) and knots[i] < knots[i + 1]:
                N[mask_end, :] = 0.0
                N[mask_end, i] = 1.0
                break

    for p in range(1, degree + 1):
        N_new = np.zeros((n_t, n_total - p))
        for i in range(n_total - p):
            d1 = knots[i + p] - knots[i]
            d2 = knots[i + p + 1] - knots[i + 1]
            if d1 > 0:
                N_new[:, i] += (t_array - knots[i]) / d1 * N[:, i]
            if d2 > 0:
                N_new[:, i] += (knots[i + p + 1] - t_array) / d2 * N[:, i + 1]
        N = N_new

    return N  # (n_t, n_ctrl)


def loose_loft_surface(ctrl_curves, degree_v=3, N_t=50):
    """Rhino 'Loose Loft': B-spline oppervlak graad degree_v in V.
    ctrl_curves: lijst van (N_u, 3) arrays — het control net.
    Geeft (N_t, N_u) arrays X, Y, Z terug."""
    n_ctrl_v = len(ctrl_curves)
    N_u = ctrl_curves[0].shape[0]
    knots_v = _make_open_knot_vector(n_ctrl_v, degree_v)

    t_array = np.linspace(0, 1, N_t)
    N_v = _bspline_basis_matrix(knots_v, n_ctrl_v, degree_v, t_array)  # (N_t, n_ctrl_v)

    ctrl_net = np.array(ctrl_curves)  # (n_ctrl_v, N_u, 3)

    # S[i, u] = sum_j  N_v[i,j] * ctrl_net[j, u, :]
    X = np.einsum('ij,jk->ik', N_v, ctrl_net[:, :, 0])
    Y = np.einsum('ij,jk->ik', N_v, ctrl_net[:, :, 1])
    Z = np.einsum('ij,jk->ik', N_v, ctrl_net[:, :, 2])

    return X, Y, Z

def _compute_surface_area(X, Y, Z):
    """Vectorised oppervlakteberekening via driehoeksmethode (één romp-helft)."""
    P = np.stack([X, Y, Z], axis=-1)          # (N_t, N_u, 3)
    p00 = P[:-1, :-1];  p10 = P[1:, :-1]
    p11 = P[1:,  1:];   p01 = P[:-1, 1:]
    a1 = 0.5 * np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
    a2 = 0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1)
    return float(np.sum(a1 + a2))


def _polygon_area_2d(y, z):
    """Shoelace-formule voor een vlak polygoon in het Y-Z vlak."""
    return 0.5 * abs(float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1))))


def _wall_effective_height(y_cs, z_cs, y_inner):
    """Effectieve hoogte van de binnenwand bij y_inner op één dwarsdoorsnede.
    Geeft het z-bereik terug waarbinnen de romp minstens y_inner breed is."""
    idx = np.where(y_cs >= y_inner - 1e-9)[0]
    if len(idx) == 0:
        return 0.0
    return max(0.0, float(z_cs[-1] - z_cs[idx[0]]))


def _longitudinal_wall_area(X_surf, Y_surf, Z_surf, y_inner, x_min, x_max):
    """Oppervlak van een langsvlak bij y=y_inner via trapezium-integratie over alle dwarsdoorsneden."""
    x_mid = X_surf.mean(axis=1)
    mask  = (x_mid >= x_min - 1e-6) & (x_mid <= x_max + 1e-6)
    xs    = x_mid[mask]
    hs    = np.array([_wall_effective_height(Y_surf[t], Z_surf[t], y_inner)
                      for t in np.where(mask)[0]])
    return float(np.trapezoid(hs, xs)) if len(xs) >= 2 else 0.0


def _interp_section(X_surf, Y_surf, Z_surf, x_pos):
    """Lineair geïnterpoleerde dwarsdoorsnede (Y, Z) op een willekeurige x-positie."""
    x_mid = X_surf.mean(axis=1)
    idx   = int(np.clip(np.searchsorted(x_mid, x_pos), 1, len(x_mid) - 1))
    t0, t1 = idx - 1, idx
    f = (x_pos - x_mid[t0]) / max(float(x_mid[t1] - x_mid[t0]), 1e-12)
    return (Y_surf[t0] * (1.0 - f) + Y_surf[t1] * f,
            Z_surf[t0] * (1.0 - f) + Z_surf[t1] * f)


def _transverse_wall_area(y_cs, z_cs, y_sb, y_port):
    """2D-oppervlak van een dwarswand (middelste tank) via shoelace op ongeresampled polygoon.
    Knipt de rompcontour naar y_sb en y_port en berekent het vlak direct."""
    port_y = np.minimum(y_cs, y_port)
    sb_y   = np.minimum(y_cs, y_sb)
    doa    = float(z_cs[-1])
    poly_y = np.concatenate([port_y[::-1], -sb_y,   [-sb_y[-1],  port_y[-1]]])
    poly_z = np.concatenate([z_cs[::-1],    z_cs,   [doa,         doa      ]])
    return _polygon_area_2d(poly_y, poly_z)


def get_curve_from_config(config_pts_key, default_pts, scale_x=1.0, scale_y=1.0, N=100):
    pad = Path(__file__).parent / CONFIG_FILE
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if config_pts_key in data:
        pts = np.array(data[config_pts_key])
        # Scale dimensionless points
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
    else:
        pts = np.array(default_pts)

    t_vals = np.linspace(0, 1, N)
    # Gebruik Rhino-style B-Spline wiskunde
    curve = get_rhino_style_spline(t_vals, pts, degree=3)
    
    val1_all = curve[:, 0]
    val2_all = curve[:, 1]
    
    return val1_all, val2_all

def get_bow_rounding_curve(N=100):
    """
    Roteert de Bow Centerline rond (0,0) met BOW_ROUNDING_DEG, 
    en schaalt de X terug zodat Punt B z'n X behoudt.
    """
    pad = Path(__file__).parent / CONFIG_FILE
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pts = np.array(data["Bow_Centerline_Points"])
    # Z is dimensionless, scale by DOA
    pts[:, 1] *= cfg.DOA
    
    # We evalueren de B-Spline van de Centerline
    t_vals = np.linspace(0, 1, N)
    curve = get_rhino_style_spline(t_vals, pts, degree=3)
    
    angle_rad = np.radians(cfg.BOW_ROUNDING_DEG)
    
    # X en Z blijven identiek aan Centerline, Y is gesweept
    x_uniform = curve[:, 0]
    y_uniform = curve[:, 0] * np.tan(angle_rad)
    z_uniform = curve[:, 1]
    
    return x_uniform, y_uniform, z_uniform

def get_midship_curve(N=100):
    B_half = cfg.BOA / 2.0
    D = cfg.DOA
    R = cfg.BILGE_RADIUS
    side_flare_deg_raw = float(getattr(cfg, "SIDE_FLARE_DEG", 0.0))
    side_flare_deg = abs(side_flare_deg_raw)  # positive = inward (V-shape)
    flare_pivot_u = float(np.clip(getattr(cfg, "SIDE_FLARE_ROTATION_POINT", 0.0), 0.0, 1.0))

    n_flat = max(4, N // 3)
    n_bilge = max(4, N // 3)
    n_side = max(4, N - n_flat - n_bilge)

    # Larger bilge radius naturally reduces usable flare.
    flare_gain = max(0.0, 1.0 - 0.9 * R / max(B_half, 1e-9))
    # Pivot effect: closer to waterline means less effective inward flare.
    flare_tan_target = np.tan(np.radians(side_flare_deg)) * flare_gain * (1.0 - 0.5 * flare_pivot_u)

    if flare_tan_target <= 1e-9:
        y_flat = np.linspace(0, max(0, B_half - R), n_flat)
        z_flat = np.zeros_like(y_flat)
        theta = np.linspace(1.5 * np.pi, 2.0 * np.pi, n_bilge)
        y_bilge = (B_half - R) + R * np.cos(theta)
        z_bilge = R + R * np.sin(theta)
        y_side = np.full(n_side, B_half)
        z_side = np.linspace(R, D, n_side)
    else:
        # Tangent-safe profile:
        # flat bottom -> circular bilge (radius R) -> inclined side.
        # We reduce flare if needed so y_flat_end never becomes negative.
        def _y_flat_end_for_m(m_local):
            alpha = np.arctan(m_local)
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            z_join_local = R * (1.0 - sin_a)
            y_join_local = B_half - m_local * (D - z_join_local)
            return y_join_local - R * cos_a

        m = float(max(flare_tan_target, 0.0))
        if _y_flat_end_for_m(m) < 0.0:
            lo, hi = 0.0, m
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if _y_flat_end_for_m(mid) >= 0.0:
                    lo = mid
                else:
                    hi = mid
            m = lo

        alpha = np.arctan(m)
        theta_end = 2.0 * np.pi - alpha
        z_join = R * (1.0 - np.sin(alpha))
        y_join = B_half - m * (D - z_join)
        y_flat_end = max(0.0, y_join - R * np.cos(alpha))

        y_flat = np.linspace(0, y_flat_end, n_flat)
        z_flat = np.zeros_like(y_flat)

        theta = np.linspace(1.5 * np.pi, theta_end, n_bilge)
        y_bilge = y_flat_end + R * np.cos(theta)
        z_bilge = R + R * np.sin(theta)

        z_side = np.linspace(z_join, D, n_side)
        y_side = y_join + m * (z_side - z_join)
    
    y_all = np.concatenate([y_flat, y_bilge[1:], y_side[1:]])
    z_all = np.concatenate([z_flat, z_bilge[1:], z_side[1:]])
    
    distances = np.sqrt(np.diff(y_all)**2 + np.diff(z_all)**2)
    cumulative_length = np.insert(np.cumsum(distances), 0, 0)
    normalized_length = cumulative_length / max(cumulative_length[-1], 1e-9)
    
    uniform_t = np.linspace(0, 1, N)
    y_uniform = np.interp(uniform_t, normalized_length, y_all)
    z_uniform = np.interp(uniform_t, normalized_length, z_all)
    
    return y_uniform, z_uniform

def _box_faces(x0, x1, y0, y1, z0, z1):
    """Geeft de 6 vlakken van een rechthoekige doos terug als lijst van polygonen."""
    return [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # bodem
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # dek
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],  # voor
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # achter
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],  # links
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],  # rechts
    ]


def draw_tank_walls(ax, x0, x1, y0, y1, z_bot, z_top, wall_color):
    """Teken alleen de tankwanden als doorzichtige gids-box (geen vulling)."""
    walls = Poly3DCollection(
        _box_faces(x0, x1, y0, y1, z_bot, z_top),
        alpha=0.10, facecolor=wall_color, edgecolor='k', linewidth=0.5,
    )
    ax.add_collection3d(walls)


def _resample_2d(y_arr, z_arr, n):
    """Hersampel een 2D-polygon naar precies n punten via booglengte."""
    dists = np.hypot(np.diff(y_arr), np.diff(z_arr))
    s = np.concatenate([[0], np.cumsum(dists)])
    if s[-1] < 1e-9:
        return np.full(n, y_arr[0]), np.full(n, z_arr[0])
    su = np.linspace(0, s[-1], n)
    return np.interp(su, s, y_arr), np.interp(su, s, z_arr)


def _port_fill_polygon(y_cs, z_cs, y_inner, z_fill, n_pts=24):
    """
    Bouw het vulfiguur voor de bakboord-zijde (positieve Y) op één dwarsdoorsnede.
    Begrenzing: binnenwand y=y_inner, rompcontour (buiten), vulhoogte z_fill.
    Geeft (poly_y, poly_z) met n_pts punten terug, of None als de romp te smal is.
    """
    # Zoek het gedeelte van de rompcontour waar y >= y_inner
    in_reg = y_cs >= (y_inner - 1e-9)
    if not np.any(in_reg):
        return None
    u0 = int(np.argmax(in_reg))
    hy = y_cs[u0:].copy()
    hz = z_cs[u0:].copy()

    # Clip aan z_fill
    over = hz > z_fill + 1e-9
    if np.any(over):
        cut = int(np.argmax(over))
        if cut == 0:
            return None
        t = (z_fill - hz[cut-1]) / max(hz[cut] - hz[cut-1], 1e-12)
        hy = np.append(hy[:cut], hy[cut-1] + t*(hy[cut]-hy[cut-1]))
        hz = np.append(hz[:cut], z_fill)

    if len(hy) < 2 or hy[-1] < y_inner - 1e-9:
        return None

    # Polygoon: binnenwand omhoog → bovenkant naar buiten → rompcontour omlaag
    raw_y = np.concatenate([[y_inner, y_inner], hy[::-1]])
    raw_z = np.concatenate([[hz[0],   z_fill  ], hz[::-1]])
    return _resample_2d(raw_y, raw_z, n_pts)


def _draw_lofted_fill(ax, px_list, py_list, pz_list, sign, fill_color, n_pts):
    """Verbind opeenvolgende dwarsdoorsnede-polygonen tot een 3D-oppervlak."""
    faces = []
    for i in range(len(px_list) - 1):
        x0, x1 = px_list[i], px_list[i+1]
        for j in range(n_pts - 1):
            faces.append([
                (x0, sign*py_list[i][j],     pz_list[i][j]),
                (x1, sign*py_list[i+1][j],   pz_list[i+1][j]),
                (x1, sign*py_list[i+1][j+1], pz_list[i+1][j+1]),
                (x0, sign*py_list[i][j+1],   pz_list[i][j+1]),
            ])
    # Eindkappen
    for idx in (0, -1):
        cap = [(px_list[idx], sign*py_list[idx][j], pz_list[idx][j]) for j in range(n_pts)]
        faces.append(cap)
    if faces:
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.85,
                                              facecolor=fill_color, edgecolor='none'))


def draw_stern_face(ax, x_pos, y_cs, z_cs, face_color='steelblue'):
    """Sluit de achtersteven af met een transomvlak."""
    # Voeg kielpunt (0,0) toe, traceer bakboord omhoog, spiegel stuurboord omlaag
    poly_y = np.concatenate([[0.0], y_cs, -y_cs[::-1], [0.0]])
    poly_z = np.concatenate([[0.0], z_cs,  z_cs[::-1], [0.0]])
    pts = [(x_pos, float(poly_y[i]), float(poly_z[i])) for i in range(len(poly_y))]
    ax.add_collection3d(Poly3DCollection([pts], alpha=0.85,
                                          facecolor=face_color, edgecolor='k', linewidth=0.4))


def draw_side_tank_fill_clipped(ax, X_surf, Y_surf, Z_surf,
                                  y_inner, z_fill, fill_color,
                                  x_min, x_max, side='port', n_pts=24):
    """
    Teken de hulpbegrensde vulling van een zijtank.
    side='port' → positieve Y; side='starboard' → negatieve Y.
    """
    x_mid = X_surf.mean(axis=1)
    sign  = 1.0 if side == 'port' else -1.0
    px, py, pz = [], [], []

    for t in range(len(x_mid)):
        xh = x_mid[t]
        if xh < x_min - 1e-6 or xh > x_max + 1e-6:
            continue
        res = _port_fill_polygon(Y_surf[t], Z_surf[t], y_inner, z_fill, n_pts)
        if res is None:
            continue
        px.append(xh); py.append(res[0]); pz.append(res[1])

    if len(px) >= 2:
        _draw_lofted_fill(ax, px, py, pz, sign, fill_color, n_pts)


def _center_fill_polygon(y_cs, z_cs, y_sb, y_port, z_fill, n_pts=24):
    """
    Hull-begrensde vulcontour voor de middelste tank in één dwarsdoorsnede.
    y_cs, z_cs: bakboord-helft van de rompcontour (y_cs[0]=0 bij kiel).
    Geeft (poly_y, poly_z) met n_pts punten terug, of None als contour leeg is.
    """
    # Clip romp aan z_fill
    over = z_cs > z_fill + 1e-9
    if np.any(over):
        c = int(np.argmax(over))
        if c == 0:
            return None
        t = (z_fill - z_cs[c-1]) / max(z_cs[c] - z_cs[c-1], 1e-12)
        hy = np.append(y_cs[:c], y_cs[c-1] + t*(y_cs[c] - y_cs[c-1]))
        hz = np.append(z_cs[:c], z_fill)
    else:
        hy, hz = y_cs.copy(), z_cs.copy()

    if len(hy) < 2:
        return None

    # Bakboord-kant: begrensd door min(rompbreedte, y_port)
    port_y = np.minimum(hy, y_port)
    # Stuurboord-kant: begrensd door min(rompbreedte, y_sb), gespiegeld naar negatief
    sb_y   = np.minimum(hy, y_sb)

    # Polygoon (Y-Z vlak):
    #   bakboord-kant omlaag (top→kiel): port_y[::-1], hz[::-1]
    #   stuurboord-kant omhoog (kiel→top): -sb_y, hz
    #   sluiting bovenaan: van -sb_y[-1] terug naar port_y[-1]
    raw_y = np.concatenate([port_y[::-1], -sb_y, [-sb_y[-1], port_y[-1]]])
    raw_z = np.concatenate([hz[::-1],      hz,   [z_fill,    z_fill    ]])

    return _resample_2d(raw_y, raw_z, n_pts)


def draw_center_tank_fill_clipped(ax, X_surf, Y_surf, Z_surf,
                                    y_sb, y_port, z_fill, fill_color,
                                    x_min, x_max, n_pts=24):
    """
    Teken de hulpbegrensde vulling van de middelste tank.
    y_sb   = binnenrand stuurboord (positief, wordt gespiegeld)
    y_port = binnenrand bakboord (positief)
    """
    x_mid = X_surf.mean(axis=1)
    px, py, pz = [], [], []

    for t in range(len(x_mid)):
        xh = x_mid[t]
        if xh < x_min - 1e-6 or xh > x_max + 1e-6:
            continue
        res = _center_fill_polygon(Y_surf[t], Z_surf[t], y_sb, y_port, z_fill, n_pts)
        if res is None:
            continue
        px.append(xh); py.append(res[0]); pz.append(res[1])

    if len(px) >= 2:
        _draw_lofted_fill(ax, px, py, pz, 1.0, fill_color, n_pts)


def draw_crane_3d(ax, crane_cfg, X_surf, Y_surf, deck_z, stowaway: bool = False):
    """Draw a simple parametric deck crane (house, mast, boom, hook, and load marker).

    When stowaway=True no lifted TP is rendered — representing the transit condition,
    but the crane remains at its configured angles.
    """
    if not isinstance(crane_cfg, dict):
        return
    swl_t = float(crane_cfg.get("swl_max_t", 0.0))
    if swl_t <= 0.0:
        return

    pivot_x = float(crane_cfg.get("pivot_x_m", 0.0))
    x_mid = X_surf.mean(axis=1)
    i_pivot = int(np.argmin(np.abs(x_mid - pivot_x)))
    local_half_beam = float(np.max(Y_surf[i_pivot]))
    pivot_y = float(crane_cfg.get("pivot_y_m", max(0.0, local_half_beam - 0.75)))
    pivot_h = float(crane_cfg.get("pivot_height_m", deck_z + 1.0))
    boom_len = float(crane_cfg.get("boom_length_m", 30.0))
    
    jib_deg = float(crane_cfg.get("jib_angle_deg", 60.0))
    slew_deg = float(crane_cfg.get("slewing_angle_deg", 90.0))

    pivot_h = max(pivot_h, deck_z + 0.5)
    boom_len = max(boom_len, 5.0)

    boom_h = boom_len * np.cos(np.radians(jib_deg))
    dx = boom_h * np.cos(np.radians(slew_deg))
    dy = boom_h * np.sin(np.radians(slew_deg))
    tip_z = pivot_h + boom_len * np.sin(np.radians(jib_deg))

    # Crane house as a compact box on deck.
    house_l = np.clip(3.0 + 0.004 * swl_t, 3.0, 8.0)
    house_w = np.clip(3.0 + 0.003 * swl_t, 3.0, 7.0)
    house_h = np.clip(2.2 + 0.0015 * swl_t, 2.2, 5.0)
    house_faces = _box_faces(
        pivot_x - 0.5 * house_l,
        pivot_x + 0.5 * house_l,
        pivot_y - 0.5 * house_w,
        pivot_y + 0.5 * house_w,
        deck_z,
        deck_z + house_h,
    )
    house = Poly3DCollection(house_faces, facecolors='dimgray', edgecolors='k', linewidths=0.4, alpha=0.85)
    ax.add_collection3d(house)

    # Mast and boom.
    ax.plot([pivot_x, pivot_x], [pivot_y, pivot_y], [deck_z + house_h, pivot_h], color='black', lw=2.2)
    ax.plot([pivot_x, pivot_x + dx], [pivot_y, pivot_y + dy], [pivot_h, tip_z], color='crimson', lw=3.2)

    hook_x = pivot_x + dx
    hook_y = pivot_y + dy

    if stowaway:
        ax.text(
            pivot_x, pivot_y, deck_z + house_h + 0.4,
            f"Kraan SWL {swl_t:.1f}t | LEEG (transit) jib {jib_deg:.0f}° slew {slew_deg:.0f}° | boom {boom_len:.0f}m",
            color='black', fontsize=8, ha='center', fontweight='bold',
        )
    else:
        # Rigging line and lifted Transition Piece (top = tip_z − 8 m rigging height).
        tp_r = 4.0
        tp_h = 20.0
        rigging_drop = 8.0
        top_tp_z = tip_z - rigging_drop
        bot_tp_z = top_tp_z - tp_h

        ax.plot([hook_x, hook_x], [hook_y, hook_y], [tip_z, top_tp_z],
                color='k', lw=1.5, ls='--')

        th = np.linspace(0.0, 2.0 * np.pi, 25)
        tx = hook_x + tp_r * np.cos(th)
        ty = hook_y + tp_r * np.sin(th)
        tp_verts = []
        for i in range(len(th) - 1):
            tp_verts.append([
                (tx[i],   ty[i],   bot_tp_z),
                (tx[i+1], ty[i+1], bot_tp_z),
                (tx[i+1], ty[i+1], top_tp_z),
                (tx[i],   ty[i],   top_tp_z),
            ])
        tp_verts.append(list(zip(tx, ty, np.full_like(tx, top_tp_z))))
        tp_verts.append(list(zip(tx, ty, np.full_like(tx, bot_tp_z))))
        ax.add_collection3d(Poly3DCollection(tp_verts, facecolors='gold', alpha=0.85,
                                             edgecolors='k', linewidths=0.5))
        ax.text(hook_x, hook_y, top_tp_z + 1.0, "TP (in luchttransport)",
                color='darkgoldenrod', fontsize=8, ha='center', fontweight='bold')

        ax.text(
            pivot_x, pivot_y, deck_z + house_h + 0.4,
            f"Kraan SWL {swl_t:.1f}t | LAADPOSITIE jib {jib_deg:.0f}° slew {slew_deg:.0f}° | boom {boom_len:.0f}m",
            color='black', fontsize=8, ha='center', fontweight='bold',
        )


def build_hull_loft(N_u=50, N_t=50):
    """Bouw de romploft en geef geometrie- en oppervlaktedata terug als dict.
    Kan door zowel plot_full_ship() als externe tools (bijv. transitiestukken) gebruikt worden."""
    LPP = cfg.LPP
    center = LPP * (cfg.MIDSHIP_LOC_PCT / 100.0)
    l_mid  = LPP * (cfg.MIDSHIP_LENGTH_PCT / 100.0)
    x_aft  = center - (l_mid / 2.0)
    x_fwd  = center + (l_mid / 2.0)

    overhang_total = cfg.LOA - LPP
    x_stern        = -(overhang_total / 2.0)
    target_bow_tip = LPP

    B_half, D = cfg.BOA / 2.0, cfg.DOA
    y_mid, z_mid = get_midship_curve(N_u)

    # Stern
    def_stern_pts = [[0.0,0.0],[2.0,1.0],[4.0,3.0],[6.0,5.0],[8.0,7.0],[10.0,9.5]]
    y_stern, z_stern = get_curve_from_config("Stern_Bezier_Points", def_stern_pts, scale_x=B_half, scale_y=D, N=N_u)
    C_stern_x = np.full(N_u, x_stern)

    x_aft_shld   = x_aft * (1.0 - cfg.AFT_SHLD_PCT / 100.0)
    x_aft_25     = x_aft * (1.0 - 0.25)
    C_aft_shld_x = np.full(N_u, x_aft_shld)
    C_aft_25_x   = np.full(N_u, x_aft_25)
    C_aft_x      = np.full(N_u, x_aft)

    # Bow Centerline
    def_bow_pts = [[1.0,0.0],[2.0,1.0],[4.0,2.0],[6.0,4.0],[8.0,6.0],[9.0,8.0],[10.0,9.5]]
    # X is NOT dimensionless, Z IS.
    local_x_bow, z_bow = get_curve_from_config("Bow_Centerline_Points", def_bow_pts, scale_x=1.0, scale_y=D, N=N_u)
    L_curve      = local_x_bow[-1]
    global_x_bow = target_bow_tip - (L_curve - local_x_bow)
    y_bow        = np.zeros(N_u)
    x_bcl_start  = global_x_bow[0]

    # Bow Intermediate
    def_bowint_pts = [[0.0,0.0],[2.0,1.0],[4.0,2.0],[6.0,4.0],[8.0,6.0],[9.0,8.0],[10.0,9.5]]
    y_bowint, z_bowint = get_curve_from_config("Bow_Intermediate_Points", def_bowint_pts, scale_x=B_half, scale_y=D, N=N_u)

    x_fwd_shld   = x_fwd + (x_bcl_start - x_fwd) * (cfg.FWD_SHLD_PCT / 100.0)
    x_fwd_25     = x_fwd + (x_bcl_start - x_fwd) * 0.25
    C_fwd_x      = np.full(N_u, x_fwd)
    C_fwd_25_x   = np.full(N_u, x_fwd_25)
    C_fwd_shld_x = np.full(N_u, x_fwd_shld)

    pct_int    = cfg.BOW_INT_PCT / 100.0
    x_bowint   = x_fwd_shld + (x_bcl_start - x_fwd_shld) * pct_int
    C_bowint_x = np.full(N_u, x_bowint)

    angle_rad      = np.radians(cfg.BOW_ROUNDING_DEG)
    y_round        = local_x_bow * np.tan(angle_rad)
    global_x_round = global_x_bow
    z_round        = z_bow

    pmc = cfg.PARALLEL_MIDSHIP_COMB
    all_ctrl_curves = [np.column_stack([C_stern_x, y_stern, z_stern])]
    if pmc >= 1:
        all_ctrl_curves.append(np.column_stack([C_aft_shld_x, y_mid, z_mid]))
    if pmc >= 2:
        all_ctrl_curves.append(np.column_stack([C_aft_25_x,   y_mid, z_mid]))
    all_ctrl_curves.append(np.column_stack([C_aft_x, y_mid, z_mid]))
    all_ctrl_curves.append(np.column_stack([C_fwd_x, y_mid, z_mid]))
    if pmc >= 2:
        all_ctrl_curves.append(np.column_stack([C_fwd_25_x,   y_mid, z_mid]))
    if pmc >= 1:
        all_ctrl_curves.append(np.column_stack([C_fwd_shld_x, y_mid, z_mid]))
    all_ctrl_curves.append(np.column_stack([C_bowint_x,    y_bowint, z_bowint]))
    all_ctrl_curves.append(np.column_stack([global_x_round, y_round, z_round]))
    all_ctrl_curves.append(np.column_stack([global_x_bow,   y_bow,   z_bow]))

    X_surf, Y_surf, Z_surf = loose_loft_surface(all_ctrl_curves, degree_v=3, N_t=N_t)

    return dict(
        X_surf=X_surf, Y_surf=Y_surf, Z_surf=Z_surf,
        x_stern=x_stern, x_bow=float(target_bow_tip),
        LPP=LPP, B_half=B_half, D=D,
        x_aft=x_aft, x_fwd=x_fwd,
        C_stern_x=C_stern_x, y_stern=y_stern, z_stern=z_stern,
        global_x_bow=global_x_bow, y_bow=y_bow, z_bow=z_bow,
        C_aft_x=C_aft_x, C_fwd_x=C_fwd_x, y_mid=y_mid, z_mid=z_mid,
        C_aft_shld_x=C_aft_shld_x, C_fwd_shld_x=C_fwd_shld_x,
        C_aft_25_x=C_aft_25_x, C_fwd_25_x=C_fwd_25_x,
        C_bowint_x=C_bowint_x, y_bowint=y_bowint, z_bowint=z_bowint,
        global_x_round=global_x_round, y_round=y_round, z_round=z_round,
        pmc=pmc,
    )


def plot_full_ship():
    N_u, N_t = 50, 50
    geo = build_hull_loft(N_u, N_t)

    X_surf, Y_surf, Z_surf = geo['X_surf'], geo['Y_surf'], geo['Z_surf']
    x_stern        = geo['x_stern']
    target_bow_tip = geo['x_bow']
    LPP, B_half, D = geo['LPP'], geo['B_half'], geo['D']
    x_aft, x_fwd   = geo['x_aft'], geo['x_fwd']
    C_stern_x      = geo['C_stern_x'];  y_stern   = geo['y_stern'];   z_stern   = geo['z_stern']
    global_x_bow   = geo['global_x_bow']; y_bow   = geo['y_bow'];     z_bow     = geo['z_bow']
    C_aft_x        = geo['C_aft_x'];    C_fwd_x   = geo['C_fwd_x']
    y_mid          = geo['y_mid'];      z_mid     = geo['z_mid']
    C_aft_shld_x   = geo['C_aft_shld_x']; C_fwd_shld_x = geo['C_fwd_shld_x']
    C_aft_25_x     = geo['C_aft_25_x']; C_fwd_25_x   = geo['C_fwd_25_x']
    C_bowint_x     = geo['C_bowint_x']; y_bowint  = geo['y_bowint'];  z_bowint  = geo['z_bowint']
    global_x_round = geo['global_x_round']; y_round = geo['y_round']; z_round  = geo['z_round']
    pmc            = geo['pmc']

    # --- 4. PLOTTEN ---
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X_surf,  Y_surf, Z_surf, color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.2)
    ax.plot_surface(X_surf, -Y_surf, Z_surf, color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.2)

    # Sluit achtersteven
    draw_stern_face(ax, x_stern, Y_surf[0], Z_surf[0], face_color='steelblue')

    # Vaste stuurcurves (dik)
    ax.plot(C_stern_x,   y_stern, z_stern, 'm-', lw=4, label='Vast: Stern')
    ax.plot(global_x_bow, y_bow,  z_bow,   'darkgreen', lw=4, label='Vast: Bow Centerline')
    ax.plot([x_stern, global_x_bow[0]], [0, 0], [0, 0], 'darkgreen', lw=4, label='Vast: Keel Extension')

    # Attractor curves — alleen actieve curves tonen (afhankelijk van pmc)
    ax.plot(C_aft_x,  y_mid, z_mid, 'r:', lw=2, label='Attractor: Midship Aft')
    ax.plot(C_fwd_x,  y_mid, z_mid, 'r:', lw=2, label='Attractor: Midship Fwd')
    ax.plot(C_aft_x,  -y_mid, z_mid, 'r:', lw=2)
    ax.plot(C_fwd_x,  -y_mid, z_mid, 'r:', lw=2)

    if pmc >= 1:
        ax.plot(C_aft_shld_x, y_mid,  z_mid, 'b:', lw=2, label=f'Attractor: Aft Shoulder ({cfg.AFT_SHLD_PCT:.0f}%)')
        ax.plot(C_fwd_shld_x, y_mid,  z_mid, 'c:', lw=2, label=f'Attractor: Fwd Shoulder ({cfg.FWD_SHLD_PCT:.0f}%)')
        ax.plot(C_aft_shld_x, -y_mid, z_mid, 'b:', lw=2)
        ax.plot(C_fwd_shld_x, -y_mid, z_mid, 'c:', lw=2)
    if pmc >= 2:
        ax.plot(C_aft_25_x, y_mid,  z_mid, 'g:', lw=2, label='Attractor: Aft 25%')
        ax.plot(C_fwd_25_x, y_mid,  z_mid, 'y:', lw=2, label='Attractor: Fwd 25%')
        ax.plot(C_aft_25_x, -y_mid, z_mid, 'g:', lw=2)
        ax.plot(C_fwd_25_x, -y_mid, z_mid, 'y:', lw=2)

    ax.plot(C_bowint_x,     y_bowint,  z_bowint, 'orange', ls='--', lw=3, label='Attractor: Bow Intermediate')
    ax.plot(global_x_round, y_round,   z_round,  'red',    ls='-.', lw=3, label='Attractor: Bow Rounding')
    ax.plot(C_bowint_x,     -y_bowint, z_bowint, 'orange', ls='--', lw=3)
    ax.plot(global_x_round, -y_round,  z_round,  'red',    ls='-.', lw=3)

    ax.set_xlabel('X [m] (Lengte)')
    ax.set_ylabel('Y [m] (Breedte)')
    ax.set_zlabel('Z [m] (Hoogte)')
    pmc_label = {0: 'geen shoulders', 1: 'alleen shoulders', 2: 'shoulders + 25%'}[pmc]
    ax.set_title(f'Volledige Romp Lofting  |  Parallel Midship: {pmc_label}\n'
                 f'Aft Shoulder: {cfg.AFT_SHLD_PCT:.0f}% | Fwd Shoulder: {cfg.FWD_SHLD_PCT:.0f}% | Bow Rounding: {cfg.BOW_ROUNDING_DEG}° | '
                 f'Side flare: {cfg.SIDE_FLARE_DEG:.1f}° @ {cfg.SIDE_FLARE_ROTATION_POINT:.2f}')
    
    x_range = target_bow_tip - x_stern
    y_range = cfg.BOA
    z_range = cfg.DOA
    
    ax.set_box_aspect((x_range, y_range, z_range))
    ax.set_xlim(x_stern, target_bow_tip)
    ax.set_ylim(-cfg.BOA/2, cfg.BOA/2)
    ax.set_zlim(0, cfg.DOA)
    
    ax.view_init(elev=20, azim=230)
    ax.legend(loc='upper right')

    # --- TANKS ---
    B_half = cfg.BOA / 2.0
    D      = cfg.DOA

    # --- TRANSITION PIECES ---
    pad = Path(__file__).parent / CONFIG_FILE
    with open(pad, "r", encoding="utf-8") as f:
        data = json.load(f)
    tps = data.get("Transition_Pieces", [])
    crane_cfg = data.get("Crane", {})
    optimizer_metrics = None
    optimizer_metrics_source = None
    live_optimizer_mode = isinstance(data.get("_optimizer_metrics"), dict)
    if live_optimizer_mode:
        optimizer_metrics = dict(data.get("_optimizer_metrics"))
        optimizer_metrics_source = "live_optimizer_generation"
    else:
        optimizer_metrics, optimizer_metrics_source = _lookup_optimizer_metrics_for_cfg(data)

    # live_eng_result is computed AFTER plt.show() to avoid blocking the window from appearing.
    live_eng_result = None

    eng_strength = _load_engineering_strength()
    
    tp_height = 20.0
    for tp in tps:
        tx, ty, tw = tp['x'], tp['y'], tp['weight_t']
        # Render als cylinder op het dek (Z=D)
        r = 4.0 # Radius uit interactive_transition_pieces.py
        theta = np.linspace(0, 2*np.pi, 25)
        px = tx + r * np.cos(theta)
        py = ty + r * np.sin(theta)
        
        # Zijwanden
        verts = []
        for i in range(len(theta)-1):
            verts.append([
                (px[i],   py[i],   D),
                (px[i+1], py[i+1], D),
                (px[i+1], py[i+1], D + tp_height),
                (px[i],   py[i],   D + tp_height)
            ])
        
        # Bovenkant
        verts.append(list(zip(px, py, np.full_like(px, D + tp_height))))
        
        poly = Poly3DCollection(verts, facecolors='gold', alpha=0.8, edgecolors='k', linewidths=0.5)
        ax.add_collection3d(poly)
        
        ax.text(tx, ty, D + tp_height + 0.3, f"{tw:.0f}t", color='black', fontsize=8, ha='center', fontweight='bold')

    # --- CRANE ---
    draw_crane_3d(ax, crane_cfg=crane_cfg, X_surf=X_surf, Y_surf=Y_surf, deck_z=D,
                  stowaway=(CRANE_MODE == "stowaway"))

    # Tank 1 — stuurboord (negatieve Y-kant)
    t1_w      = cfg.TANK1_WIDTH
    t1_y_inner = B_half - t1_w          # binnenwand afstand tot middenlijn
    t1_z_fill  = D * cfg.TANK1_FILL / 100.0
    draw_tank_walls(ax, x0=0, x1=LPP,
                    y0=-B_half, y1=-(B_half - t1_w),
                    z_bot=0, z_top=D, wall_color='limegreen')
    draw_side_tank_fill_clipped(ax, X_surf, Y_surf, Z_surf,
                                 y_inner=t1_y_inner, z_fill=t1_z_fill,
                                 fill_color='lime',
                                 x_min=0, x_max=LPP, side='starboard')

    # Tank 3 — bakboord (positieve Y-kant)
    t3_w      = cfg.TANK3_WIDTH
    t3_y_inner = B_half - t3_w
    t3_z_fill  = D * cfg.TANK3_FILL / 100.0
    draw_tank_walls(ax, x0=0, x1=LPP,
                    y0=B_half - t3_w, y1=B_half,
                    z_bot=0, z_top=D, wall_color='tomato')
    draw_side_tank_fill_clipped(ax, X_surf, Y_surf, Z_surf,
                                 y_inner=t3_y_inner, z_fill=t3_z_fill,
                                 fill_color='orangered',
                                 x_min=0, x_max=LPP, side='port')

    # Tank 2 — midden
    t2_len   = cfg.LOA * cfg.TANK2_LEN_PCT / 100.0
    t2_cx    = cfg.TANK2_CENTER
    t2_x0    = t2_cx - t2_len / 2.0
    t2_x1    = t2_cx + t2_len / 2.0
    t2_z_fill = D * cfg.TANK2_FILL / 100.0
    draw_tank_walls(ax, x0=t2_x0, x1=t2_x1,
                    y0=-(B_half - t1_w), y1=(B_half - t3_w),
                    z_bot=0, z_top=D, wall_color='royalblue')
    draw_center_tank_fill_clipped(ax, X_surf, Y_surf, Z_surf,
                                   y_sb=t1_y_inner, y_port=t3_y_inner,
                                   z_fill=t2_z_fill, fill_color='deepskyblue',
                                   x_min=t2_x0, x_max=t2_x1)

    # --- GEWICHTSBEREKENING ---
    RHO    = 7850.0
    FAC    = 2.1
    HULL_T = max(cfg.HULL_THICKNESS_MM, 8.0) / 1000.0
    WALL_T = 0.010

    # Romplaat: beide helften (numerieke oppervlakteberekening via driehoeksom)
    hull_half_area = _compute_surface_area(X_surf, Y_surf, Z_surf)
    hull_area      = 2.0 * hull_half_area
    W_hull         = hull_area * HULL_T * RHO * FAC

    # Tank 1 (SB) binnenwand — trapezium-integratie van effectieve hoogte over x
    A_t1 = _longitudinal_wall_area(X_surf, Y_surf, Z_surf, t1_y_inner, 0.0, LPP)
    W_t1 = A_t1 * WALL_T * RHO * FAC

    # Tank 3 (BB) binnenwand — zelfde methode
    A_t3 = _longitudinal_wall_area(X_surf, Y_surf, Z_surf, t3_y_inner, 0.0, LPP)
    W_t3 = A_t3 * WALL_T * RHO * FAC

    # Tank 2 voor- en achterwand — shoelace op rompgeknipt polygoon op geïnterpoleerde positie
    y_cs_fwd, z_cs_fwd = _interp_section(X_surf, Y_surf, Z_surf, t2_x0)
    y_cs_aft, z_cs_aft = _interp_section(X_surf, Y_surf, Z_surf, t2_x1)
    A_t2_fwd = _transverse_wall_area(y_cs_fwd, z_cs_fwd, t1_y_inner, t3_y_inner)
    A_t2_aft = _transverse_wall_area(y_cs_aft, z_cs_aft, t1_y_inner, t3_y_inner)
    W_t2 = (A_t2_fwd + A_t2_aft) * WALL_T * RHO * FAC

    # Achtersteven — shoelace op het volledige transompolygoon (beide helften)
    stern_poly_y = np.concatenate([[0.0], Y_surf[0], -Y_surf[0][::-1]])
    stern_poly_z = np.concatenate([[0.0], Z_surf[0],  Z_surf[0][::-1]])
    A_stern = _polygon_area_2d(stern_poly_y, stern_poly_z)
    W_stern = A_stern * WALL_T * RHO * FAC

    # Dek — trapezium-integratie van 2·y_deck over x (plan-oppervlak op z=DOA)
    x_deck  = X_surf[:, 0]
    y_deck  = Y_surf[:, -1]
    A_deck  = 2.0 * float(np.trapezoid(y_deck, x_deck))
    W_deck  = A_deck * HULL_T * RHO * FAC   # zelfde dikte en factor als romplaat

    # Kraanmassa conform opdrachtverhouding (zonder extra FAC, directe equipment mass).
    crane_swl = float(crane_cfg.get("swl_max_t", 0.0)) if isinstance(crane_cfg, dict) else 0.0
    W_crane = (0.34 + 0.17 + 0.06) * crane_swl * 1000.0

    W_total = W_hull + W_deck + W_t1 + W_t3 + W_t2 + W_stern + W_crane

    lines = [
        "=" * 52,
        "  CONSTRUCTIEGEWICHTEN",
        "=" * 52,
        f"  Romplaat    {cfg.HULL_THICKNESS_MM:.0f} mm   A = {hull_area:7.1f} m²   {W_hull/1000:7.1f} t",
        f"  Dek         {cfg.HULL_THICKNESS_MM:.0f} mm   A = {A_deck:7.1f} m²   {W_deck/1000:7.1f} t",
        f"  Tank1 wand  10 mm   A = {A_t1:7.1f} m²   {W_t1/1000:7.1f} t",
        f"  Tank3 wand  10 mm   A = {A_t3:7.1f} m²   {W_t3/1000:7.1f} t",
        f"  Tank2 voor  10 mm   A = {A_t2_fwd:7.1f} m²   {W_t2/2/1000:7.1f} t",
        f"  Tank2 acht  10 mm   A = {A_t2_aft:7.1f} m²   {W_t2/2/1000:7.1f} t",
        f"  Achtersteven10 mm   A = {A_stern:7.1f} m²   {W_stern/1000:7.1f} t",
        f"  Kraan equip.         SWL={crane_swl:5.0f} t      {W_crane/1000:7.1f} t",
        "-" * 52,
        f"  TOTAAL                                {W_total/1000:7.1f} t",
        "=" * 52,
    ]
    print("\n".join(lines))

    if isinstance(optimizer_metrics, dict):
        if "Design_Speed_kn" not in optimizer_metrics and isinstance(data.get("Design_Speed_kn"), (int, float)):
            optimizer_metrics["Design_Speed_kn"] = float(data.get("Design_Speed_kn"))

    if live_optimizer_mode:
        # Phase 1: show the window immediately with fast optimizer metrics.
        panel_txt = _render_optimizer_metrics_text(
            optimizer_metrics,
            optimizer_metrics_source or "live_optimizer_generation",
            eng_data=eng_strength,
        )
        panel_obj = fig.text(0.01, 0.97, panel_txt, transform=fig.transFigure,
                             fontsize=9, verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85, edgecolor='gray'))
        plt.tight_layout()
        plt.ion()
        plt.show()
        plt.pause(0.1)

        # Phase 2: run full engineering while window is already visible (3-10s).
        live_eng_result = _run_engineering_for_live_plot(data)
        if isinstance(live_eng_result, dict):
            gen = data.get("Optimizer_Generation", "?")
            full_txt = _render_full_engineering_text(live_eng_result, geo, data, source_label=f"gen {gen}")
            panel_obj.set_text(full_txt)
            panel_obj.get_bbox_patch().set_facecolor('lightyellow')
            plt.draw()
            plt.pause(0.1)

        # Phase 3: keep window alive until the next generation kills this process.
        try:
            while plt.fignum_exists(fig.number):
                plt.pause(1.0)
        except Exception:
            pass
    else:
        if isinstance(optimizer_metrics, dict):
            panel_txt = _render_optimizer_metrics_text(
                optimizer_metrics,
                optimizer_metrics_source or "optim_results",
                eng_data=eng_strength,
            )
            panel_face = "lightcyan"
        else:
            # Build legacy construction-weight panel with tank fills and strength.
            def _eng_strength_line(key, label, nd=1, sfx=""):
                if not isinstance(eng_strength, dict):
                    return None
                v = eng_strength.get(key)
                if isinstance(v, (int, float)):
                    return f"{label}: {float(v):.{nd}f}{sfx}"
                return None

            strength_lines = [
                _eng_strength_line("Max_sigma_bodem_MPa", "Sigma bodem max", 1, " MPa"),
                _eng_strength_line("Max_sigma_dek_MPa",   "Sigma dek max",   1, " MPa"),
                _eng_strength_line("Force_residual_kN",   "Krachtrestant",   3, " kN"),
                _eng_strength_line("Moment_residual_MNm", "Momentrestant",   3, " MNm"),
            ]
            strength_block = "\n".join(l for l in strength_lines if l)

            panel_txt = (
                f"Romplaat ({cfg.HULL_THICKNESS_MM:.0f} mm, {hull_area:.0f} m²): {W_hull/1000:.1f} t\n"
                f"Dek       ({cfg.HULL_THICKNESS_MM:.0f} mm, {A_deck:.0f} m²): {W_deck/1000:.1f} t\n"
                f"Tank1+3 wanden (10 mm): {(W_t1+W_t3)/1000:.1f} t\n"
                f"Tank2 voor+acht (10 mm): {W_t2/1000:.1f} t\n"
                f"Achtersteven (10 mm): {W_stern/1000:.1f} t\n"
                f"Kraan equipment: {W_crane/1000:.1f} t\n"
                f"── TOTAAL: {W_total/1000:.1f} t ──\n"
                f"────────────────────────────────\n"
                f"Tank 1 (SB): {cfg.TANK1_FILL:.1f}%  "
                f"Tank 2: {cfg.TANK2_FILL:.1f}%  "
                f"Tank 3 (BB): {cfg.TANK3_FILL:.1f}%"
            )
            if strength_block:
                panel_txt += f"\n────────────────────────────────\n{strength_block}"
            panel_face = "lightyellow"

        fig.text(0.01, 0.97, panel_txt, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor=panel_face, alpha=0.85, edgecolor='gray'))

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot de volledige 3D-romp.")
    parser.add_argument("--config", type=str, default="config.json", help="Pad naar config JSON bestand.")
    parser.add_argument(
        "--crane-mode",
        type=str,
        choices=["loading", "stowaway"],
        default="stowaway",
        help="Kraanpositie: 'loading' toont de operationele laadpositie, "
             "'stowaway' toont de gestuwd transportpositie (jib=0°, slew=0°).",
    )
    args = parser.parse_args()

    CONFIG_FILE = args.config
    CRANE_MODE = args.crane_mode
    reload_config_module(CONFIG_FILE)

    plot_full_ship()
