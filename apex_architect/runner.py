from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from scipy.stats import qmc

from .constants import (
    DESIGN_SPEED_KN,
    MORPH_VAR_COUNT,
    N_VAR,
    STRENGTH_SIGMA_ALLOW_MPA,
    STRENGTH_FORCE_RESIDUAL_MAX_KN,
    STRENGTH_MOMENT_RESIDUAL_MAX_MNM,
    TRIM_LCG_LCB_TOL_M,
    HEEL_TCG_TOL_M,
)
from .evaluation import evaluate_one_unit
from .io_utils import DesignContext, save_json
from .mcdm import enrich_scores, select_best
from .monitoring import HypervolumeEarlyStopCallback
from .paths import BEST_CFG_FILE, CFG_PATH, REPORT_FILE, RESULTS_FILE
from .problem import ApexArchitectProblem


def sobol_sampling_unit(pop_size: int, n_var: int, seed: int) -> np.ndarray:
    m = int(np.ceil(np.log2(pop_size)))
    sampler = qmc.Sobol(d=n_var, scramble=True, seed=seed)
    u = sampler.random_base2(m=m)
    return u[:pop_size, :]


def heuristic_seed_unit_vectors(n_var: int) -> np.ndarray:
    """A few hand-crafted feasible-leaning seeds to help heavy-load convergence."""
    if n_var < 44:
        return np.empty((0, n_var), dtype=float)

    seeds = []

    # Wide/deep, thin steel, low tank1 fill, long boom, moderate SWL.
    s0 = np.full(n_var, 0.5, dtype=float)
    s0[0] = 0.70   # LOA
    s0[1] = 0.95   # BOA
    s0[2] = 0.75   # DOA
    s0[11] = 0.05  # hull thickness
    s0[12] = 0.92  # draft fraction
    s0[13] = 0.70  # tank1 width frac
    s0[14] = 0.85  # tank3 width frac
    s0[15] = 0.55  # tank2 length
    s0[16] = 0.02  # tank1 fill
    s0[20] = 0.78  # crane pivot x frac
    s0[21] = 0.80  # payload control fraction
    s0[22] = 0.55  # crane boom length (~55 m)
    s0[23] = 0.75  # crane pivot Y frac → ~+0.5 (near port edge)
    s0[24] = 0.0   # jib angle → 60 deg (minimum, widest outreach)
    s0[25] = 0.25  # slew angle → 90 deg (full transverse reach)
    seeds.append(s0)

    # Payload-biased variant.
    s1 = s0.copy()
    s1[0] = 0.85
    s1[1] = 1.00
    s1[2] = 0.85
    s1[20] = 0.70
    s1[21] = 0.95
    s1[22] = 0.65  # slightly longer boom for wider ship
    s1[23] = 0.75  # near port edge
    s1[24] = 0.0   # 60 deg jib
    s1[25] = 0.25  # 90 deg slew
    seeds.append(s1)

    # Resistance-biased variant: smaller ship, shorter boom.
    s2 = s0.copy()
    s2[0] = 0.60
    s2[1] = 0.80
    s2[2] = 0.70
    s2[7] = 0.35
    s2[8] = 0.35
    s2[9] = 0.55
    s2[21] = 0.60
    s2[22] = 0.35  # shorter boom suits narrower ship
    s2[23] = 0.75  # near port edge
    s2[24] = 0.0   # 60 deg jib
    s2[25] = 0.25  # 90 deg slew
    seeds.append(s2)

    return np.array(seeds, dtype=float)


def collect_feasible_front(X: np.ndarray, ctx: DesignContext, min_tps: int, ship_mode: str = "both") -> list[dict]:
    candidates: list[dict] = []
    for row in X:
        e = evaluate_one_unit(row, ctx, min_tps=min_tps, ship_mode=ship_mode)
        if np.all(e["G"] <= 0.0) and e["cfg"] is not None:
            m = dict(e["metrics"])
            m["F"] = e["F"].tolist()
            m["G"] = e["G"].tolist()
            m = enrich_scores(m)
            candidates.append({"x": row.tolist(), "cfg": e["cfg"], "metrics": m})
    return candidates


def post_filter_strict_equilibrium(candidates: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    dropped = 0
    for c in candidates:
        ok = bool(c["metrics"].get("StrictEq_ok", False))
        if ok:
            kept.append(c)
        else:
            dropped += 1
    return kept, dropped


def post_filter_groep10_strength(candidates: list[dict]) -> tuple[list[dict], int]:
    """Verify candidates via full engineering+groep10 pipeline and drop violators."""
    if not candidates:
        return candidates, 0

    from engineering.run import run_all
    import plot_full_surface as _pfs

    kept: list[dict] = []
    dropped = 0
    n_total = len(candidates)
    errors: list[str] = []

    # Per-constraint diagnostics: list of (g_value, raw_value) for failing candidates
    diag: dict[str, list] = {
        "sigma": [], "force": [], "moment": [], "trim": [], "heel": []
    }
    all_results: list[dict] = []  # for "best candidate" summary

    original_cfg_text = CFG_PATH.read_text(encoding="utf-8")
    try:
        for i, c in enumerate(candidates):
            print(f"  Strength filter: {i+1}/{n_total} ...", end="\r", flush=True)
            save_json(CFG_PATH, dict(c["cfg"]))
            # Reset the config path in plot_full_surface — may have been set to a
            # stale PID-specific temp file by collect_feasible_front worker evals.
            _pfs.CONFIG_FILE = "config.json"
            _pfs.reload_config_module("config.json")
            try:
                result = run_all(verbose=False)
            except (ValueError, RuntimeError) as exc:
                c["metrics"]["Strength_Source"] = "error"
                c["metrics"]["Strength_error"] = str(exc)
                errors.append(str(exc))
                dropped += 1
                continue

            max_sigma_bodem = float(result.get("max_sigma_bodem", 0.0))
            max_sigma_dek = float(result.get("max_sigma_dek", 0.0))
            force_residual_kn = abs(float(result.get("krachtrestant_kn", 0.0)))
            moment_residual_mnm = abs(float(result.get("momentrestant_mnm", 0.0)))
            lcg = float(result.get("lcg_total", 0.0))
            lcb = float(result.get("lcb", 0.0))
            tcg = abs(float(result.get("tcg_total", 0.0)))

            g_sigma = max(max_sigma_bodem, max_sigma_dek) - STRENGTH_SIGMA_ALLOW_MPA
            g_force = force_residual_kn - STRENGTH_FORCE_RESIDUAL_MAX_KN
            g_moment = moment_residual_mnm - STRENGTH_MOMENT_RESIDUAL_MAX_MNM
            g_trim = abs(lcg - lcb) - TRIM_LCG_LCB_TOL_M
            g_heel = tcg - HEEL_TCG_TOL_M

            c["metrics"]["Strength_Source"] = str(result.get("strength_source", "unknown"))
            c["metrics"]["Strength_Max_sigma_bodem_MPa"] = max_sigma_bodem
            c["metrics"]["Strength_Max_sigma_dek_MPa"] = max_sigma_dek
            c["metrics"]["Strength_Force_residual_kN"] = force_residual_kn
            c["metrics"]["Strength_Moment_residual_MNm"] = moment_residual_mnm
            c["metrics"]["Strength_LCG_LCB_delta_m"] = abs(lcg - lcb)
            c["metrics"]["Strength_TCG_abs_m"] = tcg
            c["metrics"]["Strength_g_sigma"] = g_sigma
            c["metrics"]["Strength_g_force"] = g_force
            c["metrics"]["Strength_g_moment"] = g_moment
            c["metrics"]["Strength_g_trim"] = g_trim
            c["metrics"]["Strength_g_heel"] = g_heel

            all_results.append({
                "g_sigma": g_sigma, "g_force": g_force, "g_moment": g_moment,
                "g_trim": g_trim, "g_heel": g_heel,
                "sigma": max(max_sigma_bodem, max_sigma_dek),
                "force_kn": force_residual_kn,
                "moment_mnm": moment_residual_mnm,
                "trim_m": abs(lcg - lcb),
                "heel_m": tcg,
            })

            passes = all(g <= 0.0 for g in (g_sigma, g_force, g_moment, g_trim, g_heel))
            if passes:
                kept.append(c)
            else:
                for key, g_val, raw_val in (
                    ("sigma",  g_sigma,  max(max_sigma_bodem, max_sigma_dek)),
                    ("force",  g_force,  force_residual_kn),
                    ("moment", g_moment, moment_residual_mnm),
                    ("trim",   g_trim,   abs(lcg - lcb)),
                    ("heel",   g_heel,   tcg),
                ):
                    if g_val > 0.0:
                        diag[key].append((g_val, raw_val))
                dropped += 1
    finally:
        Path(CFG_PATH).write_text(original_cfg_text, encoding="utf-8")

    print()  # clear progress line

    if dropped > 0 and not kept:
        _print_strength_filter_diagnostics(diag, errors, all_results, n_total)

    return kept, dropped


def _print_strength_filter_diagnostics(
    diag: dict,
    errors: list[str],
    all_results: list[dict],
    n_total: int,
) -> None:
    labels = {
        "sigma":  (f"sigma > {STRENGTH_SIGMA_ALLOW_MPA:.0f} MPa",   "MPa"),
        "force":  (f"krachtrestant > {STRENGTH_FORCE_RESIDUAL_MAX_KN:.1f} kN", "kN"),
        "moment": (f"momentrestant > {STRENGTH_MOMENT_RESIDUAL_MAX_MNM:.1f} MNm", "MNm"),
        "trim":   (f"|LCG-LCB| > {TRIM_LCG_LCB_TOL_M:.2f} m",     "m"),
        "heel":   (f"|TCG| > {HEEL_TCG_TOL_M:.2f} m",              "m"),
    }
    print("-" * 65)
    print(f"  Strength filter diagnostics  ({n_total} kandidaten)")
    print("-" * 65)
    for key, (desc, unit) in labels.items():
        vals = diag[key]
        if not vals:
            print(f"  ✓ {desc:48s}  0/{n_total}")
        else:
            raw_vals = [v for _, v in vals]
            print(
                f"  ✗ {desc:48s}  {len(vals)}/{n_total}"
                f"  range [{min(raw_vals):.2f}, {max(raw_vals):.2f}] {unit}"
            )
    if errors:
        from collections import Counter
        err_counts = Counter(e.split(":")[0] for e in errors)
        print(f"  ! Fouten (ValueError/RuntimeError): {len(errors)}/{n_total}")
        for msg, cnt in err_counts.most_common(3):
            print(f"      {cnt}× {msg[:60]}")
    if all_results:
        # Find candidate closest to passing (minimum total violation)
        def _total_viol(r: dict) -> float:
            return sum(max(0.0, r[k]) for k in ("g_sigma", "g_force", "g_moment", "g_trim", "g_heel"))
        best = min(all_results, key=_total_viol)
        print("  Beste kandidaat (dichtstbij halen):")
        print(f"    force={best['force_kn']:.1f} kN  sigma={best['sigma']:.1f} MPa"
              f"  moment={best['moment_mnm']:.3f} MNm"
              f"  trim={best['trim_m']:.3f} m  heel={best['heel_m']:.4f} m")
    print("-" * 65)


def write_report(
    best: dict,
    ht_idx: int | None,
    n_pareto: int,
    n_postfilter_dropped: int,
    n_strength_dropped: int,
    elapsed_s: float,
    stopped_early: bool,
    hv_last: float | None,
) -> None:
    m = best["metrics"]
    lines = [
        "=" * 70,
        "APEX ARCHITECT RESULTAAT",
        "=" * 70,
        f"Pareto oplossingen (haalbaar): {n_pareto}",
        f"Post-filter gesneuveld (strict equilibrium): {n_postfilter_dropped}",
        f"Post-filter gesneuveld (groep10 strength): {n_strength_dropped}",
        f"Runtime: {elapsed_s/60.0:.1f} min",
        f"Early stop (HV plateau): {'ja' if stopped_early else 'nee'}",
        f"Laatste HV: {hv_last:.6f}" if hv_last is not None else "Laatste HV: -",
        f"HighTradeoff index in front: {ht_idx if ht_idx is not None else '-'}",
        "-" * 70,
        f"Rtot @ {DESIGN_SPEED_KN:.1f} kn : {m['Rtot_kN']:.1f} kN",
        f"N_TPs (payload)         : {m['N_TPs']}",
        f"LSW (incl. ballast)     : {m['LSW_t']:.1f} t",
        f"Payload                 : {m['Payload_t']:.1f} t",
        f"Empty Ship Weight       : {m.get('ShipWeight_t', 0.0):.1f} t",
        f"Crane load capacity     : {m.get('Crane_Load_MN', 0.0):.3f} MN",
        f"LSW/Payload (diagnostic): {m['LSW_per_payload']:.3f} t/t",
        f"GM                      : {m['GM_m']:.2f} m",
        f"T_roll                  : {m['T_roll_s']:.2f} s",
        f"Crane heel @ lift       : {m.get('Crane_heel_deg', 0.0):.2f} deg",
        f"Crane SWL eff/max       : {m.get('Crane_Swl_effective_t', 0.0):.1f} / {m.get('Crane_Swl_max_t', 0.0):.1f} t",
        f"Vrijboord               : {m['Freeboard_m']:.2f} m",
        f"Fn                      : {m['Fn']:.3f}",
        f"Massareserve            : {m['MassReserve_t']:.1f} t",
        f"Ballast                 : {m['Ballast_t']:.1f} t ({100.0*m['Ballast_frac_disp']:.1f}% van disp)",
        "-" * 70,
        f"10-punten weerstand     : {m['score_R']:.2f}/10",
        f"10-punten payload       : {m['score_P']:.2f}/10",
        f"10-punten ShipWeight    : {m['score_LSW']:.2f}/10",
        f"Totaalscore             : {m['score_total_30']:.2f}/30",
        "-" * 70,
        f"Competitie weerstand    : {m.get('score_comp_resistance', 0.0):.2f}/10",
        f"Competitie lading       : {m.get('score_comp_payload', 0.0):.2f}/10",
        f"Competitie kraanlast    : {m.get('score_comp_crane', 0.0):.2f}/10",
        f"Competitie scheepsgew.  : {m.get('score_comp_shipweight', 0.0):.2f}/10",
        f"Competitie totaal       : {m.get('score_comp_total_40', 0.0):.2f}/40",
        f"Benchmark wins          : {m.get('beats_count_4', 0)}/4  |  all four beaten: {'ja' if m.get('beats_all_4', False) else 'nee'}",
        f"Harrington D            : {m['desirability_D']:.4f}",
        f"ASF waarde              : {m['asf_value']:.4f}",
        "=" * 70,
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _serialize_candidate(c: dict) -> dict:
    return {
        "metrics": c["metrics"],
        "objectives": {
            "f1_resistance_kN": c["metrics"]["F"][0],
            "f2_empty_ship_weight_t": c["metrics"]["F"][1],
            "f3_negative_payload": c["metrics"]["F"][2],
        },
        "constraints": {
            "g1_gm": c["metrics"]["G"][0],
            "g2_freeboard": c["metrics"]["G"][1],
            "g3_tp_deck_width": c["metrics"]["G"][2],
            "g4_mass_reserve": c["metrics"]["G"][3],
            "g5_min_draft": c["metrics"]["G"][4],
            "g6_froude_limit": c["metrics"]["G"][5],
            "g7_ballast_fraction": c["metrics"]["G"][6],
            "g8_roll_period": c["metrics"]["G"][7],
            "g9_strict_equilibrium": c["metrics"]["G"][8],
            "g10_crane_swl_and_geometry": c["metrics"]["G"][9],
            "g11_crane_heel_limit": c["metrics"]["G"][10],
            "g12_max_draft": c["metrics"]["G"][11],
            "g13_min_transition_pieces": c["metrics"]["G"][12],
            "g14_roll_period_max": c["metrics"]["G"][13],
            "g15_fast_strength": c["metrics"]["G"][14],
            "g16_transverse_moment": c["metrics"]["G"][15],
        },
        "cfg": c["cfg"],
    }


def run(
    pop_size: int,
    n_gen: int,
    seed: int,
    verbose: bool,
    ctx: DesignContext,
    n_threads: int,
    min_tps: int = 0,
    ship_mode: str = "both",
    plot_mode: str = "auto",
) -> int:
    if ctx.n_var != N_VAR:
        print(f"FOUT: bounds mismatch. verwacht {N_VAR}, kreeg {ctx.n_var}")
        return 2

    print("=" * 70)
    print("APEX ARCHITECT - NSGA-II PARETO OPTIMISATIE")
    print(f"Populatie: {pop_size}  |  Generaties: {n_gen}  |  n_var: {N_VAR}")
    print(f"Morph variabelen: {MORPH_VAR_COUNT}  |  Sobol init: ja")
    print(f"Minimum transition pieces: {max(0, int(min_tps))}")
    print(f"Ship mode: {ship_mode}")
    print("=" * 70)

    sampling = sobol_sampling_unit(pop_size=pop_size, n_var=N_VAR, seed=seed)
    anchors = heuristic_seed_unit_vectors(n_var=N_VAR)
    if len(anchors) > 0:
        k = min(len(anchors), len(sampling))
        sampling[:k, :] = np.clip(anchors[:k, :], 0.0, 1.0)
    problem = ApexArchitectProblem(ctx=ctx, n_threads=n_threads, min_tps=min_tps, ship_mode=ship_mode)
    algorithm = NSGA2(pop_size=pop_size, sampling=sampling, eliminate_duplicates=True)
    hv_cb = HypervolumeEarlyStopCallback(ctx=ctx, live_plot=True, plot_mode=plot_mode)

    t0 = time.time()
    try:
        result = minimize(
            problem,
            algorithm,
            termination=("n_gen", n_gen),
            seed=seed,
            verbose=verbose,
            save_history=False,
            return_least_infeasible=True,
            callback=hv_cb,
        )
    finally:
        hv_cb.close()
        problem.close()
    elapsed = time.time() - t0

    X_res = None
    if getattr(result, "pop", None) is not None:
        try:
            X_res = result.pop.get("X")
        except Exception:
            X_res = None
    if X_res is None:
        X_res = result.X
    if X_res is None:
        print("Geen oplossingen teruggekregen van NSGA-II.")
        return 3

    X_res = np.atleast_2d(X_res)
    candidates = collect_feasible_front(X_res, ctx=ctx, min_tps=min_tps, ship_mode=ship_mode)
    if not candidates:
        print("Geen haalbare Pareto-oplossingen (alle constraints voldeden niet).")
        return 4
    candidates_filtered, dropped = post_filter_strict_equilibrium(candidates)
    if not candidates_filtered:
        print("Geen ontwerpen over na strict-equilibrium post-filter.")
        print(f"Gesneuveld in post-filter: {dropped}")
        return 5
    candidates_strength, dropped_strength = post_filter_groep10_strength(candidates_filtered)
    if not candidates_strength:
        print(f"Geen ontwerpen over na groep10-strength post-filter (gesneuveld: {dropped_strength}).")
        return 6

    best, ht_idx = select_best(candidates_strength)
    save_json(BEST_CFG_FILE, dict(best["cfg"]))

    top = sorted(candidates_strength, key=lambda c: (-c["metrics"]["desirability_D"], c["metrics"]["asf_value"]))
    by_resistance = min(candidates_strength, key=lambda c: c["metrics"]["Rtot_kN"])
    by_payload = max(candidates_strength, key=lambda c: c["metrics"]["N_TPs"])
    by_efficiency = min(candidates_strength, key=lambda c: c["metrics"]["ShipWeight_t"])

    hv_valid = [d["hv"] for d in hv_cb.hv_history if np.isfinite(d["hv"])]
    hv_last = float(hv_valid[-1]) if hv_valid else None
    stopped_early = bool(hv_cb.hv_history and hv_cb.hv_history[-1]["gen"] < n_gen)

    payload = {
        "selected_best": _serialize_candidate(best),
        "extremes": {
            "min_resistance": _serialize_candidate(by_resistance),
            "max_payload": _serialize_candidate(by_payload),
            "min_empty_ship_weight": _serialize_candidate(by_efficiency),
        },
        "hypervolume_history": hv_cb.hv_history,
        "pareto_top": [_serialize_candidate(c) for c in top[:100]],
    }
    save_json(RESULTS_FILE, payload)
    write_report(
        best,
        ht_idx=ht_idx,
        n_pareto=len(candidates_strength),
        n_postfilter_dropped=dropped,
        n_strength_dropped=dropped_strength,
        elapsed_s=elapsed,
        stopped_early=stopped_early,
        hv_last=hv_last,
    )

    print(f"\nBeste config opgeslagen in: {BEST_CFG_FILE.name}")
    print(f"Pareto resultaten opgeslagen in: {RESULTS_FILE.name}")
    print(f"Rapport opgeslagen in: {REPORT_FILE.name}")
    return 0
