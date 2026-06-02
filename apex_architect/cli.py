from __future__ import annotations

import argparse
import sys
import traceback

from .design import build_physical_bounds
from .io_utils import load_design_context, load_limits
from .paths import EVAL_CFG_PATH, HERE
from .runner import run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apex Architect optimizer (pymoo + NSGA-II)")
    parser.add_argument("--quick", action="store_true", help="snelle smoke run")
    parser.add_argument("--pop-size", type=int, default=None, help="override populatiegrootte")
    parser.add_argument("--n-gen", type=int, default=None, help="override aantal generaties")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--threads", type=int, default=8, help="aantal evaluatie-threads")
    parser.add_argument("--min-tps", type=int, default=0, help="minimum aantal transition pieces (hard constraint)")
    parser.add_argument(
        "--ship-mode",
        choices=["both", "crane-only", "tps-only"],
        default="both",
        help="welke componenten meenemen: both (standaard), crane-only (geen TPs), tps-only (geen kraan)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: F401
    except ImportError:
        print("FOUT: pymoo ontbreekt. Installeer met: python -m pip install pymoo")
        return 1

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.quick:
        pop = args.pop_size if args.pop_size is not None else 40
        n_gen = args.n_gen if args.n_gen is not None else 20
    else:
        pop = args.pop_size if args.pop_size is not None else 200
        n_gen = args.n_gen if args.n_gen is not None else 500

    try:
        # bounds are physical, problem is normalized [0,1] and decode rescales
        limits = load_limits()
        xl, xu = build_physical_bounds(limits)
        ctx = load_design_context(xl=xl, xu=xu)
        return run(
            pop_size=pop,
            n_gen=n_gen,
            seed=args.seed,
            verbose=True,
            ctx=ctx,
            n_threads=args.threads,
            min_tps=max(0, int(args.min_tps)),
            ship_mode=args.ship_mode,
        )
    except KeyboardInterrupt:
        print("\nAfgebroken door gebruiker.")
        return 130
    except Exception as exc:
        print(f"\nOnverwachte fout: {exc}")
        traceback.print_exc()
        return 1
    finally:
        try:
            for p in HERE.glob(f"{EVAL_CFG_PATH.stem}*{EVAL_CFG_PATH.suffix}"):
                p.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
