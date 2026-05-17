"""CLI wrapper for the Four-Rooms experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core import run as core_run
from exp1_four_rooms import four_rooms_mdp as FRM
from core import output as OUT


def parse_args() -> argparse.Namespace:
    """Define the Four-Rooms experiment command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the Four-Rooms experiment."
    )
    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--metric-kind",
        choices=["one_step", "fixed_point"],
        default="fixed_point",
        help="Which concrete pair metric to use.",
    )
    parser.add_argument(
        "--abstract-alphabet-size",
        "--abstract-size",
        dest="abstract_alphabet_size",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--abstraction-solver",
        choices=["flat", "lambda"],
        default="flat",
        help="Which BA bottleneck fitter to use.",
    )
    parser.add_argument(
        "--abstraction-error-mode",
        choices=["average", "max"],
        default="average",
        help="How to aggregate encoder/decoder distortion into the scalar abstraction error.",
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="0,5,10,15,20",
        #default="0,2,4,6,8,10,15,20",
    )
    parser.add_argument(
        "--adaptive-beta-step",
        type=float,
        default=5,
        help="If positive, build a dense adaptive ladder using this beta step size.",
    )
    parser.add_argument(
        "--ba-max-outer",
        type=int,
        default=500,
        help="Maximum number of outer BA alternating updates per beta.",
    )
    parser.add_argument(
        "--ba-max-inner",
        type=int,
        default=50,
        help="Maximum number of inner encoder/marginal BA updates per outer step.",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=150,
        help="Compute budget measured in equivalent full-MDP Bellman sweeps.",
    )
    parser.add_argument(
        "--eval-interval",
        dest="eval_interval",
        type=int,
        default=1,
        help="Save one trace row every this many Bellman sweeps.",
    )
    parser.add_argument(
        "--save-policies",
        action="store_true",
        help="Save greedy policies at every evaluation checkpoint to policies.csv.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="If greater than one, fit fixed betas independently in parallel and trace fixed betas in parallel.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--precompute-metric-only",
        action="store_true",
        help="Build or load the Four-Rooms metric cache and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Four-Rooms experiment."""
    args = parse_args()
    beta_schedule = OUT.parse_beta_schedule(args.beta_schedule)
    adaptive_beta_schedule = OUT.build_dense_beta_schedule(
        beta_schedule,
        args.adaptive_beta_step,
    )
    output_dir = OUT.resolve_output_dir(
        args.output_dir,
        Path(__file__).resolve().parent.parent.parent / "results" / "four_rooms",
    )
    mdp = FRM.build_four_rooms_mdp(eta=args.eta, gamma=args.gamma)

    if args.precompute_metric_only:
        diagnostics = FRM.precompute_metric_cache(
            eta=args.eta,
            gamma=args.gamma,
            metric_kind=args.metric_kind,
            verbose=True,
        )
        print("Metric cache")
        print(f"  metric_kind: {diagnostics['metric_kind']}")
        print(f"  cache_hit: {bool(diagnostics['cache_hit'])}")
        print(f"  elapsed_seconds: {float(diagnostics['elapsed_seconds']):.1f}")
        pair_cache = diagnostics.get("pair_cache_path")
        if pair_cache is not None:
            print(f"  pair_cache: {Path(pair_cache)}")
        return

    distortion = FRM.load_distortion(
        mdp,
        metric_kind=args.metric_kind,
        verbose=True,
        num_workers=max(1, int(args.num_workers)),
    )

    core_run.run(
        mdp=mdp,
        distortion=distortion,
        abstract_alphabet_size=args.abstract_alphabet_size,
        beta_schedule=beta_schedule,
        adaptive_beta_schedule=adaptive_beta_schedule,
        abstraction_error_mode=args.abstraction_error_mode,
        ba_max_outer=args.ba_max_outer,
        ba_max_inner=args.ba_max_inner,
        max_sweeps=args.max_sweeps,
        eval_interval=args.eval_interval,
        save_policies=bool(args.save_policies),
        num_workers=max(1, int(args.num_workers)),
        abstraction_solver=str(args.abstraction_solver),
        output_dir=output_dir,
        summary_mutator=FRM.build_summary_mutator(
            eta=args.eta,
            gamma=args.gamma,
            metric_kind=args.metric_kind,
            beta_schedule=beta_schedule,
            adaptive_beta_schedule=adaptive_beta_schedule,
        ),
    )


if __name__ == "__main__":
    main()
