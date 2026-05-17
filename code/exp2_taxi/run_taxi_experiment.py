"""CLI wrapper for the Taxi experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core import run as core_run
from exp2_taxi import taxi_mdp
from core import output as OUT


def parse_args() -> argparse.Namespace:
    """Define the Taxi experiment command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Taxi experiment using the shared model-based pipeline."
        )
    )
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
        default="0.02,0.04,0.06,0.08,0.10",
    )
    parser.add_argument(
        "--adaptive-beta-step",
        type=float,
        default=0.0,
        help="If positive, build a dense adaptive ladder using this beta step size.",
    )
    parser.add_argument(
        "--ba-max-outer",
        type=int,
        default=10,
        help="Maximum number of outer BA alternating updates per beta.",
    )
    parser.add_argument(
        "--ba-max-inner",
        type=int,
        default=100,
        help="Maximum number of inner encoder/marginal BA updates per outer step.",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=100,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "results" / "taxi",
    )
    return parser.parse_args()


def make_summary_mutator(gamma: float, metric_kind: str):
    """Attach Taxi-specific metadata to the shared experiment summary."""
    distance = (
        "one-step Bellman-compatible state-action metric"
        if metric_kind == "one_step"
        else "fixed-point state-action bisimulation metric"
    )

    def mutate(summary_payload: dict[str, object]) -> None:
        config = summary_payload.setdefault("config", {})
        notes = summary_payload.setdefault("notes", {})
        if isinstance(config, dict):
            config["scenario"] = "taxi"
            config["abstraction"] = "sa"
            config["solver"] = "modelbased_state_action"
            config["environment"] = "taxi"
            config["grid_size"] = 5
            config["gamma"] = float(gamma)
            config["metric_kind"] = metric_kind
            config["distance"] = distance
            config["mu"] = "uniform over concrete state-action pairs"
            config["terminal_state"] = "single absorbing success state"
            config["reward_structure"] = {
                "step": -1.0,
                "illegal_pickup_or_dropoff": -10.0,
                "successful_dropoff": 20.0,
            }
        if isinstance(notes, dict):
            notes["metric"] = distance
        summary_payload["adaptive_trigger_metric"] = (
            "abstract_residual <= abstraction_error on the current beta stage"
        )
        summary_payload["abstraction_notes"] = (
            "Compression is performed over concrete state-action pairs, then planning "
            "proceeds directly in abstract Q-space via the deterministic-decoder "
            "specialization of Fbar_eta = L F Gamma."
        )

    return mutate


def main() -> None:
    """Run the Taxi experiment."""
    args = parse_args()
    beta_schedule = OUT.parse_beta_schedule(args.beta_schedule)
    adaptive_beta_schedule = OUT.build_dense_beta_schedule(
        beta_schedule,
        args.adaptive_beta_step,
    )
    output_dir = OUT.resolve_output_dir(
        args.output_dir,
        Path(__file__).resolve().parent.parent.parent / "results" / "taxi",
    )
    mdp = taxi_mdp.build_taxi_mdp(gamma=args.gamma)
    distortion = taxi_mdp.load_distortion(
        mdp,
        metric_kind=args.metric_kind,
        cache_dir=Path(__file__).resolve().parent / ".cache",
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
        summary_mutator=make_summary_mutator(args.gamma, args.metric_kind),
    )


if __name__ == "__main__":
    main()
