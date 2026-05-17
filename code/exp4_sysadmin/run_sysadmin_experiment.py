"""CLI wrapper for the SysAdmin state-action model-based experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core import output as OUT
from core import run as core_run
from exp4_sysadmin import sysadmin_mdp


def parse_args() -> argparse.Namespace:
    """Define the SysAdmin experiment command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the SysAdmin state-action model-based experiment."
    )
    parser.add_argument("--num-machines", type=int, default=4)
    parser.add_argument("--p-base", type=float, default=0.95)
    parser.add_argument("--neighbor-penalty", type=float, default=0.15)
    parser.add_argument("--p-recover", type=float, default=0.05)
    parser.add_argument("--p-reboot", type=float, default=0.95)
    parser.add_argument("--reboot-cost", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--metric-kind",
        choices=["one_step", "fixed_point"],
        default="fixed_point",
        help="Which concrete pair metric to use. The fixed-point variant is much slower.",
    )
    parser.add_argument(
        "--abstract-alphabet-size",
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
        default="0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"
    )
    parser.add_argument(
        "--adaptive-beta-step",
        type=float,
        default=0.1,
        help="If positive, build a dense adaptive ladder using this beta step size.",
    )
    parser.add_argument(
        "--ba-max-outer",
        type=int,
        default=200,
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
        default=100,
        help="Compute budget measured in equivalent full-MDP Bellman sweeps.",
    )
    parser.add_argument(
        "--eval-interval",
        dest="eval_interval",
        type=int,
        default=10,
        help="Save one trace row every this many Bellman sweeps.",
    )
    parser.add_argument(
        "--save-policies",
        action="store_true",
        help=(
            "Save greedy policies at every evaluation checkpoint to policies.csv "
            "and persist the matching abstraction encoders under policy_encoders/."
        ),
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
        default=Path(__file__).resolve().parent.parent.parent / "results" / "sysadmin",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs during metric construction and experiment execution.",
    )
    return parser.parse_args()


def make_summary_mutator(args: argparse.Namespace):
    """Attach SysAdmin-specific metadata to the shared experiment summary."""
    distance = (
        "one-step Bellman-compatible state-action distortion"
        if args.metric_kind == "one_step"
        else "fixed-point state-action bisimulation metric"
    )

    def mutate(summary_payload: dict[str, object]) -> None:
        config = summary_payload.setdefault("config", {})
        notes = summary_payload.setdefault("notes", {})
        if isinstance(config, dict):
            config["scenario"] = "sysadmin"
            config["abstraction"] = "sa"
            config["solver"] = "modelbased_state_action"
            config["environment"] = "ring_sysadmin"
            config["num_machines"] = int(args.num_machines)
            config["gamma"] = float(args.gamma)
            config["p_base"] = float(args.p_base)
            config["neighbor_penalty"] = float(args.neighbor_penalty)
            config["p_recover"] = float(args.p_recover)
            config["p_reboot"] = float(args.p_reboot)
            config["reboot_cost"] = float(args.reboot_cost)
            config["metric_kind"] = args.metric_kind
            config["distance"] = distance
            config["mu"] = "uniform over concrete state-action pairs"
            config["reward_structure"] = {
                "alive_fraction": "mean fraction of machines currently up",
                "reboot_penalty": float(args.reboot_cost),
            }
        if isinstance(notes, dict):
            notes["metric"] = distance
            notes["topology"] = "ring interaction graph with local neighbor-dependent failures"
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
    """Run the SysAdmin experiment."""
    args = parse_args()
    beta_schedule = OUT.parse_beta_schedule(args.beta_schedule)
    adaptive_beta_schedule = OUT.build_dense_beta_schedule(
        beta_schedule,
        args.adaptive_beta_step,
    )

    mdp = sysadmin_mdp.build_ring_sysadmin_mdp(
        num_machines=args.num_machines,
        p_base=args.p_base,
        neighbor_penalty=args.neighbor_penalty,
        p_recover=args.p_recover,
        p_reboot=args.p_reboot,
        reboot_cost=args.reboot_cost,
        gamma=args.gamma,
    )
    distortion = sysadmin_mdp.load_distortion(
        mdp,
        metric_kind=args.metric_kind,
        verbose=not args.quiet,
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
        output_dir=args.output_dir,
        summary_mutator=make_summary_mutator(args),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
