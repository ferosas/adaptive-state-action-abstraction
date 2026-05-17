"""Run the experiments used in the paper.

The default mode reproduces the paper-facing settings. The ``--quick`` mode is
only a smoke test: it lowers metrics, beta schedules, BA iterations, and sweep
budgets so users can verify that the code path works quickly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


TOY_EXPERIMENTS = {
    "four_rooms": {
        "runner": ROOT / "code" / "exp1_four_rooms" / "run_four_rooms_experiment.py",
        "postprocess": ROOT / "code" / "exp1_four_rooms" / "results.py",
        "figures": ROOT / "code" / "exp1_four_rooms" / "make_figures.py",
        "quick_args": [
            "--metric-kind",
            "one_step",
            "--beta-schedule",
            "0,5",
            "--adaptive-beta-step",
            "5",
            "--ba-max-outer",
            "3",
            "--ba-max-inner",
            "5",
            "--max-sweeps",
            "3",
        ],
    },
    "taxi": {
        "runner": ROOT / "code" / "exp2_taxi" / "run_taxi_experiment.py",
        "postprocess": ROOT / "code" / "exp2_taxi" / "results.py",
        "figures": ROOT / "code" / "exp2_taxi" / "make_figures.py",
        "quick_args": [
            "--metric-kind",
            "one_step",
            "--beta-schedule",
            "0.02,0.04",
            "--ba-max-outer",
            "2",
            "--ba-max-inner",
            "5",
            "--max-sweeps",
            "3",
        ],
    },
    "doorkey": {
        "runner": ROOT / "code" / "exp3_doorkey" / "run_doorkey_experiment.py",
        "postprocess": ROOT / "code" / "exp3_doorkey" / "results.py",
        "figures": ROOT / "code" / "exp3_doorkey" / "make_figures.py",
        "quick_args": [
            "--grid-size",
            "5",
            "--metric-kind",
            "one_step",
            "--beta-schedule",
            "6,7",
            "--adaptive-beta-step",
            "1",
            "--ba-max-outer",
            "2",
            "--ba-max-inner",
            "5",
            "--max-sweeps",
            "3",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce paper experiment outputs.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["all", "toy", "four_rooms", "taxi", "doorkey", "sysadmin"],
        default=["all"],
        help="Which experiment groups to run.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Root output directory. Defaults to results/ or results_quick/.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel worker count passed to experiment runners.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced smoke test instead of the paper configuration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun experiments even when summary/traces outputs already exist.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing toy run artifacts when summary.json and traces.csv exist.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG/PDF figure rendering after CSV postprocessing.",
    )
    parser.add_argument(
        "--no-paper-tables",
        action="store_true",
        help="Do not build compact paper_data CSV tables at the end.",
    )
    return parser.parse_args()


def selected_experiments(tokens: list[str]) -> set[str]:
    requested = set(tokens)
    if "all" in requested:
        return {"four_rooms", "taxi", "doorkey", "sysadmin"}
    selected: set[str] = set()
    if "toy" in requested:
        selected.update({"four_rooms", "taxi", "doorkey"})
    selected.update(name for name in requested if name in TOY_EXPERIMENTS)
    if "sysadmin" in requested:
        selected.add("sysadmin")
    return selected


def run_command(command: list[str]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n[reproduce] {printable}", flush=True)
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def root_relative(path: Path) -> str:
    """Return a stable path argument relative to the public-code root when possible."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def toy_artifacts_exist(output_dir: Path) -> bool:
    return (output_dir / "summary.json").exists() and (output_dir / "traces.csv").exists()


def run_toy_experiment(
    name: str,
    *,
    results_root: Path,
    num_workers: int,
    quick: bool,
    force: bool,
    skip_existing: bool,
    no_figures: bool,
) -> None:
    spec = TOY_EXPERIMENTS[name]
    output_dir = results_root / name
    if name == "four_rooms":
        output_dir = results_root / "four_rooms"

    if toy_artifacts_exist(output_dir) and skip_existing and not force:
        print(f"[reproduce] Reusing existing {name} artifacts in {output_dir}")
    else:
        command = [
            sys.executable,
            root_relative(spec["runner"]),
            "--output-dir",
            root_relative(output_dir),
            "--num-workers",
            str(max(1, int(num_workers))),
        ]
        if quick:
            command.extend(spec["quick_args"])
        run_command(command)

    final_results_dir = output_dir / "final_results"
    run_command(
        [
            sys.executable,
            root_relative(spec["postprocess"]),
            "--results-dir",
            root_relative(output_dir),
            "--output-dir",
            root_relative(final_results_dir),
        ]
    )
    if not no_figures:
        run_command(
            [
                sys.executable,
                root_relative(spec["figures"]),
                "--results-dir",
                root_relative(final_results_dir),
                "--output-dir",
                root_relative(output_dir / "figures"),
            ]
        )


def run_sysadmin(
    *,
    results_root: Path,
    num_workers: int,
    quick: bool,
    force: bool,
    no_figures: bool,
) -> None:
    if quick:
        output_dir = results_root / "sysadmin_N2"
        command = [
            sys.executable,
            root_relative(ROOT / "code" / "exp4_sysadmin" / "run_sysadmin_experiment.py"),
            "--num-machines",
            "2",
            "--metric-kind",
            "one_step",
            "--beta-schedule",
            "0,1",
            "--adaptive-beta-step",
            "1",
            "--ba-max-outer",
            "3",
            "--ba-max-inner",
            "5",
            "--max-sweeps",
            "3",
            "--eval-interval",
            "1",
            "--output-dir",
            root_relative(output_dir),
            "--num-workers",
            str(max(1, int(num_workers))),
        ]
        run_command(command)
        return

    output_dir = results_root / "sysadmin_scaling"
    command = [
        sys.executable,
        root_relative(ROOT / "code" / "exp4_sysadmin" / "exp_scaling.py"),
        "--n-min",
        "2",
        "--n-max",
        "7",
        "--output-dir",
        root_relative(output_dir),
        "--metric-kind",
        "fixed_point",
        "--eval-interval",
        "1",
        "--num-workers",
        str(max(1, int(num_workers))),
    ]
    if force:
        command.append("--force")
    run_command(command)

    if not no_figures:
        run_command(
            [
                sys.executable,
                root_relative(ROOT / "code" / "exp4_sysadmin" / "make_scaling_figure.py"),
                "--results-dir",
                root_relative(output_dir),
                "--output-dir",
                root_relative(output_dir / "figures"),
                "--basename",
                "sysadmin_scaling_summary",
                "--n-min",
                "2",
                "--n-max",
                "7",
            ]
        )


def build_paper_tables(results_root: Path) -> None:
    run_command(
        [
            sys.executable,
            root_relative(ROOT / "scripts" / "build_paper_tables.py"),
            "--results-dir",
            root_relative(results_root),
            "--output-dir",
            "paper_data",
        ]
    )


def main() -> None:
    args = parse_args()
    results_root = args.results_dir
    if results_root is None:
        results_root = ROOT / ("results_quick" if args.quick else "results")
    else:
        results_root = results_root if results_root.is_absolute() else ROOT / results_root
    results_root.mkdir(parents=True, exist_ok=True)

    selected = selected_experiments(args.experiments)
    for name in ("four_rooms", "taxi", "doorkey"):
        if name in selected:
            run_toy_experiment(
                name,
                results_root=results_root,
                num_workers=args.num_workers,
                quick=args.quick,
                force=args.force,
                skip_existing=args.skip_existing,
                no_figures=args.no_figures,
            )
    if "sysadmin" in selected:
        run_sysadmin(
            results_root=results_root,
            num_workers=args.num_workers,
            quick=args.quick,
            force=args.force,
            no_figures=args.no_figures,
        )

    if not args.quick and not args.no_paper_tables:
        build_paper_tables(results_root)


if __name__ == "__main__":
    main()
