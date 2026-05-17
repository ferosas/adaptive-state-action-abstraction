"""Run and aggregate SysAdmin scaling experiments across machine counts.

The script is intentionally a thin driver around ``run_sysadmin_experiment.py``:
each individual run is produced by the existing experiment CLI, while this file
only varies ``--num-machines`` and postprocesses the saved artifacts into compact
scaling tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = REPO_DIR / "results" / "sysadmin_scaling"
RUNNER = SCRIPT_DIR / "run_sysadmin_experiment.py"
PROJECTED_INFORMATION_SPLIT = "I(S,A;Sbar)+I(S,A;Abar|Sbar)"


def parse_args() -> argparse.Namespace:
    """Define the scaling-driver command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run SysAdmin state-action experiments for N_min,...,N_max and "
            "aggregate normalized rate-distortion and adaptive-threshold data."
        )
    )
    parser.add_argument("--n-min", type=int, default=4)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--near-optimal-fraction",
        type=float,
        default=0.99,
        help="Adaptive return threshold as a fraction of the best saved exact policy return.",
    )
    parser.add_argument(
        "--metric-kind",
        choices=["one_step", "fixed_point"],
        default="fixed_point",
        help=(
            "Optional override passed to run_sysadmin_experiment.py. If omitted, "
            "the runner's default metric is used."
        ),
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help=(
            "Optional override for the runner's evaluation interval. Use 1 to "
            "identify the first near-optimal adaptive checkpoint exactly."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional parallel-worker override passed to the runner.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun an N even if summary/traces/policies artifacts already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the runner commands without executing them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass --quiet to each SysAdmin run.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write dictionaries to CSV using a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: Any, default: float = math.nan) -> float:
    """Parse a floating-point value, returning ``default`` for missing entries."""
    if value is None or value == "":
        return default
    return float(value)


def as_int(value: Any, default: int = 0) -> int:
    """Parse an integer value from the runner's CSV/JSON artifacts."""
    if value is None or value == "":
        return default
    return int(float(value))


def build_runner_command(args: argparse.Namespace, num_machines: int, run_dir: Path) -> List[str]:
    """Build the command used for one concrete SysAdmin run."""
    command = [
        sys.executable,
        str(RUNNER),
        "--num-machines",
        str(int(num_machines)),
        "--output-dir",
        str(run_dir),
        "--save-policies",
    ]
    if args.metric_kind is not None:
        command.extend(["--metric-kind", str(args.metric_kind)])
    if args.eval_interval is not None:
        command.extend(["--eval-interval", str(int(args.eval_interval))])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(int(args.num_workers))])
    if args.quiet:
        command.append("--quiet")
    return command


def run_artifacts_exist(run_dir: Path) -> bool:
    """Return whether one run already has all artifacts needed for aggregation."""
    return all(
        (run_dir / filename).exists()
        for filename in ("summary.json", "traces.csv", "policies.csv")
    )


def run_one_experiment(args: argparse.Namespace, num_machines: int) -> Path:
    """Run, or reuse, the SysAdmin experiment for one machine count."""
    run_dir = args.output_dir / f"N{int(num_machines)}"
    command = build_runner_command(args, num_machines, run_dir)
    if run_artifacts_exist(run_dir) and not args.force:
        print(f"[scaling] Reusing existing artifacts for N={num_machines}: {run_dir}")
        return run_dir

    print(f"[scaling] Running N={num_machines}")
    print("[scaling] " + " ".join(command))
    if args.dry_run:
        return run_dir
    subprocess.run(command, check=True, cwd=REPO_DIR)
    return run_dir


def fixed_family_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the runner's beta/rate/distortion table."""
    rows = summary.get("fixed_family_table")
    if isinstance(rows, list):
        return [dict(row) for row in rows]
    return []


def normalized_rate_distortion_rows(
    *,
    num_machines: int,
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build normalized rate-distortion rows for one run."""
    config = dict(summary.get("config", {}))
    max_distance = float(summary.get("metric_summary", {}).get("max_state_action_distance", 0.0))
    scale = max_distance if max_distance > 0.0 else 1.0

    rows: List[Dict[str, Any]] = []
    for row in fixed_family_rows(summary):
        raw_rate_bits = as_float(row.get("mutual_information"))
        has_projected_split = row.get("bar_information_split") == PROJECTED_INFORMATION_SPLIT
        projected_rate_bits = (
            as_float(row.get("bar_information_sum")) if has_projected_split else math.nan
        )
        state_rate_bits = (
            as_float(row.get("bar_state_information")) if has_projected_split else math.nan
        )
        action_rate_bits = (
            as_float(row.get("bar_conditional_action_information"))
            if has_projected_split
            else math.nan
        )
        rate_bits = projected_rate_bits if math.isfinite(projected_rate_bits) else raw_rate_bits
        num_states = as_int(config.get("num_states"))
        num_actions = as_int(config.get("num_actions"))
        state_effective_size = 2.0**state_rate_bits if math.isfinite(state_rate_bits) else math.nan
        action_effective_size = (
            2.0**action_rate_bits if math.isfinite(action_rate_bits) else math.nan
        )
        raw_distortion = as_float(row.get("abstraction_error"))
        normalized_distortion = raw_distortion / scale
        rows.append(
            {
                "num_machines": int(num_machines),
                "num_states": num_states,
                "num_actions": num_actions,
                "num_state_action_pairs": as_int(config.get("num_state_action_pairs")),
                "beta": as_float(row.get("beta")),
                "rate_bits": rate_bits,
                "raw_rate_bits": raw_rate_bits,
                "state_rate_bits": state_rate_bits,
                "action_rate_bits": action_rate_bits,
                "effective_rate_size": 2.0**rate_bits if math.isfinite(rate_bits) else math.nan,
                "state_effective_size": state_effective_size,
                "action_effective_size": action_effective_size,
                "normalized_state_effective_size": (
                    state_effective_size / float(num_states)
                    if math.isfinite(state_effective_size) and num_states > 0
                    else math.nan
                ),
                "normalized_action_effective_size": (
                    action_effective_size / float(num_actions)
                    if math.isfinite(action_effective_size) and num_actions > 0
                    else math.nan
                ),
                "distortion": normalized_distortion,
                "normalized_distortion": normalized_distortion,
                "raw_distortion": raw_distortion,
                "max_pair_distance": max_distance,
                "effective_abstract_pairs": as_int(row.get("effective_abstract_pairs")),
                "information_split": (
                    PROJECTED_INFORMATION_SPLIT if has_projected_split else "raw_code"
                ),
                "solver_kind": row.get("solver_kind", config.get("abstraction_solver", "")),
            }
        )
    return rows


def reference_return(summary: Dict[str, Any]) -> float:
    """Return the saved reference return used to define near-optimality."""
    values = summary.get("exact_policy_values", {})
    if not isinstance(values, dict) or not values:
        return math.nan
    if "Base MDP" in values:
        return float(values["Base MDP"])
    return max(float(value) for value in values.values())


def first_near_optimal_adaptive_row(
    *,
    traces: List[Dict[str, str]],
    optimal_return: float,
    near_optimal_fraction: float,
) -> Dict[str, str] | None:
    """Find the first adaptive checkpoint whose return reaches the threshold."""
    threshold = float(near_optimal_fraction) * float(optimal_return)
    adaptive_rows = [
        row
        for row in traces
        if row.get("method_type") == "adaptive" or row.get("method_label") == "Adaptive"
    ]
    adaptive_rows.sort(
        key=lambda row: (
            as_float(row.get("bellman_backup_units"), math.inf),
            as_float(row.get("sweep"), math.inf),
        )
    )
    for row in adaptive_rows:
        if as_float(row.get("policy_return"), -math.inf) >= threshold:
            return row
    return None


def match_policy_row(
    *,
    policies: List[Dict[str, str]],
    trace_row: Dict[str, str],
) -> Dict[str, str] | None:
    """Find the saved policy corresponding to a selected trace row."""
    target_sweep = as_float(trace_row.get("sweep"))
    target_beta = as_float(trace_row.get("beta"))
    target_stage = as_float(trace_row.get("stage_index"))
    target_compute = as_float(trace_row.get("bellman_backup_units"))

    best: Dict[str, str] | None = None
    best_score = math.inf
    for row in policies:
        if not (row.get("method_type") == "adaptive" or row.get("method_label") == "Adaptive"):
            continue
        score = (
            abs(as_float(row.get("sweep")) - target_sweep)
            + abs(as_float(row.get("beta")) - target_beta)
            + abs(as_float(row.get("bellman_backup_units")) - target_compute)
        )
        row_stage = as_float(row.get("stage_index"))
        if math.isfinite(target_stage) and math.isfinite(row_stage):
            score += abs(row_stage - target_stage)
        if score < best_score:
            best = row
            best_score = score
    return best


def row_for_beta(rd_rows: List[Dict[str, Any]], beta: float) -> Dict[str, Any] | None:
    """Return the closest rate-distortion row to a selected beta."""
    if not rd_rows:
        return None
    return min(rd_rows, key=lambda row: abs(float(row["beta"]) - float(beta)))


def aggregate_one_run(
    *,
    run_dir: Path,
    num_machines: int,
    near_optimal_fraction: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Aggregate one run into RD rows, a threshold row, and a policy payload."""
    summary = json.loads((run_dir / "summary.json").read_text())
    traces = read_csv_rows(run_dir / "traces.csv")
    policies = read_csv_rows(run_dir / "policies.csv")
    rd_rows = normalized_rate_distortion_rows(num_machines=num_machines, summary=summary)

    optimal_return = reference_return(summary)
    if not math.isfinite(optimal_return):
        optimal_return = max(as_float(row.get("policy_return"), -math.inf) for row in traces)
    threshold = near_optimal_fraction * optimal_return
    near_trace = first_near_optimal_adaptive_row(
        traces=traces,
        optimal_return=optimal_return,
        near_optimal_fraction=near_optimal_fraction,
    )
    if near_trace is None:
        near_row = {
            "num_machines": int(num_machines),
            "run_dir": str(run_dir),
            "near_optimal_fraction": float(near_optimal_fraction),
            "optimal_return": float(optimal_return),
            "return_threshold": float(threshold),
            "reached": False,
        }
        return rd_rows, near_row, {"num_machines": int(num_machines), "reached": False}

    beta = as_float(near_trace.get("beta"))
    selected_rd = row_for_beta(rd_rows, beta) or {}
    policy_row = match_policy_row(policies=policies, trace_row=near_trace)
    policy_json = policy_row.get("policy_json", "[]") if policy_row is not None else "[]"
    policy = json.loads(policy_json)

    near_row = {
        "num_machines": int(num_machines),
        "run_dir": str(run_dir),
        "near_optimal_fraction": float(near_optimal_fraction),
        "optimal_return": float(optimal_return),
        "return_threshold": float(threshold),
        "reached": True,
        "sweep": as_float(near_trace.get("sweep")),
        "bellman_backup_units": as_float(near_trace.get("bellman_backup_units")),
        "stage_index": as_float(near_trace.get("stage_index")),
        "policy_return": as_float(near_trace.get("policy_return")),
        "beta": beta,
        "rate_bits": selected_rd.get("rate_bits", math.nan),
        "raw_rate_bits": selected_rd.get("raw_rate_bits", math.nan),
        "state_rate_bits": selected_rd.get("state_rate_bits", math.nan),
        "action_rate_bits": selected_rd.get("action_rate_bits", math.nan),
        "effective_rate_size": selected_rd.get("effective_rate_size", math.nan),
        "state_effective_size": selected_rd.get("state_effective_size", math.nan),
        "action_effective_size": selected_rd.get("action_effective_size", math.nan),
        "normalized_state_effective_size": selected_rd.get(
            "normalized_state_effective_size", math.nan
        ),
        "normalized_action_effective_size": selected_rd.get(
            "normalized_action_effective_size", math.nan
        ),
        "distortion": selected_rd.get("distortion", math.nan),
        "normalized_distortion": selected_rd.get("normalized_distortion", math.nan),
        "raw_distortion": selected_rd.get(
            "raw_distortion", as_float(near_trace.get("abstraction_error"))
        ),
        "max_pair_distance": selected_rd.get("max_pair_distance", math.nan),
        "effective_abstract_pairs": selected_rd.get("effective_abstract_pairs", math.nan),
        "information_split": selected_rd.get("information_split", ""),
        "policy_length": len(policy),
    }
    policy_payload = {
        "num_machines": int(num_machines),
        "run_dir": str(run_dir),
        "near_optimal_fraction": float(near_optimal_fraction),
        "optimal_return": float(optimal_return),
        "return_threshold": float(threshold),
        "sweep": near_row["sweep"],
        "bellman_backup_units": near_row["bellman_backup_units"],
        "beta": beta,
        "rate_bits": near_row["rate_bits"],
        "raw_rate_bits": near_row["raw_rate_bits"],
        "state_rate_bits": near_row["state_rate_bits"],
        "action_rate_bits": near_row["action_rate_bits"],
        "distortion": near_row["distortion"],
        "normalized_distortion": near_row["normalized_distortion"],
        "raw_distortion": near_row["raw_distortion"],
        "policy": policy,
    }
    return rd_rows, near_row, policy_payload


def write_aggregate_outputs(
    *,
    output_dir: Path,
    rd_rows: List[Dict[str, Any]],
    near_rows: List[Dict[str, Any]],
    policy_payloads: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write the scaling CSV/JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "rate_distortion.csv",
        rd_rows,
        [
            "num_machines",
            "num_states",
            "num_actions",
            "num_state_action_pairs",
            "beta",
            "rate_bits",
            "raw_rate_bits",
            "state_rate_bits",
            "action_rate_bits",
            "effective_rate_size",
            "state_effective_size",
            "action_effective_size",
            "normalized_state_effective_size",
            "normalized_action_effective_size",
            "distortion",
            "normalized_distortion",
            "raw_distortion",
            "max_pair_distance",
            "effective_abstract_pairs",
            "information_split",
            "solver_kind",
        ],
    )
    write_csv_rows(
        output_dir / "adaptive_near_optimal.csv",
        near_rows,
        [
            "num_machines",
            "run_dir",
            "near_optimal_fraction",
            "optimal_return",
            "return_threshold",
            "reached",
            "sweep",
            "bellman_backup_units",
            "stage_index",
            "policy_return",
            "beta",
            "rate_bits",
            "raw_rate_bits",
            "state_rate_bits",
            "action_rate_bits",
            "effective_rate_size",
            "state_effective_size",
            "action_effective_size",
            "normalized_state_effective_size",
            "normalized_action_effective_size",
            "distortion",
            "normalized_distortion",
            "raw_distortion",
            "max_pair_distance",
            "effective_abstract_pairs",
            "information_split",
            "policy_length",
        ],
    )
    write_csv_rows(
        output_dir / "near_optimal_policies.csv",
        [
            {
                "num_machines": payload.get("num_machines"),
                "beta": payload.get("beta"),
                "policy_json": json.dumps(payload.get("policy", []), separators=(",", ":")),
            }
            for payload in policy_payloads
        ],
        ["num_machines", "beta", "policy_json"],
    )
    summary = {
        "config": {
            "n_min": int(args.n_min),
            "n_max": int(args.n_max),
            "near_optimal_fraction": float(args.near_optimal_fraction),
            "metric_kind_override": args.metric_kind,
            "eval_interval_override": args.eval_interval,
            "num_workers_override": args.num_workers,
        },
        "outputs": {
            "rate_distortion": "rate_distortion.csv",
            "adaptive_near_optimal": "adaptive_near_optimal.csv",
            "near_optimal_policies_csv": "near_optimal_policies.csv",
            "near_optimal_policies_json": "near_optimal_policies.json",
        },
        "near_optimal": near_rows,
    }
    (output_dir / "near_optimal_policies.json").write_text(
        json.dumps(policy_payloads, indent=2) + "\n"
    )
    (output_dir / "scaling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    """Run the scaling sweep and write aggregate outputs."""
    args = parse_args()
    if args.n_min <= 0 or args.n_max < args.n_min:
        raise ValueError("Require 0 < n_min <= n_max.")
    if not (0.0 < args.near_optimal_fraction <= 1.0):
        raise ValueError("near_optimal_fraction must lie in (0, 1].")

    all_rd_rows: List[Dict[str, Any]] = []
    near_rows: List[Dict[str, Any]] = []
    policy_payloads: List[Dict[str, Any]] = []

    for num_machines in range(int(args.n_min), int(args.n_max) + 1):
        run_dir = run_one_experiment(args, num_machines)
        if args.dry_run:
            continue
        rd_rows, near_row, policy_payload = aggregate_one_run(
            run_dir=run_dir,
            num_machines=num_machines,
            near_optimal_fraction=float(args.near_optimal_fraction),
        )
        all_rd_rows.extend(rd_rows)
        near_rows.append(near_row)
        policy_payloads.append(policy_payload)

    if not args.dry_run:
        write_aggregate_outputs(
            output_dir=args.output_dir,
            rd_rows=all_rd_rows,
            near_rows=near_rows,
            policy_payloads=policy_payloads,
            args=args,
        )
        print(f"[scaling] Wrote aggregate outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
