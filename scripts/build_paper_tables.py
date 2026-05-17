"""Collect experiment outputs into compact paper-facing CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

TOY_SCENARIOS = {
    "four_rooms": "Four Rooms",
    "taxi": "Taxi",
    "doorkey": "DoorKey",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact CSV tables for paper figures.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_data")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an expected experiment output is missing.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[paper-data] wrote {path}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def maybe_effective(value: float) -> float:
    return 2.0**value if math.isfinite(value) else math.nan


def relative_output_path(path_text: str, *, base: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        try:
            return str(path.resolve().relative_to(base.parent.resolve()))
        except Exception:
            return path.name


def load_optional(path: Path, *, strict: bool) -> list[dict[str, str]]:
    if path.exists():
        return read_rows(path)
    if strict:
        raise FileNotFoundError(path)
    print(f"[paper-data] skipping missing {path}")
    return []


def build_toy_tables(results_dir: Path, output_dir: Path, *, strict: bool) -> None:
    env_rows: list[dict[str, Any]] = []
    rd_rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []

    for scenario_key, scenario_label in TOY_SCENARIOS.items():
        scenario_dir = results_dir / scenario_key
        summary_path = scenario_dir / "summary.json"
        final_dir = scenario_dir / "final_results"
        if not summary_path.exists():
            if strict:
                raise FileNotFoundError(summary_path)
            print(f"[paper-data] skipping missing {summary_path}")
            continue

        summary = read_json(summary_path)
        config = dict(summary.get("config", {}))
        exact_returns = dict(summary.get("exact_policy_values", {}))
        adaptive_summary = dict(summary.get("adaptive_summary", {}))
        num_states = int(config.get("num_states", 0))
        num_actions = int(config.get("num_actions", 0))
        num_pairs = int(config.get("num_state_action_pairs", 0))
        base_return = float(exact_returns.get("Base MDP", math.nan))
        adaptive_return = float(exact_returns.get("Adaptive", math.nan))
        if not math.isfinite(adaptive_return):
            adaptive_return = float(adaptive_summary.get("final_policy_return", math.nan))

        table_rows = load_optional(final_dir / "report_figure0.csv", strict=strict)
        selected_beta = float(adaptive_summary.get("final_beta", math.nan))
        selected_rate_row = (
            min(table_rows, key=lambda row: abs(as_float(row, "beta") - selected_beta))
            if table_rows and math.isfinite(selected_beta)
            else {}
        )
        state_rate = as_float(selected_rate_row, "bar_state_information")
        action_rate = as_float(selected_rate_row, "bar_conditional_action_information")
        state_effective_size = maybe_effective(state_rate)
        action_effective_size = maybe_effective(action_rate)
        effective_pairs = as_float(selected_rate_row, "effective_abstraction_size")

        env_rows.append(
            {
                "scenario": scenario_key,
                "label": scenario_label,
                "num_states": num_states,
                "num_actions": num_actions,
                "num_state_action_pairs": num_pairs,
                "metric_kind": config.get("metric_kind", ""),
                "max_sweeps": config.get("max_sweeps", ""),
                "base_return": base_return,
                "adaptive_return": adaptive_return,
                "adaptive_return_fraction": (
                    adaptive_return / base_return
                    if math.isfinite(base_return) and abs(base_return) > 0.0
                    else math.nan
                ),
                "selected_beta": selected_beta,
                "selected_effective_state_action_fraction": (
                    effective_pairs / float(num_pairs)
                    if math.isfinite(effective_pairs) and num_pairs > 0
                    else math.nan
                ),
                "selected_effective_state_fraction": (
                    state_effective_size / float(num_states)
                    if math.isfinite(state_effective_size) and num_states > 0
                    else math.nan
                ),
                "selected_effective_action_fraction": (
                    action_effective_size / float(num_actions)
                    if math.isfinite(action_effective_size) and num_actions > 0
                    else math.nan
                ),
                "selected_abstraction_error": as_float(
                    selected_rate_row, "abstraction_error"
                ),
            }
        )

        for row in table_rows:
            state_rate = as_float(row, "bar_state_information")
            action_rate = as_float(row, "bar_conditional_action_information")
            state_effective_size = maybe_effective(state_rate)
            action_effective_size = maybe_effective(action_rate)
            out = {
                "scenario": scenario_key,
                "label": scenario_label,
                "beta": as_float(row, "beta"),
                "abstraction_error": as_float(row, "abstraction_error"),
                "mutual_information": as_float(row, "mutual_information"),
                "effective_abstraction_size": as_float(row, "effective_abstraction_size"),
                "normalized_effective_abstraction_size": as_float(
                    row, "normalized_effective_abstraction_size"
                ),
                "state_rate_bits": state_rate,
                "action_rate_bits": action_rate,
                "state_effective_size": state_effective_size,
                "action_effective_size": action_effective_size,
                "state_effective_fraction": (
                    state_effective_size / float(num_states)
                    if math.isfinite(state_effective_size) and num_states > 0
                    else math.nan
                ),
                "action_effective_fraction": (
                    action_effective_size / float(num_actions)
                    if math.isfinite(action_effective_size) and num_actions > 0
                    else math.nan
                ),
                "num_state_action_pairs": num_pairs,
            }
            rd_rows.append(out)

        for row in load_optional(final_dir / "report_figure1.csv", strict=strict):
            beta_rows.append(
                {
                    "scenario": scenario_key,
                    "label": scenario_label,
                    "sweep": as_float(row, "sweep"),
                    "beta": as_float(row, "beta"),
                    "stage_index": row.get("stage_index", ""),
                    "abstract_residual": row.get("abstract_residual", ""),
                    "abstraction_error": as_float(row, "abstraction_error"),
                }
            )

        for row in load_optional(final_dir / "report_figure3.csv", strict=strict):
            performance_rows.append(
                {
                    "scenario": scenario_key,
                    "label": scenario_label,
                    "method_label": row.get("method_label", ""),
                    "method_type": row.get("method_type", ""),
                    "beta": as_float(row, "beta"),
                    "sweep": as_float(row, "sweep"),
                    "policy_return": as_float(row, "policy_return"),
                    "optimal_mean_value": as_float(row, "optimal_mean_value", base_return),
                }
            )

    if env_rows:
        write_rows(output_dir / "fig1_environment_metadata.csv", env_rows)
    if rd_rows:
        write_rows(output_dir / "fig1_rate_distortion_frontiers.csv", rd_rows)
    if beta_rows:
        write_rows(output_dir / "fig1_adaptive_beta_schedule.csv", beta_rows)
    if performance_rows:
        write_rows(output_dir / "fig1_policy_return.csv", performance_rows)


def build_sysadmin_tables(results_dir: Path, output_dir: Path, *, strict: bool) -> None:
    scaling_dir = results_dir / "sysadmin_scaling"
    rd_path = scaling_dir / "rate_distortion.csv"
    near_path = scaling_dir / "adaptive_near_optimal.csv"
    summary_path = scaling_dir / "scaling_summary.json"
    if not rd_path.exists() or not near_path.exists():
        if strict:
            missing = rd_path if not rd_path.exists() else near_path
            raise FileNotFoundError(missing)
        print(f"[paper-data] skipping missing SysAdmin scaling outputs in {scaling_dir}")
        return

    rd_rows = read_rows(rd_path)
    near_rows = read_rows(near_path)
    full_size_by_n = {
        int(float(row["num_machines"])): as_float(row, "num_state_action_pairs")
        for row in rd_rows
        if row.get("num_machines") not in (None, "")
    }

    out_rd: list[dict[str, Any]] = []
    for row in rd_rows:
        num_machines = int(float(row["num_machines"]))
        full_size = full_size_by_n.get(num_machines, (num_machines + 1) * (2**num_machines))
        effective_rate_size = as_float(row, "effective_rate_size")
        out = dict(row)
        out["scenario"] = "sysadmin_scaling"
        out["full_state_action_pairs"] = full_size
        out["total_effective_fraction"] = (
            effective_rate_size / float(full_size)
            if math.isfinite(effective_rate_size) and full_size > 0
            else math.nan
        )
        out_rd.append(out)

    selected_rows: list[dict[str, Any]] = []
    for row in near_rows:
        num_machines = int(float(row["num_machines"]))
        full_size = full_size_by_n.get(num_machines, (num_machines + 1) * (2**num_machines))
        effective_rate_size = as_float(row, "effective_rate_size")
        out = dict(row)
        out["scenario"] = "sysadmin_scaling"
        out["run_dir"] = relative_output_path(str(row.get("run_dir", "")), base=ROOT)
        out["full_state_action_pairs"] = full_size
        out["total_effective_fraction"] = (
            effective_rate_size / float(full_size)
            if math.isfinite(effective_rate_size) and full_size > 0
            else math.nan
        )
        out["state_effective_fraction"] = as_float(row, "normalized_state_effective_size")
        out["action_effective_fraction"] = as_float(row, "normalized_action_effective_size")
        out["selected_distortion"] = as_float(row, "distortion")
        selected_rows.append(out)

    write_rows(output_dir / "fig2_rate_distortion_frontiers.csv", out_rd)
    write_rows(output_dir / "fig2_selected_compression_distortion.csv", selected_rows)

    if summary_path.exists():
        summary = read_json(summary_path)
        config = dict(summary.get("config", {}))
        write_rows(
            output_dir / "fig2_topology_metadata.csv",
            [
                {
                    "scenario": "sysadmin_scaling",
                    "n_min": config.get("n_min", ""),
                    "n_max": config.get("n_max", ""),
                    "near_optimal_fraction": config.get("near_optimal_fraction", ""),
                    "metric_kind": config.get("metric_kind_override", ""),
                    "eval_interval": config.get("eval_interval_override", ""),
                    "num_workers": config.get("num_workers_override", ""),
                    "source": relative_output_path(str(summary_path), base=ROOT),
                }
            ],
        )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    build_toy_tables(results_dir, output_dir, strict=args.strict)
    build_sysadmin_tables(results_dir, output_dir, strict=args.strict)


if __name__ == "__main__":
    main()

