"""Shared helpers for post-hoc analysis scripts that read experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from core import planning as PL
from core import output as OUT
from exp1_four_rooms import four_rooms_mdp
from exp2_taxi import taxi_mdp
from exp3_doorkey import doorkey_mdp
from exp4_sysadmin import sysadmin_mdp


def parse_results_dir(description: str) -> argparse.Namespace:
    """Parse a single required results-directory argument."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    """Load a CSV trace or table into a list of dictionaries."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    return add_policy_return_alias(add_abstraction_error_alias(rows))


def add_abstraction_error_alias(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Backfill `abstraction_error` from older `distortion` exports when needed."""
    aliased_rows: List[Dict[str, object]] = []
    for row in rows:
        aliased = dict(row)
        if aliased.get("abstraction_error") in (None, "") and aliased.get("distortion") not in (
            None,
            "",
        ):
            aliased["abstraction_error"] = aliased["distortion"]
        aliased_rows.append(aliased)
    return aliased_rows


def add_policy_return_alias(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Backfill `policy_return` from older `mc_return` exports when needed."""
    aliased_rows: List[Dict[str, object]] = []
    for row in rows:
        aliased = dict(row)
        if aliased.get("policy_return") in (None, "") and aliased.get("mc_return") not in (
            None,
            "",
        ):
            aliased["policy_return"] = aliased["mc_return"]
        aliased_rows.append(aliased)
    return aliased_rows


def load_summary(path: Path) -> Dict[str, object]:
    """Load a JSON experiment summary."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    adaptive_summary = summary.get("adaptive_summary")
    if isinstance(adaptive_summary, dict):
        if adaptive_summary.get("final_policy_return") in (None, "") and adaptive_summary.get(
            "final_mc_return"
        ) not in (None, ""):
            adaptive_summary["final_policy_return"] = adaptive_summary["final_mc_return"]
    return summary


def load_saved_fixed_family_table(summary: Dict[str, object]) -> List[Dict[str, object]]:
    """Return saved fixed-family rate-distortion rows when present."""
    rows = summary.get("fixed_family_table")
    if not isinstance(rows, list):
        return []
    return add_abstraction_error_alias(
        [dict(row) for row in rows if isinstance(row, dict)]
    )


def add_normalized_bellman_compute(
    rows: Sequence[Dict[str, object]],
    summary: Dict[str, object],
) -> List[Dict[str, object]]:
    """Derive equivalent full-MDP sweeps from transition-summation Bellman costs.

    Saved `bellman_backup_units` count Bellman backup entries, e.g. one unit per
    concrete or abstract state-action update. To express compute in terms of the
    dense tabular transition-summation cost, we multiply both the numerator and
    the base full-MDP sweep cost by `|S|`. This makes the convention explicit
    while preserving the historical normalization values.
    """
    config = summary.get("config", {})
    if not isinstance(config, dict):
        return [dict(row) for row in rows]

    base_backup_units = config.get("num_state_action_pairs")
    num_states = config.get("num_states")
    if base_backup_units in (None, ""):
        num_actions = config.get("num_actions")
        if num_states in (None, "") or num_actions in (None, ""):
            return [dict(row) for row in rows]
        base_backup_units = int(num_states) * int(num_actions)
    if num_states in (None, ""):
        return [dict(row) for row in rows]

    base_transition_units = float(base_backup_units) * float(num_states)
    if base_transition_units <= 0.0:
        return [dict(row) for row in rows]

    augmented_rows: List[Dict[str, object]] = []
    for row in rows:
        augmented = dict(row)
        if augmented.get("normalized_bellman_compute") in (None, ""):
            bellman_backup_units = augmented.get("bellman_backup_units")
            if bellman_backup_units not in (None, ""):
                transition_units = float(bellman_backup_units) * float(num_states)
                augmented["normalized_bellman_compute"] = (
                    transition_units / base_transition_units
                )
        augmented_rows.append(augmented)
    return augmented_rows


def save_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write rows using the shared CSV helper."""
    OUT.save_rows(path, rows)


def compute_optimal_mean_value(summary: Dict[str, object]) -> float:
    """Recompute the optimal mean value from the saved experiment config."""
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("Expected summary['config'] to be a dictionary.")

    scenario = config.get("scenario")
    if scenario == "chain":
        raise ValueError("This code bundle does not include the legacy chain experiment.")
    elif scenario == "four_rooms":
        mdp = four_rooms_mdp.build_four_rooms_mdp(
            eta=float(config["slip_probability"]),
            gamma=float(config["gamma"]),
        )
    elif scenario == "taxi":
        mdp = taxi_mdp.build_taxi_mdp(gamma=float(config["gamma"]))
    elif scenario == "sysadmin":
        mdp = sysadmin_mdp.build_ring_sysadmin_mdp(
            num_machines=int(config["num_machines"]),
            p_base=float(config["p_base"]),
            neighbor_penalty=float(config["neighbor_penalty"]),
            p_recover=float(config["p_recover"]),
            p_reboot=float(config["p_reboot"]),
            reboot_cost=float(config["reboot_cost"]),
            gamma=float(config["gamma"]),
        )
    elif scenario == "doorkey":
        mdp = doorkey_mdp.build_doorkey_mdp(
            grid_size=int(config["grid_size"]),
            gamma=float(config["gamma"]),
            goal_reward=float(config.get("goal_reward", 1.0)),
        )
    else:
        raise ValueError(f"Unsupported scenario for optimal mean value: {scenario!r}")

    optimal_values = PL.solve_optimal_values(mdp)
    return float(optimal_values.mean())
