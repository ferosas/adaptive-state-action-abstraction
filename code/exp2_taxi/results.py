"""Build report-style CSV tables from saved Taxi experiment results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis import common as AC
from core import abstraction as AB
from core import output as OUT
from exp2_taxi import taxi_mdp


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "taxi"


def parse_args() -> argparse.Namespace:
    """Define the CLI for Taxi report-table extraction."""
    parser = argparse.ArgumentParser(
        description="Build report-style CSV tables from saved Taxi results."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _as_float(row: Dict[str, object], key: str) -> float:
    """Read a numeric row field as float."""
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing numeric field {key!r} in row: {row}")
    return float(value)


def _as_float_any(row: Dict[str, object], *keys: str) -> float:
    """Read the first available numeric row field from a list of keys."""
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"Missing numeric fields {keys!r} in row: {row}")


def _sorted_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Sort rows in the common method/beta/sweep order used by the report."""
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("method_type", "")),
            _as_float(row, "beta") if row.get("beta") not in (None, "") else 0.0,
            _as_float(row, "sweep") if row.get("sweep") not in (None, "") else 0.0,
        ),
    )


def _rebuild_taxi_fixed_family_table(summary: Dict[str, object]) -> List[Dict[str, object]]:
    """Reconstruct the fixed abstraction family to recover rate-distortion data."""
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("Expected summary['config'] to be a dictionary.")

    mdp = taxi_mdp.build_taxi_mdp(gamma=float(config["gamma"]))
    distortion = taxi_mdp.load_distortion(
        mdp,
        metric_kind=str(config.get("metric_kind", "fixed_point")),
    )

    num_pairs = int(mdp.num_state_action_pairs)
    mu_uniform = np.full(num_pairs, 1.0 / float(num_pairs), dtype=float)
    beta_schedule = [float(beta) for beta in config["beta_schedule"]]
    abstract_alphabet_size = int(config.get("abstract_alphabet_size_cap", num_pairs))
    abstraction_error_mode = str(config.get("abstraction_error_mode", "average"))
    abstraction_solver = AB.normalize_solver_kind(str(config.get("abstraction_solver", "flat")))
    ba_solver_limits = config.get("ba_solver_limits", {})
    if not isinstance(ba_solver_limits, dict):
        ba_solver_limits = {}
    ba_max_outer = int(ba_solver_limits.get("max_outer", 10))
    ba_max_inner = int(ba_solver_limits.get("max_inner", 30))

    decoder = None
    encoder = None
    rows: List[Dict[str, object]] = []
    for beta in beta_schedule:
        abstraction = AB.fit_soft_abstraction(
            distortion=distortion,
            mu=mu_uniform,
            beta=beta,
            num_abstract=abstract_alphabet_size,
            decoder_init=decoder,
            encoder_init=encoder,
            max_outer=ba_max_outer,
            max_inner=ba_max_inner,
            tolerance=1e-6,
            solver_kind=abstraction_solver,
            num_actions=int(mdp.num_actions),
        )
        decoder = abstraction.full_decoder
        encoder = abstraction.full_encoder
        row = {
            "beta": float(beta),
            "effective_abstract_pairs": int(abstraction.num_abstract),
            "abstraction_error": AB.compute_abstraction_error(
                mu_uniform,
                abstraction.encoder,
                abstraction.decoder,
                distortion,
                mode=abstraction_error_mode,
            ),
            "mutual_information": AB.mutual_information(
                mu_uniform,
                abstraction.encoder,
            ),
        }
        row.update(
            AB.state_action_information_decomposition(
                mu_uniform,
                abstraction.encoder,
                abstraction.decoder,
                int(mdp.num_actions),
            )
        )
        rows.append(row)
    return rows


def build_figure0_rows(
    table1_rows: Sequence[Dict[str, object]],
    total_state_action_pairs: int,
) -> List[Dict[str, object]]:
    """Build the rate-distortion CSV used by Figure 0."""
    optional_information_keys = [
        "joint_code_state_information",
        "joint_code_conditional_action_information",
        "joint_code_information",
        "bar_state_information",
        "bar_conditional_action_information",
        "bar_information_sum",
    ]
    rows: List[Dict[str, object]] = []
    for row in sorted(table1_rows, key=lambda row: _as_float(row, "beta")):
        figure_row = {
            "beta": _as_float(row, "beta"),
            "mutual_information": _as_float(row, "mutual_information"),
            "abstraction_error": _as_float(row, "abstraction_error"),
            "effective_abstraction_size": 2.0 ** _as_float(row, "mutual_information"),
            "normalized_effective_abstraction_size": (
                (2.0 ** _as_float(row, "mutual_information")) / float(total_state_action_pairs)
            ),
            "total_state_action_pairs": int(total_state_action_pairs),
            "effective_abstract_pairs": int(round(_as_float(row, "effective_abstract_pairs"))),
        }
        for key in optional_information_keys:
            if row.get(key) not in (None, ""):
                figure_row[key] = _as_float(row, key)
        rows.append(figure_row)
    return rows


def build_figure1_rows(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Build Figure 1 data: adaptive beta-switching against Bellman update."""
    rows: List[Dict[str, object]] = []
    for row in trace_rows:
        if str(row.get("method_type", "")) != "adaptive":
            continue
        rows.append(
            {
                "method_label": str(row["method_label"]),
                "method_type": str(row["method_type"]),
                "sweep": _as_float(row, "sweep"),
                "beta": _as_float(row, "beta"),
                "stage_index": (
                    int(round(_as_float(row, "stage_index")))
                    if row.get("stage_index") not in (None, "")
                    else ""
                ),
                "abstract_residual": (
                    _as_float(row, "abstract_residual")
                    if row.get("abstract_residual") not in (None, "")
                    else ""
                ),
                "abstraction_error": _as_float(row, "abstraction_error"),
            }
        )
    return _sorted_rows(rows)


def build_figure2_rows(
    trace_rows: Sequence[Dict[str, object]],
    optimal_mean_value: float,
) -> List[Dict[str, object]]:
    """Build Figure 2 data: policy-value against normalized compute."""
    rows: List[Dict[str, object]] = []
    for row in trace_rows:
        if row.get("normalized_bellman_compute") in (None, ""):
            continue
        if row.get("policy_return") in (None, ""):
            continue
        rows.append(
            {
                "method_label": str(row["method_label"]),
                "method_type": str(row["method_type"]),
                "beta": _as_float(row, "beta"),
                "sweep": _as_float(row, "sweep"),
                "normalized_bellman_compute": _as_float(row, "normalized_bellman_compute"),
                "policy_return": _as_float(row, "policy_return"),
                "optimal_mean_value": float(optimal_mean_value),
            }
        )
    return _sorted_rows(rows)


def build_figure3_rows(
    trace_rows: Sequence[Dict[str, object]],
    optimal_mean_value: float,
) -> List[Dict[str, object]]:
    """Build Figure 3 data: policy-value against raw Bellman sweeps."""
    rows: List[Dict[str, object]] = []
    for row in trace_rows:
        if row.get("policy_return") in (None, ""):
            continue
        rows.append(
            {
                "method_label": str(row["method_label"]),
                "method_type": str(row["method_type"]),
                "beta": _as_float(row, "beta"),
                "sweep": _as_float(row, "sweep"),
                "policy_return": _as_float(row, "policy_return"),
                "optimal_mean_value": float(optimal_mean_value),
            }
        )
    return _sorted_rows(rows)


def build_figure4_rows(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Build the induced-value-error data against Bellman update."""
    rows: List[Dict[str, object]] = []
    for row in trace_rows:
        if str(row.get("method_type", "")) not in {"fixed", "base"}:
            continue
        if row.get("abstract_value_error") in (None, "") and row.get("abstract_q_error") in (None, ""):
            continue
        rows.append(
            {
                "method_label": str(row["method_label"]),
                "method_type": str(row["method_type"]),
                "beta": _as_float(row, "beta"),
                "sweep": _as_float(row, "sweep"),
                "abstract_value_error": _as_float_any(row, "abstract_value_error", "abstract_q_error"),
                "active_abstracts": int(round(_as_float(row, "active_abstracts"))),
                "abstraction_error": _as_float(row, "abstraction_error"),
            }
        )
    return _sorted_rows(rows)


def build_figure5_rows(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Build Figure 5 data: grounded-value-error against Bellman update."""
    rows: List[Dict[str, object]] = []
    for row in trace_rows:
        rows.append(
            {
                "method_label": str(row["method_label"]),
                "method_type": str(row["method_type"]),
                "beta": _as_float(row, "beta"),
                "sweep": _as_float(row, "sweep"),
                "concrete_value_error": _as_float_any(row, "concrete_value_error", "concrete_q_error"),
            }
        )
    return _sorted_rows(rows)


def build_report_csvs(results_dir: Path, output_dir: Path) -> None:
    """Build and save the Taxi report CSVs."""
    summary = AC.load_summary(results_dir / "summary.json")
    trace_rows = AC.load_rows(results_dir / "traces.csv")
    trace_rows = AC.add_normalized_bellman_compute(trace_rows, summary)
    config = summary.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Expected summary['config'] to be a dictionary.")
    total_state_action_pairs = int(config["num_state_action_pairs"])

    table1_rows = AC.load_saved_fixed_family_table(summary)
    if not table1_rows:
        table1_rows = _rebuild_taxi_fixed_family_table(summary)
    optimal_mean_value = AC.compute_optimal_mean_value(summary)

    figure0_rows = build_figure0_rows(table1_rows, total_state_action_pairs)
    for legacy_path in [
        output_dir / "report_figure8.csv",
        output_dir / "report_figure7.csv",
        output_dir / "report_figure6.csv",
        output_dir / "report_figure5.csv",
    ]:
        if legacy_path.exists():
            legacy_path.unlink()

    figure1_rows = build_figure1_rows(trace_rows)
    figure2_rows = build_figure2_rows(trace_rows, optimal_mean_value)
    figure3_rows = build_figure3_rows(trace_rows, optimal_mean_value)
    figure4_rows = build_figure4_rows(trace_rows)
    figure5_rows = build_figure5_rows(trace_rows)

    OUT.save_rows(output_dir / "report_table1.csv", table1_rows)
    OUT.save_rows(output_dir / "report_figure0.csv", figure0_rows)
    OUT.save_rows(output_dir / "report_figure1.csv", figure1_rows)
    OUT.save_rows(output_dir / "report_figure2.csv", figure2_rows)
    OUT.save_rows(output_dir / "report_figure3.csv", figure3_rows)
    OUT.save_rows(output_dir / "report_figure4.csv", figure4_rows)
    OUT.save_rows(output_dir / "report_figure5.csv", figure5_rows)


def main() -> None:
    """Entry point for rebuilding Taxi report CSVs."""
    args = parse_args()
    output_dir = (
        args.results_dir / "final_results"
        if args.output_dir is None
        else args.output_dir
    )
    build_report_csvs(args.results_dir, output_dir)
    print(f"Wrote report CSVs to {output_dir}")


if __name__ == "__main__":
    main()
