"""Build the SysAdmin scaling summary figure from ``exp_scaling.py`` outputs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis import plotting as AP
from exp4_sysadmin import exp_scaling as ES


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "results" / "sysadmin_scaling"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "figures"
PANEL_TITLE_KW = {"loc": "left", "fontweight": "bold", "fontsize": 11.5, "pad": 7}
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 8.2


def parse_args() -> argparse.Namespace:
    """Define the plotting CLI."""
    parser = argparse.ArgumentParser(
        description="Build a single-row three-panel SysAdmin scaling figure."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--basename",
        type=str,
        default="sysadmin_scaling_summary",
        help="Output filename stem for the PDF and PNG figures.",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=None,
        help="Minimum machine count to include in the plots.",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Maximum machine count to include in the plots.",
    )
    parser.add_argument(
        "--no-selected-markers",
        action="store_true",
        help="Do not highlight the near-optimal beta on the RD curves.",
    )
    parser.add_argument(
        "--refresh-aggregates",
        action="store_true",
        help=(
            "Deprecated: aggregate scaling CSVs are rebuilt by default before plotting."
        ),
    )
    parser.add_argument(
        "--no-refresh-aggregates",
        action="store_true",
        help=(
            "Use the existing aggregate CSVs instead of rebuilding them from N*/ "
            "run folders before plotting."
        ),
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    """Read a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, str], key: str, default: float = math.nan) -> float:
    """Read a float column with a safe missing-value fallback."""
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    """Read an integer column saved by the scaling scripts."""
    value = row.get(key)
    if value is None or value == "":
        return default
    return int(float(value))


def is_true(value: str | None) -> bool:
    """Parse booleans written by Python's CSV writer."""
    return str(value).strip().lower() in {"1", "true", "yes"}


def group_by_n(rows: Iterable[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    """Group rows by machine count."""
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(as_int(row, "num_machines"), []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: as_float(row, "effective_rate_size"))
    return grouped


def filter_rows_by_n(
    rows: Iterable[Dict[str, str]],
    *,
    n_min: int | None,
    n_max: int | None,
) -> List[Dict[str, str]]:
    """Keep only rows whose machine count lies in the requested interval."""
    filtered: List[Dict[str, str]] = []
    for row in rows:
        num_machines = as_int(row, "num_machines")
        if n_min is not None and num_machines < int(n_min):
            continue
        if n_max is not None and num_machines > int(n_max):
            continue
        filtered.append(row)
    return filtered


def selected_by_n(rows: Iterable[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    """Map each machine count to its near-optimal adaptive row."""
    selected: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if is_true(row.get("reached")):
            selected[as_int(row, "num_machines")] = row
    return selected


def full_size_by_n(rd_by_n: Dict[int, List[Dict[str, str]]]) -> Dict[int, float]:
    """Read the concrete state-action size for each machine count."""
    sizes: Dict[int, float] = {}
    for num_machines, rows in rd_by_n.items():
        if not rows:
            continue
        saved_size = as_float(rows[0], "num_state_action_pairs")
        if math.isfinite(saved_size) and saved_size > 0.0:
            sizes[num_machines] = saved_size
        else:
            sizes[num_machines] = float((num_machines + 1) * (2**num_machines))
    return sizes


def available_run_ns(results_dir: Path) -> List[int]:
    """Return machine counts with complete per-run artifacts under results_dir."""
    ns: List[int] = []
    for child in results_dir.iterdir() if results_dir.exists() else []:
        if not child.is_dir() or not child.name.startswith("N"):
            continue
        try:
            num_machines = int(child.name[1:])
        except ValueError:
            continue
        if ES.run_artifacts_exist(child):
            ns.append(num_machines)
    return sorted(ns)


def refresh_aggregate_csvs(
    *,
    results_dir: Path,
    n_min: int | None,
    n_max: int | None,
) -> None:
    """Rebuild aggregate CSVs from existing per-N run folders."""
    ns = [
        num_machines
        for num_machines in available_run_ns(results_dir)
        if (n_min is None or num_machines >= int(n_min))
        and (n_max is None or num_machines <= int(n_max))
    ]
    if not ns:
        raise ValueError(
            "No complete N*/ run folders found for aggregation. "
            "Run exp_scaling.py first."
        )

    all_rd_rows = []
    near_rows = []
    policy_payloads = []
    near_optimal_fraction = 0.99
    for num_machines in ns:
        run_dir = results_dir / f"N{num_machines}"
        rd_rows, near_row, policy_payload = ES.aggregate_one_run(
            run_dir=run_dir,
            num_machines=num_machines,
            near_optimal_fraction=near_optimal_fraction,
        )
        all_rd_rows.extend(rd_rows)
        near_rows.append(near_row)
        policy_payloads.append(policy_payload)

    class _Args:
        pass

    args = _Args()
    args.n_min = min(ns)
    args.n_max = max(ns)
    args.near_optimal_fraction = near_optimal_fraction
    args.metric_kind = None
    args.eval_interval = None
    args.num_workers = None
    ES.write_aggregate_outputs(
        output_dir=results_dir,
        rd_rows=all_rd_rows,
        near_rows=near_rows,
        policy_payloads=policy_payloads,
        args=args,
    )


def draw_sysadmin_scenario(ax, *, n_values: List[int]) -> None:
    """Draw a compact Ring SysAdmin machine diagram."""
    from matplotlib.patches import Circle

    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    del n_values
    center = np.array([0.5, 0.5], dtype=float)
    radius = 0.34
    num_drawn_machines = 6
    angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, num_drawn_machines, endpoint=False)
    positions = np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ]
    )

    for index in range(num_drawn_machines):
        start = positions[index]
        end = positions[(index + 1) % num_drawn_machines]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="0.65",
            linewidth=2.5,
            zorder=1,
        )

    for index, (x_value, y_value) in enumerate(positions):
        is_up = index not in {2, 5}
        facecolor = "#4daf7c" if is_up else "#d95f5f"
        node = Circle(
            (x_value, y_value),
            0.075,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.1,
            zorder=3,
        )
        ax.add_patch(node)
        ax.text(
            x_value,
            y_value,
            rf"$M_{index + 1}$",
            fontsize=10,
            ha="center",
            va="center",
            color="white",
            zorder=4,
        )


def make_scaling_figure(
    *,
    results_dir: Path,
    output_dir: Path,
    basename: str,
    n_min: int | None = None,
    n_max: int | None = None,
    show_selected_markers: bool = True,
) -> List[Path]:
    """Build and save the three-panel SysAdmin scaling figure."""
    rd_rows = filter_rows_by_n(
        read_rows(results_dir / "rate_distortion.csv"),
        n_min=n_min,
        n_max=n_max,
    )
    near_rows = filter_rows_by_n(
        read_rows(results_dir / "adaptive_near_optimal.csv"),
        n_min=n_min,
        n_max=n_max,
    )
    rd_by_n = group_by_n(rd_rows)
    near_by_n = selected_by_n(near_rows)
    full_sizes_by_n = full_size_by_n(rd_by_n)
    if not rd_by_n:
        raise ValueError(
            "No rate-distortion rows found for the requested N range. "
            "Check --results-dir, --n-min, and --n-max."
        )
    if not near_by_n:
        raise ValueError(
            "No reached near-optimal rows found for the requested N range. "
            "Check --results-dir, --n-min, and --n-max."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    AP.setup_matplotlib(output_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_values = sorted(rd_by_n)
    selected_n_values = sorted(near_by_n)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(n_values)))
    color_by_n = {num_machines: colors[index] for index, num_machines in enumerate(n_values)}

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.2, 4.9),
        gridspec_kw={"width_ratios": [0.82, 1.34, 1.28], "wspace": 0.34},
    )
    ax_scenario, ax_rd, ax_tradeoff = axes
    draw_sysadmin_scenario(ax_scenario, n_values=n_values)
    ax_scenario.set_title("(A) Ring SysAdmin", **PANEL_TITLE_KW)
    ax_rd.set_title("(B) Rate-distortion frontiers", **PANEL_TITLE_KW)
    ax_tradeoff.set_title("(C) Selected compression and distortion", **PANEL_TITLE_KW)

    # (B) Rate-distortion curves across N.
    for num_machines in n_values:
        rows = rd_by_n[num_machines]
        x_values = np.array([as_float(row, "distortion") for row in rows], dtype=float)
        full_size = full_sizes_by_n.get(
            num_machines,
            float((num_machines + 1) * (2**num_machines)),
        )
        y_values = np.array(
            [as_float(row, "effective_rate_size") / full_size for row in rows],
            dtype=float,
        )
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        ax_rd.plot(
            x_values[mask],
            y_values[mask],
            marker="o",
            markersize=4.2,
            linewidth=2.0,
            color=color_by_n[num_machines],
            label=rf"$N={num_machines}$",
        )
        if show_selected_markers and num_machines in near_by_n:
            selected = near_by_n[num_machines]
            ax_rd.scatter(
                [as_float(selected, "distortion")],
                [as_float(selected, "effective_rate_size") / full_size],
                marker="*",
                s=140,
                color=color_by_n[num_machines],
                edgecolor="black",
                linewidth=0.65,
                zorder=5,
            )
    ax_rd.set_xlabel("Normalized distortion", fontsize=AXIS_LABEL_SIZE)
    ax_rd.set_ylabel("Normalised effective state-action pairs", fontsize=AXIS_LABEL_SIZE)
    ax_rd.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax_rd.grid(True, which="both", alpha=0.22)
    ax_rd.legend(frameon=False, fontsize=LEGEND_SIZE)

    # (C) Total/state/action compression and distortion at the near-optimal checkpoint.
    n_axis = np.array(selected_n_values, dtype=float)
    total_effective_fraction = np.array(
        [
            as_float(near_by_n[num_machines], "effective_rate_size")
            / full_sizes_by_n.get(
                num_machines,
                float((num_machines + 1) * (2**num_machines)),
            )
            for num_machines in selected_n_values
        ],
        dtype=float,
    )
    state_effective_fraction = np.array(
        [
            as_float(near_by_n[num_machines], "normalized_state_effective_size")
            for num_machines in selected_n_values
        ],
        dtype=float,
    )
    action_effective_fraction = np.array(
        [
            as_float(near_by_n[num_machines], "normalized_action_effective_size")
            for num_machines in selected_n_values
        ],
        dtype=float,
    )
    selected_distortion = np.array(
        [as_float(near_by_n[num_machines], "distortion") for num_machines in selected_n_values],
        dtype=float,
    )
    ax_tradeoff.plot(
        n_axis,
        total_effective_fraction,
        marker="D",
        linewidth=2.2,
        color="black",
        label="Total compression",
    )
    ax_tradeoff.plot(
        n_axis,
        state_effective_fraction,
        marker="o",
        linewidth=2.2,
        linestyle="--",
        color="#1f77b4",
        label="State compression",
    )
    ax_tradeoff.plot(
        n_axis,
        action_effective_fraction,
        marker="^",
        linewidth=2.2,
        linestyle="--",
        color="#2ca02c",
        label="Action compression",
    )
    ax_tradeoff.plot(
        n_axis,
        selected_distortion,
        marker="s",
        linewidth=2.2,
        color="#d62728",
        label="Normalized distortion",
    )
    ax_tradeoff.set_xlabel("Number of machines", fontsize=AXIS_LABEL_SIZE)
    ax_tradeoff.set_ylabel("Selected normalized value", fontsize=AXIS_LABEL_SIZE)
    ax_tradeoff.set_xticks(selected_n_values)
    ax_tradeoff.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    finite_state_action = np.concatenate(
        [
            state_effective_fraction[np.isfinite(state_effective_fraction)],
            action_effective_fraction[np.isfinite(action_effective_fraction)],
        ]
    )
    if finite_state_action.size == 0:
        ax_tradeoff.text(
            0.5,
            0.54,
            "Re-run scaling experiments\nfor projected state/action rates",
            ha="center",
            va="center",
            transform=ax_tradeoff.transAxes,
            fontsize=9,
            color="0.35",
        )
    y_max = float(
        np.nanmax(
            np.concatenate(
                [
                    total_effective_fraction[np.isfinite(total_effective_fraction)],
                    state_effective_fraction[np.isfinite(state_effective_fraction)],
                    action_effective_fraction[np.isfinite(action_effective_fraction)],
                    selected_distortion[np.isfinite(selected_distortion)],
                    np.array([0.05]),
                ]
            )
        )
    )
    ax_tradeoff.set_ylim(0.0, min(1.0, max(0.08, 1.15 * y_max)))
    ax_tradeoff.grid(True, alpha=0.22)
    ax_tradeoff.legend(frameon=False, fontsize=LEGEND_SIZE, loc="upper right")

    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.17)
    output_paths = [
        output_dir / f"{basename}.pdf",
        output_dir / f"{basename}.png",
    ]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_paths


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.n_min is not None and args.n_max is not None and args.n_min > args.n_max:
        raise ValueError("Require --n-min <= --n-max.")
    if not args.no_refresh_aggregates:
        refresh_aggregate_csvs(
            results_dir=args.results_dir,
            n_min=args.n_min,
            n_max=args.n_max,
        )
    output_paths = make_scaling_figure(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        basename=args.basename,
        n_min=args.n_min,
        n_max=args.n_max,
        show_selected_markers=not args.no_selected_markers,
    )
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
