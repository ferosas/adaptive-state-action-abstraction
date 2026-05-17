"""Build report-style figures from DoorKey postprocessed CSV tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis import common as AC
from analysis import plotting as AP


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "results" / "doorkey" / "final_results"
)
DEFAULT_FIGURES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "results" / "doorkey" / "figures"
)
SHORT_X_MAX = 30.0
def parse_args() -> argparse.Namespace:
    """Define the CLI for rebuilding DoorKey report figures."""
    parser = argparse.ArgumentParser(
        description="Build report-style figures from DoorKey final_results CSVs."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def _load_required_rows(results_dir: Path, filename: str):
    """Load one required CSV from the final-results directory."""
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return AC.load_rows(path)


def _load_summary_for_figures(results_dir: Path) -> dict[str, object]:
    """Load the saved experiment summary associated with a final-results directory."""
    candidates = [
        results_dir / "summary.json",
        results_dir.parent / "summary.json",
    ]
    for path in candidates:
        if path.exists():
            return AC.load_summary(path)
    raise FileNotFoundError(
        f"Could not find summary.json next to figure CSVs in {results_dir} or {results_dir.parent}"
    )


def _max_sweeps_x_limit(results_dir: Path) -> float:
    """Read the saved max-sweeps budget for x-axis clipping."""
    summary = _load_summary_for_figures(results_dir)
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("Expected summary['config'] to be a dictionary.")
    return float(config["max_sweeps"])


def make_figure0(results_dir: Path, output_dir: Path) -> Path:
    """Build Figure 0: rate-distortion view of the fixed abstraction family."""
    rows = _load_required_rows(results_dir, "report_figure0.csv")
    output_path = output_dir / "report_figure0.png"

    AP.setup_matplotlib(output_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_rows = sorted(rows, key=lambda row: float(row["beta"]))
    x_values = np.array([float(row["abstraction_error"]) for row in sorted_rows], dtype=float)
    y_values = np.array(
        [float(row["normalized_effective_abstraction_size"]) for row in sorted_rows],
        dtype=float,
    )
    betas = [float(row["beta"]) for row in sorted_rows]

    plt.figure(figsize=(7.4, 4.8))
    plt.plot(x_values, y_values, linewidth=2.0, color="black", alpha=0.9)
    plt.scatter(x_values, y_values, s=24, color="black")
    for index, (x_value, y_value, beta) in enumerate(zip(x_values, y_values, betas)):
        plt.annotate(
            rf"$\beta={beta:g}$",
            (x_value, y_value),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            va="bottom" if index % 2 == 0 else "top",
        )
    plt.xlabel("Abstraction error")
    plt.ylabel("Normalized effective state-action pairs")
    plt.title("Rate-distortion view of the fixed DoorKey abstraction family")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def make_figure1(results_dir: Path, output_dir: Path) -> Path:
    """Build Figure 1: adaptive beta schedule against Bellman update."""
    rows = _load_required_rows(results_dir, "report_figure1.csv")
    output_path = output_dir / "report_figure1.png"
    AP.plot_adaptive_switching(
        adaptive_rows=rows,
        output_path=output_path,
        x_max=SHORT_X_MAX,
    )
    return output_path


def make_figure2(results_dir: Path, output_dir: Path) -> Path:
    """Build Figure 2: policy return against normalized compute."""
    rows = _load_required_rows(results_dir, "report_figure2.csv")
    output_path = output_dir / "report_figure2.png"
    summary = _load_summary_for_figures(results_dir)
    optimal_mean_value = AC.compute_optimal_mean_value(summary)
    AP.plot_comparison_metric(
        rows=rows,
        metric_key="policy_return",
        y_label="Exact discounted return (tabular policy evaluation)",
        title="Adaptive, fixed, and base policy value vs normalized compute",
        output_path=output_path,
        use_log_scale=False,
        x_key="normalized_bellman_compute",
        x_label="Equivalent full-MDP Bellman-summation sweeps",
        x_max=SHORT_X_MAX,
        optimal_line=optimal_mean_value,
    )
    return output_path


def make_figure3(results_dir: Path, output_dir: Path) -> Path:
    """Build Figure 3: policy return against Bellman update."""
    rows = _load_required_rows(results_dir, "report_figure3.csv")
    output_path = output_dir / "report_figure3.png"
    summary = _load_summary_for_figures(results_dir)
    optimal_mean_value = AC.compute_optimal_mean_value(summary)
    AP.plot_comparison_metric(
        rows=rows,
        metric_key="policy_return",
        y_label="Exact discounted return (tabular policy evaluation)",
        title="Adaptive, fixed, and base policy value vs Bellman update",
        output_path=output_path,
        use_log_scale=False,
        x_key="sweep",
        x_label="Bellman update",
        x_max=SHORT_X_MAX,
        optimal_line=optimal_mean_value,
    )
    return output_path


def make_figure4(results_dir: Path, output_dir: Path) -> Path:
    """Build Figure 4: induced value error against Bellman update."""
    rows = _load_required_rows(results_dir, "report_figure4.csv")
    output_path = output_dir / "report_figure4.png"
    x_max = _max_sweeps_x_limit(results_dir)

    AP.setup_matplotlib(output_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row.get("abstract_value_error") in (None, "") or row.get("sweep") in (None, ""):
            continue
        grouped.setdefault(str(row["method_label"]), []).append(dict(row))

    adaptive_label = "Adaptive"
    ordered_labels = [label for label in grouped if label != adaptive_label]
    ordered_labels.sort(key=lambda label: float(grouped[label][0]["beta"]))
    if adaptive_label in grouped:
        ordered_labels.append(adaptive_label)
    plt.figure(figsize=(9.0, 5.4))
    x_series: list[np.ndarray] = []
    y_series: list[np.ndarray] = []
    line_by_label: dict[str, object] = {}

    for label in ordered_labels:
        method_rows = grouped[label]
        by_sweep: dict[float, list[float]] = {}
        for row in method_rows:
            by_sweep.setdefault(float(row["sweep"]), []).append(float(row["abstract_value_error"]))
        x_values = np.array(sorted(by_sweep), dtype=float)
        raw_y_values = np.array(
            [float(np.mean(by_sweep[x_value])) for x_value in x_values],
            dtype=float,
        )
        y_values = np.maximum(raw_y_values, AP.LOG_FLOOR)
        x_series.append(x_values)
        y_series.append(raw_y_values)
        linewidth = 3.0 if label == adaptive_label else 1.6
        color = "black" if label == adaptive_label else None
        (line,) = plt.plot(
            x_values,
            y_values,
            linewidth=linewidth,
            alpha=0.9 if label == adaptive_label else 0.85,
            color=color,
            label=AP.display_method_label(label),
        )
        line_by_label[label] = line

    for label in ordered_labels:
        method_rows = grouped[label]
        abstraction_error = method_rows[0].get("abstraction_error")
        if abstraction_error in (None, ""):
            continue
        color = line_by_label[label].get_color()
        plt.axhline(
            float(abstraction_error),
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            color=color,
        )

    plt.yscale("log")
    limits = AP.compute_visible_y_limits(
        x_series=x_series,
        y_series=y_series,
        x_max=x_max,
        use_log_scale=True,
        lower_floor=AP.LOG_FLOOR,
        include_values=[
            float(row["abstraction_error"])
            for row in rows
            if row.get("abstraction_error") not in (None, "")
        ],
    )
    if limits is not None:
        plt.ylim(*limits)
    plt.xlim(0.0, x_max)
    plt.xlabel("Bellman update")
    plt.ylabel(r"Induced value error $\|V_{\Gamma \bar Q_k} - V_{\Gamma \bar Q^*_\beta}\|_\infty$")
    plt.title("Induced value error against Bellman update")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True, ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def make_figure5(results_dir: Path, output_dir: Path) -> Path:
    """Build Figure 5: grounded value error against Bellman update."""
    rows = _load_required_rows(results_dir, "report_figure5.csv")
    output_path = output_dir / "report_figure5.png"
    x_max = _max_sweeps_x_limit(results_dir)
    AP.plot_comparison_metric(
        rows,
        metric_key="concrete_value_error",
        y_label=r"Grounded value error $\|V_{\Gamma \bar Q} - V^*\|_\infty$",
        title="Adaptive, fixed, and base grounded value error against Bellman update",
        output_path=output_path,
        use_log_scale=True,
        x_key="sweep",
        x_label="Bellman update",
        x_max=x_max,
    )
    return output_path


def main() -> None:
    """Build every DoorKey report figure."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_path in [
        args.output_dir / "report_figure8.png",
        args.output_dir / "report_figure7.png",
        args.output_dir / "report_figure6.png",
        args.output_dir / "report_figure5.png",
    ]:
        if legacy_path.exists():
            legacy_path.unlink()
    make_figure0(args.results_dir, args.output_dir)
    make_figure1(args.results_dir, args.output_dir)
    make_figure2(args.results_dir, args.output_dir)
    make_figure3(args.results_dir, args.output_dir)
    make_figure4(args.results_dir, args.output_dir)
    make_figure5(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
