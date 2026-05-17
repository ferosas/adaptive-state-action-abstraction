"""Post-hoc plotting helpers for reading saved experiment results."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from core.abstraction import Array
from core import output as OUT


LOG_FLOOR = 1e-10


def display_method_label(label: str) -> str:
    """Normalize saved method labels for cleaner figure legends."""
    if label.startswith("Fixed "):
        return label[len("Fixed ") :]
    return label


def setup_matplotlib(output_dir: Path) -> None:
    """Configure matplotlib to use per-run cache directories and the Agg backend."""
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    OUT.ensure_output_dir(output_dir / ".mplconfig")
    OUT.ensure_output_dir(output_dir / ".cache")


def compute_visible_y_limits(
    x_series: Sequence[Array],
    y_series: Sequence[Array],
    *,
    x_max: float | None,
    use_log_scale: bool,
    lower_floor: float = LOG_FLOOR,
    include_values: Sequence[float] | None = None,
) -> tuple[float, float] | None:
    """Choose y-axis limits using only the portion of each series that is visible."""
    visible_values: List[float] = []
    for x_values, y_values in zip(x_series, y_series):
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        if x_max is not None:
            mask &= x_values <= float(x_max) + 1e-12
        if np.any(mask):
            visible_values.extend(float(value) for value in y_values[mask])
    if include_values is not None:
        for value in include_values:
            if math.isfinite(float(value)):
                visible_values.append(float(value))
    if not visible_values:
        return None
    values = np.array(visible_values, dtype=float)
    if use_log_scale:
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size == 0:
            return (lower_floor, lower_floor * 10.0)
        min_value = max(lower_floor, float(np.min(values)))
        max_value = max(min_value, float(np.max(values)))
        if max_value <= min_value:
            return (max(lower_floor, min_value / 1.5), max_value * 1.5)
        return (max(lower_floor, min_value / 1.35), max_value * 1.15)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        padding = max(0.1, 0.15 * max(1.0, abs(max_value)))
        return (min_value - padding, max_value + padding)
    padding = 0.08 * (max_value - min_value)
    return (min_value - padding, max_value + padding)


def plot_metric_vs_bellman_update(
    rows: Sequence[Dict[str, object]],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
    use_log_scale: bool,
    x_max: float | None = None,
) -> None:
    """Plot a fixed-beta metric against Bellman updates."""
    setup_matplotlib(output_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row["beta"]), []).append(dict(row))

    plt.figure(figsize=(8.0, 5.0))
    x_series: List[Array] = []
    y_series: List[Array] = []
    for beta in sorted(grouped):
        beta_rows = grouped[beta]
        x_values = np.array([float(row["sweep"]) for row in beta_rows], dtype=float)
        raw_y_values = np.array([float(row[metric_key]) for row in beta_rows], dtype=float)
        y_values = raw_y_values
        if use_log_scale:
            y_values = np.maximum(y_values, LOG_FLOOR)
        x_series.append(x_values)
        y_series.append(raw_y_values)
        plt.plot(x_values, y_values, linewidth=2.0, label=rf"$\beta={beta:g}$")

    if use_log_scale:
        plt.yscale("log")
    limits = compute_visible_y_limits(
        x_series=x_series,
        y_series=y_series,
        x_max=x_max,
        use_log_scale=use_log_scale,
        lower_floor=LOG_FLOOR,
    )
    if limits is not None:
        plt.ylim(*limits)
    if x_max is not None:
        plt.xlim(0.0, float(x_max))
    plt.xlabel("Bellman update")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_adaptive_switching(
    adaptive_rows: Sequence[Dict[str, object]],
    output_path: Path,
    x_max: float | None,
    x_min: float | None = None,
) -> None:
    """Plot the beta selected by the adaptive controller over time."""
    setup_matplotlib(output_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = np.array([float(row["sweep"]) for row in adaptive_rows], dtype=float)
    betas = np.array([float(row["beta"]) for row in adaptive_rows], dtype=float)

    plt.figure(figsize=(8.0, 4.2))
    plt.step(sweeps, betas, where="post", linewidth=2.6, color="black")
    plt.scatter(sweeps, betas, s=12, color="black")
    limits = compute_visible_y_limits(
        x_series=[sweeps],
        y_series=[betas],
        x_max=x_max,
        use_log_scale=False,
    )
    if limits is not None:
        plt.ylim(*limits)
    if x_min is not None or x_max is not None:
        left = 0.0 if x_min is None else float(x_min)
        right = float(x_max) if x_max is not None else float(np.max(sweeps))
        plt.xlim(left, right)
    plt.xlabel("Bellman update")
    plt.ylabel(r"Current $\beta$")
    plt.title("Adaptive beta schedule during Bellman updates")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


plot_adaptive_beta_switching = plot_adaptive_switching


def plot_comparison_metric(
    rows: Sequence[Dict[str, object]],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
    use_log_scale: bool,
    x_key: str = "sweep",
    x_label: str = "Bellman update",
    x_max: float | None = None,
    optimal_line: float | None = None,
    adaptive_linewidth: float = 3.0,
    fixed_linewidth: float = 1.6,
) -> None:
    """Compare base, fixed-beta, and adaptive methods on a common x-axis."""
    setup_matplotlib(output_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if metric_key not in row or x_key not in row:
            continue
        x_value = row.get(x_key)
        y_value = row.get(metric_key)
        if x_value in (None, "") or y_value in (None, ""):
            continue
        grouped.setdefault(str(row["method_label"]), []).append(dict(row))

    plt.figure(figsize=(9.0, 5.4))
    adaptive_label = "Adaptive"
    ordered_labels = [label for label in grouped if label != adaptive_label]
    ordered_labels.sort(key=lambda label: float(grouped[label][0]["beta"]))
    if adaptive_label in grouped:
        ordered_labels.append(adaptive_label)

    x_series: List[Array] = []
    y_series: List[Array] = []
    for label in ordered_labels:
        method_rows = grouped[label]
        by_x: Dict[float, List[float]] = {}
        for row in method_rows:
            by_x.setdefault(float(row[x_key]), []).append(float(row[metric_key]))
        x_values = np.array(sorted(by_x), dtype=float)
        raw_y_values = np.array([float(np.mean(by_x[x_value])) for x_value in x_values], dtype=float)
        y_values = raw_y_values
        if use_log_scale:
            y_values = np.maximum(y_values, LOG_FLOOR)
        x_series.append(x_values)
        y_series.append(raw_y_values)
        linewidth = adaptive_linewidth if label == adaptive_label else fixed_linewidth
        color = "black" if label == adaptive_label else None
        plt.plot(
            x_values,
            y_values,
            linewidth=linewidth,
            alpha=0.9 if label == adaptive_label else 0.85,
            color=color,
            label=display_method_label(label),
        )

    if optimal_line is not None:
        plt.axhline(optimal_line, color="#374151", linestyle="--", linewidth=1.4, label="Optimal mean value")

    if use_log_scale:
        plt.yscale("log")
    limits = compute_visible_y_limits(
        x_series=x_series,
        y_series=y_series,
        x_max=x_max,
        use_log_scale=use_log_scale,
        lower_floor=LOG_FLOOR,
        include_values=[] if optimal_line is None else [optimal_line],
    )
    if limits is not None:
        plt.ylim(*limits)
    if x_max is not None:
        plt.xlim(0.0, float(x_max))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True, ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
