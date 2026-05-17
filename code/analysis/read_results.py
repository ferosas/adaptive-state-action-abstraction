"""Post-hoc analysis for model-based state-action experiment results."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis import common as AC
from analysis import plotting as AP


def analyze_results(results_dir: Path) -> None:
    """Generate standard plots and tables from a saved results directory."""
    summary = AC.load_summary(results_dir / "summary.json")
    trace_rows = AC.load_rows(results_dir / "traces.csv")
    if not trace_rows:
        trace_rows = AC.load_rows(results_dir / "comparison_traces.csv")
    trace_rows = AC.add_normalized_bellman_compute(trace_rows, summary)

    fixed_rows = [row for row in trace_rows if str(row.get("method_type", "")) == "fixed"]
    adaptive_rows = [row for row in trace_rows if str(row.get("method_type", "")) == "adaptive"]

    metric_summary = summary.get("metric_summary")
    if isinstance(metric_summary, dict):
        AC.save_rows(
            results_dir / "metric_summary.csv",
            [
                {"name": key, "value": value}
                for key, value in metric_summary.items()
            ],
        )

    AP.plot_metric_vs_bellman_update(
        fixed_rows,
        metric_key="abstraction_error",
        y_label="Expected abstraction error",
        title="State-action abstraction error vs Bellman update",
        output_path=results_dir / "abstraction_error_vs_bellman_update.png",
        use_log_scale=True,
    )
    AP.plot_metric_vs_bellman_update(
        fixed_rows,
        metric_key="abstract_q_error",
        y_label=r"Abstract Q error $\|\bar Q_k - \bar Q^*_\beta\|_\infty$",
        title="Abstract Q error vs Bellman update",
        output_path=results_dir / "abstract_q_error_vs_bellman_update.png",
        use_log_scale=True,
    )
    AP.plot_metric_vs_bellman_update(
        fixed_rows,
        metric_key="concrete_q_error",
        y_label=r"Grounded Q error $\|\Gamma \bar Q_k - Q^*\|_\infty$",
        title="Grounded concrete Q error vs Bellman update",
        output_path=results_dir / "concrete_q_error_vs_bellman_update.png",
        use_log_scale=True,
    )
    AP.plot_adaptive_beta_switching(
        adaptive_rows,
        output_path=results_dir / "adaptive_beta_switching_vs_bellman_update.png",
        x_max=None,
    )
    AP.plot_comparison_metric(
        trace_rows,
        metric_key="concrete_q_error",
        y_label=r"Grounded Q error $\|\Gamma \bar Q - Q^*\|_\infty$",
        title="Adaptive vs fixed grounded Q error",
        output_path=results_dir / "adaptive_vs_fixed_q_error.png",
        use_log_scale=True,
    )
    AP.plot_comparison_metric(
        trace_rows,
        metric_key="concrete_q_error",
        y_label=r"Grounded Q error $\|\Gamma \bar Q - Q^*\|_\infty$",
        title="Adaptive vs fixed grounded Q error (normalized compute)",
        output_path=results_dir / "adaptive_vs_fixed_q_error_normalized_compute.png",
        use_log_scale=True,
        x_key="normalized_bellman_compute",
        x_label="Equivalent full-MDP Bellman-summation sweeps",
    )
    optimal_mean_value = AC.compute_optimal_mean_value(summary)
    AP.plot_comparison_metric(
        trace_rows,
        metric_key="policy_return",
        y_label="Exact discounted return (tabular policy evaluation)",
        title="Adaptive vs fixed policy value from grounded greedy policies",
        output_path=results_dir / "adaptive_vs_fixed_policy_return.png",
        use_log_scale=False,
        optimal_line=optimal_mean_value,
    )
    AP.plot_comparison_metric(
        trace_rows,
        metric_key="policy_return",
        y_label="Exact discounted return (tabular policy evaluation)",
        title="Adaptive vs fixed policy value (zoomed to 10 equivalent Q sweeps)",
        output_path=results_dir / "adaptive_vs_fixed_policy_return_zoom10.png",
        use_log_scale=False,
        x_key="normalized_bellman_compute",
        x_label="Equivalent full-MDP Bellman-summation sweeps",
        x_max=10.0,
        optimal_line=optimal_mean_value,
    )


def main() -> None:
    args = AC.parse_results_dir("Analyze model-based state-action experiment results.")
    analyze_results(args.results_dir)


if __name__ == "__main__":
    main()
