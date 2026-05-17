"""Shared model-based pipeline for experiments that abstract state-action pairs.

Unlike the state-abstraction pipeline, the abstraction here lives on concrete
state-action pairs and planning is done directly in Q-space via the paper's
decoder-based operator

    \bar F_\eta = L F \Gamma,

specialized to a deterministic decoder representative map g(z). Grounding uses
the full soft encoder nu(z | s, a), while abstract readout evaluates the
concrete Bellman update at the decoder representatives. Experiment folders such
as `exp2_taxi` provide the MDP, metric, and CLI wrapper; this module owns the
reusable solver and plotting logic.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Sequence
import numpy as np
from math import ceil
from time import perf_counter
from core import adaptive as AD
from core import analysis as AN
from core import abstraction as AB
from core import planning as PL
from core import output as OUT
Array = np.ndarray


def _log_progress(logger: Callable[[str], None] | None, message: str) -> None:
    """Emit a progress message when a logger is provided."""
    if logger is not None:
        logger(message)


def _beta_schedules_match(
    fixed_beta_schedule: Sequence[float],
    adaptive_beta_schedule: Sequence[float] | None,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether the adaptive ladder is absent or numerically identical."""
    if adaptive_beta_schedule is None:
        return True
    if len(fixed_beta_schedule) != len(adaptive_beta_schedule):
        return False
    return all(
        abs(float(fixed_beta) - float(adaptive_beta)) <= tolerance
        for fixed_beta, adaptive_beta in zip(fixed_beta_schedule, adaptive_beta_schedule)
    )


########################################################
# EXPERIMENT MISC STUFF
def _build_baseline(
    mdp: AB.TabularMDP,
    distortion: Array,
) -> Dict[str, object]:
    """Build the concrete MDP baseline and the distortion metric it uses."""
    num_pairs = int(mdp.num_states * mdp.num_actions)
    mu_uniform = np.full(num_pairs, 1.0 / float(num_pairs), dtype=float)
    optimal_values = PL.solve_optimal_values(mdp)
    optimal_q = mdp.rewards + mdp.gamma * np.einsum("sak,k->sa", mdp.transitions, optimal_values)
    return {
        "mdp": mdp,
        "num_pairs": num_pairs,
        "mu_uniform": mu_uniform,
        "optimal_concrete_values": np.asarray(optimal_values, dtype=float),
        "optimal_concrete_q": np.asarray(optimal_q, dtype=float).reshape(-1),
        "distortion": distortion,
    }


def _fit_abstraction_job(
    distortion: Array,
    mu_uniform: Array,
    beta: float,
    abstract_alphabet_size: int,
    ba_max_outer: int,
    ba_max_inner: int,
    abstraction_solver: str,
    num_actions: int,
) -> AB.StateActionAbstraction:
    """Fit one abstraction independently for one beta."""
    return AB.fit_soft_abstraction(
        distortion=distortion,
        mu=mu_uniform,
        beta=beta,
        num_abstract=abstract_alphabet_size,
        decoder_init=None,
        encoder_init=None,
        max_outer=ba_max_outer,
        max_inner=ba_max_inner,
        tolerance=1e-6,
        solver_kind=abstraction_solver,
        num_actions=int(num_actions),
    )


def _solve_optimal_abstract_q_job(
    mdp: AB.TabularMDP,
    abstraction: AB.StateActionAbstraction,
) -> Array:
    """Solve one abstract fixed point independently."""
    return PL.solve_optimal_abstract_q(mdp, abstraction)


def _compute_abstraction_error_job(
    mu_uniform: Array,
    abstraction: AB.StateActionAbstraction,
    distortion: Array,
    abstraction_error_mode: str,
) -> float:
    """Compute one abstraction error independently."""
    return AB.compute_abstraction_error(
        mu_uniform,
        abstraction.encoder,
        abstraction.decoder,
        distortion,
        mode=abstraction_error_mode,
    )


def _trace_fixed_job(
    mdp: AB.TabularMDP,
    abstraction: AB.StateActionAbstraction,
    optimal_abstract_q: Array,
    optimal_concrete_q: Array,
    optimal_concrete_values: Array,
    abstraction_error: float,
    max_sweeps: int,
    evaluation_sweeps: Sequence[int],
    save_policies: bool,
    encoder_path: str | None,
) -> tuple[List[Dict[str, float | str]], List[Dict[str, object]]]:
    """Trace one fixed-beta family member independently."""
    policy_rows: List[Dict[str, object]] = [] if save_policies else []
    rows = AN.trace_fixed(
        mdp=mdp,
        abstraction=abstraction,
        optimal_abstract_q=optimal_abstract_q,
        optimal_concrete_q=optimal_concrete_q,
        optimal_concrete_values=optimal_concrete_values,
        abstraction_error=abstraction_error,
        max_sweeps=max_sweeps,
        evaluation_sweeps=evaluation_sweeps,
        policy_value=AN.evaluate_policy_value_exact,
        policy_rows=policy_rows if save_policies else None,
        encoder_path=encoder_path,
    )
    return rows, policy_rows


def _build_abstractions(
    baseline: Dict[str, object],
    abstract_alphabet_size: int,
    beta_schedule: Sequence[float],
    abstraction_error_mode: str,
    ba_max_outer: int,
    ba_max_inner: int,
    abstraction_solver: str,
    num_workers: int,
    logger: Callable[[str], None] | None = None,
    family_name: str = "abstraction",
) -> Dict[str, object]:
    """Fit and solve the abstraction ladder for one beta schedule."""
    abstraction_solver = AB.normalize_solver_kind(abstraction_solver)
    abstractions: List[AB.StateActionAbstraction] = []
    decoder: Array | None = None
    encoder: Array | None = None
    distortion = baseline["distortion"]

    worker_count = max(1, int(num_workers))
    num_betas = len(beta_schedule)
    if worker_count > 1 and num_betas > 1:
        _log_progress(
            logger,
            f"Fitting {family_name} independently across {num_betas} betas with {worker_count} workers",
        )
        with ProcessPoolExecutor(max_workers=min(worker_count, num_betas)) as executor:
            abstractions = list(
                executor.map(
                    _fit_abstraction_job,
                    [distortion] * num_betas,
                    [baseline["mu_uniform"]] * num_betas,
                    [float(beta) for beta in beta_schedule],
                    [int(abstract_alphabet_size)] * num_betas,
                    [int(ba_max_outer)] * num_betas,
                    [int(ba_max_inner)] * num_betas,
                    [abstraction_solver] * num_betas,
                    [int(baseline["mdp"].num_actions)] * num_betas,
                )
            )
        with ProcessPoolExecutor(max_workers=min(worker_count, num_betas)) as executor:
            optimal_abstract_qs = list(
                executor.map(
                    _solve_optimal_abstract_q_job,
                    [baseline["mdp"]] * num_betas,
                    abstractions,
                )
            )
        with ProcessPoolExecutor(max_workers=min(worker_count, num_betas)) as executor:
            abstraction_errors = list(
                executor.map(
                    _compute_abstraction_error_job,
                    [baseline["mu_uniform"]] * num_betas,
                    abstractions,
                    [baseline["distortion"]] * num_betas,
                    [abstraction_error_mode] * num_betas,
                )
            )
    else:
        for index, beta in enumerate(beta_schedule, start=1):
            _log_progress(
                logger,
                f"Fitting beta={beta:g} ({index}/{num_betas})",
            )
            abstraction = AB.fit_soft_abstraction(
                distortion=distortion,
                mu=baseline["mu_uniform"],
                beta=beta,
                num_abstract=abstract_alphabet_size,
                decoder_init=decoder,
                encoder_init=encoder,
                max_outer=ba_max_outer,
                max_inner=ba_max_inner,
                tolerance=1e-6,
                solver_kind=abstraction_solver,
                num_actions=int(baseline["mdp"].num_actions),
            )
            abstractions.append(abstraction)
            decoder = abstraction.full_decoder
            encoder = abstraction.full_encoder

        optimal_abstract_qs = []
        for index, abstraction in enumerate(abstractions, start=1):
            beta = float(abstraction.beta)
            _log_progress(
                logger,
                f"Solving abstract fixed point for beta={beta:g} ({index}/{num_betas})",
            )
            optimal_abstract_qs.append(PL.solve_optimal_abstract_q(baseline["mdp"], abstraction))

        abstraction_errors = [
            AB.compute_abstraction_error(
                baseline["mu_uniform"],
                abstraction.encoder,
                abstraction.decoder,
                baseline["distortion"],
                mode=abstraction_error_mode,
            )
            for abstraction in abstractions
        ]
    return {
        "beta_schedule": [float(beta) for beta in beta_schedule],
        "abstractions": abstractions,
        "optimal_abstract_qs": optimal_abstract_qs,
        "abstraction_errors": abstraction_errors,
        "abstraction_error_mode": str(abstraction_error_mode),
        "abstraction_solver": str(AB.normalize_solver_kind(abstraction_solver)),
    }


def _save_state_encoders(
    output_dir: Path,
    family_name: str,
    abstractions: Sequence[AB.StateActionAbstraction],
) -> None:
    """Persist one state-encoder array per abstraction when the structured solver is used."""
    family_dir = output_dir / "state_encoders" / family_name
    family_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []
    for index, abstraction in enumerate(abstractions):
        if abstraction.state_encoder is None:
            continue
        beta_tag = f"beta_{float(abstraction.beta):g}".replace(".", "p")
        filename = f"{beta_tag}.npy"
        path = family_dir / filename
        np.save(path, np.asarray(abstraction.state_encoder, dtype=float))
        manifest_rows.append(
            {
                "family": family_name,
                "index": int(index),
                "beta": float(abstraction.beta),
                "path": str(path.relative_to(output_dir)),
                "num_states": int(np.asarray(abstraction.state_encoder).shape[0]),
                "num_abstract_states": int(np.asarray(abstraction.state_encoder).shape[1]),
            }
        )
    if manifest_rows:
        OUT.save_rows(family_dir / "manifest.csv", manifest_rows)


def _save_policy_encoders(
    output_dir: Path,
    family_name: str,
    abstractions: Sequence[AB.StateActionAbstraction],
) -> List[str]:
    """Persist one active flat encoder array per abstraction for policy analysis."""
    family_dir = output_dir / "policy_encoders" / family_name
    family_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []
    encoder_paths: List[str] = []
    for index, abstraction in enumerate(abstractions):
        beta_tag = f"beta_{float(abstraction.beta):g}".replace(".", "p")
        filename = f"{beta_tag}.npy"
        path = family_dir / filename
        np.save(path, np.asarray(abstraction.encoder, dtype=float))
        relative_path = str(path.relative_to(output_dir))
        encoder_paths.append(relative_path)
        manifest_rows.append(
            {
                "family": family_name,
                "index": int(index),
                "beta": float(abstraction.beta),
                "path": relative_path,
                "num_concrete_pairs": int(np.asarray(abstraction.encoder).shape[0]),
                "num_abstract_pairs": int(np.asarray(abstraction.encoder).shape[1]),
            }
        )
    OUT.save_rows(family_dir / "manifest.csv", manifest_rows)
    return encoder_paths


def _build_fixed_family_report_rows(
    baseline: Dict[str, object],
    fixed_family: Dict[str, object],
) -> List[Dict[str, object]]:
    """Summarize the fitted fixed ladder into report-table rows."""
    return AB.summarize_abstraction_family(
        baseline["mu_uniform"],
        fixed_family["abstractions"],
        baseline["distortion"],
        mode=str(fixed_family["abstraction_error_mode"]),
        num_actions=int(baseline["mdp"].num_actions),
    )


def _run_traces(
    *,
    baseline: Dict[str, object],
    fixed_family: Dict[str, object],
    adaptive_family: Dict[str, object],
    max_sweeps: int,
    eval_interval: int,
    save_policies: bool,
    num_workers: int,
    logger: Callable[[str], None] | None = None,
    fixed_policy_encoder_paths: Sequence[str] | None = None,
    adaptive_policy_encoder_paths: Sequence[str] | None = None,
) -> tuple[List[Dict[str, float | str]], Dict[str, object], List[Dict[str, object]]]:
    """Run and save trace rows for the base, fixed, and adaptive methods."""

    traces: List[Dict[str, float | str]] = []
    policy_rows: List[Dict[str, object]] = [] if save_policies else []
    policy_value_cache: Dict[bytes, float] = {}
    mdp = baseline["mdp"]
    optimal_concrete_q = baseline["optimal_concrete_q"]
    optimal_concrete_values = baseline["optimal_concrete_values"]

    # 0) Helper function to evaluate policy value and cache results
    def policy_value(mdp, policy: Array) -> float:
        key = np.asarray(policy, dtype=np.int16).tobytes()
        if key not in policy_value_cache:
            policy_value_cache[key] = float(AN.evaluate_policy_value_exact(mdp, policy))
        return policy_value_cache[key]

    # 1) Collect trace rows for the base concrete method.    
    base_trace_sweeps = int(max_sweeps)
    base_evaluation_sweeps = AN.build_evaluation_sweeps(
        max_sweeps=base_trace_sweeps,
        eval_interval=eval_interval,
    )
    _log_progress(
        logger,
        f"Tracing base MDP over {base_trace_sweeps} sweeps ({len(base_evaluation_sweeps)} checkpoints)",
    )
    traces.extend(
        AN.trace_base(
            mdp=mdp,
            optimal_concrete_q=optimal_concrete_q,
            optimal_concrete_values=optimal_concrete_values,
            max_sweeps=base_trace_sweeps,
            evaluation_sweeps=base_evaluation_sweeps,
            policy_value=policy_value,
            policy_rows=policy_rows if save_policies else None,
        )
    )

    # 2) Collect fixed-beta trace rows.
    fixed_abstractions = fixed_family["abstractions"]
    fixed_optimal_abstract_qs = fixed_family["optimal_abstract_qs"]
    fixed_abstraction_errors = fixed_family["abstraction_errors"]
    max_budget_units = int(max_sweeps) * int(mdp.num_state_action_pairs)
    
    fixed_specs = []
    for abstraction, optimal_abstract_q, abstraction_error in zip(
        fixed_abstractions,
        fixed_optimal_abstract_qs,
        fixed_abstraction_errors,
    ):
        fixed_trace_sweeps = int(ceil(max_budget_units / float(abstraction.num_abstract)))
        fixed_evaluation_sweeps = AN.build_evaluation_sweeps(
            max_sweeps=fixed_trace_sweeps,
            eval_interval=eval_interval,
        )
        fixed_specs.append(
            (
                abstraction,
                optimal_abstract_q,
                abstraction_error,
                fixed_trace_sweeps,
                fixed_evaluation_sweeps,
            )
        )
        _log_progress(
            logger,
            (
                f"Tracing fixed beta={float(abstraction.beta):g} "
                f"over {fixed_trace_sweeps} sweeps "
                f"({len(fixed_evaluation_sweeps)} checkpoints)"
            ),
        )
    worker_count = max(1, int(num_workers))
    if worker_count > 1 and len(fixed_specs) > 1:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(fixed_specs))) as executor:
            fixed_results = list(
                executor.map(
                    _trace_fixed_job,
                    [mdp] * len(fixed_specs),
                    [spec[0] for spec in fixed_specs],
                    [spec[1] for spec in fixed_specs],
                    [optimal_concrete_q] * len(fixed_specs),
                    [optimal_concrete_values] * len(fixed_specs),
                    [float(spec[2]) for spec in fixed_specs],
                    [int(spec[3]) for spec in fixed_specs],
                    [spec[4] for spec in fixed_specs],
                    [bool(save_policies)] * len(fixed_specs),
                    list(fixed_policy_encoder_paths or [None] * len(fixed_specs)),
                )
            )
        for fixed_rows, fixed_policy_rows in fixed_results:
            traces.extend(fixed_rows)
            if save_policies:
                policy_rows.extend(fixed_policy_rows)
    else:
        for index, (
            abstraction,
            optimal_abstract_q,
            abstraction_error,
            fixed_trace_sweeps,
            fixed_evaluation_sweeps,
        ) in enumerate(fixed_specs):
            fixed_rows = AN.trace_fixed(
                mdp=mdp,
                abstraction=abstraction,
                optimal_abstract_q=optimal_abstract_q,
                optimal_concrete_q=optimal_concrete_q,
                optimal_concrete_values=optimal_concrete_values,
                abstraction_error=abstraction_error,
                max_sweeps=fixed_trace_sweeps,
                evaluation_sweeps=fixed_evaluation_sweeps,
                policy_value=policy_value,
                policy_rows=policy_rows if save_policies else None,
                encoder_path=(
                    fixed_policy_encoder_paths[index]
                    if fixed_policy_encoder_paths is not None
                    else None
                ),
            )
            traces.extend(fixed_rows)

    # 3) Collect adaptive trace rows and final adaptive summary.
    adaptive_abstractions = adaptive_family["abstractions"]
    adaptive_abstraction_errors = adaptive_family["abstraction_errors"]

    adaptive_min_sweep_units = min(int(abstraction.num_abstract) for abstraction in adaptive_abstractions)
    adaptive_trace_sweep_upper_bound = int(
        ceil(max_budget_units / float(adaptive_min_sweep_units))
    )
    adaptive_evaluation_sweeps = AN.build_evaluation_sweeps(
        max_sweeps=adaptive_trace_sweep_upper_bound,
        eval_interval=eval_interval,
    )
    adaptive_ladder = AD.AdaptiveLadder(
        abstractions=adaptive_abstractions,
        abstraction_errors=adaptive_abstraction_errors,
    )
    _log_progress(
        logger,
        (
            "Tracing adaptive controller over "
            f"{adaptive_trace_sweep_upper_bound} sweep slots "
            f"({len(adaptive_evaluation_sweeps)} checkpoints)"
        ),
    )

    adaptive_rows, adaptive_summary = AN.trace_adaptive(
        mdp=mdp,
        ladder=adaptive_ladder,
        optimal_concrete_q=optimal_concrete_q,
        optimal_concrete_values=optimal_concrete_values,
        backup_budget_units=max_budget_units,
        max_trace_sweeps=adaptive_trace_sweep_upper_bound,
        evaluation_sweeps=adaptive_evaluation_sweeps,
        policy_value=policy_value,
        policy_rows=policy_rows if save_policies else None,
        encoder_paths_by_stage=adaptive_policy_encoder_paths,
    )
    traces.extend(adaptive_rows)

    # 4) Return all trace rows plus the adaptive summary artifact.
    return traces, adaptive_summary, policy_rows


def _build_summary(
    *,
    baseline: Dict[str, object],
    fixed_family_rows: Sequence[Dict[str, object]],
    fixed_beta_schedule: Sequence[float],
    adaptive_beta_schedule: Sequence[float],
    adaptive_summary: Dict[str, object],
    traces: Sequence[Dict[str, float | str]],
    abstract_alphabet_size: int,
    max_sweeps: int,
    eval_interval: int,
    save_policies: bool,
    num_workers: int,
    abstraction_solver: str,
    abstraction_error_mode: str,
    ba_max_outer: int,
    ba_max_inner: int,
    summary_mutator: Callable[[Dict[str, object]], None] | None,
) -> Dict[str, object]:

    """Assemble the saved summary payload for one run."""
    exact_policy_values: Dict[str, float] = {}
    best_compute_by_label: Dict[str, float] = {}
    for row in traces:
        label = str(row["method_label"])
        if "policy_return" not in row:
            continue
        bellman_compute = float(row.get("bellman_backup_units", -np.inf))
        candidate = float(row["policy_return"])
        if bellman_compute >= best_compute_by_label.get(label, -np.inf):
            best_compute_by_label[label] = bellman_compute
            exact_policy_values[label] = candidate

    config_payload = {
        "num_states": int(baseline["mdp"].num_states),
        "num_actions": int(baseline["mdp"].num_actions),
        "num_state_action_pairs": int(baseline["num_pairs"]),
        "beta_schedule": [float(beta) for beta in fixed_beta_schedule],
        "adaptive_beta_schedule": [float(beta) for beta in adaptive_beta_schedule],
        "abstract_alphabet_size_cap": int(abstract_alphabet_size),
        "max_sweeps": int(max_sweeps),
        "eval_interval": int(eval_interval),
        "save_policies": bool(save_policies),
        "num_workers": int(num_workers),
        "abstraction_solver": str(abstraction_solver),
        "abstraction_error_mode": str(abstraction_error_mode),
        "reference_distribution": "uniform over concrete state-action pairs",
        "abstraction_space": "state_action_pairs",
        "planner": (
            "deterministic-decoder specialization of the decoder-based "
            "operator Fbar_eta = L F Gamma, with soft encoder grounding "
            "and deterministic decoder representatives"
        ),
        "ba_solver_limits": {
            "max_outer": int(ba_max_outer),
            "max_inner": int(ba_max_inner),
        },
    }
    summary_payload = {
        "config": config_payload,
        "adaptive_summary": {
            "switch_updates": [int(value) for value in adaptive_summary["switch_updates"]],
            "switch_betas": [float(value) for value in adaptive_summary["switch_betas"]],
            "final_beta": float(adaptive_summary["final_beta"]),
            "final_concrete_q_error": float(adaptive_summary["final_concrete_q_error"]),
            "final_concrete_value_error": float(adaptive_summary["final_concrete_value_error"]),
            "final_policy_return": float(adaptive_summary["final_policy_return"]),
        },
        "fixed_family_table": [dict(row) for row in fixed_family_rows],
        "metric_summary": {
            "max_state_action_distance": float(np.max(baseline["distortion"])),
        },
        "exact_policy_values": exact_policy_values,
        "notes": {
            "policy_value_estimator": (
                "exact tabular evaluation: solve (I - gamma P^pi) V = r^pi in closed "
                "form on the original MDP"
            ),
            "metric": "experiment-specific metric description supplied by the summary mutator",
            "grounding": (
                "Gamma barQ is formed by averaging barQ(z) under the soft encoder "
                "nu(z|s,a); abstract updates are then read off at the deterministic "
                "BA decoder representatives g(z)"
            ),
            "abstraction_error": (
                "scalar abstraction error used in traces and adaptive switching: "
                "average(mu-uniform) encoder/decoder distortion"
                if abstraction_error_mode == "average"
                else "max per-pair encoder/decoder distortion"
            ),
            "sweep_budget": (
                "max_sweeps denotes the compute budget of that many full-MDP "
                "Bellman sweeps; fixed and adaptive traces expand this budget "
                "internally using their own per-sweep backup costs"
            ),
        },
    }
    if summary_mutator is not None:
        summary_mutator(summary_payload)
    return summary_payload


########################################################
# EXPERIMENT RUNNER
def run(
    mdp,
    distortion: Array,
    abstract_alphabet_size: int,
    beta_schedule: Sequence[float],
    adaptive_beta_schedule: Sequence[float] | None,
    abstraction_error_mode: str,
    ba_max_outer: int,
    ba_max_inner: int,
    max_sweeps: int,
    eval_interval: int,
    output_dir: Path,
    save_policies: bool = False,
    num_workers: int = 1,
    abstraction_solver: str = "flat",
    summary_mutator: Callable[[Dict[str, object]], None] | None = None,
    verbose: bool = True,
) -> None:
    """Run the full model-based state-action experiment workflow."""

    start_time = perf_counter()

    def logger(message: str) -> None:
        if verbose:
            print(f"[run] {message}")

    # 1) Build concrete MDP baseline (optimal values and Q-values)
    logger("Building concrete baseline.")
    baseline = _build_baseline(mdp=mdp,distortion=distortion)

    # 2) Fit fixed-beta abstraction family and adaptive family
    if abstract_alphabet_size <= 0:
        abstract_alphabet_size = baseline["num_pairs"]
    logger(
        f"Fitting fixed abstraction ladder with alphabet cap {abstract_alphabet_size} "
        f"across {len(beta_schedule)} betas."
    )
    fixed_family = _build_abstractions(
        baseline=baseline,
        abstract_alphabet_size=abstract_alphabet_size,
        beta_schedule=beta_schedule,
        abstraction_error_mode=abstraction_error_mode,
        ba_max_outer=ba_max_outer,
        ba_max_inner=ba_max_inner,
        abstraction_solver=abstraction_solver,
        num_workers=num_workers,
        logger=logger,
        family_name="fixed ladder",
    )

    if _beta_schedules_match(beta_schedule, adaptive_beta_schedule):
        logger("Reusing the fixed ladder for the adaptive controller.")
        adaptive_family = fixed_family
    else:
        logger(
            f"Fitting adaptive abstraction ladder across {len(adaptive_beta_schedule)} betas."
        )
        adaptive_family = _build_abstractions(
            baseline=baseline,
            abstract_alphabet_size=abstract_alphabet_size,
            beta_schedule=adaptive_beta_schedule,
            abstraction_error_mode=abstraction_error_mode,
            ba_max_outer=ba_max_outer,
            ba_max_inner=ba_max_inner,
            abstraction_solver=abstraction_solver,
            num_workers=num_workers,
            logger=logger,
            family_name="adaptive ladder",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_policy_encoder_paths: List[str] | None = None
    adaptive_policy_encoder_paths: List[str] | None = None
    if save_policies:
        logger("Saving abstraction encoders for policy checkpoints.")
        fixed_policy_encoder_paths = _save_policy_encoders(
            output_dir,
            "fixed",
            fixed_family["abstractions"],
        )
        adaptive_policy_encoder_paths = (
            fixed_policy_encoder_paths
            if adaptive_family is fixed_family
            else _save_policy_encoders(
                output_dir,
                "adaptive",
                adaptive_family["abstractions"],
            )
        )

    # 3) Produce the saved traces for fixed, base, and adaptive methods.
    logger("Running Bellman traces.")
    traces, adaptive_summary, policy_rows = _run_traces(
        baseline=baseline,
        fixed_family=fixed_family,
        adaptive_family=adaptive_family,
        max_sweeps=max_sweeps,
        eval_interval=eval_interval,
        save_policies=save_policies,
        num_workers=num_workers,
        logger=logger,
        fixed_policy_encoder_paths=fixed_policy_encoder_paths,
        adaptive_policy_encoder_paths=adaptive_policy_encoder_paths,
    )

    # 4) Aggregate outputs and persist experiment artifacts
    logger("Assembling summary and writing outputs.")
    abstraction_solver = AB.normalize_solver_kind(abstraction_solver)
    if abstraction_solver == "lambda":
        _save_state_encoders(output_dir, "fixed", fixed_family["abstractions"])
        if adaptive_family is not fixed_family:
            _save_state_encoders(output_dir, "adaptive", adaptive_family["abstractions"])

    fixed_family_rows = _build_fixed_family_report_rows(
        baseline=baseline,
        fixed_family=fixed_family,
    )
    summary_payload = _build_summary(
        baseline=baseline,
        fixed_family_rows=fixed_family_rows,
        fixed_beta_schedule=fixed_family["beta_schedule"],
        adaptive_beta_schedule=adaptive_family["beta_schedule"],
        adaptive_summary=adaptive_summary,
        traces=traces,
        abstract_alphabet_size=abstract_alphabet_size,
        max_sweeps=max_sweeps,
        eval_interval=eval_interval,
        save_policies=save_policies,
        num_workers=num_workers,
        abstraction_solver=abstraction_solver,
        abstraction_error_mode=abstraction_error_mode,
        ba_max_outer=ba_max_outer,
        ba_max_inner=ba_max_inner,
        summary_mutator=summary_mutator,
    )

    final_results = {
        "output_dir": str(output_dir),
        "adaptive_summary": {
            "switch_updates": [int(x) for x in adaptive_summary["switch_updates"]],
            "switch_betas": [float(x) for x in adaptive_summary["switch_betas"]],
            "final_beta": float(adaptive_summary["final_beta"]),
            "final_concrete_q_error": float(adaptive_summary["final_concrete_q_error"]),
            "final_concrete_value_error": float(adaptive_summary["final_concrete_value_error"]),
            "final_policy_return": float(adaptive_summary["final_policy_return"]),
        },
    }
    OUT.save_rows(output_dir / "traces.csv", traces)
    policies_path = output_dir / "policies.csv"
    if save_policies:
        OUT.save_rows(policies_path, policy_rows)
    elif policies_path.exists():
        policies_path.unlink()
    OUT.save_json(output_dir / "summary.json", summary_payload)
    OUT.save_json(output_dir / "results.json", final_results)

    print(f"Finished. Results saved to {output_dir}")
    logger(f"Done in {perf_counter() - start_time:.1f}s.")

    return
