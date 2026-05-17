"""Evaluation and trace-building helpers for the state-action experiments."""

from __future__ import annotations
import json
from typing import Callable, Dict, List, Sequence
import numpy as np
from core import adaptive as AD
from core import abstraction as AB
from core import planning as PL
from core.abstraction import Array, TabularMDP


def evaluate_policy_value_exact(mdp: TabularMDP, policy: Array) -> float:
    """
    Evaluate a deterministic policy exactly under the tabular model, by solving
    (I - gamma * P^π) v = r^π
    Then returns the mean value over all states as the average return for the policy.
    """
    states = np.arange(mdp.num_states, dtype=int)
    rewards = mdp.rewards[states, policy]
    transitions = mdp.transitions[states, policy, :]
    system = np.eye(mdp.num_states, dtype=float) - mdp.gamma * transitions
    values = np.linalg.solve(system, rewards)
    return float(np.mean(values))


def build_evaluation_sweeps(max_sweeps: int, eval_interval: int) -> List[int]:
    """Select evaluation sweeps at a fixed interval, always including endpoints."""
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    sweeps = list(range(0, max_sweeps + 1, int(eval_interval)))
    if not sweeps or sweeps[0] != 0:
        sweeps.insert(0, 0)
    if sweeps[-1] != max_sweeps:
        sweeps.append(max_sweeps)
    return sweeps


def _greedy_policy_from_flat_q(mdp: TabularMDP, flat_q: Array) -> Array:
    """Extract a greedy state policy from flattened concrete pair-values."""
    q_matrix = np.asarray(flat_q, dtype=float).reshape(mdp.num_states, mdp.num_actions)
    return np.argmax(q_matrix, axis=1)


def _values_from_flat_q(mdp: TabularMDP, flat_q: Array) -> Array:
    """Collapse flattened concrete Q-values into concrete state values by maximization."""
    q_matrix = np.asarray(flat_q, dtype=float).reshape(mdp.num_states, mdp.num_actions)
    return np.max(q_matrix, axis=1)


def _build_trace_row(
    *,
    method_label: str,
    method_type: str,
    sweep: int,
    beta: float,
    active_abstracts: int,
    abstraction_error: float,
    abstract_q_error: float | None,
    abstract_value_error: float | None,
    concrete_q_error: float,
    concrete_value_error: float,
    bellman_backup_units: float,
    policy_return: float,
) -> Dict[str, float | str]:
    """Build one saved trace row shared by fixed, base, and adaptive runs."""
    row: Dict[str, float | str] = {
        "method_label": method_label,
        "method_type": method_type,
        "sweep": float(sweep),
        "beta": float(beta),
        "active_abstracts": float(active_abstracts),
        "abstraction_error": float(abstraction_error),
        "concrete_q_error": float(concrete_q_error),
        "concrete_value_error": float(concrete_value_error),
        "bellman_backup_units": float(bellman_backup_units),
        "policy_return": float(policy_return),
    }
    if abstract_q_error is not None:
        row["abstract_q_error"] = float(abstract_q_error)
    if abstract_value_error is not None:
        row["abstract_value_error"] = float(abstract_value_error)
    return row


def _build_policy_row(
    *,
    method_label: str,
    method_type: str,
    sweep: int,
    beta: float,
    bellman_backup_units: float,
    policy: Array,
    active_abstracts: int | None = None,
    stage_index: int | None = None,
    encoder_path: str | None = None,
) -> Dict[str, object]:
    """Build one serialized-policy checkpoint row."""
    row: Dict[str, object] = {
        "method_label": method_label,
        "method_type": method_type,
        "sweep": float(sweep),
        "beta": float(beta),
        "bellman_backup_units": float(bellman_backup_units),
        "policy_json": json.dumps(
            np.asarray(policy, dtype=int).tolist(),
            separators=(",", ":"),
        ),
    }
    if active_abstracts is not None:
        row["active_abstracts"] = float(active_abstracts)
    if stage_index is not None:
        row["stage_index"] = float(stage_index)
    if encoder_path is not None:
        row["encoder_path"] = str(encoder_path)
    return row


def trace_base(
    mdp: TabularMDP,
    optimal_concrete_q: Array,
    optimal_concrete_values: Array,
    max_sweeps: int,
    evaluation_sweeps: Sequence[int],
    policy_value: Callable[[TabularMDP, Array], float],
    policy_rows: List[Dict[str, object]] | None = None,
) -> List[Dict[str, float | str]]:
    """Trace the base concrete Q-iteration baseline at selected sweeps."""

    rows: List[Dict[str, float | str]] = []
    sweep_cost = float(mdp.num_state_action_pairs)
    concrete_q = np.zeros_like(optimal_concrete_q)
    evaluation_sweep_set = set(int(sweep) for sweep in evaluation_sweeps)

    for sweep in range(max_sweeps + 1):
        is_final_sweep = sweep == max_sweeps
        if sweep in evaluation_sweep_set or is_final_sweep:
            error = float(np.max(np.abs(concrete_q - optimal_concrete_q)))
            concrete_values = _values_from_flat_q(mdp, concrete_q)
            value_error = float(np.max(np.abs(concrete_values - optimal_concrete_values)))
            policy = _greedy_policy_from_flat_q(mdp, concrete_q)
            if policy_rows is not None:
                policy_rows.append(
                    _build_policy_row(
                        method_label="Base MDP",
                        method_type="base",
                        sweep=sweep,
                        beta=0.0,
                        bellman_backup_units=sweep * sweep_cost,
                        active_abstracts=int(mdp.num_state_action_pairs),
                        policy=policy,
                    )
                )
            rows.append(
                _build_trace_row(
                    method_label="Base MDP",
                    method_type="base",
                    sweep=sweep,
                    bellman_backup_units=sweep * sweep_cost,
                    beta=0.0,
                    active_abstracts=int(mdp.num_state_action_pairs),
                    abstraction_error=0.0,
                    abstract_q_error=error,
                    abstract_value_error=value_error,
                    concrete_q_error=error,
                    concrete_value_error=value_error,
                    policy_return=float(policy_value(mdp, policy)),
                )
            )
        if is_final_sweep:
            break
        concrete_q = PL.bellman_update(mdp, concrete_q)

    return rows


def trace_fixed(
    mdp: TabularMDP,
    abstraction: AB.StateActionAbstraction,
    optimal_abstract_q: Array,
    optimal_concrete_q: Array,
    optimal_concrete_values: Array,
    abstraction_error: float,
    max_sweeps: int,
    evaluation_sweeps: Sequence[int],
    policy_value: Callable[[TabularMDP, Array], float],
    policy_rows: List[Dict[str, object]] | None = None,
    encoder_path: str | None = None,
) -> List[Dict[str, float | str]]:
    """Trace fixed-beta updates at selected evaluation sweeps."""

    rows: List[Dict[str, float | str]] = []
    beta = float(abstraction.beta)
    active_abstracts = abstraction.num_abstract
    abstract_q = np.zeros(active_abstracts, dtype=float)
    optimal_grounded_q = AB.ground_state_action_abstract_q(abstraction, optimal_abstract_q)
    optimal_grounded_values = _values_from_flat_q(mdp, optimal_grounded_q)
    evaluation_sweep_set = set(int(sweep) for sweep in evaluation_sweeps)
    sweep_cost = float(abstraction.num_abstract)
    method_label = rf"Fixed $\beta={beta:g}$"

    for sweep in range(max_sweeps + 1):
        is_final_sweep = sweep == max_sweeps
        grounded_q = AB.ground_state_action_abstract_q(abstraction, abstract_q)
        abstract_q_error = float(np.max(np.abs(abstract_q - optimal_abstract_q)))
        concrete_q_error = float(np.max(np.abs(grounded_q - optimal_concrete_q)))
        grounded_values = _values_from_flat_q(mdp, grounded_q)
        abstract_value_error = float(np.max(np.abs(grounded_values - optimal_grounded_values)))
        concrete_value_error = float(np.max(np.abs(grounded_values - optimal_concrete_values)))

        if sweep in evaluation_sweep_set or is_final_sweep:
            policy = _greedy_policy_from_flat_q(mdp, grounded_q)
            if policy_rows is not None:
                policy_rows.append(
                    _build_policy_row(
                        method_label=method_label,
                        method_type="fixed",
                        sweep=sweep,
                        beta=beta,
                        bellman_backup_units=float(sweep) * sweep_cost,
                        active_abstracts=active_abstracts,
                        policy=policy,
                        encoder_path=encoder_path,
                    )
                )
            rows.append(
                _build_trace_row(
                    method_label=method_label,
                    method_type="fixed",
                    sweep=sweep,
                    beta=beta,
                    active_abstracts=active_abstracts,
                    abstraction_error=abstraction_error,
                    abstract_q_error=abstract_q_error,
                    abstract_value_error=abstract_value_error,
                    concrete_q_error=concrete_q_error,
                    concrete_value_error=concrete_value_error,
                    bellman_backup_units=float(sweep) * sweep_cost,
                    policy_return=float(policy_value(mdp, policy)),
                )
            )
        if is_final_sweep:
            break
        abstract_q = PL.abstract_state_action_bellman_update(mdp, abstraction, abstract_q)

    return rows


def trace_adaptive(
    mdp: TabularMDP,
    ladder: AD.AdaptiveLadder,
    optimal_concrete_q: Array,
    optimal_concrete_values: Array,
    backup_budget_units: float,
    max_trace_sweeps: int,
    evaluation_sweeps: Sequence[int],
    policy_value: Callable[[TabularMDP, Array], float],
    policy_rows: List[Dict[str, object]] | None = None,
    encoder_paths_by_stage: Sequence[str | None] | None = None,
) -> tuple[List[Dict[str, float | str]], Dict[str, object]]:
    """Trace the adaptive beta controller until it exhausts the base-sweep budget."""
    adaptive_run = AD.run_adaptive_controller(
        mdp=mdp,
        ladder=ladder,
        optimal_concrete_q=optimal_concrete_q,
        backup_budget_units=backup_budget_units,
        max_trace_sweeps=max_trace_sweeps,
        record_sweeps=evaluation_sweeps,
    )

    rows: List[Dict[str, float | str]] = []
    for snapshot in adaptive_run.snapshots:
        policy = _greedy_policy_from_flat_q(mdp, snapshot.grounded_q)
        if policy_rows is not None:
            encoder_path = None
            if encoder_paths_by_stage is not None:
                stage_index = int(snapshot.stage_index)
                if 0 <= stage_index < len(encoder_paths_by_stage):
                    encoder_path = encoder_paths_by_stage[stage_index]
            policy_rows.append(
                _build_policy_row(
                    method_label="Adaptive",
                    method_type="adaptive",
                    sweep=int(snapshot.sweep),
                    beta=float(snapshot.beta),
                    bellman_backup_units=float(snapshot.bellman_backup_units),
                    active_abstracts=int(snapshot.active_abstracts),
                    stage_index=int(snapshot.stage_index),
                    policy=policy,
                    encoder_path=encoder_path,
                )
            )
        policy_return = policy_value(mdp, policy)
        grounded_values = _values_from_flat_q(mdp, snapshot.grounded_q)
        concrete_value_error = float(np.max(np.abs(grounded_values - optimal_concrete_values)))
        row = _build_trace_row(
            method_label="Adaptive",
            method_type="adaptive",
            sweep=int(snapshot.sweep),
            beta=float(snapshot.beta),
            active_abstracts=int(snapshot.active_abstracts),
            abstraction_error=float(snapshot.abstraction_error),
            abstract_q_error=None,
            abstract_value_error=None,
            concrete_q_error=float(snapshot.concrete_q_error),
            concrete_value_error=concrete_value_error,
            bellman_backup_units=float(snapshot.bellman_backup_units),
            policy_return=float(policy_return),
        )
        row["abstract_residual"] = float(snapshot.abstract_residual)
        row["stage_index"] = float(snapshot.stage_index)
        rows.append(row)

    final_snapshot = adaptive_run.final_snapshot
    summary = {
        "switch_updates": [int(value) for value in adaptive_run.final_state.switch_updates],
        "switch_betas": [float(value) for value in adaptive_run.final_state.switch_betas],
        "final_beta": float(final_snapshot.beta),
        "final_concrete_q_error": float(final_snapshot.concrete_q_error),
        "final_concrete_value_error": float(np.max(np.abs(_values_from_flat_q(mdp, final_snapshot.grounded_q) - optimal_concrete_values))),
        "final_policy_return": float(rows[-1]["policy_return"]),
    }
    return rows, summary
