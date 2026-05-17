"""Adaptive abstraction controller for the state-action experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core import abstraction as AB
from core import planning as PL
from core.abstraction import Array, TabularMDP


def _values_from_flat_q(mdp: TabularMDP, flat_q: Array) -> Array:
    """Collapse flattened concrete Q-values into concrete state values by maximization."""
    q_matrix = np.asarray(flat_q, dtype=float).reshape(mdp.num_states, mdp.num_actions)
    return np.max(q_matrix, axis=1)


@dataclass(frozen=True)
class AdaptiveLadder:
    """One ordered abstraction ladder used by the adaptive controller."""

    abstractions: Sequence[AB.StateActionAbstraction]
    abstraction_errors: Sequence[float]


@dataclass
class AdaptiveState:
    """Mutable controller state as it progresses through the abstraction ladder."""

    stage_index: int  # Current abstraction level index in the ladder.
    abstract_q: Array  # Current Q iterate in the active abstract space.
    cumulative_backup_units: float  # Total Bellman backup cost accumulated so far.
    switch_updates: list[int]  # Sweep indices where the controller switched levels.
    switch_betas: list[float]  # Beta values active at each recorded switch.


@dataclass(frozen=True)
class AdaptiveSnapshot:
    """One recorded controller snapshot for the current Bellman update."""

    sweep: int
    stage_index: int
    beta: float
    active_abstracts: int
    abstraction_error: float
    abstract_q: Array
    grounded_q: Array
    next_abstract_q: Array
    abstract_residual: float
    concrete_q_error: float
    bellman_backup_units: float


@dataclass(frozen=True)
class AdaptiveRun:
    """Recorded adaptive controller snapshots plus the final mutable state."""

    snapshots: list[AdaptiveSnapshot]
    final_state: AdaptiveState

    @property
    def final_snapshot(self) -> AdaptiveSnapshot:
        """Return the final recorded controller snapshot."""
        return self.snapshots[-1]


def initialize_adaptive_state(ladder: AdaptiveLadder) -> AdaptiveState:
    """Start at the coarsest abstraction with a zero abstract Q iterate."""
    first_abstraction = ladder.abstractions[0]
    return AdaptiveState(
        stage_index=0,
        abstract_q=np.zeros(first_abstraction.num_abstract, dtype=float),
        cumulative_backup_units=0.0,
        switch_updates=[],
        switch_betas=[float(first_abstraction.beta)],
    )


def build_adaptive_snapshot(
    mdp: TabularMDP,
    ladder: AdaptiveLadder,
    state: AdaptiveState,
    optimal_concrete_q: Array,
    sweep: int,
) -> AdaptiveSnapshot:
    """Build one adaptive snapshot from the current controller state."""
    abstraction = ladder.abstractions[state.stage_index]
    grounded_q = AB.ground_state_action_abstract_q(abstraction, state.abstract_q)
    next_abstract_q = PL.abstract_state_action_bellman_update(
        mdp,
        abstraction,
        state.abstract_q,
    )
    next_grounded_q = AB.ground_state_action_abstract_q(abstraction, next_abstract_q)
    grounded_values = _values_from_flat_q(mdp, grounded_q)
    next_grounded_values = _values_from_flat_q(mdp, next_grounded_q)
    abstract_residual = float(np.max(np.abs(next_grounded_values - grounded_values)))
    concrete_q_error = float(np.max(np.abs(grounded_q - optimal_concrete_q)))
    return AdaptiveSnapshot(
        sweep=int(sweep),
        stage_index=int(state.stage_index),
        beta=float(abstraction.beta),
        active_abstracts=int(abstraction.num_abstract),
        abstraction_error=float(ladder.abstraction_errors[state.stage_index]),
        abstract_q=np.asarray(state.abstract_q, dtype=float).copy(),
        grounded_q=np.asarray(grounded_q, dtype=float),
        next_abstract_q=np.asarray(next_abstract_q, dtype=float),
        abstract_residual=abstract_residual,
        concrete_q_error=concrete_q_error,
        bellman_backup_units=float(state.cumulative_backup_units),
    )


def should_switch_stage(snapshot: AdaptiveSnapshot, ladder: AdaptiveLadder) -> bool:
    """Decide whether the current stage has reached its abstraction-error floor."""
    return (
        snapshot.stage_index < len(ladder.abstractions) - 1
        and snapshot.abstract_residual <= snapshot.abstraction_error
    )


def find_next_stage(
    mdp: TabularMDP,
    ladder: AdaptiveLadder,
    current_stage_index: int,
    grounded_q: Array,
) -> tuple[int, Array]:
    """Choose the first later abstraction whose transferred iterate still has a large value residual."""
    candidate_stage_index = current_stage_index + 1
    last_projected_q: Array | None = None

    while candidate_stage_index < len(ladder.abstractions):
        candidate_abstraction = ladder.abstractions[candidate_stage_index]
        candidate_abstract_q = AB.lift_state_action_concrete_q(
            candidate_abstraction,
            grounded_q,
        )
        candidate_next_abstract_q = PL.abstract_state_action_bellman_update(
            mdp,
            candidate_abstraction,
            candidate_abstract_q,
        )
        candidate_grounded_q = AB.ground_state_action_abstract_q(
            candidate_abstraction,
            candidate_abstract_q,
        )
        candidate_next_grounded_q = AB.ground_state_action_abstract_q(
            candidate_abstraction,
            candidate_next_abstract_q,
        )
        candidate_residual = float(
            np.max(
                np.abs(
                    _values_from_flat_q(mdp, candidate_next_grounded_q)
                    - _values_from_flat_q(mdp, candidate_grounded_q)
                )
            )
        )
        last_projected_q = candidate_abstract_q
        if candidate_residual > float(ladder.abstraction_errors[candidate_stage_index]):
            return candidate_stage_index, candidate_abstract_q
        candidate_stage_index += 1

    if last_projected_q is None:
        raise ValueError("Adaptive stage search requires a later abstraction.")
    return len(ladder.abstractions) - 1, last_projected_q


def step_adaptive_state(
    mdp: TabularMDP,
    ladder: AdaptiveLadder,
    state: AdaptiveState,
    snapshot: AdaptiveSnapshot,
) -> AdaptiveState:
    """Advance the adaptive controller by one Bellman update."""
    current_abstraction = ladder.abstractions[state.stage_index]
    next_stage_index = state.stage_index
    next_abstract_q = np.asarray(snapshot.next_abstract_q, dtype=float)
    switch_updates = list(state.switch_updates)
    switch_betas = list(state.switch_betas)
    updated_grounded_q = AB.ground_state_action_abstract_q(
        current_abstraction,
        next_abstract_q,
    )

    if should_switch_stage(snapshot, ladder):
        next_stage_index, next_abstract_q = find_next_stage(
            mdp,
            ladder,
            state.stage_index,
            updated_grounded_q,
        )
        switch_updates.append(int(snapshot.sweep) + 1)
        switch_betas.append(float(ladder.abstractions[next_stage_index].beta))

    return AdaptiveState(
        stage_index=int(next_stage_index),
        abstract_q=np.asarray(next_abstract_q, dtype=float),
        cumulative_backup_units=(
            float(state.cumulative_backup_units) + float(current_abstraction.num_abstract)
        ),
        switch_updates=switch_updates,
        switch_betas=switch_betas,
    )


def run_adaptive_controller(
    mdp: TabularMDP,
    ladder: AdaptiveLadder,
    optimal_concrete_q: Array,
    backup_budget_units: float,
    max_trace_sweeps: int,
    record_sweeps: Sequence[int],
) -> AdaptiveRun:
    """Run the adaptive abstraction controller and record the beta used at each update."""

    state = initialize_adaptive_state(ladder)
    record_sweep_set = set(int(sweep) for sweep in record_sweeps)
    snapshots: list[AdaptiveSnapshot] = []

    initial_snapshot = build_adaptive_snapshot(
        mdp=mdp,
        ladder=ladder,
        state=state,
        optimal_concrete_q=optimal_concrete_q,
        sweep=-1,
    )
    if -1 in record_sweep_set or 0 in record_sweep_set:
        snapshots.append(initial_snapshot)

    if max_trace_sweeps == 0 or state.cumulative_backup_units >= float(backup_budget_units) - 1e-12:
        if not snapshots:
            snapshots.append(initial_snapshot)
        return AdaptiveRun(snapshots=snapshots, final_state=state)

    for sweep in range(max_trace_sweeps):
        current_snapshot = build_adaptive_snapshot(
            mdp=mdp,
            ladder=ladder,
            state=state,
            optimal_concrete_q=optimal_concrete_q,
            sweep=sweep,
        )
        next_state = step_adaptive_state(
            mdp=mdp,
            ladder=ladder,
            state=state,
            snapshot=current_snapshot,
        )
        did_switch = int(next_state.stage_index) != int(state.stage_index)
        reached_budget = (
            next_state.cumulative_backup_units >= float(backup_budget_units) - 1e-12
        )
        is_final_sweep = sweep == max_trace_sweeps - 1

        if sweep in record_sweep_set or did_switch or reached_budget or is_final_sweep:
            snapshots.append(current_snapshot)
        state = next_state
        if reached_budget or is_final_sweep:
            break

    return AdaptiveRun(snapshots=snapshots, final_state=state)
