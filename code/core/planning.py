"""Bellman updates and fixed-point solvers for tabular control."""

from __future__ import annotations
import numpy as np
from core.abstraction import (
    Array,
    TabularMDP,
    ground_state_action_abstract_q,
)


def solve_optimal_values(
    mdp: TabularMDP,
    tol: float = 1e-12,
    max_sweeps: int = 20000,
) -> Array:
    """Solve for the optimal state-value fixed point by value iteration."""
    values = np.zeros(mdp.num_states, dtype=float)
    for _ in range(max_sweeps):
        q_values = mdp.rewards + mdp.gamma * np.einsum("sak,k->sa", mdp.transitions, values)
        updated = np.max(q_values, axis=1)
        if float(np.max(np.abs(updated - values))) < tol:
            return updated
        values = updated
    return values


def bellman_update(mdp: TabularMDP, grounded_q: Array) -> Array:
    """Apply one concrete Bellman optimality update in flattened pair space."""
    q_matrix = np.asarray(grounded_q, dtype=float).reshape(mdp.num_states, mdp.num_actions)
    successor_values = np.max(q_matrix, axis=1)
    updated = mdp.rewards + mdp.gamma * np.einsum("sak,k->sa", mdp.transitions, successor_values)
    return np.asarray(updated, dtype=float).reshape(-1)


def abstract_state_action_bellman_update(
    mdp: TabularMDP,
    abstraction,
    abstract_q: Array,
) -> Array:
    """Apply one abstract state-action Bellman update and return abstract Q-values."""
    grounded_q = ground_state_action_abstract_q(abstraction, abstract_q)
    updated_flat = bellman_update(mdp, grounded_q)
    return np.asarray(updated_flat[abstraction.decoder], dtype=float)


def solve_optimal_abstract_q(
    mdp: TabularMDP,
    abstraction,
    tol: float = 1e-12,
    max_sweeps: int = 20000,
) -> Array:
    """Solve the abstract state-action fixed point of \bar F_\eta."""
    abstract_q = np.zeros(abstraction.num_abstract, dtype=float)
    for _ in range(max_sweeps):
        updated = abstract_state_action_bellman_update(mdp, abstraction, abstract_q)
        if float(np.max(np.abs(updated - abstract_q))) < tol:
            return updated
        abstract_q = updated
    return abstract_q
