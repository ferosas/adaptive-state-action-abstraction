"""Tabular Taxi environment matching Taxi-v3 dynamics as closely as possible.

The only deliberate deviation from the classic Gym/Gymnasium Taxi-v3 setup is
that we add a single absorbing terminal state so the episodic task becomes a
discounted tabular MDP compatible with the existing experiment pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core import abstraction as AB


Array = np.ndarray
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


TAXI_MAP: Tuple[str, ...] = (
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
)
GRID_SIZE = 5
LOCATIONS: Tuple[Tuple[int, int], ...] = (
    (0, 0),  # R
    (0, 4),  # G
    (4, 0),  # Y
    (4, 3),  # B
)
LOCATION_LABELS: Tuple[str, ...] = ("R", "G", "Y", "B")
PASSENGER_IN_TAXI = 4
NONTERMINAL_STATE_COUNT = GRID_SIZE * GRID_SIZE * (len(LOCATIONS) + 1) * len(LOCATIONS)
ACTION_LABELS = ["south", "north", "east", "west", "pickup", "dropoff"]
ABSORBING_LABEL = (-1, -1, -1, -1)


def encode_state(row: int, col: int, passenger: int, destination: int) -> int:
    """Pack Taxi state factors into the canonical tabular state index."""
    state = row
    state *= GRID_SIZE
    state += col
    state *= len(LOCATIONS) + 1
    state += passenger
    state *= len(LOCATIONS)
    state += destination
    return int(state)


def decode_state(state: int) -> Tuple[int, int, int, int]:
    """Invert ``encode_state`` back into row, column, passenger, and destination."""
    destination = state % len(LOCATIONS)
    state //= len(LOCATIONS)
    passenger = state % (len(LOCATIONS) + 1)
    state //= len(LOCATIONS) + 1
    col = state % GRID_SIZE
    state //= GRID_SIZE
    row = state
    return int(row), int(col), int(passenger), int(destination)


def passenger_label(passenger: int) -> str:
    """Convert the passenger factor into a readable label."""
    if passenger == PASSENGER_IN_TAXI:
        return "in_taxi"
    return LOCATION_LABELS[passenger]


def destination_label(destination: int) -> str:
    """Convert the destination factor into a readable label."""
    return LOCATION_LABELS[destination]


def encode_state_action(state: int, action: int, num_actions: int | None = None) -> int:
    """Flatten a concrete ``(state, action)`` pair into a single index."""
    if num_actions is None:
        num_actions = len(ACTION_LABELS)
    return int(state) * int(num_actions) + int(action)


def decode_state_action(
    pair_index: int,
    num_actions: int | None = None,
) -> Tuple[int, int]:
    """Invert ``encode_state_action`` back into ``(state, action)``."""
    if num_actions is None:
        num_actions = len(ACTION_LABELS)
    state = int(pair_index) // int(num_actions)
    action = int(pair_index) % int(num_actions)
    return state, action


def build_taxi_mdp(gamma: float = 0.99) -> AB.TabularMDP:
    """Build a discounted Taxi-v3 MDP with one absorbing terminal state."""
    num_nonterminal = NONTERMINAL_STATE_COUNT
    terminal_state = num_nonterminal
    num_states = num_nonterminal + 1
    num_actions = len(ACTION_LABELS)
    desc = np.asarray([list(row) for row in TAXI_MAP], dtype="U1")
    location_to_index = {loc: idx for idx, loc in enumerate(LOCATIONS)}

    transitions = np.zeros((num_states, num_actions, num_states), dtype=float)
    rewards = np.zeros((num_states, num_actions), dtype=float)
    state_labels: List[Tuple[int, int, int, int]] = []

    for state in range(num_nonterminal):
        row, col, passenger, destination = decode_state(state)
        state_labels.append((row, col, passenger, destination))
        taxi_loc = (row, col)

        for action in range(num_actions):
            next_row = row
            next_col = col
            next_passenger = passenger
            reward = -1.0
            next_state = state

            if action == 0:  # south
                next_row = min(row + 1, GRID_SIZE - 1)
            elif action == 1:  # north
                next_row = max(row - 1, 0)
            elif action == 2:  # east
                if desc[1 + row, 2 * col + 2] == ":":
                    next_col = min(col + 1, GRID_SIZE - 1)
            elif action == 3:  # west
                if desc[1 + row, 2 * col] == ":":
                    next_col = max(col - 1, 0)
            elif action == 4:  # pickup
                if passenger < len(LOCATIONS) and taxi_loc == LOCATIONS[passenger]:
                    next_passenger = PASSENGER_IN_TAXI
                else:
                    reward = -10.0
            elif action == 5:  # dropoff
                if passenger == PASSENGER_IN_TAXI and taxi_loc == LOCATIONS[destination]:
                    reward = 20.0
                    next_state = terminal_state
                elif passenger == PASSENGER_IN_TAXI and taxi_loc in location_to_index:
                    next_passenger = location_to_index[taxi_loc]
                else:
                    reward = -10.0
            else:
                raise ValueError(f"Unknown action {action}")

            if action <= 3 or action == 4 or (action == 5 and next_state != terminal_state):
                next_state = encode_state(
                    next_row,
                    next_col,
                    next_passenger,
                    destination,
                )

            transitions[state, action, next_state] = 1.0
            rewards[state, action] = reward

    state_labels.append(ABSORBING_LABEL)
    transitions[terminal_state, :, terminal_state] = 1.0
    rewards[terminal_state, :] = 0.0

    return AB.TabularMDP(
        transitions=transitions,
        rewards=rewards,
        gamma=gamma,
        state_labels=state_labels,
        action_labels=list(ACTION_LABELS),
    )


def compute_taxi_one_step_bisimulation_metric(mdp: AB.TabularMDP) -> Array:
    """Build the Taxi one-step pair distortion on concrete state-action pairs.

    The pair distortion is the one-step Bellman-compatible surrogate

      d_sa((s,a),(t,b))
        = |r(s,a)-r(t,b)| + gamma * TV(P(.|s,a), P(.|t,b)).
    """
    num_pairs = mdp.num_state_action_pairs
    num_states = mdp.num_states
    pair_rows = np.asarray(mdp.transitions, dtype=float).reshape(num_pairs, num_states)
    reward_flat = np.asarray(mdp.rewards, dtype=float).reshape(num_pairs)
    distortion = np.zeros((num_pairs, num_pairs), dtype=float)

    for start in range(num_pairs):
        reward_gap = np.abs(reward_flat[start] - reward_flat[start + 1 :])
        total_variation = 0.5 * np.sum(
            np.abs(pair_rows[start] - pair_rows[start + 1 :]),
            axis=1,
        )
        distortion[start, start + 1 :] = reward_gap + mdp.gamma * total_variation
        distortion[start + 1 :, start] = distortion[start, start + 1 :]

    np.fill_diagonal(distortion, 0.0)
    return distortion


def compute_taxi_fixed_point_bisimulation_metric(
    mdp: AB.TabularMDP,
    tol: float = 1e-8,
    max_iter: int = 2000,
    verbose: bool = True,
) -> Array:
    """Build the Taxi fixed-point pair metric on concrete `(state, action)` pairs.

    The fixed-point iterate keeps an internal state semimetric `d_s` only so
    the Wasserstein term can close on itself. The public output is just the
    converged pair metric `d_sa`.

      d_sa((s,a),(t,b))
        = |r(s,a) - r(t,b)| + gamma * W_{d_s}(P(.|s,a), P(.|t,b))

    The Taxi transition kernel is deterministic, so the Wasserstein term
    collapses to the induced state distance between successor states.
    """
    row_sums = np.sum(mdp.transitions, axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise ValueError("Taxi transitions must be row-stochastic.")

    next_states = np.argmax(mdp.transitions, axis=2)
    if not np.allclose(
        mdp.transitions[
            np.arange(mdp.num_states)[:, None],
            np.arange(mdp.num_actions)[None, :],
            next_states,
        ],
        1.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Taxi state-action metric currently assumes deterministic transitions."
        )

    num_states = mdp.num_states
    num_actions = mdp.num_actions
    num_pairs = num_states * num_actions
    d_s = np.zeros((num_states, num_states), dtype=float)

    per_action_reward_gap = [
        np.abs(mdp.rewards[:, action][:, None] - mdp.rewards[:, action][None, :])
        for action in range(num_actions)
    ]

    for iteration in range(max_iter):
        d_s_new = np.zeros_like(d_s)
        for action in range(num_actions):
            successors = next_states[:, action]
            candidate = per_action_reward_gap[action] + mdp.gamma * d_s[np.ix_(successors, successors)]
            d_s_new = np.maximum(d_s_new, candidate)
        np.fill_diagonal(d_s_new, 0.0)

        delta = float(np.max(np.abs(d_s_new - d_s)))
        d_s = d_s_new
        if verbose and (iteration % 10 == 0 or delta < tol):
            print(f"    pair-bisim iter {iteration}: max change = {delta:.3e}")
        if delta < tol:
            break

    rewards_flat = mdp.rewards.reshape(num_pairs)
    successor_flat = next_states.reshape(num_pairs)
    d_sa = np.abs(rewards_flat[:, None] - rewards_flat[None, :]) + mdp.gamma * d_s[
        np.ix_(successor_flat, successor_flat)
    ]
    np.fill_diagonal(d_sa, 0.0)
    return d_sa


def _pair_metric_cache_path(
    mdp: AB.TabularMDP,
    metric_kind: str = "fixed_point",
    cache_dir: Path = CACHE_DIR,
) -> Path:
    """Return the Taxi cache path for the requested pair metric."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.blake2b(digest_size=10)
    h.update(mdp.rewards.astype(np.float64).tobytes())
    h.update(mdp.transitions.astype(np.float64).tobytes())
    fingerprint = h.hexdigest()
    if metric_kind == "fixed_point":
        stem = (
            f"taxi_sa_metric_n{mdp.num_states}_a{mdp.num_actions}_"
            f"g{float(mdp.gamma):.4f}_{fingerprint}"
        )
    else:
        stem = (
            f"taxi_sa_{metric_kind}_n{mdp.num_states}_a{mdp.num_actions}_"
            f"g{float(mdp.gamma):.4f}_{fingerprint}"
        )
    return cache_dir / f"{stem}.npy"


def load_or_compute_taxi_metric(
    mdp: AB.TabularMDP,
    metric_kind: str = "fixed_point",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
    num_workers: int = 1,
) -> Array:
    """Load the cached Taxi pair distortion, or compute and persist it."""
    pair_path = _pair_metric_cache_path(
        mdp,
        metric_kind=metric_kind,
        cache_dir=cache_dir,
    )
    if pair_path.exists():
        if verbose:
            print(f"  Loading cached Taxi {metric_kind} metric from {pair_path.name}")
        return np.load(pair_path)

    if verbose:
        if metric_kind == "fixed_point":
            print("  Computing Taxi fixed-point pair metric (no cache hit)...")
        else:
            print("  Computing Taxi one-step pair metric (no cache hit)...")
    if metric_kind == "fixed_point":
        distortion = compute_taxi_fixed_point_bisimulation_metric(mdp, verbose=verbose)
    elif metric_kind == "one_step":
        distortion = compute_taxi_one_step_bisimulation_metric(mdp)
    else:
        raise ValueError(f"Unknown Taxi metric kind: {metric_kind}")
    np.save(pair_path, distortion)
    return distortion


def load_distortion(
    mdp: AB.TabularMDP,
    metric_kind: str = "fixed_point",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
    num_workers: int = 1,
) -> Array:
    """Return the pair distortion used by Taxi experiments."""
    return load_or_compute_taxi_metric(
        mdp,
        metric_kind=metric_kind,
        cache_dir=cache_dir,
        verbose=verbose,
        num_workers=num_workers,
    )


def state_factor_summary(label: Sequence[int]) -> Dict[str, object]:
    """Summarize one Taxi state label into readable categorical factors."""
    row, col, passenger, destination = [int(value) for value in label]
    if tuple(label) == ABSORBING_LABEL:
        return {
            "row": "terminal",
            "col": "terminal",
            "passenger": "terminal",
            "destination": "terminal",
            "taxi_zone": "terminal",
            "taxi_at_special": False,
        }
    taxi_loc = (row, col)
    return {
        "row": row,
        "col": col,
        "passenger": passenger_label(passenger),
        "destination": destination_label(destination),
        "taxi_zone": f"r{row}_c{col}",
        "taxi_at_special": taxi_loc in LOCATIONS,
    }


def state_action_factor_summary(
    mdp: TabularMDP,
    pair_index: int,
) -> Dict[str, object]:
    """Summarize one Taxi pair index into readable state and action factors."""
    state, action = decode_state_action(pair_index, mdp.num_actions)
    summary = state_factor_summary(mdp.state_labels[state])
    summary["action"] = ACTION_LABELS[action]
    summary["state_index"] = int(state)
    summary["pair_index"] = int(pair_index)
    return summary
