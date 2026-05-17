"""Fully observable tabular DoorKey-style MDP and its pair-distortion builders."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core import abstraction as AB


Array = np.ndarray
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

ACTION_LABELS = ["turn_left", "turn_right", "forward", "pickup", "toggle"]
DIR_LABELS = ("north", "east", "south", "west")
DIR_DELTAS: Tuple[Tuple[int, int], ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
ABSORBING_LABEL = ("terminal",)


def _door_position(grid_size: int) -> Tuple[int, int]:
    """Return the unique locked-door cell."""
    return grid_size // 2, grid_size // 2


def _goal_position(grid_size: int) -> Tuple[int, int]:
    """Return the goal cell behind the door."""
    return grid_size - 2, grid_size - 2


def _key_position(grid_size: int) -> Tuple[int, int]:
    """Return the key cell in the start room."""
    return 1, grid_size - 2


def _wall_cells(grid_size: int) -> set[Tuple[int, int]]:
    """Return the internal wall cells that separate the two rooms."""
    wall_x, door_y = _door_position(grid_size)
    return {
        (wall_x, y)
        for y in range(grid_size)
        if y != door_y
    }


def _traversable_positions(grid_size: int, door_open: int) -> List[Tuple[int, int]]:
    """Return the legal agent positions for the requested door status."""
    door_position = _door_position(grid_size)
    goal_position = _goal_position(grid_size)
    blocked = _wall_cells(grid_size)
    if int(door_open) == 0:
        blocked = set(blocked)
        blocked.add(door_position)

    positions: List[Tuple[int, int]] = []
    for y in range(grid_size):
        for x in range(grid_size):
            position = (x, y)
            if position == goal_position:
                continue
            if position in blocked:
                continue
            positions.append(position)
    return positions


def _forward_position(x: int, y: int, direction: int) -> Tuple[int, int]:
    """Move one grid step in the requested direction."""
    dx, dy = DIR_DELTAS[int(direction)]
    return x + dx, y + dy


def _is_within_bounds(position: Tuple[int, int], grid_size: int) -> bool:
    """Check whether one cell lies inside the square grid."""
    x, y = position
    return 0 <= x < grid_size and 0 <= y < grid_size


def _is_blocked(
    position: Tuple[int, int],
    grid_size: int,
    door_open: int,
) -> bool:
    """Check whether one cell is currently blocked by a wall or a closed door."""
    if not _is_within_bounds(position, grid_size):
        return True
    if position in _wall_cells(grid_size):
        return True
    if int(door_open) == 0 and position == _door_position(grid_size):
        return True
    return False


def build_doorkey_mdp(
    grid_size: int = 6,
    gamma: float = 0.99,
    goal_reward: float = 1.0,
) -> AB.TabularMDP:
    """Build a fully observable tabular DoorKey-style MDP.

    The environment is inspired by MiniGrid DoorKey but stays fully observable
    and exactly tabular:

    - one internal wall splits the grid into two rooms
    - a single locked door sits in that wall
    - a single key lies in the left room
    - the goal lies in the right room and ends the episode on arrival
    """
    if grid_size < 5:
        raise ValueError("grid_size must be at least 5.")

    door_position = _door_position(grid_size)
    key_position = _key_position(grid_size)
    goal_position = _goal_position(grid_size)

    if key_position == door_position or key_position == goal_position:
        raise ValueError("DoorKey layout must keep key, door, and goal distinct.")

    state_labels: List[Tuple[int, ...] | Tuple[str]] = []
    state_to_index: Dict[Tuple[int, int, int, int, int], int] = {}
    reachable_modes = (
        (0, 0),  # key not picked up, door closed
        (1, 0),  # key carried, door still closed
        (1, 1),  # key carried, door opened
    )

    for has_key, door_open in reachable_modes:
        for x, y in _traversable_positions(grid_size, door_open):
            for direction in range(len(DIR_LABELS)):
                state = (x, y, direction, has_key, door_open)
                state_to_index[state] = len(state_labels)
                state_labels.append(state)

    terminal_state = len(state_labels)
    state_labels.append(ABSORBING_LABEL)

    num_states = len(state_labels)
    num_actions = len(ACTION_LABELS)
    transitions = np.zeros((num_states, num_actions, num_states), dtype=float)
    rewards = np.zeros((num_states, num_actions), dtype=float)

    for state, state_index in state_to_index.items():
        x, y, direction, has_key, door_open = state

        for action in range(num_actions):
            next_state = state
            reward = 0.0

            if action == 0:  # turn_left
                next_state = (x, y, (direction - 1) % 4, has_key, door_open)
            elif action == 1:  # turn_right
                next_state = (x, y, (direction + 1) % 4, has_key, door_open)
            elif action == 2:  # forward
                front = _forward_position(x, y, direction)
                if front == goal_position:
                    transitions[state_index, action, terminal_state] = 1.0
                    rewards[state_index, action] = float(goal_reward)
                    continue
                if not _is_blocked(front, grid_size, door_open):
                    next_state = (front[0], front[1], direction, has_key, door_open)
            elif action == 3:  # pickup
                front = _forward_position(x, y, direction)
                if has_key == 0 and door_open == 0 and front == key_position:
                    next_state = (x, y, direction, 1, 0)
            elif action == 4:  # toggle
                front = _forward_position(x, y, direction)
                if has_key == 1 and door_open == 0 and front == door_position:
                    next_state = (x, y, direction, 1, 1)
            else:
                raise ValueError(f"Unknown action {action}")

            transitions[state_index, action, state_to_index[next_state]] = 1.0
            rewards[state_index, action] = reward

    transitions[terminal_state, :, terminal_state] = 1.0
    rewards[terminal_state, :] = 0.0

    return AB.TabularMDP(
        transitions=transitions,
        rewards=rewards,
        gamma=gamma,
        state_labels=state_labels,
        action_labels=list(ACTION_LABELS),
    )


def compute_doorkey_one_step_distortion(mdp: AB.TabularMDP) -> Array:
    """Build the one-step Bellman-compatible pair distortion on concrete pairs."""
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


def compute_doorkey_fixed_point_bisimulation_metric(
    mdp: AB.TabularMDP,
    tol: float = 1e-4,
    max_iter: int = 200,
    verbose: bool = True,
) -> Array:
    """Build the fixed-point state-action bisimulation metric on concrete pairs.

    DoorKey is deterministic, so the Wasserstein term collapses to the induced
    state semimetric between successor states, just as in the Taxi benchmark.
    """
    row_sums = np.sum(mdp.transitions, axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise ValueError("DoorKey transitions must be row-stochastic.")

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
            "DoorKey fixed-point metric currently assumes deterministic transitions."
        )

    num_states = mdp.num_states
    num_actions = mdp.num_actions
    num_pairs = num_states * num_actions
    state_metric = np.zeros((num_states, num_states), dtype=float)

    per_action_reward_gap = [
        np.abs(mdp.rewards[:, action][:, None] - mdp.rewards[:, action][None, :])
        for action in range(num_actions)
    ]

    for iteration in range(max_iter):
        updated = np.zeros_like(state_metric)
        for action in range(num_actions):
            successors = next_states[:, action]
            candidate = (
                per_action_reward_gap[action]
                + mdp.gamma * state_metric[np.ix_(successors, successors)]
            )
            updated = np.maximum(updated, candidate)
        np.fill_diagonal(updated, 0.0)

        delta = float(np.max(np.abs(updated - state_metric)))
        state_metric = updated
        if verbose and (iteration % 10 == 0 or delta < tol):
            print(f"    doorkey bisim iter {iteration}: max change = {delta:.3e}")
        if delta < tol:
            break

    rewards_flat = np.asarray(mdp.rewards, dtype=float).reshape(num_pairs)
    successor_flat = next_states.reshape(num_pairs)
    distortion = np.abs(rewards_flat[:, None] - rewards_flat[None, :]) + mdp.gamma * state_metric[
        np.ix_(successor_flat, successor_flat)
    ]
    np.fill_diagonal(distortion, 0.0)
    return distortion


def _metric_cache_path(
    mdp: AB.TabularMDP,
    metric_kind: str,
    cache_dir: Path,
) -> Path:
    """Return the cached pair-metric path for one concrete DoorKey MDP."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = mdp.transitions.tobytes() + mdp.rewards.tobytes()
    digest = hashlib.sha1(payload).hexdigest()[:16]
    if metric_kind == "one_step":
        filename = (
            f"doorkey_sa_one_step_n{mdp.num_states}_a{mdp.num_actions}_"
            f"g{mdp.gamma:.4f}_{digest}.npy"
        )
    elif metric_kind == "fixed_point":
        filename = (
            f"doorkey_sa_fixed_point_n{mdp.num_states}_a{mdp.num_actions}_"
            f"g{mdp.gamma:.4f}_{digest}.npy"
        )
    else:
        raise ValueError(f"Unknown DoorKey metric kind: {metric_kind}")
    return cache_dir / filename


def load_or_compute_doorkey_metric(
    mdp: AB.TabularMDP,
    metric_kind: str = "fixed_point",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
    num_workers: int = 1,
) -> Array:
    """Load the cached DoorKey pair distortion or compute and persist it."""
    pair_path = _metric_cache_path(
        mdp,
        metric_kind=metric_kind,
        cache_dir=cache_dir,
    )
    if pair_path.exists():
        if verbose:
            print(f"  Loading cached DoorKey {metric_kind} metric from {pair_path.name}")
        return np.load(pair_path)

    if verbose:
        if metric_kind == "fixed_point":
            print("  Computing DoorKey fixed-point pair metric (no cache hit)...")
        else:
            print("  Computing DoorKey one-step pair metric (no cache hit)...")
    if metric_kind == "fixed_point":
        distortion = compute_doorkey_fixed_point_bisimulation_metric(mdp, verbose=verbose)
    elif metric_kind == "one_step":
        distortion = compute_doorkey_one_step_distortion(mdp)
    else:
        raise ValueError(f"Unknown DoorKey metric kind: {metric_kind}")
    np.save(pair_path, distortion)
    return distortion


def load_distortion(
    mdp: AB.TabularMDP,
    metric_kind: str = "fixed_point",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
    num_workers: int = 1,
) -> Array:
    """Return the pair distortion used by DoorKey experiments."""
    return load_or_compute_doorkey_metric(
        mdp,
        metric_kind=metric_kind,
        cache_dir=cache_dir,
        verbose=verbose,
        num_workers=num_workers,
    )
