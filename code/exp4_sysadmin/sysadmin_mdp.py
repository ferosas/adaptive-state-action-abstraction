"""Fully observable ring SysAdmin MDP and its state-action distortion builder."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from core import abstraction as AB


Array = np.ndarray
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


@dataclass(frozen=True)
class _KantorovichDualTemplate:
    """Cached sparse constraint template for Wasserstein dual LPs."""

    A_ub: csr_matrix
    row_i: Array
    row_j: Array
    bounds: tuple[tuple[float | None, float | None], ...]


def decode_state(state: int, num_machines: int) -> Tuple[int, ...]:
    """Decode one integer state index into its binary machine-status tuple."""
    return tuple((int(state) >> machine) & 1 for machine in range(int(num_machines)))


def encode_state(bits: Sequence[int]) -> int:
    """Encode a binary machine-status tuple into one integer state index."""
    state = 0
    for machine, value in enumerate(bits):
        state |= (int(value) & 1) << machine
    return int(state)


def _validate_neighbors(neighbors: Sequence[Sequence[int]], num_machines: int) -> List[Tuple[int, ...]]:
    """Validate and normalize an undirected machine-neighborhood list."""
    if len(neighbors) != int(num_machines):
        raise ValueError("neighbors must contain one entry per machine.")
    normalized: List[Tuple[int, ...]] = []
    for machine, raw_neighbors in enumerate(neighbors):
        unique = sorted({int(neighbor) for neighbor in raw_neighbors})
        if machine in unique:
            raise ValueError("Self-neighbors are not supported.")
        if any(neighbor < 0 or neighbor >= int(num_machines) for neighbor in unique):
            raise ValueError("Neighbor indices must be valid machine ids.")
        normalized.append(tuple(unique))

    for machine, machine_neighbors in enumerate(normalized):
        for neighbor in machine_neighbors:
            if machine not in normalized[neighbor]:
                raise ValueError("SysAdmin topology must be undirected.")
    return normalized


def ring_neighbors(num_machines: int) -> List[Tuple[int, ...]]:
    """Return the default ring topology neighbors."""
    if num_machines == 1:
        return [tuple()]
    if num_machines == 2:
        return [(1,), (0,)]
    return [
        ((machine - 1) % int(num_machines), (machine + 1) % int(num_machines))
        for machine in range(int(num_machines))
    ]


def build_graph_sysadmin_mdp(
    num_machines: int = 7,
    p_base: float = 0.95,
    neighbor_penalty: float = 0.15,
    p_recover: float = 0.05,
    p_reboot: float = 0.95,
    reboot_cost: float = 0.2,
    gamma: float = 0.99,
    neighbors: Sequence[Sequence[int]] | None = None,
    degree_normalized: bool = True,
) -> AB.TabularMDP:
    """Build a graph-structured SysAdmin MDP with local failure interactions."""
    if num_machines <= 0:
        raise ValueError("num_machines must be positive.")
    if not (0.0 <= p_base <= 1.0):
        raise ValueError("p_base must be in [0, 1].")
    if not (0.0 <= neighbor_penalty <= 1.0):
        raise ValueError("neighbor_penalty must be in [0, 1].")
    if not (0.0 <= p_recover <= 1.0):
        raise ValueError("p_recover must be in [0, 1].")
    if not (0.0 <= p_reboot <= 1.0):
        raise ValueError("p_reboot must be in [0, 1].")
    if reboot_cost < 0.0:
        raise ValueError("reboot_cost must be non-negative.")

    topology = _validate_neighbors(
        ring_neighbors(num_machines) if neighbors is None else neighbors,
        num_machines,
    )
    num_states = 1 << int(num_machines)
    action_labels = ["noop"] + [f"reboot_{machine}" for machine in range(num_machines)]
    num_actions = len(action_labels)

    all_state_bits = np.asarray(
        [decode_state(state, num_machines) for state in range(num_states)],
        dtype=int,
    )
    transitions = np.zeros((num_states, num_actions, num_states), dtype=float)
    rewards = np.zeros((num_states, num_actions), dtype=float)
    state_labels: List[Tuple[int, ...]] = [tuple(bits.tolist()) for bits in all_state_bits]

    for state in range(num_states):
        current_bits = all_state_bits[state]
        alive_fraction = float(np.mean(current_bits))
        for action in range(num_actions):
            reboot_target = None if action == 0 else action - 1
            rewards[state, action] = alive_fraction - (reboot_cost if reboot_target is not None else 0.0)

            up_probabilities = np.zeros(num_machines, dtype=float)
            for machine in range(num_machines):
                if reboot_target == machine:
                    up_probabilities[machine] = p_reboot
                    continue

                if current_bits[machine] == 1:
                    neighbor_ids = topology[machine]
                    failed_neighbors = sum(
                        1 - int(current_bits[neighbor])
                        for neighbor in neighbor_ids
                    )
                    degree = len(neighbor_ids)
                    pressure = (
                        float(failed_neighbors) / float(degree)
                        if degree_normalized and degree > 0
                        else float(failed_neighbors)
                    )
                    up_probabilities[machine] = float(
                        np.clip(p_base - neighbor_penalty * pressure, 0.0, 1.0)
                    )
                else:
                    up_probabilities[machine] = p_recover

            next_state_probabilities = np.where(
                all_state_bits == 1,
                up_probabilities[None, :],
                1.0 - up_probabilities[None, :],
            )
            transitions[state, action, :] = np.prod(next_state_probabilities, axis=1)

    return AB.TabularMDP(
        transitions=transitions,
        rewards=rewards,
        gamma=gamma,
        state_labels=state_labels,
        action_labels=action_labels,
    )


def build_ring_sysadmin_mdp(
    num_machines: int = 7,
    p_base: float = 0.95,
    neighbor_penalty: float = 0.15,
    p_recover: float = 0.05,
    p_reboot: float = 0.95,
    reboot_cost: float = 0.2,
    gamma: float = 0.99,
) -> AB.TabularMDP:
    """Build the default ring-structured SysAdmin MDP."""
    return build_graph_sysadmin_mdp(
        num_machines=num_machines,
        p_base=p_base,
        neighbor_penalty=neighbor_penalty,
        p_recover=p_recover,
        p_reboot=p_reboot,
        reboot_cost=reboot_cost,
        gamma=gamma,
        neighbors=ring_neighbors(num_machines),
        degree_normalized=False,
    )


def _total_variation_distance(row_p: Array, row_q: Array) -> float:
    """Compute total variation distance between two tabular transition rows."""
    return 0.5 * float(np.sum(np.abs(np.asarray(row_p) - np.asarray(row_q))))


def _build_kantorovich_dual_template(num_states: int) -> _KantorovichDualTemplate:
    """Pre-build the sparse dual LP template for finite-state Wasserstein."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row_i: list[int] = []
    row_j: list[int] = []
    constraint = 0
    for source in range(num_states):
        for target in range(num_states):
            if source == target:
                continue
            rows.extend([constraint, constraint])
            cols.extend([source, target])
            data.extend([1.0, -1.0])
            row_i.append(source)
            row_j.append(target)
            constraint += 1

    A_ub = csr_matrix((data, (rows, cols)), shape=(constraint, num_states))
    bounds = tuple([(0.0, 0.0)] + [(None, None)] * (num_states - 1))
    return _KantorovichDualTemplate(
        A_ub=A_ub,
        row_i=np.asarray(row_i, dtype=int),
        row_j=np.asarray(row_j, dtype=int),
        bounds=bounds,
    )


def _wasserstein_dense(
    row_p: Array,
    row_q: Array,
    metric: Array,
    template: _KantorovichDualTemplate,
) -> float:
    """Solve Wasserstein exactly on a dense finite state space via the dual LP."""
    diff = np.asarray(row_p, dtype=float) - np.asarray(row_q, dtype=float)
    if float(np.max(np.abs(diff))) <= 1e-15:
        return 0.0

    result = linprog(
        c=-diff,
        A_ub=template.A_ub,
        b_ub=metric[template.row_i, template.row_j],
        bounds=template.bounds,
        method="highs",
    )
    if not result.success or result.fun is None:
        raise RuntimeError(
            f"Failed to solve SysAdmin Wasserstein LP: {result.message}"
        )
    return max(0.0, float(-result.fun))


def _one_step_block(
    pair_rows: Array,
    reward_flat: Array,
    gamma: float,
    start: int,
    end: int,
) -> tuple[int, int, Array]:
    """Compute one contiguous block of the dense one-step pair distortion."""
    reward_gap = np.abs(reward_flat[start:end, None] - reward_flat[None, :])
    total_variation = 0.5 * np.sum(
        np.abs(pair_rows[start:end, None, :] - pair_rows[None, :, :]),
        axis=2,
    )
    block = reward_gap + float(gamma) * total_variation
    return int(start), int(end), np.asarray(block, dtype=float)


def _pair_lift_block(
    pair_rows: Array,
    reward_flat: Array,
    state_metric: Array,
    gamma: float,
    template: _KantorovichDualTemplate,
    start: int,
    end: int,
) -> tuple[int, Array]:
    """Compute one contiguous upper-triangular block of the lifted pair metric."""
    num_pairs = int(pair_rows.shape[0])
    block = np.zeros((int(end) - int(start), num_pairs), dtype=float)
    for local_offset, pair_index in enumerate(range(int(start), int(end))):
        for other_pair_index in range(pair_index + 1, num_pairs):
            reward_gap = abs(reward_flat[pair_index] - reward_flat[other_pair_index])
            transport = _wasserstein_dense(
                pair_rows[pair_index],
                pair_rows[other_pair_index],
                state_metric,
                template,
            )
            block[local_offset, other_pair_index] = reward_gap + float(gamma) * transport
    return int(start), block


def compute_sysadmin_distortion(
    mdp: AB.TabularMDP,
    block_size: int = 16,
    verbose: bool = True,
) -> Array:
    """Compute the one-step Bellman-compatible distortion on concrete pairs."""
    return compute_sysadmin_one_step_distortion(
        mdp,
        block_size=block_size,
        verbose=verbose,
    )


def compute_sysadmin_one_step_distortion(
    mdp: AB.TabularMDP,
    block_size: int = 16,
    verbose: bool = True,
    num_workers: int = 1,
) -> Array:
    """Compute the one-step Bellman-compatible distortion on concrete pairs."""
    num_pairs = mdp.num_state_action_pairs
    num_states = mdp.num_states
    pair_rows = np.asarray(mdp.transitions, dtype=float).reshape(num_pairs, num_states)
    reward_flat = np.asarray(mdp.rewards, dtype=float).reshape(num_pairs)
    distortion = np.zeros((num_pairs, num_pairs), dtype=float)

    worker_count = max(1, int(num_workers))
    blocks = [(start, min(start + block_size, num_pairs)) for start in range(0, num_pairs, block_size)]
    if worker_count > 1 and len(blocks) > 1:
        if verbose:
            print(
                f"    parallelizing SysAdmin one-step distortion over {len(blocks)} row blocks with {min(worker_count, len(blocks))} workers",
                flush=True,
            )
        with ProcessPoolExecutor(max_workers=min(worker_count, len(blocks))) as executor:
            results = list(
                executor.map(
                    _one_step_block,
                    [pair_rows] * len(blocks),
                    [reward_flat] * len(blocks),
                    [float(mdp.gamma)] * len(blocks),
                    [start for start, _ in blocks],
                    [end for _, end in blocks],
                )
            )
        for start, end, block in results:
            distortion[start:end, :] = block
    else:
        for start, end in blocks:
            if verbose:
                print(
                    f"    distortion rows {start}:{end} / {num_pairs}",
                    flush=True,
                )
            _, _, block = _one_step_block(pair_rows, reward_flat, float(mdp.gamma), start, end)
            distortion[start:end, :] = block

    np.fill_diagonal(distortion, 0.0)
    return distortion


def compute_sysadmin_fixed_point_bisimulation_metric(
    mdp: AB.TabularMDP,
    tol: float = 1e-6,
    max_iter: int = 40,
    verbose: bool = True,
    num_workers: int = 1,
) -> Array:
    """Compute the fixed-point state-action bisimulation metric on concrete pairs.

    This follows the same action-aligned state-metric iteration used by the Taxi
    state-action metric:

      d_s^{k+1}(s,t)
        = max_a |r(s,a)-r(t,a)| + gamma * W_{d_s^k}(P(.|s,a), P(.|t,a))

    and then defines the pair metric by

      d_sa((s,a),(t,b))
        = |r(s,a)-r(t,b)| + gamma * W_{d_s^*}(P(.|s,a), P(.|t,b)).

    Because SysAdmin has dense stochastic kernels, this exact fixed-point
    variant is substantially more expensive than the one-step surrogate.
    """
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    transitions = np.asarray(mdp.transitions, dtype=float)
    reward_gaps = [
        np.abs(mdp.rewards[:, action][:, None] - mdp.rewards[:, action][None, :])
        for action in range(num_actions)
    ]
    lp_template = _build_kantorovich_dual_template(num_states)
    state_metric = np.zeros((num_states, num_states), dtype=float)

    if verbose:
        print("    iterating SysAdmin state metric fixed point...", flush=True)
    for iteration in range(max_iter):
        updated = np.zeros_like(state_metric)
        for state in range(num_states):
            for other_state in range(state + 1, num_states):
                best = 0.0
                for action in range(num_actions):
                    transport = _wasserstein_dense(
                        transitions[state, action],
                        transitions[other_state, action],
                        state_metric,
                        lp_template,
                    )
                    candidate = reward_gaps[action][state, other_state] + mdp.gamma * transport
                    if candidate > best:
                        best = candidate
                updated[state, other_state] = best
                updated[other_state, state] = best
        delta = float(np.max(np.abs(updated - state_metric)))
        state_metric = updated
        if verbose and (iteration % 5 == 0 or delta < tol):
            print(f"    sysadmin bisim iter {iteration}: max change = {delta:.3e}", flush=True)
        if delta < tol:
            break

    num_pairs = mdp.num_state_action_pairs
    pair_rows = transitions.reshape(num_pairs, num_states)
    reward_flat = np.asarray(mdp.rewards, dtype=float).reshape(num_pairs)
    distortion = np.zeros((num_pairs, num_pairs), dtype=float)

    if verbose:
        print("    lifting fixed-point state metric to pair metric...", flush=True)
    worker_count = max(1, int(num_workers))
    progress_stride = max(1, num_pairs // 8)
    if worker_count > 1 and num_pairs > 1:
        blocks = [(start, min(start + progress_stride, num_pairs)) for start in range(0, num_pairs, progress_stride)]
        if verbose:
            print(
                f"    parallelizing lifted SysAdmin pair metric over {len(blocks)} row blocks with {min(worker_count, len(blocks))} workers",
                flush=True,
            )
        with ProcessPoolExecutor(max_workers=min(worker_count, len(blocks))) as executor:
            results = list(
                executor.map(
                    _pair_lift_block,
                    [pair_rows] * len(blocks),
                    [reward_flat] * len(blocks),
                    [state_metric] * len(blocks),
                    [float(mdp.gamma)] * len(blocks),
                    [lp_template] * len(blocks),
                    [start for start, _ in blocks],
                    [end for _, end in blocks],
                )
            )
        for start, block in results:
            end = start + block.shape[0]
            distortion[start:end, :] = block
        distortion = np.maximum(distortion, distortion.T)
    else:
        for pair_index in range(num_pairs):
            if verbose and (pair_index % progress_stride == 0 or pair_index + 1 == num_pairs):
                print(f"    pair rows {pair_index + 1}/{num_pairs}", flush=True)
            for other_pair_index in range(pair_index + 1, num_pairs):
                reward_gap = abs(reward_flat[pair_index] - reward_flat[other_pair_index])
                transport = _wasserstein_dense(
                    pair_rows[pair_index],
                    pair_rows[other_pair_index],
                    state_metric,
                    lp_template,
                )
                distortion[pair_index, other_pair_index] = reward_gap + mdp.gamma * transport
                distortion[other_pair_index, pair_index] = distortion[pair_index, other_pair_index]

    np.fill_diagonal(distortion, 0.0)
    return distortion


def _distortion_cache_path(mdp: AB.TabularMDP, cache_dir: Path) -> Path:
    """Return the cached distortion path for one concrete SysAdmin MDP."""
    return _metric_cache_path(mdp, metric_kind="one_step", cache_dir=cache_dir)


def _metric_cache_path(
    mdp: AB.TabularMDP,
    metric_kind: str,
    cache_dir: Path,
) -> Path:
    """Return the cached pair-metric path for one concrete SysAdmin MDP."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = mdp.transitions.tobytes() + mdp.rewards.tobytes()
    digest = hashlib.sha1(payload).hexdigest()[:16]
    if metric_kind == "one_step":
        filename = (
            f"distortion_sa_state_n{mdp.num_states}_a{mdp.num_actions}_"
            f"g{mdp.gamma:.4f}_{digest}.npy"
        )
    elif metric_kind == "fixed_point":
        filename = (
            f"distortion_sa_fixed_point_n{mdp.num_states}_a{mdp.num_actions}_"
            f"g{mdp.gamma:.4f}_{digest}.npy"
        )
    else:
        raise ValueError(f"Unknown SysAdmin metric kind: {metric_kind}")
    return cache_dir / filename


def load_or_compute_sysadmin_metric(
    mdp: AB.TabularMDP,
    metric_kind: str = "one_step",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
    block_size: int = 16,
    num_workers: int = 1,
) -> Array:
    """Load the cached SysAdmin pair metric or compute and save it."""
    path = _metric_cache_path(mdp, metric_kind=metric_kind, cache_dir=cache_dir)
    if path.exists():
        if verbose:
            print(f"Loading cached SysAdmin distortion from {path}", flush=True)
        return np.load(path)

    if verbose:
        if metric_kind == "one_step":
            print("Computing SysAdmin one-step state-action distortion...", flush=True)
        else:
            print("Computing SysAdmin fixed-point state-action bisimulation metric...", flush=True)
    if metric_kind == "one_step":
        distortion = compute_sysadmin_one_step_distortion(
            mdp,
            block_size=block_size,
            verbose=verbose,
            num_workers=num_workers,
        )
    elif metric_kind == "fixed_point":
        distortion = compute_sysadmin_fixed_point_bisimulation_metric(
            mdp,
            verbose=verbose,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown SysAdmin metric kind: {metric_kind}")
    np.save(path, distortion)
    if verbose:
        print(f"Saved SysAdmin distortion to {path}", flush=True)
    return distortion


def load_distortion(
    mdp: AB.TabularMDP,
    metric_kind: str = "one_step",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
    block_size: int = 16,
    num_workers: int = 1,
) -> Array:
    """Return the pair distortion used by SysAdmin experiments."""
    return load_or_compute_sysadmin_metric(
        mdp,
        metric_kind=metric_kind,
        cache_dir=cache_dir,
        verbose=verbose,
        block_size=block_size,
        num_workers=num_workers,
    )
