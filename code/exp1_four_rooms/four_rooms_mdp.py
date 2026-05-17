"""Four-Rooms MDP with uniform-reset random goal chaining.

State: (row, col, goal_room) on an 11x11 grid with walls dividing four rooms.
Goal resets to a uniformly-drawn room (any of the four) and agent position
resets uniformly over all playable cells whenever the agent lands on the
current goal center.

This module also owns the Four-Rooms metric-cache helpers used by the main
experiment runner. In particular, ``precompute_metric_cache(...)`` replaces the
old standalone ``compute_sa_bisim.py`` helper script.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core import abstraction as AB


Array = np.ndarray


GRID_SIZE = 11
CLASSIC_LAYOUT: Tuple[str, ...] = (
    ".....#.....",
    ".....#.....",
    "..C..H.....",
    ".....#..C..",
    ".....#.....",
    "#H####.....",
    ".....###H##",
    ".....#.....",
    "..C..#..C..",
    ".....H.....",
    ".....#.....",
)
ROOM_CENTERS: Tuple[Tuple[int, int], ...] = (
    (2, 2),   # room 1: top-left
    (3, 8),   # room 2: top-right
    (8, 2),   # room 3: bottom-left
    (8, 8),   # room 4: bottom-right
)
# Hallways: one cell per wall connecting adjacent rooms.
HALLWAYS: Tuple[Tuple[int, int], ...] = (
    (2, 5),   # between rooms 1 and 2 (top rooms)
    (5, 1),   # between rooms 1 and 3 (left rooms)
    (6, 8),   # between rooms 2 and 4 (right rooms)
    (9, 5),   # between rooms 3 and 4 (bottom rooms)
)
NUM_GOALS = len(ROOM_CENTERS)
# Actions: N, E, S, W.
ACTION_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),  # up (N)
    (0, 1),   # right (E)
    (1, 0),   # down (S)
    (0, -1),  # left (W)
)
ACTION_LABELS = ["up", "right", "down", "left"]
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _build_wall_mask() -> Array:
    """Return an 11x11 boolean mask where True marks walls."""
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for r, row in enumerate(CLASSIC_LAYOUT):
        for c, char in enumerate(row):
            walls[r, c] = char == "#"
    return walls


def _build_cell_index(walls: Array) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Enumerate playable cells and return a (row, col) -> cell_index lookup."""
    cells: List[Tuple[int, int]] = []
    cell_of = -np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if walls[r, c]:
                continue
            cell_of[r, c] = len(cells)
            cells.append((r, c))
    return cells, cell_of


def _goal_of_cell(cells: List[Tuple[int, int]]) -> List[int]:
    """Return the goal index g for each cell such that cell == center_g, else -1."""
    center_to_goal = {center: g for g, center in enumerate(ROOM_CENTERS)}
    return [center_to_goal.get(cell, -1) for cell in cells]


def room_label(row: int, col: int) -> str:
    """Assign every playable cell to a room label, or ``hallway``."""
    if row <= 4 and col <= 4:
        return "top-left"
    if row <= 4 and col >= 6:
        return "top-right"
    if row >= 6 and col <= 4:
        return "bottom-left"
    if row >= 6 and col >= 6:
        return "bottom-right"
    return "hallway"


def sigma_permutation(mdp: AB.TabularMDP, cells: List[Tuple[int, int]]) -> Array:
    """Return the vertical-reflection plus goal-swap permutation of states when defined."""
    goal_perm = {0: 2, 1: 3, 2: 0, 3: 1}
    cell_to_index = {cell: idx for idx, cell in enumerate(cells)}
    sigma = np.zeros(mdp.num_states, dtype=int)
    for state in range(mdp.num_states):
        cell_idx, goal_idx = mdp.state_labels[state]
        row, col = cells[cell_idx]
        reflected_cell_idx = cell_to_index.get((10 - row, col))
        if reflected_cell_idx is None:
            sigma[state] = state
            continue
        sigma[state] = mdp.state_labels.index((reflected_cell_idx, goal_perm[goal_idx]))
    return sigma


def build_four_rooms_mdp(eta: float = 0.1, gamma: float = 0.99) -> AB.TabularMDP:
    """Construct the Four-Rooms TabularMDP with uniform-reset goal chaining.

    Semantics
    ---------
    * With probability 1-eta the agent's intended move happens (bouncing on
      walls); with probability eta the agent stays in place.
    * If the resulting landing cell is the goal-room centre, reward fires
      and the state resets uniformly over all `(cell, goal)` pairs ---
      cell uniform over all playable cells, new goal uniform over all
      4 rooms (including the same one).

    Structured fields
    -----------------
    The returned MDP carries three extra attributes used by the
    bisimulation-metric fixed point:
        mdp.p_hit       : (S, A) array of goal-hit probabilities.
        mdp.R_support   : list-of-list of non-reset destination indices.
        mdp.R_probs     : list-of-list of corresponding probabilities.
    """
    if not (0.0 <= eta < 1.0):
        raise ValueError(f"Slip probability eta must lie in [0, 1); got {eta}.")

    walls = _build_wall_mask()
    cells, cell_of = _build_cell_index(walls)
    num_cells = len(cells)
    num_states = num_cells * NUM_GOALS
    num_actions = len(ACTION_DELTAS)

    cell_center_goal = _goal_of_cell(cells)

    transitions = np.zeros((num_states, num_actions, num_states), dtype=float)
    rewards = np.zeros((num_states, num_actions), dtype=float)
    state_labels: List[Tuple[int, int]] = []

    p_hit = np.zeros((num_states, num_actions), dtype=float)
    # Non-reset (``R``) components, indexed by [state][action] -> dict of
    # dest_state -> prob.
    R_map: List[List[dict]] = [[dict() for _ in range(num_actions)] for _ in range(num_states)]

    def state_index(cell_idx: int, goal_idx: int) -> int:
        return cell_idx * NUM_GOALS + goal_idx

    # Pre-compute the uniform reset distribution once.
    reset_share = 1.0 / float(num_states)

    for cell_idx, (r, c) in enumerate(cells):
        for goal_idx in range(NUM_GOALS):
            s = state_index(cell_idx, goal_idx)
            state_labels.append((cell_idx, goal_idx))
            for action_idx, (dr, dc) in enumerate(ACTION_DELTAS):
                intended_r = r + dr
                intended_c = c + dc
                in_bounds = 0 <= intended_r < GRID_SIZE and 0 <= intended_c < GRID_SIZE
                if in_bounds and not walls[intended_r, intended_c]:
                    intended_cell = cell_of[intended_r, intended_c]
                else:
                    intended_cell = cell_idx  # bounce: stay in place

                # Outcome 1: intended move with probability 1-eta.
                _accumulate(
                    transitions, rewards, p_hit, R_map,
                    s, action_idx,
                    prob=1.0 - eta,
                    dest_cell=intended_cell,
                    current_goal=goal_idx,
                    cell_center_goal=cell_center_goal,
                    state_index=state_index,
                    num_cells=num_cells,
                    reset_share=reset_share,
                )
                # Outcome 2: slip (stay in place) with probability eta.
                _accumulate(
                    transitions, rewards, p_hit, R_map,
                    s, action_idx,
                    prob=eta,
                    dest_cell=cell_idx,
                    current_goal=goal_idx,
                    cell_center_goal=cell_center_goal,
                    state_index=state_index,
                    num_cells=num_cells,
                    reset_share=reset_share,
                )

    # Sanity: each (s,a) row sums to 1.
    row_sums = transitions.sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        bad = np.argwhere(np.abs(row_sums - 1.0) > 1e-10)
        raise RuntimeError(f"Transition rows do not sum to 1 at {bad[:5].tolist()}")

    # Freeze R representation.
    R_support: List[List[List[int]]] = [
        [sorted(R_map[s][a].keys()) for a in range(num_actions)] for s in range(num_states)
    ]
    R_probs: List[List[List[float]]] = [
        [[R_map[s][a][k] for k in R_support[s][a]] for a in range(num_actions)]
        for s in range(num_states)
    ]

    mdp = AB.TabularMDP(
        transitions=transitions,
        rewards=rewards,
        gamma=gamma,
        state_labels=state_labels,
        action_labels=list(ACTION_LABELS),
    )
    # Structured representation used by the fast bisim helper.
    mdp.p_hit = p_hit  # type: ignore[attr-defined]
    mdp.R_support = R_support  # type: ignore[attr-defined]
    mdp.R_probs = R_probs  # type: ignore[attr-defined]
    return mdp


def _accumulate(
    transitions: Array,
    rewards: Array,
    p_hit: Array,
    R_map: List[List[dict]],
    s: int,
    action_idx: int,
    prob: float,
    dest_cell: int,
    current_goal: int,
    cell_center_goal: List[int],
    state_index,
    num_cells: int,
    reset_share: float,
) -> None:
    if prob <= 0.0:
        return
    landing_goal = cell_center_goal[dest_cell]
    if landing_goal == current_goal:
        # Arrived at the current goal centre: reward fires, (cell, goal)
        # resets uniformly over all 4 * num_cells pairs.
        rewards[s, action_idx] += prob * 1.0
        p_hit[s, action_idx] += prob
        share = prob * reset_share
        # Uniformly distribute prob across all (cell, goal) pairs.
        num_states = transitions.shape[2]
        transitions[s, action_idx, :] += share
    else:
        dest_state = state_index(dest_cell, current_goal)
        transitions[s, action_idx, dest_state] += prob
        R_map[s][action_idx][dest_state] = R_map[s][action_idx].get(dest_state, 0.0) + prob


# -----------------------------------------------------------------------------
# Lax-bisimulation metric (Ferns--Panangaden--Precup style fixed point) with
# reset-aware fast Wasserstein.
# -----------------------------------------------------------------------------


def _wasserstein_small(
    support_p: List[int],
    probs_p: List[float],
    support_q: List[int],
    probs_q: List[float],
    metric: Array,
) -> float:
    """Wasserstein between two small-support (<=2) distributions."""
    m = len(support_p)
    n = len(support_q)
    if m == 0 or n == 0:
        return 0.0
    if m == 1 and n == 1:
        return float(metric[support_p[0], support_q[0]])
    if m == 1:
        src = support_p[0]
        return float(sum(q * metric[src, t] for q, t in zip(probs_q, support_q)))
    if n == 1:
        tgt = support_q[0]
        return float(sum(p * metric[s, tgt] for p, s in zip(probs_p, support_p)))
    # m == n == 2, both masses sum to same value.
    if (
        support_p == support_q
        and abs(probs_p[0] - probs_q[0]) <= 1e-15
        and abs(probs_p[1] - probs_q[1]) <= 1e-15
    ):
        return 0.0
    p1, p2 = probs_p
    q1, q2 = probs_q
    s1, s2 = support_p
    t1, t2 = support_q
    c11 = metric[s1, t1]
    c12 = metric[s1, t2]
    c21 = metric[s2, t1]
    c22 = metric[s2, t2]
    x_low = max(0.0, p1 - q2)
    x_high = min(p1, q1)
    slope = c11 - c12 - c21 + c22
    x = x_low if slope >= 0.0 else x_high
    return float(x * c11 + (p1 - x) * c12 + (q1 - x) * c21 + (p2 - q1 + x) * c22)


def _transport_to_single_point(
    source_indices: Array,
    source_probs: Array,
    y: int,
    metric: Array,
) -> float:
    return float(np.sum(source_probs * metric[source_indices, y]))


def _transport_to_two_points(
    source_indices: Array,
    source_probs: Array,
    y1: int,
    y2: int,
    q1: float,
    metric: Array,
) -> float:
    """Solve W(source, q1 δ_{y1} + q2 δ_{y2}) where q2 = sum(source_probs) - q1,
    via sorting.  O(n log n) in the source support size."""
    if q1 <= 1e-15:
        return _transport_to_single_point(source_indices, source_probs, y2, metric)
    total = float(source_probs.sum())
    q2 = total - q1
    if q2 <= 1e-15:
        return _transport_to_single_point(source_indices, source_probs, y1, metric)
    c1 = metric[source_indices, y1]
    c2 = metric[source_indices, y2]
    diffs = c1 - c2
    order = np.argsort(diffs, kind="stable")
    probs_s = source_probs[order]
    c1_s = c1[order]
    c2_s = c2[order]
    cumulative = np.cumsum(probs_s)
    idx = int(np.searchsorted(cumulative, q1, side="right"))
    if idx >= len(order):
        # All mass goes to y1 (shouldn't happen unless q1 >= total).
        return float(np.sum(probs_s * c1_s))
    cost = float(np.sum(probs_s[:idx] * c1_s[:idx]))
    filled_before = float(cumulative[idx - 1]) if idx > 0 else 0.0
    fill = q1 - filled_before
    if fill > 0:
        cost += fill * float(c1_s[idx])
    remainder = float(probs_s[idx]) - fill
    if remainder > 0:
        cost += remainder * float(c2_s[idx])
    cost += float(np.sum(probs_s[idx + 1:] * c2_s[idx + 1:]))
    return cost


def _wasserstein_structured(
    p_s: float,
    R_s_support: List[int],
    R_s_probs: List[float],
    p_t: float,
    R_t_support: List[int],
    R_t_probs: List[float],
    metric: Array,
    reset_mean_dist: Array,
) -> float:
    """Wasserstein between (p_s U + (1-p_s) R_s) and (p_t U + (1-p_t) R_t),
    where U is uniform over {0,...,N-1} (all MDP states). Uses the dual
    identity: W(a U + b R_1, a U + c R_2) = W(b R_1, c R_2) after cancelling
    the shared reset mass.

    ``reset_mean_dist[y]`` must equal (1/N) sum_k metric[k, y] --- the
    expected distance from a uniform source to state y. It is passed in so
    the caller can amortise the broadcast cost across many calls."""
    if p_s == 0.0 and p_t == 0.0:
        return _wasserstein_small(R_s_support, R_s_probs, R_t_support, R_t_probs, metric)
    if abs(p_s - p_t) <= 1e-15:
        # Reset mass cancels; only the non-reset component contributes.
        return (1.0 - p_s) * _wasserstein_small(
            R_s_support, R_s_probs, R_t_support, R_t_probs, metric
        )

    # Mixed case: WLOG p_s > p_t.
    if p_s < p_t:
        p_s, p_t = p_t, p_s
        R_s_support, R_t_support = R_t_support, R_s_support
        R_s_probs, R_t_probs = R_t_probs, R_s_probs

    # Side A = (p_s - p_t) U + (1 - p_s) R_s,  total mass (1 - p_t).
    # Side B = (1 - p_t) R_t,                  total mass (1 - p_t).
    gap = p_s - p_t
    weight_s = 1.0 - p_s
    weight_t = 1.0 - p_t

    N = metric.shape[0]
    uniform_probs = np.full(N, gap / float(N))
    if R_s_support:
        r_s_idx = np.asarray(R_s_support, dtype=int)
        r_s_p = np.asarray(R_s_probs, dtype=float) * weight_s
    else:
        r_s_idx = np.empty(0, dtype=int)
        r_s_p = np.empty(0, dtype=float)
    uniform_idx = np.arange(N, dtype=int)
    source_idx = np.concatenate([uniform_idx, r_s_idx])
    source_probs = np.concatenate([uniform_probs, r_s_p])

    if len(R_t_support) == 0:
        return 0.0
    if len(R_t_support) == 1:
        y = int(R_t_support[0])
        # Exploit reset_mean_dist to avoid the full broadcast for the
        # uniform-source part.
        uniform_cost = gap * float(reset_mean_dist[y])
        r_s_cost = 0.0
        if r_s_idx.size > 0:
            r_s_cost = float(np.sum(r_s_p * metric[r_s_idx, y]))
        return uniform_cost + r_s_cost
    # R_t has two points.
    y1 = int(R_t_support[0])
    y2 = int(R_t_support[1])
    q1 = float(R_t_probs[0]) * weight_t
    return _transport_to_two_points(source_idx, source_probs, y1, y2, q1, metric)


# -----------------------------------------------------------------------------
# Fixed-point pair metric with internal induced state updates via bottleneck
# assignment over actions. This version captures action relabelings, so the
# Z/2 vertical-reflection symmetry of Four-Rooms collapses to zero distance
# between mirror partners.
# -----------------------------------------------------------------------------


def compute_four_rooms_fixed_point_bisimulation_metric(
    mdp: AB.TabularMDP,
    tol: float = 1e-4,
    max_iter: int = 200,
    verbose: bool = True,
) -> Array:
    """Build the Four-Rooms fixed-point pair metric on concrete `(state, action)` pairs.

    The fixed-point iterate keeps an internal state semimetric `d_s` only so
    the Wasserstein term and the action-relabeling bottleneck update can close
    on themselves. The public output is just the converged pair metric `d_sa`.
    """
    from itertools import permutations

    num_states = mdp.num_states
    num_actions = mdp.num_actions
    NA = num_states * num_actions
    N = num_states

    p_hit: Array = mdp.p_hit  # type: ignore[attr-defined]
    R_support: List[List[List[int]]] = mdp.R_support  # type: ignore[attr-defined]
    R_probs: List[List[List[float]]] = mdp.R_probs  # type: ignore[attr-defined]

    d_sa = np.zeros((NA, NA), dtype=float)
    d_s = np.zeros((N, N), dtype=float)

    # Precompute the list of action bijections. For |A|=4 this is 24 perms.
    bijections = [tuple(p) for p in permutations(range(num_actions))]

    # Precompute the upper-triangle of state-action pairs and the reward
    # difference matrix once.  ``rewards_flat[sa]`` is the scalar
    # r(s, a) for sa = s*A + a.
    rewards_flat = mdp.rewards.reshape(-1)  # shape (NA,)

    for iteration in range(max_iter):
        # Amortise the uniform-source broadcast used by the structured
        # Wasserstein helper: reset_mean_dist[y] = (1/N) * sum_k d_s[k, y].
        reset_mean_dist = d_s.mean(axis=0)

        # --- Step 1: update d_sa from current d_s. -----------------------
        d_sa_new = np.zeros_like(d_sa)
        for sa in range(NA):
            s = sa // num_actions
            a = sa % num_actions
            p_s = p_hit[s, a]
            R_s_sup = R_support[s][a]
            R_s_pr = R_probs[s][a]
            r_sa = rewards_flat[sa]
            # Iterate over upper triangle (sap > sa).
            for sap in range(sa + 1, NA):
                sp = sap // num_actions
                ap = sap % num_actions
                r_diff = abs(r_sa - rewards_flat[sap])
                transport = _wasserstein_structured(
                    p_s, R_s_sup, R_s_pr,
                    p_hit[sp, ap], R_support[sp][ap], R_probs[sp][ap],
                    d_s, reset_mean_dist,
                )
                val = r_diff + mdp.gamma * transport
                d_sa_new[sa, sap] = val
                d_sa_new[sap, sa] = val

        # --- Step 2: induce d_s from d_sa via bottleneck assignment ------
        d_s_new = np.zeros_like(d_s)
        A = num_actions
        for s in range(N):
            base_s = s * A
            for sp in range(s + 1, N):
                base_sp = sp * A
                # 4x4 cost matrix C[a, ap] = d_sa_new[s*A+a, sp*A+ap]
                C = d_sa_new[base_s:base_s + A, base_sp:base_sp + A]
                # min over bijections phi of max_a C[a, phi(a)].
                best = np.inf
                for phi in bijections:
                    # C[a, phi(a)] for a in 0..A-1
                    vmax = max(C[a, phi[a]] for a in range(A))
                    if vmax < best:
                        best = vmax
                d_s_new[s, sp] = best
                d_s_new[sp, s] = best

        delta = float(max(
            np.max(np.abs(d_sa_new - d_sa)),
            np.max(np.abs(d_s_new - d_s)),
        ))
        d_sa = d_sa_new
        d_s = d_s_new
        if verbose and (iteration % 5 == 0 or delta < tol):
            print(f"    pair-bisim iter {iteration}: max change = {delta:.3e}")
        if delta < tol:
            break
    return d_sa


def compute_four_rooms_one_step_bisimulation_metric(mdp: AB.TabularMDP) -> Array:
    """Build the Four-Rooms one-step pair distortion on concrete state-action pairs.

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


def induced_state_metric_from_pair_metric(
    distortion: Array,
    num_states: int,
    num_actions: int,
) -> Array:
    """Induce a state metric from a pair metric via bottleneck action matching."""
    from itertools import permutations

    state_metric = np.zeros((num_states, num_states), dtype=float)
    bijections = [tuple(p) for p in permutations(range(num_actions))]
    for state in range(num_states):
        base_state = state * num_actions
        for other_state in range(state + 1, num_states):
            base_other = other_state * num_actions
            cost_matrix = distortion[
                base_state : base_state + num_actions,
                base_other : base_other + num_actions,
            ]
            best = np.inf
            for phi in bijections:
                vmax = max(cost_matrix[action, phi[action]] for action in range(num_actions))
                if vmax < best:
                    best = vmax
            state_metric[state, other_state] = best
            state_metric[other_state, state] = best
    return state_metric


def mdp_fingerprint(mdp: AB.TabularMDP) -> str:
    """Fingerprint the concrete MDP so metric caches follow the dynamics."""
    h = hashlib.blake2b(digest_size=10)
    h.update(mdp.rewards.astype(np.float64).tobytes())
    h.update(mdp.transitions.astype(np.float64).tobytes())
    return h.hexdigest()


def pair_metric_cache_path(
    mdp: AB.TabularMDP,
    metric_kind: str,
    cache_dir: Path = CACHE_DIR,
) -> Path:
    """Return the cache path for the requested Four-Rooms pair metric."""
    cache_dir.mkdir(exist_ok=True)
    gamma = float(mdp.gamma)
    fingerprint = mdp_fingerprint(mdp)
    if metric_kind == "fixed_point":
        return cache_dir / (
            f"bisim_fixed_point_pairs_n{mdp.num_states}_a{mdp.num_actions}_"
            f"g{gamma:.4f}_{fingerprint}.npy"
        )
    if metric_kind == "one_step":
        return cache_dir / (
            f"bisim_one_step_pairs_n{mdp.num_states}_a{mdp.num_actions}_"
            f"g{gamma:.4f}_{fingerprint}.npy"
        )
    raise ValueError(f"Unknown Four-Rooms metric kind: {metric_kind}")

def load_or_compute_pair_metric(
    mdp: AB.TabularMDP,
    metric_kind: str = "fixed_point",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
    num_workers: int = 1,
) -> Array:
    """Load or compute the requested Four-Rooms pair metric."""
    pair_path = pair_metric_cache_path(mdp, metric_kind=metric_kind, cache_dir=cache_dir)
    if pair_path.exists():
        if verbose:
            print(f"  Loading cached Four-Rooms {metric_kind} metric from {pair_path.name}")
        return np.load(pair_path)

    if verbose:
        if metric_kind == "fixed_point":
            print("  Computing Four-Rooms fixed-point pair metric (no cache hit)...")
        else:
            print("  Computing Four-Rooms one-step pair metric (no cache hit)...")
    if metric_kind == "fixed_point":
        distortion = compute_four_rooms_fixed_point_bisimulation_metric(
            mdp,
            tol=1e-4,
            max_iter=200,
            verbose=verbose,
        )
    elif metric_kind == "one_step":
        distortion = compute_four_rooms_one_step_bisimulation_metric(mdp)
    else:
        raise ValueError(f"Unknown Four-Rooms metric kind: {metric_kind}")
    np.save(pair_path, distortion)
    return distortion


def _summarize_metric_distribution(metric: Array) -> dict[str, object]:
    """Summarize one symmetric metric matrix for terminal diagnostics."""
    iu, ju = np.triu_indices(metric.shape[0], k=1)
    off_diag = metric[iu, ju]
    return {
        "shape": tuple(int(x) for x in metric.shape),
        "min_off_diag": float(off_diag.min()) if off_diag.size else 0.0,
        "max_off_diag": float(off_diag.max()) if off_diag.size else 0.0,
        "num_lt_1e-6": int(np.sum(off_diag < 1e-6)),
        "num_lt_1e-3": int(np.sum(off_diag < 1e-3)),
        "num_lt_1e-1": int(np.sum(off_diag < 1e-1)),
    }


def precompute_metric_cache(
    eta: float = 0.10,
    gamma: float = 0.99,
    metric_kind: str = "fixed_point",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
) -> dict[str, object]:
    """Materialize the Four-Rooms pair metric cache and return diagnostics."""
    mdp = build_four_rooms_mdp(eta=eta, gamma=gamma)
    pair_cache = pair_metric_cache_path(mdp, metric_kind=metric_kind, cache_dir=cache_dir)
    cache_hit = pair_cache.exists()
    if verbose:
        if cache_hit:
            print(f"Loading cached pair metric from {pair_cache.name}")
        else:
            if metric_kind == "fixed_point":
                print("Computing Four-Rooms fixed-point pair metric (no cache hit)...")
            else:
                print("Computing Four-Rooms one-step pair metric (no cache hit)...")
    start = time.perf_counter()
    distortion = load_or_compute_pair_metric(
        mdp,
        metric_kind=metric_kind,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    state_metric = induced_state_metric_from_pair_metric(
        distortion,
        num_states=mdp.num_states,
        num_actions=mdp.num_actions,
    )
    elapsed = time.perf_counter() - start
    diagnostics: dict[str, object] = {
        "mdp": mdp,
        "metric_kind": metric_kind,
        "pair_cache_path": pair_cache,
        "cache_hit": cache_hit,
        "elapsed_seconds": float(elapsed),
        "state_metric": state_metric,
        "distortion": distortion,
        "state_metric_summary": _summarize_metric_distribution(state_metric),
    }
    sigma = sigma_permutation(mdp, _build_cell_index(_build_wall_mask())[0])
    sigma_distances = state_metric[np.arange(mdp.num_states), sigma]
    moved = sigma != np.arange(mdp.num_states)
    diagnostics["distortion_summary"] = _summarize_metric_distribution(distortion)
    diagnostics["sigma_summary"] = {
        "num_moved_states": int(np.sum(moved)),
        "num_sigma_lt_1e-6": int(np.sum(sigma_distances[moved] < 1e-6)),
        "num_sigma_lt_1e-3": int(np.sum(sigma_distances[moved] < 1e-3)),
        "min_sigma_distance": float(np.min(sigma_distances[moved])) if np.any(moved) else 0.0,
        "median_sigma_distance": float(np.median(sigma_distances[moved])) if np.any(moved) else 0.0,
        "max_sigma_distance": float(np.max(sigma_distances[moved])) if np.any(moved) else 0.0,
    }
    return diagnostics


def load_distortion(
    mdp: AB.TabularMDP,
    metric_kind: str = "fixed_point",
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
    num_workers: int = 1,
) -> Array:
    """Return the state-action pair distortion used by Four-Rooms experiments."""
    return load_or_compute_pair_metric(
        mdp,
        metric_kind=metric_kind,
        cache_dir=cache_dir,
        verbose=verbose,
        num_workers=num_workers,
    )


def build_summary_mutator(
    eta: float,
    gamma: float,
    metric_kind: str,
    beta_schedule: list[float],
    adaptive_beta_schedule: list[float] | None,
):
    """Attach Four-Rooms-specific metadata to the shared summary payload."""
    distance = (
        "one-step Bellman-compatible state-action metric"
        if metric_kind == "one_step"
        else "fixed-point state-action bisimulation metric"
    )

    def mutate(summary_payload: dict[str, object]) -> None:
        config = summary_payload.setdefault("config", {})
        notes = summary_payload.setdefault("notes", {})
        if isinstance(config, dict):
            config["scenario"] = "four_rooms"
            config["abstraction"] = "sa"
            config["solver"] = "modelbased"
            config["environment"] = "four_rooms_random_goal"
            config["grid_size"] = 11
            config["num_goals"] = int(NUM_GOALS)
            config["room_centers"] = [list(center) for center in ROOM_CENTERS]
            config["slip_probability"] = float(eta)
            config["gamma"] = float(gamma)
            config["metric_kind"] = metric_kind
            config["distance"] = distance
            config["mu"] = "uniform over concrete state-action pairs"
            config["goal_reset"] = (
                "uniform over other three rooms; agent stays at former goal cell"
            )
            config["reward"] = "+1 on arrival at current goal center; 0 otherwise"
            config["beta_schedule"] = [float(beta) for beta in beta_schedule]
            config["adaptive_beta_schedule"] = [
                float(beta) for beta in (adaptive_beta_schedule or beta_schedule)
            ]
        if isinstance(notes, dict):
            notes["metric"] = distance
        summary_payload["adaptive_trigger_metric"] = (
            "abstract_residual = ||Fbar_eta Qbar - Qbar||_inf"
        )
        summary_payload["adaptive_transition_rule"] = (
            "advance by multiple beta rungs in one step when the transferred iterate at each skipped rung already has residual below that rung's abstraction-error scale"
        )
        summary_payload["policy_value_estimator"] = (
            "exact tabular evaluation: solve (I - gamma P^pi) V = r^pi in closed form"
        )

    return mutate
