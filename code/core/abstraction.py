"""Tabular abstraction objects, BA fitting, and abstract/concrete operators."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import numpy as np
Array = np.ndarray
PROJECTED_INFORMATION_SPLIT = "I(S,A;Sbar)+I(S,A;Abar|Sbar)"


def normalize_solver_kind(solver_kind: str) -> str:
    """Map user-facing and legacy solver names onto canonical internal identifiers."""
    if solver_kind == "structured_lambda0":
        return "lambda"
    return str(solver_kind)


################################################################################
# Class definitions
@dataclass
class TabularMDP:
    """Tabular discounted MDP with explicit transition and reward tensors."""

    transitions: Array
    rewards: Array
    gamma: float
    state_labels: List[Any]
    action_labels: List[str]

    @property
    def num_states(self) -> int:
        """Return the number of states."""
        return int(self.transitions.shape[0])

    @property
    def num_actions(self) -> int:
        """Return the number of actions."""
        return int(self.transitions.shape[1])

    @property
    def num_state_action_pairs(self) -> int:
        """Return the number of state-action pairs."""
        return int(self.transitions.shape[0] * self.transitions.shape[1])

@dataclass
class StateActionAbstraction:
    """State-action abstraction with soft encoder grounding and hard decoder representatives."""

    beta: float
    encoder: Array
    posterior: Array
    decoder: Array
    full_encoder: Array
    full_decoder: Array
    solver_kind: str = "flat"
    state_encoder: Array | None = None
    action_encoder: Array | None = None
    state_decoder: Array | None = None
    action_decoder: Array | None = None

    @property
    def num_abstract(self) -> int:
        """Return the number of abstract action-state pairs."""
        return int(self.encoder.shape[1])


################################################################################
# Helper functions
def _logsumexp(values: Array, axis: int) -> Array:
    """Compute log(sum(exp(values))) in a numerically stable way over a specified axis."""
    maxima = np.max(values, axis=axis, keepdims=True)
    stable = values - maxima
    summed = np.sum(np.exp(stable), axis=axis, keepdims=True)
    return maxima + np.log(summed)


def _sanitize_row_stochastic(matrix: Array) -> Array:
    """Ensure a matrix is row-stochastic (rows sum to 1), handling NaNs/Infs safely."""
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return matrix / row_sums


def _compute_abstract_marginal(mu: Array, encoder: Array) -> Array:
    """Compute the abstract marginal induced by `mu` and the encoder."""
    with np.errstate(all="ignore"):
        marginal = np.nan_to_num(mu @ encoder, nan=0.0)
    return np.clip(marginal, 1e-15, None)


def _compute_posterior(mu: Array, encoder: Array, abstract_marginal: Array) -> Array:
    """Compute p(concrete | abstract) from the encoder and abstract marginal."""
    return (mu[:, None] * encoder) / abstract_marginal[None, :]


################################################################################
# Abstraction utils
def distortion_profile(encoder: Array, decoder: Array, distance: Array) -> Array:
    """Compute the per-concrete-pair distortion induced by an encoder/decoder pair."""
    encoder = np.asarray(encoder, dtype=float)
    decoder = np.asarray(decoder, dtype=int)
    with np.errstate(all="ignore"):
        profile = np.sum(encoder * distance[:, decoder], axis=1)
    return np.asarray(
        np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=float,
    )


def expected_distortion_components(mu: Array, encoder: Array, decoder: Array, distance: Array) -> float:
    """Compute the expected distortion induced by an encoder/decoder pair."""
    profile = distortion_profile(encoder, decoder, distance)
    with np.errstate(all="ignore"):
        distortion = np.sum(np.asarray(mu, dtype=float) * profile)
    return float(np.nan_to_num(distortion, nan=0.0, posinf=0.0, neginf=0.0))


def max_distortion_components(mu: Array, encoder: Array, decoder: Array, distance: Array) -> float:
    """Compute the worst-case per-pair distortion induced by an encoder/decoder pair."""
    del mu
    profile = distortion_profile(encoder, decoder, distance)
    return float(np.max(profile)) if profile.size else 0.0


def compute_abstraction_error(
    mu: Array,
    encoder: Array,
    decoder: Array,
    distance: Array,
    mode: str = "average",
) -> float:
    """Compute the scalar abstraction error under the requested aggregation mode."""
    if mode == "average":
        return expected_distortion_components(mu, encoder, decoder, distance)
    if mode == "max":
        return max_distortion_components(mu, encoder, decoder, distance)
    raise ValueError(f"Unsupported abstraction error mode: {mode!r}")


def mutual_information(mu: Array, encoder: Array) -> float:
    """Compute I_mu(X;Z) in bits for a soft encoder and concrete prior `mu`."""
    encoder = np.asarray(encoder, dtype=float)
    mu = np.asarray(mu, dtype=float)
    abstract_marginal = _compute_abstract_marginal(mu, encoder)
    with np.errstate(all="ignore"):
        log_ratio = (
            np.log(np.clip(encoder, 1e-15, None)) - np.log(abstract_marginal)[None, :]
        ) / np.log(2.0)
        contribution = mu[:, None] * encoder * log_ratio
    value = float(np.nan_to_num(np.sum(contribution), nan=0.0, posinf=0.0, neginf=0.0))
    return max(0.0, value)


def _information_sum(joint: Array, numerator: Array, denominator: Array) -> float:
    """Compute sum joint * log2(numerator / denominator), skipping zero-mass terms."""
    joint = np.asarray(joint, dtype=float)
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    mask = (joint > 0.0) & (numerator > 0.0) & (denominator > 0.0)
    if not np.any(mask):
        return 0.0
    with np.errstate(all="ignore"):
        value = float(
            np.sum(joint[mask] * (np.log(numerator[mask] / denominator[mask]) / np.log(2.0)))
        )
    return max(0.0, float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)))


def joint_code_information_decomposition(
    mu: Array,
    encoder: Array,
    num_actions: int,
) -> Dict[str, float]:
    """Decompose a flat state-action encoder rate as I(S;Z) + I(A;Z|S).

    Here `Z` is the abstract code produced by the flat state-action encoder
    `encoder[z | s,a]`. This is an exact chain-rule decomposition of
    `mutual_information(mu, encoder)` when `mu` is the concrete state-action
    prior.
    """
    encoder = np.asarray(encoder, dtype=float)
    mu = np.asarray(mu, dtype=float)
    num_actions = int(num_actions)
    if num_actions <= 0:
        raise ValueError("num_actions must be positive.")
    if encoder.shape[0] != mu.size:
        raise ValueError("encoder and mu must have the same concrete-pair count.")
    if mu.size % num_actions != 0:
        raise ValueError("mu size must be divisible by num_actions.")

    num_states = int(mu.size // num_actions)
    encoder_saz = encoder.reshape(num_states, num_actions, encoder.shape[1])
    mu_sa = mu.reshape(num_states, num_actions)

    joint_saz = mu_sa[:, :, None] * encoder_saz
    joint_sz = np.sum(joint_saz, axis=1)
    marginal_s = np.sum(mu_sa, axis=1)
    marginal_z = np.sum(joint_sz, axis=0)

    state_denominator = marginal_s[:, None] * marginal_z[None, :]
    state_information = _information_sum(joint_sz, joint_sz, state_denominator)

    action_numerator = joint_saz * marginal_s[:, None, None]
    action_denominator = mu_sa[:, :, None] * joint_sz[:, None, :]
    conditional_action_information = _information_sum(
        joint_saz,
        action_numerator,
        action_denominator,
    )

    return {
        "joint_code_state_information": state_information,
        "joint_code_conditional_action_information": conditional_action_information,
        "joint_code_information": state_information + conditional_action_information,
    }


def decoder_projected_information_decomposition(
    mu: Array,
    encoder: Array,
    decoder: Array,
    num_actions: int,
) -> Dict[str, object]:
    """Compute I(S,A;Sbar) and I(S,A;Abar|Sbar) from decoder labels.

    The flat decoder maps each abstract code to a representative concrete
    state-action pair. We use the representative's state and action components
    as post-hoc `Sbar` and `Abar` labels, then aggregate the soft encoder onto
    those labels before computing the chain-rule terms whose sum is
    I(S,A;Sbar,Abar).
    """
    encoder = np.asarray(encoder, dtype=float)
    decoder = np.asarray(decoder, dtype=int)
    mu = np.asarray(mu, dtype=float)
    num_actions = int(num_actions)
    if num_actions <= 0:
        raise ValueError("num_actions must be positive.")
    if encoder.shape[0] != mu.size:
        raise ValueError("encoder and mu must have the same concrete-pair count.")
    if encoder.shape[1] != decoder.size:
        raise ValueError("decoder size must match the encoder abstract-code count.")
    if mu.size % num_actions != 0:
        raise ValueError("mu size must be divisible by num_actions.")

    num_states = int(mu.size // num_actions)
    decoder_states = decoder // num_actions
    decoder_actions = decoder % num_actions
    if np.any(decoder_states < 0) or np.any(decoder_states >= num_states):
        raise ValueError("decoder contains state indices outside the concrete state range.")
    if np.any(decoder_actions < 0) or np.any(decoder_actions >= num_actions):
        raise ValueError("decoder contains action indices outside the concrete action range.")

    # joint_bar[s, a, sbar, abar] = p(s,a,sbar,abar)
    joint_bar = np.zeros(
        (num_states, num_actions, num_states, num_actions),
        dtype=float,
    )
    weighted_encoder = mu[:, None] * encoder
    flat_joint_bar = joint_bar.reshape(mu.size, num_states, num_actions)
    for abstract_index, (bar_state, bar_action) in enumerate(
        zip(decoder_states.tolist(), decoder_actions.tolist())
    ):
        flat_joint_bar[:, int(bar_state), int(bar_action)] += weighted_encoder[:, abstract_index]

    mu_sa = mu.reshape(num_states, num_actions)
    joint_s_a_bar_state = np.sum(joint_bar, axis=3)
    marginal_bar_state = np.sum(joint_s_a_bar_state, axis=(0, 1))
    state_denominator = mu_sa[:, :, None] * marginal_bar_state[None, None, :]
    bar_state_information = _information_sum(
        joint_s_a_bar_state,
        joint_s_a_bar_state,
        state_denominator,
    )

    joint_bar_state_bar_action = np.sum(joint_bar, axis=(0, 1))
    action_numerator = joint_bar * marginal_bar_state[None, None, :, None]
    action_denominator = (
        joint_s_a_bar_state[:, :, :, None]
        * joint_bar_state_bar_action[None, None, :, :]
    )
    bar_conditional_action_information = _information_sum(
        joint_bar,
        action_numerator,
        action_denominator,
    )

    return {
        "bar_state_information": bar_state_information,
        "bar_conditional_action_information": bar_conditional_action_information,
        "bar_information_sum": (
            bar_state_information + bar_conditional_action_information
        ),
        "bar_information_split": PROJECTED_INFORMATION_SPLIT,
    }


def state_action_information_decomposition(
    mu: Array,
    encoder: Array,
    decoder: Array,
    num_actions: int,
) -> Dict[str, object]:
    """Return flat-code and decoder-projected information splits in bits."""
    rows = joint_code_information_decomposition(mu, encoder, num_actions)
    rows.update(
        decoder_projected_information_decomposition(
            mu,
            encoder,
            decoder,
            num_actions,
        )
    )
    return rows


def summarize_abstraction_family(
    mu: Array,
    abstractions: Sequence[StateActionAbstraction],
    distance: Array,
    mode: str = "average",
    num_actions: int | None = None,
) -> List[Dict[str, object]]:
    """Build compact per-beta metadata for a fitted abstraction family."""
    rows: List[Dict[str, object]] = []
    for abstraction in abstractions:
        row: Dict[str, object] = {
            "beta": float(abstraction.beta),
            "effective_abstract_pairs": int(abstraction.num_abstract),
            "abstraction_error": compute_abstraction_error(
                mu,
                abstraction.encoder,
                abstraction.decoder,
                distance,
                mode=mode,
            ),
            "mutual_information": mutual_information(
                mu,
                abstraction.encoder,
            ),
            "solver_kind": str(abstraction.solver_kind),
        }
        if abstraction.state_encoder is not None:
            row["effective_abstract_states"] = int(np.asarray(abstraction.state_encoder).shape[1])
        if abstraction.action_encoder is not None:
            row["effective_abstract_actions"] = int(np.asarray(abstraction.action_encoder).shape[1])
        if num_actions is not None:
            row.update(
                state_action_information_decomposition(
                    mu,
                    abstraction.encoder,
                    abstraction.decoder,
                    int(num_actions),
                )
            )
        rows.append(row)
    return rows


def ground_state_action_abstract_q(abstraction, abstract_q: Array) -> Array:
    """Ground abstract pair-values onto concrete pairs using the full soft encoder."""
    abstract_q = np.asarray(abstract_q, dtype=float)
    encoder = np.asarray(abstraction.encoder, dtype=float)
    if not np.isfinite(encoder).all():
        raise FloatingPointError("Non-finite entries found in abstraction encoder.")
    if not np.isfinite(abstract_q).all():
        raise FloatingPointError("Non-finite entries found in abstract Q iterate.")
    with np.errstate(all="ignore"):
        grounded_q = np.einsum("ij,j->i", encoder, abstract_q, optimize=True)
    if not np.isfinite(grounded_q).all():
        raise FloatingPointError("Grounded concrete Q contains non-finite entries.")
    return np.asarray(grounded_q, dtype=float)


def lift_state_action_concrete_q(abstraction, concrete_q: Array) -> Array:
    """Read out concrete pair-values at deterministic decoder representatives."""
    concrete_q = np.asarray(concrete_q, dtype=float)
    return np.asarray(concrete_q[np.asarray(abstraction.decoder, dtype=int)], dtype=float)


################################################################################
# BA helper functions
def _initialize_decoder(num_states: int, num_abstract: int) -> Array:
    """Initialize decoder representatives by spreading them over the concrete index range."""
    return np.linspace(0, num_states - 1, num=num_abstract, dtype=int)


def _initialize_encoder(num_states: int, num_abstract: int) -> Array:
    """Initialize the encoder with a uniform distribution over abstract states."""
    return np.full((num_states, num_abstract), 1.0 / float(num_abstract), dtype=float)


def diversify_decoder(decoder: Array, distance: Array) -> Array:
    """Move duplicate decoder representatives to uncovered high-distortion states."""
    diversified = np.array(decoder, dtype=int, copy=True)
    seen = set()
    unique_states: List[int] = []
    duplicate_positions: List[int] = []
    for position, state in enumerate(diversified.tolist()):
        if state in seen:
            duplicate_positions.append(position)
        else:
            seen.add(state)
            unique_states.append(state)

    if not duplicate_positions:
        return diversified

    current_cover = np.min(distance[:, unique_states], axis=1)
    for position in duplicate_positions:
        candidate = int(np.argmax(current_cover))
        diversified[position] = candidate
        current_cover = np.minimum(current_cover, distance[:, candidate])
    return diversified


def diversify_encoder(encoder: Array, smoothing: float = 1e-3) -> Array:
    """Slightly smooth an encoder before warm-starting the next BA solve."""
    num_abstract = encoder.shape[1]
    smoothed = (1.0 - smoothing) * encoder + smoothing / float(num_abstract)
    return _sanitize_row_stochastic(smoothed)


def _restrict_to_active_abstracts(
    mu: Array,
    encoder: Array,
    decoder: Array,
    abstract_marginal: Array,
    prune_mass_threshold: float,
) -> tuple[Array, Array, Array]:
    """Prune negligible abstract states and renormalize the surviving encoder."""
    active_mask = abstract_marginal > prune_mass_threshold
    if not np.any(active_mask):
        active_mask[np.argmax(abstract_marginal)] = True

    encoder_active = _sanitize_row_stochastic(encoder[:, active_mask])
    abstract_marginal_active = _compute_abstract_marginal(mu, encoder_active)
    posterior_active = _compute_posterior(mu, encoder_active, abstract_marginal_active)
    decoder_active = np.asarray(decoder[active_mask], dtype=int)
    return encoder_active, posterior_active, decoder_active


################################################################################
# BA soft abstraction fit
def _fit_flat_soft_abstraction(
    distortion: Array,
    mu: Array,
    beta: float,
    num_abstract: int,
    decoder_init: Array | None = None,
    encoder_init: Array | None = None,
    diversify_warm_start: bool = True,
    prune_mass_threshold: float = 1e-4,
    tolerance: float = 1e-10,
    max_outer: int = 50,
    max_inner: int = 1000,
) -> StateActionAbstraction:
    """Run the flat BA-style soft abstraction fit with alternating encoder/decoder updates.

    When warm-start arrays are provided, they are diversified by default before
    optimization so duplicate representatives and overly sharp reused encoders do
    not stall the next solve.
    """

    # Initialize decoder and encoder.
    num_concrete = distortion.shape[0]
    decoder = (
        np.array(decoder_init, dtype=int, copy=True)
        if decoder_init is not None
        else _initialize_decoder(num_concrete, num_abstract)
    )
    encoder = (
        np.array(encoder_init, dtype=float, copy=True)
        if encoder_init is not None
        else _initialize_encoder(num_concrete, num_abstract)
    )
    if diversify_warm_start:
        if decoder_init is not None:
            decoder = diversify_decoder(decoder, distortion)
        if encoder_init is not None:
            encoder = diversify_encoder(encoder)

    # Blahut-Arimoto-style optimization of the encoder and decoder.
    for _ in range(max_outer):
        concrete_to_decoder_distortion = distortion[:, decoder]
        abstract_marginal = _compute_abstract_marginal(mu, encoder)

        for _ in range(max_inner):
            # Solving fixed point equation for encoder and marginal
            logits = np.log(abstract_marginal)[None, :] - beta * concrete_to_decoder_distortion
            log_norm = _logsumexp(logits, axis=1)
            updated_encoder = _sanitize_row_stochastic(np.exp(logits - log_norm))
            updated_abstract_marginal = _compute_abstract_marginal(mu, updated_encoder)
            # Check for convergence
            if (
                float(np.max(np.abs(updated_encoder - encoder))) < tolerance
                and float(np.max(np.abs(updated_abstract_marginal - abstract_marginal))) < tolerance
            ):
                encoder = updated_encoder
                abstract_marginal = updated_abstract_marginal
                break
            encoder = updated_encoder
            abstract_marginal = updated_abstract_marginal

        # Update decoder
        posterior = _compute_posterior(mu, encoder, abstract_marginal)
        updated_decoder = np.array(decoder, copy=True)
        for abstract in range(num_abstract):
            weights = posterior[:, abstract]
            if float(np.sum(weights)) <= 1e-14: # Skip if the abstract is effectively inactive
                continue
            # For each abstract, compute expected distortion to every concrete and pick the minimal one
            with np.errstate(all="ignore"):
                objective = np.nan_to_num(weights @ distortion, nan=np.inf, posinf=np.inf)
            updated_decoder[abstract] = int(np.argmin(objective))

        # Stop if the decoder did not change
        if np.array_equal(updated_decoder, decoder):
            decoder = updated_decoder
            break
        decoder = updated_decoder

    # Restrict to active abstracts for downstream planning/analysis.
    abstract_marginal = _compute_abstract_marginal(mu, encoder)
    encoder_active, posterior_active, decoder_active = _restrict_to_active_abstracts(
        mu,
        encoder,
        decoder,
        abstract_marginal,
        prune_mass_threshold,
    )

    return StateActionAbstraction(
        beta=beta,
        encoder=encoder_active,
        posterior=posterior_active,
        decoder=decoder_active,
        full_encoder=encoder,  # used for iteration of family of abstractions
        full_decoder=decoder,  # used for iteration of family of abstractions
        solver_kind="flat",
    )


def _build_state_action_marginals(mu: Array, num_states: int, num_actions: int) -> tuple[Array, Array]:
    """Factor a concrete pair prior into state marginals and conditional action priors."""
    mu_pairs = np.asarray(mu, dtype=float).reshape(num_states, num_actions)
    mu_states = np.sum(mu_pairs, axis=1)
    safe_state_mass = np.where(mu_states > 1e-15, mu_states, 1.0)
    mu_action = mu_pairs / safe_state_mass[:, None]
    zero_mass_states = mu_states <= 1e-15
    if np.any(zero_mass_states):
        mu_action[zero_mass_states, :] = 1.0 / float(num_actions)
    return np.asarray(mu_states, dtype=float), _sanitize_row_stochastic(mu_action)


def _initialize_state_encoder(num_states: int, num_abstract_states: int) -> Array:
    """Initialize the structured state encoder uniformly."""
    return np.full(
        (num_states, num_abstract_states),
        1.0 / float(num_abstract_states),
        dtype=float,
    )


def _initialize_action_encoder(num_pairs: int, num_abstract_actions: int) -> Array:
    """Initialize the structured action encoder uniformly."""
    return np.full(
        (num_pairs, num_abstract_actions),
        1.0 / float(num_abstract_actions),
        dtype=float,
    )


def _initial_action_matching_encoder(
    num_states: int,
    num_actions: int,
    num_abstract_actions: int,
) -> Array:
    """Initialize the action encoder by a simple action-identity matching rule."""
    num_pairs = int(num_states * num_actions)
    encoder = np.zeros((num_pairs, num_abstract_actions), dtype=float)
    for state in range(num_states):
        for action in range(num_actions):
            pair_index = int(state * num_actions + action)
            encoder[pair_index, int(action % num_abstract_actions)] = 1.0
    return encoder


def _structured_flat_decoder(
    state_decoder: Array,
    action_decoder: Array,
    num_actions: int,
) -> Array:
    """Flatten structured decoder slices into concrete pair representatives."""
    state_decoder = np.asarray(state_decoder, dtype=int)
    action_decoder = np.asarray(action_decoder, dtype=int)
    return (
        state_decoder[:, None] * int(num_actions) + action_decoder
    ).reshape(-1)


def _structured_flat_encoder(
    state_encoder: Array,
    action_encoder: Array,
    num_states: int,
    num_actions: int,
) -> Array:
    """Flatten structured state/action encoders into a pair-to-abstract encoder."""
    state_encoder = np.asarray(state_encoder, dtype=float)
    action_encoder = np.asarray(action_encoder, dtype=float)
    num_abstract_states = int(state_encoder.shape[1])
    num_abstract_actions = int(action_encoder.shape[1])
    num_pairs = int(num_states * num_actions)
    flat_encoder = np.zeros((num_pairs, num_abstract_states * num_abstract_actions), dtype=float)
    for state in range(num_states):
        state_slice = slice(state * num_actions, (state + 1) * num_actions)
        pair_encoder = (
            state_encoder[state, :, None] * action_encoder[state_slice, None, :]
        ).reshape(num_actions, num_abstract_states * num_abstract_actions)
        flat_encoder[state_slice, :] = pair_encoder
    return _sanitize_row_stochastic(flat_encoder)


def _warm_start_structured_state_encoder(
    encoder_init: Array | None,
    mu_action: Array,
    num_states: int,
    num_actions: int,
    num_abstract_states: int,
    num_abstract_actions: int,
) -> Array:
    """Derive a structured state-encoder warm start from a flat warm start when available."""
    if encoder_init is None:
        return _initialize_state_encoder(num_states, num_abstract_states)
    flat_encoder = np.asarray(encoder_init, dtype=float)
    if flat_encoder.shape[0] != num_states * num_actions:
        return _initialize_state_encoder(num_states, num_abstract_states)
    usable_columns = min(flat_encoder.shape[1], num_abstract_states * num_abstract_actions)
    reshaped = np.zeros((num_states, num_actions, num_abstract_states, num_abstract_actions), dtype=float)
    reshaped.reshape(num_states, num_actions, -1)[..., :usable_columns] = flat_encoder[:, :usable_columns].reshape(
        num_states,
        num_actions,
        usable_columns,
    )
    aggregated = np.sum(reshaped, axis=3)
    state_encoder = np.einsum("sa,sab->sb", mu_action, aggregated)
    return _sanitize_row_stochastic(state_encoder)


def _warm_start_structured_action_encoder(
    encoder_init: Array | None,
    num_states: int,
    num_actions: int,
    num_abstract_states: int,
    num_abstract_actions: int,
) -> Array:
    """Derive a structured action-encoder warm start from a flat warm start when available."""
    if encoder_init is None:
        return _initialize_action_encoder(num_states * num_actions, num_abstract_actions)
    flat_encoder = np.asarray(encoder_init, dtype=float)
    if flat_encoder.shape[0] != num_states * num_actions:
        return _initialize_action_encoder(num_states * num_actions, num_abstract_actions)
    usable_columns = min(flat_encoder.shape[1], num_abstract_states * num_abstract_actions)
    reshaped = np.zeros((num_states * num_actions, num_abstract_states, num_abstract_actions), dtype=float)
    reshaped.reshape(num_states * num_actions, -1)[:, :usable_columns] = flat_encoder[:, :usable_columns]
    action_encoder = np.sum(reshaped, axis=1)
    return _sanitize_row_stochastic(action_encoder)


def _warm_start_structured_decoder(
    decoder_init: Array | None,
    num_states: int,
    num_actions: int,
    num_abstract_states: int,
    num_abstract_actions: int,
) -> tuple[Array, Array]:
    """Derive structured decoder warm starts from a flat decoder when available."""
    total_abstract_pairs = int(num_abstract_states * num_abstract_actions)
    if decoder_init is None:
        state_decoder = _initialize_decoder(num_states, num_abstract_states)
        action_decoder = np.tile(
            np.arange(num_abstract_actions, dtype=int) % int(num_actions),
            (num_abstract_states, 1),
        )
        return state_decoder, action_decoder

    flat_decoder = np.asarray(decoder_init, dtype=int)
    if flat_decoder.size < total_abstract_pairs:
        state_decoder = _initialize_decoder(num_states, num_abstract_states)
        action_decoder = np.tile(
            np.arange(num_abstract_actions, dtype=int) % int(num_actions),
            (num_abstract_states, 1),
        )
        return state_decoder, action_decoder

    reshaped = flat_decoder[:total_abstract_pairs].reshape(num_abstract_states, num_abstract_actions)
    state_decoder = np.asarray(reshaped[:, 0] // int(num_actions), dtype=int)
    action_decoder = np.asarray(reshaped % int(num_actions), dtype=int)
    state_decoder = np.clip(state_decoder, 0, num_states - 1)
    action_decoder = np.clip(action_decoder, 0, num_actions - 1)
    return state_decoder, action_decoder


def _compute_structured_state_cost(
    distortion: Array,
    mu_action: Array,
    action_encoder: Array,
    state_decoder: Array,
    action_decoder: Array,
    num_states: int,
    num_actions: int,
) -> Array:
    """Compute the structured state cost used by the lambda-zero state-encoder step."""
    num_abstract_states = int(state_decoder.shape[0])
    num_abstract_actions = int(action_decoder.shape[1])
    state_cost = np.zeros((num_states, num_abstract_states), dtype=float)

    for state in range(num_states):
        for abstract_state in range(num_abstract_states):
            total = 0.0
            for action in range(num_actions):
                pair_index = int(state * num_actions + action)
                action_weights = action_encoder[pair_index]
                pair_cost = 0.0
                for abstract_action in range(num_abstract_actions):
                    decoded_pair = int(
                        state_decoder[abstract_state] * num_actions
                        + action_decoder[abstract_state, abstract_action]
                    )
                    pair_cost += float(action_weights[abstract_action]) * float(
                        distortion[pair_index, decoded_pair]
                    )
                total += float(mu_action[state, action]) * pair_cost
            state_cost[state, abstract_state] = total
    return state_cost


def _initialize_structured_state_encoder_from_cost(
    state_cost: Array,
    mu_states: Array,
    beta: float,
) -> Array:
    """Initialize the structured state encoder from a non-uniform state-cost matrix."""
    num_states, num_abstract_states = state_cost.shape
    if num_abstract_states <= 1:
        return np.ones((num_states, num_abstract_states), dtype=float)

    abstract_marginal = np.full(num_abstract_states, 1.0 / float(num_abstract_states), dtype=float)
    logits = np.log(abstract_marginal)[None, :] - float(max(beta, 1.0)) * np.asarray(state_cost, dtype=float)
    # Add a tiny deterministic column bias to avoid exact symmetry under tied costs.
    logits += 1e-9 * np.arange(num_abstract_states, dtype=float)[None, :]
    log_norm = _logsumexp(logits, axis=1)
    encoder = _sanitize_row_stochastic(np.exp(logits - log_norm))
    # One BA refinement against the induced marginal is enough to break symmetry cleanly.
    updated_marginal = _compute_abstract_marginal(mu_states, encoder)
    logits = np.log(updated_marginal)[None, :] - float(max(beta, 1.0)) * np.asarray(state_cost, dtype=float)
    logits += 1e-9 * np.arange(num_abstract_states, dtype=float)[None, :]
    log_norm = _logsumexp(logits, axis=1)
    return _sanitize_row_stochastic(np.exp(logits - log_norm))


def _diversify_state_decoder_from_state_cost(
    state_decoder: Array,
    state_cost: Array,
) -> Array:
    """Move duplicate decoded states to poorly covered concrete states."""
    diversified = np.array(state_decoder, dtype=int, copy=True)
    seen = set()
    unique_states: List[int] = []
    duplicate_positions: List[int] = []
    for position, state in enumerate(diversified.tolist()):
        if state in seen:
            duplicate_positions.append(position)
        else:
            seen.add(state)
            unique_states.append(state)

    if not duplicate_positions:
        return diversified

    current_cover = np.min(state_cost[:, unique_states], axis=1)
    for position in duplicate_positions:
        candidate = int(np.argmax(current_cover))
        diversified[position] = candidate
        current_cover = np.minimum(current_cover, state_cost[:, candidate])
    return diversified


def _update_lambda_zero_action_encoder(
    distortion: Array,
    state_encoder: Array,
    state_decoder: Array,
    action_decoder: Array,
    num_states: int,
    num_actions: int,
) -> Array:
    """Best-response tied action matching update for the structured lambda-zero solver."""
    num_pairs = int(num_states * num_actions)
    num_abstract_states = int(state_encoder.shape[1])
    num_abstract_actions = int(action_decoder.shape[1])
    action_encoder = np.zeros((num_pairs, num_abstract_actions), dtype=float)

    for state in range(num_states):
        state_weights = state_encoder[state]
        for action in range(num_actions):
            pair_index = int(state * num_actions + action)
            costs = np.zeros(num_abstract_actions, dtype=float)
            for abstract_action in range(num_abstract_actions):
                total = 0.0
                for abstract_state in range(num_abstract_states):
                    decoded_pair = int(
                        state_decoder[abstract_state] * num_actions
                        + action_decoder[abstract_state, abstract_action]
                    )
                    total += float(state_weights[abstract_state]) * float(
                        distortion[pair_index, decoded_pair]
                    )
                costs[abstract_action] = total
            best_action = int(np.argmin(costs))
            action_encoder[pair_index, best_action] = 1.0
    return action_encoder


def _update_lambda_zero_state_encoder(
    distortion: Array,
    mu_states: Array,
    mu_action: Array,
    state_encoder: Array,
    action_encoder: Array,
    state_decoder: Array,
    action_decoder: Array,
    beta: float,
    tolerance: float,
    max_inner: int,
    num_states: int,
    num_actions: int,
) -> Array:
    """BA state-encoder update for the structured lambda-zero solver."""
    state_cost = _compute_structured_state_cost(
        distortion,
        mu_action,
        action_encoder,
        state_decoder,
        action_decoder,
        num_states,
        num_actions,
    )
    encoder = np.asarray(state_encoder, dtype=float)
    abstract_marginal = _compute_abstract_marginal(mu_states, encoder)
    for _ in range(max_inner):
        logits = np.log(abstract_marginal)[None, :] - beta * state_cost
        log_norm = _logsumexp(logits, axis=1)
        updated_encoder = _sanitize_row_stochastic(np.exp(logits - log_norm))
        updated_marginal = _compute_abstract_marginal(mu_states, updated_encoder)
        if (
            float(np.max(np.abs(updated_encoder - encoder))) < tolerance
            and float(np.max(np.abs(updated_marginal - abstract_marginal))) < tolerance
        ):
            encoder = updated_encoder
            break
        encoder = updated_encoder
        abstract_marginal = updated_marginal
    return encoder


def _update_lambda_zero_decoder(
    distortion: Array,
    mu_pairs: Array,
    state_encoder: Array,
    action_encoder: Array,
    num_states: int,
    num_actions: int,
    state_decoder: Array,
    action_decoder: Array,
) -> tuple[Array, Array]:
    """Blockwise medoid-style structured decoder update for the lambda-zero solver."""
    num_abstract_states = int(state_encoder.shape[1])
    num_abstract_actions = int(action_decoder.shape[1])
    num_pairs = int(num_states * num_actions)
    mu_pairs_flat = np.asarray(mu_pairs, dtype=float).reshape(num_pairs)

    updated_state_decoder = np.array(state_decoder, dtype=int, copy=True)
    updated_action_decoder = np.array(action_decoder, dtype=int, copy=True)

    for abstract_state in range(num_abstract_states):
        block_objective_by_state = np.full(num_states, np.inf, dtype=float)
        block_actions_by_state = np.zeros((num_states, num_abstract_actions), dtype=int)

        for candidate_state in range(num_states):
            total_block_objective = 0.0
            candidate_actions = np.zeros(num_abstract_actions, dtype=int)
            for abstract_action in range(num_abstract_actions):
                weights = np.zeros(num_pairs, dtype=float)
                for state in range(num_states):
                    state_mass = float(state_encoder[state, abstract_state])
                    if state_mass <= 1e-15:
                        continue
                    state_slice = slice(state * num_actions, (state + 1) * num_actions)
                    weights[state_slice] = (
                        mu_pairs_flat[state_slice]
                        * state_mass
                        * action_encoder[state_slice, abstract_action]
                    )
                if float(np.sum(weights)) <= 1e-15:
                    candidate_actions[abstract_action] = int(abstract_action % num_actions)
                    continue

                best_action = 0
                best_value = np.inf
                for candidate_action in range(num_actions):
                    decoded_pair = int(candidate_state * num_actions + candidate_action)
                    value = float(weights @ distortion[:, decoded_pair])
                    if value < best_value:
                        best_value = value
                        best_action = int(candidate_action)
                candidate_actions[abstract_action] = best_action
                total_block_objective += best_value

            block_objective_by_state[candidate_state] = total_block_objective
            block_actions_by_state[candidate_state, :] = candidate_actions

        chosen_state = int(np.argmin(block_objective_by_state))
        updated_state_decoder[abstract_state] = chosen_state
        updated_action_decoder[abstract_state, :] = block_actions_by_state[chosen_state, :]

    return updated_state_decoder, updated_action_decoder


def _fit_structured_lambda_zero_soft_abstraction(
    distortion: Array,
    mu: Array,
    beta: float,
    num_abstract: int,
    num_actions: int,
    decoder_init: Array | None = None,
    encoder_init: Array | None = None,
    diversify_warm_start: bool = True,
    prune_mass_threshold: float = 1e-4,
    tolerance: float = 1e-10,
    max_outer: int = 50,
    max_inner: int = 1000,
) -> StateActionAbstraction:
    """Run the paper-style structured lambda-zero solver and flatten it for planning."""
    num_concrete = int(distortion.shape[0])
    if num_concrete % int(num_actions) != 0:
        raise ValueError("Structured lambda-zero solver requires num_concrete divisible by num_actions.")
    num_states = int(num_concrete // int(num_actions))
    num_abstract_actions = max(1, min(int(num_actions), int(num_abstract)))
    num_abstract_states = max(1, int(num_abstract) // int(num_abstract_actions))
    total_abstract_pairs = int(num_abstract_states * num_abstract_actions)

    mu_states, mu_action = _build_state_action_marginals(mu, num_states, num_actions)
    mu_pairs = np.asarray(mu, dtype=float).reshape(num_states, num_actions)

    state_encoder = _warm_start_structured_state_encoder(
        encoder_init,
        mu_action,
        num_states,
        num_actions,
        num_abstract_states,
        num_abstract_actions,
    )
    action_encoder = _warm_start_structured_action_encoder(
        encoder_init,
        num_states,
        num_actions,
        num_abstract_states,
        num_abstract_actions,
    )
    state_decoder, action_decoder = _warm_start_structured_decoder(
        decoder_init,
        num_states,
        num_actions,
        num_abstract_states,
        num_abstract_actions,
    )
    if encoder_init is None:
        action_encoder = _initial_action_matching_encoder(
            num_states,
            num_actions,
            num_abstract_actions,
        )
        initial_state_cost = _compute_structured_state_cost(
            distortion,
            mu_action,
            action_encoder,
            state_decoder,
            action_decoder,
            num_states,
            num_actions,
        )
        state_encoder = _initialize_structured_state_encoder_from_cost(
            initial_state_cost,
            mu_states,
            beta,
        )

    if diversify_warm_start and decoder_init is not None:
        state_distance = np.abs(
            np.arange(num_states, dtype=float)[:, None]
            - np.arange(num_states, dtype=float)[None, :]
        )
        state_decoder = diversify_decoder(state_decoder, state_distance)

    for _ in range(max_outer):
        previous_state_decoder = np.array(state_decoder, copy=True)
        previous_action_decoder = np.array(action_decoder, copy=True)
        previous_state_encoder = np.array(state_encoder, copy=True)
        previous_action_encoder = np.array(action_encoder, copy=True)

        action_encoder = _update_lambda_zero_action_encoder(
            distortion,
            state_encoder,
            state_decoder,
            action_decoder,
            num_states,
            num_actions,
        )
        state_encoder = _update_lambda_zero_state_encoder(
            distortion,
            mu_states,
            mu_action,
            state_encoder,
            action_encoder,
            state_decoder,
            action_decoder,
            beta,
            tolerance,
            max_inner,
            num_states,
            num_actions,
        )
        state_decoder, action_decoder = _update_lambda_zero_decoder(
            distortion,
            mu_pairs,
            state_encoder,
            action_encoder,
            num_states,
            num_actions,
            state_decoder,
            action_decoder,
        )
        state_cost = _compute_structured_state_cost(
            distortion,
            mu_action,
            action_encoder,
            state_decoder,
            action_decoder,
            num_states,
            num_actions,
        )
        state_decoder = _diversify_state_decoder_from_state_cost(
            state_decoder,
            state_cost,
        )

        if (
            np.array_equal(previous_state_decoder, state_decoder)
            and np.array_equal(previous_action_decoder, action_decoder)
            and float(np.max(np.abs(previous_state_encoder - state_encoder))) < tolerance
            and float(np.max(np.abs(previous_action_encoder - action_encoder))) < tolerance
        ):
            break

    flat_encoder = _structured_flat_encoder(
        state_encoder,
        action_encoder,
        num_states,
        num_actions,
    )
    flat_decoder = _structured_flat_decoder(
        state_decoder,
        action_decoder,
        num_actions,
    )
    abstract_marginal = _compute_abstract_marginal(mu, flat_encoder)
    encoder_active, posterior_active, decoder_active = _restrict_to_active_abstracts(
        mu,
        flat_encoder,
        flat_decoder,
        abstract_marginal,
        prune_mass_threshold,
    )

    return StateActionAbstraction(
        beta=beta,
        encoder=encoder_active,
        posterior=posterior_active,
        decoder=decoder_active,
        full_encoder=flat_encoder[:, :total_abstract_pairs],
        full_decoder=flat_decoder[:total_abstract_pairs],
        solver_kind="lambda",
        state_encoder=np.asarray(state_encoder, dtype=float),
        action_encoder=np.asarray(action_encoder, dtype=float),
        state_decoder=np.asarray(state_decoder, dtype=int),
        action_decoder=np.asarray(action_decoder, dtype=int),
    )


def fit_soft_abstraction(
    distortion: Array,
    mu: Array,
    beta: float,
    num_abstract: int,
    decoder_init: Array | None = None,
    encoder_init: Array | None = None,
    diversify_warm_start: bool = True,
    prune_mass_threshold: float = 1e-4,
    tolerance: float = 1e-10,
    max_outer: int = 50,
    max_inner: int = 1000,
    *,
    solver_kind: str = "flat",
    num_actions: int | None = None,
) -> StateActionAbstraction:
    """Dispatch to either the flat BA baseline or the structured lambda-zero solver."""
    solver_kind = normalize_solver_kind(solver_kind)
    if solver_kind == "flat":
        return _fit_flat_soft_abstraction(
            distortion=distortion,
            mu=mu,
            beta=beta,
            num_abstract=num_abstract,
            decoder_init=decoder_init,
            encoder_init=encoder_init,
            diversify_warm_start=diversify_warm_start,
            prune_mass_threshold=prune_mass_threshold,
            tolerance=tolerance,
            max_outer=max_outer,
            max_inner=max_inner,
        )
    if solver_kind == "lambda":
        if num_actions is None or int(num_actions) <= 0:
            raise ValueError("Structured lambda-zero solver requires a positive num_actions.")
        return _fit_structured_lambda_zero_soft_abstraction(
            distortion=distortion,
            mu=mu,
            beta=beta,
            num_abstract=num_abstract,
            num_actions=int(num_actions),
            decoder_init=decoder_init,
            encoder_init=encoder_init,
            diversify_warm_start=diversify_warm_start,
            prune_mass_threshold=prune_mass_threshold,
            tolerance=tolerance,
            max_outer=max_outer,
            max_inner=max_inner,
        )
    raise ValueError(f"Unsupported abstraction solver kind: {solver_kind!r}")
