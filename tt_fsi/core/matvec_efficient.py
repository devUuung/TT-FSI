"""
Efficient TT-operator-vector multiplication via sequential left-to-right pass.

Standard complexity: O(D^2 × 2^d) where D is max bond dimension.
This is much faster than naive O(4^d) iteration over all pairs.
"""

import numpy as np
from typing import List


_PRUNED_PLAN_CACHE = {}


def core_transitions(core):
    if hasattr(core, 'transitions'):
        yield from core.transitions()
    else:
        for alpha, sigma, tau, beta in np.argwhere(core != 0):
            yield int(alpha), int(sigma), int(tau), int(beta), core[alpha, sigma, tau, beta]



def mpo_matvec_sweep(cores: List[np.ndarray], v_tensor: np.ndarray) -> np.ndarray:
    """
    Efficient left-to-right sequential TT-operator-vector multiplication.

    We maintain an intermediate tensor that accumulates processed dimensions.
    At step k, we have:
        I[α_k, σ_1:k, τ_{k+1}:d] = Σ_{τ_1:k, α_0:k-1} Π_j G_j[...] × v[τ]

    Complexity: O(D_max^2 × 2^d) where D_max is maximum bond dimension.

    Args:
        cores: List of d TT operator cores, each of shape [D_left, 2, 2, D_right].
        v_tensor: Input tensor of shape (2, 2, ..., 2).

    Returns:
        Output tensor of shape (2, 2, ..., 2).
    """
    d = len(cores)
    if d == 0:
        return v_tensor.copy()

    # Initial shape: (1, τ_1, τ_2, ..., τ_d) = (1,) + v.shape
    intermediate = v_tensor.reshape((1,) + v_tensor.shape)

    for k in range(d):
        core = cores[k]  # (D_left, σ, τ, D_right)
        D_left, _, _, D_right = core.shape

        # At step k=0: shape is (1, τ_1, τ_2, ..., τ_d) = (1, 2, 2, ..., 2) [1+d dims]
        #   0 output dims, d input dims

        n_out_dims = k  # number of σ dimensions we've produced
        n_in_dims = d - k  # number of τ dimensions remaining

        # Reshape to (D_left, 2^n_out, 2, 2^{n_in-1})
        # = (D_left, out_block, τ_k, in_remaining)
        n_out = 2 ** n_out_dims if n_out_dims > 0 else 1
        n_in_remaining = 2 ** (n_in_dims - 1) if n_in_dims > 1 else 1

        inter_reshaped = intermediate.reshape(D_left, n_out, 2, n_in_remaining)
        # inter_reshaped[α, out, τ_k, in_rest]

        # Contract with core[α, σ_k, τ_k, β]
        # Result: Σ_{α, τ_k} inter[α, out, τ_k, in_rest] × core[α, σ_k, τ_k, β]
        # = new[out, σ_k, β, in_rest]

        new_inter = np.einsum('aotr,astb->osbr', inter_reshaped, core, optimize='optimal')
        # Shape: (n_out, 2, D_right, n_in_remaining)

        # Reshape to standard form: (D_right, 2^{n_out+1}, 2^{n_in_dims-1})
        new_n_out = n_out * 2  # we added one σ dimension
        new_inter = new_inter.transpose(2, 0, 1, 3)  # (D_right, n_out, 2, n_in_remaining)
        new_inter = new_inter.reshape(D_right, new_n_out, n_in_remaining)

        # Reshape to tensor form
        new_n_out_dims = n_out_dims + 1
        new_n_in_dims = n_in_dims - 1
        new_shape = (D_right,) + (2,) * new_n_out_dims + (2,) * new_n_in_dims
        intermediate = new_inter.reshape(new_shape)

    # Final: (1, σ_1, ..., σ_d) -> squeeze first dim
    result = intermediate.squeeze(0)
    return result


def mpo_matvec_sweep_pruned_by_order(
    cores: List[np.ndarray], v_tensor: np.ndarray, ell: int
) -> np.ndarray:
    """Apply an TT operator while keeping only output prefixes with popcount <= ell.

    FSI only needs outputs up to order ell.  The generic sweep materializes all
    2^k output prefixes at step k and masks at the end, which is wasteful for
    small ell.  This variant keeps the same left-to-right contraction but prunes
    impossible output prefixes during the sweep.
    """
    d = len(cores)
    if d == 0:
        return v_tensor.copy()

    prefixes = np.array([0], dtype=np.int64)
    counts = np.array([0], dtype=np.int16)
    intermediate = v_tensor.reshape(1, 1, 1 << d)

    for k, core in enumerate(cores):
        d_left, _, _, d_right = core.shape
        n_remaining = 1 << (d - k - 1)
        current = intermediate.reshape(d_left, len(prefixes), 2, n_remaining)

        next_prefix_parts = []
        next_count_parts = []
        next_indices = []
        for sigma in (0, 1):
            valid = counts + sigma <= ell
            idx = np.nonzero(valid)[0]
            next_indices.append(idx)
            next_prefix_parts.append(prefixes[idx] * 2 + sigma)
            next_count_parts.append(counts[idx] + sigma)

        next_prefixes = np.concatenate(next_prefix_parts)
        next_counts = np.concatenate(next_count_parts)
        order = np.argsort(next_prefixes)
        next_prefixes = next_prefixes[order]
        next_counts = next_counts[order]
        prefix_to_pos = {int(prefix): pos for pos, prefix in enumerate(next_prefixes)}
        sigma_targets = []
        for sigma, idx in enumerate(next_indices):
            target = np.fromiter(
                (prefix_to_pos[int(prefixes[j] * 2 + sigma)] for j in idx),
                dtype=np.int64,
                count=len(idx),
            )
            sigma_targets.append((idx, target))

        new_intermediate = np.zeros((d_right, len(next_prefixes), n_remaining), dtype=v_tensor.dtype)
        for alpha, sigma, tau, beta, coefficient in core_transitions(core):
            source_idx, target_idx = sigma_targets[int(sigma)]
            if len(source_idx) == 0:
                continue
            new_intermediate[beta, target_idx, :] += (
                coefficient * current[alpha, source_idx, tau, :]
            )

        prefixes = next_prefixes
        counts = next_counts
        intermediate = new_intermediate

    flat = np.zeros(1 << d, dtype=v_tensor.dtype)
    flat[prefixes] = intermediate.reshape(-1)
    return flat.reshape(v_tensor.shape)


def _build_pruned_plan(cores, ell: int):
    d = len(cores)
    prefixes = np.array([0], dtype=np.int64)
    counts = np.array([0], dtype=np.int16)
    steps = []

    for k, core in enumerate(cores):
        _, _, _, d_right = core.shape
        n_remaining = 1 << (d - k - 1)

        next_prefix_parts = []
        next_count_parts = []
        next_indices = []
        for sigma in (0, 1):
            valid = counts + sigma <= ell
            idx = np.nonzero(valid)[0]
            next_indices.append(idx)
            next_prefix_parts.append(prefixes[idx] * 2 + sigma)
            next_count_parts.append(counts[idx] + sigma)

        next_prefixes = np.concatenate(next_prefix_parts)
        next_counts = np.concatenate(next_count_parts)
        order = np.argsort(next_prefixes)
        next_prefixes = next_prefixes[order]
        next_counts = next_counts[order]
        prefix_to_pos = {int(prefix): pos for pos, prefix in enumerate(next_prefixes)}
        sigma_targets = []
        for sigma, idx in enumerate(next_indices):
            target = np.fromiter(
                (prefix_to_pos[int(prefixes[j] * 2 + sigma)] for j in idx),
                dtype=np.int64,
                count=len(idx),
            )
            sigma_targets.append((idx, target))

        transitions = []
        for alpha, sigma, tau, beta, coefficient in core_transitions(core):
            source_idx, target_idx = sigma_targets[int(sigma)]
            if len(source_idx) == 0:
                continue
            transitions.append(
                (
                    int(alpha),
                    int(tau),
                    int(beta),
                    np.float32(coefficient),
                    source_idx,
                    target_idx,
                )
            )

        steps.append((d_right, n_remaining, len(prefixes), len(next_prefixes), transitions))
        prefixes = next_prefixes
        counts = next_counts

    return tuple(steps), prefixes


def _get_pruned_plan(cores, ell: int):
    key = (id(cores), ell)
    plan = _PRUNED_PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_pruned_plan(cores, ell)
        _PRUNED_PLAN_CACHE[key] = (cores, plan)
        return plan
    return plan[1]


def mpo_matvec_sweep_pruned_by_order_cached(
    cores: List[np.ndarray], v_tensor: np.ndarray, ell: int
) -> np.ndarray:
    d = len(cores)
    if d == 0:
        return v_tensor.copy()
    steps, prefixes = _get_pruned_plan(cores, ell)
    intermediate = v_tensor.reshape(1, 1, 1 << d)

    for d_right, n_remaining, n_prefixes, n_next_prefixes, transitions in steps:
        current = intermediate.reshape(-1, n_prefixes, 2, n_remaining)
        new_intermediate = np.zeros((d_right, n_next_prefixes, n_remaining), dtype=v_tensor.dtype)
        for alpha, tau, beta, coeff, source_idx, target_idx in transitions:
            new_intermediate[beta, target_idx, :] += coeff * current[alpha, source_idx, tau, :]
        intermediate = new_intermediate

    flat = np.zeros(1 << d, dtype=v_tensor.dtype)
    flat[prefixes] = intermediate.reshape(-1)
    return flat.reshape(v_tensor.shape)


def mpo_matvec_sweep_pruned_by_order_sparse(
    cores: List[np.ndarray], v_tensor: np.ndarray, ell: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return only valid output flat indices and values."""
    d = len(cores)
    if d == 0:
        return np.array([0], dtype=np.int64), v_tensor.reshape(1).copy()
    steps, prefixes = _get_pruned_plan(cores, ell)
    intermediate = v_tensor.reshape(1, 1, 1 << d)

    for d_right, n_remaining, n_prefixes, n_next_prefixes, transitions in steps:
        current = intermediate.reshape(-1, n_prefixes, 2, n_remaining)
        new_intermediate = np.zeros((d_right, n_next_prefixes, n_remaining), dtype=v_tensor.dtype)
        for alpha, tau, beta, coeff, source_idx, target_idx in transitions:
            new_intermediate[beta, target_idx, :] += coeff * current[alpha, source_idx, tau, :]
        intermediate = new_intermediate

    return prefixes.copy(), intermediate.reshape(-1)
