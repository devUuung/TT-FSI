"""
A_trunc TT operator for FSI correction term.

The correction term computes:
C(S) = Σ_{T⊃S, |T|>ℓ} w(|S|, |T|) × a(T)

where a(T) is the Möbius transform and w(s,t) is the weight function.

State space: (s_k, t_k) where
- s_k = prefix sum of |S| up to position k
- t_k = prefix sum of |T| up to position k

Truncation: s_k ∈ {0, ..., min(k, ℓ)} since we only need |S| ≤ ℓ.

Bond dimension: D_k = (min(k, ℓ) + 1) × (k + 1)
"""

import numpy as np
from typing import List, Tuple
from .weight import weight_w


from .generic_operator import D_k, idx_state, state_from_idx, FSI_SPEC, build_generic_mpo_cores, build_generic_precontract_cores


def build_correction_mpo_cores(d: int, ell: int) -> List[np.ndarray]:
    return build_generic_mpo_cores(d, ell, FSI_SPEC)


def mpo_matvec_general(cores: List[np.ndarray], v_tensor: np.ndarray) -> np.ndarray:
    """
    Apply general TT operator (with rank > 1) to a tensor.

    Contracts from left to right using intermediate tensors.

    Args:
        cores: List of d TT operator cores, each of shape [r_l, 2, 2, r_r].
        v_tensor: Input tensor of shape (2, 2, ..., 2).

    Returns:
        Output tensor of shape (2, 2, ..., 2).
    """
    d = len(cores)
    assert v_tensor.ndim == d

    # Result tensor
    result = np.zeros_like(v_tensor)

    # Iterate over all output configurations
    for out_idx in np.ndindex(*([2] * d)):
        # For each output configuration, compute the sum over inputs
        total = 0.0

        for in_idx in np.ndindex(*([2] * d)):
            # Compute the TT operator path weight
            # Contract: Σ_{α_0,...,α_d} Π_k G_k[α_{k-1}, σ_k, τ_k, α_k]
            # For this, we trace through the bond dimensions

            # Left boundary: α_0 corresponds to index 0 in first core
            path_weight = np.zeros((1,), dtype=np.float32)
            path_weight[0] = 1.0

            valid = True
            for k in range(d):
                sigma = out_idx[k]
                tau = in_idx[k]
                core = cores[k]

                # Contract: new_path[α_k] = Σ_{α_{k-1}} path[α_{k-1}] × G[α_{k-1}, σ, τ, α_k]
                new_path = np.einsum('a,abcd->d', path_weight,
                                     core[:, sigma:sigma+1, tau:tau+1, :])
                new_path = new_path.flatten()

                if np.all(new_path == 0):
                    valid = False
                    break

                path_weight = new_path

            if valid and len(path_weight) == 1:
                total += path_weight[0] * v_tensor[in_idx]

        result[out_idx] = total

    return result


def correction_mpo(a_tensor: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute correction term C = A_trunc × a using TT operator.

    Args:
        a_tensor: Möbius coefficients, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        Correction tensor, same shape.
    """
    d = a_tensor.ndim
    cores = build_correction_mpo_cores(d, ell)
    return mpo_matvec_general(cores, a_tensor)


def build_precontract_mpo_cores(d: int, ell: int) -> List[np.ndarray]:
    return build_generic_precontract_cores(d, ell, FSI_SPEC)


def correction_precontract_mpo(v_tensor: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute correction term C = A_trunc × μ_↓ × v using precontracted TT operator.

    This is equivalent to:
        a = μ_↓ × v  (Möbius transform)
        C = A_trunc × a  (correction)

    But more efficient as it combines the operations into a single TT operator application.

    Args:
        v_tensor: Set function values, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        Correction tensor, same shape.
    """
    d = v_tensor.ndim
    cores = build_precontract_mpo_cores(d, ell)
    return mpo_matvec_general(cores, v_tensor)
