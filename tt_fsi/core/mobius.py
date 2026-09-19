"""
TT-based Möbius transform implementation.

μ_↓ has Kronecker product structure:
μ_↓ = ⊗_{i=1}^{d} M_i, where M_i = [[1, 0], [-1, 1]]

This gives TT-rank 1, with each core having shape [1, 2, 2, 1].
"""

import numpy as np
from typing import List


def build_mobius_mpo_cores(d: int) -> List[np.ndarray]:
    """
    Build the TT operator cores for μ_↓.

    Each core has shape [r_left, phys_out, phys_in, r_right] = [1, 2, 2, 1]
    where:
      - phys_out (σ): output physical index (0 or 1, indicates if element is in S)
      - phys_in (τ): input physical index (0 or 1, indicates if element is in T)

    The local matrix M = [[1, 0], [-1, 1]] encodes:
      M[σ, τ] = (-1)^{σ-τ} if τ ≤ σ else 0

    This means: for each element i,
      - If σ=0, τ must be 0 (element not in S, so not in T)
      - If σ=1, τ can be 0 or 1 (element in S, T can have it or not)

    Returns:
        List of d cores, each of shape [1, 2, 2, 1].
    """
    # Local Möbius matrix
    M = np.array([[1.0, 0.0],
                  [-1.0, 1.0]], dtype=np.float32)

    cores = []
    for k in range(d):
        # Shape: [r_left, phys_out, phys_in, r_right] = [1, 2, 2, 1]
        core = np.zeros((1, 2, 2, 1), dtype=np.float32)
        for sigma in range(2):
            for tau in range(2):
                core[0, sigma, tau, 0] = M[sigma, tau]
        cores.append(core)

    return cores


def mpo_matvec(cores: List[np.ndarray], v_tensor: np.ndarray) -> np.ndarray:
    """
    Apply TT operator to a tensor (TT-vector with rank 1).

    This computes a = M × v where M is given by TT operator cores
    and v is given as a tensor of shape (2, 2, ..., 2).

    The computation contracts from left to right:
        result = Σ_τ Π_k G_k[α_{k-1}, σ_k, τ_k, α_k] × v[τ_1, ..., τ_d]

    For rank-1 TT operator and rank-1 input, this simplifies to element-wise:
        a[σ_1, ..., σ_d] = Π_k M[σ_k, τ_k] × v[τ_1, ..., τ_d]
                         = Σ_{τ_1,...,τ_d} Π_k M_k[σ_k, τ_k] × v[τ]

    Args:
        cores: List of d TT operator cores, each of shape [r_l, 2, 2, r_r].
        v_tensor: Input tensor of shape (2, 2, ..., 2).

    Returns:
        Output tensor of shape (2, 2, ..., 2).
    """
    d = len(cores)
    assert v_tensor.ndim == d

    # For rank-1 TT operator, we can compute this efficiently using einsum
    # The TT operator is essentially M = M_1 ⊗ M_2 ⊗ ... ⊗ M_d
    # So (M × v)[σ_1,...,σ_d] = Σ_{τ_1,...,τ_d} Π_k M_k[σ_k, τ_k] × v[τ]
    #                        = Π_k (Σ_{τ_k} M_k[σ_k, τ_k] × ...) (via separability)

    # Extract local matrices (squeeze out bond dimensions for rank-1)
    local_mats = []
    for core in cores:
        # Shape [1, 2, 2, 1] -> [2, 2]
        local_mats.append(core[0, :, :, 0])

    # Apply each local matrix along its dimension
    result = v_tensor.copy()
    for k in range(d):
        # Contract dimension k with local_mats[k]
        # result[..., σ_k, ...] = Σ_{τ_k} M[σ_k, τ_k] × result[..., τ_k, ...]
        result = np.tensordot(local_mats[k], result, axes=([1], [k]))
        # tensordot moves the contracted axis to position 0, need to move it back
        result = np.moveaxis(result, 0, k)

    return result


def mobius_mpo(v_tensor: np.ndarray) -> np.ndarray:
    """
    Compute Möbius transform a = μ_↓ v using TT operator representation.

    Args:
        v_tensor: Array of shape (2, 2, ..., 2) with d dimensions.

    Returns:
        a_tensor: Array of same shape with Möbius coefficients.
    """
    from tt_fsi._validation import tensor_input
    v_tensor = tensor_input(v_tensor)
    d = v_tensor.ndim
    cores = build_mobius_mpo_cores(d)
    return mpo_matvec(cores, v_tensor)
