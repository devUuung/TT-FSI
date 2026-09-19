"""
Efficient FSI computation using optimized TT contraction.

Uses sequential left-to-right pass for O(D^2 × 2^d) complexity instead of O(4^d).
"""

import numpy as np
from typing import Tuple

from .mobius import mobius_mpo
from .correction import build_correction_mpo_cores, build_precontract_mpo_cores
from .matvec_efficient import mpo_matvec_sweep


def correction_mpo_efficient(a_tensor: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute correction term C = A_trunc × a using efficient TT contraction.

    Args:
        a_tensor: Möbius coefficients, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        Correction tensor, same shape.
    """
    from tt_fsi._validation import tensor_input
    a_tensor = tensor_input(a_tensor, ell)
    d = a_tensor.ndim
    cores = build_correction_mpo_cores(d, ell)
    return mpo_matvec_sweep(cores, a_tensor)


def correction_precontract_efficient(v_tensor: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute correction term C = A_trunc × μ_↓ × v using precontracted TT operator.

    More efficient than separate Möbius + correction for large d.

    Args:
        v_tensor: Set function values, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        Correction tensor, same shape.
    """
    from tt_fsi._validation import tensor_input
    v_tensor = tensor_input(v_tensor, ell)
    d = v_tensor.ndim
    cores = build_precontract_mpo_cores(d, ell)
    return mpo_matvec_sweep(cores, v_tensor)


def fsi_tt_efficient(v_tensor: np.ndarray, ell: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FSI using efficient TT pipeline.

    Pipeline:
    1. Compute Möbius transform: a = μ_↓ v
    2. Compute correction: C = A_trunc × a (sequential core multiplication)
    3. Combine: FSI = a + C for |S| ≤ ℓ, else 0

    Args:
        v_tensor: Set function values, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        Tuple of (fsi_tensor, a_tensor, correction_tensor)
    """
    from tt_fsi._validation import tensor_input
    v_tensor = tensor_input(v_tensor, ell)
    d = v_tensor.ndim

    # Step 1: Möbius transform (already efficient for rank-1)
    a_tensor = mobius_mpo(v_tensor)

    # Step 2: Correction term (sequential core multiplication)
    correction_tensor = correction_mpo_efficient(a_tensor, ell)

    # Step 3: Combine (with masking for |S| > ℓ)
    fsi_tensor = np.zeros_like(v_tensor)

    for idx in np.ndindex(*([2] * d)):
        s = sum(idx)  # |S| = number of 1s in index
        if s <= ell:
            fsi_tensor[idx] = a_tensor[idx] + correction_tensor[idx]
        # else: fsi_tensor[idx] = 0 (already initialized)

    return fsi_tensor, a_tensor, correction_tensor


def fsi_tt_efficient_simple(v_tensor: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute FSI using efficient TT pipeline (simple interface).

    Args:
        v_tensor: Set function values, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        FSI tensor, same shape as input.
    """
    fsi_tensor, _, _ = fsi_tt_efficient(v_tensor, ell)
    return fsi_tensor


def fsi_precontract_efficient(v_tensor: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute FSI using precontracted B = A_trunc ∘ μ_↓.

    This is the most efficient approach as it:
    1. Combines Möbius and correction into single TT operator
    2. Uses efficient sequential left-to-right pass

    Args:
        v_tensor: Set function values, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        FSI tensor, same shape as input.
    """
    from tt_fsi._validation import tensor_input
    v_tensor = tensor_input(v_tensor, ell)
    d = v_tensor.ndim

    # Compute a = μ_↓ v (efficient for rank-1)
    a_tensor = mobius_mpo(v_tensor)

    # Compute correction using precontracted TT operator
    correction_tensor = correction_precontract_efficient(v_tensor, ell)

    # Combine with masking
    fsi_tensor = np.zeros_like(v_tensor)

    for idx in np.ndindex(*([2] * d)):
        s = sum(idx)
        if s <= ell:
            fsi_tensor[idx] = a_tensor[idx] + correction_tensor[idx]

    return fsi_tensor
