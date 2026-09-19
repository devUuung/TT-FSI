"""
Naive (brute-force) FSI implementation.

FSI closed form (Theorem 19, Tsai et al. JMLR 2023):
For |S| ≤ ℓ:
E^{F-Shap}_S(v, ℓ) = a(v, S) + correction(S)

where:
correction(S) = (-1)^{ℓ-|S|} × (|S|/(ℓ+|S|)) × C(ℓ,|S|) ×
                Σ_{T⊃S, |T|>ℓ} [C(|T|-1,ℓ)/C(|T|+ℓ-1,ℓ+|S|)] × a(v,T)

This simplifies to:
correction(S) = Σ_{T⊃S, |T|>ℓ} w(|S|, |T|) × a(T)

where w(s, t) is the weight function.
"""

import numpy as np
from tt_fsi._validation import flat_input, tensor_input
from typing import List

from .mobius_naive import mobius_naive, popcount, is_subset


def weight_w_naive(s: int, t: int, ell: int) -> float:
    """
    Compute the correction weight w(s, t, ℓ) directly.

    Uses scipy.special.comb for comparison (less stable for large values).
    """
    from scipy.special import comb

    if s > ell or t <= ell or s <= 0 or t < s:
        return 0.0

    sign = (-1) ** (ell - s)
    scalar = s / (ell + s)
    binom_val = comb(ell, s, exact=False) * comb(t - 1, ell, exact=False) / comb(t + ell - 1, ell + s, exact=False)

    return sign * scalar * binom_val


def correction_naive(a: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute the correction term for FSI using brute force.

    For each S with |S| ≤ ℓ:
    correction(S) = Σ_{T⊃S, |T|>ℓ} w(|S|, |T|) × a(T)

    Args:
        a: Möbius coefficients, shape (2^d,).
        ell: Interaction order bound.

    Returns:
        Correction values, shape (2^d,).
    """
    a = flat_input(a, ell)
    n = len(a)
    d = int(np.log2(n))
    assert n == 2**d, "a must have length 2^d"

    correction = np.zeros(n, dtype=a.dtype)

    for S_idx in range(n):
        s = popcount(S_idx)

        if s > ell:
            # Only compute for |S| ≤ ℓ
            continue

        total = np.float32(0.0)
        # Iterate over all T that are supersets of S with |T| > ℓ
        for T_idx in range(n):
            if not is_superset(T_idx, S_idx):
                continue
            if T_idx == S_idx:
                # T must be strict superset for correction (T ⊃ S, not T = S)
                # Actually, the formula includes T ⊇ S, but weight is 0 when |T| ≤ ℓ
                # So we don't need to skip T = S explicitly
                pass

            t = popcount(T_idx)
            if t <= ell:
                continue

            w = weight_w_naive(s, t, ell)
            total += np.float32(w) * a[T_idx]

        correction[S_idx] = total

    return correction


def is_superset(T_idx: int, S_idx: int) -> bool:
    """Check if T ⊇ S using bitwise operations."""
    return (T_idx & S_idx) == S_idx


def fsi_naive(v: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute FSI values using brute force.

    E^{F-Shap}_S(v, ℓ) = a(S) + correction(S) for |S| ≤ ℓ
                       = 0 for |S| > ℓ (by definition)

    Args:
        v: Set function values, shape (2^d,).
        ell: Interaction order bound.

    Returns:
        FSI values, shape (2^d,). Values for |S| > ℓ are set to 0.
    """
    v = flat_input(v, ell)
    # Compute Möbius transform
    a = mobius_naive(v)

    # Compute correction term
    corr = correction_naive(a, ell)

    # FSI = a + correction for |S| ≤ ℓ, else 0
    n = len(v)
    d = int(np.log2(n))
    fsi = np.zeros(n, dtype=v.dtype)

    for S_idx in range(n):
        s = popcount(S_idx)
        if s <= ell:
            fsi[S_idx] = a[S_idx] + corr[S_idx]
        # else: fsi[S_idx] = 0 (already initialized)

    return fsi


def correction_naive_tensor(a_tensor: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute correction term on tensor representation.

    Args:
        a_tensor: Möbius coefficients, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        Correction tensor of same shape.
    """
    a_tensor = tensor_input(a_tensor, ell)
    a_flat = a_tensor.flatten()
    corr_flat = correction_naive(a_flat, ell)
    return corr_flat.reshape(a_tensor.shape)


def fsi_naive_tensor(v_tensor: np.ndarray, ell: int) -> np.ndarray:
    """
    Compute FSI on tensor representation.

    Args:
        v_tensor: Set function values, shape (2, 2, ..., 2).
        ell: Interaction order bound.

    Returns:
        FSI tensor of same shape.
    """
    v_tensor = tensor_input(v_tensor, ell)
    v_flat = v_tensor.flatten()
    fsi_flat = fsi_naive(v_flat, ell)
    return fsi_flat.reshape(v_tensor.shape)
