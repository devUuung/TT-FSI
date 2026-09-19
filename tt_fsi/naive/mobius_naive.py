"""
Naive (brute-force) Möbius transform implementation.

μ_↓ is the subset-direction Möbius transform:
(μ_↓)_{S,T} = (-1)^{|S|-|T|} × 1[T ⊆ S]

For a function v: 2^[d] → R, the Möbius transform a = μ_↓ v is:
a(S) = Σ_{T ⊆ S} (-1)^{|S|-|T|} v(T)
"""

import numpy as np
from tt_fsi._validation import flat_input, tensor_input
from typing import List


def subset_to_index(subset: List[int], d: int) -> int:
    """Convert subset (list of 0-indexed elements) to binary index."""
    idx = 0
    for elem in subset:
        idx |= (1 << elem)
    return idx


def index_to_subset(idx: int, d: int) -> List[int]:
    """Convert binary index to subset (list of 0-indexed elements)."""
    subset = []
    for i in range(d):
        if idx & (1 << i):
            subset.append(i)
    return subset


def is_subset(T_idx: int, S_idx: int) -> bool:
    """Check if T ⊆ S using bitwise operations."""
    return (T_idx & S_idx) == T_idx


def popcount(x: int) -> int:
    """Count number of 1 bits."""
    return bin(x).count('1')


def mobius_naive(v: np.ndarray) -> np.ndarray:
    """
    Compute Möbius transform a = μ_↓ v using brute force.

    Args:
        v: Array of shape (2^d,) representing set function values.
           v[idx] = v(S) where S is the subset with binary representation idx.

    Returns:
        a: Array of shape (2^d,) with Möbius coefficients.
           a[idx] = Σ_{T ⊆ S} (-1)^{|S|-|T|} v(T)
    """
    v = flat_input(v)
    n = len(v)
    d = int(np.log2(n))
    assert n == 2**d, "v must have length 2^d"

    a = np.zeros(n, dtype=v.dtype)

    for S_idx in range(n):
        S_size = popcount(S_idx)
        total = np.float32(0.0)
        # Iterate over all subsets T of S
        T_idx = S_idx
        while True:
            T_size = popcount(T_idx)
            sign = (-1) ** (S_size - T_size)
            total += sign * v[T_idx]
            if T_idx == 0:
                break
            # Next subset of S: T = (T - 1) & S
            T_idx = (T_idx - 1) & S_idx
        a[S_idx] = total

    return a


def mobius_naive_tensor(v_tensor: np.ndarray) -> np.ndarray:
    """
    Compute Möbius transform on tensor representation.

    Args:
        v_tensor: Array of shape (2, 2, ..., 2) with d dimensions.
                  v_tensor[σ_1, σ_2, ..., σ_d] = v(S) where S = {i : σ_i = 1}

    Returns:
        a_tensor: Array of same shape with Möbius coefficients.
    """
    v_tensor = tensor_input(v_tensor)
    v_flat = v_tensor.flatten()
    a_flat = mobius_naive(v_flat)
    return a_flat.reshape(v_tensor.shape)
