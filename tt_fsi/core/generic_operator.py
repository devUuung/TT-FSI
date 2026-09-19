"""
Generic cardinality-dependent TT operator operator compiler.

All operators act on Mobius coefficients a = mu_down v.
State space tracks (s_k, t_k) with s_k = |S∩[k]| and t_k = |T∩[k]|.
Transitions enforce T ⊇ S with (sigma, tau) in {(0,0), (0,1), (1,1)}.
Only the final weight differs per operator.
"""

import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from scipy.special import gammaln

from .mobius import mobius_mpo
from .matvec_efficient import (
    mpo_matvec_sweep_pruned_by_order_cached,
    mpo_matvec_sweep_pruned_by_order_sparse,
)


@dataclass
class OperatorSpec:
    name: str
    final_weight: Callable[[int, int, int, int], float]
    needs_mobius: bool = True
    needs_correction_only: bool = False


def log_binom(n: int, k: int) -> float:
    if k < 0 or k > n or n < 0:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def weight_fsi_correction(s: int, t: int, d: int, ell: int) -> float:
    if s > ell:
        return 0.0
    if t <= ell:
        return 0.0
    if s <= 0:
        return 0.0
    if t < s:
        return 0.0

    sign = (-1) ** (ell - s)
    scalar = s / (ell + s)
    log_val = (log_binom(ell, s)
               + log_binom(t - 1, ell)
               - log_binom(t + ell - 1, ell + s))
    return sign * scalar * np.exp(log_val)


FSI_SPEC = OperatorSpec(
    name="fsi",
    final_weight=weight_fsi_correction,
    needs_mobius=True,
    needs_correction_only=True,
)


@lru_cache(maxsize=None)
def _cached_fsi_cores(d: int, ell: int):
    return build_generic_sparse_cores(d, ell, FSI_SPEC)


@dataclass(frozen=True)
class SparseCore:
    shape: tuple
    keys: np.ndarray
    coefficients: np.ndarray

    @property
    def nbytes(self):
        return self.keys.nbytes + self.coefficients.nbytes

    def transitions(self):
        for key, coefficient in zip(self.keys, self.coefficients):
            yield (*map(int, key), coefficient)

    def to_dense(self):
        core = np.zeros(self.shape, dtype=np.float32)
        if len(self.keys):
            core[tuple(self.keys.T)] = self.coefficients
        return core


def D_k(k: int, ell: int) -> int:
    smax = min(k, ell)
    return (smax + 1) * (k + 1) - smax * (smax + 1) // 2


def idx_state(k: int, s: int, t: int, ell: int) -> int:
    if not 0 <= s <= min(k, ell) or not s <= t <= k:
        raise ValueError('Invalid FSM state')
    return s * (k + 1) - s * (s - 1) // 2 + t - s


def state_from_idx(k: int, alpha: int, ell: int):
    for s in range(min(k, ell) + 1):
        width = k - s + 1
        if alpha < width:
            return s, s + alpha
        alpha -= width
    raise ValueError('Invalid FSM state index')


def _sparse_core(shape, entries):
    items = [(key, np.float32(value)) for key, value in sorted(entries.items()) if value != 0]
    return SparseCore(shape, np.asarray([key for key, _ in items], dtype=np.int32).reshape(-1, 4),
                      np.asarray([value for _, value in items], dtype=np.float32))


def build_generic_sparse_cores(d: int, ell: int, spec: OperatorSpec, precontract=False):
    from tt_fsi._validation import validate_order
    validate_order(ell, d)
    cores = []
    for k in range(d):
        shape = (D_k(k, ell), 2, 2, 1 if k == d-1 else D_k(k+1, ell))
        entries = {}
        for s in range(min(k, ell) + 1):
            for t in range(s, k + 1):
                source = idx_state(k, s, t, ell)
                for sigma, tau in ((0, 0), (0, 1), (1, 1)):
                    sn, tn = s + sigma, t + tau
                    if sn > ell:
                        continue
                    last = k == d - 1
                    target = 0 if last else idx_state(k+1, sn, tn, ell)
                    coefficient = np.float32(spec.final_weight(sn, tn, d, ell) if last else 1)
                    inputs = ((0, -1), (1, 1)) if precontract and spec.needs_mobius and tau == 1 else ((tau, 1),)
                    for rho, sign in inputs:
                        key = (source, sigma, rho, target)
                        entries[key] = np.float32(entries.get(key, np.float32(0)) + sign * coefficient)
        cores.append(_sparse_core(shape, entries))
    return tuple(cores)


def build_generic_mpo_cores(d: int, ell: int, spec: OperatorSpec):
    return [core.to_dense() for core in build_generic_sparse_cores(d, ell, spec)]


def build_generic_precontract_cores(d: int, ell: int, spec: OperatorSpec):
    return [core.to_dense() for core in build_generic_sparse_cores(d, ell, spec, precontract=True)]


@lru_cache(maxsize=None)
def _order_mask(d: int, ell: int):
    n = 1 << d
    pop = np.fromiter((i.bit_count() for i in range(n)), dtype=np.int16, count=n)
    return (pop <= ell).reshape((2,) * d)


def mask_by_order(tensor, ell: int):
    return np.where(_order_mask(tensor.ndim, ell), tensor, np.zeros((), dtype=tensor.dtype))


def compute_interaction_index(v_tensor, ell: int, index_type: str = "fsi"):
    """Dense FSI over the flat bitmask order; feature i is mask bit i."""
    if index_type.lower() != "fsi":
        raise ValueError(f"Unsupported index_type: {index_type}")
    from tt_fsi._validation import tensor_input
    v_tensor = tensor_input(v_tensor, ell)
    d = v_tensor.ndim
    a_tensor = mobius_mpo(v_tensor)
    correction = mpo_matvec_sweep_pruned_by_order_cached(_cached_fsi_cores(d, ell), a_tensor, ell)
    return mask_by_order(a_tensor + correction, ell)


def compute_fsi_sparse(v_tensor, ell: int):
    """Return FSI values only for subsets with order <= ell.

    Returns a dictionary with flat subset indices, values, and the dense tensor
    shape.  Use this when downstream code does not need a full 2^d tensor.
    """
    from tt_fsi._validation import tensor_input
    v_tensor = tensor_input(v_tensor, ell)
    d = v_tensor.ndim
    a_tensor = mobius_mpo(v_tensor)
    indices, correction = mpo_matvec_sweep_pruned_by_order_sparse(_cached_fsi_cores(d, ell), a_tensor, ell)
    values = a_tensor.reshape(-1)[indices] + correction
    return {"indices": indices, "values": values, "shape": v_tensor.shape}


def sparse_fsi_to_dense(sparse_result, dtype=None):
    shape = tuple(sparse_result["shape"])
    values = sparse_result["values"]
    out = np.zeros(np.prod(shape, dtype=np.int64), dtype=values.dtype if dtype is None else dtype)
    out[sparse_result["indices"]] = values
    return out.reshape(shape)
