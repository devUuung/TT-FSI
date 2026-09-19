"""Shared public input contract for FP32 operators."""
from numbers import Integral
import numpy as np


def validate_order(ell, d):
    if isinstance(ell, (bool, np.bool_)) or not isinstance(ell, Integral) or not 1 <= ell <= d:
        raise ValueError('ell must be an integer in [1, d]')
    return int(ell)


def validate_chunk(chunk_size):
    if chunk_size is not None and (isinstance(chunk_size, (bool, np.bool_)) or
            not isinstance(chunk_size, Integral) or chunk_size < 1):
        raise ValueError('chunk_size must be a positive integer')


def fp32_dtype(dtype):
    if dtype is not None and np.dtype(dtype) != np.dtype('float32'):
        raise ValueError('Only float32 is supported')
    return np.dtype('float32')


def real_array(values, xp=np):
    arr = xp.asarray(values)
    if arr.dtype.kind not in 'fiu':
        raise ValueError('values must contain real floating-point or integer numbers')
    with np.errstate(over='ignore', invalid='ignore'):
        arr = xp.ascontiguousarray(arr, dtype=xp.float32)
    if not bool(xp.isfinite(arr).all()):
        raise ValueError('values must be finite and representable in float32')
    return arr


def flat_input(values, ell=None):
    arr = real_array(values)
    if arr.ndim != 1 or arr.size < 2 or arr.size & (arr.size - 1):
        raise ValueError('values must have shape (2**d,), d >= 1')
    if ell is not None:
        validate_order(ell, arr.size.bit_length() - 1)
    return arr


def tensor_input(values, ell=None):
    arr = real_array(values)
    if arr.ndim < 1 or any(n != 2 for n in arr.shape):
        raise ValueError('values must have tensor shape (2,) * d, d >= 1')
    if ell is not None:
        validate_order(ell, arr.ndim)
    return arr
