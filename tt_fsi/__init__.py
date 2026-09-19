"""Input-validated CPU API for bounded and dense FSI computation."""
from numbers import Integral
import numpy as np
from tt_fsi.core.generic_operator import compute_fsi_sparse, sparse_fsi_to_dense
from tt_fsi.core.fsi_efficient import fsi_tt_efficient_simple


def _validate(values, ell):
    from tt_fsi._validation import flat_input
    arr = flat_input(values, ell)
    return arr.reshape((2,) * (arr.size.bit_length()-1))


def fsi(values, ell, *, method="bounded", sparse=False):
    """FP32 FSI, using flat bitmask order; feature i corresponds to mask bit i.

    bounded uses CPU prefix-pruned correction with a dense Mobius intermediate.
    dense uses the full-prefix TT reference. Both are CPU implementations;
    the fused CUDA schedule is provided separately in tt_fsi.core.gpu_tt.
    """
    tensor = _validate(values, ell)
    if method == "bounded":
        result = compute_fsi_sparse(tensor, ell)
        if sparse:
            return {"indices": result["indices"], "values": result["values"],
                    "shape": (tensor.size,)}
        return sparse_fsi_to_dense(result).reshape(-1)
    if method == "dense" and not sparse:
        return fsi_tt_efficient_simple(tensor, ell).reshape(-1)
    raise ValueError("use method='bounded' (optionally sparse) or method='dense'")
