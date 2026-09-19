"""CuPy TT contraction for FSI.

The CPU implementation builds the Möbius TT operator and the FSI correction TT operator as
lists of cores.  This module keeps those same cores and performs the
left-to-right contraction on GPU with CuPy.  It supports both a single value
table and a batch of value tables.
"""

from __future__ import annotations

from functools import lru_cache, wraps
from math import comb
from typing import Any

import numpy as np

from .generic_operator import (
    FSI_SPEC,
    build_generic_sparse_cores,
    build_generic_mpo_cores,
    build_generic_precontract_cores,
    idx_state,
)
from .mobius import build_mobius_mpo_cores
from .matvec_efficient import core_transitions
from tt_fsi._validation import fp32_dtype, real_array, validate_order, validate_chunk


def _device_cached(fn):
    @lru_cache(maxsize=None)
    def cached(device, *args):
        return fn(*args)
    @wraps(fn)
    def wrapped(*args):
        cp = __import__('cupy')
        return cached(cp.cuda.Device().id, *args)
    wrapped.cache_clear = cached.cache_clear
    return wrapped


def _batch_input(values, ell, dtype=None):
    cp = __import__('cupy')
    fp32_dtype(dtype)
    arr = real_array(values, cp)
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError('values must have shape (B, 2**d), B >= 1')
    n = arr.shape[1]
    if n < 2 or n & (n-1):
        raise ValueError('values must have length 2**d, d >= 1')
    d = n.bit_length()-1
    validate_order(ell, d)
    if arr.device.id != cp.cuda.Device().id:
        raise ValueError('Input must reside on the active CUDA device')
    return arr, d



@lru_cache(maxsize=None)
def _popcounts(n: int) -> np.ndarray:
    return np.fromiter((i.bit_count() for i in range(n)), dtype=np.int16, count=n)


def _to_cupy_cores(cores: list[np.ndarray], cp: Any, dtype: Any) -> list[Any]:
    return [cp.asarray(core, dtype=dtype) for core in cores]


def _dtype_name(dtype: Any) -> str:
    return np.dtype(dtype).name


@_device_cached
def _cached_cupy_cores(d: int, ell: int, dtype_name: str):
    cp = __import__("cupy")
    dtype = cp.dtype(dtype_name)
    mobius = tuple(_to_cupy_cores(build_mobius_mpo_cores(d), cp, dtype))
    correction = tuple(_to_cupy_cores(build_generic_mpo_cores(d, ell, FSI_SPEC), cp, dtype))
    return mobius, correction


@_device_cached
def _cached_cupy_precontract_cores(d: int, ell: int, dtype_name: str):
    cp = __import__("cupy")
    dtype = cp.dtype(dtype_name)
    mobius = tuple(_to_cupy_cores(build_mobius_mpo_cores(d), cp, dtype))
    correction = tuple(_to_cupy_cores(build_generic_precontract_cores(d, ell, FSI_SPEC), cp, dtype))
    return mobius, correction


@_device_cached
def _cached_cupy_popcounts(n: int):
    cp = __import__("cupy")
    return cp.asarray(_popcounts(n), dtype=cp.int16)


def _mpo_matvec_batch(cores: list[Any], values: Any) -> Any:
    """Apply TT operator cores to a batched flat tensor of shape (batch, 2^d)."""
    cp = __import__("cupy")
    batch, n = values.shape
    d = len(cores)
    if n != 1 << d:
        raise ValueError(f"values length must be 2^d={1 << d}, got {n}")

    intermediate = values.reshape((batch, 1) + (2,) * d)
    for k, core in enumerate(cores):
        d_left, _, _, d_right = core.shape
        n_out_dims = k
        n_in_dims = d - k
        n_out = 1 << n_out_dims
        n_in_remaining = 1 << (n_in_dims - 1)

        inter = intermediate.reshape(batch, d_left, n_out, 2, n_in_remaining)
        new_inter = cp.einsum("baotr,astc->boscr", inter, core, optimize=True)
        new_inter = new_inter.transpose(0, 3, 1, 2, 4)
        new_inter = new_inter.reshape(batch, d_right, n_out * 2, n_in_remaining)
        intermediate = new_inter.reshape((batch, d_right) + (2,) * (n_out_dims + 1) + (2,) * (n_in_dims - 1))
    return intermediate.reshape(batch, n)


@lru_cache(maxsize=None)
def _pruned_plan(d: int, ell: int, core_kind: str):
    if core_kind == "mobius":
        cores = build_mobius_mpo_cores(d)
    elif core_kind == "correction":
        cores = build_generic_sparse_cores(d, ell, FSI_SPEC)
    elif core_kind == "precontract":
        cores = build_generic_sparse_cores(d, ell, FSI_SPEC, precontract=True)
    else:
        raise ValueError(core_kind)

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
            idx = np.nonzero(valid)[0].astype(np.int64)
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
        steps.append((d_right, n_remaining, len(prefixes), len(next_prefixes), tuple(transitions)))
        prefixes = next_prefixes
        counts = next_counts
    return tuple(steps), prefixes


@_device_cached
def _cached_cupy_pruned_plan(d: int, ell: int, core_kind: str, dtype_name: str):
    cp = __import__("cupy")
    dtype = cp.dtype(dtype_name)
    steps, prefixes = _pruned_plan(d, ell, core_kind)
    gpu_steps = []
    for d_right, n_remaining, n_prefixes, n_next_prefixes, transitions in steps:
        gpu_transitions = []
        for alpha, tau, beta, coeff, source_idx, target_idx in transitions:
            gpu_transitions.append(
                (
                    alpha,
                    tau,
                    beta,
                    dtype.type(coeff),
                    cp.asarray(source_idx, dtype=cp.int64),
                    cp.asarray(target_idx, dtype=cp.int64),
                )
            )
        gpu_steps.append((d_right, n_remaining, n_prefixes, n_next_prefixes, tuple(gpu_transitions)))
    return tuple(gpu_steps), cp.asarray(prefixes, dtype=cp.int64)


def _mpo_matvec_batch_pruned(steps: tuple, values: Any) -> tuple[Any, Any]:
    cp = __import__("cupy")
    batch, n = values.shape
    intermediate = values.reshape(batch, 1, 1, n)
    for d_right, n_remaining, n_prefixes, n_next_prefixes, transitions in steps:
        current = intermediate.reshape(batch, -1, n_prefixes, 2, n_remaining)
        new_intermediate = cp.zeros((batch, d_right, n_next_prefixes, n_remaining), dtype=values.dtype)
        for alpha, tau, beta, coeff, source_idx, target_idx in transitions:
            new_intermediate[:, beta, target_idx, :] += coeff * current[:, alpha, source_idx, tau, :]
        intermediate = new_intermediate
    return intermediate.reshape(batch, -1)


@_device_cached
def _transition_kernel(dtype_name: str):
    cp = __import__("cupy")
    fp32_dtype(dtype_name)
    ctype = "float"
    return cp.RawKernel(
        rf'''
        extern "C" __global__
        void transition_add(
            {ctype}* out,
            const {ctype}* cur,
            const long long* src,
            const long long* tgt,
            const int source_len,
            const int n_remaining,
            const int d_left,
            const int n_prefixes,
            const int d_right,
            const int n_next_prefixes,
            const int alpha,
            const int tau,
            const int beta,
            const {ctype} coeff,
            const int batch
        ) {{
            long long total = (long long) batch * source_len * n_remaining;
            long long gid = (long long) blockDim.x * blockIdx.x + threadIdx.x;
            for (long long idx = gid; idx < total; idx += (long long) blockDim.x * gridDim.x) {{
                int r = idx % n_remaining;
                long long tmp = idx / n_remaining;
                int j = tmp % source_len;
                int b = tmp / source_len;
                long long cur_idx =
                    (((((long long)b * d_left + alpha) * n_prefixes + src[j]) * 2 + tau)
                     * n_remaining + r);
                long long out_idx =
                    ((((long long)b * d_right + beta) * n_next_prefixes + tgt[j])
                     * n_remaining + r);
                out[out_idx] += coeff * cur[cur_idx];
            }}
        }}
        ''',
        "transition_add",
    )


def _mpo_matvec_batch_pruned_raw(steps: tuple, values: Any) -> Any:
    cp = __import__("cupy")
    batch, n = values.shape
    dtype_name = _dtype_name(values.dtype)
    kernel = _transition_kernel(dtype_name)
    intermediate = values.reshape(batch, 1, 1, 2, n // 2)
    # The first loop below expects the unprocessed input length in the step,
    # so reset to the common shape used by the vectorized pruned implementation.
    intermediate = values.reshape(batch, 1, 1, n)
    threads = 256
    for d_right, n_remaining, n_prefixes, n_next_prefixes, transitions in steps:
        current = intermediate.reshape(batch, -1, n_prefixes, 2, n_remaining)
        new_intermediate = cp.zeros((batch, d_right, n_next_prefixes, n_remaining), dtype=values.dtype)
        for alpha, tau, beta, coeff, source_idx, target_idx in transitions:
            source_len = int(source_idx.size)
            if source_len == 0:
                continue
            total = batch * source_len * n_remaining
            blocks = min(65535, max(1, (total + threads - 1) // threads))
            kernel(
                (blocks,),
                (threads,),
                (
                    new_intermediate,
                    current,
                    source_idx,
                    target_idx,
                    source_len,
                    n_remaining,
                    current.shape[1],
                    n_prefixes,
                    d_right,
                    n_next_prefixes,
                    alpha,
                    tau,
                    beta,
                    values.dtype.type(coeff),
                    batch,
                ),
            )
        intermediate = new_intermediate
    return intermediate.reshape(batch, -1)


class PreparedGPUFSI:
    """Validated device input and resident plans; run() performs only the operator.

    The caller must not mutate the prepared input to non-finite values. Timings
    use run(out=...) with a separately allocated output buffer.
    """
    def __init__(self, values, ell, method='bounded', dtype=None):
        cp = __import__('cupy')
        self.values, self.d = _batch_input(values, ell, dtype)
        self.ell = int(ell)
        self.method = method
        self.device = cp.cuda.Device().id
        if method == 'bounded':
            self.plans = (_strat_gpu_plan(self.d, self.ell, 'mobius', 'float32'),
                          _strat_gpu_plan(self.d, self.ell, 'correction', 'float32'))
            self.indices = _strat_gpu_indices(self.d, self.ell)
            self.output_shape = (self.values.shape[0], self.indices.size)
            _strat_kernel('float32').compile()
        elif method == 'dense':
            self.plans = _cached_cupy_precontract_cores(self.d, self.ell, 'float32')
            self.mask = _cached_cupy_order_mask(self.values.shape[1], self.ell)
            self.indices = None
            self.output_shape = self.values.shape
        else:
            raise ValueError('method must be bounded or dense')

    def allocate_output(self):
        cp = __import__('cupy')
        return cp.empty(self.output_shape, dtype=cp.float32)

    def memory_breakdown(self):
        arrays = {}
        def visit(obj):
            if hasattr(obj, 'data') and hasattr(obj.data, 'ptr'):
                arrays[obj.data.ptr] = obj.nbytes
            elif isinstance(obj, dict):
                for x in obj.values(): visit(x)
            elif isinstance(obj, (tuple, list)):
                for x in obj: visit(x)
        visit(self.plans)
        if self.indices is None: visit(self.mask)
        # Sparse indices are part of the final output, even though cached.
        return {'resident_input_bytes': int(self.values.nbytes),
                'output_bytes': int(np.prod(self.output_shape))*4 + (0 if self.indices is None else int(self.indices.nbytes)),
                'resident_operator_bytes': int(sum(arrays.values())),
                'reusable_workspace_bytes': 0}

    def run(self, out=None):
        cp = __import__('cupy')
        if cp.cuda.Device().id != self.device:
            raise ValueError('Prepared operator belongs to a different CUDA device')
        if out is None: out = self.allocate_output()
        if not isinstance(out, cp.ndarray) or out.shape != self.output_shape or out.dtype != cp.float32 or not out.flags.c_contiguous or out.device.id != self.device:
            raise ValueError('out must be a contiguous float32 device array with the prepared output shape')
        if cp.may_share_memory(out, self.values):
            raise ValueError('out must not overlap the input')
        if self.method == 'bounded':
            _strat_run('mobius', self.values, self.d, self.ell, out=out)
            _strat_run('correction', self.values, self.d, self.ell, out=out, add=True)
            return {'indices': self.indices, 'values': out, 'shape': self.values.shape}
        mobius = _mpo_matvec_batch(self.plans[0], self.values)
        correction = _mpo_matvec_batch(self.plans[1], self.values)
        cp.add(mobius, correction, out=out)
        out[:, self.mask] = 0
        return out


def prepare_fsi_gpu(values, ell, method='bounded', dtype=None):
    return PreparedGPUFSI(values, ell, method, dtype)


def fsi_tt_gpu_batch(values: Any, ell: int, dtype: Any | None = None, *, out=None) -> Any:
    """FP32 full-prefix FSI with precontracted correction on the original input."""
    return prepare_fsi_gpu(values, ell, 'dense', dtype).run(out=out)


def _sparse_batch_run(arr, ell, use_raw):
    cp = __import__('cupy')
    d = arr.shape[1].bit_length()-1
    mobius_steps, indices = _cached_cupy_pruned_plan(d, ell, 'mobius', 'float32')
    steps, _ = _cached_cupy_pruned_plan(d, ell, 'precontract', 'float32')
    runner = _mpo_matvec_batch_pruned_raw if use_raw else _mpo_matvec_batch_pruned
    return {'indices': indices, 'values': runner(mobius_steps, arr) + runner(steps, arr), 'shape': arr.shape}


def fsi_tt_gpu_sparse_batch(values: Any, ell: int, dtype: Any | None = None, use_raw: bool = False):
    arr, _ = _batch_input(values, ell, dtype)
    return _sparse_batch_run(arr, int(ell), use_raw)


def _chunked(values, ell, dtype, chunk_size, method, use_raw=False):
    cp = __import__('cupy')
    validate_chunk(chunk_size)
    arr, d = _batch_input(values, ell, dtype)
    chunk_size = arr.shape[0] if chunk_size is None else int(chunk_size)
    sparse = method != 'dense'
    width = sum(comb(d, s) for s in range(ell+1)) if sparse else arr.shape[1]
    result = cp.empty((arr.shape[0], width), dtype=cp.float32)
    indices = None
    for start in range(0, arr.shape[0], chunk_size):
        part = arr[start:start+chunk_size]
        if method == 'sparse':
            output = _sparse_batch_run(part, ell, use_raw)
            result[start:start+len(part)] = output['values']
            indices = output['indices']
        else:
            op = prepare_fsi_gpu(part, ell, method)
            op.run(out=result[start:start+len(part)])
            indices = op.indices
    return {'indices': indices, 'values': result, 'shape': arr.shape} if sparse else result


def fsi_tt_gpu_batch_chunked(values, ell, dtype=None, chunk_size=None):
    return _chunked(values, ell, dtype, chunk_size, 'dense')


def fsi_tt_gpu_sparse_batch_chunked(values, ell, dtype=None, chunk_size=None, use_raw=False):
    return _chunked(values, ell, dtype, chunk_size, 'sparse', use_raw)


def fsi_tt_gpu(values, ell, dtype=None):
    cp = __import__('cupy')
    arr = cp.asarray(values)
    if arr.ndim != 1 and (arr.ndim < 1 or any(n != 2 for n in arr.shape)):
        raise ValueError('values must be a flat coalition table or a binary tensor')
    return fsi_tt_gpu_batch(arr.reshape(1, -1), ell, dtype).reshape(arr.shape)


# ---------------------------------------------------------------------------
# Weight-stratified (charge-sector) fused-kernel TT contraction
#
# The dense sweep materializes the full 2^k out-prefix axis and masks |S|>ell
# only at the end. Because the out-prefix weight equals the bond charge s, the
# live state at every step is confined to the weight-graded sectors
# {(s, t): s <= ell}. Ordering each weight block in the combinatorial number
# system (colex) makes the sigma_k bit-append map onto contiguous front/back
# slices of the next block with an identity row map, so the contraction needs
# no gather/scatter and no atomics. All sectors of a level are packed into one
# buffer and advanced by a single fused kernel (one thread per destination
# element, summing its <=3 source contributions). Working set is O(d^ell)
# output-side state instead of Theta(ell k * 2^d).
# ---------------------------------------------------------------------------


def _strat_corr_sectors(k: int, ell: int):
    return [(s, t) for s in range(0, min(k, ell) + 1) for t in range(s, k + 1)]


def _strat_mob_sectors(k: int, ell: int):
    return [(s, 0) for s in range(0, min(k, ell) + 1)]


@lru_cache(maxsize=None)
def _strat_plan(d: int, ell: int, kind: str):
    """CSR-style per-step plan. kind in {'mobius', 'correction'}.

    Each destination sector carries an arbitrary number of
    (src_off, tau, coeff, side) contributions, read directly from the core so
    the plan is robust to operator-specific tau mixing (Mobius folds tau down,
    correction folds it up).
    """
    if kind == "correction":
        cores = build_generic_sparse_cores(d, ell, FSI_SPEC, precontract=True)
        sectors = lambda k: _strat_corr_sectors(k, ell)

        def bidx(level, s, t, last):
            return 0 if last else (0 if level == 0 else idx_state(level, s, t, ell))
    elif kind == "mobius":
        cores = build_mobius_mpo_cores(d)
        sectors = lambda k: _strat_mob_sectors(k, ell)

        def bidx(level, s, t, last):
            return 0
    else:
        raise ValueError(kind)

    steps = []
    for k in range(d):
        last = (k == d - 1)
        half = 1 << (d - k - 1)
        suffix = 1 << (d - k)
        core = cores[k]
        coefficients = {tuple(key): coefficient for *key, coefficient in core_transitions(core)}
        D_left, _, _, D_right = core.shape

        src_secs = sectors(k)
        src_off = {}
        off = 0
        for key in src_secs:
            src_off[key] = off
            off += comb(k, key[0])
        src_rows_total = off

        # On the last step the bond collapses (an=0) and the final |T| is
        # summed out, so key the destination by weight s2 only.
        if last:
            dst_secs = [(s2, 0) for s2 in range(0, ell + 1)]
        else:
            dst_secs = sectors(k + 1)
        dst_off = {}
        off = 0
        for key in dst_secs:
            dst_off[key] = off
            off += comb(k + 1, key[0])
        dst_rows_total = off

        n = len(dst_secs)
        rowbase = np.zeros(n, np.int32)
        front = np.zeros(n, np.int32)
        nrows = np.zeros(n, np.int32)
        cstart = np.zeros(n + 1, np.int32)
        c_off, c_tau, c_side, c_co = [], [], [], []

        for di, (s2, t2) in enumerate(dst_secs):
            rowbase[di] = dst_off[(s2, t2)]
            front[di] = comb(k, s2)
            nrows[di] = comb(k + 1, s2)
            for sigma in (0, 1):
                s_src = s2 - sigma
                if s_src < 0:
                    continue
                t_cands = [0] if kind == "mobius" else range(0, k + 1)
                for t_src in t_cands:
                    sk = (s_src, t_src if kind != "mobius" else 0)
                    if sk not in src_off:
                        continue
                    a = bidx(k, sk[0], sk[1], False)
                    if a >= D_left:
                        continue
                    for tau in (0, 1):
                        if last:
                            an = 0
                        else:
                            an = bidx(k + 1, s2, t2, False)
                            if an >= D_right:
                                continue
                        co = float(coefficients.get((a, sigma, tau, an), 0))
                        if co == 0.0:
                            continue
                        c_off.append(src_off[sk])
                        c_tau.append(tau)
                        c_side.append(sigma)
                        c_co.append(co)
            cstart[di + 1] = len(c_off)

        sec_of_row = np.zeros(dst_rows_total, np.int32)
        for di in range(n):
            sec_of_row[rowbase[di]:rowbase[di] + nrows[di]] = di

        steps.append(dict(
            half=half, suffix=suffix, last=last,
            src_rows_total=src_rows_total, dst_rows_total=dst_rows_total,
            dst_secs=dst_secs, dst_off=dst_off,
            sec_of_row=sec_of_row, rowbase=rowbase, front=front, cstart=cstart,
            c_off=np.asarray(c_off, np.int32), c_tau=np.asarray(c_tau, np.int32),
            c_side=np.asarray(c_side, np.int32), c_co=np.asarray(c_co, np.float32),
        ))
    return steps


_STRAT_KERNEL_SRC = r'''
extern "C" __global__ void strat_step(
    const {ct}* src, {ct}* dst,
    const int* sec_of_row, const int* rowbase, const int* front,
    const int* cstart, const int* c_off, const int* c_tau,
    const int* c_side, const {ct}* c_co,
    const int dst_rows_total, const int half, const int suffix,
    const int src_rows_total, const int batch, const int add_to_output)
{{
  long long total = (long long) batch * dst_rows_total * half;
  long long stride = (long long) blockDim.x * gridDim.x;
  for (long long gid = (long long) blockDim.x * blockIdx.x + threadIdx.x;
       gid < total; gid += stride) {{
    int h = gid % half;
    long long tmp = gid / half;
    int grow = tmp % dst_rows_total;
    int b = tmp / dst_rows_total;
    int sec = sec_of_row[grow];
    int local = grow - rowbase[sec];
    int fr = front[sec];
    int in_front = (local < fr);
    int lr = in_front ? local : (local - fr);
    long long sb = (long long) b * src_rows_total * suffix;
    {ct} acc = 0;
    int j0 = cstart[sec], j1 = cstart[sec + 1];
    for (int j = j0; j < j1; j++) {{
      if ((c_side[j] == 0) != in_front) continue;
      acc += c_co[j] * src[sb + (long long)(c_off[j] + lr) * suffix
                            + (long long) c_tau[j] * half + h];
    }}
    long long dest = (long long) b * dst_rows_total * half + (long long) grow * half + h;
    if (add_to_output) dst[dest] += acc;
    else dst[dest] = acc;
  }}
}}
'''


@_device_cached
def _strat_kernel(dtype_name: str):
    cp = __import__("cupy")
    fp32_dtype(dtype_name)
    ct = "float"
    return cp.RawKernel(_STRAT_KERNEL_SRC.format(ct=ct), "strat_step")


@_device_cached
def _strat_gpu_plan(d: int, ell: int, kind: str, dtype_name: str):
    cp = __import__("cupy")
    steps = _strat_plan(d, ell, kind)
    dt = cp.dtype(dtype_name)
    g = []
    for st in steps:
        g.append(dict(
            half=st["half"], suffix=st["suffix"], last=st["last"],
            src_rows_total=st["src_rows_total"], dst_rows_total=st["dst_rows_total"],
            dst_secs=st["dst_secs"], dst_off=st["dst_off"],
            sec_of_row=cp.asarray(st["sec_of_row"]),
            rowbase=cp.asarray(st["rowbase"]), front=cp.asarray(st["front"]),
            cstart=cp.asarray(st["cstart"]), c_off=cp.asarray(st["c_off"]),
            c_tau=cp.asarray(st["c_tau"]), c_side=cp.asarray(st["c_side"]),
            c_co=cp.asarray(st["c_co"], dtype=dt),
        ))
    return g


def _strat_run(kind: str, v_cp: Any, d: int, ell: int, out=None, add=False):
    """Contract one branch, writing its final site into the caller output."""
    cp = __import__("cupy")
    batch = v_cp.shape[0]
    dtype_name = _dtype_name(v_cp.dtype)
    plan = _strat_gpu_plan(d, ell, kind, dtype_name)
    kern = _strat_kernel(dtype_name)
    threads = 256
    cur = v_cp.reshape(batch, -1)
    for st in plan:
        dst = out if st["last"] and out is not None else cp.empty((batch, st["dst_rows_total"] * st["half"]), dtype=v_cp.dtype)
        total = batch * st["dst_rows_total"] * st["half"]
        blocks = min(65535, max(1, (total + threads - 1) // threads))
        kern(
            (blocks,), (threads,),
            (
                cur, dst, st["sec_of_row"], st["rowbase"], st["front"],
                st["cstart"], st["c_off"], st["c_tau"], st["c_side"], st["c_co"],
                np.int32(st["dst_rows_total"]), np.int32(st["half"]),
                np.int32(st["suffix"]), np.int32(st["src_rows_total"]),
                np.int32(batch), np.int32(add and st["last"]),
            ),
        )
        cur = dst
    return cur


def _colex_unrank(s: int, r: int):
    """Return ascending element positions of the rank-r weight-s colex subset."""
    bits = []
    for i in range(s, 0, -1):
        c = i - 1
        while comb(c + 1, i) <= r:
            c += 1
        bits.append(c)
        r -= comb(c, i)
    return sorted(bits)


@lru_cache(maxsize=None)
def _strat_index_map(d: int, ell: int):
    """Per weight s, the flat subset indices for colex rows 0..C(d,s)-1.

    Step k decides sigma_{k+1} = bit (d-1-k); colex element 'k' is bit (d-1-k).
    """
    out = {}
    for s in range(0, ell + 1):
        idxs = np.empty(comb(d, s), dtype=np.int64)
        for r in range(comb(d, s)):
            flat = 0
            for k in _colex_unrank(s, r):
                flat |= 1 << (d - 1 - k)
            idxs[r] = flat
        out[s] = idxs
    return out


@_device_cached
def _strat_gpu_indices(d, ell):
    cp = __import__('cupy')
    return cp.asarray(np.concatenate(list(_strat_index_map(d, ell).values())))


@_device_cached
def _cached_cupy_order_mask(n, ell):
    cp = __import__('cupy')
    return cp.asarray(_popcounts(n) > ell)


def fsi_tt_gpu_stratified(values, ell, dtype=None, *, out=None):
    """FP32 bounded FSI; use prepare_fsi_gpu to amortize input validation."""
    return prepare_fsi_gpu(values, ell, 'bounded', dtype).run(out=out)


def fsi_tt_gpu_stratified_chunked(values, ell, dtype=None, chunk_size=None):
    return _chunked(values, ell, dtype, chunk_size, 'bounded')
