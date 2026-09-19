"""Timing, GPU-allocation and correctness helpers shared by the table measurements.

Nothing here runs on import and no optional dependency is loaded until used.
"""
from __future__ import annotations

import gc
import random
from numerical_protocol import TIMING_PROTOCOL
import statistics
import time

METHODS = {
    "cpu_dense": "tt_fsi.core.fsi_efficient.fsi_precontract_efficient: full-prefix Mobius and precontracted correction",
    "cpu_prefix_pruned": "tt_fsi.core.generic_operator.compute_interaction_index: prefix-pruned correction; dense Mobius/result",
    "gpu_fused_bounded": "tt_fsi.core.gpu_tt.fsi_tt_gpu_stratified: fused cardinality-bounded sparse output",
    "gpu_dense_ablation": "tt_fsi.core.gpu_tt.fsi_tt_gpu_batch: dense TT ablation",
    "direct": "tt_fsi.naive.fsi_naive.fsi_naive_tensor: brute-force direct FSI",
    "shapiq_exact": "shapiq.ExactComputer: fresh constructor and FSII call each repetition",
}


def expected_indices(d, ell):
    return [i for i in range(1 << d) if i.bit_count() <= ell]


def validate_indices(indices, d, ell):
    import numpy as np
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise AssertionError('Sparse indices must be a one-dimensional integer array')
    actual = [int(i) for i in indices]
    if sorted(actual) != expected_indices(d, ell):
        raise AssertionError("Sparse indices must cover each order<=ell coalition exactly once")
    return actual


def timing_summary(samples, required=10, seeds=None):
    complete = len(samples) == required
    return {"samples_ms": list(samples), "sample_seeds": list(range(len(samples))) if seeds is None else list(seeds), "n_runs": len(samples), "required_runs": required,
            "mean_ms": statistics.mean(samples) if complete else None,
            "std_ms": statistics.pstdev(samples) if complete else None,
            "std_ddof": 0, "complete": complete}


def timed(fn, sync, samples, warmups=2, runs=10, *, check=None, checks=None, require_allclose=True, pass_seed=False):
    if check is not None and checks is None:
        raise ValueError('checks is required when check is supplied')
    # Release every output BEFORE starting another call, including warmups.
    def reset_seed(seed):
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    for warmup in range(warmups):
        seed = TIMING_PROTOCOL["warmup_seeds"][warmup % len(TIMING_PROTOCOL["warmup_seeds"])]
        reset_seed(seed)
        out = fn(seed) if pass_seed else fn()
        sync()
        del out
    for seed in range(runs):
        gc.collect()
        reset_seed(seed)
        sync()
        start = time.perf_counter()
        out = fn(seed) if pass_seed else fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
        try:
            if check is not None:
                result = {**check(out), "seed": seed}
                checks.append(result)
                if result.get('status') != 'passed' and (require_allclose or result.get('error') is not None):
                    raise ValidationError(result)
        finally:
            del out


class ValidationError(AssertionError):
    def __init__(self, record):
        self.record = record
        super().__init__(record.get('error', 'Timed output failed numerical validation'))


def validation_summary(checks, required=10):
    return {'n_checked_runs': len(checks),
            'all_passed': len(checks) == required and all(r['status'] == 'passed' for r in checks),
            'per_run': list(checks),
            'max_abs_error': max((r['max_abs_error'] for r in checks if r.get('max_abs_error') is not None), default=None),
            'max_rmse': max((r['rmse'] for r in checks if r.get('rmse') is not None), default=None)}


class LiveAllocations:
    """Track only successful allocations born inside this hook's lifetime."""
    def __init__(self):
        self.live = {}
        self.current = 0
        self.peak = 0

    def allocated(self, key, size, pointer):
        if pointer:
            if key in self.live:
                raise RuntimeError("Duplicate live allocation identity")
            self.live[key] = size
            self.current += size
            self.peak = max(self.peak, self.current)

    def freed(self, key):
        self.current -= self.live.pop(key, 0)


def workspace_peak_bytes(cp, fn, reusable_workspace_bytes=0):
    """Fresh warmed call, input/caches alive; excludes non-CuPy allocations.

    Tracks pool malloc/free events, including reused blocks, not pool reserve,
    driver memory, process peak, or an input-inclusive GPU peak.
    """
    tracker = LiveAllocations()

    class Hook(cp.cuda.memory_hook.MemoryHook):
        name = "tt_fsi_fresh_call_incremental_live"

        def malloc_postprocess(self, *, mem_size, mem_ptr, pmem_id, **kwargs):
            tracker.allocated(pmem_id, mem_size, mem_ptr)

        def free_postprocess(self, *, pmem_id, **kwargs):
            tracker.freed(pmem_id)

    gc.collect()
    cp.cuda.get_current_stream().synchronize()
    with Hook():
        out = fn()
        cp.cuda.get_current_stream().synchronize()
        del out
    return tracker.peak + reusable_workspace_bytes


def incremental_live_mib(cp, fn):
    """Legacy allocation metric; includes any output allocated inside fn."""
    return workspace_peak_bytes(cp, fn) / 2**20


def exact_shapiq(values, d, ell):
    import numpy as np
    import shapiq

    def game(coalitions):
        masks = np.asarray(coalitions, dtype=np.int64) @ (1 << np.arange(d, dtype=np.int64))
        return values[masks]

    return shapiq.ExactComputer(game=game, n_players=d)(index="FSII", order=ell)


def correctness(out, reference, d, ell, method, cp, config, batch=1):
    import numpy as np
    try:
        reference = np.asarray(reference)
        if reference.shape != (1 << d,) or not np.isfinite(reference).all():
            raise AssertionError('Reference must be a finite flat coalition table')
        if method == "gpu_fused_bounded":
            indices = validate_indices(cp.asnumpy(out["indices"]), d, ell)
            actual = cp.asnumpy(out["values"])
            shape = (batch, len(indices))
            if tuple(out['shape']) != (batch, 1 << d):
                raise AssertionError('Sparse output metadata has the wrong shape')
            target = np.broadcast_to(reference[indices][None, :], shape)
        elif method == "shapiq_exact":
            indices = [sum(1 << i for i in coalition) for coalition in out.interaction_lookup]
            validate_indices(indices, d, ell)
            positions = list(out.interaction_lookup.values())
            if sorted(positions) != list(range(len(indices))) or np.asarray(out.values).shape != (len(indices),):
                raise AssertionError('shapiq value positions must cover the output exactly once')
            actual = np.asarray([out.values[pos] for pos in positions])
            shape = (len(indices),)
            target = reference[indices]
        else:
            actual = cp.asnumpy(out) if method.startswith("gpu_") else np.asarray(out)
            shape = (batch, 1 << d) if method.startswith('gpu_') else (2,) * d
            target = np.broadcast_to(reference[None, :], shape) if method.startswith('gpu_') else reference.reshape(shape)
        if actual.shape != shape:
            raise AssertionError(f'Output shape {actual.shape} does not match {shape}')
        if not np.isfinite(actual).all():
            raise AssertionError('Output contains NaN or Inf')
    except (AssertionError, KeyError, IndexError, TypeError, ValueError) as exc:
        return {'status': 'failed', 'error': str(exc), 'outside_timing': True,
                'max_abs_error': None, 'rmse': None}
    with np.errstate(over='ignore', invalid='ignore'):
        passed = bool(np.allclose(actual, target, rtol=config["rtol"], atol=config["atol"]))
        diff = actual - target
    if not np.isfinite(diff).all():
        return {'status': 'failed', 'error': 'Non-finite comparison residual',
                'outside_timing': True, 'max_abs_error': None, 'rmse': None}
    maximum = float(np.max(np.abs(diff)))
    rmse = maximum * float(np.sqrt(np.mean((diff / maximum)**2))) if maximum else 0.0
    return {"status": "passed" if passed else "failed",
            "max_abs_error": maximum, "rmse": rmse, 'output_dtype': str(actual.dtype),
            'rtol': config['rtol'], 'atol': config['atol'],
            "outside_timing": True, "reference": METHODS["cpu_prefix_pruned"],
            "checked_values": int(actual.size)}
