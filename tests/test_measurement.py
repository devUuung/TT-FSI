import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from measurement import correctness, timed, validation_summary, ValidationError


def test_validates_each_timed_output_after_sync():
    events = []
    samples, checks = [], []
    def operator():
        events.append('operator')
        return len([x for x in events if x == 'operator'])
    def sync():
        events.append('sync')
    def check(output):
        assert events[-1] == 'sync'
        assert len(samples) == len(checks) + 1
        events.append('check')
        return {'status': 'passed', 'output': output, 'max_abs_error': 0, 'rmse': 0}
    timed(operator, sync, samples, check=check, checks=checks)
    assert [r['output'] for r in checks] == list(range(3, 13))
    assert events.count('operator') == 12
    assert validation_summary(checks)['all_passed']


@pytest.mark.parametrize('bad_run', [2, 5, 10])
def test_detects_incorrect_middle_output(bad_run):
    samples, checks = [], []
    calls = 0
    def operator():
        nonlocal calls
        calls += 1
        return np.full((2,), calls == bad_run, dtype=np.float32)
    with pytest.raises(ValidationError):
        timed(operator, lambda: None, samples, warmups=0,
              check=lambda out: correctness(out, np.zeros(2), 1, 1, 'direct', None,
                                            {'rtol': 1e-8, 'atol': 1e-8}), checks=checks)
    assert len(checks) == len(samples) == bad_run
    assert not validation_summary(checks)['all_passed']
    assert checks[-1]['status'] == 'failed'


@pytest.mark.parametrize('actual', [np.zeros((4,)), np.full((2, 2), np.nan),
                                    np.full((2, 2), np.inf)])
def test_rejects_shape_and_nonfinite_outputs(actual):
    assert correctness(actual, np.zeros(4), 2, 2, 'direct', None,
                       {'rtol': 1e-8, 'atol': 1e-8})['status'] == 'failed'


@pytest.mark.parametrize('indices,shape', [([0, 1, 1], (2, 3)),
                                          ([0, 1], (2, 2)),
                                          ([0, 1, 2], (1, 3)),
                                          ([0., 1., 2.], (2, 3))])
def test_rejects_sparse_indices_and_batch_broadcast(indices, shape):
    output = {'indices': np.array(indices), 'values': np.zeros(shape), 'shape': (2, 4)}
    fake_cp = SimpleNamespace(asnumpy=np.asarray)
    assert correctness(output, np.zeros(4), 2, 1, 'gpu_fused_bounded', fake_cp,
                       {'rtol': 1e-8, 'atol': 1e-8}, batch=2)['status'] == 'failed'


def test_shapiq_has_no_numerical_exemption():
    output = SimpleNamespace(interaction_lookup={(): 0, (0,): 1}, values=np.ones(2))
    assert correctness(output, np.zeros(2), 1, 1, 'shapiq_exact', None,
                       {'rtol': 1e-8, 'atol': 1e-8})['status'] == 'failed'


def test_diagnostic_only_collects_all_failed_numerical_outputs():
    from experiments.measurement import timed
    samples, checks = [], []
    timed(lambda: 1, lambda: None, samples, warmups=0, runs=10,
          check=lambda out: {'status': 'failed', 'max_abs_error': 0.1, 'rmse': 0.1},
          checks=checks, require_allclose=False)
    assert len(samples) == len(checks) == 10
    assert all(c['status'] == 'failed' for c in checks)


def test_diagnostic_only_still_rejects_structural_errors():
    from experiments.measurement import timed, ValidationError
    import pytest
    with pytest.raises(ValidationError):
        timed(lambda: 1, lambda: None, [], warmups=0, runs=10,
              check=lambda out: {'status': 'failed', 'error': 'Wrong shape'},
              checks=[], require_allclose=False)


def test_timing_reinitializes_each_measured_seed():
    import random
    seen=[];checks=[];samples=[]
    def invoke(seed):
        seen.append((seed, np.random.rand(), random.random()))
        return seed
    timed(invoke,lambda:None,samples,2,10,
          check=lambda out:{'status':'passed'},checks=checks,pass_seed=True)
    assert [x[0] for x in seen]==[0,1]+list(range(10))
    assert [c['seed'] for c in checks]==list(range(10))
    for seed,n,p in seen:
        assert n==np.random.RandomState(seed).rand()
        assert p==random.Random(seed).random()
