import numpy as np
import pytest

cp = pytest.importorskip('cupy')
try:
    DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip('CUDA is unavailable', allow_module_level=True)
if DEVICE_COUNT == 0:
    pytest.skip('CUDA is unavailable', allow_module_level=True)

from tt_fsi import fsi
from tt_fsi.core import gpu_tt as gpu

TOL = dict(rtol=1.3e-6,atol=1e-5)


@pytest.mark.parametrize('dtype',[np.int32,np.float32,np.float64])
@pytest.mark.parametrize('ell',[1,2,3,4])
def test_distinct_rows_and_all_gpu_schedules(dtype,ell):
    values = np.random.default_rng(8).integers(-2,3,size=(3,32)).astype(dtype)[:,::2]
    device = cp.asarray(np.ascontiguousarray(values))
    expected = np.stack([fsi(row,ell) for row in values])
    for actual in [gpu.fsi_tt_gpu_batch(device,ell),gpu.fsi_tt_gpu_batch_chunked(device,ell,chunk_size=2)]:
        assert actual.dtype == cp.float32
        np.testing.assert_allclose(cp.asnumpy(actual),expected,**TOL)
    for actual in [gpu.fsi_tt_gpu_stratified(values,ell),
                   gpu.fsi_tt_gpu_sparse_batch(device,ell),
                   gpu.fsi_tt_gpu_sparse_batch(device,ell,use_raw=True),
                   gpu.fsi_tt_gpu_stratified_chunked(device,ell,chunk_size=2),
                   gpu.fsi_tt_gpu_sparse_batch_chunked(device,ell,chunk_size=2,use_raw=True)]:
        indices = cp.asnumpy(actual['indices'])
        assert sorted(indices) == [s for s in range(16) if s.bit_count()<=ell]
        assert actual['values'].dtype == cp.float32
        np.testing.assert_allclose(cp.asnumpy(actual['values']),expected[:,indices],**TOL)
    np.testing.assert_allclose(cp.asnumpy(gpu.fsi_tt_gpu(device[1],ell)),expected[1],**TOL)


def test_device_strided_input_and_invalid_inputs():
    x = cp.arange(32,dtype=cp.float32).reshape(2,16)[:,::2]
    for fn in [gpu.fsi_tt_gpu_batch,gpu.fsi_tt_gpu_stratified,gpu.fsi_tt_gpu_sparse_batch]:
        fn(x,2)
        for bad in [cp.empty((0,8)),cp.empty((2,0)),cp.empty((2,3)),cp.ones(8),
                    cp.full((1,8),cp.nan),cp.ones((1,8),dtype=cp.complex64)]:
            with pytest.raises(ValueError):fn(bad,2)
        for ell in [0,4,True,1.5]:
            with pytest.raises(ValueError):fn(x,ell)
        with pytest.raises(ValueError):fn(x,2,dtype=cp.float64)
    for fn in [gpu.fsi_tt_gpu_batch_chunked,gpu.fsi_tt_gpu_sparse_batch_chunked,gpu.fsi_tt_gpu_stratified_chunked]:
        for size in [0,-1,True,1.5]:
            with pytest.raises(ValueError):fn(x,2,chunk_size=size)
    for kernel in [gpu._strat_kernel,gpu._transition_kernel]:
        with pytest.raises(ValueError):kernel('int32')


@pytest.mark.parametrize('method',['bounded','dense'])
def test_prepared_workspace_and_output(method):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
    from measurement import workspace_peak_bytes
    x = cp.arange(32,dtype=cp.float32).reshape(2,16)
    op = gpu.prepare_fsi_gpu(x,2,method)
    out = op.allocate_output()
    first = op.run(out=out)
    returned = first['values'] if method=='bounded' else first
    assert returned.data.ptr == out.data.ptr
    footprint = op.memory_breakdown()
    assert footprint['resident_input_bytes'] == x.nbytes
    assert footprint['resident_operator_bytes'] > 0
    assert footprint['output_bytes'] >= out.nbytes
    assert workspace_peak_bytes(cp,lambda:op.run(out=out)) > 0
    with pytest.raises(ValueError):op.run(out=cp.empty((1,1)))
    # Preparation must absorb conversion and the full finite-input scan.
    from unittest.mock import patch
    with patch.object(gpu,'_batch_input',side_effect=AssertionError('Repeated preparation')):
        op.run(out=out)


@pytest.mark.skipif(DEVICE_COUNT<2,reason='Needs two CUDA devices')
def test_plan_cache_is_device_specific():
    with cp.cuda.Device(0):
        first = gpu.prepare_fsi_gpu(np.arange(8)[None,:],2)
    with cp.cuda.Device(1):
        second = gpu.prepare_fsi_gpu(np.arange(8)[None,:],2)
        assert second.indices.device.id == 1
        with pytest.raises(ValueError):first.run()
    assert first.indices.device.id == 0
