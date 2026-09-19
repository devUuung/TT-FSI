import numpy as np
import pytest
from tt_fsi.core.fsi_efficient import fsi_precontract_efficient
from tt_fsi.naive.fsi_naive import fsi_naive_tensor

@pytest.mark.parametrize("d", range(1, 6))
def test_full_prefix_cpu_against_literal(d):
    x = np.random.default_rng(d).normal(size=(2,) * d).astype(np.float32)
    for ell in range(1, d + 1):
        actual = fsi_precontract_efficient(x, ell)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, fsi_naive_tensor(x, ell), rtol=1.3e-6, atol=1e-5)

def test_dense_never_uses_pruned_contraction(monkeypatch):
    import tt_fsi.core.matvec_efficient as mv
    def forbidden(*args, **kwargs):
        raise AssertionError("Dense baseline invoked pruning")
    monkeypatch.setattr(mv, 'mpo_matvec_sweep_pruned_by_order', forbidden)
    assert fsi_precontract_efficient(np.arange(16, dtype=np.float32).reshape((2,)*4), 2).shape == (2,)*4
