from fractions import Fraction
from math import comb
import numpy as np
import pytest

from tt_fsi import fsi
from tt_fsi.core import generic_operator as generic
from tt_fsi.core.matvec_efficient import mpo_matvec_sweep
from tt_fsi.naive.fsi_naive import fsi_naive

TOL = dict(rtol=1.3e-6, atol=1e-5)


def rational_wls(values, ell):
    """Independent exact-rational Shapley-weighted constrained regression."""
    n = len(values); d = n.bit_length()-1
    masks = [s for s in range(1, n) if s.bit_count() <= ell]
    m = len(masks)
    matrix = [[Fraction(0) for _ in range(m+2)] for _ in range(m+1)]
    for coalition in range(1, n-1):
        k = coalition.bit_count()
        weight = Fraction(1, comb(d, k)*k*(d-k))
        active = [j for j, s in enumerate(masks) if coalition & s == s]
        y = Fraction(int(values[coalition])-int(values[0]))
        for j in active:
            matrix[j][-1] += weight*y
            for i in active: matrix[j][i] += weight
    for j in range(m): matrix[j][m] = matrix[m][j] = Fraction(1)
    matrix[m][-1] = Fraction(int(values[-1])-int(values[0]))
    for j in range(m+1):
        pivot = next(i for i in range(j,m+1) if matrix[i][j])
        matrix[j], matrix[pivot] = matrix[pivot], matrix[j]
        scale = matrix[j][j]
        matrix[j] = [x/scale for x in matrix[j]]
        for i in range(m+1):
            if i != j:
                factor = matrix[i][j]
                matrix[i] = [a-factor*b for a,b in zip(matrix[i],matrix[j])]
    out = np.zeros(n, np.float32); out[0] = values[0]
    out[masks] = [float(matrix[j][-1]) for j in range(m)]
    return out


@pytest.mark.parametrize('d', range(1,5))
def test_all_orders_against_brute_and_independent_wls(d):
    values = np.random.default_rng(d).integers(-3,4,size=1<<d)
    for ell in range(1,d+1):
        expected = rational_wls(values,ell)
        for actual in [fsi(values,ell), fsi(values,ell,method='dense'),fsi_naive(values,ell)]:
            assert actual.dtype == np.float32
            np.testing.assert_allclose(actual, expected, **TOL)


def test_integer_fractional_game():
    values = np.zeros(8, dtype=int); values[-1] = 1
    expected = np.array([0,-1/6,-1/6,1/2,-1/6,1/2,1/2,0],np.float32)
    for dtype in [int,np.float32,np.float64]:
        for fn in [fsi,fsi_naive]:
            np.testing.assert_allclose(fn(values.astype(dtype),2),expected,**TOL)


@pytest.mark.parametrize('ell',[0,4,-1,True,1.5])
def test_invalid_order(ell):
    with pytest.raises(ValueError): fsi(np.zeros(8),ell)
    with pytest.raises(ValueError): fsi_naive(np.zeros(8),ell)


@pytest.mark.parametrize('values',[[],[1],np.zeros((2,2)),np.zeros(3),
                                  [0,np.nan],[0,np.inf],[0,1j],np.array([0,1],object)])
def test_invalid_flat_inputs(values):
    with pytest.raises(ValueError): fsi(values,1)
    with pytest.raises(ValueError): fsi_naive(values,1)


def test_noncontiguous_input():
    values = np.arange(16,dtype=np.float32)[::2]
    np.testing.assert_array_equal(fsi(values,2),fsi(values.copy(),2))


@pytest.mark.parametrize('d',range(1,6))
def test_sparse_cores_and_precontraction(d):
    for ell in range(1,d+1):
        cores = generic.build_generic_sparse_cores(d,ell,generic.FSI_SPEC)
        pre = generic.build_generic_sparse_cores(d,ell,generic.FSI_SPEC,precontract=True)
        for core,b in zip(cores,pre):
            dense = core.to_dense()
            expected = dense.copy()
            expected[:,:,0,:] = dense[:,:,0,:]-dense[:,:,1,:]
            np.testing.assert_array_equal(b.to_dense(),expected)
            assert len(set(map(tuple,b.keys))) == len(b.keys)
            assert np.all(b.coefficients != 0)
            assert b.nbytes == b.keys.nbytes+b.coefficients.nbytes
        v = np.arange(1<<d,dtype=np.float32).reshape((2,)*d)
        from tt_fsi.core.mobius import mobius_mpo
        a = mpo_matvec_sweep([c.to_dense() for c in cores],mobius_mpo(v))
        b = mpo_matvec_sweep([c.to_dense() for c in pre],v)
        np.testing.assert_allclose(a,b,**TOL)


def test_bounded_plans_never_materialize_dense_correction(monkeypatch):
    from tt_fsi.core import gpu_tt
    def forbidden(*args,**kwargs):
        raise AssertionError('Dense correction core materialized')
    monkeypatch.setattr(generic.SparseCore,'to_dense',forbidden)
    generic._cached_fsi_cores.cache_clear()
    gpu_tt._strat_plan.cache_clear()
    fsi(np.arange(16),2)
    gpu_tt._strat_plan(4,2,'correction')
    gpu_tt._pruned_plan.cache_clear()
    gpu_tt._pruned_plan(4,2,'precontract')
    indices = np.concatenate(list(gpu_tt._strat_index_map(4,2).values()))
    assert sorted(indices) == [s for s in range(16) if s.bit_count()<=2]


@pytest.mark.parametrize('d',range(1,6))
def test_stratified_plan_maps_distinct_rows_to_cpu_reference(d):
    from tt_fsi.core.gpu_tt import _strat_plan,_strat_index_map
    values=np.random.default_rng(d).integers(-2,3,size=(3,1<<d)).astype(np.float32)
    for ell in range(1,d+1):
        outputs=[]
        for kind in ['mobius','correction']:
            current=values
            for st in _strat_plan(d,ell,kind):
                destination=np.zeros((3,st['dst_rows_total'],st['half']),np.float32)
                source=current.reshape(3,st['src_rows_total'],2,st['half'])
                for row,sector in enumerate(st['sec_of_row']):
                    local=row-st['rowbase'][sector]
                    front=st['front'][sector]
                    side=int(local>=front)
                    source_row=local-front if side else local
                    for j in range(st['cstart'][sector],st['cstart'][sector+1]):
                        if st['c_side'][j]==side:
                            destination[:,row,:]+=st['c_co'][j]*source[:,st['c_off'][j]+source_row,st['c_tau'][j],:]
                current=destination
            outputs.append(current.reshape(3,-1))
        indices=np.concatenate(list(_strat_index_map(d,ell).values()))
        expected=np.stack([fsi_naive(row,ell)[indices] for row in values])
        np.testing.assert_allclose(outputs[0]+outputs[1],expected,**TOL)


def test_compact_core_equals_reachable_part_of_legacy_rectangular_core():
    d,ell=5,3
    for k,core in enumerate(generic.build_generic_sparse_cores(d,ell,generic.FSI_SPEC)):
        left=[(s,t) for s in range(min(k,ell)+1) for t in range(s,k+1)]
        right=[(s,t) for s in range(min(k+1,ell)+1) for t in range(s,k+2)]
        legacy=np.zeros((1 if k==0 else (min(k,ell)+1)*(k+1),2,2,
                         1 if k==d-1 else (min(k+1,ell)+1)*(k+2)),np.float32)
        for source,(s,t) in enumerate((s,t) for s in range(min(k,ell)+1) for t in range(k+1)):
            for sigma,tau in [(0,0),(0,1),(1,1)]:
                sn,tn=s+sigma,t+tau
                if sn>ell:continue
                if k==d-1:
                    w=Fraction(0) if sn==0 or tn<=ell else Fraction((-1)**(ell-sn)*sn*comb(ell,sn)*comb(tn-1,ell),
                                                                          (ell+sn)*comb(tn+ell-1,ell+sn))
                    legacy[source,sigma,tau,0]=float(w)
                else:legacy[source,sigma,tau,sn*(k+2)+tn]=1
        selected=legacy[[s*(k+1)+t for s,t in left]]
        if k!=d-1:selected=selected[:,:,:,[s*(k+2)+t for s,t in right]]
        np.testing.assert_array_equal(core.to_dense(),selected)
