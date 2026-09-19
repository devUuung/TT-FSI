import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import paper_measure as pm


def test_sampling_has_two_warmups_and_ten_distinct_seed_calls(monkeypatch):
    calls=[]
    lookup={(i,): i for i in range(14)}
    def result(seed):
        return SimpleNamespace(interaction_lookup=lookup, values=np.arange(14, dtype=float)+seed*0.01)
    class Exact:
        def __init__(self, **kwargs): pass
        def __call__(self, **kwargs): return result(0)
    class Approx:
        def __init__(self, *, n, max_order, random_state): self.seed=random_state
        def approximate(self, *, budget, game):
            calls.append(self.seed)
            game(np.zeros((1,14),dtype=bool))
            return result(self.seed)
    monkeypatch.setitem(sys.modules,'shapiq',SimpleNamespace(ExactComputer=Exact,RegressionFSII=Approx))
    monkeypatch.setattr(pm,'paper_input',lambda *args:(np.zeros(1<<14),{'sha256':'test'}))
    monkeypatch.setitem(pm.PROTOCOLS,4,{**pm.PROTOCOLS[4],'budgets':[1000]})
    rows=pm.measure_sampling(4)
    assert calls==[0,1]+list(range(10))
    assert [r['seed'] for r in rows]==list(range(10))
    assert all(r['timing']['n_runs']==1 and len(r['approximation_per_run'])==1 for r in rows)
    assert all(r['approximation_per_run'][0]['outside_timing'] for r in rows)
    assert rows[0]['rmse_nonempty']==0
    assert rows[-1]['rmse_nonempty']>0


def test_all_active_tables_share_timing_protocol():
    from paper_tables import PROTOCOLS
    for table in [1,2,3,4,6]:
        assert {k:PROTOCOLS[table][k] for k in ['seeds','warmups','runs']} == {'seeds':list(range(10)),'warmups':2,'runs':10}
    assert PROTOCOLS[4]['estimator_seeds']==list(range(10))


def test_generation_uses_two_full_warmups_and_ten_checked_calls(monkeypatch):
    calls=[]
    saved=np.arange(4,dtype=float)
    def generate(*args):
        calls.append(1)
        return saved.copy()
    model=SimpleNamespace(reference_game=lambda:(None,None))
    monkeypatch.setitem(sys.modules,'model_games',SimpleNamespace(settings=lambda:{'model_by_experiment':{'table6':'xgb'},'dataset_by_dimension':{'2':'test'}},generate_values=generate))
    monkeypatch.setitem(sys.modules,'checkpoints',SimpleNamespace(load_checkpoint=lambda *args:model))
    monkeypatch.setattr(pm,'DIMENSIONS',[2])
    monkeypatch.setattr(pm,'paper_input',lambda *args:(saved,{'sha256':'test'}))
    rows=pm.measure_generation(None)
    assert len(calls)==12
    assert len(rows[0]['samples_ms'])==len(rows[0]['per_run'])==10
    assert rows[0]['timing_protocol']=={'seeds':list(range(10)),'warmup_seeds':[0,1],'warmups':2,'runs':10}
