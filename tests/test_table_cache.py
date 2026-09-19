import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import paper_tables as tables
from measurement import timing_summary, validation_summary


def valid_table():
    checks = [{'status':'passed','max_abs_error':0.,'rmse':0.,'outside_timing':True,'seed':seed} for seed in range(10)]
    return tables.envelope(1,[{'d':d,'method':method,'status':'ok',
        'timing':timing_summary([1.]*10),'validation':validation_summary(checks)}
        for d in tables.DIMENSIONS for method in tables.METHODS],'test',{})


def checksum(data):
    data['content_sha256']=tables.canonical_hash({k:v for k,v in data.items() if k!='content_sha256'})


@pytest.mark.parametrize('change',['missing','failed','false_summary'])
def test_rejects_incomplete_or_failed_success_cells(change):
    data=valid_table();validation=data['records'][0]['validation']
    if change=='missing': validation['per_run'].pop()
    elif change=='failed': validation['per_run'][3]['status']='failed'
    else:validation['max_abs_error']=1.
    checksum(data)
    with pytest.raises(ValueError):tables.validate(data,1)


def test_operator_source_change_invalidates_cache(tmp_path,monkeypatch):
    data=valid_table()
    old=data['execution_fingerprint']
    changed=copy.deepcopy(old)
    changed['sources']['tt_fsi/core/gpu_tt.py']='changed-source'
    monkeypatch.setattr(tables,'execution_fingerprint',lambda:changed)
    with pytest.raises(ValueError,match='fingerprint'):tables.validate(data,1)
    output=tmp_path/'result'
    tables.write_json(output/'cache/table1.json',data)
    calls=[]
    def measure(table,root):
        calls.append(table)
        return valid_table()
    tables.run_table(1,output=output,root=tmp_path,measure=measure)
    assert calls==[1]
    assert list((output/'history').glob('*.json'))


def test_failures_never_render_as_success_or_invent_oom():
    data=valid_table()
    for i,status in enumerate(['oom','validation_failed','error']):
        data['records'][i].update(status=status,exception_type='TestError',
                                 error_message='test',failure_stage='validation',memory_limit_bytes=None)
    checksum(data);tables.validate(data,1)
    _,rows=tables.display(data)
    assert rows[0][1:4]==['OOM','FAIL','FAIL']


def test_fingerprint_reads_actual_operator_source(tmp_path):
    core=tmp_path/'tt_fsi/core'
    core.mkdir(parents=True)
    source=core/'gpu_tt.py'
    source.write_text('version = 1\n')
    before=tables.execution_fingerprint(tmp_path)
    source.write_text('version = 2\n')
    after=tables.execution_fingerprint(tmp_path)
    assert before['sources']['tt_fsi/core/gpu_tt.py']!=after['sources']['tt_fsi/core/gpu_tt.py']


def test_operator_cache_rejects_repeated_seed():
    data=valid_table()
    data['records'][0]['validation']['per_run'][4]['seed']=0
    checksum(data)
    with pytest.raises(ValueError,match='seed schedule'):tables.validate(data,1)
