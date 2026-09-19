"""One paper table per experiment: validated cache -> published data -> measurement."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import math
import json
from pathlib import Path
import statistics
import tempfile
from datetime import datetime, timezone

from numerical_protocol import FP32_TOLERANCES, PROTOCOL_VERSION, POLICY_SOURCE, TIMING_PROTOCOL
ROOT=Path(__file__).resolve().parents[1]


def execution_fingerprint(root=ROOT):
    paths=sorted(list((root/'tt_fsi').rglob('*.py')) + list((root/'experiments').glob('*.py')))
    hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    versions={}
    for name in ['numpy','scipy','shapiq','cupy-cuda12x','scikit-learn','xgboost','lightgbm','torch']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return {'protocol_version': PROTOCOL_VERSION,'sources':hashes,'dependency_versions':versions}

DIMENSIONS=[8,10,11,14,16,20]
BATCHES=[1,2,4,8,16,32,64,100]
METHODS=['cpu_dense','gpu_fused_bounded','direct','shapiq_exact']
GAME_CONFIG=json.loads((ROOT/'configs/paper_inputs.json').read_text())
for entry in GAME_CONFIG['checkpoints'].values():
    dataset_root=ROOT/'outputs/checkpoints'/entry['dataset']
    for name,expected in entry['files'].items():
        if hashlib.sha256((dataset_root/entry['model']/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Paper checkpoint changed; old result caches are invalid')
    for name,expected in entry['dataset_files'].items():
        if hashlib.sha256((dataset_root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Paper dataset inputs changed; old result caches are invalid')
GAME_HASH=hashlib.sha256(json.dumps(GAME_CONFIG,sort_keys=True).encode()).hexdigest()
MODEL_BY_EXPERIMENT=GAME_CONFIG['models_by_experiment']
DATASET_BY_DIMENSION=GAME_CONFIG['datasets']
BASE={**TIMING_PROTOCOL,'schema':7,'direct_validation':'diagnostic_only','input':'saved model game, centered interventional','game_config_sha256':GAME_HASH,
      'datasets':DATASET_BY_DIMENSION,'dtype':'float32','validation':FP32_TOLERANCES,
      'validation_policy_source':POLICY_SOURCE,'timing_api':'GPU prepared run(out=buffer); CPU returned-output API',
      'warmups':2,'runs':10,'gpu_sync':True,'allocation':'intermediate workspace live CuPy bytes; input/operator/output excluded'}
SAMPLING={**TIMING_PROTOCOL,'schema':5,'input':'saved Adult game','game_config_sha256':GAME_HASH,
          'dataset':DATASET_BY_DIMENSION['14'],'d':14,'ell':3,
          'ranking':'nonempty absolute magnitude, stable order/tuple tie break'}
PROTOCOLS={
  1:{**BASE,'model':MODEL_BY_EXPERIMENT['table1'],'dimensions':DIMENSIONS,'ell':3,'methods':METHODS},
  2:{**BASE,'model':MODEL_BY_EXPERIMENT['table2'],'d':20,'ell':3,'batches':BATCHES,'methods':['gpu_fused_bounded','gpu_dense_ablation']},
  3:{**BASE,'model':MODEL_BY_EXPERIMENT['table3'],'d':14,'orders':[2,3,4,5,6],'methods':['shapiq_exact','cpu_dense','gpu_dense_ablation','gpu_fused_bounded']},
  4:{**SAMPLING,'timed_calls_per_budget':10,'timing_api':'fresh estimator per seed0..9; metrics outside timing','model':MODEL_BY_EXPERIMENT['table4'],'budgets':[1000,4000,10000,16384,100000],'estimator_seeds':list(range(10))},
  5:{**SAMPLING,'model':MODEL_BY_EXPERIMENT['table5'],'budgets':[1000,4000,10000,16384,100000],'estimator_seeds':list(range(10))},
  6:{**TIMING_PROTOCOL,'schema':5,'dimensions':DIMENSIONS,'game_config_sha256':GAME_HASH,'background_rows':100,'chunk_coalitions':256,'warmup_scope':'full value generation','model':MODEL_BY_EXPERIMENT['table6'],'datasets':DATASET_BY_DIMENSION,'input':'six real datasets, one family for the whole sweep','timing':'full centered value generation, excluding fitting/loading'},
}
TITLES={1:'Operator runtime (ms)',2:'Batched GPU throughput (ms/instance, MiB)',3:'Interaction-order runtime (ms)',
        4:'RegressionFSII approximation',5:'Ranking stability over ten seeds',6:'Coalition-value generation (ms)'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False,suffix='.json') as handle:
        json.dump(data,handle,indent=2,allow_nan=False);handle.write('\n');tmp=Path(handle.name)
    tmp.replace(path)


def published_record(table,root=ROOT):
    """Only evidence produced for this mixed-model protocol is eligible for reuse."""
    path=root/'outputs/mixed-model-paper'/f'table{table}.json'
    if not path.is_file():return None
    data=json.loads(path.read_text())
    if data.get('protocol')!=PROTOCOLS[table] or data.get('execution_fingerprint')!=execution_fingerprint():return None
    validate(data,table)
    return data


def envelope(table,records,origin,provenance):
    data={'schema_version':1,'table':table,'protocol':PROTOCOLS[table],'protocol_sha256':canonical_hash(PROTOCOLS[table]),
          'execution_fingerprint':execution_fingerprint(),'origin':origin,'created_at':datetime.now(timezone.utc).isoformat(),'records':records,'provenance':provenance}
    data['content_sha256']=canonical_hash(data)
    validate(data,table)
    return data


def validate(data,table):
    payload={k:v for k,v in data.items() if k!='content_sha256'}
    if data.get('content_sha256')!=canonical_hash(payload):raise ValueError('Table cache content checksum mismatch')
    if data.get('table')!=table or data.get('protocol_sha256')!=canonical_hash(PROTOCOLS[table]) or data.get('protocol')!=PROTOCOLS[table]:
        raise ValueError('Table cache protocol mismatch')
    if data.get('execution_fingerprint')!=execution_fingerprint():raise ValueError('Execution fingerprint mismatch; remeasurement required')
    rows=data['records'];p=PROTOCOLS[table]
    if table in [1,2,3]:
        coordinate={1:'d',2:'batch',3:'ell'}[table]
        keys={(r[coordinate],r['method']) for r in rows}
        values={1:DIMENSIONS,2:BATCHES,3:[2,3,4,5,6]}[table]
        expected={(v,m) for v in values for m in p['methods']}
        if keys!=expected or len(rows)!=len(expected):raise ValueError('Missing/duplicate table cells')
        for r in rows:
            if r['status'] == 'ok':
                t=r['timing'];samples=t['samples_ms']
                if any(not math.isfinite(x) or x<0 for x in samples):raise ValueError('Invalid timed sample')
                if len(samples)!=10 or not t['complete']:raise ValueError('Incomplete timed cell')
                v=r.get('validation',{});checks=v.get('per_run',[])
                if t.get('sample_seeds')!=p['seeds'] or [c.get('seed') for c in checks]!=p['seeds']:raise ValueError('Operator seed schedule mismatch')
                if v.get('n_checked_runs')!=10 or len(checks)!=10 or (r['method']!='direct' and v.get('all_passed') is not True):
                    raise ValueError('Every timed output must have a passing validation record')
                if any((r['method']!='direct' and c.get('status')!='passed') or c.get('error') is not None or c.get('outside_timing') is not True for c in checks):
                    raise ValueError('A timed output failed validation')
                if any(not math.isfinite(c.get(key,float('nan'))) or c[key]<0 for c in checks for key in ['max_abs_error','rmse']):raise ValueError('Invalid validation metric')
                if table==2 and any(not isinstance(r.get(key),int) or r[key]<0 for key in ['workspace_peak_bytes','resident_input_bytes','output_bytes','resident_operator_bytes']):raise ValueError('Missing memory breakdown')
                if v.get('max_abs_error')!=max(c['max_abs_error'] for c in checks) or v.get('max_rmse')!=max(c['rmse'] for c in checks):
                    raise ValueError('Validation summary is inconsistent with per-run errors')
                if abs(statistics.mean(samples)-t['mean_ms'])>1e-8*max(1,abs(t['mean_ms'])):raise ValueError('Timing mean inconsistent with samples')
            elif r['status'] not in ['oom','validation_failed','error']:
                raise ValueError(f'Incomplete or failed measurement: {r["status"]}')
            elif not all(key in r for key in ['exception_type','error_message','failure_stage']):
                raise ValueError('Failed cells must preserve their failure cause and stage')
            elif r['status']=='oom' and 'memory_limit_bytes' not in r:
                raise ValueError('OOM cells must record the configured memory limit')
    elif table in [4,5]:
        expected={(b,s) for b in p['budgets'] for s in p['estimator_seeds']}
        if {(r['requested_budget'],r['seed']) for r in rows}!=expected or len(rows)!=len(expected):raise ValueError('Incomplete sampling grid')
        if table==4:
            for r in rows:
                t=r.get('timing',{});samples=t.get('samples_ms',[]);checks=r.get('approximation_per_run',[])
                if len(samples)!=1 or len(checks)!=1 or t.get('complete') is not True:
                    raise ValueError('Each seed requires one timed approximation output')
                if any(not math.isfinite(x) or x<0 for x in samples):raise ValueError('Invalid approximation timing')
                if checks[0].get('seed')!=r['seed'] or t.get('sample_seeds')!=[r['seed']]:raise ValueError('Approximation seed mismatch')
                if any(c.get('outside_timing') is not True or c.get('status')!='passed' for c in checks):raise ValueError('Invalid approximation output record')
                if not math.isclose(statistics.mean(samples),r['runtime_ms'],rel_tol=1e-10):raise ValueError('Approximation mean mismatch')
                if not math.isclose(statistics.pstdev(samples),t['std_ms'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Approximation standard deviation mismatch')
    else:
        if sorted(r['d'] for r in rows)!=DIMENSIONS:raise ValueError('Incomplete generation sweep')
        for r in rows:
            if len(r['samples_ms'])!=10 or r['status']!='complete':raise ValueError('Incomplete generation timing')
            if [c.get('seed') for c in r.get('per_run',[])]!=p['seeds']:raise ValueError('Generation seed schedule mismatch')
            if abs(statistics.mean(r['samples_ms'])-r['mean_ms'])>1e-8*max(1,abs(r['mean_ms'])):raise ValueError('Generation timing mean mismatch')


def sci(value,latex=False):
    if value==0:return '0'
    a,b=f'{value:.2e}'.split('e')
    return f'\\({a}\\times10^{{{int(b)}}}\\)' if latex else f'{a}e{int(b):+d}'


def display(data,latex=False):
    table=data['table'];rows=data['records']
    def timing(r,div=1):
        if r['status']=='oom':return 'OOM'
        if r['status']!='ok':return 'FAIL'
        number=(str(round(r['timing']['mean_ms'])) if table==1 and r['method']=='direct' and r['d']>=14
                else f'{r["timing"]["mean_ms"]/div:.2f}')
        return number
    def mem(r):return timing(r) if r['status']!='ok' else f'{r["workspace_peak_bytes"]/2**20:.2f}'
    if table in [1,2,3]:
        coordinate={1:'d',2:'batch',3:'ell'}[table];lookup={(r[coordinate],r['method']):r for r in rows}
        if table==2:
            header=['Batch','Bounded ms/instance','Bounded MiB','Dense ms/instance','Dense MiB']
            values=[]
            for b in BATCHES:
                p=lookup[b,'gpu_fused_bounded'];q=lookup[b,'gpu_dense_ablation']
                values.append([str(b),timing(p,b),mem(p),timing(q,b),mem(q)])
        else:
            header=([r'\(d\)' if latex else 'd','CPU dense','GPU bounded','Direct FSI','FSII'] if table==1 else
                    [r'\(\ell\)' if latex else 'ell','FSII','CPU dense','GPU dense','GPU bounded'])
            values=[[str(v)]+[timing(lookup[v,m]) for m in PROTOCOLS[table]['methods']] for v in (DIMENSIONS if table==1 else range(2,7))]
    elif table==4:
        header=['Budget','Mean ms','Mean RMSE','Max. abs.','Mean displaced'];values=[]
        for budget in PROTOCOLS[4]['budgets']:
            group=[r for r in rows if r['requested_budget']==budget]
            values.append([f'{budget:,}',f'{statistics.mean(r["runtime_ms"] for r in group):.2f}',
                           sci(statistics.mean(r['rmse_nonempty'] for r in group),latex),
                           sci(max(r['max_abs_nonempty'] for r in group),latex),
                           f'{statistics.mean(r["top10_displaced"] for r in group):.2f}'])
    elif table==5:
        header=['Budget','Set matches','Order matches','Mean RMSE','Max. unique'];values=[]
        for budget in PROTOCOLS[5]['budgets']:
            group=[r for r in rows if r['requested_budget']==budget]
            values.append([f'{budget:,}',f'{sum(r["top10_set_match"] for r in group)}/10',f'{sum(r["top10_order_match"] for r in group)}/10',
                           sci(statistics.mean(r['rmse_nonempty'] for r in group),latex),f'{max(r["observed_unique_coalitions"] for r in group):,}'])
    else:
        header=['d','Generation time (ms)'];values=[[str(r['d']),f'{r["mean_ms"]:,.2f}'] for r in sorted(rows,key=lambda r:r['d'])]
    return header,values


def export(data,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);table=data['table'];base=directory/f'table{table}'
    head,rows=display(data);lh,lr=display(data,True)
    notes=f'Origin: {data["origin"]}. Protocol SHA256: {data["protocol_sha256"]}. OOM denotes failure to complete under the configured host or device memory constraints, including the host-memory cap used for baseline runs. FAIL denotes validation failure or another recorded error; no runtime is reported for those cells.'
    markdown=f'# Table {table}: {TITLES[table]}\n\n'+notes+'\n\n| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'
    markdown+=''.join('| '+' | '.join(r)+' |\n' for r in rows)
    base.with_suffix('.md').write_text(markdown)
    with base.with_suffix('.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(head);writer.writerows(rows)
    tex='% '+notes+'\n'+r'\begin{table}[htbp]'+'\n'+r'\centering'+'\n'+r'\caption{'+TITLES[table]+'}\n'+r'\begin{tabular}{'+'r'*len(head)+'}\n'+r'\toprule'+'\n'
    tex+=' & '.join(lh)+r' \\'+'\n'+r'\midrule'+'\n'+''.join(' & '.join(r)+r' \\'+'\n' for r in lr)+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table}'+'\n'
    base.with_suffix('.tex').write_text(tex);write_json(base.with_suffix('.json'),data)
    print(markdown,flush=True)


def run_table(table,output=None,refresh=False,published_only=False,root=ROOT,measure=None):
    output=Path(output) if output is not None else root/'paper-results/mixed-model'
    cache=output/'cache'/f'table{table}.json'
    if refresh and published_only:raise ValueError('refresh and published_only are mutually exclusive')
    if cache.is_file() and not refresh and not published_only:
        data=json.loads(cache.read_text())
        if data.get('protocol')!=PROTOCOLS[table] or data.get('execution_fingerprint')!=execution_fingerprint():
            return run_table(table,output,True,False,root,measure)
        validate(data,table);route='local_cache'
    else:
        data=None if refresh else published_record(table,root)
        if data is None:
            if published_only:raise FileNotFoundError('Published evidence is incomplete')
            if measure is None:
                from paper_measure import measure_table
                measure=measure_table
            print('PAPER_TABLE_CACHE_MISS '+json.dumps({'table':table,'protocol':PROTOCOLS[table]}),flush=True)
            data=measure(table,root);validate(data,table);route='measured'
        else:route='published_evidence'
        if cache.exists():
            prior=json.loads(cache.read_text());write_json(output/'history'/f'table{table}-{prior["content_sha256"]}.json',prior)
        write_json(cache,data)
    export(data,output)
    print('PAPER_TABLE_RESULT '+json.dumps({'table':table,'route':route,'origin':data['origin'],'rows':len(data['records']),'content_sha256':data['content_sha256']}),flush=True)
    return data


def main(table=None):
    ap=argparse.ArgumentParser(description=__doc__)
    if table is None:ap.add_argument('table',type=int,choices=list(PROTOCOLS))
    ap.add_argument('--output',type=Path,default=ROOT/'paper-results/mixed-model');ap.add_argument('--refresh',action='store_true')
    ap.add_argument('--published',action='store_true',help='Replay bundled mixed-model measurements even if a newer local cache exists')
    args=ap.parse_args();run_table(table or args.table,args.output,args.refresh,args.published)


if __name__=='__main__':main()
