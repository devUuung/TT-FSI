"""Same-protocol measurements used only when a requested paper table has no cache."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import numpy as np

from paper_tables import ROOT,PROTOCOLS,DIMENSIONS,BATCHES,envelope,write_json
from numerical_protocol import FP32_TOLERANCES, TIMING_PROTOCOL


def paper_input(d,experiment):
    """Values at dimension d for one experiment, using that experiment's family."""
    from model_games import experiment_input,model_for
    from paper_tables import GAME_CONFIG
    from checkpoints import ROOT as checkpoint_root
    dataset=GAME_CONFIG['datasets'][str(d)];family=GAME_CONFIG['models_by_experiment'][experiment]
    if model_for(experiment)!=family:
        raise ValueError('Assigned model family differs from the recorded input protocol')
    if d not in GAME_CONFIG['dimensions_by_experiment'][experiment]:
        raise ValueError(f'{experiment} does not declare dimension {d}')
    expected=GAME_CONFIG['checkpoints'][f'{dataset}/{family}']
    manifest=json.loads((checkpoint_root/'outputs/checkpoints'/dataset/family/'manifest.json').read_text())
    if manifest['files']!=expected['files'] or manifest['dataset_files']!=expected['dataset_files']:
        raise ValueError('Assigned checkpoint changed; update the input protocol before measuring')
    values,info=experiment_input(d,experiment)
    if info['identity']['dataset']!=dataset or info['identity']['model']!=family:
        raise ValueError('Paper experiments require the assigned checkpoint')
    return values,info


def metadata():
    versions={}
    for name in ['numpy','scipy','scikit-learn','shapiq','cupy-cuda12x']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'platform':platform.platform(),'python':platform.python_version(),'versions':versions,
            'measurement_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'historical_measurements_overwritten':False}


def measure_one(job):
    from measurement import (timed,timing_summary,correctness,workspace_peak_bytes,
                             exact_shapiq,validation_summary,ValidationError)
    from tt_fsi.core.generic_operator import compute_interaction_index
    from tt_fsi.naive.fsi_naive import fsi_naive_tensor
    d,ell,batch,method=job['d'],job['ell'],job['batch'],job['method']
    row={**job,'status':'running','timing_protocol':dict(TIMING_PROTOCOL)}
    samples=[];checks=[];cp=None;out=x=fn=operator=output=None;stage='input_load'
    try:
        values,info=paper_input(d,job['experiment']);row['input_sha256']=info['sha256']
        values=np.ascontiguousarray(values,dtype=np.float32)
        row['operator_input_sha256']=hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()
        stage='reference'
        reference=compute_interaction_index(values.reshape((2,)*d),ell,'fsi').reshape(-1)
        sync=lambda:None
        if method.startswith('gpu_'):
            import cupy as cp
            from tt_fsi.core.gpu_tt import prepare_fsi_gpu
            cp.get_default_memory_pool().free_all_blocks()
            stage='input_upload'
            x=cp.broadcast_to(cp.asarray(values)[None,:],(batch,values.size)).copy()
            sync=cp.cuda.get_current_stream().synchronize;sync()
            stage='operator_preparation'
            operator=prepare_fsi_gpu(x,ell,'bounded' if method=='gpu_fused_bounded' else 'dense')
            stage='output_allocation'
            output=operator.allocate_output()
            row.update(operator.memory_breakdown())
            fn=lambda:operator.run(out=output)
        elif method=='cpu_dense':
            from tt_fsi.core.fsi_efficient import fsi_precontract_efficient
            fn=lambda:fsi_precontract_efficient(values.reshape((2,)*d),ell)
        elif method=='cpu_prefix_pruned':fn=lambda:compute_interaction_index(values.reshape((2,)*d),ell,'fsi')
        elif method=='direct':fn=lambda:fsi_naive_tensor(values.reshape((2,)*d),ell)
        else:fn=lambda:exact_shapiq(values,d,ell)
        def check(output):
            nonlocal stage
            stage='validation'
            result=correctness(output,reference,d,ell,method,cp,FP32_TOLERANCES,batch=batch)
            if result['status']=='passed' or (method=='direct' and 'error' not in result):stage='operator'
            return result
        stage='operator'
        timed(fn,sync,samples,TIMING_PROTOCOL["warmups"],TIMING_PROTOCOL["runs"],check=check,checks=checks,require_allclose=method!='direct')
        if method.startswith('gpu_'):
            stage='memory_measurement'
            row['workspace_peak_bytes']=workspace_peak_bytes(cp,fn,row['reusable_workspace_bytes'])
        row['status']='ok'
        row['validation_policy']='diagnostic_only' if method=='direct' else 'require_allclose'
    except ValidationError as exc:
        row.update(status='validation_failed',exception_type=type(exc).__name__,
                   error_message=str(exc),failure_stage=stage)
    except MemoryError as exc:
        row.update(status='oom',exception_type=type(exc).__name__,error_message=str(exc),
                   memory_limit_bytes=job.get('memory_limit_bytes'),failure_stage=stage)
    except Exception as exc:
        status='oom' if cp is not None and isinstance(exc,cp.cuda.memory.OutOfMemoryError) else 'error'
        row.update(status=status,exception_type=type(exc).__name__,error_message=str(exc),
                   memory_limit_bytes=job.get('memory_limit_bytes'),failure_stage=stage)
    finally:
        row['timing']=timing_summary(samples)
        row['validation']=validation_summary(checks)
        out=fn=x=operator=output=None
        if cp is not None:cp.get_default_memory_pool().free_all_blocks()
    return row


def bounded_worker(job):
    """Protect the host; preserve the cap and actual exception in the raw result."""
    import psutil
    cap=int(min(psutil.virtual_memory().available*.9,psutil.virtual_memory().total*.9))
    job={**job,'memory_limit_bytes':cap}
    with tempfile.TemporaryDirectory(prefix='tt-fsi-fsii-') as temp:
        out=Path(temp)/'result.json'
        result=subprocess.run([sys.executable,str(Path(__file__)),'--worker',json.dumps(job),'--result',str(out)])
        if result.returncode!=0:
            return {**job,'status':'error','exception_type':'WorkerProcessExit',
                    'error_message':f'FSII worker exited {result.returncode}',
                    'failure_stage':'worker_process','returncode':result.returncode}
        return json.loads(out.read_text())


def measure_operator_table(table):
    jobs=[]
    if table==1:
        jobs=[{'section':'table1','experiment':'table1','d':d,'ell':3,'batch':1,'method':m} for d in DIMENSIONS for m in PROTOCOLS[1]['methods']]
    elif table==2:
        jobs=[{'section':'batch_memory','experiment':'table2','d':20,'ell':3,'batch':b,'method':m} for b in BATCHES for m in PROTOCOLS[2]['methods']]
    else:
        jobs=[{'section':'table3','experiment':'table3','d':14,'ell':ell,'batch':1,'method':m} for ell in PROTOCOLS[3]['orders'] for m in PROTOCOLS[3]['methods']]
    rows=[]
    for job in jobs:
        print('PAPER_CELL_START '+json.dumps(job),flush=True)
        row=bounded_worker(job) if job['method']=='shapiq_exact' and job['d']>14 else measure_one(job)
        rows.append(row);print('PAPER_CELL_RESULT '+json.dumps(row),flush=True)
    return rows


def measure_sampling(table):
    import shapiq
    p=PROTOCOLS[table];d=p['d'];ell=p['ell']
    values,input_info=paper_input(d,f'table{table}');bits=1<<np.arange(d)
    def game(coalitions):return values[np.asarray(coalitions,dtype=np.int64)@bits]
    exact=shapiq.ExactComputer(n_players=d,game=game)(index='FSII',order=ell)
    keys=sorted((k for k in exact.interaction_lookup if k),key=lambda k:(len(k),k))
    truth=np.array([exact.values[exact.interaction_lookup[k]] for k in keys]);order=np.argsort(-np.abs(truth),kind='stable')[:10]
    rows=[]
    from measurement import timed, timing_summary
    for budget in p['budgets']:
        count=0;seen=set();samples=[];checks=[]
        def tracked(coalitions):
            nonlocal count
            masks=np.asarray(coalitions,dtype=np.int64)@bits;count+=len(masks);seen.update(masks.tolist())
            return values[masks]
        def invoke(seed):
            nonlocal count
            count=0;seen.clear()
            return shapiq.RegressionFSII(n=d,max_order=ell,random_state=seed).approximate(budget=budget,game=tracked)
        def assess(approx):
            prediction=np.array([approx.values[approx.interaction_lookup[k]] for k in keys])
            if prediction.shape!=truth.shape or not np.all(np.isfinite(prediction)):
                raise ValueError('Invalid approximation output')
            rank=np.argsort(-np.abs(prediction),kind='stable')[:10]
            overlap=len(set(order.tolist())&set(rank.tolist()))
            return {'status':'passed','outside_timing':True,
                    'observed_evaluations':count,'observed_unique_coalitions':len(seen),
                    'top10_overlap':overlap,'top10_displaced':10-overlap,'top10_set_match':overlap==10,
                    'top10_order_match':bool(np.array_equal(rank,order)),
                    'rmse_nonempty':float(np.sqrt(np.mean((prediction-truth)**2))),
                    'max_abs_nonempty':float(np.max(np.abs(prediction-truth)))}
        timed(invoke,lambda:None,samples,p['warmups'],p['runs'],check=assess,checks=checks,pass_seed=True)
        for elapsed,check in zip(samples,checks):
            seed=check['seed']
            row={'input_sha256':input_info['sha256'],'requested_budget':budget,'seed':seed,
                 'runtime_ms':elapsed,'timing':timing_summary([elapsed],required=1,seeds=[seed]),
                 'timing_protocol':dict(TIMING_PROTOCOL),'approximation_per_run':[check],
                 **{k:v for k,v in check.items() if k not in ['status','outside_timing']}}
            rows.append(row);print('PAPER_SAMPLING_RESULT '+json.dumps(row),flush=True)
    return rows


def measure_generation(root):
    from model_games import settings,generate_values
    from checkpoints import load_checkpoint
    config=settings();rows=[];family=config['model_by_experiment']['table6']
    for d in DIMENSIONS:
        saved,info=paper_input(d,'table6')
        dataset=config['dataset_by_dimension'][str(d)]
        model=load_checkpoint(dataset,family);instance,background=model.reference_game()
        from measurement import timed, timing_summary
        samples=[];checks=[];hashes=[]
        def invoke():
            return generate_values(model,instance,background,256)
        def assess(values):
            np.testing.assert_array_equal(values,saved)
            digest=hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
            hashes.append(digest)
            print('PAPER_GENERATION_SAMPLE '+json.dumps({'d':d,'dataset':dataset,'model':family,'repeat':len(hashes)-1,'ms':samples[-1],'input_sha256':digest}),flush=True)
            return {'status':'passed','outside_timing':True,'input_sha256':digest}
        timed(invoke,lambda:None,samples,TIMING_PROTOCOL['warmups'],TIMING_PROTOCOL['runs'],check=assess,checks=checks)
        summary=timing_summary(samples)
        rows.append({'d':d,'dataset':dataset,'model':family,'status':'complete','samples_ms':samples,
                     'mean_ms':summary['mean_ms'],'std_ms':summary['std_ms'],'value_hashes':hashes,
                     'timing_protocol':dict(TIMING_PROTOCOL),'per_run':checks,
                     'input_sha256':info['sha256'],'input':info})
    return rows


def measure_table(table,root=ROOT):
    records=measure_operator_table(table) if table<=3 else measure_sampling(table) if table<=5 else measure_generation(root)
    provenance=metadata()
    from tt_fsi.core.generic_operator import build_generic_sparse_cores,FSI_SPEC
    coordinates=sorted({(row['d'],row['ell']) for row in records if 'd' in row and 'ell' in row})
    provenance['sparse_core_storage']={}
    for d,ell in coordinates:
        cores=build_generic_sparse_cores(d,ell,FSI_SPEC,precontract=True)
        provenance['sparse_core_storage'][f'{d}/{ell}']={
            'nonzeros':sum(len(c.coefficients) for c in cores),
            'array_bytes':sum(c.nbytes for c in cores),'dtype':'float32',
            'includes':'CPU transition keys and coefficients; excludes Python object overhead'}
    print('PAPER_PROVENANCE '+json.dumps(provenance),flush=True)
    experiment=f'table{table}'
    provenance['model']=PROTOCOLS[table]['model']
    provenance['inputs']={str(d):paper_input(d,experiment)[1] for d in (DIMENSIONS if table in [1,6] else [20] if table==2 else [14])}
    return envelope(table,records,'measured',provenance)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',required=True);ap.add_argument('--result',type=Path,required=True);args=ap.parse_args()
    job=json.loads(args.worker)
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(job['memory_limit_bytes'],job['memory_limit_bytes']))
    write_json(args.result,measure_one(job))
