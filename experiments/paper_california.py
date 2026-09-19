"""Cache-first California validation case supplying the paper's Figures 2 and 3."""
import argparse
import json
from pathlib import Path
import shutil
import time
from paper_tables import ROOT,file_hash,write_json,GAME_HASH,execution_fingerprint
from numerical_protocol import FP32_TOLERANCES


def run(output=None,refresh=False,root=ROOT):
    output=Path(output) if output is not None else root/'paper-results/mixed-model/california'
    manifest=output/'cache.json'
    if manifest.exists() and not refresh:
        record=json.loads(manifest.read_text())
        if record.get('execution_fingerprint')==execution_fingerprint() and record.get('game_config_sha256')==GAME_HASH and all((output/name).is_file() for name in record['files']):
            if any(file_hash(output/name)!=sha for name,sha in record['files'].items()):raise ValueError('California cache checksum mismatch')
            print('PAPER_CALIFORNIA_RESULT '+json.dumps(record),flush=True);return record
    bundled=root/'outputs/mixed-model-paper/california/cache.json'
    if not refresh and bundled.is_file() and json.loads(bundled.read_text()).get('execution_fingerprint')==execution_fingerprint():
        record=json.loads(bundled.read_text())
        if record.get('game_config_sha256')!=GAME_HASH:raise ValueError('Bundled California game protocol mismatch')
        if all((bundled.parent/name).is_file() for name in record['files']):
            for name,sha in record['files'].items():
                if file_hash(bundled.parent/name)!=sha:raise ValueError('Bundled California checksum mismatch')
            output.mkdir(parents=True,exist_ok=True)
            for name in record['files']:shutil.copy2(bundled.parent/name,output/name)
            write_json(manifest,record)
            print('PAPER_CALIFORNIA_RESULT '+json.dumps(record),flush=True);return record
    import numpy as np
    from model_games import checkpoint_game,model_for
    from checkpoints import load_checkpoint
    family=model_for('california')
    model=load_checkpoint('california',family)
    instance,background=model.reference_game()
    values,input_info=checkpoint_game('california',family)
    values=np.ascontiguousarray(values,dtype=np.float32)
    data={'instance':instance,'background':background,'values':values}
    from tt_fsi import fsi
    from tt_fsi.naive.fsi_naive import fsi_naive
    import shapiq
    import cupy as cp
    from tt_fsi.core.gpu_tt import fsi_tt_gpu_stratified
    elapsed=input_info['generation_seconds']*1000
    output.mkdir(parents=True,exist_ok=True);data['values']=values;np.savez_compressed(output/'game.npz',**data)
    def game(coalitions):return values[np.asarray(coalitions,dtype=np.int64)@(1<<np.arange(8))]
    results=[];x=cp.asarray(values[None,:])
    for ell in [1,2,3]:
        cpu=fsi(values,ell);brute=fsi_naive(values,ell)
        exact=shapiq.ExactComputer(n_players=8,game=game)(index='FSII',order=ell)
        masks=np.asarray([sum(1<<i for i in key) for key in exact.interaction_lookup]);positions=np.asarray(list(exact.interaction_lookup.values()))
        gpu=fsi_tt_gpu_stratified(x,ell,cp.float32);cp.cuda.Stream.null.synchronize()
        gi=cp.asnumpy(gpu['indices']).reshape(-1);gv=cp.asnumpy(gpu['values']).reshape(-1)
        np.testing.assert_allclose(cpu,brute,**FP32_TOLERANCES)
        np.testing.assert_allclose(cpu[masks],exact.values[positions],**FP32_TOLERANCES)
        np.testing.assert_allclose(gv,cpu[gi],**FP32_TOLERANCES)
        np.savez_compressed(output/f'scores-ell{ell}.npz',cpu=cpu,gpu_indices=gi,gpu_values=gv,shapiq_masks=masks,shapiq_values=exact.values[positions])
        results.append({'ell':ell,'max_abs_vs_brute':float(np.max(np.abs(cpu-brute))),
                        'max_abs_vs_shapiq':float(np.max(np.abs(cpu[masks]-exact.values[positions]))),
                        'max_abs_gpu_vs_cpu':float(np.max(np.abs(gv-cpu[gi])))})
    write_json(output/'summary.json',{'model':'saved '+family,'input':input_info,'dataset':'California Housing','value_generation_ms':elapsed,
                                     'game_sha256':__import__('hashlib').sha256(values.astype('<f8').tobytes()).hexdigest(),'results':results})
    origin='measured'
    files={p.name:file_hash(p) for p in output.iterdir() if p.name=='summary.json' or p.suffix=='.npz'}
    record={'execution_fingerprint':execution_fingerprint(),'validation':FP32_TOLERANCES,'dtype':'float32','case':'california_'+family,'game_config_sha256':GAME_HASH,'origin':origin,'model':'saved '+family,
            'files':files,'summary':json.loads((output/'summary.json').read_text())}
    write_json(manifest,record);print('PAPER_CALIFORNIA_RESULT '+json.dumps(record),flush=True);return record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'paper-results/mixed-model/california');ap.add_argument('--refresh',action='store_true')
    args=ap.parse_args();run(args.output,args.refresh)


if __name__=='__main__':main()
