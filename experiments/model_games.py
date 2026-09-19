"""Checkpoint-backed model games shared by every active data experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from checkpoints import ROOT, digest, load_checkpoint


def settings():
    return json.loads((ROOT/'configs/model_experiments.json').read_text())


def generate_values(predictor, instance, background, chunk_coalitions=256):
    """Full centered interventional table; original feature i is mask bit i."""
    d = len(instance)
    if not 1 <= d <= 20 or background.ndim != 2 or background.shape[1] != d or not len(background):
        raise ValueError('Expected instance (d,) and nonempty background (B,d), 1 <= d <= 20')
    if chunk_coalitions < 1: raise ValueError('chunk_coalitions must be positive')
    values = np.empty(1 << d, dtype=np.float64)
    for start in range(0, len(values), chunk_coalitions):
        stop = min(start+chunk_coalitions, len(values))
        mask = ((np.arange(start,stop,dtype=np.uint64)[:,None] >> np.arange(d,dtype=np.uint64)) & 1).astype(bool)
        hybrids = np.where(mask[:,None,:],instance[None,None,:],background[None,:,:])
        values[start:stop] = predictor.predict(hybrids.reshape(-1,d)).reshape(stop-start,len(background)).mean(axis=1)
    values -= values[0]
    if not np.isfinite(values).all(): raise ValueError('Nonfinite generated value table')
    return values


def model_for(experiment):
    """Committed family for one experiment; a dimension sweep keeps it at every d."""
    assignment = settings()['model_by_experiment']
    if experiment not in assignment: raise ValueError(f'No model family assigned to experiment {experiment}')
    return assignment[experiment]


def dataset_for(d):
    dataset = settings()['dataset_by_dimension'].get(str(d))
    if dataset is None: raise ValueError(f'No saved dataset game for d={d}; synthetic fallback is disabled')
    return dataset


def checkpoint_game(dataset, model, *, use_cache=True):
    c = settings()
    predictor = load_checkpoint(dataset, model, ROOT/c['checkpoint_directory'])
    instance, background = predictor.reference_game()
    identity = {'dataset':dataset, 'model':model, 'checkpoint_manifest_sha256':digest(predictor.path/'manifest.json'),
                'generator_source_sha256':digest(Path(__file__)),
                'prepared_sha256':digest(predictor.base/dataset/'prepared.npz'), 'definition':'centered interventional; positive probability for classification',
                'd':len(instance), 'background_rows':len(background), 'chunk_coalitions':c['chunk_coalitions']}
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=ROOT/c['cache_directory']/dataset/model; path=folder/(key+'.npz'); meta=folder/(key+'.json')
    if use_cache and path.exists() and meta.exists():
        info=json.loads(meta.read_text())
        if info['identity'] != identity or info['npz_sha256'] != digest(path): raise ValueError('Cached game identity/checksum mismatch')
        with np.load(path,allow_pickle=False) as data: values=data['values'].copy()
    else:
        t=time.perf_counter();values=generate_values(predictor,instance,background,c['chunk_coalitions'])
        info={'identity':identity,'generation_seconds':time.perf_counter()-t}
        folder.mkdir(parents=True,exist_ok=True)
        import tempfile
        with tempfile.NamedTemporaryFile(dir=folder,suffix='.npz',delete=False) as handle:
            temporary=Path(handle.name)
        try:
            np.savez_compressed(temporary,values=values,instance=instance,background=background)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        info['npz_sha256']=digest(path)
        with tempfile.NamedTemporaryFile(mode='w',dir=folder,suffix='.json',delete=False) as handle:
            json.dump(info,handle,indent=2);temporary_meta=Path(handle.name)
        temporary_meta.replace(meta)
    info = {**info, 'd':len(instance), 'source':'saved_model_checkpoint',
            'sha256':hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()}
    return values, info


def experiment_input(d, experiment):
    """The game one experiment uses at dimension d: its family, that d's dataset."""
    return checkpoint_game(dataset_for(d), model_for(experiment))
