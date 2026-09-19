"""Train each committed dataset/model recipe once and verify saved predictors."""
from __future__ import annotations
import copy
import importlib.metadata
import json
from pathlib import Path
import random
import tarfile
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from checkpoints import digest, load_checkpoint, mlp_network
from model_data import load_dataset, fit_preprocessor, encode

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def train_mlp(X, y, train, val, task, config, seed):
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = mlp_network(X.shape[1], config['hidden_sizes'])
    target_mean = float(y[train].mean()) if task == 'regression' else 0.
    target_scale = float(y[train].std()) if task == 'regression' else 1.
    if target_scale == 0: target_scale = 1.
    xt = torch.from_numpy(X)
    yt = torch.tensor((y-target_mean)/target_scale, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss() if task == 'regression' else torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, stale = float('inf'), None, None, 0
    trajectory = []
    for epoch in range(1, config['max_epochs']+1):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), config['batch_size']):
            idx = order[start:start+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xt[idx]).reshape(-1), yt[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad(): value = float(criterion(model(xt[val]).reshape(-1), yt[val]))
        if not np.isfinite(value): raise ValueError('Nonfinite MLP validation loss')
        trajectory.append({'epoch': epoch, 'validation_loss': value})
        if value < best-config['min_delta']:
            best, best_state, best_epoch, stale = value, copy.deepcopy(model.state_dict()), epoch, 0
        else: stale += 1
        if epoch % 25 == 0: print('MLP_PROGRESS '+json.dumps(trajectory[-1]), flush=True)
        if stale >= config['patience']: break
    model.load_state_dict(best_state)
    return model, {'target_mean': target_mean, 'target_scale': target_scale, 'hidden_sizes': config['hidden_sizes'],
                   'best_epoch': best_epoch, 'epochs_run': epoch, 'best_validation_loss': best, 'trajectory': trajectory}


def main():
    import torch
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    config = json.loads((ROOT/'configs/model_training.json').read_text())
    print('MODEL_TRAINING_CONFIG '+json.dumps(config), flush=True)
    dest = ROOT/config['output_directory']
    if (dest/'summary.json').exists():
        saved = json.loads((dest/'summary.json').read_text())
        if saved['config'] != config: raise ValueError('Existing checkpoints use another configuration; choose a new output directory')
        from verify_checkpoints import verify_all
        verify_all(dest)
        print('MODEL_TRAINING_REUSED '+json.dumps(saved), flush=True)
        return
    # Never overwrite a partial run or silently train during inference.
    if dest.exists() and any(dest.iterdir()): raise FileExistsError(f'Checkpoint destination is not empty: {dest}')
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['threads'])
    random.seed(config['seed']); np.random.seed(config['seed'])
    versions = {n: importlib.metadata.version(n) for n in ['numpy', 'scikit-learn', 'pandas', 'torch', 'lightgbm', 'xgboost', 'joblib']}
    summary = {'schema_version': 1, 'kind': 'new_saved_model_suite', 'config': config, 'versions': versions,
               'training_source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/'experiments/model_data.py', ROOT/'experiments/checkpoints.py']},
               'rows': [], 'status': 'running'}
    for dataset in config['datasets']:
        print('MODEL_DATASET_START '+dataset, flush=True)
        frame, y, data_info = load_dataset(dataset, ROOT/'outputs/download-cache')
        indices = np.arange(len(y)); stratify = y if data_info['task'] == 'classification' else None
        trainval, test = train_test_split(indices, test_size=config['test_fraction'], random_state=config['seed'], stratify=stratify)
        train, val = train_test_split(trainval, test_size=config['validation_fraction_of_training'], random_state=config['seed'],
                                      stratify=y[trainval] if stratify is not None else None)
        schema = fit_preprocessor(frame.iloc[train]); Xraw = encode(frame, schema)
        mean, scale = Xraw[train].mean(axis=0), Xraw[train].std(axis=0)
        scale[scale == 0] = 1.
        schema.update(mean=mean.tolist(), scale=scale.tolist(), fit_rows='training split only')
        X = np.asarray((Xraw-mean)/scale, dtype=np.float32)
        folder = dest/dataset; folder.mkdir()
        write_json(folder/'preprocessor.json', schema)
        data_info.update(split_seed=config['seed'], n_train=len(train), n_validation=len(val), n_test=len(test),
                         background_selection='first 100 training split indices', explained_row=int(test[0]))
        write_json(folder/'dataset.json', data_info)
        np.savez_compressed(folder/'prepared.npz', X=Xraw, y=y, train_indices=train, validation_indices=val, test_indices=test,
                            background=Xraw[train[:config['background_rows']]], instance=Xraw[test[0]])
        dataset_files = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
        for kind in config['models']:
            print('MODEL_FIT_START '+json.dumps({'dataset': dataset, 'model': kind}), flush=True)
            path = folder/kind; path.mkdir()
            task = data_info['task']; extra = {}; t = time.perf_counter()
            if kind == 'dt':
                cls = DecisionTreeRegressor if task == 'regression' else DecisionTreeClassifier
                model = cls(max_depth=config['tree_max_depth'], random_state=config['seed']).fit(X[train], y[train])
                expected = model.predict(X[test]) if task == 'regression' else model.predict_proba(X[test])[:,1]
                joblib.dump(model, path/'model.joblib', compress=3)
            elif kind == 'lgbm':
                cls = lgb.LGBMRegressor if task == 'regression' else lgb.LGBMClassifier
                model = cls(n_estimators=config['boosting_estimators'], max_depth=config['tree_max_depth'],
                            num_leaves=2**config['tree_max_depth'], random_state=config['seed'],
                            n_jobs=config['threads'], deterministic=True, force_col_wise=True, verbosity=-1)
                model.fit(X[train], y[train])
                expected = model.booster_.predict(X[test], num_threads=config['threads'])
                model.booster_.save_model(str(path/'model.txt'))
            elif kind == 'xgb':
                cls = xgb.XGBRegressor if task == 'regression' else xgb.XGBClassifier
                model = cls(n_estimators=config['boosting_estimators'], max_depth=config['tree_max_depth'],
                            random_state=config['seed'], n_jobs=config['threads'], tree_method='hist', device=config['xgb_device'])
                model.fit(X[train], y[train])
                fitted_config = json.loads(model.get_booster().save_config())
                actual_device = fitted_config['learner']['generic_param']['device']
                if config['xgb_device'].startswith('cuda') and not actual_device.startswith('cuda'):
                    raise RuntimeError('XGBoost silently fell back from requested CUDA training')
                extra['training_device'] = actual_device
                # Record native CPU inference, the portable consumer path, after GPU fitting.
                model.get_booster().set_param({'device': 'cpu'})
                expected = model.get_booster().predict(xgb.DMatrix(X[test], nthread=config['threads']))
                model.save_model(path/'model.ubj')
            else:
                model, extra = train_mlp(X, y, train, val, task, config['mlp'], config['seed'])
                with torch.no_grad():
                    expected = model(torch.from_numpy(X[test])).reshape(-1)
                    if task == 'classification': expected = torch.sigmoid(expected)
                    expected = expected.numpy().astype(np.float64)
                if task == 'regression': expected = expected*extra['target_scale']+extra['target_mean']
                torch.save(model.state_dict(), path/'weights.pt')
            fit_seconds = time.perf_counter()-t
            np.save(path/'test_predictions.npy', np.asarray(expected, dtype=np.float64))
            manifest = {'schema_version': 1, 'dataset': dataset, 'model': kind, 'task': task, 'threads': config['threads'],
                        'seed': config['seed'], 'versions': versions, 'dataset_files': dataset_files,
                        'files': {p.name: digest(p) for p in path.iterdir()}, **extra}
            write_json(path/'manifest.json', manifest)
            predictor = load_checkpoint(dataset, kind, dest)
            actual = predictor.predict(Xraw[test]); np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            if task == 'regression': metrics = {'rmse': float(np.sqrt(mean_squared_error(y[test],actual))), 'r2': float(r2_score(y[test],actual))}
            else: metrics = {'accuracy': float(accuracy_score(y[test],actual>=.5)), 'roc_auc': float(roc_auc_score(y[test],actual)),
                             'log_loss': float(log_loss(y[test],actual,labels=[0,1]))}
            row = {'dataset': dataset, 'model': kind, 'task': task, 'fit_seconds': fit_seconds,
                   'reload_max_abs_error': float(np.max(np.abs(actual-expected))), 'test_metrics': metrics,
                   'manifest_sha256': digest(path/'manifest.json')}
            summary['rows'].append(row)
            print('MODEL_CHECKPOINT_SAVED '+json.dumps(row), flush=True)
    summary['status'] = 'complete'; write_json(dest/'summary.json', summary)
    from verify_checkpoints import verify_all
    verify_all(dest)
    # One compact, checksummed export for transport back into the repository.
    archive = ROOT/'outputs/checkpoints.tar.gz'
    with tarfile.open(archive,'w:gz') as tar: tar.add(dest,arcname='checkpoints')
    print('MODEL_ARCHIVE '+json.dumps({'path':str(archive.relative_to(ROOT)), 'bytes':archive.stat().st_size, 'sha256':digest(archive)}),flush=True)
    print('MODEL_TRAINING_SUMMARY '+json.dumps(summary),flush=True)


if __name__ == '__main__': main()
