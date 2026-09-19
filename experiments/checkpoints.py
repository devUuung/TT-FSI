"""Portable scalar predictors over saved, original-feature-space encodings.

Loading never trains, downloads data, or silently creates a checkpoint.
Classification returns the manifest's positive-class probability.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mlp_network(d, hidden):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]),
                         nn.ReLU(), nn.Linear(hidden[1], 1))


class CheckpointPredictor:
    def __init__(self, dataset, model, directory=None):
        self.base = Path(directory) if directory is not None else ROOT/'outputs/checkpoints'
        self.path = self.base/dataset/model
        self.manifest = json.loads((self.path/'manifest.json').read_text())
        if (self.manifest['dataset'], self.manifest['model']) != (dataset, model):
            raise ValueError('Checkpoint identity mismatch')
        for filename, expected in self.manifest['files'].items():
            if digest(self.path/filename) != expected: raise ValueError(f'Checkpoint checksum mismatch: {filename}')
        for filename, expected in self.manifest['dataset_files'].items():
            if digest(self.base/dataset/filename) != expected: raise ValueError(f'Dataset checksum mismatch: {filename}')
        self.schema = json.loads((self.base/dataset/'preprocessor.json').read_text())
        self.mean = np.asarray(self.schema['mean'])
        self.scale = np.asarray(self.schema['scale'])
        self.kind, self.task = model, self.manifest['task']
        if model == 'dt':
            import joblib
            self.model = joblib.load(self.path/'model.joblib')
        elif model == 'lgbm':
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(self.path/'model.txt'))
        elif model == 'xgb':
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(self.path/'model.ubj')
            self.model.set_param({'device': 'cpu', 'nthread': self.manifest['threads']})
        elif model == 'mlp':
            import torch
            torch.set_num_threads(self.manifest['threads'])
            self.model = mlp_network(len(self.mean), self.manifest['hidden_sizes'])
            self.model.load_state_dict(torch.load(self.path/'weights.pt', map_location='cpu', weights_only=True))
            self.model.eval()
        else: raise ValueError(f'Unknown model {model}')

    def predict(self, encoded):
        x = np.asarray(encoded, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Expected finite (n, original_feature_count) encoded input')
        x = np.asarray((x-self.mean)/self.scale, dtype=np.float32)
        if self.kind == 'dt':
            pred = self.model.predict(x) if self.task == 'regression' else self.model.predict_proba(x)[:, 1]
        elif self.kind == 'lgbm':
            pred = self.model.predict(x, num_threads=self.manifest['threads'])
        elif self.kind == 'xgb':
            import xgboost as xgb
            pred = self.model.predict(xgb.DMatrix(x, nthread=self.manifest['threads']))
        else:
            import torch
            with torch.no_grad():
                tensor = self.model(torch.from_numpy(x)).reshape(-1)
                if self.task == 'classification': tensor = torch.sigmoid(tensor)
                pred = tensor.numpy()
            if self.task == 'regression':
                pred = pred.astype(np.float64)*self.manifest['target_scale']+self.manifest['target_mean']
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != (len(x),) or not np.isfinite(pred).all(): raise ValueError('Invalid checkpoint prediction')
        return pred

    def predict_frame(self, frame):
        from model_data import encode
        return self.predict(encode(frame, self.schema))

    def reference_game(self):
        with np.load(self.base/self.manifest['dataset']/'prepared.npz', allow_pickle=False) as data:
            return data['instance'].copy(), data['background'].copy()


def load_checkpoint(dataset, model='dt', directory=None):
    return CheckpointPredictor(dataset, model, directory)
