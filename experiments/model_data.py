"""Explicit six-dataset recipe. Preprocessing is fitted on training rows only."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

DIMENSIONS = {"california": 8, "diabetes": 10, "compas": 11,
              "adult": 14, "bank": 16, "german": 20}
ADULT = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
         "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
         "hours_per_week", "native_country", "income"]
GERMAN = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
          "savings_status", "employment", "installment_commitment", "personal_status",
          "other_parties", "residence_since", "property_magnitude", "age", "other_payment_plans",
          "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker", "class"]
COMPAS = ["age", "sex", "race", "juv_fel_count", "juv_misd_count", "juv_other_count",
          "priors_count", "c_charge_degree", "c_charge_desc", "days_b_screening_arrest", "length_of_stay"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / hashlib.sha256(url.encode()).hexdigest()
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "TT-FSI-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        temp = path.with_suffix('.tmp')
        temp.write_bytes(data)
        temp.replace(path)
    return path.read_bytes(), {"url": url, "sha256": sha256(path)}


def load_dataset(name, cache):
    from sklearn.datasets import fetch_california_housing, load_diabetes
    sources = []
    if name in ("california", "diabetes"):
        ds = (fetch_california_housing(data_home=str(cache), as_frame=True) if name == 'california'
              else load_diabetes(as_frame=True, scaled=False))
        X, y = ds.data.copy(), ds.target.to_numpy(dtype=np.float64)
        sources = [{"provider": "sklearn.datasets." + ('fetch_california_housing' if name == 'california' else 'load_diabetes(scaled=False)')}]
        task = "regression"
    elif name == 'compas':
        raw, source = download('https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv', cache)
        sources.append(source)
        df = pd.read_csv(io.BytesIO(raw))
        # ProPublica's analysis cohort; no risk scores or outcome-derived features.
        df = df.loc[df.days_b_screening_arrest.between(-30, 30) & (df.is_recid != -1)
                    & (df.c_charge_degree != 'O') & (df.score_text != 'N/A')].copy()
        df['length_of_stay'] = (pd.to_datetime(df.c_jail_out) - pd.to_datetime(df.c_jail_in)).dt.total_seconds() / 86400
        X, y, task = df[COMPAS].reset_index(drop=True), df.two_year_recid.to_numpy(dtype=np.int64), 'classification'
    elif name == 'adult':
        frames = []
        for filename, skip in [('adult.data', 0), ('adult.test', 1)]:
            raw, source = download('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/' + filename, cache)
            sources.append(source)
            frames.append(pd.read_csv(io.BytesIO(raw), names=ADULT, skiprows=skip, skipinitialspace=True, na_values='?'))
        df = pd.concat(frames, ignore_index=True).dropna(subset=['income'])
        X = df.drop(columns='income').reset_index(drop=True)
        y, task = df.income.str.rstrip('.').eq('>50K').to_numpy(dtype=np.int64), 'classification'
    elif name == 'bank':
        raw, source = download('https://archive.ics.uci.edu/static/public/222/bank+marketing.zip', cache)
        sources.append(source)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if 'bank-full.csv' in z.namelist(): data = z.read('bank-full.csv')
            else:
                with zipfile.ZipFile(io.BytesIO(z.read('bank.zip'))) as inner: data = inner.read('bank-full.csv')
        df = pd.read_csv(io.BytesIO(data), sep=';')
        X, y, task = df.drop(columns='y'), df.y.eq('yes').to_numpy(dtype=np.int64), 'classification'
    elif name == 'german':
        raw, source = download('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', cache)
        sources.append(source)
        df = pd.read_csv(io.BytesIO(raw), sep=r'\s+', names=GERMAN)
        X, y, task = df.drop(columns='class'), df['class'].eq(2).to_numpy(dtype=np.int64), 'classification'
    else:
        raise ValueError(f'Unknown dataset {name!r}')
    assert X.shape[1] == DIMENSIONS[name] and len(X) == len(y)
    assert np.isfinite(y).all()
    # Fingerprint the parsed source in addition to each downloaded file.
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes() + y.tobytes()).hexdigest()
    return X, y, {"dataset": name, "task": task, "n_rows": len(X), "dimension": X.shape[1],
                  "feature_names": list(X.columns), "sources": sources, "parsed_data_sha256": fingerprint,
                  "positive_class": {"compas": "two_year_recid=1", "adult": ">50K", "bank": "yes", "german": "bad credit (class=2)"}.get(name)}


def fit_preprocessor(frame):
    columns = []
    for name in frame.columns:
        col = frame[name]
        if pd.api.types.is_numeric_dtype(col):
            median = float(col.median())
            if not np.isfinite(median): raise ValueError(f'No finite training values for {name}')
            columns.append({"name": name, "kind": "numeric", "median": median})
        else:
            categories = sorted(col.dropna().astype(str).unique().tolist())
            columns.append({"name": name, "kind": "categorical", "categories": categories,
                            "missing_or_unknown": -1})
    return {"schema_version": 1, "encoding": "train-only ordinal categories; one original feature per FSI player",
            "columns": columns}


def encode(frame, schema):
    expected = [c['name'] for c in schema['columns']]
    if list(frame.columns) != expected: raise ValueError('Feature names/order do not match checkpoint schema')
    out = np.empty((len(frame), len(expected)), dtype=np.float64)
    for j, col in enumerate(schema['columns']):
        s = frame[col['name']]
        if col['kind'] == 'numeric':
            out[:, j] = pd.to_numeric(s, errors='raise').fillna(col['median']).to_numpy(dtype=float)
        else:
            lookup = {v: i for i, v in enumerate(col['categories'])}
            out[:, j] = s.map(lambda v: -1 if pd.isna(v) else lookup.get(str(v), -1)).to_numpy(dtype=float)
    if not np.isfinite(out).all(): raise ValueError('Nonfinite encoded feature')
    return out
