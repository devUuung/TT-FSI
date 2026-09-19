#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --without-pip .venv-reproduction
.venv-reproduction/bin/python - <<'PYBOOT'
import hashlib, pathlib, urllib.request
path = pathlib.Path('.venv-reproduction/pip.pyz')
url = 'https://bootstrap.pypa.io/pip/pip.pyz'
with urllib.request.urlopen(url, timeout=60) as response:
    path.write_bytes(response.read())
print('PIP_BOOTSTRAP', url, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
PYBOOT
.venv-reproduction/bin/python .venv-reproduction/pip.pyz install pip==25.3
.venv-reproduction/bin/python -m pip install --disable-pip-version-check -r requirements.txt -c constraints-verified-linux-py312.txt
.venv-reproduction/bin/python -m pip install --disable-pip-version-check --no-deps -e .
.venv-reproduction/bin/python -m pip check
.venv-reproduction/bin/python -m pip freeze
.venv-reproduction/bin/python - <<'PYGPU'
import json, cupy as cp
properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
name = properties['name']
print('GPU_VALIDATION_DEVICE', json.dumps({'name': name.decode() if isinstance(name, bytes) else name,
      'device': cp.cuda.Device().id, 'memory_bytes': properties['totalGlobalMem'],
      'runtime': cp.cuda.runtime.runtimeGetVersion(), 'driver': cp.cuda.runtime.driverGetVersion()}), flush=True)
PYGPU
.venv-reproduction/bin/python -m pytest -q tests
.venv-reproduction/bin/python -u experiments/run_revision.py
