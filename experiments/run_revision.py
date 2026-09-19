"""Run the reproduction stages enabled in configs/execution.json."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
config=json.loads((root/'configs/execution.json').read_text())
print('EXECUTION_CONFIG '+json.dumps(config),flush=True)
for key,script in [('run_model_training','train_models.py'),
                   ('run_checkpoint_verification','verify_checkpoints.py'),
                   ('run_prepare_model_games','prepare_model_games.py'),
                   ('run_paper_reproduction','run_paper.py')]:
    if config.get(key,False):subprocess.run([sys.executable,'-u',str(root/'experiments'/script)],check=True)
