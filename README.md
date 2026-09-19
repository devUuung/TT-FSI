# TT-FSI: Scalable Faithful Shapley Interactions via Tensor-Train

Reproduction code for five active numerical experiments corresponding to
manuscript Tables 2–6, plus the California case and figures. CPU dense is the
timed CPU baseline. The default workflow remeasures all five experiments and
the California case using saved checkpoints and coalition values; model training
remains disabled.

## 1. Installation

The pinned reproduction environment was verified on Linux with Python 3.12.
Run from the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-verified-linux-py312.txt
python -m pip install --no-deps -e .
```

On a CPU-only host, omit the `cupy-cuda12x[ctk]` requirement. Replaying saved
measurements and redrawing saved figures do not require a GPU. Remeasuring GPU
methods and the California CPU/GPU comparison requires CuPy and a compatible
CUDA GPU. Fresh XGBoost training also requires CUDA under the committed settings.

## 2. Run the FP32 paper reproduction

```sh
bash scripts/run.sh
```

The script installs the pinned environment, checks the CUDA device, runs the
CPU/GPU regression tests, verifies saved game inputs and remeasures all five
active numerical experiments and the California case, then regenerates the
figures. This takes hours because the literal direct baseline is retained. Outputs are written to `paper-results/seeds-0-9/`.
Note the two result locations: `outputs/seeds-0-9/` holds the committed reference results that
the manuscript tables were taken from and is never overwritten, while `paper-results/`
(git-ignored) receives every fresh run, so a new run can be compared against the reference
side by side.

Set `refresh_tables` to an empty list and `refresh_california` to false to
reuse a current, validated local result. Old local results are archived
before replacement.

Each result fingerprint includes all method and experiment Python sources,
the protocol version and dependency versions. Tables export Markdown, CSV,
LaTeX and JSON; the final manifest ties the table checksums together. Figures
require California scores with a matching fingerprint and file checksums.

## 3. Run new measurements

```sh
python experiments/table4_approximation.py --refresh --output paper-results/seeds-0-9
python experiments/paper_california.py --refresh --output paper-results/seeds-0-9/california
```

For a configured run, select script IDs in `tables` and `refresh_tables` in
`configs/paper_reproduction.json`, then run `python experiments/run_paper.py`.
The defaults already refresh IDs 1, 2, 3, 4 and 6, set `refresh_california` to
true, and regenerate the figures. These options bypass saved measurement results;
they do not retrain models or force regeneration of cached coalition values.

**Measuring manuscript Table 2 (script `table1_runtime.py`) takes hours:** it includes ten exhaustive direct
FSI calls plus warm-ups at d=20. Remeasurement can be triggered by `--refresh`,
by `refresh_tables`, or by missing local and bundled results.

## 4. Model checkpoints and coalition values

### Train or verify models

Committed checkpoints make training optional. The training configuration covers
24 combinations (six datasets and four model families); the paper uses nine.

```sh
python experiments/train_models.py        # reuse and verify, or train into an empty destination
python experiments/verify_checkpoints.py  # reload and compare held-out predictions
```

### Generate or verify coalition values

```sh
python experiments/prepare_model_games.py   # nine games under outputs/games
```

Each game explains the saved first test instance against the saved first 100
training rows. Feature i is coalition mask bit i. For coalition S, replace the
background columns in S with the instance's corresponding values, then compute:

```text
raw_value[S] = mean(model.predict(hybrid_background_S))
value[S] = raw_value[S] - raw_value[0]
```

All 2**d coalitions are enumerated and saved as FP64 in an NPZ, with a JSON
identity based on the checkpoint manifest, prepared data, generator source and
chunk size. A valid cache hit does not call the generator. The preparation script
independently recomputes selected coalitions, checks exact reload equality, and
checks that shared (dataset, model) pairs use identical values. Spot-check
tolerances are 1e-12 for tree models and 1e-6 for the MLP, whose float32 matrix
multiplication can vary with batch shape; observed errors are recorded.

### Run the configured stages together

`bash scripts/run.sh` builds `.venv-reproduction` using `python3` and runs the
stages enabled in `configs/execution.json`. Ensure `python3` is Python 3.12 to
match the verified environment. The defaults prepare or verify paper games,
remeasure all five active numerical experiments and the California case, and
regenerate the figures. Training and full checkpoint verification are disabled.

## 5. Experiment inputs and measurement protocol

The dimension determines the dataset; the experiment determines the model family.
Dimension sweeps use the same model family across datasets, not the same fitted
model or coalition value function.

| Manuscript item | Script | Dimensions | Model family |
|---|---|---|---|
| Table 2: operator runtime | `table1_runtime.py` | 8, 10, 11, 14, 16, 20 | XGBoost |
| Table 3: batching and workspace | `table2_batch.py` | 20 | LightGBM |
| Table 4: order sweep | `table3_order.py` | 14 | Decision tree |
| Table 5: multi-seed approximation | `table4_approximation.py` | 14 | MLP |
| Table 6: value generation | `table6_generation.py` | 8, 10, 11, 14, 16, 20 | XGBoost |
| California case and figures | `paper_california.py`, `plot_figures.py` | 8 | XGBoost |

The separate multi-seed ranking experiment (`table5_stability.py`) is retired
from the paper and excluded from the default workflow. It remains only for
historical compatibility. Theoretical manuscript Table 1 is not a numerical
experiment. Script IDs are retained, so they do not match manuscript numbering.

| Dimension | Dataset | Prediction |
|---:|---|---|
| 8 | California Housing | Regression value |
| 10 | Diabetes | Regression value |
| 11 | COMPAS | Positive-class probability |
| 14 | Adult | Positive-class probability |
| 16 | Bank Marketing | Positive-class probability |
| 20 | German Credit | Positive-class probability |

## Repository layout

```text
tt_fsi/       CPU dense TT baseline and internal prefix-pruned reference, CuPy GPU schedules,
              and brute-force references
experiments/  training, game generation, measurement and figure scripts
configs/      execution settings and fixed paper input protocols
outputs/      committed checkpoints, coalition values and reference measurements (read-only)
paper-results/ fresh measurement runs (created by the scripts, git-ignored)
scripts/      environment setup and configured end-to-end execution
```

## License

This repository is released under the MIT License; see [LICENSE](LICENSE).
