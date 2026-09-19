# Common paper timing protocol

All five active numerical tables use the same repetition policy: the same saved input within a cell, two untimed full-call warmups using seeds 0 and 1, and ten timed calls using seeds 0 through 9 respectively. We report the arithmetic mean. Raw records retain all samples and the population standard deviation. Checkpoints are reused; this seed policy does not retrain or alter the saved models.

For stochastic approximation, each measured call constructs a fresh RegressionFSII estimator with random_state equal to that call's seed (0 through 9). These ten calls measure both timing and estimator variability. Approximation errors and Top-10 displacement are evaluated outside timing for every output. Per-seed displacement is 10 minus the overlap with the exact Top-10 set; the table reports its mean across seeds. RMSE is averaged across seeds, and maximum absolute error is the worst over all seeds. Set/order matches remain in the raw records. The separate multi-seed ranking table is retired.

For value generation, each warmup and each timed call generates the complete coalition-value table. Loading and model fitting are excluded. Every timed result is checked against the saved table outside timing and then released.

CPU timings use the full-prefix dense precontracted implementation. GPU timings use prepared device-resident operators and preallocated output buffers, with synchronization around timing. Each timed output is checked outside timing. TT-FSI and shapiq retain the numerical acceptance gate. Direct allclose errors are recorded diagnostically and do not interrupt completed timings; structural and nonfinite errors remain failures. Approximation is evaluated by its approximation error, not an exact-allclose gate.

## Suggested experimental-setup paragraph

All reported runtimes are arithmetic means of ten calls following two untimed warm-up calls, using the same input within each experimental cell. The ten measured calls use seeds 0 through 9 respectively, and the two warm-up calls use seeds 0 and 1. Stochastic estimators are reconstructed using the corresponding seed. Saved inputs and checkpoints remain unchanged; seeds do not define ten separately trained models or input games. Deterministic operators receive the same inputs throughout. Each warm-up invokes the complete operation measured by the corresponding timed call. Output checks and approximation-error calculations are performed outside the timed region. Model fitting and loading are excluded; coalition-value generation is measured separately under the same repetition policy.

## Table mapping

Paper Table 2: dimension runtime (script table1). Paper Table 3: batch throughput/workspace (table2). Paper Table 4: interaction order (table3). Paper Table 5: multi-seed approximation runtime/error/displacement (table4). Paper Table 6: coalition-value generation (table6). Figures retain the California case study. No standalone ranking table is included.
