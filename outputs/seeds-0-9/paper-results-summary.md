# Paper results: seeds 0–9

Authoritative run: 8d7090a0-70b9-4099-8339-f7eefd0e90f9. All five numerical experiments were freshly measured in this run. Two full-call warmups use seeds0 and1; measured calls use seeds0–9 once each. Saved inputs/checkpoints are unchanged. CPU dense is timed; prefix-pruned CPU is the internal reference. Direct numerical errors are diagnostic. Table5 shows means without standard deviation; raw samples and errors remain available.

At budget1000, seeds5 and6 each miss one Top-10 interaction; average displacement is0.20. Larger budgets miss none. This is a result for this game and these seeds, not a universal threshold. Figures2/3 are the regenerated PDFs in figures/. Use manuscript-table2.tex through manuscript-table6.tex for paper Tables2–6.

## Paper Table 2

| d | CPU dense | GPU bounded | Direct FSI | FSII |
| --- | --- | --- | --- | --- |
| 8 | 1.10 | 0.24 | 21.24 | 2.31 |
| 10 | 2.01 | 0.27 | 160.57 | 11.60 |
| 11 | 2.31 | 0.29 | 419.46 | 27.69 |
| 14 | 12.32 | 0.33 | 6761 | 897.34 |
| 16 | 36.35 | 0.36 | 41884 | OOM |
| 20 | 1322.51 | 0.50 | 1647940 | OOM |

## Paper Table 3

| Batch | Bounded ms/instance | Bounded MiB | Dense ms/instance | Dense MiB |
| --- | --- | --- | --- | --- |
| 1 | 0.48 | 25.00 | 13.53 | 876.00 |
| 2 | 0.34 | 50.00 | 13.02 | 1752.00 |
| 4 | 0.26 | 100.00 | 12.61 | 3504.00 |
| 8 | 0.24 | 200.00 | 12.74 | 7008.00 |
| 16 | 0.22 | 400.00 | 13.05 | 14016.00 |
| 32 | 0.21 | 800.00 | 13.07 | 28032.00 |
| 64 | 0.22 | 1600.00 | OOM | OOM |
| 100 | 0.23 | 2500.00 | OOM | OOM |

## Paper Table 4

| ell | FSII | CPU dense | GPU dense | GPU bounded |
| --- | --- | --- | --- | --- |
| 2 | 552.32 | 7.54 | 2.30 | 0.32 |
| 3 | 896.89 | 9.07 | 2.32 | 0.32 |
| 4 | 2188.29 | 10.81 | 2.31 | 0.31 |
| 5 | 8574.36 | 12.36 | 2.30 | 0.31 |
| 6 | 33600.33 | 13.95 | 2.27 | 0.31 |

## Paper Table 5

| Budget | Mean ms | Mean RMSE | Max. abs. | Mean displaced |
| --- | --- | --- | --- | --- |
| 1,000 | 14.47 | 1.56e-3 | 6.27e-3 | 0.20 |
| 4,000 | 53.26 | 4.27e-4 | 1.75e-3 | 0.00 |
| 10,000 | 159.86 | 1.74e-4 | 6.38e-4 | 0.00 |
| 16,384 | 91.33 | 6.75e-7 | 3.02e-6 | 0.00 |
| 100,000 | 91.04 | 6.75e-7 | 3.02e-6 | 0.00 |

## Paper Table 6

| d | Generation time (ms) |
| --- | --- |
| 8 | 4.36 |
| 10 | 13.11 |
| 11 | 25.39 |
| 14 | 218.85 |
| 16 | 911.14 |
| 20 | 16,294.51 |

