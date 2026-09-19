# Table 2: Batched GPU throughput (ms/instance, MiB)

Origin: measured. Protocol SHA256: d0895acc470f03f4fcbf6dd4274b4675110da0a62a6aba970174875bc325e383. OOM denotes failure to complete under the configured host or device memory constraints, including the host-memory cap used for baseline runs. FAIL denotes validation failure or another recorded error; no runtime is reported for those cells.

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
