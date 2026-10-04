# Reruns under protocol Section 8 (a crashed run is rerun once, unchanged)

| run | first attempt | cause | rerun |
|---|---|---|---|
| SLGNN-epinions-T32-s10004-spectral | GPU 4 (V100 16 GB), failed after 3085 s | CUDA out of memory in the Jacobian stage (log: confirmatory/logs/SLGNN-epinions-32-10004-spectral.log) | unchanged command on GPU 1 (V100 32 GB), started 18:39 UTC, completed 01:07 UTC (exit 0), record verifies; runs are bitwise reproducible across 16 GB and 32 GB V100s (development/20261003-detcheck3) |
| relay-4-SIDNET-32-10000-2 | queue attempt failed (see confirmatory/logs/relay-4-SIDNET-32-10000-2.log) | out of memory on a shared card | rerun on GPU 1 from 03:14 with --memory-efficient --chunk 1; completed, record written |
| relay-4-SIDNET-32-10001-2 | queue attempt failed (see confirmatory/logs/relay-4-SIDNET-32-10001-2.log) | out of memory on a shared card | rerun on GPU 1 from 03:47 with --memory-efficient --chunk 1; completed, record written |
| relay-4-SIDNET-32-10003-2 | queue attempt failed (see confirmatory/logs/relay-4-SIDNET-32-10003-2.log) | out of memory on a shared card | rerun on GPU 0 from 04:21 with --memory-efficient --chunk 1; completed, record written |
| relay-8-SIDNET-32-10003-2 | queue attempt failed (see confirmatory/logs/relay-8-SIDNET-32-10003-2.log) | out of memory on a shared card | rerun on GPU 0 from 04:48 with --memory-efficient --chunk 1; completed, record written |
| relay-8-SIDNET-32-10004-2 | queue attempt failed (see confirmatory/logs/relay-8-SIDNET-32-10004-2.log) | out of memory on a shared card | rerun on GPU 0 from 05:24 with --memory-efficient --chunk 1; completed, record written |
| relay-4-SIDNET-32-10004-2 | queue attempt failed (see confirmatory/logs/relay-4-SIDNET-32-10004-2.log) | out of memory on a shared card | rerun on GPU 0 from 05:55 with --memory-efficient --chunk 1; completed, record written |
