# Post-hoc diagnostics (not in PLAN.md)

PLAN.md fixed E1-E4 before any exploratory run. The two scripts below were written
afterwards, in response to a result, and are reported as post hoc.

| When (UTC, 2026-10-04) | What was seen | What was added |
|---|---|---|
| 08:42 | E3: on chains planted in real networks the mass summaries (sign D, R90) fall short of the required distance even in solved cells | |
| 08:43 | | `posthoc_per_edge.py`: per-edge mean influence per shell, from the stored profiles. Fails on planted chains: the sign flip samples 8 edges per shell and misses the one decisive chain edge |
| ~08:50 | | `posthoc_reach_max.py`: sign gradient on every edge, recomputed from the stored checkpoints; reach = farthest shell whose strongest edge is at least tau of the query's strongest edge. Run on the 40 E3 records and the 160 confirmatory T=32 tree records (`posthoc-reach/`) |
| 09:05 | Decisive reach recovers the required distance in every solved tree and planted cell | `--native`: the same measure on the confirmatory network checkpoints (4 smaller networks, T in {8, 32}, all seeds; `posthoc-reach-native/`), with the uniform-influence ceiling per query |

Every result regenerates the test logits from the checkpoint and stores whether they
match the record bitwise (`test_logits_match`). Nothing is retrained.
