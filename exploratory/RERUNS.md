# Exploratory reruns (out of memory; rerun once with --memory-efficient --chunk 1, which does not change results)
| SIDNET planted-wiki_elec r4-s20004 | out of memory in the queue | rerun on GPU 4: exit 0 |
| SIDNET planted-wiki_elec r4-s20002 | out of memory in the queue | rerun on GPU 2: exit 0 |
| SIDNET planted-wiki_elec r8-s20004 | out of memory in the queue | rerun on GPU 5: exit 0 |
| SIDNET planted-wiki_elec r8-s20003 | out of memory in the queue | rerun on GPU 3: exit 0 |
| SIDNET planted-wiki_elec r8-s20000 | out of memory in the queue | rerun on GPU 0: exit 0 |
| SIDNET planted-wiki_elec r8-s20001 | out of memory in the queue | rerun on GPU 1: exit 0 |
| SLGNN epinions T32 s10004 random (E6) | out of memory on a 16 GB card in the queue (log: exploratory/logs/e6/004-SLGNN-epinions-32-10004-random-0.005-1e-05.log) | rerun once, unchanged, on GPU 0 (32 GB), 2026-10-05: completed (test AUC 0.925), measured (logits match), E6 re-collected |
| E7 audit shard 7/8 | out of memory on a 16 GB card (SLGNN on a large network) | rerun by the driver's rerun step on the 32 GB cards: all 32 audits written |
