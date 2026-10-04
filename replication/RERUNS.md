# Replication reruns (PROTOCOL.md Section 4)

| item | first attempt | cause | rerun |
|---|---|---|---|
| measure native SLGNN-epinions-T32-s10003-spectral | shard 38/40 on a 16 GB V100, 2026-10-04 ~19:20 UTC | CUDA out of memory in the sign gradient (log: replication/logs/measure-native-38of40.log); the shard's other items completed | rerun once on a 32 GB card by the measurement-rerun step of replication/run_all.sh (19:56-20:12 UTC): completed, test logits match the record (log: replication/logs/measure-rerun-2of4.log) |
| measure native SLGNN-epinions-T32-s10004-spectral | shard 39/40 on a 16 GB V100, 2026-10-04 ~19:20 UTC | CUDA out of memory in the sign gradient (log: replication/logs/measure-native-39of40.log); the shard's other items completed | rerun once on a 32 GB card by the measurement-rerun step of replication/run_all.sh (19:56-20:12 UTC): completed, test logits match the record (log: replication/logs/measure-rerun-3of4.log) |

All 120 planted-chain training runs completed on their first attempt.
