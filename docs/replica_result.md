# Recovered replica: negative acceptance result

Training finished on the original T4 and was recovered from Drive after runtime expiration. Validation loss 1.510824 fails 1.330 ± 0.010 (gap 0.180824). Downstream v2 experiments stop under the preregistered Phase 1 rule. No thresholds changed.

On the same 140 rows and response mask, Trainer averages examples: base 1.941817, trained 1.510824. Independently recomputed example means are 1.941813 and 1.510834. Token-weighted NLL is 1.880016 and 1.733213, over 9,316 targets. Different denominators explain the difference within this rerun; missing historical row IDs/masks prevent attributing the historical 1.33 versus 2.35 gap conclusively. Reference fit improved, but active-adapter, merged and NF4 comparisons remain unexecuted.

Pre-run envelope: 4.656–6.512 GB. Measured allocator allocated/reserved: 3.103/3.695 GB; raw GPU peak 3,667 MiB (3.845 GB). The conservative activation/workspace and CUDA allowances overestimated this batch's observed lengths and allocations; allocator reserve and GPU process overhead are different accounting scopes. Wall time 881.7s; host process high-water RSS 4.467 GB. Budget formulas and raw samples are adjacent files.

Drive held train.complete with its stage record, and merge.complete without stage_merge.json. The merge marker is therefore invalid; copy interruption is possible but unproven. Quantized exports and verifier outputs were absent. Recover or rebuild export only during a documented replica diagnosis; do not count the marker as success. Training code revision is recorded in logs/git_revision.txt.

Next required: reconcile the historical split/mask and training defaults, rerun the replica with frozen criteria, then complete saved-artifact verification before task selection or the sweep. The artifact is not releasable.
