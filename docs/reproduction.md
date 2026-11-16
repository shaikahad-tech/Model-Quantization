# Reproduction and current limits

Use the Python supplied by Colab; Python 3.11 is not mandatory. The clean entry point is `colab/run_all.ipynb`, or the authenticated official Colab CLI. Configure a mounted `DRIVE_DIR` or private `HF_REPO` before starting. Credentials belong in Colab secrets/environment, never Git.

Bootstrap with `python -m pip install -r code/requirements-colab.in`, then save the actual runtime's `pip freeze` as requirements-lock.txt on persistent storage. The actual Python 3.13 Colab lock is recorded in evidence/runs/20261115-colab/requirements-lock.txt; it includes the preinstalled image packages and known unrelated dependency conflicts. Restart the runtime after replacing already-imported libraries.

```sh
bash reproduce.sh --dry-run --config configs/train_replica.yaml
bash reproduce.sh --stage replicate --run-dir runs/replica-20261115 --config configs/train_replica.yaml --resume
```

The replica records a fresh config-derived budget before training; it does not copy the retrospective report. It records environment versions, initial GPU output, compact GPU samples, process-tree RSS, allocator maxima, baseline validation and same-mask response NLL. Checkpoints are saved every 50 steps and copied to persistence. Completed stages are skipped only when marker and output hashes validate. A failed or incomplete export is preserved and rebuilt. Off-VM disconnect recovery still requires a real GPU test.

Replica tolerances: validation loss 1.330±0.01; historical-format eight-row merged NLL 2.352±0.01, NF4 NLL2.378±0.01; sampled q delta0.00628±5%. Historical validation row IDs and loss masks were lost, so this is not assured bit-for-bit reproduction. A tolerance failure stops progression. The original 100-row draft is byte-preserved at `code/historical/verify_artifacts_100.py`.

Later commands support `--stage eval --suite-dir PATH`, `--stage publish` (private persistence/fresh-process reload), and `--stage gate`. Publish/reload must precede the final gate to supply durability evidence. Missing shuffled-label weights or licensed suites are errors, not omitted successful checks. Human refusal labels must be entered by a reviewer using the blank sheet from tools/make_manual_audit.py.

**Incomplete:** source-specific candidate conversions are not completed; sweep orchestration is CPU-tested but not GPU-validated; v2 primary-dev/safety-dev checkpoint selection is CPU-tested but not GPU-validated. The runner requires a verified passing replica and safe memory budget before v2 training; v2 selects checkpoints by primary development score subject to safety constraints. The GPU replica finished and failed validation-loss acceptance; see docs/replica_result.md. Downstream experiments are stopped; full suite evaluation, persistent reload and release gate have not run. These limitations are tracked in status.json; DON'T SHIP remains in effect.

Run the preregistered sweep with `python code/sweep.py --out runs/sweep-v2 --resume` after the replica, task selection and dataset prerequisites pass. It records the 2×8 memory fallback when needed, preserves unsafe candidates for review, selects only safety-eligible candidates, and retains a shuffled-label negative control.

On the current Colab image, the preinstalled torchvision is incompatible with pinned Torch 2.6. The text-only experiment succeeded at importing Trainer after `python -m pip uninstall -y torchvision`; retain both import logs and capture the environment lock after this repair. Other image dependency conflicts remain visible in the installation log.

`requirements-lock.txt` is the full observed image freeze, copied byte-for-byte from Colab. It is an environment record, not a clean resolver lock: preinstalled gradio/diffusers/gcsfs constraints conflict with this stack. Installing that entire image freeze into a fresh environment is not verified. The experiment dependencies in `code/requirements-rerun.txt` are pinned to versions observed in that freeze.
