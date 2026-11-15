# Reproduction and current limits

Use the Python supplied by Colab; Python 3.11 is not mandatory. The clean entry point is `colab/run_all.ipynb`, or the authenticated official Colab CLI. Configure a mounted `DRIVE_DIR` or private `HF_REPO` before starting. Credentials belong in Colab secrets/environment, never Git.

Bootstrap with `python -m pip install -r code/requirements-colab.in`, then save the actual runtime's `pip freeze` as requirements-lock.txt on persistent storage. A Colab lock has not yet been captured in this repository; do not describe the Mac CPU dependency list as one. Restart the runtime after replacing already-imported libraries.

```sh
bash reproduce.sh --dry-run --config configs/train_replica.yaml
bash reproduce.sh --stage replicate --run-dir runs/replica-20261115 --config configs/train_replica.yaml --resume
```

The replica records a fresh config-derived budget before training; it does not copy the retrospective report. It records environment versions, initial GPU output, compact GPU samples, process-tree RSS, allocator maxima, baseline validation and same-mask response NLL. Checkpoints are saved every 50 steps and copied to persistence. Completed stages are skipped only when marker and output hashes validate. A failed or incomplete export is preserved and rebuilt. Off-VM disconnect recovery still requires a real GPU test.

Replica tolerances: validation loss 1.330±0.01; historical-format eight-row merged NLL 2.352±0.01, NF4 NLL2.378±0.01; sampled q delta0.00628±5%. Historical validation row IDs and loss masks were lost, so this is not assured bit-for-bit reproduction. A tolerance failure stops progression. The original 100-row draft is byte-preserved at `code/historical/verify_artifacts_100.py`.

Later commands support `--stage eval --suite-dir PATH`, `--stage publish` (private persistence/fresh-process reload), and `--stage gate`. Publish/reload must precede the final gate to supply durability evidence. Missing shuffled-label weights or licensed suites are errors, not omitted successful checks. Human refusal labels must be entered by a reviewer using the blank sheet from tools/make_manual_audit.py.

**Incomplete:** source-specific candidate conversions, v2 primary-dev/safety-dev checkpoint selection and sweep orchestration are not validated. The runner deliberately blocks v2 training instead of silently choosing by loss. The GPU replica, full suite evaluation, persistent reload and release gate have not run. These limitations are tracked in status.json; DON'T SHIP remains in effect.
