# Model quantization experiment

**Decision: DON'T SHIP.** A 1.5B instruction model was fine-tuned with LoRA, merged, and exported to NF4 4-bit. The adapter changed model weights and improved reference likelihood on an eight-example held-out slice. The live release gate rejected it: six task checks tied at 1.00, while mini safety fell from 1.00 to 0.45 and common sense and tool calling also regressed.

## Files

- [Report (PDF)](report.pdf) and [source](report.md): memory estimate, measured peaks, verification, silent failures, and decision.
- [Notebook](code/experiment_notebook_redacted.ipynb): saved experiment code and first-runtime outputs, followed by a provenance note for the retry. Product identifiers in the public copy are replaced, so it is **for review, not direct execution**.
- [Verification scripts](code/verify_artifacts.py): the eight-example check that ran. The [100-example extension](code/verify_artifacts_100.py) was prepared but did **not** run.
- [Evidence](evidence/): timestamped training, merge, quantization, verification, and release-gate logs; GPU samples; and machine-readable verdicts.
- [Sampled dataset](data/sampled_dataset.zip): exact train and held-out JSONL files in a ZIP. The data is unchanged; its original hashes appear in the report.
- [Status](status.md): completed work and outstanding checks.

The published logs are **redacted copies**: product names and documentation domains were replaced. Their failures, metrics, timestamps, and numerical results were preserved. Untouched originals remain local and are not represented here as public raw logs. GPU sampling files are unchanged.

The model weights are not in this repository. The hosted runtime reset before the 4-bit release gate, 100-example evaluation, and per-case safety comparison could be completed. This repository documents the negative result and cannot establish release readiness.
