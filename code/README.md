# Script inventory and compatibility

| Script | Status | Environment / path assumptions | Last-run evidence |
|---|---|---|---|
| verify_artifacts.py | Historical, frozen | Transformers 5.x dictionary-style template output; cwd needs output/, merged/, quantized/, data/heldout.jsonl | evidence/verification.json |
| historical/verify_artifacts_100.py | Archived unexecuted draft, bytes preserved | Same 5.x/cwd assumptions; one-newline formatter | NOT RUN |
| trainer.yaml | Historical, frozen | Original redacted CLI config, not the new runner | evidence/train_log_redacted.txt |
| experiment_data.py / mask_audit.py | Current | Explicit return_dict=False; newline-safe JSONL; no GPU required for tokenization | New CPU logs |
| rerun.py / run_pipeline.py | Current replica workflow | CUDA + real environment lock + explicit persistent destination; configs/; checkpoints and hash-validated markers | T4 replica RUNNING; see status.json |
| verify_extended.py / variants.py / model_checks.py | Current | FP16/NF4 variants explicit; tiny-model controls run on CPU; full variants need CUDA | New CPU logs; GPU NOT RUN |
| budget.py / hostmon.py | Current | Config-based analytic/allowance budget; psutil process-tree RSS | CPU budget/tests; host GPU-run monitor active in replica |
| persist.py / remote_smoke.py | Current | DRIVE_DIR or private HF_REPO, HF_TOKEN only from environment; fresh-process reload requires CUDA | Remote GPU reload NOT RUN |
| select_task.py | Current saved-baseline selector | Requires canonical 300-case baseline outputs per task | NOT RUN |
| build_dataset.py | Current canonical-input builder | Requires licensed source metadata and actual base replay; source-specific conversion not completed | NOT RUN |
| audit_evidence.py | Current read-only audit | Repository root derived from script; local tools dependencies | New audit output |
| test_*.py | CPU tests | Local test environment; real-tokenizer test skips only if import/network unavailable | New CPU logs |

Unspecified historical fields (seed42/default, cosine decay, ~0.03 warmup, effective batch4 and optimizer behavior) are not fully encoded by trainer.yaml. They are inferred from the trace/defaults, not a recovered complete environment. The replica uses explicit settings and stops if loose preregistered tolerances fail. A historical formatter is retained only for the eight-row comparability diagnostic; the new response-only mask stops after assistant EOS.

The seven core variants plus no-op and randomized-B are nine variants when the shuffled-label adapter is available. Without that adapter, its result is missing, not silently counted as executed. No compatibility claim is made for running the frozen 5.x scripts with the new 4.x environment.

| Additional script | Status | Evidence / limit |
|---|---|---|
| sweep.py | CPU-tested | Frozen LR selection and 3-seed orchestration; GPU sweep NOT RUN |
| dev_selection.py | CPU-tested | Primary-dev score constrained by paired harmful/benign safety checks |
| eval/run_eval.py | CPU-tested resume checks | Completed suites validated against case and artifact hashes; partial suites preserved and restarted |
| eval/ship_gate.py | Gate implementation | Saved NF4 gate NOT RUN; defaults to negative on missing evidence |
