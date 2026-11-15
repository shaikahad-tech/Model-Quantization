# Silent-failure matrix

`trainer` is the original CLI's public alias. “Unknown” means no retained test establishes the built-in check's behavior. Explicit crashes are retained as failures, not mislabeled as silent success.

| Stage | Failure / symptom | Check run | Result | Built-in check caught? | Evidence |
|---|---|---|---|---|---|
| Data | Truncation removes answer despite sampled validation success | Sampled doctor, then training | Doctor passed; train failed row 4. Filter corrected. | Doctor missed; train caught zero target explicitly | [Failure excerpt](../evidence/initial_failed_train_excerpt_redacted.txt) |
| Data | Leakage/duplicates make NLL optimistic | Exact-row hashes/disjointness | Exact rows disjoint; semantic/pretraining contamination unknown | No evidence of semantic checks | [Dataset](../data/sampled_dataset.zip), [local audit](../evidence/local_audit_20261115.json) |
| Template | `len(BatchEncoding)` counts keys, not tokens | Inspect `input_ids` | Initial filter wrong; fixed before retry | Filter missed; doctor sampled | [Notebook](../code/experiment_notebook_redacted.ipynb), cells 8–11 |
| Labels | Wrong EOS/padding/mask trains on prompts | Doctor generation-marker warning; evaluator prefix assertion | Warning retained; actual all-row training mask audit missing | Doctor warned without blocking; final mask coverage unknown | [Doctor](../evidence/data_doctor_redacted.json) |
| Formatter | Training/eval template mismatch | Source comparison in audit | Evaluator joins context with one newline, filter uses two; actual training formatter not captured | No retained built-in mismatch check | [Evaluator](../code/verify_artifacts.py), notebook cell 11 |
| Adapter | Wrong targets/frozen/no-op adapter with falling loss | 56 nonzero B tensors; 1 merged q delta | Nonzero change; random/wrong adapter can pass | No independent adversarial built-in test | [Verification](../evidence/verification.json) |
| Merge | Wrong base/adapter/scaling gives plausible checkpoint | One matrix delta; no merged adapter file | Full W+alpha/r×BA equality and active-adapter equivalence absent | Unknown | [Merge log](../evidence/merge_log_redacted.txt), verification |
| Quantization | Skipped layers/incorrect dtype/wrong export path | Config, saved reload, NLL | NF4/double-quant/FP16 configured; no module census; NLL +0.02647 on 8 rows | Export success is not conversion-coverage evidence | [Config/output](../evidence/verification.json) |
| Inference | Wrong tokenizer/adapter silently loaded | Base tokenizer reused for all artifacts | Saved tokenizer identity not compared; adapter equivalence absent | Gate used bf16 adapter, not NF4 | [Gate scope](gate.md) |
| Evaluation | Saturated tasks hide regression | Six tasks, eight aggregate mini checks | 6/6 ties; safety 1→0.45; per-case evidence absent | Gate rejected task_win; flagged regressions | [Verdict](../evidence/ship_verdict_redacted.json) |
| Dependencies | Retrying hides incompatible export environment | Failures retained | Cached symlink and dependency mismatch failed explicitly; retry succeeded | Exceptions caught failures, not silent correctness | [Dependency failure](../evidence/initial_failed_merge_dependency_excerpt_redacted.txt), [install log](../evidence/torchao_install_log.txt) |

Final environment doctor/plan, full training masks, final-runtime dependency imports, negative controls and four-artifact per-case safety remain unverified. New source code does not establish passed checks.
