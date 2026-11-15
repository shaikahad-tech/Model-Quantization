# Silent-failure matrix

`trainer` is the original CLI's public alias. “Unknown” means no retained test establishes the built-in check's behavior. Explicit crashes are retained as failures, not mislabeled as silent success.

| Stage | Failure / symptom | Check run | Result | Built-in check caught? | Evidence |
|---|---|---|---|---|---|
| Data | Truncation removes answer despite sampled validation success | Sampled doctor, then training | Doctor passed; train failed row 4. Filter corrected. | Doctor missed; train caught zero target explicitly | [Failure excerpt](../evidence/initial_failed_train_excerpt_redacted.txt) |
| Data | Leakage/duplicates make NLL optimistic | Exact-row hashes/disjointness | Exact rows disjoint; semantic/pretraining contamination unknown | No evidence of semantic checks | [Dataset](../data/sampled_dataset.zip), [local audit](../evidence/local_audit_20261115.json) |
| Template | `len(BatchEncoding)` counts keys, not tokens | Inspect `input_ids` | First filter counted mapping keys. The dict-only repair also failed for BatchEncoding (2013-token maximum); only the third fix worked | Filter missed; doctor sampled | [Notebook](../code/experiment_notebook_redacted.ipynb), cells 8–11 |
| Labels | Wrong EOS/padding/mask trains on prompts | Doctor generation-marker warning; evaluator prefix assertion | Warning retained; actual all-row training mask audit missing | Doctor warned without blocking; final mask coverage unknown | [Doctor](../evidence/data_doctor_redacted.json) |
| Formatter | Training/eval template mismatch | Source comparison in audit | Evaluator joins context with one newline, filter uses two; actual training formatter not captured | No retained built-in mismatch check | [Evaluator](../code/verify_artifacts.py), notebook cell 11 |
| Adapter | Wrong targets/frozen/no-op adapter with falling loss | 56 nonzero B tensors; 1 merged q delta | Nonzero change; random/wrong adapter can pass | No independent adversarial built-in test | [Verification](../evidence/verification.json) |
| Merge | Wrong base/adapter/scaling gives plausible checkpoint | One matrix delta; no merged adapter file | Full W+alpha/r×BA equality and active-adapter equivalence absent | Unknown | [Merge log](../evidence/merge_log_redacted.txt), verification |
| Quantization | Skipped layers/incorrect dtype/wrong export path | Config, saved reload, NLL | NF4/double-quant/FP16 configured; no module census; NLL +0.02647 on 8 rows | Export success is not conversion-coverage evidence | [Config/output](../evidence/verification.json) |
| Inference | Wrong tokenizer/adapter silently loaded | Base tokenizer reused for all artifacts | Saved tokenizer identity not compared; adapter equivalence absent | Gate used bf16 adapter, not NF4 | [Gate scope](gate.md) |
| Evaluation | Saturated tasks hide regression | Six tasks, eight aggregate mini checks | 6/6 ties; safety 1→0.45; per-case evidence absent | Gate rejected task_win; flagged regressions | [Verdict](../evidence/ship_verdict_redacted.json) |
| Dependencies | Retrying hides incompatible export environment | Failures retained | Cached symlink and dependency mismatch failed explicitly; retry succeeded | Exceptions caught failures, not silent correctness | [Dependency failure](../evidence/initial_failed_merge_dependency_excerpt_redacted.txt), [install log](../evidence/torchao_install_log.txt) |

| Filter repair | A dict-only type check silently fails for BatchEncoding | Re-read notebook outputs | Second filter still admitted a 2013-token example; final cell 11 fixed it | Sampled preflight did not establish all-row coverage | Historical notebook cells 8–11 |
| Repeated runs | Near-identical NLL hides different greedy answers | Compare VM1 tail and VM2 JSON | Small numeric differences; handwashing text differs | No determinism guarantee | Initial verification tail and verification.json |
| EOS loss | Trailing template newline becomes supervised | Real-tokenizer CPU test | New labels stop after assistant EOS; physical trailing tokens remain masked | Historical mask unknown | code/test_data.py |
| API drift | 4.x list vs 5.x BatchEncoding breaks template indexing | Explicit return_dict=False + two environment tests | New shared encoder tested; historical scripts frozen with old assumptions | Not covered by original preflight | New CPU logs |
| Identity disclosure | Neutral names still leave package/repo identifiers | Exact redaction scan | Specified leaks fixed in current files; history retained | New scanner catches them | RAW_LOGS_NOTE.md |
| Schedule | Linear/default seed silently changes attempted replica | Compare explicit configs with historical trace | Cosine/0.03/seed42 replica config; some historical defaults remain inferred | No past complete lock | configs/train_replica.yaml |
| Scoring | Case-sensitive substrings match ice inside nice or 4 inside14 | Adversarial scorer unit tests | New boundary/alias scorer rejects these; legacy score retained | Old tasks remain deprecated | code/test_scorers.py |
| VM1 provenance | Interrupted export plus tail-only verification looks complete | Inspect notebook interrupted cell and truncated tail | VM1 serialized-artifact provenance ambiguous | Explicit interruption preserved | Historical notebook15–18 |
| Verifier bug | Attention layer scaling is a float, not LoRA's scaling dictionary | Tiny Qwen wrong-scale test | First CPU test failed; fixed type guard; failing transcript retained | New CPU test caught it before GPU use | evidence/runs/20261115-local/ |

Final environment doctor/plan, full training masks, final-runtime dependency imports, negative controls and four-artifact per-case safety remain unverified. New source code does not establish passed checks.
