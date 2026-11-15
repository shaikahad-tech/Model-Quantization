# Model quantization experiment

**DON'T SHIP — NOT RELEASED.** Historical adapter training and export ran, but the saved gate evaluated **base + active adapter in bfloat16**, not either exported artifact. The new CUDA replica, v2 training and saved-NF4 release evaluation are **NOT RUN**. Source code and CPU checks are not model-improvement evidence. [Generated status](status.md) / [source](status.json).

## Evidence that exists

| Check | Observed scope | Result / evidence |
|---|---|---|
| Historical smoke test | Base, merged FP16, saved NF4; 8 rows, {{target_tokens}} targets each | NLL {{base_nll:.6f}} / {{merged_nll:.6f}} / {{nf4_nll:.6f}}; [JSON](evidence/verification.json), [script](code/verify_artifacts.py) |
| Adapter activity | {{adapter_B_nonzero}} nonzero B tensors; one target matrix | Relative q_proj delta {{q_delta:.9f}}; not proof of meaningful learning |
| Historical gate | Active adapter, {{gate_numerics}}; {{task_items}} contains tasks + eight mini-suite aggregates | Task {{task_base:.2f}}→{{task_tuned:.2f}}; safety {{mini_safety_base:.2f}}→{{mini_safety_tuned:.2f}}; [log](evidence/ship_log_redacted.txt), [scope](docs/gate.md) |
| New release gate | Saved NF4, all preregistered criteria | NOT RUN on a real artifact; [gate implementation](eval/ship_gate.py), [criteria](configs/success_criteria.yaml) |

The old strict task-win gate was unsatisfiable: the base already scored 6/6. Safety is a serious warning; its cause is untested. Mini-suite denominators and per-case outputs were not retained. **If** n=40, conditional Wilson intervals suggest a large safety drop while the three-case common-sense/tool-call differences are inconclusive. Forty is only the smallest denominator consistent with displayed fractions, not an observed sample size. [Conditional analysis](docs/gate.md).

**Replication evidence:** the two historical VMs reported validation losses {{vm1_val_loss:.4f}} / {{vm2_val_loss:.4f}}, merged NLL {{vm1_merged_nll:.6f}} / {{merged_nll:.6f}}, NF4 NLL {{vm1_nf4_nll:.6f}} / {{nf4_nll:.6f}}, and sampled q deltas {{vm1_q_delta:.9f}} / {{q_delta:.9f}}. Greedy text differed. The handwashing output is a run-specific anecdote, not an isolated quantization defect. VM1 export was interrupted and only a verification tail survives; this limits provenance.

## Materials and selection bias

[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, has {{parameters:,}} base parameters. Its scale fits the historical T4 and its chat template is inspectable. [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) is English, so the author can read prompts and responses; comprehensive manual scoring is not claimed.

The source was shuffled with seed 20261111; the first 1,500 unique complete rows with **≤480 rendered tokens** were selected. Exact-row hashes deduplicated selections. Long rows were excluded and `category` was dropped. Training input has {{train_rows}} rows (10% internal validation: 1,260/140); heldout has {{heldout_rows}}. Computed warnings: {{duplicate_instructions}} duplicate instruction occurrences beyond the first, {{near_pairs}} train/holdout instruction pairs at word-set Jaccard≥0.8, and {{exact_overlap}} exact full-row overlaps. [Data card](data/DATA_CARD.md), [frozen ZIP](data/sampled_dataset.zip).

SHA-256: train `f2851c4d747445d727e86fd03483bfce30250b0843db25b38f8a86d3a4aeaab3`; heldout `87fb5abbb936b05909856c6f22ec56fdc84d106b0d737aee3e0d49b60bcf1002`. Upstream pinned template string: `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`, rechecked by a real-tokenizer CPU test; this does not recover the lost runtime's tokenizer hash.

Historical CLI 0.77.2 is named only by the public alias `trainer`; Transformers was 5.22.1. The new pipeline uses Transformers/PEFT/bitsandbytes directly so masks, merges and artifact identity are inspectable. [Code license](LICENSE), [notices](NOTICE.md), [data/model licenses](DATA_LICENSES.md). MIT applies to original code, not Dolly data or model weights.

## Memory: analytic, measured and unknown

| Quantity | Kind | Value / explanation |
|---|---|---|
| NF4 linears | Analytic | {{packed_linear_bytes:,}} bytes |
| Tied FP16 embedding | Analytic | {{embedding_fp16_bytes:,}} bytes; embeddings are not packed to four bits |
| NF4 floor before stats | Analytic | {{nf4_floor_without_stats_bytes:,}} bytes (~1.122 GB), replacing the unattainable all-parameters-at-four-bits estimate |
| Double-quant stats + bias/norm | Analytic | {{double_quant_stats_bytes:,}} + {{bias_norm_fp16_bytes:,}} bytes |
| NF4 file estimate | Analytic | {{nf4_analytic_bytes:,}} bytes versus historical 1,143,327,349 bytes (~0.035% difference) |
| FP32 nonquantized upcast | Expected, needs GPU census | Additional {{expected_fp32_nonquantized_upcast_bytes:,}} bytes during k-bit training |
| Training device peak | Sampled | {{train_peak_mib:,}} MiB = {{train_peak_gb:.3f}} GB; {{train_samples}} samples, mean interval {{train_interval_seconds:.3f}} s |
| CLI GPU peak display | Reported, semantics unverified | {{vm2_cli_peak_gb:.1f}}/{{vm2_cli_total_gb:.1f}} GB in both VM logs; not independently identified as allocated or reserved memory |
| Merge / quantize device peaks | Sampled | {{merge_peak_mib}} / {{quantize_peak_mib:,}} MiB; host peaks absent |

[Full memory budget](docs/memory.md) separates weights, adapters, optimizer, logits, activations and export buffers. No timestamped historical pre-run worksheet exists. The new [budget generator](code/budget.py) records actual config assumptions before training; new CPU arithmetic is not a measured GPU peak. Raw GPU output: [train](evidence/nvidia_smi_train.txt), [merge](evidence/nvidia_smi_merge.txt), [quantize](evidence/nvidia_smi_quantize.txt).

## Running and reviewing

- [Colab notebook](colab/run_all.ipynb) and [reproduction instructions](docs/reproduction.md): explicit persistence, config, seeds, checkpoint resume and failure preservation. The Colab lock must come from a real runtime; it has not been fabricated.
- [Report](report.pdf) / [source](report.md), [failure matrix](docs/silent-failures.md), [AI disclosure](docs/disclosure.md).
- [Evaluation/scorers/statistics](eval/), [configs](configs/), [release-gate specification](docs/gate_v2.md).
- [Historical script inventory](code/README.md); the old 100-row draft is archived, not evidence of execution.
- [Public ZIP](experiment-evidence.zip), [release hashes](RELEASE_HASHES.txt), [raw-log/redaction note](RAW_LOGS_NOTE.md), [model card: NOT RELEASED](MODEL_CARD.md).

## What must change before ship

Run and pass the historical-recipe replica; investigate any same-mask validation/NLL discrepancy before progressing. Select a non-saturated task from saved base baselines, build licensed disjoint replay data, and execute the LR/seed/shuffled-label experiments. Evaluate seven core variants plus no-op/randomized-B controls on identical cases. Complete the human refusal-label audit and durable remote reload. Only [G1–G9](docs/gate_v2.md), evaluated on the saved NF4 artifact under the committed criteria, may produce SHIP. Missing evidence keeps DON'T SHIP. No public model release or email delivery is claimed.
