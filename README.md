# Model quantization experiment

**DON'T SHIP — negative result; submission and reproduction remain incomplete.** The saved release gate compared the base against the **active LoRA adapter in bfloat16**. It did **not** test either exported artifact. The separate eight-example smoke test loaded base, merged FP16, and saved NF4 files and found a sampled weight change and lower reference NLL. Neither result proves useful learning or release readiness. The full 4-bit gate, 100-example comparison, negative controls and per-case safety investigation have **not run**.

## What actually ran

| Check | Artifact / numerics | Coverage and result | Evidence |
|---|---|---|---|
| Historical verification | Base, merged FP16, serialized NF4; FP16 compute | 8 held-out rows, 457 target tokens each; 2 generated answers each; 1 weight matrix; 56 adapter B tensors | [Script](code/verify_artifacts.py), [output](evidence/verification.json), [redacted log](evidence/verification_log_redacted.txt) |
| Historical release gate | Base vs base + `output/` adapter, bfloat16 | 6 contains-match tasks: 6/6 each; 8 mini benchmark aggregates; exit 2, `failed_rule=task_win` | [Command/scope](docs/gate.md), [redacted log](evidence/ship_log_redacted.txt), [verdict](evidence/ship_verdict_redacted.json) |
| Exported-model release gate | Merged FP16 / saved NF4 | Neither evaluated by that gate | No result |
| Extended evaluation | Base, active adapter, merged, NF4, controls | Not run; historical per-case safety outputs and denominators absent | [Outstanding work](status.md) |

The adapter gate reported safety **1.00 → 0.45**, common sense **0.825 → 0.750**, and tool calling **0.700 → 0.625**. These are warnings about the adapter configuration, not measured safety scores for the NF4 file. Exact mini-suite counts cannot be established from the saved output; we do not infer denominators from rounded scores.

```text
Historical smoke test: 8 rows, 457 answer tokens per artifact
adapter_B_nonzero = 56/56
layer_0_q_proj_relative_delta = 0.0062767863273620605
NLL/token: base=2.5134674371 merged=2.3520263926 NF4=2.3784927409
structural_pass=true; quantized_close_nll=true
```

These booleans are narrow script checks, **not** a shipping decision. Random changes, incorrect merge scaling, contamination, wrong masks and unsafe behavior can pass them. The evaluator also joins context with one newline whereas the data filter uses two; scores are not a validated match to the training formatter.

## Materials

[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, was chosen for its T4-compatible size and inspectable chat template. Data: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), English, CC BY-SA 3.0 according to its dataset card. English permits direct reading of prompts and answers; this does not imply all answers were manually scored. Seed 20261111; 1,500 selected rows: 1,400 submitted to training (10% validation: 1,260 train / 140 validation) and 100 separate holdouts.

[Exact sampled data](data/sampled_dataset.zip), SHA-256:

- `train.jsonl`: `f2851c4d747445d727e86fd03483bfce30250b0843db25b38f8a86d3a4aeaab3`
- `heldout.jsonl`: `87fb5abbb936b05909856c6f22ec56fdc84d106b0d737aee3e0d49b60bcf1002`
- Pinned upstream chat-template UTF-8 string: `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`. Retrieved during this audit; **not** a recorded hash of the lost runtime tokenizer. EOS `<|im_end|>`; padding `<|endoftext|>`.

Historical stack: CLI 0.77.2 (public alias `trainer`), Transformers 5.22.1, T4, r=16 / alpha=32 q/v LoRA, NF4 training, length 512, one epoch / 315 optimizer steps. The CLI automated training/export/gating; its identity is redacted at the owner's request. The alias is not an installable replacement. [Redaction note](RAW_LOGS_NOTE.md).

## Memory: estimates versus sampled peaks

Decimal GB below; [detailed assumptions](docs/memory.md). These are reconstructed estimates: no timestamped pre-training budget was retained.

| Component | Formula / allowance | Estimate GB | Observation |
|---|---|---:|---|
| Quantized base | 1.544B × 0.5 bytes lower bound; scales/high-precision tensors extra | ≥0.772; ~1.14 file-size proxy | Allocation unmeasured |
| Adapter + gradients | 2,179,072 × 2 × 2 bytes | 0.0087 | Not separately measured |
| Adam + possible master | 2,179,072 × (8+4) bytes | 0.0261 | Not separately measured |
| Logits | 512 × 151,936 × 2 or 4 bytes | 0.156 / 0.311 | May overlap during loss |
| Activations / backward / workspace | Shape-dependent allowance | 2–3 | Not separately measured |
| CUDA / allocator | Allowance | 0.7–1.4 | Not separately measured |
| Training total | Rounded envelope | 4–6 | **5,395 MiB = 5.657 GB**; 235 samples / 2 s |
| CPU merge | 2 × 1.544B × 2 bytes + buffers | 7–10 host; 0–1 GPU | **115 MiB = 0.121 GB GPU**; 111 samples / 1 s; host peak missing |
| Quantization | 3.09 source + 1.14 packed + 1–3 workspace | 5.2–7.2 host; ~1–2 GPU allowance | **1,257 MiB = 1.318 GB GPU**; 185 samples / 1 s; host peak missing |

Unchanged raw GPU output: [train](evidence/nvidia_smi_train.txt), [merge](evidence/nvidia_smi_merge.txt), [quantize](evidence/nvidia_smi_quantize.txt). Sampling misses short peaks. CUDA context and reserved memory make device usage exceed tensor bytes; CPU export explains low merge GPU use. Exact gaps cannot be apportioned without allocator and host measurements.

## Deliverables and reproduction

- [Report PDF](report.pdf) / [source](report.md): at most two pages.
- [Silent-failure matrix](docs/silent-failures.md), [AI disclosure and reflection](docs/disclosure.md).
- [Historical notebook](code/experiment_notebook_redacted.ipynb): failed cells / first-runtime outputs retained, retry provenance stated; redacted and not a run-all workflow.
- [Historical verification](code/verify_artifacts.py) and [saved output](evidence/verification.json): the check that actually ran.
- [New rerun instructions](docs/reproduction.md): a public Hugging Face workflow. **GPU execution pending; new code is not historical evidence.**
- [Public evidence ZIP](experiment-evidence.zip): documentation, code, redacted logs, raw GPU samples and exact sampled data. Model weights are absent.
- [Raw-log delivery status](RAW_LOGS_NOTE.md): originals exist locally; email delivery unverified.

## What must change before ship

1. Recover or regenerate adapter, merged and NF4 weights; save durable artifacts, hashes, tokenizer files and dependency versions.
2. Run the gate on the **saved NF4 artifact** (`--tuned ./quantized` in the original CLI); verify loaded modules and reported numerics. Compare base, active adapter, merged and NF4 on identical cases.
3. Obtain safety cases and denominators; score each with a documented rubric, diagnose regressions, fix data/training/masking and rerun. Do not simply exclude failing cases.
4. Run all 100 heldouts with per-row NLL, negative controls including shuffled-label training, all-module merge checks and fixed-prompt logits. Use representative, non-saturated tasks.
5. Audit all training loss masks, EOS/padding and template; rerun final preflight. Record host/GPU peaks and a budget **before** the next training run.
6. Execute the workflow end to end, preserve failures, deliver original raw logs privately, and establish accessible weight links. Then reconsider the verdict.

Documentation corrections do not complete these experiment requirements.
