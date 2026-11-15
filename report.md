# Model quantization evidence audit

Historical run: 11 November 2026. Revised: 15 November 2026.

**DON'T SHIP.** The saved gate evaluated the **base plus active adapter in bfloat16**, not the merged FP16 or NF4 export. It exited 2 (`failed_rule=task_win`): six contains-match tasks tied 6/6, while aggregate safety fell 1.00→0.45, common sense 0.825→0.750 and tool calling 0.700→0.625. All eight mini-suite aggregates were saved; per-case outputs and denominators were not. Neither exported artifact received this gate. Missing 4-bit validation and the adapter regression block release.

## Materials and provenance

Qwen2.5-1.5B-Instruct (revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) fits a T4 and has an inspectable template. Databricks Dolly 15k provides English instructions that can be read directly; its dataset card specifies CC BY-SA 3.0. A deterministic seed (20261111) selected 1,500 rows: 1,400 input to training, with 10% validation (1,260/140), plus 100 separate holdouts. Direct English inspection does not mean all outputs were manually scored. Exact file hashes and the newly retrieved upstream template hash are in README; the historical tokenizer hash was not saved.

The historical CLI (public alias `trainer`, version 0.77.2) automated training/export/gating; Transformers was 5.22.1. Names are redacted at the owner's request; aliases are not runnable commands. LoRA r16/alpha32 targeted q_proj/v_proj; 4-bit training, length 512, one epoch, 315 steps. Retry loss fell 1.7595→1.6234. FP16 merge: 3,087,466,808 bytes; NF4 double-quant export: 1,143,327,349 bytes. Logs document exports and reloads; weights were lost after runtime reset. A new public HF rerun workflow exists but has not run on GPU.

## Memory: retrospective estimate versus sampled use

**No timestamped pre-training budget survives.** These reconstructed estimates do not satisfy that timing requirement. GB is decimal. Model: ~1.544B parameters, 28 layers, h1536, 12 heads, 2 KV heads, vocabulary151936. Batch1, sequence512. Frozen base weights have no optimizer/gradient states.

| Component | Calculation / estimate |
|---|---|
| Base | FP16: 1.544B×2=3.088 GB; ideal 4-bit: ×0.5=0.772 GB; actual file 1.143 GB is only a storage proxy |
| LoRA | A=28×16×[(1536+1536)+(1536+256)]=2,179,072; FP16 weights+grads=A×4=0.0087 GB |
| Optimizer | FP32 moments A×8=0.0174 GB; possible master A×4=0.0087 GB |
| Logits | 512×151936×2=0.156 GB; FP32 loss copy 0.311 GB; may overlap |
| Activations | Hidden L×S×h×2=0.044 GB; dense attention L×12×S²×2=0.176 GB; other saved tensors/workspace allowance brings estimate to 2–3 GB |
| Training total | CUDA/allocator allowance 0.7–1.4 GB; total ~4–6 GB; observed 5395 MiB=5.657 GB (235 samples, 2 s) |
| Merge peak | Two FP16 copies 6.176 GB + buffers ≈7–10 GB host; GPU 0–1 GB allowance, observed115 MiB=0.121 GB (111 samples, 1 s) |
| Quantize peak | 3.088 source+1.143 packed+1–3 buffers≈5.2–7.2 GB host; GPU ~1–2 GB allowance, observed1257 MiB=1.318 GB (185 samples, 1 s) |

Activation formulas anchor shapes, not actual simultaneous allocations. Checkpointing, FFN intermediates, dtype upcasts and kernels affect memory. Training is within the broad envelope; CUDA context/reserved memory explain part of the gap from tensor bytes. Low export GPU use reflects CPU paths, not low host peaks. Host peaks were not measured; a 6.8 GB UI snapshot is insufficient. Polling misses short spikes. Raw samples are in `evidence/nvidia_smi_*.txt`; no independently verified throughput is claimed.

<!-- PAGEBREAK -->

## Verification: what the output proves and misses

A falling loss can reflect easier batches, leakage or incorrect labels; it does not establish an active adapter or correct exported checkpoint. Historical `code/verify_artifacts.py` exited0 at08:52:06 UTC. Saved output:

| Historical smoke check | Result |
|---|---|
| Adapter B tensors | 56/56 nonzero |
| Layer0 q_proj relative weight delta | 0.0062767863273620605 |
| Base / merged / NF4 NLL per target token | 2.5134674371 / 2.3520263926 / 2.3784927409 |
| Denominator | Eight heldout rows; 457 target tokens per artifact |
| Other checks | No adapter file in merged; two greedy answers each; structural_pass=true, quantized_close_nll=true |

This establishes a sampled nonzero change and lower NLL on that slice, not meaningful learning. Random changes/wrong adapters can pass. No negative controls, all-module merge algebra, active-adapter equivalence, safety, contamination or generalization proof exists. The original evaluator uses one newline for context; the data filter uses two. Actual training-mask fidelity remains unknown. The quantized handwashing answer produced five steps despite a request for three and was truncated. Historical logs and JSON are retained unchanged apart from declared name redaction.

New code adds a 100-row comparison, no-op/randomized-B controls, 56 target merge residuals, fixed-prompt logits, module/dtype census and per-case task output. **No GPU results exist for it.** Shuffled-label training and per-case safety remain absent; new source and CPU tests are not evidence of passed model checks.

## Silent failures and built-in coverage

| Stage | Check / result / built-in coverage |
|---|---|
| Data | Sampled doctor passed; training later rejected a zero-target row. Token-length filter counted mapping keys, then was corrected to input_ids. Exact overlap checks do not detect semantic leakage. |
| Template/labels | Doctor warned about missing generation markers without blocking. EOS/pad and all-row actual training loss masks were not audited. Formatting mismatch discovered later. |
| Training | Nonzero B and one weight delta detect some no-op failures, not wrong/nonmeaningful updates. Built-in adversarial coverage was not tested. |
| Merge | Single sampled delta cannot establish alpha/r scaling or base identity. Symlink and dependency failures raised exceptions and were preserved; successful retry is not equivalence proof. |
| Quantization | Config says NF4/double-quant/FP16; saved artifact loaded. Historical layer-conversion census missing; +0.02647 NLL on eight rows is insufficient. |
| Inference/gate | Base tokenizer reused, saved-tokenizer equality untested. Adapter bf16 gate rejected task_win and flagged regressions; no gate on saved NF4. Final preflight coverage unknown. |

The full matrix and evidence links are in `docs/silent-failures.md`. Failed install/train/merge results remain. Retry timestamped text is public as redacted copies, not raw logs; original private logs exist locally. Email delivery is unverified. GPU samples are untouched. Text timestamps may lag buffered output. The historical notebook preserves first-runtime output and retry provenance; it is not a clean runnable notebook.

**Before release:** preserve durable weights/hashes; run the saved NF4 gate with verified numerics; compare safety cases across base/adapter/merged/NF4 and fix regressions without hiding failures; run 100-row evaluation plus shuffled-label/control comparisons; audit full masks and final dependencies; record a pre-run budget and host/GPU peaks; execute the replacement workflow and deliver original evidence. Then reconsider DON'T SHIP.

**AI disclosure/reflection:** ChatGPT proposed the setup, wrote code, operated Colab and assembled evidence. Actual errors/hashes/configs/outputs drove corrections to the dependency pin and token-length bug. No independent human rewrite or full manual scoring is claimed. This revision checked gate scope, peaks and hashes locally; CUDA execution remains pending. The surprise was attributing a gate to an artifact it never loaded. The concerns are missing safety cases, masks, controls, weights and host peaks: a polished report cannot replace them.
