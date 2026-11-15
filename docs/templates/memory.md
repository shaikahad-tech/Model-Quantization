# Memory: reconstructed history and a new pre-run calculation

No timestamped pre-training worksheet survives from the historical run. The new JSON budget generator timestamps and hashes a config-specific estimate before training. GB is decimal; MiB is 2^20 bytes. Every estimate is labeled analytic or allowance; actual measurements remain separate.

Qwen dimensions: 28 layers, hidden1536, FFN8960, 12 query / 2 KV heads, vocabulary151936; base parameters {{parameters:,}}. Linear weights: {{linear_parameters:,}}; tied embedding: {{embedding_parameters:,}}; bias/norm: {{bias_norm_parameters:,}}.

| Component | Kind | Calculation / bytes |
|---|---|---|
| Packed linears | Analytic | {{linear_parameters:,}}×0.5 = {{packed_linear_bytes:,}} |
| FP16 embedding | Analytic | 151936×1536×2 = {{embedding_fp16_bytes:,}} |
| FP16 bias/norm | Analytic | {{bias_norm_parameters:,}}×2 = {{bias_norm_fp16_bytes:,}} |
| Double-quant stats | Analytic | One uint8 scale /64 weights + FP32 second-level /256 scale blocks = {{double_quant_stats_bytes:,}} |
| NF4 total | Analytic | {{nf4_analytic_bytes:,}}; measured historical file 1,143,327,349 (~0.035% larger); floor before stats {{nf4_floor_without_stats_bytes:,}} (~1.122 GB) |
| FP32 nonquantized upcast | Expected | Additional {{expected_fp32_nonquantized_upcast_bytes:,}}; actual prepared-model dtype census must verify |
| q/v LoRA | Analytic | A=28×16×[(1536+1536)+(1536+256)]={{trainable_parameters:,}}; FP32 weights/gradients {{adapter_weights_bytes:,}} each |
| FP32 Adam moments | Analytic | A×8 = {{optimizer_state_bytes:,}}; no extra master copy assumed for FP32 adapters |
| Logits FP16 / FP32 | Analytic, B1 S512 | {{logits_fp16_bytes:,}} / {{logits_fp32_bytes:,}}; may overlap |
| Checkpointed hidden anchor | Analytic | L×B×S×h×2 = {{checkpointed_hidden_anchor_bytes:,}} |
| Activations / backward / workspaces | Allowance | 2–3 GB at B1 S512, scaled with B×S; not a measured checkpointing model |
| CUDA / allocator overhead | Allowance | 0.7–1.4 GB; must calibrate on GPU |
| Merge host | Allowance + weights | Two FP16 copies ~6.175 GB +1–4 GB temporary buffers |
| Quantize host | Allowance + weights | FP16 source ~3.087 GB + packed~1.143 GB +1–3 GB temporary buffers |

`budget.py` computes envelopes for each config. Four-bit training can use FP32 embeddings and LoRA states, so the deployment-file size is not training VRAM. The all-linear recipe has more adapter state and a larger batch; the runner must respect the >13 GB budget warning and record any 2×8 fallback instead of silently changing effective batch.

| Historical observation | Kind | Result |
|---|---|---|
| Training | Raw sampled GPU | {{train_peak_mib:,}} MiB = {{train_peak_gb:.3f}} GB; {{train_samples}} samples; mean {{train_interval_seconds:.3f}} s |
| Merge | Raw sampled GPU | {{merge_peak_mib}} MiB = {{merge_peak_gb:.3f}} GB; {{merge_samples}} samples; mean {{merge_interval_seconds:.3f}} s |
| Quantize | Raw sampled GPU | {{quantize_peak_mib:,}} MiB = {{quantize_peak_gb:.3f}} GB; {{quantize_samples}} samples; mean {{quantize_interval_seconds:.3f}} s |
| CLI display, both VMs | Reported | GPU peak {{vm2_cli_peak_gb:.1f}}/{{vm2_cli_total_gb:.1f}} GB; implementation/units not independently traced |
| Host RAM | Missing peak | A 6.8/12.67 GB UI snapshot is not a phase peak |

Both VMs' summaries report the same GPU maxima, but VM1's raw sampling files are unavailable. A plateau is consistent with retained allocations/allocator reserve; `nvidia-smi` alone cannot prove that attribution or minimum memory need. Sample intervals are observed means, including delays; nominal waits were 2 s / 1 s. Polling misses brief spikes. The new runner logs allocator allocated/reserved maxima, summed process-tree RSS (shared pages may be double-counted), compact GPU samples and phase times. None of these new GPU measurements exists yet.
