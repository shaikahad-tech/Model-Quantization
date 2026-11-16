# Memory: reconstructed history and a new pre-run calculation

No timestamped pre-training worksheet survives from the historical run. The new JSON budget generator timestamps and hashes a config-specific estimate before training. GB is decimal; MiB is 2^20 bytes. Every estimate is labeled analytic or allowance; actual measurements remain separate.

Qwen dimensions: 28 layers, hidden1536, FFN8960, 12 query / 2 KV heads, vocabulary151936; base parameters 1,543,714,304. Linear weights: 1,310,195,712; tied embedding: 233,373,696; bias/norm: 144,896.

| Component | Kind | Calculation / bytes |
|---|---|---|
| Packed linears | Analytic | 1,310,195,712×0.5 = 655,097,856 |
| FP16 embedding | Analytic | 151936×1536×2 = 466,747,392 |
| FP16 bias/norm | Analytic | 144,896×2 = 289,792 |
| Double-quant stats | Analytic | One uint8 scale /64 weights + FP32 second-level /256 scale blocks = 20,791,680 |
| NF4 total | Analytic | 1,142,926,720; measured historical file 1,143,327,349 (~0.035% larger); floor before stats 1,122,135,040 (~1.122 GB) |
| FP32 nonquantized upcast | Expected | Additional 467,037,184; actual prepared-model dtype census must verify |
| q/v LoRA | Analytic | A=28×16×[(1536+1536)+(1536+256)]=2,179,072; FP32 weights/gradients 8,716,288 each |
| FP32 Adam moments | Analytic | A×8 = 17,432,576; no extra master copy assumed for FP32 adapters |
| Logits FP16 / FP32 | Analytic, B1 S512 | 155,582,464 / 311,164,928; may overlap |
| Checkpointed hidden anchor | Analytic | L×B×S×h×2 = 44,040,192 |
| Activations / backward / workspaces | Allowance | 2–3 GB at B1 S512, scaled with B×S; not a measured checkpointing model |
| CUDA / allocator overhead | Allowance | 0.7–1.4 GB; must calibrate on GPU |
| Merge host | Allowance + weights | Two FP16 copies ~6.175 GB +1–4 GB temporary buffers |
| Quantize host | Allowance + weights | FP16 source ~3.087 GB + packed~1.143 GB +1–3 GB temporary buffers |

`budget.py` computes envelopes for each config. Four-bit training can use FP32 embeddings and LoRA states, so the deployment-file size is not training VRAM. The all-linear recipe has more adapter state and a larger batch; the runner must respect the >13 GB budget warning and record any 2×8 fallback instead of silently changing effective batch.

| Historical observation | Kind | Result |
|---|---|---|
| Training | Raw sampled GPU | 5,395 MiB = 5.657 GB; 235 samples; mean 2.167 s |
| Merge | Raw sampled GPU | 115 MiB = 0.121 GB; 111 samples; mean 1.269 s |
| Quantize | Raw sampled GPU | 1,257 MiB = 1.318 GB; 185 samples; mean 1.348 s |
| CLI display, both VMs | Reported | GPU peak 4.5/14.6 GB; implementation/units not independently traced |
| Host RAM | Missing peak | A 6.8/12.67 GB UI snapshot is not a phase peak |

Both VMs' summaries report the same GPU maxima, but VM1's raw sampling files are unavailable. A plateau is consistent with retained allocations/allocator reserve; `nvidia-smi` alone cannot prove that attribution or minimum memory need. Sample intervals are observed means, including delays; nominal waits were 2 s / 1 s. Polling misses brief spikes. The new runner logs allocator allocated/reserved maxima, summed process-tree RSS (shared pages may be double-counted), compact GPU samples and phase times. Recovered replica: allocated/reserved 3.103/3.695 GB, sampled GPU peak 3,667 MiB, versus pre-run 4.656–6.512 GB envelope. See docs/replica_result.md.
