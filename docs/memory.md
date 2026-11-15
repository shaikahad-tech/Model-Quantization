# Memory reconstruction and measurement limits

No timestamped pre-training worksheet survives. This is retrospective arithmetic, not evidence the timing requirement was met. GB means 1e9 bytes; MiB means 2^20 bytes.

Qwen: L=28, hidden h=1536, 12 query heads, 2 KV heads, V=151936, batch B=1, sequence S=512; rounded parameter count P=1.544 billion. LoRA r=16 on q/v: A=28×16×[(1536+1536)+(1536+256)]=2,179,072 parameters. Frozen base weights have no gradient/Adam states.

| Component | Formula | GB |
|---|---|---:|
| FP16 base | P×2 | 3.088 |
| Ideal packed 4-bit lower bound | P×0.5 | 0.772 |
| Actual quantized file-size proxy | 1,143,327,349 bytes | 1.143; not VRAM measurement |
| LoRA weights / gradients, FP16 | A×2 each | 0.00436 each |
| Adam moments | A×8 | 0.01743 |
| Possible FP32 master | A×4 | 0.00872 |
| Logits FP16 / FP32 | B×S×V×2 / ×4 | 0.156 / 0.311 |
| One hidden tensor/layer | L×B×S×h×2 | 0.0440 |
| One dense attention score tensor/layer | L×B×12×S²×2 | 0.1762 |
| Saved activations/backward/workspace | 0.0440 + 0.1762 + 1.8–2.8 allowance | ~2–3 |
| CUDA/allocator reserve | Unmeasured allowance | 0.7–1.4 |

The activation formulas are shape anchors, not simultaneous allocations. Checkpointing, attention kernels, FFN intermediates and upcasts change lifetimes. FP32 adapter weights/gradients would double those small terms. File size includes metadata/high-precision tensors. Approximate total: 1.143+0.035+(0.156–0.467)+(2–3)+(0.7–1.4) ≈4–6 GB.

| Phase | Estimate | Actual sampled GPU maximum | Gap / limits |
|---|---|---|---|
| Train | 4–6 GB GPU | 5,395 MiB = 5.657 GB; 235 samples / 2 s | Within band, 0.343 GB below upper bound; no allocator breakdown |
| Merge | 2×3.088 + 1–4 buffers ≈7–10 GB host; 0–1 GB GPU | 115 MiB = 0.121 GB; 111 samples / 1 s | CPU path explains low GPU; host peak missing |
| Quantize | 3.088 source + 1.143 packed + 1–3 buffers ≈5.2–7.2 GB host; ~1–2 GB GPU allowance | 1,257 MiB = 1.318 GB; 185 samples / 1 s | Device packed weights/context plausible; host peak missing |

A Colab UI snapshot of 6.8/12.67 GB host memory is not a phase peak. GPU samples are lower bounds on maxima; short peaks can be missed. Text timestamps measure capture time and can lag buffered subprocess output. No independent tokens/second measurement was retained. Rerun with process-tree RSS, GPU sampling, allocated/reserved maxima, phase boundaries, actual nonpadding target counts and synchronized elapsed time.

Raw samples: [train](../evidence/nvidia_smi_train.txt), [merge](../evidence/nvidia_smi_merge.txt), [quantize](../evidence/nvidia_smi_quantize.txt).
