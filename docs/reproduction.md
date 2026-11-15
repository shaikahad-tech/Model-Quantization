# Reproduction: historical evidence versus new workflow

The historical redacted notebook is an execution record, not runnable software. Product names cannot be inferred from the public aliases. Its original dependency snapshot is incomplete. The model weights were lost. Reusing the original eight-row script would require recovering `output/`, `merged/` and `quantized/` plus the original environment; do not mistake a new run for recovery of those artifacts.

## New public workflow — GPU execution pending

The replacement uses [Transformers 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer), [PEFT LoRA](https://huggingface.co/docs/peft/v0.15.0/package_reference/lora) and [bitsandbytes NF4](https://huggingface.co/docs/transformers/v4.51.3/quantization/bitsandbytes). This makes masking, adapter merging and serialized 4-bit reload visible without proprietary command aliases. It is a new implementation, **not a bit-for-bit reproduction** of the original CLI. Different optimizer defaults, explicit train/validation order and masking can change results.

Use Python 3.11 on Linux with an NVIDIA CUDA GPU (T4-class 16 GB), adequate host RAM (~16 GB or more) and ~20 GB free disk. No Mac MPS support is claimed. In a fresh environment at the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-rerun.txt
python -m unittest discover -s code -p 'test_*.py' -v
bash reproduce.sh
```

The top-level packages are pinned; this is not a complete transitive lockfile. Actual installed versions are saved in each run. CUDA/driver compatibility and the full pipeline have **not been validated here**. Installation may need adjustment on the selected GPU image; preserve any errors and record changes.

`reproduce.sh` creates a fresh run directory and refuses overwrite. It downloads the pinned base revision, validates exact data hashes and all 1,500 tokenized rows, trains q/v LoRA (r16/alpha32), saves the adapter, merges into a fresh FP16 base on CPU, quantizes the saved merge to NF4/double-quant, and reloads the saved export for verification. The first 1,260 training-file rows train and the last 140 validate; the 100 heldouts stay separate. Batch 1 requires no padding; prompt labels are −100 and assistant EOS is supervised. Training and evaluation use the same two-newline context formatter.

Each phase saves UTC capture-time logs, untouched `nvidia-smi` samples and allocator maxima. The memory worksheet is copied/timestamped before training. **Its allowance is inherited from the older run, not newly calibrated; revise it for the new dtype/kernel choices before relying on it.** Host RSS peaks are not instrumented. Buffered child output and GPU polling remain limitations. Preserve the whole directory if a stage fails; partial outputs are not success evidence.

## Extended check — no result yet

[verify_extended.py](../code/verify_extended.py) evaluates 100 heldouts on base, no-op LoRA, randomized-B LoRA, trained active adapter, merged and saved NF4. Randomized B retains trained A and matches each B norm; it is one seeded control, not shuffled-label training. Outputs include per-row answer NLL/token counts, six task generations/contains scores, full fixed-prompt logits, all 56 target-matrix deltas and merge residuals, tokenizer identity, and the expected 196 quantized linear-module names / compute dtypes.

No-op logits must match base at atol/rtol 1e-5. Merge comparison allows FP16 rounding (atol 1e-4, rtol 1e-3); these provisional tolerances need calibration against true no-op/wrong-scale artifacts on hardware. NF4 must expose the expected converted modules and FP16 compute. A diagnostic training-signal check requires finite NLL, 56 nonzero B tensors, a relative delta above 1e-6 in each target matrix, and active-adapter NLL below base/no-op/randomized-B. It fails if these signals are absent; passing is not a shipping certificate or causal proof of useful learning. Logit differences are reported. Partial JSON is written after each evaluated artifact; a crash still leaves phase logs. Final JSON is written before structural assertions so a failed check is preserved.

Blind spots: no shuffled-label-trained adapter; one random seed; one fixed logit prompt; no bootstrap/uncertainty test; no safety rubric or original safety cases; six saturated contains tasks are not instruction-quality evaluation. A cleverly wrong checkpoint may pass. No real learning, safety or release-ready claim follows from this code until it runs and the missing tests are completed. The output decision remains DON'T SHIP.

Historical code/output stays unchanged. New outputs go under `runs/`, which is excluded from Git to avoid publishing weights accidentally. Publish selected reviewed logs and durable model links after the actual run.
