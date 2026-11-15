# AI assistance and reflection

ChatGPT helped select the experiment, wrote notebook cells and verification code, proposed memory arithmetic, operated the authorized Colab session, and assembled documentation. Accepted suggestions included the small Qwen model, English instruction data, LoRA and the three-artifact comparison. Tool-based checks inspected actual errors, sizes, saved configs, hashes, GPU samples and smoke-test outputs. The incompatible version pin and token-length calculation were corrected after observed failures. These are AI-assisted actions; the record does not establish that the author independently rewrote or reviewed every line or answer.

For this revision, ChatGPT compared the gate command with its log/JSON, recalculated sampled GPU maxima and hashes, identified the newline mismatch, and removed overclaims about gate scope, learning and a documented pre-training budget. New code was checked locally but has not run on CUDA. Missing results and email confirmations were not invented.

**Surprise and remaining concern.** A successful export and lower small-slice NLL coexisted with a large safety drop in a different inference configuration. More surprising was how easily a gate result could be described as applying to a file it never loaded. Artifact identity, numerics and per-case evidence matter as much as a completed run. Remaining concerns are the response mask, missing negative controls, lost weights, absent safety diagnoses, unmeasured host peaks and treating a polished report as experimental proof.


## This revision

ChatGPT/Codex read the owner-supplied audit, checked claims against files, wrote the data/statistics/scorer/verifier/persistence/gate/tooling code, prepared configs and notebook, and ran local tests through terminal tools. No other AI system was invoked by this agent. The origin of the supplied audit is not independently established. The owner chose MIT; the agent did not choose a license on the owner's behalf.

The actual CPU run caught a bug in the newly written wrong-scale merge test: it mistook Qwen attention's scalar `scaling` for a LoRA scaling dictionary. The type guard was corrected and both the failed transcript and subsequent result were retained. A pinned real tokenizer was downloaded and used to check masks/template, rather than assuming the audit's claims. Computed duplicate counts distinguish extra occurrences from total matching rows; observed mean sample intervals differ from nominal sleep intervals. Conditional n=40 statistics were kept conditional.

NOT EXECUTED unless a later status/evidence entry states otherwise: Colab replica, candidate task baselines, source-specific dataset conversion, v2 data generation, LR sweep, three seeds, shuffled-label training, full GPU variants, human refusal-label audit, durable weight upload and remote GPU reload. Browser inspection confirmed a signed-in Colab page; it did not establish a completed CUDA experiment. No independent human line-by-line review or full manual scoring is claimed.
