# Model card — NOT RELEASED

**DON'T SHIP.** No new trained weights or public model artifact is released. Base: Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Historical data: selected English Databricks Dolly rows (CC BY-SA 3.0); model license Apache-2.0; original code MIT. See NOTICE.md and DATA_LICENSES.md.

Historical results are a small likelihood smoke test and a negative bf16 adapter gate, not a saved-NF4 safety evaluation. New configurations and verification code are development artifacts. S1–S7 results with CIs, durable weight hashes, a selected primary task, safety review and remote reload proof are **NOT AVAILABLE**. Intended use: inspect and reproduce an incomplete experiment, not serve this model to end users.

Only replace this status after `eval/ship_gate.py` records a real passing saved-artifact verdict; include the exact model hashes, data manifest, measured results and remaining limitations. No license opinion about ShareAlike's effect on trained weights is implied.
