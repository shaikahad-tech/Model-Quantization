# Third-party attribution

The MIT license covers this repository's original code and documentation; it does not relicense third-party data, model weights, or third-party excerpts in experiment logs.

**Databricks Dolly 15k** — Databricks, Inc. and its contributors, English version 1.0. Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k . The redistributed selected rows in `data/sampled_dataset.zip` are provided under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode). Changes: deterministic shuffle and selection, exact-row deduplication, a ≤480 rendered-token filter, mapping instruction/context/response to instruction/input/output, and removal of category. No endorsement is implied. See `data/DATA_CARD.md` for retained hashes and limitations. The original dataset includes some Wikipedia-derived contexts; retain its source attribution and license obligations.

**Qwen2.5-1.5B-Instruct** — Qwen, pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The [pinned LICENSE](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) specifies Apache-2.0. Weights are not redistributed in this repository. Downloading or publishing weights must retain the applicable license and notices.

No new external evaluation or training dataset is redistributed yet. New sources must be resolved, licensed and recorded before use. Not legal advice: how ShareAlike applies to derived model weights is unsettled and is not resolved by licensing the repository code under MIT.
