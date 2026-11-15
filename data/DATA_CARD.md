# Historical selected Dolly data

Source: Databricks Dolly 15k, English, CC BY-SA 3.0. See NOTICE.md and DATA_LICENSES.md. The source revision was not captured; the selected bytes are retained verbatim in `sampled_dataset.zip`.

The source was shuffled with seed 20261111. The first 1,500 unique complete rows whose rendered user+assistant exchange had at most 480 tokens were retained, mapped to instruction/input/output, and split into 1,400 input-to-training and 100 heldouts. Context was joined with two newlines. Category was dropped. This systematically excludes longer instructions/responses; it is not a random sample of all Dolly lengths. Internal validation consumed 10% of the training input.

Hashes are verified by `code/experiment_data.py`. `tools/metrics.py` computes exact overlap, duplicate instruction counts and Jaccard≥0.8 train/holdout instruction pairs; these are warnings about split quality, not proof of semantic independence. The historical six-item `ship_tasks.jsonl` is deprecated: the base already scored 6/6, making a strict-win gate impossible.

`fixed_prompts.jsonl` contains 20 new, simple diagnostic prompts authored with AI assistance for logit comparisons; it is not an accuracy benchmark or historical dataset. New v2 data must never overwrite the selected ZIP.
