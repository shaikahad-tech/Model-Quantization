# Exact release-gate scope

Historical notebook cell 20 records this command, with the public CLI alias (`trainer` is a redaction, not an executable supplied here):

```sh
trainer ship --base Qwen/Qwen2.5-1.5B-Instruct --adapter output --task-eval ship_tasks.jsonl --output ship_verdict.json --device cuda
```

The retry [log](../evidence/ship_log_redacted.txt) starts 2026-11-11 08:52:57 UTC and ends 09:07:39 UTC, exit 2. Its banner says full precision bfloat16. The [verdict](../evidence/ship_verdict_redacted.json) agrees: `numerics=bfloat16`, `failed_rule=task_win`, `noise_floor=null`.

| Item | Evidence-supported conclusion |
|---|---|
| Tuned input | Base + active adapter `output`; neither `merged` nor `quantized` |
| Tasks | [Six contains-match prompts](../data/ship_tasks.jsonl); 6/6 each; no strict win |
| General checks | Arithmetic, common sense, JSON format, instruction, MMLU, over-refusal, safety, tool calling; eight aggregate scores saved |
| Regressions | Common sense −0.075, safety −0.55, tool calling −0.075; threshold 0.05 |
| Unknown | Mini-suite case denominators, per-case generations, actual module census, repeated-run uncertainty |
| Not run | Gate on merged export, gate on saved NF4, four-artifact per-case safety comparison |

The gate completed for its adapter input. It was **not a partial merged-model gate**; describing it that way would introduce another error. Its scores cannot be attributed to the saved NF4 artifact. A separate `python -u verify_artifacts.py` smoke test loaded three artifacts and evaluated eight rows / 457 targets each. No 100-row result exists.
