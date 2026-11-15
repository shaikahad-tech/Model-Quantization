# Historical gate: exact scope and limits

The redacted notebook command is `trainer ship --base Qwen/Qwen2.5-1.5B-Instruct --adapter output --task-eval ship_tasks.jsonl --output ship_verdict.json --device cuda`. The alias is not executable software in this repository. The retry log spans 08:52:57–09:07:39 UTC on 11 November, exit 2. Its banner and JSON specify bfloat16; the saved gate evaluated an active adapter, not the merged or NF4 export. The banner also offered `--config` to reuse training quantization. That option was not used.

The base was already 6/6 on contains-match tasks. A strict improvement above 1.0 is impossible, so `task_win` could not pass with those inputs. That is a defect in task selection, not proof every possible tuned model fails to learn.

The mini-suite denominators are unknown. Forty is the smallest common denominator consistent with the displayed fractions; larger denominators are also possible. **Conditionally assuming n=40**, safety corresponds to 40/40→18/40. Wilson 95% intervals are approximately [0.912,1.000] and [0.307,0.602]. The common-sense and tool-call differences are only three cases each; they are inconclusive at that assumed size. Overlapping/nonoverlapping Wilson intervals are descriptive, not a substitute for saved paired cases. These inferred counts are not recorded observations, and no unconditional significance claim follows.

| Old gate could show | Old gate could not establish |
|---|---|
| Six task scores tied and aggregate mini scores changed under bf16 adapter inference | Task improvement on a non-saturated representative test |
| Safety point estimate dropped 1.00→0.45 | Why particular safety cases failed or whether labels were reliable |
| Common-sense/tool-call point estimates fell | Their significance without denominators and paired outcomes |
| The adapter-input gate rejected task_win | Safety or quality of the saved NF4 artifact |
| The CLI advertised a training-quantization config option | Native T4 bf16 support or actual module/dtype behavior |

Sources: [log](../evidence/ship_log_redacted.txt), [verdict](../evidence/ship_verdict_redacted.json), [six tasks](../data/ship_tasks.jsonl). The [new gate](gate_v2.md) binds results to saved artifacts and keeps missing evidence negative.
