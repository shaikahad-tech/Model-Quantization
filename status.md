# Status — updated 15 November 2026

**DON'T SHIP. Submission incomplete.** Historical run: 11 November. Documentation audit and new code are later work.

| Requirement | Status |
|---|---|
| Train / merge / NF4 | Retry logs exist; weights lost after runtime reset |
| Historical verification | 8 rows / 457 targets; smoke test, not learning proof |
| Historical gate | Base + active adapter, bf16; exit 2/task_win; safety, common-sense and tool-call regressions |
| Exact NF4 gate | Not run |
| 100-row evaluation / controls | Not run; new code is not evidence |
| Four-artifact safety cases | Not run; original per-case outputs missing |
| Final preflight / full loss-mask audit | Missing |
| Memory | Retrospective arithmetic + sampled GPU peaks; no retained pre-training worksheet or host peak |
| Raw-log delivery | Originals local; ZIP prepared; email delivery unverified |
| Runnable workflow | New public code; complete GPU execution pending |
| Documentation | Scope, memory table, failure matrix, AI disclosure and report corrected |

The earlier 100-row extension is retained as an unexecuted historical draft. The notebook has first-runtime output plus explicit retry provenance; retry results are separate files. Failed results remain.

The Mac used for this revision has no NVIDIA CUDA device. No GPU rerun was performed; no missing weights or results were fabricated. See [release blockers](README.md#what-must-change-before-ship).
