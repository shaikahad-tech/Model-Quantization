# Public redactions and original evidence

Public text logs and the historical notebook are **redacted copies, not raw logs**. The original product name uses neutral aliases. This revision additionally replaces the leaked package specifier with `[vendor-package-redacted][train]` and the leaked repository URL with `[vendor-repo-redacted]`, including notebook hyperlink escapes. Exactly the specified notebook lines and one initial-failure-log line were changed; cell order, execution counts and all numerical results remain intact. Raw `nvidia_smi_*.txt` files are unchanged.

**History choice: preserved.** No force-push or history rewrite was performed. Old strings remain recoverable from earlier commits and old downloads of the ZIP. Current files and current ZIP are corrected. Removing historical copies would require a separately authorized history rewrite or repository recreation and cannot recall copies others already downloaded.

Private delivery checklist (not a claim of sending):

- Recipient: submission team; contact address retained privately to honor public redaction.
- Date sent: **UNVERIFIED**.
- Private prepared file: `Model_Quantization_Evidence_for_Team.zip`.
- SHA-256 of the existing private ZIP: `82291036da2ed8b71b4e4dce051b3de712c4d350ffc4e816df8dd0a7085c3602` (checked locally during this revision).
- Acknowledgement of this package received: **NO VERIFIED ACKNOWLEDGEMENT**.

Original retry logs/config and the original notebook remain local. The historical notebook records failures and is not a clean run-all workflow. New CPU test failure transcripts are retained in the ignored local run directory; public copies normalize the workspace path and are labeled as such. No failed training/evaluation cases were discarded to improve a score.
