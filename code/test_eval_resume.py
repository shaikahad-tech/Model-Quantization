import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.run_eval import validate_completed
from variants import artifact_fingerprint

class EvalResumeChecks(unittest.TestCase):
    def test_completed_evidence_must_match(self):
        item={'id':'a','prompt':'Answer','scorer':'contains','expected':'yes'}
        from eval.scorers import score
        row=dict(id='a',item=item,prompt='Answer',variant='merged_nf4',suite='primary',suite_sha256='suite',artifact_fingerprint='weights',generation='yes',score=score('yes',item))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.jsonl';p.write_text(json.dumps(row)+'\n')
            validate_completed(p,[item],{'content_sha256':'suite'},'merged_nf4','primary','weights')
            with self.assertRaises(ValueError):validate_completed(p,[item],{'content_sha256':'suite'},'merged_nf4','primary','changed')
            with self.assertRaises(ValueError):validate_completed(p,[item,item],{'content_sha256':'suite'},'merged_nf4','primary','weights')
    def test_artifact_change_invalidates_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'quantized';p.mkdir();(p/'weights').write_bytes(b'a')
            before=artifact_fingerprint('merged_nf4',Path(d));(p/'weights').write_bytes(b'b')
            self.assertNotEqual(before,artifact_fingerprint('merged_nf4',Path(d)))
