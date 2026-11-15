import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dev_selection import selection_metrics

class SelectionTests(unittest.TestCase):
    def test_safety_overrides_primary_gain(self):
        base={'primary':[0]*50,'safety_harmful':[1]*50,'safety_benign':[0]*50}
        improved=dict(base,primary=[1]*50)
        self.assertEqual(selection_metrics(improved,base)['selection_score'],1)
        unsafe=dict(improved,safety_harmful=[0]*50)
        self.assertEqual(selection_metrics(unsafe,base)['selection_score'],-1)
        overrefusal=dict(improved,safety_benign=[1]*50)
        self.assertFalse(selection_metrics(overrefusal,base)['safety_eligible'])
    def test_unpaired_rejected(self):
        with self.assertRaises(ValueError):selection_metrics({'primary':[1]}, {'primary':[1,0]})
