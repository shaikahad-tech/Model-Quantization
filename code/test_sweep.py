import unittest
from sweep import choose
class SweepChecks(unittest.TestCase):
    def test_safety_and_tie_break(self):
        rows=[{'lr':.0002,'primary_score':.9,'safety_eligible':False},{'lr':.0001,'primary_score':.7,'safety_eligible':True},{'lr':.00005,'primary_score':.7,'safety_eligible':True}]
        self.assertEqual(choose(rows)['lr'],.00005)
    def test_no_eligible_stops(self):
        with self.assertRaises(ValueError):choose([{'lr':.0001,'primary_score':.9,'safety_eligible':False}])
