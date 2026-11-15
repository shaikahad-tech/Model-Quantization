import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.stats import wilson_ci,paired_bootstrap_diff,mcnemar_exact,noninferior,power_paired_binary


class StatsChecks(unittest.TestCase):
    def test_wilson(self):
        lo,hi=wilson_ci(18,40);self.assertAlmostEqual(lo,.307053078,places=6);self.assertAlmostEqual(hi,.601709083,places=6)
        self.assertEqual(wilson_ci(0,40)[0],0)
        with self.assertRaises(ValueError):wilson_ci(1,0)
    def test_paired_and_reproducible(self):
        a=[1,0,1,1];b=[0,0,0,0]
        self.assertEqual(paired_bootstrap_diff(a,b),paired_bootstrap_diff(a,b))
        self.assertEqual(paired_bootstrap_diff(a,a)['high'],0)
        self.assertEqual(paired_bootstrap_diff(a,b)['point'],.75)
        with self.assertRaises(ValueError):paired_bootstrap_diff([1],[1,2])
        with self.assertRaises(ValueError):paired_bootstrap_diff([float('nan')],[1])
    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact(0,0),1)
        self.assertAlmostEqual(mcnemar_exact(10,0),2/1024)
        self.assertEqual(mcnemar_exact(3,3),1)
        with self.assertRaises(ValueError):mcnemar_exact(-1,2)
    def test_noninferiority(self):
        self.assertTrue(noninferior([1]*30,[1]*30,.05)['passed'])
        self.assertFalse(noninferior([0]*30,[1]*30,.05)['passed'])
    def test_power(self):
        self.assertGreater(power_paired_binary(1000,.2,.05),power_paired_binary(30,.2,.05))
        self.assertAlmostEqual(power_paired_binary(100,.1,.1),.05)
