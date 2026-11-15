import random,unittest
from build_dataset import select_replay

class ReplaySelectionChecks(unittest.TestCase):
    def test_rare_required_categories_survive_sampling(self):
        rows=[{'category':'dolly','id':i} for i in range(500)]+[{'category':'harmful','id':500},{'category':'benign_near_miss','id':501}]
        selected=select_replay(rows,random.Random(42))
        self.assertEqual(len(selected),420)
        self.assertEqual(len({r['id'] for r in selected}),420)
        self.assertEqual({r['category'] for r in selected},{'dolly','harmful','benign_near_miss'})
        with self.assertRaises(ValueError):select_replay(rows[:500],random.Random(42))
