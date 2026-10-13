import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulations.ai2thor.execution import build_action_sequence


class Map6ConfigurationTests(unittest.TestCase):
    def test_cooperative_fridge_chain_and_independent_coffee_chain(self):
        plan = json.loads((ROOT / "maps_ai2thor/map6/actions.json").read_text())
        commands = build_action_sequence(plan)
        self.assertEqual(len(commands), 7)
        tasks = {task['id']: task for task in plan['tasks']}
        self.assertEqual(plan['robot_plans'], {'R1': [1,5], 'R2': [2,4], 'R3': [3,6,7]})
        self.assertTrue(all(tasks[tid]['node'] == [] for tid in (1,2,3)))
        self.assertEqual(tasks[4]['node'], [1,2])
        self.assertEqual(tasks[5]['node'], [4])
        self.assertEqual(tasks[6]['node'], [3])
        self.assertEqual(tasks[6]['duration'], 2.)
        self.assertEqual(tasks[7]['node'], [6])
        self.assertEqual([tasks[tid]['action'] for tid in (3,6,7)], ['toggle_on','wait','toggle_off'])
        self.assertFalse(any('wait_for_on_conflict' in c for c in commands))

    def test_different_kitchen_and_three_distinct_starts(self):
        scene = json.loads((ROOT / "maps_ai2thor/map6/scene.json").read_text())
        self.assertEqual(scene['scene'], 'FloorPlan2')
        self.assertEqual(scene['agent_count'], 3)
        self.assertEqual(len({(p['position']['x'], p['position']['z'])
                              for p in scene['robot_placements']}), 3)
        apple = next(p for p in scene['object_placements'] if p['name'] == 'Apple')
        self.assertEqual(apple['position'], {'x': .45, 'y': 1.06, 'z': -1.44})
        self.assertTrue(apple['apply'])


if __name__ == '__main__':
    unittest.main()
