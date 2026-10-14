"""An opt-in execution evaluator must precede selection; grid defaults stay put."""
from copy import deepcopy
import unittest
from tests.test_runtime_policies import fixture
from simulations.grid.app import _compute_conflict_plan


class ExecutionEvaluatorTests(unittest.TestCase):
    def test_none_preserves_grid_selection_and_pool(self):
        first = _compute_conflict_plan(**deepcopy(fixture()))
        second = _compute_conflict_plan(**deepcopy(fixture()), execution_candidate_refiner=None)
        self.assertEqual(first['candidate_id'], second['candidate_id'])
        self.assertEqual(first['estimated_makespan'], second['estimated_makespan'])
        self.assertEqual(first['planned_paths'], second['planned_paths'])

    def test_evaluator_sees_pool_and_controls_selection(self):
        seen = []
        def evaluate(candidates, stats, deadline, finishes, distance):
            seen.extend(c['candidate_id'] for c in candidates)
            wanted = candidates[-1]['candidate_id']
            for candidate in candidates:
                candidate['estimated_makespan'] = 0. if candidate['candidate_id'] == wanted else 100.
            return candidates
        selected = _compute_conflict_plan(**deepcopy(fixture()), execution_candidate_refiner=evaluate)
        self.assertGreater(len(seen), 1)
        self.assertEqual(selected['candidate_id'], seen[-1])

    def test_evaluator_timeout_never_selects_an_unmatched_grid_plan(self):
        def evaluate(*args):
            raise TimeoutError('execution model budget exhausted')
        with self.assertRaisesRegex(TimeoutError, 'execution model'):
            _compute_conflict_plan(**deepcopy(fixture()), execution_candidate_refiner=evaluate)

    def test_only_dag_aware_accepts_execution_evaluator(self):
        with self.assertRaises(ValueError):
            _compute_conflict_plan(**fixture(), selection_policy='local_delay',
                                   execution_candidate_refiner=lambda *args: [])
