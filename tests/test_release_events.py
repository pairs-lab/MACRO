import io
import time
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from simulations.standalone import planner
from simulations.standalone.app import _compute_conflict_plan, RobotState
from simulations.standalone.release_events import (
    predict_release_events, continuation_motion_is_clear,
)
from simulations.standalone.relocation_search import _remaining_dag_cost
import test_runtime_policies as fixtures


class ReleaseEventTests(unittest.TestCase):
    @staticmethod
    def candidate():
        return {'strategy': 'relocate', 'priority_robot': 'R1',
                'relocations': [{'robot_id': 'R2', 'path': [(1, 0), (1, 1)],
                                 'delay': 1., 'distance': 1.}],
                'moved_agents': ['R2'], 'delay_by_robot': {'R1': 1., 'R2': 7.},
                'preferred_paths': {'R1': [(x, 0) for x in range(7)],
                                    'R2': [(1, 1), (1, 0), (0, 0), (-1, 0)]}}

    def test_safe_release_does_not_wait_for_priority_goal(self):
        candidate = self.candidate()
        before = deepcopy(candidate)
        robots = {'R1': SimpleNamespace(cell=(0, 0)), 'R2': SimpleNamespace(cell=(1, 0))}
        delays, info = predict_release_events(candidate, robots, .2, 1.)
        self.assertEqual(delays, {'R1': 0., 'R2': 1.})
        self.assertTrue(info['early_staging_overlap_credited'])
        self.assertEqual(info['unknown_agents'], [])
        self.assertEqual(candidate, before)
        self.assertFalse(continuation_motion_is_clear(
            candidate['preferred_paths']['R1'], candidate['preferred_paths']['R2'], .2, 1.))
        self.assertTrue(continuation_motion_is_clear(
            candidate['preferred_paths']['R1'][1:], candidate['preferred_paths']['R2'], .2, 1.))

    def test_joint_staging_barrier_remains_serial(self):
        candidate = self.candidate()
        candidate['relocations'].insert(0, {'robot_id': 'R1', 'path': [(0, -1), (0, 0)],
                                          'delay': 1., 'distance': 1.})
        delays, info = predict_release_events(candidate, {}, .2, 1.)
        self.assertEqual(delays['R1'], 2.)
        self.assertGreaterEqual(delays['R2'], 2.)
        self.assertFalse(info['early_staging_overlap_credited'])

    def test_unknown_clearance_is_reported_not_certified(self):
        candidate = self.candidate()
        candidate['preferred_paths']['R2'] = [(1, 1), (6, 0)]
        candidate['relocations'].append({'robot_id': 'R3', 'path': [(8, 0), (8, 1)],
                                         'delay': 1., 'distance': 1.})
        candidate['moved_agents'].append('R3')
        _, info = predict_release_events(candidate, {}, 2., 1.)
        self.assertIn('R2', info['unknown_agents'])
        self.assertFalse(info['future_conflicts_modelled'])

    def test_all_policies_keep_same_candidate_geometry(self):
        pools = []
        for policy in ('dag_aware', 'local_distance', 'local_delay'):
            with redirect_stdout(io.StringIO()):
                plan = _compute_conflict_plan(**fixtures.fixture(), selection_policy=policy)
            pools.append([c['geometry_key'] for c in plan['search_stats']['candidate_evaluations']])
        self.assertEqual(pools[0], pools[1])
        self.assertEqual(pools[1], pools[2])





class GeometryCacheTests(unittest.TestCase):
    def test_cache_preserves_results_and_returns_independent_routes(self):
        grid = frozenset((x, y) for x in range(5) for y in range(5))
        uncached = planner.astar(grid, (0, 0), (4, 4))
        @planner.with_geometry_cache
        def run():
            first = planner.astar(grid, (0, 0), (4, 4))
            first.append((99, 99))
            second = planner.astar(grid, (0, 0), (4, 4))
            return {'route': second, 'search_stats': {}}
        result = run()
        self.assertEqual(result['route'], uncached)
        self.assertEqual(result['search_stats']['geometry_cache'], {'hits': 1, 'misses': 1})

    def test_cache_keys_include_dynamic_obstacles_and_scope(self):
        grid = {(x, y) for x in range(4) for y in range(4)}
        @planner.with_geometry_cache
        def run():
            obstacles = {(1, 1)}
            first = planner.astar(grid, (0, 0), (3, 3), obstacles)
            obstacles.add((2, 2))
            second = planner.astar(grid, (0, 0), (3, 3), obstacles)
            self.assertNotEqual(first, second)
            return {'search_stats': {}}
        self.assertEqual(run()['search_stats']['geometry_cache']['hits'], 0)
        self.assertEqual(run()['search_stats']['geometry_cache']['hits'], 0)


if __name__ == '__main__':
    unittest.main()
