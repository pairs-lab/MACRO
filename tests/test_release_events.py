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
    predict_release_events, refine_release_events, continuation_motion_is_clear,
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

    def test_event_dag_incremental_matches_full_for_every_candidate(self):
        snapshot = fixtures.fixture()
        captured = []
        from simulations.standalone.app import _make_release_refiner
        def capture(*args, **kwargs):
            real = _make_release_refiner(*args, **kwargs)
            def refine(candidates, *extra):
                result = real(candidates, *extra)
                captured.extend(deepcopy(result))
                return result
            return refine
        with patch('simulations.standalone.app._make_release_refiner', side_effect=capture), redirect_stdout(io.StringIO()):
            _compute_conflict_plan(**snapshot)
        robots = {r.robot_id: r for r in snapshot['dag_robots']}
        initial = {rid: r.cell for rid, r in robots.items()}
        for candidate in captured:
            positions = dict(initial)
            positions.update(candidate.get('final_positions', {}))
            paths = dict(candidate['preferred_paths'])
            paths.update(candidate.get('release_prediction', {}).get('continuation_paths', {}))
            result = _remaining_dag_cost(
                tasks=snapshot['tasks'], task_status=snapshot['task_status'],
                robots_by_id=robots, positions=positions,
                preferred_paths=paths, objects=snapshot['objects'],
                navigation_cells=snapshot['navigation_cells'], grid_size=1.,
                speed_cells_per_second=2., relocation_delay_by_robot=candidate['delay_by_robot'],
                include_travel=True)
            self.assertEqual(candidate['estimated_makespan'], result[0])
        self.assertTrue(captured)

    def test_certified_incremental_reuses_independent_branch(self):
        candidate = self.candidate()
        candidate['preferred_paths']['R2'] = [(1, 1), (1, 2)]
        candidate.update(estimated_makespan=10., total_distance=7., total_delay=7.,
                         final_positions={'R1': (0, 0), 'R2': (1, 1)})
        tasks = [dict(id=1, robot='R1', action='toggle_on', object='a', node=[], duration=1.),
                 dict(id=2, robot='R2', action='toggle_on', object='b', node=[], duration=1.),
                 dict(id=3, robot='R3', action='toggle_on', object='c', node=[], duration=10.)]
        robots = [RobotState('R1', (0, 0), (0, 0, 0), goal=(6, 0),
                             path=[(x, 0) for x in range(7)], phase='moving', current_task=tasks[0]),
                  RobotState('R2', (1, 0), (0, 0, 0), goal=(1, 2),
                             path=[(1, 0), (1, 1), (1, 2)], phase='moving', current_task=tasks[1]),
                  RobotState('R3', (8, 3), (0, 0, 0), goal=(8, 3),
                             path=[(8, 3)], phase='acting', current_task=tasks[2], action_remaining=10.)]
        grid = frozenset((x, y) for x in range(-1, 10) for y in range(5))
        objects = {name: dict(cell=goal, cells=frozenset({goal}), interaction_cells=frozenset({goal}))
                   for name, goal in (('a', (6, 0)), ('b', (1, 2)), ('c', (8, 3)))}
        status = {1: 'running', 2: 'running', 3: 'acting'}
        by_id = {r.robot_id: r for r in robots}
        initial = {r.robot_id: r.cell for r in robots}
        kwargs = dict(tasks=tasks, task_status=status, robots_by_id=by_id, objects=objects,
                      navigation_cells=grid, grid_size=1., speed_cells_per_second=1.,
                      include_travel=True, return_details=True, return_evaluation=True)
        baseline = _remaining_dag_cost(**kwargs, positions=initial, preferred_paths={}, relocation_delay_by_robot={})
        candidates = [dict(deepcopy(candidate), candidate_id=i) for i in ('a', 'b')]
        stats = {}
        refine_release_events(candidates, robots=robots, tasks=tasks, task_status=status,
                              objects=objects, navigation_cells=grid, robot_radius=.1, grid_size=1., speed=1.,
                              participants={'R1', 'R2'}, stats=stats, search_deadline=None,
                              baseline_finishes=baseline[2], baseline_distance=baseline[1], budget_s=1.)
        self.assertEqual(stats['release_prediction']['status'], 'applied')
        for c in candidates:
            paths = dict(c['preferred_paths'])
            paths.update(c['release_prediction']['continuation_paths'])
            full = _remaining_dag_cost(**kwargs, positions=c['final_positions'] | {'R3': (8, 3)},
                                       preferred_paths=paths, relocation_delay_by_robot=c['delay_by_robot'])
            logs = []
            inc = _remaining_dag_cost(**kwargs, positions=c['final_positions'] | {'R3': (8, 3)},
                                      preferred_paths=paths, relocation_delay_by_robot=c['delay_by_robot'],
                                      baseline_evaluation=baseline[3], affected_robot_ids={'R1', 'R2'}, debug_logger=logs.append)
            self.assertEqual(full, inc)
            self.assertEqual(c['estimated_makespan'], full[0])
            self.assertEqual(inc[3]['start_times'][3], baseline[3]['start_times'][3])
            self.assertEqual(inc[2][3], baseline[2][3])
            self.assertEqual(logs[0]['affected_tasks'], 2)

    def test_uncertified_batch_restores_all_scores_atomically(self):
        snapshot = fixtures.fixture()
        candidates = [dict(self.candidate(), estimated_makespan=1.),
                      dict(self.candidate(), estimated_makespan=2.)]
        before = deepcopy(candidates)
        stats = {}
        with patch('simulations.standalone.release_events.predict_release_events',
                   side_effect=[({'R1': 1., 'R2': 2.}, {'unknown_agents': []}),
                                ({}, {'unknown_agents': ['R2']})]):
            refine_release_events(candidates, robots=snapshot['dag_robots'], tasks=snapshot['tasks'],
                task_status=snapshot['task_status'], objects=snapshot['objects'],
                navigation_cells=snapshot['navigation_cells'], robot_radius=.4, grid_size=1., speed=2.,
                participants={'R1', 'R2'}, stats=stats, search_deadline=None,
                baseline_finishes={}, baseline_distance=0.)
        self.assertEqual(candidates, before)
        self.assertEqual(stats['release_prediction']['reason'], 'uncertified_continuation')


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
