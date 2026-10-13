import io
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from unittest.mock import patch

from simulations.standalone.app import RobotState, _compute_conflict_plan, _make_release_refiner
from simulations.standalone.anticipated_delay import (
    Segment, ReservationTable, project_trajectories, estimate_future_delays,
)
from simulations.standalone.dynamic_release import refine_dynamic_release_events, stationary_service_chain
from simulations.standalone.relocation_search import _remaining_dag_cost
import test_runtime_policies as fixtures


def departure_fixture():
    tasks = [dict(id=1, robot='R1', action='toggle_on', object='a', duration=1., node=[]),
             dict(id=2, robot='R2', action='toggle_on', object='b', duration=2., node=[]),
             dict(id=3, robot='R2', action='toggle_on', object='c', duration=1., node=[2])]
    robots = [RobotState('R1', (0, 0), (0, 0, 0), goal=(1, 0),
                         path=[(0, 0), (1, 0)], phase='moving', current_task=tasks[0]),
              RobotState('R2', (1, 0), (0, 0, 0), goal=(1, 0),
                         path=[(1, 0)], phase='acting', current_task=tasks[1], action_remaining=2.)]
    grid = frozenset((x, y) for x in range(5) for y in range(2))
    objects = {name: dict(cell=cell, cells=frozenset({cell}), interaction_cells=frozenset({cell}))
               for name, cell in [('a', (1, 0)), ('b', (1, 0)), ('c', (3, 0))]}
    common = dict(tasks=tasks, task_status={1: 'running', 2: 'acting', 3: 'pending'},
                  robots_by_id={r.robot_id: r for r in robots}, objects=objects,
                  navigation_cells=grid, grid_size=1., speed_cells_per_second=1.,
                  include_travel=True, return_details=True, return_evaluation=True)
    candidate = dict(strategy='short_replan', priority_robot='R1', candidate_id='a',
                     preferred_paths={'R1': [(0, 0), (1, 0)]}, final_positions={},
                     delay_by_robot={}, rerouted_agents=['R1'], relocations=[],
                     estimated_makespan=5., total_distance=3., total_delay=0.)
    return tasks, robots, common, candidate


class DynamicReleaseTests(unittest.TestCase):
    def test_stationary_chain_counts_service_until_first_departure(self):
        tasks, robots, common, _ = departure_fixture()
        robots[1].action_remaining = .5
        tasks.insert(2, dict(id=4, robot='R2', action='toggle_off', object='b', duration=3., node=[2]))
        tasks[3]['node'] = [4]
        common['task_status'][4] = 'pending'
        evaluation = _remaining_dag_cost(**common, positions={r.robot_id:r.cell for r in robots},
            preferred_paths={}, relocation_delay_by_robot={})[3]
        self.assertEqual(stationary_service_chain(robots, tasks, evaluation)['R2'], 3.5)
        # Later service at another goal must not inflate occupancy here.
        self.assertGreater(evaluation['task_travel_times'][3], 0.)

    def test_current_model_uses_guarded_release_and_five_second_budget(self):
        with patch('simulations.standalone.dynamic_release.refine_dynamic_release_events') as refine:
            for model, guarded in [('dynamic_events_guarded', True)]:
                callback = _make_release_refiner([], [], {}, {}, frozenset(), .1, 1., 1., set(), model)
                callback([], {}, None, {}, 0.)
                self.assertEqual(refine.call_args.kwargs['budget_s'], 5. if guarded else .30)

    def test_future_goal_occupancy_delays_arrival_but_initial_body_is_not_overridden(self):
        tasks, robots, _, candidate = departure_fixture()
        robots[1].cell = (1, 1)
        evaluation = dict(start_times={1: 1., 2: 0.}, finish_times={1: 4., 2: 4.},
                          task_paths={1: ((0, 0), (1, 0), (2, 0)),
                                      2: ((1, 1), (1, 0))},
                          task_travel_times={1: 2., 2: 1.})
        delays, _ = estimate_future_delays(candidate=candidate, evaluation=evaluation,
            robots=robots, tasks=tasks, speed=1., grid_size=1., robot_radius=.1,
            horizon=5., task_limit=2, wait_before_navigation=True,
            robot_order={'R1': 0, 'R2': 1}, include_future_occupancy=True)
        self.assertGreater(delays[2], 0.)
        self.assertNotIn(1, delays)
        body = Segment('R2', 2, 0., 5., (1, 0), (1, 0), 0., anchored=True)
        route = Segment('R1', 1, 0., 2., (0, 0), (2, 0), 0.)
        table = ReservationTable([body], 1., 5., robot_order={'R1': 0, 'R2': 1})
        self.assertEqual(table.blockers(route, .2), [body])

    def test_full_equality_for_direct_wait_relocation_and_composed_staging(self):
        for strategy, steps in [('short_replan', []), ('wait', []),
                                ('relocate', [1.]), ('relocate', [1., 2.])]:
            with self.subTest(strategy=strategy, steps=steps):
                tasks, robots, common, candidate = departure_fixture()
                candidate.update(strategy=strategy, delay_by_robot={'R1': 2.},
                    relocations=[dict(delay=d, distance=d, robot_id='R1',
                                      path=[(0, 0), (0, 1)]) for d in steps],
                    moved_agents=['R1'] if steps else [])
                pool = [candidate, dict(deepcopy(candidate), candidate_id='b')]
                stats = {}
                with patch('simulations.standalone.dynamic_release.estimate_future_delays',
                           return_value=({}, {'censored': False})):
                    refine_dynamic_release_events(pool, robots=robots, tasks=tasks,
                        task_status=common['task_status'], objects=common['objects'],
                        navigation_cells=common['navigation_cells'], robot_radius=.1,
                        grid_size=1., speed=1., participants={'R1'}, stats=stats,
                        search_deadline=None, baseline_finishes={}, baseline_distance=0., budget_s=1.)
                self.assertEqual(stats['release_prediction']['status'], 'applied')
                for c in pool:
                    full = _remaining_dag_cost(**common,
                        positions={r.robot_id: r.cell for r in robots},
                        preferred_paths=c['release_prediction']['forecast_paths'],
                        relocation_delay_by_robot=c['delay_by_robot'])
                    self.assertEqual(c['estimated_makespan'], full[0])
                    self.assertEqual(c['release_prediction']['finish_times'], full[2])
                if steps:
                    self.assertEqual(candidate['delay_by_robot']['R1'], sum(steps))

    def test_staging_end_is_not_automatically_yielder_release(self):
        tasks, robots, common, candidate = departure_fixture()
        common['navigation_cells'] = frozenset([(x, 0) for x in range(5)] + [(1, 1)])
        robots[1].phase = 'moving'
        robots[1].goal = (0, 0)
        common['objects']['a'] = dict(cell=(3, 0), cells=frozenset({(3, 0)}),
                                     interaction_cells=frozenset({(3, 0)}))
        common['objects']['b'] = dict(cell=(0, 0), cells=frozenset({(0, 0)}),
                                     interaction_cells=frozenset({(0, 0)}))
        candidate.update(strategy='relocate', moved_agents=['R2'],
            delay_by_robot={'R1': 1., 'R2': 4.}, final_positions={'R2': (1, 1)},
            relocations=[dict(robot_id='R2', path=[(1, 0), (1, 1)], delay=1., distance=1.)],
            preferred_paths={'R1': [(0, 0), (1, 0), (2, 0), (3, 0)],
                             'R2': [(1, 1), (1, 0), (0, 0)]})
        pool = [candidate, dict(deepcopy(candidate), candidate_id='b')]
        stats = {}
        with patch('simulations.standalone.dynamic_release.estimate_future_delays',
                   return_value=({}, {'censored': False})):
            refine_dynamic_release_events(pool, robots=robots, tasks=tasks,
                task_status=common['task_status'], objects=common['objects'],
                navigation_cells=common['navigation_cells'], robot_radius=.4,
                grid_size=1., speed=1., participants={'R1', 'R2'}, stats=stats,
                search_deadline=None, baseline_finishes={}, baseline_distance=0., budget_s=1.)
        self.assertEqual(stats['release_prediction']['status'], 'applied')
        self.assertGreater(candidate['delay_by_robot']['R2'], 1.)
        self.assertIn('R2', candidate['release_prediction']['forecast_paths'])
        full = _remaining_dag_cost(**common,
            positions={r.robot_id: candidate['final_positions'].get(r.robot_id, r.cell) for r in robots},
            preferred_paths=candidate['release_prediction']['forecast_paths'],
            relocation_delay_by_robot=candidate['delay_by_robot'])
        self.assertEqual(candidate['estimated_makespan'], full[0])
        self.assertEqual(candidate['release_prediction']['finish_times'], full[2])

    def test_successor_departure_reduces_wait_vs_stationary_body(self):
        tasks, robots, common, candidate = departure_fixture()
        eval_ = _remaining_dag_cost(**common, positions={r.robot_id: r.cell for r in robots},
                                    preferred_paths={}, relocation_delay_by_robot={})[3]
        args = dict(candidate=candidate, evaluation=eval_, robots=robots, tasks=tasks,
                    speed=1., grid_size=1., robot_radius=.1, horizon=5., task_limit=2,
                    wait_before_navigation=True, robot_order={'R1': 0}, window_after_staging=True)
        moving, info = estimate_future_delays(**args)
        self.assertEqual(moving, {1: 2.})
        static = deepcopy(eval_)
        static['finish_times'][2] = 10.
        static['start_times'].pop(3)
        stationary, static_info = estimate_future_delays(**dict(args, evaluation=static))
        self.assertGreater(stationary[1], moving[1])
        self.assertTrue(static_info['censored'])
        self.assertFalse(info['censored'])

    def test_projection_limits_to_current_and_one_successor(self):
        tasks, robots, common, _ = departure_fixture()
        eval_ = _remaining_dag_cost(**common, positions={r.robot_id: r.cell for r in robots},
                                    preferred_paths={}, relocation_delay_by_robot={})[3]
        tasks.append(dict(id=4, robot='R2', action='toggle_on', object='c', node=[3], duration=1.))
        eval_['start_times'][4] = 5.
        eval_['finish_times'][4] = 6.
        eval_['task_paths'][4] = ((3, 0), (4, 0))
        eval_['task_travel_times'][4] = 1.
        segments = project_trajectories(evaluation=eval_, robots=robots, tasks=tasks,
            positions={}, speed=1., grid_size=1., horizon=7., task_limit=2)
        self.assertNotIn(4, {s.task for s in segments})
        self.assertIn(3, {s.task for s in segments})

    def test_injected_wait_precedes_navigation_not_extended_service(self):
        tasks, robots, common, _ = departure_fixture()
        eval_ = _remaining_dag_cost(**common, positions={r.robot_id: r.cell for r in robots},
            preferred_paths={}, relocation_delay_by_robot={}, future_delay_by_task={1: 2.})[3]
        segments = project_trajectories(evaluation=eval_, robots=robots, tasks=tasks,
            positions={}, speed=1., grid_size=1., horizon=5., task_limit=2, wait_before_navigation=True)
        movement = next(s for s in segments if s.task == 1 and s.moving)
        self.assertEqual(movement.start, 2.)
        self.assertEqual(movement.end, 3.)
        self.assertTrue(any(s.robot == 'R1' and s.first == (0, 0) and s.start == 0. and s.end == 2.
                            for s in segments if not s.moving))

    def test_refinement_matches_full_dag_and_does_not_double_charge_wait(self):
        tasks, robots, common, candidate = departure_fixture()
        baseline = _remaining_dag_cost(**common, positions={r.robot_id: r.cell for r in robots},
                                       preferred_paths={}, relocation_delay_by_robot={})
        pool = [candidate, dict(deepcopy(candidate), candidate_id='b')]
        stats = {}
        refine_dynamic_release_events(pool, robots=robots, tasks=tasks, task_status=common['task_status'],
            objects=common['objects'], navigation_cells=common['navigation_cells'], robot_radius=.1,
            grid_size=1., speed=1., participants={'R1'}, stats=stats, search_deadline=None,
            baseline_finishes=baseline[2], baseline_distance=baseline[1], budget_s=1.)
        self.assertEqual(stats['release_prediction']['status'], 'applied')
        for c in pool:
            info = c['release_prediction']
            self.assertEqual(info['future_delay_by_task'], {1: 2.})
            full = _remaining_dag_cost(**common, positions={r.robot_id: r.cell for r in robots},
                preferred_paths=c['preferred_paths'], relocation_delay_by_robot=c['delay_by_robot'],
                future_delay_by_task=info['future_delay_by_task'])
            self.assertEqual(c['estimated_makespan'], full[0])
            self.assertEqual(info['finish_times'], full[2])
            self.assertEqual(info['finish_times'][1], 4.)
            self.assertEqual(info['finish_times'][3], baseline[2][3])

    def test_timeout_rolls_back_entire_pool(self):
        tasks, robots, common, candidate = departure_fixture()
        pool = [candidate, dict(deepcopy(candidate), candidate_id='b')]
        before = deepcopy(pool)
        stats = {}
        with patch('simulations.standalone.dynamic_release.estimate_future_delays',
                   side_effect=[({}, {'censored': False}), TimeoutError()]):
            refine_dynamic_release_events(pool, robots=robots, tasks=tasks, task_status=common['task_status'],
                objects=common['objects'], navigation_cells=common['navigation_cells'], robot_radius=.1,
                grid_size=1., speed=1., participants={'R1'}, stats=stats, search_deadline=None,
                baseline_finishes={}, baseline_distance=0., budget_s=1.)
        self.assertEqual(pool, before)
        self.assertEqual(stats['release_prediction']['status'], 'fallback')

    def test_local_policies_never_construct_dynamic_refiner(self):
        with patch('simulations.standalone.app._make_release_refiner', side_effect=AssertionError('DAG leakage')):
            for policy in ('local_distance', 'local_delay', 'reroute_only'):
                for model in ('dynamic_events_guarded',):
                    with redirect_stdout(io.StringIO()):
                        try:
                            _compute_conflict_plan(**fixtures.fixture(), selection_policy=policy,
                                                   release_prediction_model=model)
                        except RuntimeError as error:
                            self.assertEqual(policy, 'reroute_only')
                            self.assertEqual(str(error), 'baseline_no_feasible_intervention')

    def test_candidate_geometry_remains_identical(self):
        pools = []
        for model in ('dynamic_events_guarded',):
            with redirect_stdout(io.StringIO()):
                plan = _compute_conflict_plan(**fixtures.fixture(), release_prediction_model=model)
            pools.append([c['geometry_key'] for c in plan['search_stats']['candidate_evaluations']])
        for pool in pools[1:]:
            self.assertEqual(pools[0], pool)


if __name__ == '__main__':
    unittest.main()
