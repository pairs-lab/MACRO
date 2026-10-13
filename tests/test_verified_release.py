import io
import sys
import time
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from simulations.standalone.app import StandaloneSimulation, _compute_conflict_plan
from simulations.standalone.verified_release import refine_verified_release
from simulations.standalone.dag_lookahead import LookaheadIncomplete, replay_candidate
from simulations.standalone.relocation_search import _candidate_stable_key
from test_runtime_policies import fixture


class VerifiedReleaseTests(unittest.TestCase):
    def test_rollout_replan_uses_dag_priority_and_nonrecursive_guarded_scoring(self):
        snapshot = fixture()
        captured = {}
        def step(sim, dt):
            self.assertEqual(sim.selection_policy, 'dag_aware')
            self.assertTrue(sim.use_dag_priority)
            self.assertTrue(sim.use_release_prediction)
            self.assertTrue(sim._executor_verification_disabled)
            sim._plan_snapshot(sim.robots)
        def replan(**kwargs):
            captured.update(kwargs)
            raise LookaheadIncomplete('stop_after_recourse_dispatch')
        with patch.object(StandaloneSimulation, 'step', step), \
             patch('simulations.standalone.app._compute_conflict_plan', side_effect=replan):
            with self.assertRaisesRegex(LookaheadIncomplete, 'stop_after_recourse_dispatch'):
                replay_candidate({'strategy': 'wait'}, robots=snapshot['dag_robots'],
                    tasks=snapshot['tasks'], task_status=snapshot['task_status'],
                    objects=snapshot['objects'], navigation_cells=snapshot['navigation_cells'],
                    robot_radius=.4, grid_size=1., speed=2., participants={'R1','R2'},
                    deadline=time.perf_counter()+5., recourse_policy='dag_aware')
        self.assertEqual(captured['selection_policy'], 'dag_aware')
        self.assertEqual(captured['release_prediction_model'], 'dynamic_events_guarded')
        self.assertTrue(captured['use_dag_priority'])
        self.assertTrue(captured['use_release_prediction'])
        self.assertFalse(captured['verify_release'])
        self.assertFalse(captured['use_dag_lookahead'])

    def test_recursive_verification_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonrecursive'):
            replay_candidate({'strategy': 'wait'}, robots=[], tasks=[], task_status={},
                objects={}, navigation_cells=frozenset(), robot_radius=.1,
                grid_size=1., speed=1., participants=set(), deadline=time.perf_counter()+5.,
                recourse_policy='dag_aware',
                recourse_release_prediction_model='dynamic_events_verified')

    def pool(self):
        return [dict(candidate_id=str(i), strategy='relocate', priority_robot='R1',
                     estimated_makespan=float(i), total_distance=1., additional_distance=0.,
                     local_resolution_distance=float(i), local_resolution_delay=float(i),
                     release_prediction={'censored': True}) for i in (1, 2, 3)]

    def refine(self, pool, stats, **extra):
        return refine_verified_release(pool, robots=[], tasks=[], task_status={}, objects={},
            navigation_cells=frozenset(), robot_radius=.1, grid_size=1., speed=1.,
            participants=set(), stats=stats, search_deadline=None,
            baseline_finishes={}, baseline_distance=0., **extra)

    def test_runner_up_shares_executor_model_and_can_reverse_ranking(self):
        pool, stats = self.pool(), {}
        geometry = [_candidate_stable_key(c) for c in pool]
        def replay(candidate, **kwargs):
            self.assertEqual(kwargs['max_repairs'], 4)
            self.assertEqual(kwargs['horizon_s'], 30.)
            self.assertEqual(kwargs['recourse_policy'], 'dag_aware')
            self.assertEqual(kwargs['recourse_release_prediction_model'], 'dynamic_events_guarded')
            return dict(estimated_makespan=10. if candidate['candidate_id']=='1' else 5.,
                        total_distance=2., finish_times={}, recourse_repairs=1)
        with patch('simulations.standalone.verified_release.refine_dynamic_release_events',
                   side_effect=lambda candidates, **kw: candidates), \
             patch('simulations.standalone.verified_release.replay_candidate', side_effect=replay) as evaluate:
            shortlist = self.refine(pool, stats)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(min(shortlist, key=lambda c:c['estimated_makespan'])['candidate_id'], '2')
        self.assertEqual(geometry, [_candidate_stable_key(c) for c in pool])
        self.assertEqual(stats['executor_verification']['status'], 'applied')
        self.assertEqual(stats['executor_verification']['recourse_policy'], 'dag_aware')
        self.assertFalse(stats['executor_verification']['nested_verification'])
        self.assertNotIn('dag_lookahead', stats)
        self.assertFalse(pool[2]['lookahead_shortlisted'])

    def test_partial_prefix_is_transactionally_discarded(self):
        pool, stats = self.pool(), {}
        before = deepcopy(pool)
        with patch('simulations.standalone.verified_release.refine_dynamic_release_events',
                   side_effect=lambda candidates, **kw: candidates), \
             patch('simulations.standalone.verified_release.replay_candidate', side_effect=[
                 dict(estimated_makespan=99., total_distance=2., finish_times={}, recourse_repairs=0),
                 LookaheadIncomplete('recourse_limit')]):
            result = self.refine(pool, stats)
        self.assertIs(result, pool)
        self.assertEqual(pool, before)
        self.assertEqual(stats['executor_verification']['status'], 'fallback')

    def test_disabled_or_unambiguous_does_not_run_executor(self):
        for enabled, ambiguous in ((False, True), (True, False)):
            pool, stats = self.pool(), {}
            for c in pool:
                c['release_prediction']['censored'] = ambiguous
            with patch('simulations.standalone.verified_release.refine_dynamic_release_events',
                       side_effect=lambda candidates, **kw: candidates), \
                 patch('simulations.standalone.verified_release.replay_candidate') as evaluate:
                self.refine(pool, stats, verification_enabled=enabled)
            evaluate.assert_not_called()

    def test_deadline_prevents_verification_and_disables_repeated_spending(self):
        pool, stats = self.pool(), {}
        with patch('simulations.standalone.verified_release.refine_dynamic_release_events',
                   side_effect=lambda candidates, **kw: candidates), \
             patch('simulations.standalone.verified_release.replay_candidate') as evaluate:
            self.refine(pool, stats, verification_budget_s=0.)
        evaluate.assert_not_called()
        sim = StandaloneSimulation.__new__(StandaloneSimulation)
        sim.selection_policy = 'dag_aware'
        sim.conflict_events = []
        sim._record_conflict(dict(strategy='wait', search_stats=stats), {'R1'})
        self.assertTrue(sim._executor_verification_disabled)
        self.assertFalse(getattr(sim, '_release_prediction_disabled', False))

    def test_other_policies_never_invoke_new_refiner(self):
        for policy in ('local_distance', 'local_delay', 'reroute_only', 'negotiation_adapted'):
            with patch('simulations.standalone.app._make_release_refiner', side_effect=AssertionError('DAG leak')):
                try:
                    _compute_conflict_plan(**fixture(), selection_policy=policy,
                                           release_prediction_model='dynamic_events_verified')
                except RuntimeError as error:
                    self.assertEqual(policy, 'reroute_only')
                    self.assertEqual(str(error), 'baseline_no_feasible_intervention')

    def test_room3_mission2_improves_without_pool_geometry_changes(self):
        captured = {}
        class Captured(Exception):
            pass
        def capture(sim, cluster):
            captured.update(cluster=deepcopy(cluster), dag_robots=deepcopy(sim.robots),
                tasks=deepcopy(sim.tasks), task_status=deepcopy(sim.task_status), objects=deepcopy(sim.objects),
                navigation_cells=sim.world.navigation_cells, robot_radius=sim.world.robot_radius,
                grid_size=sim.world.grid_size, speed_cells_per_second=sim.steps_per_second)
            raise Captured()
        with redirect_stdout(io.StringIO()):
            sim = StandaloneSimulation(ROOT/'maps/map3/scenario_2.json', async_planning=False)
            try:
                with patch.object(StandaloneSimulation, '_plan_snapshot', capture):
                    with self.assertRaises(Captured):
                        while sim.simulation_time_s < 20:
                            sim.step(min(sim.next_event_duration(), 1/60))
                before = deepcopy(captured)
                old = _compute_conflict_plan(**captured, release_prediction_model='dynamic_events_guarded')
                new = _compute_conflict_plan(**captured, release_prediction_model='dynamic_events_verified')
            finally:
                sim.close()
        self.assertEqual(captured, before)
        self.assertEqual(new['search_stats']['executor_verification']['status'], 'applied')
        self.assertEqual(old['strategy'], 'short_replan')
        self.assertEqual(new['strategy'], 'relocate')
        self.assertEqual([c['geometry_key'] for c in old['search_stats']['candidate_evaluations']],
                         [c['geometry_key'] for c in new['search_stats']['candidate_evaluations']])


if __name__ == '__main__':
    unittest.main()
