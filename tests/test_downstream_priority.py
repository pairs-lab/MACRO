from copy import deepcopy
import unittest
from unittest.mock import patch

from tests.test_runtime_policies import fixture
from simulations.grid.app import _compute_conflict_plan
from core.coordination import relocation_search as search


class DownstreamPriorityTests(unittest.TestCase):
    def test_priority_rank_precedes_local_delay_and_never_reads_dag_scores(self):
        candidates = [dict(priority_robot='R1', strategy='relocate', local_resolution_delay=.1,
                           estimated_makespan=1., total_delay=0.),
                      dict(priority_robot='R2', strategy='relocate', local_resolution_delay=3.,
                           estimated_makespan=999., total_delay=999.)]
        self.assertIs(search._select_downstream_candidate(candidates, ['R2', 'R1']), candidates[1])
        self.assertIs(search._select_downstream_candidate(candidates, ['R1', 'R2']), candidates[0])

    def test_longer_tail_first_without_candidate_dag_scoring_or_refinement(self):
        snapshot = deepcopy(fixture())
        snapshot['tasks'] += [dict(id=3, robot='R1', action='toggle_on', object='a', node=[1], duration=2),
                              dict(id=4, robot='R2', action='toggle_on', object='b', node=[2], duration=10)]
        snapshot['task_status'].update({3: 'pending', 4: 'pending'})
        original = search._remaining_dag_cost
        with patch.object(search, '_remaining_dag_cost', wraps=original) as dag:
            plan = _compute_conflict_plan(**snapshot, selection_policy='downstream_priority',
                                          release_prediction_model='dynamic_events_guarded')
        stats = plan['search_stats']
        self.assertEqual(stats['priority_order'], ['R2', 'R1'])
        self.assertEqual(dag.call_count, 1)  # baseline durations only
        self.assertNotIn('release_prediction', stats)
        self.assertFalse(stats['dag_priority'])  # no slack ordering
        self.assertFalse(stats['downstream_priority_diagnostics']['uses_candidate_dag_scoring'])
        self.assertEqual(plan['priority_robot'], 'R2')

    def test_equal_tails_tie_by_robot_id(self):
        plan = _compute_conflict_plan(**fixture(), selection_policy='downstream_priority')
        self.assertEqual(plan['search_stats']['priority_order'], ['R1', 'R2'])

    def test_pool_geometry_matches_local_delay(self):
        snapshot = fixture()
        local = _compute_conflict_plan(**snapshot, selection_policy='local_delay')
        downstream = _compute_conflict_plan(**snapshot, selection_policy='downstream_priority')
        self.assertEqual([c['geometry_key'] for c in local['search_stats']['candidate_evaluations']],
                         [c['geometry_key'] for c in downstream['search_stats']['candidate_evaluations']])


if __name__ == '__main__':
    unittest.main()
