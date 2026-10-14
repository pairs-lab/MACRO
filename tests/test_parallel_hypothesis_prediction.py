from copy import deepcopy
from concurrent.futures import Future
import os
import unittest
from unittest.mock import patch

from core.coordination.dynamic_release import (
    refine_dynamic_release_events, close_hypothesis_pools)
from test_dynamic_release import departure_fixture


class ParallelHypothesisTests(unittest.TestCase):
    def tearDown(self):
        close_hypothesis_pools()

    def inputs(self):
        tasks, robots, common, candidate = departure_fixture()
        pool = [deepcopy(candidate), dict(deepcopy(candidate), candidate_id='b')]
        kwargs = dict(robots=robots, tasks=tasks, task_status=common['task_status'],
            objects=common['objects'], navigation_cells=common['navigation_cells'],
            robot_radius=.1, grid_size=1., speed=1., participants={'R1'},
            search_deadline=None, baseline_finishes={}, baseline_distance=0., budget_s=20.)
        return pool, kwargs

    def test_process_scores_equal_sequential_scores(self):
        pool, kwargs = self.inputs()
        sequential, parallel = deepcopy(pool), deepcopy(pool)
        seq_stats, par_stats = {}, {}
        refine_dynamic_release_events(sequential, stats=seq_stats, **kwargs)
        refine_dynamic_release_events(parallel, stats=par_stats, hypothesis_workers=2, **kwargs)
        self.assertEqual(par_stats['release_prediction']['status'], 'applied')
        self.assertEqual(par_stats['release_prediction']['evaluation_mode'], 'process_parallel')
        for seq, par in zip(sequential, parallel):
            for field in ('candidate_id', 'estimated_makespan', 'total_distance', 'delay_by_robot'):
                self.assertEqual(seq[field], par[field])
            for field in ('finish_times', 'future_delay_by_task', 'forecast_paths'):
                self.assertEqual(seq['release_prediction'][field], par['release_prediction'][field])

    def test_incomplete_batch_never_commits_partial_scores(self):
        pool, kwargs = self.inputs()
        before = deepcopy(pool)
        class IncompletePool:
            def submit(self, fn, candidate, arguments):
                future = Future()
                changed = dict(candidate, estimated_makespan=-999)
                future.set_result((changed, dict(status='fallback')))
                return future
        stats = {}
        with patch('core.coordination.dynamic_release._POOLS',
                   {(os.getpid(), 2): IncompletePool()}):
            refine_dynamic_release_events(pool, stats=stats, hypothesis_workers=2, **kwargs)
        self.assertEqual(pool, before)
        self.assertEqual(stats['release_prediction']['status'], 'fallback')


if __name__ == '__main__':
    unittest.main()
