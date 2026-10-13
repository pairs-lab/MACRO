import io
import json
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulations.standalone.app import StandaloneSimulation, _compute_conflict_plan
from simulations.standalone.dag_lookahead import (
    LookaheadIncomplete, refine_candidates, replay_candidate, successor_reservation_risk,
)
from simulations.standalone.relocation_search import (
    _candidate_stable_key, _select_resolution_candidate,
)


class RefinementTests(unittest.TestCase):
    def candidates(self):
        return [dict(candidate_id=str(i), strategy="relocate", priority_robot=f"R{i}",
                     estimated_makespan=float(i), total_distance=1., total_delay=0.,
                     local_resolution_distance=float(3-i), local_resolution_delay=float(i))
                for i in (1, 2)]

    def refine(self, candidates, evaluator, stats, budget=1.):
        return refine_candidates(candidates, evaluator=evaluator, stats=stats,
            select=lambda pool:_select_resolution_candidate(pool, "dag_aware"),
            stable_key=_candidate_stable_key, budget_s=budget)

    def test_atomic_fallback_never_mixes_replayed_and_nominal_scores(self):
        candidates=self.candidates(); before=deepcopy(candidates); stats={}
        def evaluate(c, deadline):
            if c["candidate_id"] == "2":
                raise LookaheadIncomplete("recourse_limit")
            return dict(estimated_makespan=99., total_distance=20.)
        result=self.refine(candidates,evaluate,stats)
        self.assertIs(result,candidates)
        self.assertEqual(candidates,before)
        self.assertEqual(stats["dag_lookahead"]["status"],"fallback")

    def test_common_forecast_can_reverse_ranking_without_changing_geometry(self):
        candidates=self.candidates(); stats={}
        result=self.refine(candidates,lambda c,d:dict(
            estimated_makespan=10. if c["candidate_id"]=="1" else 5.,
            total_distance=2.),stats)
        self.assertEqual(_select_resolution_candidate(result,"dag_aware")["candidate_id"],"2")
        self.assertEqual(len(candidates),2)
        self.assertEqual(stats["dag_lookahead"]["status"],"applied")

    def test_zero_budget_preserves_original_ranking(self):
        candidates=self.candidates(); stats={}
        with patch("simulations.standalone.dag_lookahead.replay_candidate") as evaluate:
            self.assertIs(self.refine(candidates,evaluate,stats,budget=0.),candidates)
            evaluate.assert_not_called()

    def test_single_representative_does_not_spend_replay_latency(self):
        candidates=self.candidates()[:1];stats={}
        with patch("simulations.standalone.dag_lookahead.replay_candidate") as evaluate:
            self.assertIs(self.refine(candidates,evaluate,stats),candidates)
            evaluate.assert_not_called()
        self.assertEqual(stats["dag_lookahead"]["status"],"skipped_single_representative")

    def test_first_incomplete_batch_disables_repeated_refinement(self):
        sim=object.__new__(StandaloneSimulation)
        sim.conflict_events=[];sim.selection_policy="dag_aware"
        sim._dag_lookahead_disabled=False
        plan=dict(strategy="wait",search_stats={"dag_lookahead":{"status":"fallback"}})
        sim._record_conflict(plan,{"R1","R2"})
        self.assertTrue(sim._dag_lookahead_disabled)
