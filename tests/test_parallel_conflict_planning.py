"""Independent groups share workers, but never blindly commit stale corridors."""
import io
import unittest
import sys
from pathlib import Path
from concurrent.futures import Future
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulations.standalone.app import RobotState, StandaloneSimulation


class ParallelConflictPlanningTests(unittest.TestCase):
    def make_sim(self):
        sim = object.__new__(StandaloneSimulation)
        sim.robots = [RobotState(f"R{i + 1}", (i * 20, 0), (0, 0, 0),
                                 phase="moving", current_task={"id": i})
                      for i in range(5)]
        sim._next_planning_id = 1
        sim._planning_generation = 0
        sim._planner_executor = Mock()
        sim._planner_executor.submit.side_effect = lambda *a, **k: Future()
        sim.pending_conflict_plans = {}
        sim.active_resolutions = {}
        sim.tasks = []
        sim.task_status = {}
        sim.objects = {}
        sim.world = SimpleNamespace(navigation_cells=set(), robot_radius=.1,
                                    grid_size=1.)
        sim.steps_per_second = 8.
        sim.selection_policy = "dag_aware"
        sim.planning_wall_time_s = 0.
        return sim

    def submit(self, sim, robots):
        with redirect_stdout(io.StringIO()):
            return sim._submit_conflict_plan(robots)

    def test_disjoint_groups_submitted_before_either_finishes(self):
        sim = self.make_sim()
        self.submit(sim, sim.robots[:2])
        self.submit(sim, sim.robots[2:4])
        self.assertEqual(sim._planner_executor.submit.call_count, 2)
        self.assertTrue(all(not p["future"].done()
                            for p in sim.pending_conflict_plans.values()))
        self.assertTrue(all(r.phase == "waiting" for r in sim.robots[:4]))
        self.assertEqual(sim.robots[4].phase, "moving")

    def test_normal_external_progress_does_not_invalidate_plan(self):
        sim = self.make_sim()
        sim.robots[2].path = [(40, 0), (41, 0), (42, 0)]
        pid = self.submit(sim, sim.robots[:2])
        sim.robots[2].cell = (41, 0)
        sim.robots[2].path_index = 1
        self.assertTrue(sim._parallel_plan_is_current(
            {"preferred_paths": {"R1": [(0, 0), (1, 0)]}},
            sim.pending_conflict_plans[pid]))

    def test_other_groups_future_staging_corridor_invalidates_plan(self):
        sim = self.make_sim()
        pid = self.submit(sim, sim.robots[:2])
        sim.active_resolutions[1] = {
            "steps": [{"robot_id": "R3", "path": [(40, 0), (1, 0)]}],
            "index": 0, "preferred_paths": {"R3": [(1, 0), (1, 1)]}}
        self.assertFalse(sim._parallel_plan_is_current(
            {"preferred_paths": {"R1": [(0, 0), (1, 0), (2, 0)]}},
            sim.pending_conflict_plans[pid]))

    def test_relocation_route_also_checked(self):
        sim = self.make_sim()
        pid = self.submit(sim, sim.robots[:2])
        sim.robots[2].path = [(40, 0), (1, 0)]
        self.assertFalse(sim._parallel_plan_is_current(
            {"relocations": [{"path": [(0, 0), (1, 0), (2, 0)]}]},
            sim.pending_conflict_plans[pid]))

    def test_rejected_result_restores_group_without_recording_or_installing(self):
        sim = self.make_sim()
        pid = self.submit(sim, sim.robots[:2])
        sim.pending_conflict_plans[pid]["future"].set_result({
            "strategy": "short_replan", "preferred_paths": {
                "R1": [(0, 0), (1, 0), (2, 0)]},
            "search_stats": {"planning_wall_time_s": .25}})
        sim.robots[2].path = [(40, 0), (1, 0)]
        sim._record_conflict = Mock()
        sim._begin_reroute_plan = Mock()
        with redirect_stdout(io.StringIO()):
            sim._poll_conflict_plans()
        sim._record_conflict.assert_not_called()
        sim._begin_reroute_plan.assert_not_called()
        self.assertEqual(sim.robots[0].phase, "moving")
        self.assertEqual(sim.planning_wall_time_s, .25)
        self.assertFalse(sim.pending_conflict_plans)

    def test_simultaneously_completed_independent_groups_both_commit(self):
        sim = self.make_sim()
        first = self.submit(sim, sim.robots[:2])
        second = self.submit(sim, sim.robots[2:4])
        sim._record_conflict = Mock()
        sim._begin_reroute_plan = Mock()
        for pid, rid, start in [(first, "R1", 0), (second, "R3", 40)]:
            sim.pending_conflict_plans[pid]["future"].set_result({
                "strategy": "short_replan", "preferred_paths": {
                    rid: [(start, 0), (start + 1, 0)]}})
        with redirect_stdout(io.StringIO()):
            sim._poll_conflict_plans()
        self.assertEqual(sim._begin_reroute_plan.call_count, 2)
        self.assertEqual([call.args[1] for call in
                          sim._begin_reroute_plan.call_args_list],
                         [{"R1", "R2"}, {"R3", "R4"}])
        self.assertFalse(sim.pending_conflict_plans)

    def test_changed_participant_snapshot_is_not_committed(self):
        sim = self.make_sim()
        pid = self.submit(sim, sim.robots[:2])
        sim.pending_conflict_plans[pid]["future"].set_result({})
        sim.robots[0].cell = (1, 0)
        sim._record_conflict = Mock()
        with redirect_stdout(io.StringIO()):
            sim._poll_conflict_plans()
        sim._record_conflict.assert_not_called()
        self.assertEqual(sim.robots[0].phase, "moving")


if __name__ == "__main__":
    unittest.main()
