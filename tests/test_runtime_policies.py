import io
import math
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from simulations.grid.app import RobotState, GridSimulation, _compute_conflict_plan, _local_resolution_time
from core.coordination import relocation_search as search
from core.dag_priority import order_dag_priorities


def fixture():
    tasks = [dict(id=1, robot="R1", action="toggle_on", object="a", node=[], duration=1),
             dict(id=2, robot="R2", action="toggle_on", object="b", node=[], duration=3)]
    robots = [RobotState("R1", (2, 3), (0, 0, 0), goal=(6, 3),
                         path=[(x, 3) for x in range(2, 7)], phase="moving", current_task=tasks[0]),
              RobotState("R2", (3, 3), (0, 0, 0), goal=(3, 6),
                         path=[(3, y) for y in range(3, 7)], phase="moving", current_task=tasks[1])]
    return dict(cluster=robots, dag_robots=robots, tasks=tasks,
        task_status={1: "running", 2: "running"},
        objects={"a": {"cell": (6, 3), "cells": frozenset({(6, 3)})},
                 "b": {"cell": (3, 6), "cells": frozenset({(3, 6)})}},
        navigation_cells=frozenset((x, y) for x in range(9) for y in range(9)),
        robot_radius=0.4, grid_size=1.0, speed_cells_per_second=2.0)


class PoliciesTest(unittest.TestCase):
    def test_joint_staging_can_release_idle_blocker_from_cul_de_sac(self):
        task = dict(
            id=1, robot="R1", action="toggle_on", object="goal",
            node=[], duration=1,
        )
        navigation = frozenset(
            [(x, 0) for x in range(7)] + [(3, 1), (3, 2)]
        )
        pools = []
        for policy in ("dag_aware", "local_distance", "local_delay"):
            active = RobotState(
                "R1", (3, 0), (0, 0, 0), goal=(6, 0),
                path=[(3, 0), (4, 0), (5, 0), (6, 0)],
                phase="moving", current_task=task,
            )
            idle = RobotState(
                "R2", (6, 0), (0, 0, 0), goal=None,
                path=[(6, 0)], phase="idle", current_task=None,
            )
            plan = _compute_conflict_plan(
                cluster=[active, idle], dag_robots=[active, idle],
                tasks=[task], task_status={1: "running"},
                objects={
                    "goal": {
                        "cell": (6, 0), "cells": frozenset({(6, 0)}),
                    }
                },
                navigation_cells=navigation, robot_radius=0.4,
                grid_size=1.0, speed_cells_per_second=2.0,
                selection_policy=policy,
            )
            candidates = plan["search_stats"]["candidate_evaluations"]
            self.assertTrue(candidates)
            self.assertTrue(any(
                candidate["strategy"] == "relocate"
                and candidate["moved_robot"] == ["R1", "R2"]
                for candidate in candidates
            ))
            pools.append([
                candidate["geometry_key"] for candidate in candidates
            ])
        self.assertEqual(pools[0], pools[1])
        self.assertEqual(pools[1], pools[2])

    def test_completed_yield_reservation_has_no_live_parking_step(self):
        sim = GridSimulation.__new__(GridSimulation)
        sim.active_resolutions = {
            4: {
                "kind": "yield_wait",
                "steps": [{"robot_id": "R1"}],
                "index": 1,
            },
            5: {
                "kind": "relocation",
                "steps": [{"robot_id": "R2"}],
                "index": 0,
            },
        }
        self.assertEqual(sim._parking_resolution_ids("R1"), [])
        self.assertEqual(sim._parking_resolution_ids("R2"), [5])

    def test_cancelled_idle_parking_arrival_recovers_without_crashing(self):
        sim = GridSimulation.__new__(GridSimulation)
        robot = RobotState(
            "R1",
            (3, 4),
            (0, 0, 0),
            goal=(3, 4),
            path=[(2, 4), (3, 4)],
            phase="parking",
            current_task=None,
        )
        robot.path_index = 1
        robot.waiting_for = "R2"
        robot.clearing_for = ("R2",)
        robot.clearing_started = True
        with redirect_stdout(io.StringIO()):
            sim._recover_orphaned_parking(robot)
        self.assertEqual(robot.phase, "idle")
        self.assertEqual(robot.path, [robot.cell])
        self.assertIsNone(robot.goal)
        self.assertIsNone(robot.waiting_for)
        self.assertEqual(robot.clearing_for, ())

    def test_new_contact_reoptimizes_completed_yield_reservation(self):
        """A parked yielder must not suppress a later horizon forever."""
        data = fixture()
        sim = GridSimulation.__new__(GridSimulation)
        sim.robots = data["dag_robots"]
        sim.pending_conflict_plans = {}
        sim.active_resolutions = {
            7: {
                "kind": "yield_wait",
                "participants": {"R1", "R2"},
                "priority_robot": "R2",
            }
        }
        sim.selection_policy = "dag_aware"
        sim.async_conflict_planning = False
        cancelled = []
        replanned = []

        def cancel(resolution_ids):
            cancelled.extend(resolution_ids)
            sim.active_resolutions.clear()
            return {"R1", "R2"}

        plan = {"strategy": "relocate"}
        sim._cancel_resolutions = cancel
        sim._plan_snapshot = lambda cluster: plan
        sim._record_conflict = lambda selected, participants: None
        sim._begin_resolution_plan = (
            lambda selected, participants: replanned.append(set(participants))
        )

        self.assertTrue(sim._resolve_contact_clusters([sim.robots]))
        self.assertEqual(cancelled, [7])
        self.assertEqual(replanned, [{"R1", "R2"}])

    def test_receding_horizon_release_completes_shared_corridor_scene(self):
        source = ROOT / "maps/map2/scenario_4.json"
        if not source.exists():
            self.skipTest("Optional shared-corridor regression scene absent")
        with redirect_stdout(io.StringIO()):
            sim = GridSimulation(
                source,
                speed=8,
                selection_policy="dag_aware",
                async_planning=False,
            )
            elapsed = 0.0
            try:
                while (
                    not all(
                        state == "completed"
                        for state in sim.task_status.values()
                    )
                    and elapsed < 60.0
                ):
                    dt = sim.next_event_duration()
                    sim.step(dt)
                    elapsed += dt
            finally:
                sim.close()
        self.assertTrue(
            all(state == "completed" for state in sim.task_status.values())
        )
        self.assertLess(elapsed, 60.0)
        self.assertLess(len(sim.conflict_events), 20)

    def test_service_pool_is_shared_without_downstream_lookups(self):
        pools = []
        for policy in ("dag_aware", "local_distance", "local_delay"):
            data = fixture()
            sim = GridSimulation.__new__(GridSimulation)
            sim.selection_policy = policy
            sim.robots = data["dag_robots"]
            sim.robots[1].phase = "acting"
            sim.robots[1].action_remaining = 3
            sim.robots[1].goal = sim.robots[1].cell
            sim.robots[1].path = [sim.robots[1].cell]
            sim.world = SimpleNamespace(navigation_cells=data["navigation_cells"], robot_radius=0.4, grid_size=1)
            sim.steps_per_second = 2
            sim.objects = data["objects"]
            sim.tasks = data["tasks"]
            sim.task_status = data["task_status"]
            sim.conflict_events = []
            sim.active_resolutions = {}
            sim._begin_resolution_plan = lambda *args: None
            with redirect_stdout(io.StringIO()), patch.object(sim, "_service_hold_estimate", side_effect=AssertionError("future task leak")):
                if policy == "dag_aware":
                    sim._choose_service_wait_or_reroute(sim.robots)
                else:
                    with patch.object(search, "_remaining_dag_cost", side_effect=AssertionError("DAG leak")):
                        sim._choose_service_wait_or_reroute(sim.robots)
            pools.append([(c["geometry_key"], c["local_resolution_distance"], c["local_resolution_delay"])
                          for c in sim.conflict_events[0]["search_stats"]["candidate_evaluations"]])
        self.assertEqual(pools[0], pools[1])
        self.assertEqual(pools[1], pools[2])

    def test_unknown_future_departure_is_not_a_finite_local_wait(self):
        data = fixture()
        data["dag_robots"][1].phase = "acting"
        data["dag_robots"][1].path = [(3, 3)]
        data["dag_robots"][1].goal = (3, 3)
        data["dag_robots"][1].action_remaining = 1
        result = _local_resolution_time(dict(strategy="wait"), data["dag_robots"],
            {"R1", "R2"}, data["navigation_cells"], 0.6, 1.0, 2.0)
        self.assertEqual(result, float("inf"))

    def test_shared_geometry_and_no_local_dag_evaluation(self):
        snapshots = []
        for policy in ("dag_aware", "local_distance", "local_delay"):
            data = fixture()
            if policy == "dag_aware":
                plan = _compute_conflict_plan(**data, selection_policy=policy)
            else:
                with patch.object(search, "_remaining_dag_cost", side_effect=AssertionError("DAG leak")), \
                     patch.object(search, "_order_standalone_priorities", side_effect=AssertionError("priority leak")):
                    plan = _compute_conflict_plan(**data, selection_policy=policy)
            candidates = plan["search_stats"]["candidate_evaluations"]
            self.assertTrue(candidates)
            self.assertEqual(sum(c["selected"] for c in candidates), 1)
            selected = next(c for c in candidates if c["selected"])
            if policy == "dag_aware":
                key = lambda c: (c["estimated_makespan"], c["additional_distance"], c["total_delay"])
            else:
                field = "local_resolution_distance" if policy == "local_distance" else "local_resolution_delay"
                key = lambda c: c[field]
            self.assertEqual(key(selected), min(map(key, candidates)))
            snapshots.append([(c["geometry_key"], c["local_resolution_distance"],
                               c["local_resolution_delay"]) for c in candidates])
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[1], snapshots[2])

    def test_local_selection_ignores_downstream_cost_even_on_ties(self):
        candidates = [dict(strategy="short_replan", priority_robot=r,
            local_resolution_distance=1, local_resolution_delay=2,
            estimated_makespan=cost, total_delay=cost, additional_distance=cost)
            for r, cost in (("R1", 100), ("R2", 0))]
        for policy in ("local_distance", "local_delay"):
            first = search._select_resolution_candidate(candidates, policy)["priority_robot"]
            candidates.reverse()
            candidates[0]["estimated_makespan"] = -1000
            self.assertEqual(first, search._select_resolution_candidate(candidates, policy)["priority_robot"])

    def test_reroute_only_never_uses_relocation_or_dag(self):
        with patch.object(search, "_remaining_dag_cost", side_effect=AssertionError("DAG leak")), \
             patch("core.coordination.coordinator.search_relocation_plan", side_effect=AssertionError("relocation leak")):
            plan = _compute_conflict_plan(**fixture(), selection_policy="reroute_only")
        self.assertEqual(plan["strategy"], "short_replan")
        self.assertEqual(plan["relocations"], [])
        self.assertEqual(plan["total_distance"], min(c["total_distance"] for c in plan["search_stats"]["candidate_evaluations"]))

    def test_reroute_execution_does_not_hold_other_robots(self):
        data = fixture()
        plan = _compute_conflict_plan(**data, selection_policy="reroute_only")
        sim = GridSimulation.__new__(GridSimulation)
        sim.selection_policy = "reroute_only"
        sim.robots = data["dag_robots"]
        sim.pending_conflict_plans = {}
        sim.active_resolutions = {}
        sim.conflict_events = []
        sim._plan_snapshot = lambda cluster: plan
        self.assertTrue(sim._resolve_contact_clusters([sim.robots]))
        self.assertEqual(sim.active_resolutions, {})
        self.assertTrue(all(r.phase == "moving" for r in sim.robots))

    def test_reroute_only_explicit_failure(self):
        with patch("core.coordination.coordinator._astar_around_robot_bodies", return_value=None), \
             patch("core.coordination.coordinator.search_relocation_plan", side_effect=AssertionError("fallback")):
            with self.assertRaisesRegex(RuntimeError, "baseline_no_feasible_intervention"):
                _compute_conflict_plan(**fixture(), selection_policy="reroute_only")

    def test_local_delay_includes_wait_not_only_extra_distance(self):
        data = fixture()
        candidate = search.evaluate_reroute_plan(worker_id="R1", route=data["cluster"][0].path,
            robots=data["dag_robots"], tasks=data["tasks"], task_status=data["task_status"],
            objects=data["objects"], navigation_cells=data["navigation_cells"],
            grid_size=1, speed_cells_per_second=2, held_robot_ids=["R2"],
            robot_radius=0.4, evaluate_dag=False)
        self.assertEqual(candidate["local_resolution_distance"], 0)
        self.assertGreater(candidate["local_resolution_delay"], 0)


class DagPriorityOrderingTest(unittest.TestCase):
    def test_priority_key_uses_slack_downstream_then_robot_id(self):
        self.assertEqual(
            order_dag_priorities(
                ("R2", "R1"),
                {"R1": 1, "R2": 3},
                {1: [2], 2: [], 3: []},
                {1: 1, 2: 2, 3: 1},
                nominal_finish_times={1: 5, 3: 4},
                nominal_makespan=10,
            ),
            ("R1", "R2"),
        )
        self.assertEqual(
            order_dag_priorities(
                ("R2", "R1"),
                {"R1": 1, "R2": 3},
                {1: [2], 2: [], 3: [4], 4: []},
                {1: 1, 2: 4, 3: 1, 4: 2},
                nominal_finish_times={1: 2, 3: 4},
                nominal_makespan=10,
            ),
            ("R1", "R2"),
        )
        self.assertEqual(
            order_dag_priorities(
                ("R2", "R1"),
                {"R1": 1, "R2": 2},
                {1: [], 2: []},
                {1: 1, 2: 1},
                nominal_finish_times={1: 4, 2: 4},
                nominal_makespan=10,
            ),
            ("R1", "R2"),
        )

    def test_priority_debug_logging_is_opt_in(self):
        events = []
        order_dag_priorities(
            ("R2", "R1"),
            {"R1": 1, "R2": 2},
            {1: [], 2: []},
            {1: 1, 2: 1},
            nominal_finish_times={1: 2, 2: 2},
            nominal_makespan=3,
            debug_logger=events.append,
        )
        self.assertEqual(events[0]["priority_order"], ["R1", "R2"])
        self.assertEqual(
            set(events[0]["robots"][0]),
            {"robot_id", "slack", "downstream_duration"},
        )


class IncrementalDagEvaluationTest(unittest.TestCase):
    @staticmethod
    def _fixture():
        tasks = [
            dict(id=1, robot="R1", action="toggle_on", object="a", node=[], duration=1),
            dict(id=2, robot="R1", action="toggle_on", object="b", node=[1], duration=1),
            dict(id=3, robot="R2", action="toggle_on", object="c", node=[], duration=2),
            dict(id=4, robot="R2", action="toggle_on", object="d", node=[3], duration=1),
            dict(id=5, robot="R3", action="toggle_on", object="e", node=[], duration=3),
            dict(id=6, robot="R3", action="toggle_on", object="f", node=[5], duration=1),
        ]
        starts = {"R1": (0, 0), "R2": (0, 2), "R3": (0, 4)}
        goals = {"R1": (2, 0), "R2": (2, 2), "R3": (2, 4)}
        current = {"R1": tasks[0], "R2": tasks[2], "R3": tasks[4]}
        robots = {
            robot_id: RobotState(
                robot_id,
                starts[robot_id],
                (0, 0, 0),
                goal=goals[robot_id],
                path=[starts[robot_id], (1, starts[robot_id][1]), goals[robot_id]],
                phase="moving",
                current_task=current[robot_id],
            )
            for robot_id in starts
        }
        object_cells = {
            "a": (2, 0), "b": (4, 0),
            "c": (2, 2), "d": (4, 2),
            "e": (2, 4), "f": (4, 4),
        }
        objects = {
            name: {"cell": cell, "cells": frozenset({cell})}
            for name, cell in object_cells.items()
        }
        common = dict(
            tasks=tasks,
            task_status={task["id"]: "running" if task["id"] in {1, 3, 5} else "pending"
                         for task in tasks},
            robots_by_id=robots,
            objects=objects,
            navigation_cells=frozenset((x, y) for x in range(6) for y in range(6)),
            grid_size=1.0,
            speed_cells_per_second=2.0,
            include_travel=True,
        )
        positions = dict(starts)
        return common, positions

    def test_incremental_matches_full_for_all_intervention_forms(self):
        common, positions = self._fixture()
        baseline = search._remaining_dag_cost(
            **common,
            positions=positions,
            preferred_paths={},
            relocation_delay_by_robot={},
            return_details=True,
            return_execution_durations=True,
            return_evaluation=True,
        )
        baseline_evaluation = baseline[4]
        variants = {
            "reroute": (
                positions,
                {"R1": [(0, 0), (0, 1), (1, 1), (2, 1)]},
                {},
                {"R1"},
            ),
            "yield": (
                positions,
                {},
                {"R2": 0.75},
                {"R2"},
            ),
            "relocation": (
                {**positions, "R1": (0, 1)},
                {"R1": [(0, 1), (1, 1), (2, 1)]},
                {"R1": 0.5},
                {"R1"},
            ),
            "composed": (
                {**positions, "R1": (0, 1), "R2": (0, 3)},
                {
                    "R1": [(0, 1), (1, 1), (2, 1)],
                    "R2": [(0, 3), (1, 3), (2, 3)],
                },
                {"R1": 0.5, "R2": 1.0},
                {"R1", "R2"},
            ),
        }
        for name, (candidate_positions, paths, delays, affected_robots) in variants.items():
            with self.subTest(name=name):
                full = search._remaining_dag_cost(
                    **common,
                    positions=candidate_positions,
                    preferred_paths=paths,
                    relocation_delay_by_robot=delays,
                    return_details=True,
                    return_execution_durations=True,
                    return_evaluation=True,
                )
                logs = []
                with patch.object(
                    search, "preferred_path_to_object",
                    wraps=search.preferred_path_to_object,
                ) as route_planner:
                    incremental = search._remaining_dag_cost(
                        **common, positions=candidate_positions,
                        preferred_paths=paths, relocation_delay_by_robot=delays,
                        return_details=True, return_execution_durations=True,
                        return_evaluation=True,
                        baseline_evaluation=baseline_evaluation,
                        affected_robot_ids=affected_robots,
                        debug_logger=logs.append,
                    )
                self.assertEqual(route_planner.call_count, len(affected_robots))
                self.assertEqual(incremental[:3], full[:3])
                self.assertEqual(logs[0]["directly_changed_tasks"], len(affected_robots))
                self.assertEqual(
                    logs[0]["affected_tasks"],
                    2 * len(affected_robots),
                )
                unaffected = {5, 6}
                if "R3" not in affected_robots:
                    for task_id in unaffected:
                        self.assertEqual(
                            incremental[4]["start_times"][task_id],
                            baseline_evaluation["start_times"][task_id],
                        )
                        self.assertEqual(
                            incremental[4]["finish_times"][task_id],
                            baseline_evaluation["finish_times"][task_id],
                        )
                for robot_id in affected_robots:
                    first = {"R1": 1, "R2": 3, "R3": 5}[robot_id]
                    descendant = first + 1
                    self.assertIn(first, incremental[4]["execution_durations"])
                    self.assertIn(descendant, incremental[4]["execution_durations"])

    def test_ablation_switches_priority_without_disabling_dag_scoring(self):
        dag_plan = _compute_conflict_plan(
            **fixture(), selection_policy="dag_aware"
        )
        without_priority = _compute_conflict_plan(
            **fixture(), selection_policy="dag_aware", use_dag_priority=False
        )
        self.assertEqual(dag_plan["search_stats"]["priority_order"], ["R2", "R1"])
        self.assertEqual(
            without_priority["search_stats"]["priority_order"], ["R1", "R2"]
        )
        self.assertTrue(
            all(
                math.isfinite(candidate["estimated_makespan"])
                for candidate in without_priority["search_stats"][
                    "candidate_evaluations"
                ]
            )
        )

    def test_full_reevaluation_matches_incremental_candidates(self):
        incremental = _compute_conflict_plan(
            **fixture(), selection_policy="dag_aware"
        )
        complete = _compute_conflict_plan(
            **fixture(),
            selection_policy="dag_aware",
            incremental_dag_evaluation=False,
        )
        first = incremental["search_stats"]["candidate_evaluations"]
        second = complete["search_stats"]["candidate_evaluations"]
        self.assertEqual(
            [(candidate["geometry_key"], candidate["selected"]) for candidate in first],
            [(candidate["geometry_key"], candidate["selected"]) for candidate in second],
        )
        for left, right in zip(first, second):
            self.assertAlmostEqual(
                left["estimated_makespan"], right["estimated_makespan"], places=12
            )




if __name__ == "__main__":
    unittest.main()
