import unittest
import threading
import time


from simulations.ai2thor.replanning.minimum_disruption import (
    PlanningTimeoutError,
    build_priority_hypotheses,
    select_minimum_disruption_plan,
)
from simulations.ai2thor.replanning.multi_agent_scheduler import (
    MultiAgentReplanScheduler,
)
from simulations.ai2thor.replanning.spatial_conflict import (
    candidate_paths_are_safe,
    certify_concurrent_resume,
)
from simulations.ai2thor.replanning.minimum_disruption import _local_metrics
from simulations.ai2thor.replanning.dag_makespan import estimate_remaining_dag_makespan


def point(x, z=0.0):
    return {"x": float(x), "y": 0.0, "z": float(z)}


class AI2ThorRuntimeSynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.contexts = {
            0: {"object_type": "Apple", "action_duration": 1.0},
            1: {"object_type": "Tomato", "action_duration": 1.0},
        }

    @staticmethod
    def route_planner(agent, _context, overrides, avoid):
        start = dict(overrides.get(agent, point(agent)))
        if avoid and not overrides:
            return {
                "path": [start, point(agent + 1, 1), point(agent + 3)],
                "target_object": {},
            }
        return {
            "path": [start, point(agent + 3)],
            "target_object": {},
        }

    @staticmethod
    def blocker_ranker(priority, group, _path, overrides):
        return [
            agent for agent in group
            if agent != priority and agent not in overrides
        ]

    @staticmethod
    def release_planner(
        _priority, blocker, _path, overrides, _group, _context,
        blocker_context=None,
    ):
        del blocker_context
        if blocker in overrides:
            return []
        start = point(blocker)
        return [
            {
                "blocker_agent": blocker,
                "release_position": point(blocker, direction),
                "release_path": [start, point(blocker, direction)],
                "release_distance": float(direction),
                "direction_group": direction,
                "goal_directed_cost": float(direction),
            }
            for direction in (1, 2)
        ]

    @staticmethod
    def estimator(candidate, _contexts, metrics):
        return {
            **metrics,
            "estimated_makespan": (
                10.0
                + candidate["priority_agent"]
                - 2.0 * len(candidate["release_moves"])
            ),
            "total_delay": float(len(candidate["release_moves"])),
        }

    def test_shared_policies_receive_same_geometry_without_dag_leak(self):
        plans = {}
        estimator_calls = {"count": 0}

        def estimator(candidate, contexts, metrics):
            estimator_calls["count"] += 1
            return self.estimator(candidate, contexts, metrics)

        for policy in ("dag_aware", "local_distance", "local_delay"):
            estimator_calls["count"] = 0
            plan = select_minimum_disruption_plan(
                (0, 1),
                self.contexts,
                self.route_planner,
                self.blocker_ranker,
                self.release_planner,
                remaining_estimator=estimator,
                candidate_validator=lambda _candidate: True,
                selection_policy=policy,
            )
            plans[policy] = plan
            if policy == "dag_aware":
                self.assertGreater(estimator_calls["count"], 0)
            else:
                self.assertEqual(estimator_calls["count"], 0)

        def geometry(plan):
            return {
                (
                    row["strategy"],
                    row["priority_agent"],
                    tuple(row["moved_agents"]),
                    round(row["local_resolution_distance"], 6),
                    round(row["local_resolution_delay"], 6),
                )
                for row in plan["search_stats"]["candidate_evaluations"]
            }

        self.assertEqual(geometry(plans["dag_aware"]), geometry(plans["local_distance"]))
        self.assertEqual(geometry(plans["dag_aware"]), geometry(plans["local_delay"]))
        self.assertFalse(plans["dag_aware"]["search_stats"]["dag_bound_pruning"])

    def test_reroute_only_does_not_generate_relocations(self):
        release_calls = {"count": 0}

        def release(*args, **kwargs):
            release_calls["count"] += 1
            return self.release_planner(*args, **kwargs)

        plan = select_minimum_disruption_plan(
            (0, 1),
            self.contexts,
            self.route_planner,
            self.blocker_ranker,
            release,
            selection_policy="reroute_only",
        )
        self.assertEqual(plan["strategy"], "short_replan")
        self.assertEqual(plan["release_moves"], [])
        self.assertEqual(release_calls["count"], 0)

    def test_conflict_snapshot_reuses_identical_astar_queries(self):
        route_calls = {"count": 0}

        def counted_route(*args, **kwargs):
            route_calls["count"] += 1
            return self.route_planner(*args, **kwargs)

        plan = select_minimum_disruption_plan(
            (0, 1),
            self.contexts,
            counted_route,
            self.blocker_ranker,
            self.release_planner,
            candidate_validator=lambda _candidate: True,
            selection_policy="local_distance",
        )
        stats = plan["search_stats"]
        self.assertGreater(stats["route_cache_hits"], 0)
        self.assertEqual(stats["route_cache_misses"], route_calls["count"])


    def test_joint_staging_can_move_priority_before_blocker(self):
        def no_direct_route(agent, context, overrides, avoid):
            if avoid and not overrides:
                raise RuntimeError("blocked")
            return self.route_planner(agent, context, overrides, avoid)

        setup = {
            "blocker_agent": 0,
            "release_position": point(-1),
            "release_path": [point(0), point(-1)],
            "release_distance": 1.0,
            "goal_directed_cost": 1.0,
        }
        parking = {
            "blocker_agent": 1,
            "release_position": point(1, 1),
            "release_path": [point(1), point(1, 1)],
            "release_distance": 1.0,
            "goal_directed_cost": 1.0,
        }

        candidates = build_priority_hypotheses(
            0,
            (0, 1),
            self.contexts,
            no_direct_route,
            self.blocker_ranker,
            lambda *args, **kwargs: None,
            candidate_validator=lambda _candidate: True,
            joint_staging_planner=lambda *_args: [
                ([setup, parking], {0: point(-1), 1: point(1, 1)})
            ],
        )
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["joint_staging"])
        self.assertEqual(
            [move["blocker_agent"] for move in candidates[0]["release_moves"]],
            [0, 1],
        )

    def test_scheduler_replans_displaced_robot_from_observed_pose(self):
        route_calls = []

        def route_planner(agent, context, overrides, avoid):
            route_calls.append((agent, context["object_type"], overrides, avoid))
            return {"path": [point(5), point(6)], "target_object": {}}

        scheduler = MultiAgentReplanScheduler(object(), route_planner)
        with scheduler._condition:
            resolution_id = scheduler._create_resolution_locked({0, 1}, {0, 1})
        candidate = {
            "priority_agent": 0,
            "release_moves": [{
                "blocker_agent": 1,
                "release_path": [point(1), point(1, 1)],
            }],
            "priority_path": [point(0), point(2)],
            "current_paths": {
                0: [point(0), point(2)],
                1: [point(1, 1), point(4)],
            },
        }
        scheduler._install_candidate(resolution_id, candidate, self.contexts)

        self.assertEqual(scheduler.wait_for_route(1), candidate["release_moves"][0]["release_path"])
        resumed = []
        waiter = threading.Thread(
            target=lambda: resumed.append(
                scheduler.wait_for_route(1, route_completed=True)
            )
        )
        waiter.start()
        self.assertEqual(scheduler.wait_for_route(0), candidate["priority_path"])
        self.assertIsNone(scheduler.wait_for_route(0, route_completed=True))
        waiter.join(timeout=2.0)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(resumed, [[point(5), point(6)]])
        self.assertEqual(route_calls, [(1, "Tomato", {}, True)])

    def test_scheduler_propagates_planning_timeout_to_navigation_worker(self):
        scheduler = MultiAgentReplanScheduler(object(), self.route_planner)
        with scheduler._condition:
            resolution_id = scheduler._create_resolution_locked({0, 1}, {0, 1})
        scheduler._fail_resolution(resolution_id, "planning_timeout")

        with self.assertRaisesRegex(RuntimeError, "planning_timeout"):
            scheduler.wait_for_route(0)

    def test_certified_resume_releases_both_robots_before_priority_goal(self):
        candidate = {
            'priority_agent': 0,
            'release_moves': [{'blocker_agent': 1,
                               'release_path': [point(1), point(1, 1)],
                               'release_position': point(1, 1),
                               'release_distance': 1.}],
            'priority_path': [point(0), point(1), point(2), point(3)],
            'current_paths': {0: [point(0), point(1), point(2), point(3)],
                              1: [point(1, 1), point(1), point(0)]},
            'current_path_lengths': {0: 3., 1: 2.},
        }
        certify_concurrent_resume(candidate, lambda a, b: .3)
        self.assertTrue(candidate_paths_are_safe(candidate, lambda a, b: .3))
        prefix = candidate['priority_release_prefix']
        self.assertLess(len(prefix), len(candidate['priority_path']))
        metrics = _local_metrics(candidate, self.contexts, 1.)
        self.assertEqual(metrics['delay_by_agent'][1], 1. + len(prefix) - 1)
        tasks = [{'id': 1, 'robot': 'R1'}, {'id': 2, 'robot': 'R2'}]
        state = {'task_status': {1: 'running', 2: 'running'},
                 'robot_plans': {'R1': [1], 'R2': [2]},
                 'agent_to_robot': {0: 'R1', 1: 'R2'}}
        baseline = estimate_remaining_dag_makespan(tasks, {1: set(), 2: set()}, state,
                                                   {}, robot_speed=1.)
        full = estimate_remaining_dag_makespan(tasks, {1: set(), 2: set()}, state,
                                               candidate, contexts=self.contexts, robot_speed=1.)
        incremental = estimate_remaining_dag_makespan(
            tasks, {1: set(), 2: set()}, state, candidate,
            contexts=self.contexts, robot_speed=1., baseline=baseline)
        self.assertEqual(full.task_start_times[2], metrics['delay_by_agent'][1])
        self.assertEqual(full.task_finish_times[1], 5.)
        self.assertEqual(full.makespan, incremental.makespan)
        self.assertEqual(full.task_finish_times, incremental.task_finish_times)

        def no_detour(*args):
            self.fail('Certified routes must not be replaced with detours')
        scheduler = MultiAgentReplanScheduler(object(), no_detour)
        with scheduler._condition:
            resolution_id = scheduler._create_resolution_locked({0, 1}, {0, 1})
        scheduler._install_candidate(resolution_id, candidate, self.contexts)
        self.assertEqual(scheduler.wait_for_route(1), candidate['release_moves'][0]['release_path'])
        resumed = []
        waiter = threading.Thread(target=lambda: resumed.append(
            scheduler.wait_for_route(1, route_completed=True)), daemon=True)
        waiter.start()
        self.assertEqual(scheduler.wait_for_route(0), prefix)
        self.assertEqual(scheduler.wait_for_route(0, route_completed=True),
                         candidate['concurrent_resume_paths'][0])
        waiter.join(timeout=2.)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(resumed, [candidate['concurrent_resume_paths'][1]])

    def test_unsafe_concurrent_resume_does_not_receive_certificate(self):
        candidate = {
            'priority_agent': 0,
            'release_moves': [{'blocker_agent': 1, 'release_path': [point(1), point(1, 1)],
                               'release_position': point(1, 1)}],
            'priority_path': [point(0), point(1), point(2)],
            'current_paths': {0: [point(0), point(1), point(2)],
                              1: [point(1, 1), point(2)]},
        }
        certify_concurrent_resume(candidate, lambda a, b: .3)
        self.assertNotIn('concurrent_resume_paths', candidate)

    def test_executor_workers_navigate_independent_tasks_concurrently(self):
        from types import SimpleNamespace
        from simulations.ai2thor.execution import ExecutionPlanExecutor
        gate = threading.Barrier(2, timeout=2.)
        calls = []
        def navigate(robot, *args, **kwargs):
            self.assertIn(('stow', robot), calls)
            calls.append(('navigate', robot))
            # Both workers must reach navigation before either can finish it.
            gate.wait()
        actions = SimpleNamespace(
            stow_for_navigation=lambda robot: calls.append(('stow', robot)),
            pick=lambda robot, obj, **kwargs: calls.append(('pick', robot)))
        plan = {'robot_plans': {'R1': [1], 'R2': [2]}, 'tasks': [
            {'id': 1, 'action': 'pick', 'object': 'bread', 'node': [], 'duration': .001},
            {'id': 2, 'action': 'pick', 'object': 'tomato', 'node': [], 'duration': .001}]}
        executor = ExecutionPlanExecutor(plan, navigate, actions)
        executor.run()
        self.assertTrue(executor._all_tasks_finished())
        self.assertEqual(sum(kind == 'navigate' for kind, robot in calls), 2)

    def test_stationary_retry_rejects_unity_failed_direct_detour(self):
        calls = []

        def stationary_resolver(
            requesting_id,
            blocking_id,
            context,
            reject_direct=False,
        ):
            calls.append(
                (requesting_id, blocking_id, context, reject_direct)
            )
            return [point(0), point(0, 1)]

        scheduler = MultiAgentReplanScheduler(
            object(),
            self.route_planner,
            stationary_resolver=stationary_resolver,
        )
        scheduler.register(0, "Bowl")
        scheduler.register(1, "Fridge")
        scheduler.navigation_arrived(1)

        route = scheduler.resolve(0, 1, direct_path_failed=True)

        self.assertEqual(route, [point(0), point(0, 1)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], (0, 1))
        self.assertTrue(calls[0][3])

    def test_planning_timeout_is_shared_by_all_priority_roots(self):
        def slow_route(*args, **kwargs):
            time.sleep(0.01)
            return self.route_planner(*args, **kwargs)

        with self.assertRaises(PlanningTimeoutError):
            select_minimum_disruption_plan(
                (0, 1),
                self.contexts,
                slow_route,
                self.blocker_ranker,
                self.release_planner,
                selection_policy="reroute_only",
                planning_timeout_s=0.005,
            )


if __name__ == "__main__":
    unittest.main()
