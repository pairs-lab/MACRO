"""A replaced conflict barrier is not a navigation failure."""
import threading
import unittest

from simulations.ai2thor.replanning.multi_agent_scheduler import MultiAgentReplanScheduler
from unittest.mock import Mock


class MergedBarrierTests(unittest.TestCase):
    def test_wait_is_at_current_pose_without_navigation(self):
        from simulations.ai2thor.execution import ExecutionPlanExecutor
        navigation = Mock()
        navigation.robot_locks = {}
        plan = {"robot_plans": {"R1": [1]}, "tasks": [{
            "id": 1, "action": "wait", "object": "microwave",
            "target": "", "node": [], "duration": 2.}]}
        executor = ExecutionPlanExecutor(plan, navigation, Mock())
        executor._wait_for_action_duration = Mock()
        executor._execute_task(executor.task_pool[0])
        navigation.assert_not_called()
        executor._wait_for_action_duration.assert_called_once_with(1, 2.)

    def test_contact_wait_does_not_prevent_registration_or_travel(self):
        calls = []
        path = [{"x": 1., "y": 0., "z": 0.}]
        scheduler = MultiAgentReplanScheduler(
            object(), Mock(),
            stationary_resolver=lambda *a, **kw: (calls.append(("clear", kw)) or path),
            blocker_waiter=lambda rid: calls.append(("wait", rid)))
        scheduler.register(0, 'sinkbasin')
        scheduler.register(1, 'microwave', wait_for_on_conflict=(0,))
        self.assertEqual(calls, [])
        self.assertEqual(scheduler.resolve(1, 0), path)
        self.assertEqual(calls, [("wait", 0), ("clear", {"reject_direct": True})])
        self.assertFalse(scheduler._resolutions)

    def test_priority_robot_reporting_first_does_not_enter_mutual_barrier(self):
        planner = Mock(return_value={"path": [1, 2]})
        scheduler = MultiAgentReplanScheduler(object(), planner)
        scheduler.register(0, 'sinkbasin')
        scheduler.register(1, 'microwave', wait_for_on_conflict=(0,))
        self.assertEqual(scheduler.resolve(0, 1), [1, 2])
        self.assertTrue(planner.call_args.args[-1])
        self.assertFalse(scheduler._resolutions)

    def test_waiter_follows_replacement_resolution(self):
        scheduler = MultiAgentReplanScheduler(object(), lambda *a: None)
        entered = threading.Event()
        original_wait = scheduler._condition.wait

        def wait(*args, **kwargs):
            entered.set()
            return original_wait(*args, **kwargs)

        scheduler._condition.wait = wait
        with scheduler._condition:
            old = scheduler._create_resolution_locked({0, 1}, {0, 1})
        results = []
        worker = threading.Thread(target=lambda: results.append(
            scheduler.wait_for_route(0)), daemon=True)
        worker.start()
        self.assertTrue(entered.wait(2))
        path = [{"x": 0., "y": 0., "z": 0.},
                {"x": 1., "y": 0., "z": 0.}]
        with scheduler._condition:
            scheduler._delete_resolution_locked(old)
            new = scheduler._create_resolution_locked({0, 1, 2}, {0, 1, 2})
            scheduler._install_candidate(new, {
                "priority_agent": 0, "release_moves": [],
                "priority_path": path, "current_paths": {0: path}}, {})
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [path])


if __name__ == "__main__":
    unittest.main()
