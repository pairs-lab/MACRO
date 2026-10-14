"""Native port equivalence; only tiny DAGs, never the full experiment."""
import copy
import itertools
from pathlib import Path
import random
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from core.assignment import evaluate_plan, optimize_task_assignments
from core.assignment_cpp import optimize_chain_bnb_cpp, optimize_atomic_bnb_cpp, prepare_native_problem, require_native
from tests.test_assignment_optimization import fixture, task
from tests.assignment_fixtures import EndpointOracle
from simulations.grid.app import GridDistanceOracle

try:
    import _chain_bnb_native
except ImportError:
    _chain_bnb_native = None


class NativeDependencyTests(unittest.TestCase):
    def test_missing_backend_has_actionable_error(self):
        with patch.dict(sys.modules, {"_chain_bnb_native": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install ./native/chain_bnb"):
                require_native()


@unittest.skipUnless(_chain_bnb_native, "Build optional pybind11 extension to test native search")
class NativeAssignmentTests(unittest.TestCase):
    def compare(self, tasks, speed=2., service=None, mode="chain_bnb"):
        world = fixture(tasks)
        python = optimize_task_assignments(tasks, GridDistanceOracle(world), robot_speed=speed,
                                           service_time=service, search_mode=mode)
        cpp = optimize_task_assignments(tasks, GridDistanceOracle(world), robot_speed=speed,
                                        service_time=service, search_mode=mode + "_cpp")
        self.assertEqual((cpp["makespan"], cpp["total_distance"]),
                         (python["makespan"], python["total_distance"]))
        self.assertEqual(cpp["assignment"], python["assignment"])
        self.assertEqual(cpp["robot_plans"], python["robot_plans"])
        for key in ("expanded_nodes", "bound_pruned", "distance_bound_pruned",
                    "complete_evaluated", "duplicate_complete_skipped", "handoff_pruned"):
            self.assertEqual(cpp["assignment_search"][key], python["assignment_search"][key], key)
        expected = evaluate_plan(tasks, cpp["assignment"], GridDistanceOracle(world),
                                 robot_speed=speed, service_time=service)
        self.assertEqual(expected, (cpp["makespan"], cpp["total_distance"], cpp["robot_plans"]))
        return cpp

    def test_random_dags_match_python_assignments_metrics_and_search_counts(self):
        rng = random.Random(1107)
        for _ in range(32):
            tasks = []
            for i in range(1, 7):
                t = task(i, [j for j in range(1, i) if rng.random() < .25],
                         object=rng.choice(("a", "b")))
                if rng.random() < .3:
                    t["candidate"] = [rng.choice(("R1", "R2"))]
                tasks.append(t)
            cpp = self.compare(tasks)
            atomic = self.compare(tasks, mode="task_bnb")
            self.assertEqual((cpp["makespan"], cpp["total_distance"]),
                             (atomic["makespan"], atomic["total_distance"]))
            exhaustive = optimize_task_assignments(tasks, GridDistanceOracle(fixture(tasks)),
                robot_speed=2., service_time=None, search_mode="exhaustive")
            self.assertEqual((cpp["makespan"], cpp["total_distance"]),
                             (exhaustive["makespan"], exhaustive["total_distance"]))

    def test_moving_object_ownership_and_multigoal_navigation(self):
        tasks = [task(1, action="pick"), task(2, object="b"),
                 dict(task(3, (1, 2), action="place"), target="b"), task(4, (3,))]
        cpp = self.compare(tasks)
        self.assertEqual(cpp["assignment"][1], cpp["assignment"][3])
        self.assertGreater(cpp["assignment_search"]["navigation_states"], 4)

    def test_heterogeneous_speeds_and_callable_durations(self):
        self.compare([task(1), task(2), task(3, (1, 2), object="b"), task(4, (3,))],
                     speed={"R1": 1.3, "R2": 2.7},
                     service=lambda t, r: t["duration"] * (1. if r == "R1" else 1.5))

    def test_atomic_ownership_checks_and_callable_durations_match_python(self):
        tasks = [task(1, action="pick"), task(2, action="pick", object="b"),
                 dict(task(3, (1, 2), action="place"), target="b"),
                 dict(task(4, (1, 2), action="place", object="b"), target="a")]
        result = self.compare(tasks, speed={"R1": 1.3, "R2": 2.7}, mode="task_bnb",
                              service=lambda t, r: t["duration"] * (1. if r == "R1" else 1.5))
        self.assertEqual(result["assignment"][1], result["assignment"][3])
        self.assertEqual(result["assignment"][2], result["assignment"][4])
        self.assertGreater(result["assignment_search"]["handoff_pruned"], 0)
        self.assertFalse(result["assignment_search"]["exact_prefix_bound"])

    def test_both_native_modes_prepare_identical_navigation_and_bounds(self):
        tasks = [task(1), task(2, (1,)), task(3), task(4, (2, 3), object="b")]
        data = [prepare_native_problem(tasks, GridDistanceOracle(fixture(tasks)), 2., None,
                                       search_mode=m)[0] for m in ("chain_bnb", "task_bnb")]
        for key in ("routes", "parents", "services", "speeds", "pair", "first",
                    "bound_durations", "optimistic", "seed", "candidates"):
            self.assertEqual(data[0][key], data[1][key], key)
        self.assertTrue(data[0]["use_exact_prefix"])
        self.assertFalse(data[1]["use_exact_prefix"])
        self.assertEqual(len(data[1]["groups"]), len(tasks))

    def test_atomic_fixed_chain_equalities_endpoint_navigation_and_ties(self):
        tasks = [task(1), task(2, (1,), object="b"), task(3, (2,), object="c")]
        cpp = optimize_atomic_bnb_cpp(tasks, EndpointOracle(), service_time=None)
        ref = optimize_task_assignments(tasks, EndpointOracle(), service_time=None, search_mode="task_bnb")
        self.assertEqual(cpp["assignment"], ref["assignment"])
        self.assertEqual(len(set(cpp["assignment"].values())), 1)
        self.assertEqual((cpp["makespan"], cpp["total_distance"]), (ref["makespan"], ref["total_distance"]))
        self.assertEqual(cpp["assignment_search"]["expanded_nodes"], ref["assignment_search"]["expanded_nodes"])

    def test_input_order_nonconsecutive_ids_and_determinism(self):
        tasks = [task(9, (7,), object="b"), task(7), task(22)]
        a = self.compare(tasks)
        b = self.compare(copy.deepcopy(tasks))
        self.assertEqual(a["assignment"], b["assignment"])

    def test_endpoint_state_compilation_and_candidate_ties(self):
        tasks = [task(1), task(2, object="b"), task(3, (1, 2), object="c")]
        cpp = optimize_chain_bnb_cpp(tasks, EndpointOracle(), service_time=None)
        ref = optimize_task_assignments(tasks, EndpointOracle(), service_time=None)
        self.assertEqual((cpp["makespan"], cpp["total_distance"], cpp["assignment"]),
                         (ref["makespan"], ref["total_distance"], ref["assignment"]))
        class ZeroOracle:
            def plan(self, robot, location, start_position=None):
                return 0., None
        tied = [task(17)]
        tied[0]["candidate"] = ["R2", "R1"]
        result = optimize_chain_bnb_cpp(tied, ZeroOracle(), service_time=None)
        self.assertEqual(result["assignment"], {17: "R2"})

    def test_all_frozen_routes_match_oracle_for_every_assignment(self):
        tasks = [task(1), task(2, object="b"), task(3, (1, 2))]
        oracle = GridDistanceOracle(fixture(tasks))
        data, metadata, _, robots = prepare_native_problem(tasks, oracle, 2., None)
        for owners in itertools.product(range(2), repeat=3):
            state, positions = [0, 0], [None, None]
            oracle.begin_prediction()
            for i, (t, r) in enumerate(zip(metadata["ordered_tasks"], owners)):
                distance, endpoint = oracle.plan("robot" + robots[r][1:], t["object"], positions[r])
                arc = data["routes"][r][i][state[r]]
                self.assertEqual(arc[0], distance)
                self.assertGreaterEqual(arc[1], 0)
                state[r], positions[r] = arc[1], endpoint
                oracle.finish_predicted_task(t)

    def test_preparation_pruning_preserves_all_interleaved_owner_assignments(self):
        from core.assignment_milp import compile_navigation
        tasks = [task(1, action="pick"), task(2, action="pick", object="b"),
                 dict(task(3, (1, 2), action="place"), target="b"),
                 dict(task(4, (1, 2), action="place", object="b"), target="a")]
        data, metadata, groups, robots = prepare_native_problem(
            tasks, GridDistanceOracle(fixture(tasks)), 2., None)
        _, _, full = compile_navigation(tasks, GridDistanceOracle(fixture(tasks)), 2., None)
        restricted = sum(arc[1] >= 0 for rows in data["routes"] for row in rows for arc in row)
        self.assertLess(restricted, len(full))
        for owners in itertools.product(*(g["candidates"] for g in groups)):
            assignment = {tid: r for g, r in zip(groups, owners) for tid in g["task_ids"]}
            states, available, finish = [0]*len(robots), [0.]*len(robots), {}
            distance = 0.
            for i, t in enumerate(metadata["ordered_tasks"]):
                r = robots.index(assignment[t["id"]])
                travel, endpoint = data["routes"][r][i][states[r]]
                self.assertGreaterEqual(endpoint, 0)
                finish[t["id"]] = max(available[r], max((finish[p] for p in t["node"]), default=0.)) + travel/2. + t["duration"]
                available[r], states[r] = finish[t["id"]], endpoint
                distance += travel
            expected = evaluate_plan(tasks, assignment, GridDistanceOracle(fixture(tasks)),
                                     robot_speed=2., service_time=None)
            self.assertEqual((max(finish.values()), distance), expected[:2])

    def test_unreachable_routes_and_owner_infeasibility(self):
        class BlockedOracle:
            def plan(self, *args, **kwargs):
                raise RuntimeError("unreachable")
        with self.assertRaisesRegex(RuntimeError, "No feasible"):
            optimize_chain_bnb_cpp([task(1)], BlockedOracle(), service_time=None)
        tasks = [task(1, action="pick"), task(2, object="b"),
                 dict(task(3, (1, 2), action="place"), target="b")]
        tasks[0]["candidate"], tasks[2]["candidate"] = ["R1"], ["R2"]
        with self.assertRaisesRegex(RuntimeError, "No eligible common robot"):
            optimize_chain_bnb_cpp(tasks, GridDistanceOracle(fixture(tasks)), service_time=None)

    def test_budget_includes_preparation_and_native_deadline(self):
        tasks = [task(1), task(2)]
        with self.assertRaisesRegex(TimeoutError, "assignment_time_limit"):
            optimize_chain_bnb_cpp(tasks, GridDistanceOracle(fixture(tasks)), time_limit=1e-12)
        data, _, _, _ = prepare_native_problem(tasks, GridDistanceOracle(fixture(tasks)))
        result = _chain_bnb_native.solve(data, 0.)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["found"])

    def test_invalid_native_input_is_rejected_not_indexed(self):
        tasks = [task(1)]
        data, _, _, _ = prepare_native_problem(tasks, GridDistanceOracle(fixture(tasks)))
        data["parents"][0] = [99]
        with self.assertRaisesRegex(ValueError, "Malformed"):
            _chain_bnb_native.solve(data)

    @unittest.skipUnless(hasattr(signal, "SIGALRM"), "POSIX benchmark watchdog")
    def test_native_search_honors_python_watchdog_and_preserves_counters(self):
        class ZeroOracle:
            def plan(self, *args, **kwargs):
                return 0., None
        tasks = [task(i) for i in range(1, 33)]
        data, _, _, _ = prepare_native_problem(tasks, ZeroOracle(), service_time=None)
        diagnostics = {}
        def watchdog(signum, frame):
            raise TimeoutError("test_watchdog")
        previous = signal.signal(signal.SIGALRM, watchdog)
        timer = signal.setitimer(signal.ITIMER_REAL, .01)
        try:
            with self.assertRaisesRegex(TimeoutError, "test_watchdog"):
                _chain_bnb_native.solve(data, 2., diagnostics)
            self.assertIn("expanded_nodes", diagnostics)
        finally:
            signal.setitimer(signal.ITIMER_REAL, *timer)
            signal.signal(signal.SIGALRM, previous)




if __name__ == "__main__":
    unittest.main()
