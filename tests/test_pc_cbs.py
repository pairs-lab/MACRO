"""Focused checks for the adapted PC-CBS, not full benchmark execution."""
from dataclasses import replace
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core.pc_cbs import Problem, Task, Plan, solve, validate, separation


class PCCBSTests(unittest.TestCase):
    def single(self):
        return Problem(frozenset((x, 0) for x in range(3)), {"R1": (0, 0)},
                       (Task(1, "R1", frozenset({(2, 0)}), 2),), 1., 12)

    def test_navigation_then_nonpreemptive_service(self):
        problem = self.single()
        result = solve(problem, 2)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["makespan_ticks"], 4)
        self.assertEqual(result["plans"]["R1"].services[1], 2)
        self.assertTrue(validate(problem, result["plans"]))
        self.assertFalse(result["optimality_claim"])

    def test_precedence_before_navigation_and_interval_split(self):
        cells = frozenset((x, y) for x in range(4) for y in (0, 2))
        problem = Problem(cells, {"R1": (0, 0), "R2": (0, 2)}, (
            Task(1, "R1", frozenset({(2, 0)}), 2),
            Task(2, "R2", frozenset({(3, 2)}), 1, (1,))), 1., 15)
        result = solve(problem, 2)
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["makespan_ticks"], 8)
        self.assertGreater(result["stats"]["precedence_splits"], 0)
        self.assertGreaterEqual(result["plans"]["R2"].starts[2], 4)
        self.assertTrue(validate(problem, result["plans"]))

    @staticmethod
    def exhaustive_two_goal(problem):
        """Independent joint BFS oracle: one 1-tick service per robot."""
        robots = sorted(problem.starts)
        goals = {t.robot: t.goals for t in problem.tasks}
        initial = (tuple(problem.starts[r] for r in robots), (False, False))
        frontier, visited = {initial}, {initial}
        for tick in range(problem.horizon+1):
            if any(all(done) for positions, done in frontier):
                return tick
            following = set()
            for positions, done in frontier:
                choices = []
                for i, r in enumerate(robots):
                    p = positions[i]
                    options = [(p, done[i])]
                    if not done[i] and p in goals[r]:
                        options.append((p, True))
                    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                        q = p[0]+dx,p[1]+dy
                        if q in problem.cells:
                            options.append((q, done[i]))
                    choices.append(options)
                for a,b in itertools.product(*choices):
                    if separation(positions[0], a[0], positions[1], b[0]) < problem.clearance-1e-9:
                        continue
                    state = ((a[0],b[0]), (a[1],b[1]))
                    if state not in visited:
                        following.add(state)
                        visited.add(state)
            frontier = following
        return None

    def test_swept_conflicts_match_tiny_exhaustive_oracle(self):
        cells = frozenset(itertools.product(range(3), range(2)))
        cases = [((0,0),(2,0),(2,0),(0,0)), ((0,0),(2,1),(2,0),(0,1)),
                 ((0,0),(1,0),(2,0),(0,0))]
        for a,b,ga,gb in cases:
            with self.subTest(starts=(a,b)):
                problem = Problem(cells,{"R1":a,"R2":b},(
                    Task(1,"R1",frozenset({ga}),1), Task(2,"R2",frozenset({gb}),1)),1.,12)
                result = solve(problem,5)
                self.assertEqual(result["status"],"solved")
                self.assertEqual(result["makespan_ticks"],self.exhaustive_two_goal(problem))
                self.assertTrue(validate(problem,result["plans"]))

    def test_service_cannot_be_interrupted_by_collision_avoidance(self):
        result = solve(self.single(),2)
        plan = result["plans"]["R1"]
        bad = list(plan.path)
        bad[3] = (1,0)
        with self.assertRaisesRegex(ValueError,"Preempted service"):
            validate(self.single(),{"R1":Plan(tuple(bad),plan.starts,plan.finishes,plan.services)})

    def test_horizon_is_not_reported_as_unbounded_infeasibility(self):
        result = solve(replace(self.single(),horizon=3),2)
        self.assertEqual(result["status"],"infeasible_within_horizon")
        self.assertIsNone(result["makespan_ticks"])

    def test_fixed_order_and_capability_validation(self):
        problem = replace(self.single(), tasks=self.single().tasks+(
            Task(2,"R1",frozenset({(0,0)}),1),))
        with self.assertRaisesRegex(ValueError,"sequence edges"):
            solve(problem,2)
        with self.assertRaisesRegex(ValueError,"Invalid robot"):
            solve(replace(self.single(),tasks=(Task(1,"R9",frozenset({(0,0)}),1),)),2)

    def test_initial_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,"overlapping"):
            solve(replace(self.single(),starts={"R1":(0,0),"R2":(0,0)}),2)

    def test_timeout_is_explicit_and_has_partial_search_counters(self):
        result = solve(self.single(),1e-12)
        self.assertEqual(result["status"],"timeout")
        self.assertIn("low_level_expanded",result["stats"])


if __name__ == "__main__":
    unittest.main()
