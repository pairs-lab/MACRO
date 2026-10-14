import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from core.coordination import relocation_search as search
from simulations.grid.app import _compute_conflict_plan
import test_runtime_policies as old_tests


class PriorityClearanceTests(unittest.TestCase):
    def test_protects_full_current_route_plus_next_task_return(self):
        s=old_tests.fixture()
        robot=s["cluster"][0]
        successor=dict(id=3,robot="R1",action="toggle_on",object="return",node=[1],duration=1)
        objects=dict(s["objects"])
        objects["return"]={"cell":(0,3),"cells":frozenset({(0,3)}),
                           "interaction_cells":frozenset({(1,3)})}
        corridor=search._priority_clearance_corridor(robot,robot.path,
            [*s["tasks"],successor],s["task_status"],objects,
            s["navigation_cells"],s["speed_cells_per_second"])
        self.assertEqual(corridor[:len(robot.path)],robot.path)
        self.assertIn((1,3),corridor)
        self.assertTrue(search._path_conflict(corridor,[(1,3)],.8))

    def test_all_parking_candidates_clear_frozen_corridor_and_keep_priority_path(self):
        s=old_tests.fixture()
        # Ordinary yields must never call dynamic priority rerouting. Direct
        # reroute candidates are tested separately by the app's generator.
        with patch.object(search,"select_dynamic_replan_path",side_effect=AssertionError("priority detour")):
            plan=search.search_relocation_plan(robots=s["cluster"],tasks=s["tasks"],
                task_status=s["task_status"],objects=s["objects"],navigation_cells=s["navigation_cells"],
                robot_radius=s["robot_radius"],grid_size=s["grid_size"],
                speed_cells_per_second=s["speed_cells_per_second"],selection_policy="local_distance")
        self.assertTrue(plan["search_stats"]["preserve_priority_corridor"])
        corridor=plan["protected_priority_corridor"]
        for rid,pose in plan["final_positions"].items():
            if rid!=plan["priority_robot"]:
                self.assertFalse(search._path_conflict(corridor,[pose],.8))
        if len(plan["moved_agents"])==1:
            priority=next(r for r in s["cluster"] if r.robot_id==plan["priority_robot"])
            self.assertEqual(plan["preferred_paths"][priority.robot_id],priority.path)

    def test_local_policy_pool_is_shared_without_dag_priority_leakage(self):
        pools=[]
        for policy in ("dag_aware","local_distance","local_delay"):
            plan=_compute_conflict_plan(**old_tests.fixture(),selection_policy=policy)
            pools.append([c["geometry_key"] for c in plan["search_stats"]["candidate_evaluations"]])
        self.assertEqual(pools[0],pools[1]); self.assertEqual(pools[1],pools[2])

    def test_reroute_only_unchanged_by_clearance_switch(self):
        s=old_tests.fixture()
        outcomes=[]
        for enabled in (False,True):
            try:
                p=_compute_conflict_plan(**s,selection_policy="reroute_only",preserve_priority_corridor=enabled)
                outcomes.append(p["preferred_paths"])
            except RuntimeError as e:
                outcomes.append(str(e))
        self.assertEqual(outcomes[0],outcomes[1])




if __name__=="__main__":
    unittest.main()
