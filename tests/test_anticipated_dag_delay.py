import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from core.coordination import relocation_search as search
from core.coordination.anticipated_delay import Segment, ReservationTable, estimate_future_delays
from simulations.grid.app import _compute_conflict_plan
import test_runtime_policies as policy_tests


class DelayPropagationTests(unittest.TestCase):
    def test_full_and_incremental_exactly_equal_without_rerouting(self):
        common,positions=policy_tests.IncrementalDagEvaluationTest._fixture()
        common.update(positions=positions,preferred_paths={},relocation_delay_by_robot={},
                      return_details=True,return_evaluation=True)
        baseline=search._remaining_dag_cost(**common)[3]
        for delays in ({1:2.3},{3:.1,5:4.},{2:1.25},{1:0.}):
            with self.subTest(delays=delays):
                full=search._remaining_dag_cost(**common,future_delay_by_task=delays)
                logs=[]
                with patch.object(search,"preferred_path_to_object",side_effect=AssertionError("A* in timing propagation")):
                    inc=search._remaining_dag_cost(**common,future_delay_by_task=delays,
                        baseline_evaluation=baseline,timing_only=True,debug_logger=logs.append)
                self.assertEqual(inc,full)
                if set(delays)=={1}:
                    for tid in (3,4,5,6):
                        self.assertEqual(inc[3]["start_times"][tid],baseline["start_times"][tid])
                        self.assertEqual(inc[3]["finish_times"][tid],baseline["finish_times"][tid])
                    self.assertEqual(logs[0]["affected_tasks"],2)
                    self.assertEqual(inc[3]["finish_times"][2],baseline["finish_times"][2]+delays[1])

    def test_slack_absorbs_delay_instead_of_external_penalty(self):
        common,positions=policy_tests.IncrementalDagEvaluationTest._fixture()
        common.update(positions=positions,preferred_paths={},relocation_delay_by_robot={})
        baseline=search._remaining_dag_cost(**common)[0]
        self.assertEqual(search._remaining_dag_cost(**common,future_delay_by_task={1:.1})[0],baseline)
        self.assertEqual(search._remaining_dag_cost(**common,future_delay_by_task={5:.1})[0],baseline+.1)

    def test_invalid_delays_rejected(self):
        common,positions=policy_tests.IncrementalDagEvaluationTest._fixture()
        for delays in ({99:1},{1:-1},{1:float("nan")},{1:float("inf")}):
            with self.assertRaises(ValueError):
                search._remaining_dag_cost(**common,positions=positions,preferred_paths={},
                    relocation_delay_by_robot={},future_delay_by_task=delays)

    def test_serial_and_cross_robot_successors_and_replacement_delays(self):
        common,positions=policy_tests.IncrementalDagEvaluationTest._fixture()
        common["tasks"][1]["node"]=[]  # Still constrained by R1 serial order.
        common["tasks"][2]["node"]=[1] # Also propagate to R2 and its successor.
        common.update(positions=positions,preferred_paths={},relocation_delay_by_robot={},
                      return_details=True,return_evaluation=True)
        baseline=search._remaining_dag_cost(**common)[3]
        logs=[]
        delayed=search._remaining_dag_cost(**common,future_delay_by_task={1:2.3},
            baseline_evaluation=baseline,timing_only=True,debug_logger=logs.append)
        self.assertEqual(delayed,search._remaining_dag_cost(**common,future_delay_by_task={1:2.3}))
        self.assertEqual(logs[0]["affected_tasks"],4)
        reset=search._remaining_dag_cost(**common,future_delay_by_task={},
            baseline_evaluation=delayed[3],timing_only=True)
        self.assertEqual(reset,search._remaining_dag_cost(**common))

    def test_future_delay_propagation_for_each_intervention_form(self):
        common,positions=policy_tests.IncrementalDagEvaluationTest._fixture()
        variants={
            "reroute":(positions,{"R1":[(0,0),(0,1),(1,1),(2,1)]},{}),
            "yield":(positions,{}, {"R2":.75}),
            "relocation":({**positions,"R1":(0,1)},
                          {"R1":[(0,1),(1,1),(2,1)]},{"R1":.5}),
            "composed":({**positions,"R1":(0,1),"R2":(0,3)},
                        {"R1":[(0,1),(1,1),(2,1)],"R2":[(0,3),(1,3),(2,3)]},
                        {"R1":.5,"R2":1.})}
        for name,(poses,paths,waits) in variants.items():
            with self.subTest(intervention=name):
                kwargs=dict(common,positions=poses,preferred_paths=paths,
                            relocation_delay_by_robot=waits,return_details=True,return_evaluation=True)
                baseline=search._remaining_dag_cost(**kwargs)[3]
                full=search._remaining_dag_cost(**kwargs,future_delay_by_task={1:2.3,4:.25})
                with patch.object(search,"preferred_path_to_object",side_effect=AssertionError("Unexpected A*")):
                    inc=search._remaining_dag_cost(**kwargs,future_delay_by_task={1:2.3,4:.25},
                        baseline_evaluation=baseline,timing_only=True)
                self.assertEqual(inc,full)


class ReservationTests(unittest.TestCase):
    def test_vertex_edge_swap_and_crossing_use_swept_clearance(self):
        move=Segment("R2",2,0,1,(0,0),(1,0),0)
        for other in (Segment("R1",1,0,1,(1,0),(0,0),0),
                      Segment("R1",1,0,1,(.5,-.5),(.5,.5),0),
                      Segment("R1",1,0,1,(.5,0),(.5,0),0)):
            table=ReservationTable([other],.125,5)
            self.assertEqual(table.blockers(move,.2),[other])
        self.assertEqual(ReservationTable([move],.125,5).blockers(move,.2),[])

    def test_simultaneous_blockers_charge_max_not_sum_and_horizon_censors(self):
        from types import SimpleNamespace
        robots=[SimpleNamespace(robot_id=r,cell=p) for r,p in
                (("R1",(1,0)),("R2",(0,0)),("R3",(1,0)))]
        tasks=[dict(id=1,robot="R1"),dict(id=2,robot="R2"),dict(id=3,robot="R3")]
        evaluation=dict(start_times={1:0.,2:0.,3:0.},finish_times={1:5.,2:1.,3:5.},
                        task_paths={1:[(1,0)],2:[(0,0),(1,0)],3:[(1,0)]},
                        task_travel_times={1:0.,2:1.,3:0.})
        delays,diagnostics=estimate_future_delays(candidate={},evaluation=evaluation,
            robots=robots,tasks=tasks,speed=1.,grid_size=1.,robot_radius=.2,horizon=5.)
        self.assertEqual(delays,{2:5.})
        self.assertTrue(diagnostics["censored"])
        self.assertEqual(diagnostics["observations"][0]["blocked_by"],["R1","R3"])

    def test_finite_clearance_wait_and_non_overlapping_reservations(self):
        from types import SimpleNamespace
        robots=[SimpleNamespace(robot_id="R2",cell=(0,0))]
        move=Segment("R2",2,0,1,(0,0),(1,0),0)
        blockers=[Segment("R1",1,0,2,(.5,0),(.5,0),0),
                  Segment("R3",3,0,3,(.5,0),(.5,0),0)]
        with patch("core.coordination.anticipated_delay.project_trajectories",return_value=[move,*blockers]):
            delays,diag=estimate_future_delays(candidate={},evaluation={},robots=robots,tasks=[],
                speed=1.,grid_size=1.,robot_radius=.2,horizon=5.)
        self.assertEqual(delays,{2:3.})
        self.assertFalse(diag["censored"])
        other=Segment("R1",1,2,3,(.5,0),(.5,0),0)
        self.assertEqual(ReservationTable([other],.125,5).blockers(move,.4),[])




if __name__=="__main__":
    unittest.main()
