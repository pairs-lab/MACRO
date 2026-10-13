"""Admissibility and objective equivalence for optimized exact assignment."""
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from core.assignment import (
    _assignment_search_metadata, _assignment_lower_bound, _prepare_assignment_bound,
    _prepare_exact_prefix_bound, _coalesce_owner_chains,
    optimize_task_assignments, evaluate_plan,
)
from simulations.standalone.app import GridDistanceOracle
from simulations.standalone.world import World


def fixture(tasks):
    free=frozenset((x,z) for x in range(7) for z in range(4))
    return World(1.,free,free,
        ({"id":"R1","start_cell":(0,0)},{"id":"R2","start_cell":(6,3)}),
        {"a":{"cell":(1,1),"cells":frozenset({(1,1)})},
         "b":{"cell":(5,1),"cells":frozenset({(5,1)})}},tuple(tasks),.1)


def task(i,parents=(),action="wash",object="a"):
    return dict(id=i,node=list(parents),action=action,object=object,
                candidate=["R1","R2"],duration=.75+i/8)


class AssignmentOptimizationTests(unittest.TestCase):
    def test_compiled_bound_matches_original_for_all_partial_assignments(self):
        tasks=[task(1),task(2,(1,)),task(3),task(4,(2,3))]
        meta=_assignment_search_metadata(tasks,None)
        geometry=dict(task_positions={1:((1,0),(2,0)),2:((3,0),),4:((0,1),)},
                      initial_positions={"R1":((0,0),),"R2":((5,0),)},distance_scale=.3)
        chain_predecessors={2:1}
        speed={"R1":1.1,"R2":2.3}
        compiled=_prepare_assignment_bound(meta,geometry,speed,chain_predecessors)
        for values in itertools.product((None,"R1","R2"),repeat=4):
            partial={i+1:r for i,r in enumerate(values) if r is not None}
            original=_assignment_lower_bound(meta["ordered_tasks"],partial,meta["service_durations"],
                robot_speed=speed,task_positions=geometry["task_positions"],
                initial_positions=geometry["initial_positions"],chain_predecessors=chain_predecessors,distance_scale=.3)
            self.assertEqual(compiled(partial),original)

    def test_prefix_bound_admissible_for_every_completion_and_gap(self):
        tasks=[task(1),task(2),task(3,(1,2),object="b"),task(4,(3,))]
        world=fixture(tasks); oracle=GridDistanceOracle(world)
        meta=_assignment_search_metadata(tasks,None)
        bound=_prepare_exact_prefix_bound(meta,oracle.prepare_assignment_routes(meta["ordered_tasks"]),2.)
        for owners in itertools.product(("R1","R2"),repeat=4):
            assignment=dict(zip(range(1,5),owners))
            t,d,_=evaluate_plan(tasks,assignment,oracle,robot_speed=2.,service_time=None)
            for mask in range(16):
                partial={tid:r for tid,r in assignment.items() if mask & (1<<(tid-1))}
                lb=bound(partial)
                self.assertLessEqual(lb[0],t+1e-9);self.assertLessEqual(lb[1],d+1e-9)
            self.assertEqual(bound(assignment),(t,d))

    def test_cached_prediction_matches_original_for_every_assignment(self):
        tasks=[task(1,action="pick"),task(2,object="b"),
               dict(task(3,(1,2),action="place"),target="b"),task(4,(3,))]
        oracle=GridDistanceOracle(fixture(tasks))
        ordered=_assignment_search_metadata(tasks,None)["ordered_tasks"]
        route=oracle.prepare_assignment_routes(ordered)
        for owners in itertools.product(("R1","R2"),repeat=4):
            positions={}
            oracle.begin_prediction()
            for t,r in zip(ordered,owners):
                location=t.get("target") if t["action"]=="place" else t["object"]
                expected=oracle.plan("robot"+r[1:],location,positions.get(r))
                self.assertEqual(route(t["id"],r,positions.get(r)),expected)
                positions[r]=expected[1]
                oracle.finish_predicted_task(t)

    def test_owner_coalescing_enforces_existing_equalities_only(self):
        tasks=[task(1,action="pick"),task(2,object="b"),
               dict(task(3,(1,2),action="place"),target="b")]
        groups=_coalesce_owner_chains(_assignment_search_metadata(tasks,None))
        self.assertTrue(any({1,3} <= set(g["task_ids"]) for g in groups))
        plans=[optimize_task_assignments(tasks,GridDistanceOracle(fixture(tasks)),
            robot_speed=2.,service_time=None,search_mode=mode)
            for mode in ("chain_bnb","task_bnb","exhaustive")]
        self.assertEqual(len({(p["makespan"],p["total_distance"]) for p in plans}),1)

    def test_random_small_dags_match_exhaustive_and_public_evaluator(self):
        rng=random.Random(114)
        for trial in range(16):
            tasks=[]
            for i in range(1,7):
                t=task(i,[p for p in range(1,i) if rng.random()<.2],object=rng.choice(("a","b")))
                if rng.random()<.3:t["candidate"]=[rng.choice(("R1","R2"))]
                tasks.append(t)
            world=fixture(tasks)
            plans=[optimize_task_assignments(tasks,GridDistanceOracle(world),robot_speed=2.,
                service_time=None,search_mode=mode) for mode in ("chain_bnb","task_bnb","exhaustive")]
            self.assertEqual(len({(p["makespan"],p["total_distance"]) for p in plans}),1)
            for p in plans:
                t,d,seq=evaluate_plan(tasks,p["assignment"],GridDistanceOracle(world),robot_speed=2.,service_time=None)
                self.assertEqual((t,d,seq),(p["makespan"],p["total_distance"],p["robot_plans"]))


if __name__=="__main__":unittest.main()
