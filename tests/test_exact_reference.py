from dataclasses import replace
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

from core import exact_reference as exact
from core.pc_cbs_legacy import Problem, Task, Plan, validate


class ExactReferenceTests(unittest.TestCase):
    def problem(self):
        return Problem(frozenset((x,y) for x in range(3) for y in range(3)),
            {'R1':(0,0)}, (Task(1,'R1',frozenset({(2,2)}),.13),), .2,20.,8.)

    def test_exact_irrational_ordering_and_cancellation(self):
        self.assertLess(exact.ExactTime(Fraction(7,5)),exact.ExactTime(diagonal=Fraction(1)))
        self.assertLess(exact.ExactTime(diagonal=Fraction(1)),exact.ExactTime(Fraction(10,7)))
        a=exact.ExactTime(Fraction(-7,5));b=exact.ExactTime(diagonal=Fraction(-1))
        self.assertLess(b,a)
        self.assertEqual((a+b)-a,b)

    def test_diagonal_service_not_quantized_and_optimal_certificate(self):
        p=self.problem();r=exact.solve(p,2.)
        self.assertEqual(r['status'],'optimal')
        self.assertEqual(r['lower_bound_exact'],{'rational':'13/100','sqrt2_coefficient':'1/4'})
        self.assertAlmostEqual(r['upper_bound_s'],.13+math.sqrt(2)/4,places=14)
        self.assertTrue(validate(p,r['plans']))
        self.assertFalse(r['complete_joint_solver'])

    def test_collision_does_not_get_optimal_label(self):
        p=replace(self.problem(),starts={'R1':(0,1),'R2':(2,1)},tasks=(
            Task(1,'R1',frozenset({(2,1)}),.13),Task(2,'R2',frozenset({(0,1)}),.13)))
        r=exact.solve(p,2.)
        self.assertEqual(r['status'],'bounded')
        self.assertEqual(r['unresolved_reason'],'swept_collision')
        self.assertFalse(r['optimality_claim'])
        self.assertIsNone(r['plans'])
        self.assertAlmostEqual(r['lower_bound_s'],.38)

    def test_external_precedence_is_not_silently_dropped(self):
        p=replace(self.problem(),starts={'R1':(0,0),'R2':(2,2)},tasks=(
            Task(1,'R1',frozenset({(0,0)}),1.,(2,)),Task(2,'R2',frozenset({(2,2)}),1.)))
        r=exact.solve(p,2.)
        self.assertEqual(r['status'],'bounded')
        self.assertEqual(r['unresolved_reason'],'precedence_or_first_ready')

    def test_solved_incumbent_is_only_upper_bound_not_optimality(self):
        p=replace(self.problem(),starts={'R1':(0,0),'R2':(2,2)},tasks=(
            Task(1,'R1',frozenset({(0,0)}),1.,(2,)),Task(2,'R2',frozenset({(2,2)}),1.)))
        plans={'R1':Plan(((0,0),(0,0),(0,0)),(0.,1.,2.),{1:1.},{1:2.},{1:1.}),
               'R2':Plan(((2,2),(2,2)),(0.,1.),{2:0.},{2:1.},{2:0.})}
        with patch('core.exact_reference.legacy.solve',return_value={
                'status':'solved','plans':plans,'makespan_s':2.}):
            r=exact.solve(p,2.,incumbent_time_limit=1.)
        self.assertEqual(r['status'],'bounded')
        self.assertEqual(r['lower_bound_s'],1.)
        self.assertEqual(r['upper_bound_s'],2.)
        self.assertEqual(r['gap_s'],1.)
        self.assertFalse(r['optimality_claim'])

    def test_timeout_cannot_be_certified(self):
        with patch('core.exact_reference.time.perf_counter',side_effect=[0.,2.,2.,2.]):
            r=exact.solve(self.problem(),1.)
        self.assertEqual(r['status'],'bound_timeout')
        self.assertFalse(r['optimality_claim'])

    def test_swept_edge_swap_has_zero_exact_separation(self):
        self.assertEqual(exact.separation_squared((0,0),(1,0),(1,0),(0,0)),0)

    def test_disconnected_relaxation_proves_infeasible(self):
        p=replace(self.problem(),cells=frozenset({(0,0),(2,2)}))
        r=exact.solve(p,2.)
        self.assertEqual(r['status'],'infeasible')
        self.assertFalse(r['optimality_claim'])

    def test_multiple_goals_internal_precedence(self):
        p=replace(self.problem(),tasks=(Task(1,'R1',frozenset({(2,0),(0,1)}),.2),
            Task(2,'R1',frozenset({(0,2)}),.3,(1,))))
        r=exact.solve(p,2.)
        self.assertEqual(r['status'],'optimal')
        self.assertEqual(r['lower_bound_s'],.75)
        self.assertTrue(validate(p,r['plans']))


if __name__=='__main__':
    unittest.main()
