"""ROS approach-trigger geometry using the shared core; no middleware required."""
import unittest
from itertools import combinations

from simulations.ros2.three_robot_t.swept_trigger import predicted_contacts


class SweptTriggerTests(unittest.TestCase):
    def test_crossing_between_samples(self):
        # Both sample endpoints are clear; the simultaneous interiors collide.
        poses={'a':[-1,0,0],'b':[0,-1,0]}
        paths={'a':[[-1,0],[1,0]],'b':[[0,-1],[0,1]]}
        hit=predicted_contacts(poses,paths,[('a','b')],{'a':2,'b':2},1,1)
        self.assertEqual(len(hit),1)
        self.assertAlmostEqual(hit[0]['predicted_separation_m'],0.)

    def test_idle_third_robot(self):
        poses={'a':[-1,0,0],'b':[1,2,0],'idle':[0,0,0]}
        paths={'a':[[-1,0],[1,0]],'b':[[1,2],[-1,2]]}
        hits=predicted_contacts(poses,paths,list(combinations(poses,2)),
                                {'a':1,'b':1},1)
        self.assertEqual([h['robots'] for h in hits],[['a','idle']])

    def test_current_contact(self):
        hits=predicted_contacts({'a':[0,0,0],'b':[.44,0,0]}, {}, [('a','b')], {})
        self.assertEqual(len(hits),1)

    def test_visual_circles_not_trigger_radius(self):
        # Display circles at 0.25 m touch, but static 0.22 m model discs do not.
        self.assertEqual(predicted_contacts({'a':[0,0,0],'b':[.50,0,0]}, {},
                                           [('a','b')], {}),[])

    def test_same_spatial_crossing_different_time(self):
        poses={'a':[-1,0,0],'b':[0,-1,0]}
        paths={'a':[[-1,0],[1,0]],'b':[[0,-1],[0,1]]}
        self.assertEqual(predicted_contacts(poses,paths,[('a','b')],
                                           {'a':2,'b':.1},1),[])

    def test_outside_horizon(self):
        poses={'a':[-1,0,0],'b':[1,0,0]}
        paths={'a':[[-1,0],[1,0]],'b':[[1,0],[-1,0]]}
        self.assertEqual(predicted_contacts(poses,paths,[('a','b')],
                                           {'a':.1,'b':.1},1),[])

    def test_invalid_parameters(self):
        for horizon in (0.,float('nan'),11.):
            with self.assertRaises(ValueError):
                predicted_contacts({}, {}, [], {}, horizon)


if __name__=='__main__':unittest.main()
