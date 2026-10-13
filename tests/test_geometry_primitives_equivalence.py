"""Freeze pre-optimization arithmetic and rasterization for exact comparisons."""
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulations.standalone.dynamic_collision import swept_separation, swept_robot_corridor, _disk_offsets
from simulations.standalone.anticipated_delay import Segment, ReservationTable


class GeometryPrimitiveEquivalenceTests(unittest.TestCase):
    def test_cached_blockers_match_original_order_and_values(self):
        def reference(table, segment, clearance):
            seen=set(); hits=[]
            for slot in table.slots(segment.start,segment.end):
                for other in table.buckets.get(slot, ()):
                    if id(other) in seen or other.robot == segment.robot:
                        continue
                    seen.add(id(other))
                    begin=max(segment.start,other.start)
                    end=min(segment.end,other.end,table.horizon)
                    if end-begin <= 1e-9: continue
                    if other.moving or (table.robot_order is not None and not other.anchored):
                        order=table.robot_order
                        other_key=((other.task_start,other.robot) if order is None else
                                   (order.get(other.robot,-1),other.task_start,other.robot))
                        own_key=((segment.task_start,segment.robot) if order is None else
                                 (order.get(segment.robot,-1),segment.task_start,segment.robot))
                        if other_key > own_key: continue
                    if swept_separation(segment.position(begin),segment.position(end),
                                        other.position(begin),other.position(end)) < clearance-1e-8:
                        hits.append(other)
            return hits
        rng=random.Random(104)
        for trial in range(100):
            def make():
                start=rng.choice([0.,.5,1.,2.]);end=start+rng.choice([.25,1.,3.])
                first=(rng.randrange(-3,4),rng.randrange(-3,4))
                last=first if rng.random()<.4 else (rng.randrange(-3,4),rng.randrange(-3,4))
                return Segment('R'+str(rng.randrange(5)),1,start,end,first,last,start,rng.random()<.5)
            reservations=[make() for _ in range(30)]
            order=None if trial%2 else {'R'+str(i):i for i in range(5)}
            table=ReservationTable(reservations,.125,5.,order)
            table._reuse_positions = True
            for _ in range(20):
                query=make();clearance=rng.choice([.1,1.,2.])
                expected=reference(table,query,clearance)
                self.assertEqual(table.blockers(query,clearance),expected)
                self.assertEqual(table.blockers(query,clearance),expected)
            self.assertLessEqual(len(table._positions),4096)
        segment=Segment('R1',1,0.,10.,(0.,0.),(2.,0.),0.)
        table=ReservationTable([segment],1.,10.)
        for i in range(5000): table._reservation_position(segment,i/1000)
        self.assertEqual(len(table._positions),4096)

    def test_swept_distance_exact_reference(self):
        def reference(a0, a1, b0, b1):
            delta = (a0[0]-b0[0], a0[1]-b0[1])
            velocity = (a1[0]-a0[0]-b1[0]+b0[0], a1[1]-a0[1]-b1[1]+b0[1])
            norm = sum(v*v for v in velocity)
            t = max(0., min(1., -sum(d*v for d,v in zip(delta,velocity))/norm)) if norm else 0.
            return math.hypot(*(d+t*v for d,v in zip(delta,velocity)))
        rng = random.Random(91)
        for i in range(10000):
            points = [tuple(rng.uniform(-100,100) for _ in range(2)) for _ in range(4)]
            if i % 3 == 0:
                points[1] = points[0]
                points[3] = points[2]
            self.assertEqual(reference(*points), swept_separation(*points))
        self.assertEqual(swept_separation((0,0),(1,0),(1,0),(0,0)),0.)

    def test_segment_position_exact_reference(self):
        rng = random.Random(44)
        for size in (2,3):
            for _ in range(1000):
                first = tuple(rng.uniform(-100,100) for _ in range(size))
                last = tuple(rng.uniform(-100,100) for _ in range(size))
                start = rng.uniform(0,10); end = start + rng.uniform(.01,5)
                at = rng.uniform(-5,20)
                fraction = max(0., min(1., (at-start)/(end-start)))
                reference = tuple(a+fraction*(b-a) for a,b in zip(first,last))
                self.assertEqual(Segment('R1',1,start,end,first,last,start).position(at),reference)

    def test_disk_corridor_exact_and_bounded(self):
        _disk_offsets.cache_clear()
        for clearance in [0.,.2,1.,math.sqrt(2),2.,2.+1e-12] + [i/20 for i in range(100)]:
            extent = math.ceil(clearance)
            reference = tuple((x,y) for x in range(-extent,extent+1)
                              for y in range(-extent,extent+1) if math.hypot(x,y)<=clearance)
            self.assertEqual(_disk_offsets(clearance), reference)
            if clearance > 0:
                path = [(-1,0),(0,0),(1,1)]
                expected = {(cx+x,cy+y) for cx,cy in path for x,y in reference}
                self.assertEqual(swept_robot_corridor(path,clearance/2,clearance/2,1.),expected)
        self.assertLessEqual(_disk_offsets.cache_info().currsize,64)
