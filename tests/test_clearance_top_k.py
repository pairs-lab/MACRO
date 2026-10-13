import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from core.yielding import select_clearance_top_k,select_goal_directed_top_k,clearance_yield_key,_select_clearance_ranked
from simulations.standalone.relocation_search import _nearest_feasible_relocations,_path_conflict


def candidate(direction,time,margin=.2,goal_cost=0.):
    angle=direction*math.pi/4
    end=(round(math.cos(angle),6),round(math.sin(angle),6))
    return dict(destination=end,path=[(0,0),end],delay=time,distance=time,
                direction_group=direction,clearance_margin=margin,goal_directed_cost=goal_cost)


class ClearanceTopKTests(unittest.TestCase):
    def test_selection_matches_original_scan_for_randomized_pools(self):
        def reference(pool, count):
            ordered = sorted(pool, key=clearance_yield_key)
            if count is None:
                return ordered
            if count <= 0 or not ordered:
                return []
            unique, seen = [], set()
            for c in ordered:
                rank = clearance_yield_key(c)
                key = rank[3], rank[4]
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            selected, remaining = [unique[0]], unique[1:]
            while remaining and len(selected) < int(count):
                used = {c.get('direction_group') for c in selected}
                novel = [(i, c) for i, c in enumerate(remaining)
                         if c.get('direction_group') is not None
                         and c.get('direction_group') not in used]
                def separation(item):
                    i, c = item
                    gap = min((min(abs(c['direction_group'] - d),
                                   8 - abs(c['direction_group'] - d))
                               for d in used if d is not None), default=0)
                    return gap, -i
                index = max(novel, key=separation)[0] if novel else 0
                selected.append(remaining.pop(index))
            return selected
        for seed in range(150):
            rng = random.Random(seed)
            pool = []
            for i in range(rng.randrange(1, 120)):
                c = candidate(rng.randrange(8), rng.choice([.1, .2, 1., 2.]),
                              margin=rng.choice([.1, .2, .3]))
                c['destination'] = (rng.randrange(-5, 6), rng.randrange(-5, 6))
                c['path'] = [(0, 0), c['destination']]
                if rng.random() < .2:
                    c.pop('direction_group')
                pool.append(c)
            pool += pool[:5]
            for count in (None, -1, 0, .5, 1, 3, 7, 15, 200):
                self.assertEqual(select_clearance_top_k(pool, count),
                                 reference(pool, count), msg=(seed, count))
                entries = [(clearance_yield_key(c), i, c.get('direction_group'))
                           for i, c in enumerate(pool)]
                self.assertEqual([pool[i] for i in _select_clearance_ranked(entries, count)],
                                 reference(pool, count), msg=('lightweight', seed, count))

    def test_opposite_retreat_survives_forward_goal_biased_candidates(self):
        pool=[candidate(0,.1,goal_cost=1),candidate(1,.2,goal_cost=2),
              candidate(2,.3,goal_cost=3),candidate(4,.5,goal_cost=100)]
        old=select_goal_directed_top_k(pool,3)
        new=select_clearance_top_k(pool,3)
        self.assertNotIn(4,[c["direction_group"] for c in old])
        self.assertIn(4,[c["direction_group"] for c in new])
        self.assertEqual(new[0]["delay"],.1)

    def test_goal_and_dag_metadata_never_affect_selection(self):
        pool=[candidate(i,1+i/10) for i in range(8)]
        expected=[c["destination"] for c in select_clearance_top_k(pool,3)]
        for i,c in enumerate(pool):
            c.update(goal_directed_cost=-1000*i,goal_remaining=999*i,slack=-999*i)
        self.assertEqual(expected,[c["destination"] for c in select_clearance_top_k(pool,3)])

    def test_equal_time_prefers_more_body_clearance_and_is_deterministic(self):
        pool=[candidate(0,1,.2),candidate(4,1,.4),candidate(2,1,.3)]
        expected=select_clearance_top_k(pool,3)
        self.assertEqual(expected[0]["clearance_margin"],.4)
        for seed in range(10):
            shuffled=list(pool);random.Random(seed).shuffle(shuffled)
            self.assertEqual(select_clearance_top_k(shuffled,3),expected)
        self.assertEqual(select_clearance_top_k(pool,0),[])
        self.assertEqual(len(select_clearance_top_k([*pool,*pool],10)),3)

    def test_real_grid_preserves_backward_bay_and_all_safety_checks(self):
        start=(4,4);goal=(8,4);corridor=[(4,y) for y in range(2,7)]
        kwargs=dict(mover_id="R2",positions={"R2":start,"R1":(4,2)},
            protected_paths={"R1":corridor},navigation_cells=frozenset((x,y) for x in range(9) for y in range(9)),
            robot_radius=.4,grid_size=1.,speed=2.,goal=goal)
        selected=_nearest_feasible_relocations(**kwargs,path_budget=3)
        self.assertTrue(any(c["destination"][0]<start[0] for c in selected))
        for c in selected:
            self.assertFalse(_path_conflict(corridor,[c["destination"]],.8))
            self.assertGreater(c["clearance_margin"],0.)
            self.assertEqual(c["delay"],c["distance"]/2.)
            self.assertNotIn("goal_directed_cost",c)
        no_goal=_nearest_feasible_relocations(**dict(kwargs,goal=None),path_budget=3)
        self.assertEqual(selected,no_goal)


if __name__=="__main__":
    unittest.main()
