import math
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulations.standalone import planner, native_geometry
from simulations.standalone.relocation_search import _clearance_tree_prefixes


@unittest.skipIf(native_geometry._native is None, 'optional C++ extension not installed')
class NativeGeometryTests(unittest.TestCase):
    @unittest.skipUnless(native_geometry._native is not None and
                         hasattr(native_geometry._native, 'prepare_grid'),
                         'native grid preparation not installed')
    def test_prepared_grid_matches_dynamic_search_and_changed_grid(self):
        native = native_geometry._native
        for seed in range(80):
            rng = random.Random(seed + 901)
            free = {(x,y) for x in range(-5,6) for y in range(-5,6) if rng.random() > .2}
            free.add((0,0))
            prepared = native.prepare_grid(free)
            for _ in range(4):
                blocked = {p for p in free if rng.random() < .15}
                goals = {p for p in free if rng.random() < .3} | {(99,99)}
                count = rng.choice([0.,1.,3.,20.,1000.])
                self.assertEqual(native.shortest_path_tree(free,(0,0),goals,blocked,count),
                                 native.shortest_path_tree(prepared,(0,0),goals,blocked,count))
        with patch.dict(os.environ, MARCO_GEOMETRY_BACKEND='cpp', MARCO_NATIVE_GRID_REUSE='1'):
            @planner.with_geometry_cache
            def check():
                for size in range(1,12):
                    free = {(x,y) for x in range(size) for y in range(size)}
                    self.assertEqual(planner.shortest_path_tree_to_any(free,(0,0),free,path_count=1000),
                                     native.shortest_path_tree(free,(0,0),free,(),1000.))
                self.assertEqual(len(planner._GEOMETRY_CACHE.get()['native_navigation_grids']),8)
            check()
        self.assertIsNone(planner._GEOMETRY_CACHE.get())

    def test_body_astar_paths_equal_including_contact_and_unreachable_goals(self):
        for seed in range(100):
            rng = random.Random(seed)
            free = {(x,y) for x in range(-5,6) for y in range(-5,6)
                    if rng.random() > .18}
            free.add((0,0))
            goal = (rng.randrange(-5,6),rng.randrange(-5,6))
            centres = [(rng.randrange(-2,3),rng.randrange(-2,3)) for _ in range(seed % 4)]
            blocked = {p for p in free if rng.random() < .04}
            radius = rng.choice([.5,1.-1e-9,1.,1.+1e-9,math.sqrt(2),2.])
            results=[]
            for enabled in ('0','1'):
                with patch.dict(os.environ,MARCO_GEOMETRY_BACKEND='cpp',MARCO_NATIVE_ASTAR=enabled):
                    results.append(planner._astar_around_robot_bodies(
                        free,(0,0),goal,centres,radius,extra_obstacles=blocked))
            self.assertEqual(*results,msg=f'seed={seed}')

    def test_tree_and_prefixes_equal_on_random_grids(self):
        for seed in range(60):
            rng = random.Random(seed)
            free = {(x,y) for x in range(-6,7) for y in range(-6,7)
                    if rng.random() > .2}
            free.add((0,0))
            goals = {p for p in free if rng.random() < .5}
            blocked = {p for p in free if rng.random() < .05}
            forbidden = {p for p in free if rng.random() < .15}
            centres = [(1,0), (-1,2), (2,-2)]
            count = rng.choice([0,1,3,10,1000])
            results = []
            for mode in ('python','cpp'):
                with patch.dict(os.environ, MARCO_GEOMETRY_BACKEND=mode):
                    tree = planner.shortest_path_tree_to_any(
                        free,(0,0),goals,dynamic_obstacles=blocked,path_count=count)
                    prefixes = _clearance_tree_prefixes(
                        tree[0],tree[2],(0,0),forbidden,centres,
                        math.sqrt(2),.05)
                    results.append((tree,prefixes))
            self.assertEqual(*results, msg=f'seed={seed}')

    def test_epsilon_contacts_and_snapshot_cache_backend_switch(self):
        cells = {(x,y) for x in range(4) for y in range(4)}
        @planner.with_geometry_cache
        def compare():
            results=[]
            for mode in ('python','cpp'):
                with patch.dict(os.environ, MARCO_GEOMETRY_BACKEND=mode):
                    tree=planner.shortest_path_tree_to_any(cells,(0,0),cells,path_count=100)
                    for radius in (1.-1e-9,1.,1.+1e-9,math.sqrt(2)):
                        prefixes=_clearance_tree_prefixes(
                            tree[0],tree[2],(0,0),{(0,0)},[(1,0)],radius,1.)
                        results.append(prefixes)
            self.assertEqual(results[:4],results[4:])
        compare()


if __name__ == '__main__':
    unittest.main()
