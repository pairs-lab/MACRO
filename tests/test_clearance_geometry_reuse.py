import os
import random
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulations.standalone import planner, native_geometry
from simulations.standalone.clearance_geometry import protected_geometry, endpoint_margin, endpoint_margins
from simulations.standalone.relocation_search import _nearest_feasible_relocations, _clearance_tree_prefixes
from core.yielding import distance_to_polyline, _prepare_polyline, _distance_to_prepared_polyline


class ClearanceGeometryReuseTests(unittest.TestCase):
    @unittest.skipUnless(native_geometry._native is not None and
                         hasattr(native_geometry._native, 'endpoint_margins'),
                         'batch C++ extension not installed')
    def test_batch_margins_exactly_match_python_and_cache(self):
        rng = random.Random(29)
        for seed in range(100):
            paths = [[(rng.uniform(-20, 20), rng.uniform(-20, 20))
                      for _ in range(size)] for size in (0, 1, 2, seed % 30)]
            if seed % 2 == 0:
                paths[2][1] = paths[2][0]
            endpoints = [(rng.randrange(-20, 20), rng.randrange(-20, 20)) for _ in range(30)]
            radius = rng.choice([.1, .25, .5])
            grid = rng.choice([.25, .5, 1.])
            with patch.dict(os.environ, MARCO_GEOMETRY_BACKEND='python'):
                geometry = protected_geometry(paths, radius, grid)
                reference = endpoint_margins(endpoints, geometry)
            for reuse in ('0', '1'):
                with patch.dict(os.environ, MARCO_GEOMETRY_BACKEND='cpp', MARCO_CLEARANCE_REUSE=reuse):
                    @planner.with_geometry_cache
                    def check():
                        geometry = protected_geometry(paths, radius, grid)
                        self.assertEqual(reference, endpoint_margins(endpoints, geometry))
                        self.assertEqual(reference, endpoint_margins(endpoints, geometry))
                        self.assertEqual([], endpoint_margins([], geometry))
                    check()

    def test_prepared_distances_are_bit_identical_to_original_arithmetic(self):
        rng = random.Random(17)
        for size in (0, 1, 2, 3, 20):
            for _ in range(100):
                path = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(size)]
                if len(path) > 1 and rng.random() < .5:
                    path[1] = path[0]
                prepared = _prepare_polyline(path)
                for _ in range(5):
                    point = (rng.uniform(-50, 50), rng.uniform(-50, 50))
                    self.assertEqual(distance_to_polyline(point, path),
                                     _distance_to_prepared_polyline(point, prepared))
        for epsilon in (0., 1e-8, 1e-6):
            path = [(0, 0), (epsilon, epsilon)]
            self.assertEqual(distance_to_polyline((1, 1), path),
                             _distance_to_prepared_polyline((1, 1), _prepare_polyline(path)))

    def test_cached_candidates_equal_and_changed_geometry_not_reused(self):
        modes=['python'] + (['cpp'] if native_geometry._native is not None else [])
        cells={(x,y) for x in range(-6,7) for y in range(-6,7)}
        base=dict(mover_id='R2',positions={'R1':(-2,0),'R2':(0,0),'R3':(0,-2)},
                  protected_paths={'R1':[(-2,0),(0,0),(3,0)]},navigation_cells=cells,
                  robot_radius=.2,grid_size=.5,speed=8.,goal=(4,4))
        cases=[base,base,dict(base,positions=dict(base['positions'],R3=(1,-2))),
               dict(base,robot_radius=.25),dict(base,grid_size=.4),
               dict(base,protected_paths={'R1':[(-2,1),(0,1),(3,1)]}),base]
        for mode in modes:
            outputs=[]
            for enabled in ('0','1'):
                with patch.dict(os.environ,MARCO_GEOMETRY_BACKEND=mode,MARCO_CLEARANCE_REUSE=enabled):
                    @planner.with_geometry_cache
                    def execute():
                        results=[_nearest_feasible_relocations(**c) for c in cases]
                        return {'results':results,'search_stats':{}}
                    outputs.append(execute())
            self.assertEqual(outputs[0]['results'],outputs[1]['results'])
            stats=outputs[1]['search_stats']['clearance_geometry_reuse']
            self.assertGreater(stats['corridor_hits'],0)
            self.assertGreater(stats['margin_hits'],0)
            self.assertGreater(stats.get('native_body',{}).get('hits',stats['body_hits']),0)
        self.assertIsNone(planner._GEOMETRY_CACHE.get())

    def test_path_history_and_changed_corridor_remain_distinct(self):
        with patch.dict(os.environ,MARCO_GEOMETRY_BACKEND='python',MARCO_CLEARANCE_REUSE='1'):
            @planner.with_geometry_cache
            def check():
                # The same edge can occur before or after body clearance.
                a={ (0,0):(1,0), (1,0):(2,0)}
                b={ (1,0):(0,0), (2,0):(1,0)}
                first=_clearance_tree_prefixes(a,[(2,0),(1,0),(0,0)],(2,0),set(),[(0,0)],1.,1.)
                second=_clearance_tree_prefixes(b,[(0,0),(1,0),(2,0)],(0,0),set(),[(0,0)],1.,1.)
                self.assertFalse(first[(0,0)][0])
                self.assertTrue(second[(2,0)][0])
                blocked=_clearance_tree_prefixes(b,[(0,0),(1,0),(2,0)],(0,0),{(2,0)},[(0,0)],1.,1.)
                self.assertFalse(blocked[(2,0)][0])
                geometry=protected_geometry([[(0,0),(3,0)]],.2,.5)
                other=protected_geometry([[(0,1),(3,1)]],.2,.5)
                self.assertNotEqual(endpoint_margin((0,2),geometry),endpoint_margin((0,2),other))
            check()


if __name__=='__main__':unittest.main()
