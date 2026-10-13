"""Cache optimizations must preserve values and mutation isolation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulations.standalone import planner
from simulations.standalone.relocation_search import _nearest_feasible_relocations


class GeometryCacheCopyTests(unittest.TestCase):
    def test_clearance_outputs_remain_isolated_inside_snapshot_cache(self):
        cells = {(x, y) for x in range(-4, 5) for y in range(-4, 5)}
        arguments = dict(mover_id='R2', positions={'R1': (-2, 0), 'R2': (0, 0)},
                         protected_paths={'R1': [(-2, 0), (2, 0)]},
                         navigation_cells=cells, robot_radius=.2,
                         grid_size=1., speed=1., goal=(3, 3))
        expected = _nearest_feasible_relocations(**arguments)
        changed = dict(arguments, positions={'R1': (-2, 0), 'R2': (1, 0)})
        expected_changed = _nearest_feasible_relocations(**changed)

        @planner.with_geometry_cache
        def check():
            first = _nearest_feasible_relocations(**arguments)
            self.assertEqual(first, expected)
            self.assertTrue(first)
            first[0]['path'].clear()
            first[0]['delay'] = -999
            self.assertEqual(_nearest_feasible_relocations(**arguments), expected)
            self.assertEqual(_nearest_feasible_relocations(**changed), expected_changed)
        check()

    def test_copy_schemas_match_deepcopy(self):
        path = [(0, 0), (1, 0), (2, 1)]
        schemas = {
            'astar': path, '_astar_around_robot_bodies': path,
            'select_preferred_path': path, 'preferred_path_to_object': path,
            'shortest_paths_to_any': [path, [(0, 0), (0, 1)]],
            'shortest_path_distances': {(0, 0): 0.0, (1, 0): 1.0},
            'unknown': {'nested': [[1]]},
        }
        for name, value in schemas.items():
            with self.subTest(name=name):
                cloned = planner._copy_geometry_result(name, value)
                self.assertEqual(cloned, copy.deepcopy(value))
                self.assertIsNot(cloned, value)
        self.assertIsNone(planner._copy_geometry_result('astar', None))

    def test_cached_paths_and_distances_are_isolated_on_miss_and_hit(self):
        cells = {(x, y) for x in range(4) for y in range(4)}

        @planner.with_geometry_cache
        def check():
            def paths():
                return planner.shortest_paths_to_any(
                    cells, (0, 0), {(3, 3), (3, 2)}, path_count=2)
            expected = paths()
            first = paths()
            first[0].clear()
            first.append([(99, 99)])
            self.assertEqual(paths(), expected)
            second = paths()
            second[1][0] = (99, 99)
            self.assertEqual(paths(), expected)
            distances = planner.shortest_path_distances(cells, (0, 0))
            original = dict(distances)
            distances.clear()
            self.assertEqual(planner.shortest_path_distances(cells, (0, 0)), original)
            route = planner.astar(cells, (0, 0), (3, 3))
            expected_route = list(route)
            route.clear()
            self.assertEqual(planner.astar(cells, (0, 0), (3, 3)), expected_route)
            def tree():
                return planner.shortest_path_tree_to_any(
                    cells, (0, 0), {(3, 3), (3, 2)}, path_count=2)
            expected_tree = tree()
            changed_tree = tree()
            for container in changed_tree:
                container.clear()
            self.assertEqual(tree(), expected_tree)
        check()


if __name__ == '__main__':
    unittest.main()
