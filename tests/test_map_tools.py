"""Map utilities remain importable after moving out of the project root."""

import unittest
from pathlib import Path

from tools.maps import create_map, export_ai2thor_layout, map_tags, view_map


class MapToolPathsTest(unittest.TestCase):
    def test_repository_relative_paths(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(map_tags.PROJECT_ROOT, root)
        self.assertEqual(create_map.OUTPUT_DIR, root / "maps")
        self.assertEqual(view_map.PROJECT_ROOT, root)
        self.assertEqual(export_ai2thor_layout.PROJECT_ROOT, root)

    def test_scene_resolution(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(map_tags.scene_path(root / "maps/map1", "scene1"),
                         root / "maps/map1/scenario_1.json")
