import unittest
import inspect
from types import SimpleNamespace

import numpy as np

from simulations.ai2thor.config import SHOW_NAVIGATION_PATH
from simulations.ai2thor import run_simulation
from simulations.ai2thor.display import draw_hover_coordinate, draw_keyframe_note


class AI2ThorDisplayTests(unittest.TestCase):
    def test_public_runner_accepts_layout(self):
        self.assertIn("layout", inspect.signature(run_simulation).parameters)

    def test_navigation_path_overlay_is_disabled(self):
        self.assertFalse(SHOW_NAVIGATION_PATH)

    def test_keyframe_note_is_drawn_in_upper_left(self):
        frame = np.zeros((120, 640, 3), dtype=np.uint8)
        draw_keyframe_note(frame, "T7: R2 place Bowl -> Fridge")

        self.assertGreater(int(frame[:55, :500].sum()), 0)
        self.assertEqual(int(frame[70:, :, :].sum()), 0)

    def test_hovered_overhead_pixel_reports_world_coordinate(self):
        camera = SimpleNamespace(
            cursor_pixel=(640, 360),
            cursor_world={"x": -0.4, "z": 1.0},
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_hover_coordinate(frame, camera)
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
