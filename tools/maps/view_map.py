"""Read-only scenario geometry viewer using the simulation's map style."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulations.standalone.world import load_world


WINDOW_NAME = "Map view (no robots or objects)"


class MapViewer:
    """Draw only black obstacles and the fine white navigation grid."""

    def __init__(self, world_file, width=980, height=760):
        self.world_file = Path(world_file).expanduser().resolve()
        self.world = load_world(self.world_file)
        self.width = int(width)
        self.height = int(height)
        self.margin = 28
        self.hover_cell = None
        self.selected_cell = None

        xs = [cell[0] for cell in self.world.free_cells]
        zs = [cell[1] for cell in self.world.free_cells]
        self.bounds = min(xs), max(xs), min(zs), max(zs)
        min_x, max_x, min_z, max_z = self.bounds
        self.scale = min(
            (self.width - 2 * self.margin) / (max_x - min_x + 1),
            (self.height - 2 * self.margin) / (max_z - min_z + 1),
        )
        self.offset_x = (
            self.width - (max_x - min_x + 1) * self.scale
        ) / 2
        self.offset_y = (
            self.height - (max_z - min_z + 1) * self.scale
        ) / 2
        self.background = self._draw_map()

    def _pixel(self, cell):
        min_x, _, _, max_z = self.bounds
        return (
            int(round(self.offset_x + (cell[0] - min_x + 0.5) * self.scale)),
            int(round(self.offset_y + (max_z - cell[1] + 0.5) * self.scale)),
        )

    def _cell_from_pixel(self, pixel_x, pixel_y):
        min_x, _, _, max_z = self.bounds
        return (
            math.floor((pixel_x - self.offset_x) / self.scale) + min_x,
            max_z - math.floor((pixel_y - self.offset_y) / self.scale),
        )

    def _draw_map(self):
        # These colors and cell borders intentionally match main.py.
        frame = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        half = self.scale / 2
        for cell in self.world.free_cells:
            centre = self._pixel(cell)
            cv2.rectangle(
                frame,
                (int(centre[0] - half), int(centre[1] - half)),
                (int(centre[0] + half), int(centre[1] + half)),
                (248, 248, 248),
                -1,
            )
        if self.scale >= 5:
            for cell in self.world.free_cells:
                centre = self._pixel(cell)
                cv2.rectangle(
                    frame,
                    (int(centre[0] - half), int(centre[1] - half)),
                    (int(centre[0] + half), int(centre[1] + half)),
                    (220, 220, 220),
                    1,
                )
        return frame

    def _draw(self):
        frame = self.background.copy()
        cv2.rectangle(frame, (0, 0), (self.width, 24), (25, 25, 25), -1)
        status = "VIEW ONLY | Click: coordinate | Q/Esc: quit"
        if self.hover_cell is not None:
            status += f" | cell={self.hover_cell}"
        cv2.putText(
            frame,
            status,
            (12, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        if self.selected_cell is not None:
            centre = self._pixel(self.selected_cell)
            cv2.circle(frame, centre, max(3, int(self.scale * 0.35)), (20, 80, 240), 1)
        return frame

    def _on_mouse(self, event, pixel_x, pixel_y, _flags, _parameter):
        cell = self._cell_from_pixel(pixel_x, pixel_y)
        self.hover_cell = cell
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_cell = cell
            state = "FREE" if cell in self.world.free_cells else "BLOCKED"
            print(f"Selected cell=[{cell[0]}, {cell[1]}], state={state}")

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, self.width, self.height)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self._draw())
                key = cv2.waitKey(16) & 0xFF
                if key in (27, ord("q")):
                    return
        finally:
            cv2.destroyWindow(WINDOW_NAME)


def parse_args():
    parser = argparse.ArgumentParser(
        description="View scenario geometry with the main simulation grid style"
    )
    parser.add_argument("world", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    MapViewer(arguments.world).run()
