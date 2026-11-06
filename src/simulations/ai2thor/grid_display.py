"""Lightweight occupancy-grid rendering for live AI2-THOR."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .display import draw_hover_coordinate, draw_keyframe_note
from .navigation.grid_planner import prepare_global_grid_map

from .config import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    NAVIGATION_GRID_SIZE,
    ROBOT_SAFETY_RADIUS,
)


_ROBOT_COLORS = (
    (40, 80, 240),   # R1 red
    (230, 120, 30),  # R2 blue
    (60, 190, 70),   # R3 green
    (190, 70, 190),
)
_PATH_COLORS = (
    (20, 170, 255),
    (255, 150, 30),
    (40, 190, 80),
    (200, 80, 200),
)


class GridMapRenderer:
    """Render cached free cells and live robot poses in world coordinates."""

    def __init__(self, controller, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT):
        _, free_cells = prepare_global_grid_map(controller)
        if not free_cells:
            raise RuntimeError("The global navigation grid is empty")
        self.free_cells = frozenset(free_cells)
        self.width = int(width)
        self.height = int(height)
        self.margin = 32
        xs = [cell[0] for cell in self.free_cells]
        zs = [cell[1] for cell in self.free_cells]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_z, self.max_z = min(zs), max(zs)
        map_width = self.max_x - self.min_x + 1
        map_height = self.max_z - self.min_z + 1
        self.cell_pixels = min(
            (self.width - 2 * self.margin) / map_width,
            (self.height - 2 * self.margin) / map_height,
        )
        drawn_width = map_width * self.cell_pixels
        drawn_height = map_height * self.cell_pixels
        self.origin_x = (self.width - drawn_width) / 2.0
        self.origin_y = (self.height - drawn_height) / 2.0
        self.cursor_pixel = None
        self.cursor_world = None
        self._background = self._draw_static_map()

    def on_mouse(self, event, x, y, _flags, _parameter):
        if event != cv2.EVENT_MOUSEMOVE:
            return
        self.cursor_pixel = (int(x), int(y))
        self.cursor_world = {
            "x": (
                self.min_x - 0.5 + (x - self.origin_x) / self.cell_pixels
            ) * NAVIGATION_GRID_SIZE,
            "z": (
                self.max_z + 0.5 - (y - self.origin_y) / self.cell_pixels
            ) * NAVIGATION_GRID_SIZE,
        }

    def _cell_pixel_bounds(self, cell):
        x, z = cell
        left = self.origin_x + (x - self.min_x) * self.cell_pixels
        # Screen y grows downward, while world/grid z is shown upward.
        top = self.origin_y + (self.max_z - z) * self.cell_pixels
        return (
            int(math.floor(left)),
            int(math.floor(top)),
            int(math.ceil(left + self.cell_pixels)),
            int(math.ceil(top + self.cell_pixels)),
        )

    def _draw_static_map(self):
        frame = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        for cell in self.free_cells:
            left, top, right, bottom = self._cell_pixel_bounds(cell)
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (248, 248, 248),
                -1,
            )
        # A subtle grid keeps the square-cell model visible without clutter.
        if self.cell_pixels >= 5:
            for cell in self.free_cells:
                left, top, right, bottom = self._cell_pixel_bounds(cell)
                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (222, 222, 222),
                    1,
                )
        return frame

    def world_to_pixel(self, position):
        cell_x = float(position["x"]) / NAVIGATION_GRID_SIZE
        cell_z = float(position["z"]) / NAVIGATION_GRID_SIZE
        pixel_x = self.origin_x + (cell_x - self.min_x + 0.5) * self.cell_pixels
        pixel_y = self.origin_y + (self.max_z - cell_z + 0.5) * self.cell_pixels
        return int(round(pixel_x)), int(round(pixel_y))

    def _draw_paths(self, frame, paths):
        if not isinstance(paths, dict):
            return
        for agent_id, path in sorted(paths.items()):
            if len(path) < 2:
                continue
            pixels = np.array(
                [self.world_to_pixel(point) for point in path],
                dtype=np.int32,
            )
            cv2.polylines(
                frame,
                [pixels],
                False,
                _PATH_COLORS[agent_id % len(_PATH_COLORS)],
                2,
                cv2.LINE_AA,
            )

    def render(self, event, paths=None, keyframe_note=None):
        frame = self._background.copy()
        self._draw_paths(frame, paths)
        agent_events = getattr(event, "events", None) or [event]
        radius_pixels = max(
            7,
            int(round(
                ROBOT_SAFETY_RADIUS
                / NAVIGATION_GRID_SIZE
                * self.cell_pixels
            )),
        )
        for agent_id, agent_event in enumerate(agent_events):
            agent = agent_event.metadata.get("agent", {})
            position = agent.get("position")
            if position is None:
                continue
            centre = self.world_to_pixel(position)
            color = _ROBOT_COLORS[agent_id % len(_ROBOT_COLORS)]
            cv2.circle(frame, centre, radius_pixels, color, -1, cv2.LINE_AA)
            cv2.circle(frame, centre, radius_pixels, (15, 15, 15), 2, cv2.LINE_AA)

            yaw = math.radians(float(agent.get("rotation", {}).get("y", 0.0)))
            tip = (
                int(round(centre[0] + math.sin(yaw) * radius_pixels)),
                int(round(centre[1] - math.cos(yaw) * radius_pixels)),
            )
            cv2.line(frame, centre, tip, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"R{agent_id + 1}",
                (centre[0] + radius_pixels + 4, centre[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )
        draw_keyframe_note(frame, keyframe_note)
        return draw_hover_coordinate(frame, self)
