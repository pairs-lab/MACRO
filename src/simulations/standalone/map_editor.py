"""Interactive rectangle editor for standalone JSON occupancy maps."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .world import load_world


WINDOW_NAME = "Standalone Map Editor"


class MapEditor:
    """Paint black rectangular regions directly into a world overlay."""

    def __init__(self, world_file, width=1000, height=760):
        self.world_file = Path(world_file).expanduser().resolve()
        self.width = int(width)
        self.height = int(height)
        self.margin = 34
        self.drag_start = None
        self.drag_end = None
        self.session_regions = []
        self.message = "Drag left mouse to paint blocked cells"

        self.data = json.loads(self.world_file.read_text(encoding="utf-8"))
        self.region_field = (
            "extra_blocked_regions"
            if "base_world" in self.data
            else "blocked_regions"
        )
        self.data.setdefault(self.region_field, [])
        self.world = load_world(self.world_file)
        xs = [cell[0] for cell in self.world.free_cells]
        zs = [cell[1] for cell in self.world.free_cells]
        self.bounds = min(xs), max(xs), min(zs), max(zs)
        self._configure_view()
        self.background = self._draw_background()

    def _configure_view(self):
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

    def _pixel(self, cell):
        min_x, _, _, max_z = self.bounds
        return (
            int(round(self.offset_x + (cell[0] - min_x + 0.5) * self.scale)),
            int(round(self.offset_y + (max_z - cell[1] + 0.5) * self.scale)),
        )

    def _cell_from_pixel(self, pixel_x, pixel_y):
        min_x, max_x, min_z, max_z = self.bounds
        cell_x = math.floor((pixel_x - self.offset_x) / self.scale) + min_x
        cell_z = max_z - math.floor((pixel_y - self.offset_y) / self.scale)
        return (
            min(max(int(cell_x), min_x), max_x),
            min(max(int(cell_z), min_z), max_z),
        )

    def _inside_grid(self, pixel_x, pixel_y):
        min_x, max_x, min_z, max_z = self.bounds
        grid_width = (max_x - min_x + 1) * self.scale
        grid_height = (max_z - min_z + 1) * self.scale
        return (
            self.offset_x <= pixel_x < self.offset_x + grid_width
            and self.offset_y <= pixel_y < self.offset_y + grid_height
        )

    def _draw_background(self):
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

    @staticmethod
    def _normalized_rectangle(first, second):
        return (
            [min(first[0], second[0]), min(first[1], second[1])],
            [max(first[0], second[0]), max(first[1], second[1])],
        )

    def _temporary_path(self):
        return self.world_file.with_name(
            f".{self.world_file.stem}.map-editor.tmp.json"
        )

    def _save_and_reload(self):
        temporary = self._temporary_path()
        temporary.write_text(
            json.dumps(self.data, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            candidate = load_world(temporary)
            displaced = [
                robot["id"]
                for robot in candidate.robots
                if "requested_start_cell" in robot
            ]
            if displaced:
                raise ValueError(
                    "blocked region covers robot start/clearance: "
                    + ", ".join(displaced)
                )
            os.replace(temporary, self.world_file)
            self.world = candidate
            self.background = self._draw_background()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _commit_rectangle(self, first, second):
        minimum, maximum = self._normalized_rectangle(first, second)
        region = {
            "name": f"painted_block_{len(self.data[self.region_field]) + 1:03d}",
            "min_cell": minimum,
            "max_cell": maximum,
        }
        self.data[self.region_field].append(region)
        try:
            self._save_and_reload()
        except Exception as error:
            self.data[self.region_field].pop()
            self.message = f"Rejected: {error}"
            print(self.message)
            return
        self.session_regions.append(region["name"])
        self.message = (
            f"Saved {region['name']}: {minimum} -> {maximum}"
        )
        print(self.message)

    def _undo(self):
        if not self.session_regions:
            self.message = "Nothing from this editing session to undo"
            return
        name = self.session_regions.pop()
        regions = self.data[self.region_field]
        index = next(
            (i for i in range(len(regions) - 1, -1, -1)
             if regions[i].get("name") == name),
            None,
        )
        if index is None:
            self.message = f"Could not find {name}"
            return
        removed = regions.pop(index)
        try:
            self._save_and_reload()
        except Exception:
            regions.insert(index, removed)
            self.session_regions.append(name)
            raise
        self.message = f"Undid {name}"
        print(self.message)

    def _on_mouse(self, event, pixel_x, pixel_y, _flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self._inside_grid(pixel_x, pixel_y):
                return
            cell = self._cell_from_pixel(pixel_x, pixel_y)
            self.drag_start = cell
            self.drag_end = cell
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = self._cell_from_pixel(pixel_x, pixel_y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = self._cell_from_pixel(pixel_x, pixel_y)
            self._commit_rectangle(self.drag_start, self.drag_end)
            self.drag_start = None
            self.drag_end = None

    def _draw(self):
        frame = self.background.copy()
        if self.drag_start is not None and self.drag_end is not None:
            minimum, maximum = self._normalized_rectangle(
                self.drag_start, self.drag_end
            )
            first = self._pixel((minimum[0], maximum[1]))
            second = self._pixel((maximum[0], minimum[1]))
            half = int(math.ceil(self.scale / 2))
            top_left = (first[0] - half, first[1] - half)
            bottom_right = (second[0] + half, second[1] + half)
            overlay = frame.copy()
            cv2.rectangle(overlay, top_left, bottom_right, (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            cv2.rectangle(frame, top_left, bottom_right, (40, 80, 240), 2)
            self.message = f"Selection: {minimum} -> {maximum}"

        cv2.rectangle(frame, (0, 0), (self.width, 32), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "DRAG: paint black | U: undo | Q/Esc: save and quit",
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (0, self.height - 34),
            (self.width, self.height),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            self.message[:110],
            (14, self.height - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return frame

    def run(self):
        # A fixed-size image keeps mouse pixels aligned exactly with grid cells.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self._draw())
                key = cv2.waitKey(16) & 0xFF
                if key in (27, ord("q")):
                    return
                if key == ord("u"):
                    self._undo()
        finally:
            cv2.destroyWindow(WINDOW_NAME)


def run_map_editor(world_file):
    MapEditor(world_file).run()
