"""Create or edit only the base navigation grid in ``map.json``.

Change the values in CONFIG below, then run::

    python -m tools.maps.create_map

The generated JSON uses the same ``free_spans`` format as the standalone
simulator. Use ``change_map.py`` for scenario robots and objects.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np

from tools.maps.map_tags import map_directory


# ---------------------------------------------------------------------------
# CONFIG: edit only this section for a new map.
# ---------------------------------------------------------------------------
GRID_WIDTH = 75
GRID_HEIGHT = 75
GRID_MIN_X = -30
GRID_MIN_Z = -41
GRID_SIZE_METERS = 0.05
# Same physical robot radius as maps/base/floorplan1_base.json.
ROBOT_RADIUS_METERS = 0.14
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "maps"
# ---------------------------------------------------------------------------


WINDOW_NAME = "Create standalone grid map"
VIEW_WIDTH = 1000
VIEW_HEIGHT = 760
MARGIN = 34

# Kept here as shared display defaults for change_map.py.
ROBOT_COLORS = [
    [50, 80, 240],
    [235, 140, 30],
    [60, 190, 70],
    [190, 80, 190],
    [40, 190, 210],
]
OBJECT_STYLES = {
    "bowl": {"radius_cells": 2, "color_bgr": [180, 180, 40]},
    "apple": {"radius_cells": 1, "color_bgr": [40, 40, 230]},
    "tomato": {"radius_cells": 1, "color_bgr": [40, 80, 240]},
    "sink": {"radius_cells": 2, "color_bgr": [210, 180, 80]},
    "garbage": {"radius_cells": 2, "color_bgr": [70, 70, 70]},
}
BASE_MAP_TOOLS = [
    ("paint_black", "BLACK RECT"),
    ("paint_white", "WHITE RECT"),
]


class GridMapCreator:
    tools = BASE_MAP_TOOLS
    window_name = WINDOW_NAME
    instructions = "Drag rectangle | U:undo | C:clear black | S:save | Q:quit"

    def __init__(self, output_file=None):
        if GRID_WIDTH <= 0 or GRID_HEIGHT <= 0:
            raise ValueError("GRID_WIDTH and GRID_HEIGHT must be positive")

        self.map_directory = (
            self._resolve_map_directory(output_file)
            if output_file is not None
            else self._next_map_directory()
        )
        self.output_file = self.map_directory / "map.json"
        self.blocked_cells: set[tuple[int, int]] = set()
        self.robots: list[dict] = []
        self.objects: list[dict] = []
        self.tasks: list[dict] = []
        self.undo_stack: list[dict] = []
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.drag_value = True
        self.hover_cell: tuple[int, int] | None = None
        self.mode = "paint_black"
        self.custom_name_input: str | None = None
        self.custom_name: str | None = None
        self.custom_style: dict | None = None
        self.message = "Select a tool, then click or drag on the grid"

        self.width = VIEW_WIDTH
        self.height = VIEW_HEIGHT
        self.scale = min(
            (self.width - 2 * MARGIN) / GRID_WIDTH,
            (self.height - 2 * MARGIN) / GRID_HEIGHT,
        )
        self.canvas_width = GRID_WIDTH * self.scale
        self.canvas_height = GRID_HEIGHT * self.scale
        self.canvas_left = (self.width - self.canvas_width) / 2
        self.canvas_top = (self.height - self.canvas_height) / 2
        self._load_existing_map()

    @staticmethod
    def _resolve_map_directory(value):
        path = Path(value).expanduser().resolve()
        if len(Path(value).parts) == 1 and not Path(value).suffix:
            return map_directory(value)
        if path.is_dir() or not path.suffix:
            return path
        if path.name == "map.json":
            return path.parent
        raise ValueError("Pass a map folder or its map.json file")

    @staticmethod
    def _next_map_directory():
        """Return the first unused maps/mapN directory."""
        output_dir = OUTPUT_DIR.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        index = 1
        while True:
            candidate = output_dir / f"map{index}"
            if not candidate.exists():
                return candidate
            index += 1

    @property
    def max_x(self):
        return GRID_MIN_X + GRID_WIDTH - 1

    @property
    def max_z(self):
        return GRID_MIN_Z + GRID_HEIGHT - 1

    def _load_existing_map(self):
        """Load the selected output if it already exists (normally it is a new file)."""
        if not self.output_file.is_file():
            return
        try:
            data = json.loads(self.output_file.read_text(encoding="utf-8"))
            metadata = data.get("editor_grid", {})
            if (
                metadata.get("width") != GRID_WIDTH
                or metadata.get("height") != GRID_HEIGHT
            ):
                self.message = "Existing output has another size; starting blank"
                return
            stored_min_x = int(metadata.get("min_x", 0))
            stored_min_z = int(metadata.get("min_z", 0))
            shift_x = GRID_MIN_X - stored_min_x
            shift_z = GRID_MIN_Z - stored_min_z
            free_cells = {
                (x + shift_x, int(raw_z) + shift_z)
                for raw_z, spans in data.get("free_spans", {}).items()
                for start, end in spans
                for x in range(int(start), int(end) + 1)
            }
            # Base maps may keep compact rectangular edits on top of the
            # frozen free_spans. Apply them in the same order as world.py so
            # create_map displays exactly the map used by the simulator.
            for region in data.get("free_regions", ()):
                minimum = region["min_cell"]
                maximum = region["max_cell"]
                free_cells.update(
                    (x + shift_x, z + shift_z)
                    for x in range(int(minimum[0]), int(maximum[0]) + 1)
                    for z in range(int(minimum[1]), int(maximum[1]) + 1)
                )
            for region in data.get("blocked_regions", ()):
                if "min_cell" not in region:
                    continue
                minimum = region["min_cell"]
                maximum = region["max_cell"]
                free_cells.difference_update(
                    (x + shift_x, z + shift_z)
                    for x in range(int(minimum[0]), int(maximum[0]) + 1)
                    for z in range(int(minimum[1]), int(maximum[1]) + 1)
                )
            all_cells = {
                (x, z)
                for x in range(GRID_MIN_X, self.max_x + 1)
                for z in range(GRID_MIN_Z, self.max_z + 1)
            }
            self.blocked_cells = all_cells - free_cells
            if shift_x or shift_z:
                self.message = (
                    f"Loaded and aligned {self.output_file.name} to Map 1 coordinates"
                )
            else:
                self.message = f"Loaded {self.map_directory.name}"
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            self.message = f"Could not load old output: {error}"

    @staticmethod
    def _row_spans(values):
        values = sorted(values)
        if not values:
            return []
        spans = []
        start = previous = values[0]
        for value in values[1:]:
            if value != previous + 1:
                spans.append([start, previous])
                start = value
            previous = value
        spans.append([start, previous])
        return spans

    def _world_data(self):
        free_spans = {}
        for z in range(GRID_MIN_Z, self.max_z + 1):
            free_x = [
                x
                for x in range(GRID_MIN_X, self.max_x + 1)
                if (x, z) not in self.blocked_cells
            ]
            spans = self._row_spans(free_x)
            if spans:
                free_spans[str(z)] = spans
        return {
            "description": (
                f"Custom {GRID_WIDTH}x{GRID_HEIGHT} standalone grid map "
                "created with create_map.py."
            ),
            "editor_grid": {
                "width": GRID_WIDTH,
                "height": GRID_HEIGHT,
                "min_x": GRID_MIN_X,
                "min_z": GRID_MIN_Z,
            },
            "grid_size": GRID_SIZE_METERS,
            "robot_radius": ROBOT_RADIUS_METERS,
            "free_spans": free_spans,
            "free_regions": [],
            "blocked_regions": [],
            "objects": [],
            "robots": [],
            "tasks": [],
        }

    def _snapshot(self):
        return {
            "blocked_cells": set(self.blocked_cells),
            "robots": [dict(robot) for robot in self.robots],
            "objects": [dict(item) for item in self.objects],
        }

    def _push_undo(self):
        self.undo_stack.append(self._snapshot())

    def _save(self):
        self.map_directory.mkdir(parents=True, exist_ok=True)
        map_temporary = self.map_directory / ".map.create-map.tmp.json"
        map_temporary.write_text(
            json.dumps(self._world_data(), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(map_temporary, self.output_file)
        self.message = f"Saved base map: {self.output_file}"
        print(self.message)

    def _inside_canvas(self, pixel_x, pixel_y):
        return (
            self.canvas_left <= pixel_x < self.canvas_left + self.canvas_width
            and self.canvas_top <= pixel_y < self.canvas_top + self.canvas_height
        )

    def _cell_from_pixel(self, pixel_x, pixel_y):
        x = int((pixel_x - self.canvas_left) // self.scale) + GRID_MIN_X
        row = int((pixel_y - self.canvas_top) // self.scale)
        z = self.max_z - row
        return (
            min(max(x, GRID_MIN_X), self.max_x),
            min(max(z, GRID_MIN_Z), self.max_z),
        )

    def _cell_rectangle(self, first, second):
        min_x, max_x = sorted((first[0], second[0]))
        min_z, max_z = sorted((first[1], second[1]))
        return {
            (x, z)
            for x in range(min_x, max_x + 1)
            for z in range(min_z, max_z + 1)
        }

    def _apply_drag(self):
        selected = self._cell_rectangle(self.drag_start, self.drag_end)
        self._push_undo()
        if self.drag_value:
            self.blocked_cells.update(selected)
            action = "Painted black"
        else:
            self.blocked_cells.difference_update(selected)
            action = "Painted white"
        self._save()
        self.message = f"{action}: {self.drag_start} -> {self.drag_end}"
        print(self.message)

    @staticmethod
    def _item_cell(item, key):
        value = item.get(key)
        return tuple(value) if isinstance(value, list) and len(value) == 2 else None

    def _occupied(self, cell):
        return any(
            self._item_cell(robot, "start_cell") == cell
            for robot in self.robots
        ) or any(
            self._item_cell(item, "cell") == cell
            for item in self.objects
        )

    def _robot_has_clearance(self, cell, blocked_cells=None):
        blocked_cells = self.blocked_cells if blocked_cells is None else blocked_cells
        radius_cells = ROBOT_RADIUS_METERS / GRID_SIZE_METERS
        extent = int(math.ceil(radius_cells))
        return all(
            GRID_MIN_X <= cell[0] + dx <= self.max_x
            and GRID_MIN_Z <= cell[1] + dz <= self.max_z
            and (cell[0] + dx, cell[1] + dz) not in blocked_cells
            for dx in range(-extent, extent + 1)
            for dz in range(-extent, extent + 1)
            if math.hypot(dx, dz) <= radius_cells + 1e-9
        )

    def _next_robot_id(self):
        used = {str(robot.get("id")) for robot in self.robots}
        index = 1
        while f"R{index}" in used:
            index += 1
        return f"R{index}"

    def _next_object_name(self, kind):
        used = {str(item.get("name")) for item in self.objects}
        if kind not in used:
            return kind
        index = 2
        while f"{kind}_{index}" in used:
            index += 1
        return f"{kind}_{index}"

    @staticmethod
    def _random_visible_color():
        """Return a varied BGR color that remains visible on the white map."""
        while True:
            color = [random.randint(35, 225) for _ in range(3)]
            if max(color) - min(color) >= 65 and sum(color) <= 570:
                return color

    def _begin_custom_name(self):
        self.mode = "custom"
        self.custom_name_input = ""
        self.custom_name = None
        self.custom_style = None
        self.message = "Type a custom object name, then press Enter"

    def _handle_name_key(self, key):
        if self.custom_name_input is None:
            return False
        if key in (10, 13):
            name = self.custom_name_input.strip()
            if not name:
                self.message = "Object name cannot be empty"
                return True
            self.custom_name = name
            self.custom_style = {
                "radius_cells": 1,
                "color_bgr": self._random_visible_color(),
            }
            self.custom_name_input = None
            self.message = f"Custom '{name}' ready; click a white cell"
            return True
        if key == 27:
            self.custom_name_input = None
            self.custom_name = None
            self.custom_style = None
            self.mode = "paint_black"
            self.message = "Custom object cancelled"
            return True
        if key in (8, 127):
            self.custom_name_input = self.custom_name_input[:-1]
            return True
        if 32 <= key <= 126 and len(self.custom_name_input) < 28:
            self.custom_name_input += chr(key)
            return True
        return True

    def _place_item(self, cell):
        if self._occupied(cell):
            self.message = f"Rejected: {cell} already contains an item"
            return
        if self.mode == "robot":
            if cell in self.blocked_cells:
                self.message = f"Rejected: robot cannot start on black cell {cell}"
                return
            if not self._robot_has_clearance(cell):
                self.message = f"Rejected: robot has insufficient clearance at {cell}"
                return
            self._push_undo()
            robot_id = self._next_robot_id()
            color = ROBOT_COLORS[(len(self.robots)) % len(ROBOT_COLORS)]
            self.robots.append(
                {"id": robot_id, "start_cell": list(cell), "color_bgr": color}
            )
            label = robot_id
        else:
            if self.mode == "custom":
                if self.custom_name is None or self.custom_style is None:
                    self.message = "Choose ADD CUSTOM and enter a name first"
                    return
                base_name = self.custom_name
                style = self.custom_style
            else:
                base_name = self.mode
                style = OBJECT_STYLES[self.mode]
            self._push_undo()
            name = self._next_object_name(base_name)
            self.objects.append(
                {
                    "name": name,
                    "cell": list(cell),
                    "radius_cells": style["radius_cells"],
                    "color_bgr": style["color_bgr"],
                    "blocking": False,
                }
            )
            label = name
        self._save()
        self.message = f"Added {label} at {cell}"
        print(self.message)
        if self.mode == "custom":
            self.custom_name = None
            self.custom_style = None
            self.mode = "paint_black"

    def _delete_item(self, cell):
        robot_index = next(
            (
                index
                for index, robot in enumerate(self.robots)
                if self._item_cell(robot, "start_cell") == cell
            ),
            None,
        )
        object_index = next(
            (
                index
                for index, item in enumerate(self.objects)
                if self._item_cell(item, "cell") == cell
            ),
            None,
        )
        if robot_index is None and object_index is None:
            self.message = f"No robot or object at {cell}"
            return
        self._push_undo()
        if robot_index is not None:
            removed = self.robots.pop(robot_index)["id"]
        else:
            removed = self.objects.pop(object_index)["name"]
        self._save()
        self.message = f"Deleted {removed} from {cell}"
        print(self.message)

    @staticmethod
    def _tool_button(index):
        left = 8
        top = 48 + index * 39
        return left, top, 142, top + 31

    def _tool_at_pixel(self, pixel_x, pixel_y):
        for index, (mode, _label) in enumerate(self.tools):
            left, top, right, bottom = self._tool_button(index)
            if left <= pixel_x <= right and top <= pixel_y <= bottom:
                return mode
        return None

    def _on_mouse(self, event, pixel_x, pixel_y, _flags, _parameter):
        self.hover_cell = (
            self._cell_from_pixel(pixel_x, pixel_y)
            if self._inside_canvas(pixel_x, pixel_y)
            else None
        )
        if event == cv2.EVENT_LBUTTONDOWN:
            tool = self._tool_at_pixel(pixel_x, pixel_y)
            if tool is not None:
                if tool == "custom":
                    self._begin_custom_name()
                else:
                    self.mode = tool
                    self.custom_name_input = None
                    self.message = f"Selected: {tool.replace('_', ' ')}"
                self.drag_start = None
                self.drag_end = None
                return
            if not self._inside_canvas(pixel_x, pixel_y):
                return
            cell = self._cell_from_pixel(pixel_x, pixel_y)
            if self.mode == "delete":
                self._delete_item(cell)
                return
            if self.mode in OBJECT_STYLES or self.mode in {"robot", "custom"}:
                self._place_item(cell)
                return
            self.drag_value = self.mode == "paint_black"
            self.drag_start = cell
            self.drag_end = cell
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self._inside_canvas(pixel_x, pixel_y):
                return
            self.drag_value = False
            self.drag_start = self._cell_from_pixel(pixel_x, pixel_y)
            self.drag_end = self.drag_start
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = self._cell_from_pixel(pixel_x, pixel_y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            if self.drag_start is None:
                return
            if event == cv2.EVENT_LBUTTONUP and self.mode not in {
                "paint_black",
                "paint_white",
            }:
                return
            if event == cv2.EVENT_RBUTTONUP and self.drag_value:
                return
            self.drag_end = self._cell_from_pixel(pixel_x, pixel_y)
            self._apply_drag()
            self.drag_start = None
            self.drag_end = None

    def _cell_box(self, cell):
        x, z = cell
        column = x - GRID_MIN_X
        row = self.max_z - z
        left = int(round(self.canvas_left + column * self.scale))
        top = int(round(self.canvas_top + row * self.scale))
        right = int(round(self.canvas_left + (column + 1) * self.scale))
        bottom = int(round(self.canvas_top + (row + 1) * self.scale))
        return left, top, right, bottom

    def _draw(self):
        frame = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        for x in range(GRID_MIN_X, self.max_x + 1):
            for z in range(GRID_MIN_Z, self.max_z + 1):
                cell = (x, z)
                blocked = cell in self.blocked_cells
                left, top, right, bottom = self._cell_box(cell)
                color = (15, 15, 15) if blocked else (245, 245, 245)
                cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
                cv2.rectangle(frame, (left, top), (right, bottom), (115, 115, 115), 1)
                if self.scale >= 42:
                    text_color = (210, 210, 210) if blocked else (70, 70, 70)
                    cv2.putText(
                        frame,
                        f"{x},{z}",
                        (left + 5, top + 17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        text_color,
                        1,
                        cv2.LINE_AA,
                    )

        for item in self.objects:
            cell = self._item_cell(item, "cell")
            if cell is None:
                continue
            left, top, right, bottom = self._cell_box(cell)
            centre = ((left + right) // 2, (top + bottom) // 2)
            radius = max(
                5,
                int(float(item.get("radius_cells", 1)) * self.scale),
            )
            color = tuple(int(value) for value in item.get("color_bgr", [90, 90, 90]))
            cv2.circle(frame, centre, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, centre, radius, (25, 25, 25), 1, cv2.LINE_AA)
            label = str(item.get("name", "?"))
            cv2.putText(
                frame,
                label,
                (centre[0] + radius + 3, centre[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (centre[0] + radius + 3, centre[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for robot in self.robots:
            cell = self._item_cell(robot, "start_cell")
            if cell is None:
                continue
            left, top, right, bottom = self._cell_box(cell)
            centre = ((left + right) // 2, (top + bottom) // 2)
            radius = max(
                7,
                int(ROBOT_RADIUS_METERS / GRID_SIZE_METERS * self.scale),
            )
            color = tuple(int(value) for value in robot.get("color_bgr", [220, 120, 30]))
            cv2.circle(frame, centre, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, centre, radius, (10, 10, 10), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(robot.get("id", "R")),
                (centre[0] + radius + 4, centre[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )

        # Rubber-band preview: the map changes only after mouse release.
        if self.drag_start is not None and self.drag_end is not None:
            min_x, max_x = sorted((self.drag_start[0], self.drag_end[0]))
            min_z, max_z = sorted((self.drag_start[1], self.drag_end[1]))
            left, top, _, _ = self._cell_box((min_x, max_z))
            _, _, right, bottom = self._cell_box((max_x, min_z))
            overlay = frame.copy()
            preview_color = (10, 10, 10) if self.drag_value else (245, 245, 245)
            border_color = (40, 80, 240) if self.drag_value else (240, 170, 40)
            cv2.rectangle(overlay, (left, top), (right, bottom), preview_color, -1)
            cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
            cv2.rectangle(frame, (left, top), (right, bottom), border_color, 2)

        for index, (mode, label) in enumerate(self.tools):
            left, top, right, bottom = self._tool_button(index)
            selected = mode == self.mode
            fill = (70, 105, 150) if selected else (55, 55, 55)
            border = (80, 160, 245) if selected else (105, 105, 105)
            cv2.rectangle(frame, (left, top), (right, bottom), fill, -1)
            cv2.rectangle(frame, (left, top), (right, bottom), border, 1)
            cv2.putText(
                frame,
                label,
                (left + 7, top + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
            swatch = None
            if mode in OBJECT_STYLES:
                swatch = tuple(OBJECT_STYLES[mode]["color_bgr"])
            elif mode == "robot":
                swatch = tuple(
                    ROBOT_COLORS[len(self.robots) % len(ROBOT_COLORS)]
                )
            elif mode == "custom":
                swatch = tuple(
                    (self.custom_style or {"color_bgr": [180, 90, 200]})[
                        "color_bgr"
                    ]
                )
            if swatch is not None:
                centre = (right - 12, (top + bottom) // 2)
                cv2.circle(frame, centre, 8, swatch, -1, cv2.LINE_AA)
                cv2.circle(frame, centre, 8, (225, 225, 225), 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"GRID {GRID_WIDTH} x {GRID_HEIGHT}    map: {self.map_directory.name}",
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self.instructions,
            (390, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        status = self.message
        if self.custom_name_input is not None:
            status = (
                f"TYPE NAME: {self.custom_name_input}_  "
                "| Enter: confirm | Backspace: erase | Esc: cancel"
            )
        if self.hover_cell is not None:
            status = f"cell={self.hover_cell} | {status}"
        cv2.rectangle(
            frame,
            (0, self.height - 34),
            (self.width, self.height),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            status[:130],
            (14, self.height - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _undo(self):
        if not self.undo_stack:
            self.message = "Nothing to undo"
            return
        snapshot = self.undo_stack.pop()
        self.blocked_cells = snapshot["blocked_cells"]
        self.robots = snapshot["robots"]
        self.objects = snapshot["objects"]
        self._save()
        self.message = "Undid last operation"

    def _clear(self):
        self._push_undo()
        self.blocked_cells.clear()
        self._save()
        self.message = "Cleared all black cells"

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                cv2.imshow(self.window_name, self._draw())
                key = cv2.waitKey(16) & 0xFF
                if self.custom_name_input is not None:
                    self._handle_name_key(key)
                    continue
                if key in (27, ord("q")):
                    self._save()
                    return
                if key == ord("u"):
                    self._undo()
                elif key == ord("c"):
                    self._clear()
                elif key == ord("s"):
                    self._save()
        finally:
            cv2.destroyWindow(self.window_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create or edit only a standalone base map"
    )
    parser.add_argument(
        "map_file",
        nargs="?",
        type=Path,
        help=(
            "existing/new map folder (for example maps/map1); when omitted, "
            "the next unused maps/mapN folder is created"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    GridMapCreator(arguments.map_file).run()
