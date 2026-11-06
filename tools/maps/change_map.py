"""Edit robots and objects in a scenario without modifying its base map."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2

from tools.maps.create_map import (
    GRID_MIN_X,
    GRID_MIN_Z,
    GridMapCreator,
    OBJECT_STYLES,
)
from tools.maps.map_tags import map_directory as tagged_map_directory
from tools.maps.map_tags import scene_path


SCENARIO_TOOLS = [
    ("paint_black", "BLACK RECT"),
    ("paint_white", "WHITE RECT"),
    ("move", "MOVE ITEM"),
    ("robot", "ADD ROBOT"),
    ("bowl", "ADD BOWL"),
    ("apple", "ADD APPLE"),
    ("tomato", "ADD TOMATO"),
    ("sink", "ADD SINK"),
    ("garbage", "ADD GARBAGE"),
    ("custom", "ADD CUSTOM"),
    ("delete", "DELETE ITEM"),
]


class ScenarioEditor(GridMapCreator):
    tools = SCENARIO_TOOLS
    window_name = "Change robots and objects"
    instructions = "Drag black/white | MOVE: item then target | U:undo | S:save | Q:quit"

    def __init__(self, source, output_name=None):
        self.source_scenario, map_directory = self._resolve_source(source)
        super().__init__(map_directory)
        if not self.output_file.is_file():
            raise FileNotFoundError(f"Base map does not exist: {self.output_file}")

        self.scenario_file = self._resolve_output(map_directory, output_name)
        self.robots = []
        self.objects = []
        self.tasks = []
        self.terrain_overridden = False
        self.move_selection = None
        self.mode = "move"
        self._load_scenario()

    @staticmethod
    def _scenario_in_directory(directory):
        canonical = directory / "scenario.json"
        if canonical.is_file():
            return canonical
        numbered_default = directory / "scenario_1.json"
        if numbered_default.is_file():
            return numbered_default
        alternatives = sorted(directory.glob("scenario*.json"))
        if len(alternatives) == 1:
            return alternatives[0]
        if len(alternatives) > 1:
            choices = ", ".join(path.name for path in alternatives)
            raise ValueError(
                f"Multiple scenarios in {directory}: {choices}. "
                "Pass the exact source scenario path."
            )
        return None

    @classmethod
    def _resolve_source(cls, value):
        path = Path(value).expanduser().resolve()
        if path.is_dir() or not path.suffix:
            return cls._scenario_in_directory(path), path
        if path.name == "map.json":
            return cls._scenario_in_directory(path.parent), path.parent
        if path.suffix == ".json" and path.stem.startswith("scenario"):
            return (path if path.is_file() else None), path.parent
        raise ValueError("Pass a map folder, map.json, or scenario*.json")

    @staticmethod
    def _resolve_output(map_directory, output_name):
        if output_name is None:
            return None
        output = Path(output_name)
        if output.parent == Path("."):
            output = map_directory / output
        else:
            output = output.expanduser().resolve()
        if output.parent != map_directory:
            raise ValueError("The scenario output must stay in the same map folder")
        if output.suffix != ".json" or not output.stem.startswith("scenario"):
            raise ValueError("Scenario filename must look like scenario_2.json")
        return output

    def _load_scenario(self):
        if self.source_scenario is not None:
            data = json.loads(self.source_scenario.read_text(encoding="utf-8"))
            self.robots = [dict(robot) for robot in data.get("robots", [])]
            self.objects = [dict(item) for item in data.get("objects", [])]
            self.tasks = [dict(task) for task in data.get("tasks", [])]
            if "free_spans" in data:
                free_cells = {
                    (x, int(raw_z))
                    for raw_z, spans in data["free_spans"].items()
                    for start, end in spans
                    for x in range(int(start), int(end) + 1)
                }
                all_cells = {
                    (x, z)
                    for x in range(GRID_MIN_X, self.max_x + 1)
                    for z in range(GRID_MIN_Z, self.max_z + 1)
                }
                self.blocked_cells = all_cells - free_cells
                self.terrain_overridden = True
            self.message = f"Loaded {self.source_scenario.name}"
        else:
            self.message = "New empty scenario"

        if self.scenario_file is None:
            self.scenario_file = (
                self.source_scenario
                if self.source_scenario is not None
                else self.map_directory / "scenario.json"
            )
        elif self.scenario_file != self.source_scenario:
            self.message += f"; saving as {self.scenario_file.name}"

    def _scenario_data(self):
        data = {
            "base_world": "map.json",
            "description": f"Task scenario for {self.map_directory.name}.",
            "objects": self.objects,
            "robots": self.robots,
            "tasks": self.tasks,
        }
        if self.terrain_overridden:
            data["free_spans"] = self._world_data()["free_spans"]
        return data

    def _snapshot(self):
        snapshot = super()._snapshot()
        snapshot["terrain_overridden"] = self.terrain_overridden
        snapshot["tasks"] = [dict(task) for task in self.tasks]
        return snapshot

    def _save(self):
        self.map_directory.mkdir(parents=True, exist_ok=True)
        temporary = self.map_directory / f".{self.scenario_file.name}.change-map.tmp"
        temporary.write_text(
            json.dumps(self._scenario_data(), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.scenario_file)
        self.message = f"Saved scenario only: {self.scenario_file}"
        print(self.message)

    def _clear(self):
        self._push_undo()
        self.blocked_cells.clear()
        self.terrain_overridden = True
        self._save()
        self.message = f"Cleared black cells in {self.scenario_file.name}"
        print(self.message)

    def _apply_drag(self):
        selected = self._cell_rectangle(self.drag_start, self.drag_end)
        proposed = (
            self.blocked_cells | selected
            if self.drag_value
            else self.blocked_cells - selected
        )
        invalid_robot = next(
            (
                robot.get("id", "robot")
                for robot in self.robots
                if not self._robot_has_clearance(
                    self._item_cell(robot, "start_cell"), proposed
                )
            ),
            None,
        )
        if invalid_robot is not None:
            self.message = (
                f"Rejected: rectangle removes navigation clearance around "
                f"{invalid_robot}"
            )
            return
        self._push_undo()
        if self.drag_value:
            self.blocked_cells.update(selected)
            action = "Painted black"
        else:
            self.blocked_cells.difference_update(selected)
            action = "Painted white"
        self.terrain_overridden = True
        self._save()
        self.message = f"{action}: {self.drag_start} -> {self.drag_end}"
        print(self.message)

    def _undo(self):
        if not self.undo_stack:
            self.message = "Nothing to undo"
            return
        snapshot = self.undo_stack.pop()
        self.blocked_cells = snapshot["blocked_cells"]
        self.robots = snapshot["robots"]
        self.objects = snapshot["objects"]
        self.tasks = snapshot.get("tasks", self.tasks)
        self.terrain_overridden = snapshot.get("terrain_overridden", False)
        self._save()
        self.move_selection = None
        self.message = "Undid last operation"

    def _item_at(self, cell, include_visual_radius=False):
        for index, robot in enumerate(self.robots):
            centre = self._item_cell(robot, "start_cell")
            radius = self._robot_radius_cells() if include_visual_radius else 0.0
            if centre is not None and math.dist(centre, cell) <= radius + 1e-9:
                return "robot", index
        for index, item in enumerate(self.objects):
            centre = self._item_cell(item, "cell")
            radius = (
                float(item.get("radius_cells", 1))
                if include_visual_radius
                else 0.0
            )
            if centre is not None and math.dist(centre, cell) <= radius + 1e-9:
                return "object", index
        return None

    def _robot_radius_cells(self):
        map_data = json.loads(self.output_file.read_text(encoding="utf-8"))
        return float(map_data.get("robot_radius", 0.14)) / float(
            map_data.get("grid_size", 0.05)
        )

    def _delete_item(self, cell):
        found = self._item_at(cell, include_visual_radius=True)
        if found is None:
            self.message = f"No robot or object at {cell}"
            return

        kind, index = found
        self._push_undo()
        if kind == "robot":
            robot_id = str(self.robots[index].get("id"))
            removed = self.robots.pop(index)["id"]
            referenced_tasks = [
                int(task["id"])
                for task in self.tasks
                if robot_id in [str(value) for value in task.get("candidate", [])]
            ]
        else:
            object_name = str(self.objects[index].get("name"))
            referenced_tasks = [
                int(task["id"])
                for task in self.tasks
                if task.get("object") == object_name
                or task.get("target") == object_name
            ]
            removed = self.objects.pop(index)["name"]

        self.move_selection = None
        self._save()
        task_note = (
            "; tasks still reference it: "
            + ", ".join(f"T{task_id}" for task_id in sorted(referenced_tasks))
            + " (re-add the same name/ID before running)"
            if referenced_tasks
            else ""
        )
        self.message = f"Deleted {removed} at {cell}{task_note}"
        print(self.message)

    def _move_item(self, cell):
        if self.move_selection is None:
            found = self._item_at(cell, include_visual_radius=True)
            if found is None:
                self.message = f"No item at {cell}; click a robot/object first"
                return
            self.move_selection = found
            kind, index = found
            item = self.robots[index] if kind == "robot" else self.objects[index]
            label = item.get("id") if kind == "robot" else item.get("name")
            self.message = f"Moving {label}; click its new cell"
            return

        kind, index = self.move_selection
        item = self.robots[index] if kind == "robot" else self.objects[index]
        key = "start_cell" if kind == "robot" else "cell"
        old_cell = self._item_cell(item, key)
        occupied = self._item_at(cell)
        if occupied is not None and cell != old_cell:
            self.message = f"Rejected: {cell} already contains an item"
            return
        if kind == "robot" and cell in self.blocked_cells:
            self.message = f"Rejected: robot centre {cell} is a black cell"
            return
        if kind == "robot" and not self._robot_has_clearance(cell):
            self.message = (
                f"Rejected: robot body at {cell} overlaps black cells; "
                "choose a centre at least 3 cells from black"
            )
            return

        self._push_undo()
        item[key] = list(cell)
        if kind == "object" and old_cell is not None:
            dx, dz = cell[0] - old_cell[0], cell[1] - old_cell[1]
            interactions = item.get("interaction_cells")
            if isinstance(interactions, list):
                item["interaction_cells"] = [
                    [int(point[0]) + dx, int(point[1]) + dz]
                    for point in interactions
                    if isinstance(point, list) and len(point) == 2
                ]
        label = item.get("id") if kind == "robot" else item.get("name")
        self.move_selection = None
        self._save()
        self.message = f"Moved {label}: {old_cell} -> {cell}"
        print(self.message)

    def _handle_name_key(self, key):
        handled = super()._handle_name_key(key)
        if self.mode == "paint_black":
            self.mode = "move"
        if self.custom_name is not None:
            self.message = f"Custom '{self.custom_name}' ready; click any cell"
        return handled

    def _place_item(self, cell):
        previous_mode = self.mode
        count_before = len(self.objects) + len(self.robots)
        super()._place_item(cell)
        if previous_mode == "custom" and len(self.objects) + len(self.robots) > count_before:
            self.mode = "move"

    def _on_mouse(self, event, pixel_x, pixel_y, _flags, _parameter):
        self.hover_cell = (
            self._cell_from_pixel(pixel_x, pixel_y)
            if self._inside_canvas(pixel_x, pixel_y)
            else None
        )
        if event == cv2.EVENT_LBUTTONDOWN:
            tool = self._tool_at_pixel(pixel_x, pixel_y)
            if tool is not None:
                self.move_selection = None
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
            if self.mode == "move":
                self._move_item(cell)
                return
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edit a scenario while keeping map.json unchanged"
    )
    parser.add_argument(
        "source",
        help="map tag (map1) or legacy map/scenario path",
    )
    parser.add_argument(
        "scene",
        nargs="?",
        help="scene tag, for example scene1",
    )
    parser.add_argument(
        "--output",
        help="scenario filename to create, for example scenario_2.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.scene is not None:
        directory = tagged_map_directory(arguments.source)
        selected_scene = scene_path(directory, arguments.scene)
        if selected_scene.is_file():
            source = selected_scene
            output = arguments.output
        else:
            source = directory / "map.json"
            output = arguments.output or selected_scene.name
    else:
        source = Path(arguments.source)
        output = arguments.output
    ScenarioEditor(source, output).run()
