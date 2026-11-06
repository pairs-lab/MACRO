"""Load the standalone grid, robots, and user-defined objects from JSON."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


Cell = tuple[int, int]


@dataclass(frozen=True)
class World:
    grid_size: float
    free_cells: frozenset[Cell]
    navigation_cells: frozenset[Cell]
    robots: tuple[dict, ...]
    objects: dict[str, dict]
    tasks: tuple[dict, ...]
    robot_radius: float


def _cell(value, label):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise ValueError(f"{label} must be [x_cell, z_cell]")
    return tuple(value)


def _line_cells(start, end):
    """Rasterize an integer line so long furniture can be defined by endpoints."""
    x, z = start
    end_x, end_z = end
    dx = abs(end_x - x)
    dz = -abs(end_z - z)
    step_x = 1 if x < end_x else -1
    step_z = 1 if z < end_z else -1
    error = dx + dz
    cells = []
    while True:
        cells.append((x, z))
        if (x, z) == (end_x, end_z):
            return cells
        doubled = 2 * error
        if doubled >= dz:
            error += dz
            x += step_x
        if doubled <= dx:
            error += dx
            z += step_z


def _object_cells(item, name):
    shape = item.get("shape", "point")
    if shape == "point":
        return {_cell(item["cell"], f"object {name}")}
    if shape != "segment":
        raise ValueError(f"Object {name} has unsupported shape {shape!r}")
    start = _cell(item["start_cell"], f"object {name} start")
    end = _cell(item["end_cell"], f"object {name} end")
    thickness = max(0, int(item.get("thickness_cells", 0)))
    return {
        (cell[0] + dx, cell[1] + dz)
        for cell in _line_cells(start, end)
        for dx in range(-thickness, thickness + 1)
        for dz in range(-thickness, thickness + 1)
    }


def _point_in_polygon(cell, polygon):
    """Return whether an integer cell centre lies inside/on a polygon."""
    x, z = cell
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, z1 = previous
        x2, z2 = current
        cross = (x - x1) * (z2 - z1) - (z - z1) * (x2 - x1)
        if (
            cross == 0
            and min(x1, x2) <= x <= max(x1, x2)
            and min(z1, z2) <= z <= max(z1, z2)
        ):
            return True
        if (z1 > z) != (z2 > z):
            intersection_x = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if x <= intersection_x:
                inside = not inside
        previous = current
    return inside


def _load_world_data(path, ancestors=()):
    """Resolve compact world overlays relative to their instance file."""
    path = Path(path).expanduser().resolve()
    if path in ancestors:
        raise ValueError(f"Cyclic base_world reference: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    base = data.pop("base_world", None)
    overrides = data.pop("task_overrides", {})
    extra_free_regions = data.pop("extra_free_regions", ())
    extra_blocked_regions = data.pop("extra_blocked_regions", ())
    if base is not None:
        inherited = _load_world_data(path.parent / base, (*ancestors, path))
        inherited.update(data)
        data = inherited
    data["free_regions"] = [
        *data.get("free_regions", ()),
        *extra_free_regions,
    ]
    data["blocked_regions"] = [
        *data.get("blocked_regions", ()),
        *extra_blocked_regions,
    ]
    tasks = {str(task["id"]): task for task in data.get("tasks", ())}
    for task_id, changes in overrides.items():
        if task_id not in tasks:
            raise ValueError(f"Override references unknown task {task_id}")
        if set(changes) - {"candidate", "node", "duration"}:
            raise ValueError(f"Unsupported task override for {task_id}")
        tasks[task_id].update(changes)
    return data


def load_world(path, *, geometry_only=False) -> World:
    data = _load_world_data(path)
    if geometry_only:
        # Benchmark generators replace agents/tasks after sampling geometry.
        # Keep normal scenario loading strict: a missing robot list elsewhere
        # must not silently become an empty mission.
        data = dict(data, robots=[], tasks=[])
    grid_size = float(data.get("grid_size", 0.05))
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    free_cells = set()
    for raw_z, spans in data["free_spans"].items():
        z = int(raw_z)
        for start, end in spans:
            free_cells.update((x, z) for x in range(int(start), int(end) + 1))

    # Optional white rectangles can reopen small portions of the frozen map.
    for region in data.get("free_regions", ()):
        name = region.get("name", "free region")
        minimum = _cell(region["min_cell"], f"{name} min")
        maximum = _cell(region["max_cell"], f"{name} max")
        free_cells.update(
            (x, z)
            for x in range(minimum[0], maximum[0] + 1)
            for z in range(minimum[1], maximum[1] + 1)
        )

    # Rectangles/polygons customize the frozen map without AI2-THOR.
    for region in data.get("blocked_regions", ()):
        name = region.get("name", "region")
        if "polygon" in region:
            polygon = [
                _cell(point, f"{name} polygon point")
                for point in region["polygon"]
            ]
            if len(polygon) < 3:
                raise ValueError(f"{name} polygon needs at least 3 points")
            min_x = min(point[0] for point in polygon)
            max_x = max(point[0] for point in polygon)
            min_z = min(point[1] for point in polygon)
            max_z = max(point[1] for point in polygon)
            free_cells.difference_update(
                (x, z)
                for x in range(min_x, max_x + 1)
                for z in range(min_z, max_z + 1)
                if _point_in_polygon((x, z), polygon)
            )
        else:
            minimum = _cell(region["min_cell"], f"{name} min")
            maximum = _cell(region["max_cell"], f"{name} max")
            free_cells.difference_update(
                (x, z)
                for x in range(minimum[0], maximum[0] + 1)
                for z in range(minimum[1], maximum[1] + 1)
            )

    objects = {}
    for item in data.get("objects", ()):
        name = str(item["name"])
        if name in objects:
            raise ValueError(f"Duplicate object name: {name}")
        normalized = dict(item)
        footprint = _object_cells(item, name)
        normalized["cells"] = frozenset(footprint)
        if item.get("shape", "point") == "segment":
            normalized["start_cell"] = _cell(
                item["start_cell"], f"object {name} start"
            )
            normalized["end_cell"] = _cell(
                item["end_cell"], f"object {name} end"
            )
            normalized["cell"] = (
                round(
                    (normalized["start_cell"][0] + normalized["end_cell"][0])
                    / 2
                ),
                round(
                    (normalized["start_cell"][1] + normalized["end_cell"][1])
                    / 2
                ),
            )
        else:
            normalized["cell"] = next(iter(footprint))
        if "interaction_cells" in item:
            normalized["interaction_cells"] = frozenset(
                _cell(value, f"object {name} interaction cell")
                for value in item["interaction_cells"]
            )
            if not normalized["interaction_cells"]:
                raise ValueError(
                    f"Object {name} interaction_cells cannot be empty"
                )
        objects[name] = normalized
        if item.get("blocking", False):
            free_cells.difference_update(footprint)

    robot_radius = float(data.get("robot_radius", 0.10))
    radius_cells = robot_radius / grid_size
    radius = int(math.ceil(radius_cells))
    clearance_offsets = [
        (dx, dz)
        for dx in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if math.hypot(dx, dz) <= radius_cells + 1e-9
    ]
    navigation_cells = {
        cell
        for cell in free_cells
        if all(
            (cell[0] + dx, cell[1] + dz) in free_cells
            for dx, dz in clearance_offsets
        )
    }
    if not navigation_cells:
        raise ValueError("Robot-radius erosion removed the entire map")
    for name, item in objects.items():
        invalid_interactions = (
            set(item.get("interaction_cells", ())) - navigation_cells
        )
        if invalid_interactions:
            raise ValueError(
                f"Object {name} has interaction cells outside the "
                f"navigation grid: {sorted(invalid_interactions)}"
            )

    robots = []
    seen_ids = set()
    for item in data["robots"]:
        robot = dict(item)
        robot_id = str(robot["id"])
        if robot_id in seen_ids:
            raise ValueError(f"Duplicate robot ID: {robot_id}")
        seen_ids.add(robot_id)
        robot["start_cell"] = _cell(robot["start_cell"], f"{robot_id} start")
        if robot["start_cell"] not in navigation_cells:
            robot["requested_start_cell"] = robot["start_cell"]
            robot["start_cell"] = min(
                navigation_cells,
                key=lambda cell: math.dist(cell, robot["start_cell"]),
            )
        robots.append(robot)

    robot_ids = {robot["id"] for robot in robots}
    tasks = []
    task_ids = set()
    supported_actions = {
        "pick",
        "place",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        "wash",
    }
    for item in data.get("tasks", ()):
        task = dict(item)
        task_id = int(task["id"])
        if task_id in task_ids:
            raise ValueError(f"Duplicate task ID: {task_id}")
        task_ids.add(task_id)
        task["id"] = task_id
        if "candidate" in task:
            task["candidate"] = [str(value) for value in task["candidate"]]
        elif "robot" in task:
            # Backward-compatible fixed assignment.
            task["candidate"] = [str(task["robot"])]
        else:
            raise ValueError(f"Task {task_id} has no candidate robots")
        task.pop("robot", None)
        task["action"] = str(task["action"])
        task["object"] = str(task["object"])
        task["node"] = [int(value) for value in task.get("node", ())]
        task["duration"] = float(task.get("duration", 1.0))
        if not task["candidate"]:
            raise ValueError(f"Task {task_id} has no candidate robots")
        unknown_robots = set(task["candidate"]) - robot_ids
        if unknown_robots:
            raise ValueError(
                f"Task {task_id} uses unknown robots {sorted(unknown_robots)}"
            )
        if task["action"] not in supported_actions:
            raise ValueError(f"Task {task_id} has unsupported action")
        destination = (
            str(task.get("target", ""))
            if task["action"] == "place"
            else task["object"]
        )
        if destination not in objects:
            raise ValueError(f"Task {task_id} uses unknown object {destination!r}")
        if not math.isfinite(task["duration"]) or task["duration"] < 0:
            raise ValueError(f"Task {task_id} duration must be non-negative")
        tasks.append(task)
    for task in tasks:
        missing = set(task["node"]) - task_ids
        if missing:
            raise ValueError(f"Task {task['id']} has missing dependencies {missing}")
    remaining = {task["id"]: set(task["node"]) for task in tasks}
    completed = set()
    while remaining:
        ready = {task_id for task_id, deps in remaining.items() if deps <= completed}
        if not ready:
            raise ValueError("Task dependencies contain a cycle")
        completed.update(ready)
        for task_id in ready:
            del remaining[task_id]

    return World(
        grid_size=grid_size,
        free_cells=frozenset(free_cells),
        navigation_cells=frozenset(navigation_cells),
        robots=tuple(robots),
        objects=objects,
        tasks=tuple(tasks),
        robot_radius=robot_radius,
    )
