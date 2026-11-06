"""Shortest AI2-THOR path distance from a robot to an object."""

import math

from simulations.ai2thor.config import (
    NAVIGATION_GRID_SIZE,
    NAVIGATION_PADDING,
)

from .api import ROBOT_AGENT_IDS, ROBOT_NAMES
from .grid_planner import find_padded_astar_path
from .navigation import _agent_event, _interaction_goals


def calculate_path_length(path: list[dict]) -> float:
    """Sum the horizontal length of every segment in a planned path."""
    return sum(
        math.hypot(
            second["x"] - first["x"],
            second["z"] - first["z"],
        )
        for first, second in zip(path, path[1:])
    )


def plan_path_to_object(
    controller,
    object_type: str,
    agent_id: int = 0,
    start_position: dict | None = None,
):
    """Plan the same padded, collision-aware path used by navigation."""
    if not isinstance(object_type, str) or not object_type.strip():
        raise ValueError("object_type must be a non-empty string.")

    if start_position is None:
        start = _agent_event(controller.last_event, agent_id).metadata["agent"][
            "position"
        ].copy()
    else:
        start = start_position.copy()
    goals = _interaction_goals(controller, object_type.strip(), agent_id)
    path, target_object = find_padded_astar_path(
        controller,
        start,
        goals,
        grid_size=NAVIGATION_GRID_SIZE,
        padding=NAVIGATION_PADDING,
        agent_id=agent_id,
    )
    if not path:
        raise RuntimeError(
            f"No traversable path from robot{agent_id + 1} to {object_type!r}."
        )

    # Navigation skips path[0], which is the grid cell nearest the real agent.
    # Use the exact requested start so this value matches physical travel.
    executable_path = [start, *path[1:]]
    return executable_path, target_object


def get_path_distance(
    controller,
    object_type: str,
    agent_id: int = 0,
    start_position: dict | None = None,
) -> float:
    """Return shortest traversable translation distance in metres."""
    path, _ = plan_path_to_object(
        controller,
        object_type,
        agent_id,
        start_position=start_position,
    )
    return calculate_path_length(path)


def get_robot_path_distance(
    controller,
    robot: str,
    object_type: str,
    start_position: dict | None = None,
) -> float:
    """Return path distance using a generated robot1..robotN name."""
    if robot not in ROBOT_NAMES:
        raise ValueError(
            f"Unknown robot {robot!r}. Available robots: {list(ROBOT_NAMES)}"
        )
    return get_path_distance(
        controller,
        object_type,
        agent_id=ROBOT_AGENT_IDS[robot],
        start_position=start_position,
    )


class DistanceCalculator:
    """Callable path-distance calculator bound to one running controller."""

    def __init__(self, controller):
        self.controller = controller

    def optimistic_geometry(self, ordered_tasks):
        """Expose AI2-THOR pose sets for the assignment BnB lower bound.

        Distances between these points are Euclidean and therefore never
        exceed executable padded-A* travel.  Missing interaction poses are
        omitted, which weakens the bound without affecting admissibility.
        """
        initial_positions = {}
        for agent_id in ROBOT_AGENT_IDS.values():
            position = _agent_event(
                self.controller.last_event, agent_id
            ).metadata["agent"]["position"]
            initial_positions[f"R{agent_id + 1}"] = (
                (float(position["x"]), float(position["z"])),
            )

        predicted_locations = {}
        task_positions = {}
        for task in ordered_tasks:
            location = (
                task.get("target")
                if task.get("action") == "place"
                else predicted_locations.get(task.get("object"), task.get("object"))
            )
            points = set()
            if location:
                for robot in task.get("candidate", ()):
                    suffix = str(robot)[1:] if str(robot).startswith("R") else ""
                    if not suffix.isdigit():
                        continue
                    agent_id = int(suffix) - 1
                    if agent_id not in ROBOT_AGENT_IDS.values():
                        continue
                    try:
                        goals = _interaction_goals(
                            self.controller, str(location), agent_id
                        )
                    except (RuntimeError, ValueError):
                        continue
                    points.update(
                        (float(goal["x"]), float(goal["z"]))
                        for goal, _ in goals
                    )
            if points:
                task_positions[task["id"]] = tuple(sorted(points))
            if task.get("action") == "place" and task.get("target"):
                predicted_locations[str(task.get("object"))] = str(task["target"])

        return {
            "initial_positions": initial_positions,
            "task_positions": task_positions,
            "distance_scale": 1.0,
        }

    def plan(
        self,
        robot: str,
        object_type: str,
        start_position: dict | None = None,
    ) -> tuple[float, dict]:
        """Return route length and the final interaction position."""
        if robot not in ROBOT_NAMES:
            raise ValueError(
                f"Unknown robot {robot!r}. Available robots: {list(ROBOT_NAMES)}"
            )
        path, _ = plan_path_to_object(
            self.controller,
            object_type,
            agent_id=ROBOT_AGENT_IDS[robot],
            start_position=start_position,
        )
        return calculate_path_length(path), path[-1].copy()

    def __call__(
        self,
        robot: str,
        object_type: str,
        start_position: dict | None = None,
    ) -> float:
        distance, _ = self.plan(robot, object_type, start_position)
        return distance


def create_distance_function(controller) -> DistanceCalculator:
    """Create a bound callable distance calculator for schedulers."""
    return DistanceCalculator(controller)
