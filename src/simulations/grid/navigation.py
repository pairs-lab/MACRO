"""Grid-specific assignment navigation adapter."""
from copy import deepcopy
from core.coordination.planner import _interaction_goals, preferred_path_to_object, path_length

class GridDistanceOracle:
    """Expose standalone A* through the shared core distance interface."""

    def __init__(self, world):
        self.world = world
        self.starts = {
            robot["id"]: robot["start_cell"] for robot in world.robots
        }
        self._cache = {}
        self.predicted_objects = deepcopy(world.objects)

    def begin_prediction(self):
        self.predicted_objects = deepcopy(self.world.objects)

    def finish_predicted_task(self, task):
        if task["action"] == "place":
            moved = self.predicted_objects[task["object"]]
            target = self.predicted_objects[task["target"]]
            moved["cell"] = target["cell"]
            moved["cells"] = frozenset({target["cell"]})

    def _interaction_positions(self, item):
        """Return exactly the reachable poses accepted by object planning."""
        explicit = item.get("interaction_cells")
        if explicit is not None:
            return set(explicit) & set(self.world.navigation_cells)
        cells = item.get("cells")
        if not cells:
            return set()
        return _interaction_goals(self.world.navigation_cells, cells)

    def optimistic_geometry(self, ordered_tasks):
        """Expose point sets used by optimistic Euclidean assignment bounds."""
        predicted = deepcopy(self.world.objects)
        task_positions = {}
        for task in ordered_tasks:
            location = (
                task.get("target")
                if task["action"] == "place"
                else task["object"]
            )
            item = predicted.get(location)
            if item is not None:
                points = self._interaction_positions(item)
                if points:
                    task_positions[task["id"]] = tuple(sorted(points))
            if task["action"] == "place" and task.get("target") in predicted:
                target = predicted[task["target"]]
                moved = predicted.get(task["object"])
                if moved is not None:
                    moved["cell"] = target.get("cell")
                    moved["cells"] = frozenset({target["cell"]})
        return {
            "initial_positions": {
                robot: (tuple(cell),) for robot, cell in self.starts.items()
            },
            "task_positions": task_positions,
            "distance_scale": self.world.grid_size,
        }

    @staticmethod
    def _robot_id(simulation_name):
        if simulation_name.startswith("robot"):
            return f"R{simulation_name[5:]}"
        return simulation_name

    def plan(self, robot, location, start_position=None):
        robot_id = self._robot_id(robot)
        start = (
            self.starts[robot_id]
            if start_position is None
            else start_position
        )
        destination = self.predicted_objects[location]
        return self._plan_destination(robot_id, location, start, destination)

    def _plan_destination(self, robot_id, location, start, destination):
        cache_key = (robot_id, location, start, tuple(sorted(destination["cells"])),
                     tuple(sorted(destination.get("interaction_cells", ()))))
        if cache_key in self._cache:
            return self._cache[cache_key]
        route = preferred_path_to_object(
            self.world.navigation_cells,
            start,
            destination["cells"],
            interaction_cells=destination.get("interaction_cells"),
            robot_speed=8.0,
            candidate_count=3,
        )
        if route is None:
            raise RuntimeError(f"No assignment route for {robot_id} to {location}")
        result = path_length(route, self.world.grid_size), route[-1]
        self._cache[cache_key] = result
        return result

    def prepare_assignment_routes(self, ordered_tasks):
        """Freeze task-specific predictions once for repeated leaf evaluation."""
        self.begin_prediction()
        destinations = {}
        for task in ordered_tasks:
            location = task.get("target") if task["action"] == "place" else task["object"]
            destinations[task["id"]] = (location, dict(self.predicted_objects[location]))
            self.finish_predicted_task(task)

        def route(task_id, robot_id, start):
            location, destination = destinations[task_id]
            return self._plan_destination(robot_id, location,
                self.starts[robot_id] if start is None else start, destination)
        return route

