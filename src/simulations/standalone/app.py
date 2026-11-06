"""Animated standalone grid simulation; no Unity or AI2-THOR imports."""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from contextlib import redirect_stdout
import io
from dataclasses import dataclass, field
from multiprocessing import get_context
from types import SimpleNamespace

import cv2
import numpy as np

from core import optimize_task_assignments

from .dynamic_collision import (
    check_dynamic_robot_collisions,
    swept_robot_corridor,
    swept_separation,
)
from .planner import (
    _astar_around_robot_bodies,
    dynamic_replan_candidates,
    _interaction_goals,
    path_length,
    preferred_path_to_object,
    select_dynamic_replan_path,
    with_geometry_cache,
)
from .relocation_search import (
    _candidate_diagnostic,
    _candidate_stable_key,
    _select_resolution_candidate,
    _path_conflict,
    _route_exits_without_reentry,
    evaluate_reroute_plan,
    search_relocation_plan,
)
from .world import load_world
from .release_events import (
    continuation_motion_is_clear, priority_motion_is_clear, distance_prefix,
    choose_continuation_route, wait_for_nominal_is_better,
)


RUNTIME_ALGORITHM_REVISION = "bounded-release-events-v6-dag-recourse"


WINDOW_NAME = "Standalone Multi-Robot Grid Simulation"
_DEFAULT_COLORS = ((50, 80, 240), (235, 140, 30), (60, 190, 70))
PATH_PREVIEW_SECONDS = 0.0


def _snapshot_remaining_path(robot):
    """Return the untraversed centre path from an immutable robot snapshot."""
    if not robot.path:
        return [robot.cell]
    route = list(robot.path[robot.path_index:])
    if not route:
        return [robot.cell]
    if route[0] != robot.cell:
        route.insert(0, robot.cell)
    return route


def _all_robot_protected_paths(robots, excluded_ids=()):
    """Protect every robot's full remaining path, or its stationary body."""
    excluded = set(excluded_ids)
    return {
        robot.robot_id: _snapshot_remaining_path(robot)
        for robot in robots
        if robot.robot_id not in excluded
    }


def _route_clears_protected_corridors(
    route,
    protected_paths,
    robot_radius,
    grid_size,
):
    """Require a route to leave every footprint-wide corridor without reentry."""
    forbidden = set()
    for protected in protected_paths.values():
        forbidden.update(
            swept_robot_corridor(
                protected,
                robot_radius,
                robot_radius,
                grid_size,
            )
        )
    return _route_exits_without_reentry(route, forbidden)


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


@dataclass
class RobotState:
    robot_id: str
    cell: tuple[int, int]
    color: tuple[int, int, int]
    goal: tuple[int, int] | None = None
    path: list[tuple[int, int]] = field(default_factory=list)
    path_index: int = 0
    travelled: float = 0.0
    phase: str = "idle"
    current_task: dict | None = None
    action_remaining: float = 0.0
    interrupted_action_remaining: float | None = None
    preview_remaining: float = 0.0
    carrying: str | None = None
    waiting_for: str | None = None
    clearing_for: tuple[str, ...] = ()
    clearing_started: bool = False
    movement_credit: float = 0.0
    movement_edge: tuple | None = None


@with_geometry_cache
def _compute_conflict_plan(
    *,
    cluster,
    dag_robots,
    tasks,
    task_status,
    objects,
    navigation_cells,
    robot_radius,
    grid_size,
    speed_cells_per_second,
    selection_policy="dag_aware",
    planning_timeout_s=None,
    diagnostics=None,
    use_dag_priority=True,
    incremental_dag_evaluation=True,
    use_dag_lookahead=False,
    use_anticipated_delays=False,
    anticipated_delay_horizon=5.,
    preserve_priority_corridor=True,
    use_release_prediction=True,
    release_prediction_model="events",
    hypothesis_workers=1,
    verify_release=True,
    parallel_reserved_paths=None,
    execution_candidate_refiner=None,
):
    """Plan from immutable snapshots without touching live simulation state.

    An execution adapter may replace the grid release evaluator with its own
    candidate evaluator *before* selection. None preserves the grid method.
    The adapter must return only executable, scored candidates and must fail
    closed if its evaluation budget is exhausted.
    """
    if execution_candidate_refiner is not None and selection_policy != "dag_aware":
        raise ValueError("An execution evaluator requires dag_aware selection")
    workers = [
        robot
        for robot in cluster
        if robot.current_task is not None
        and robot.phase in {"preview", "moving", "waiting"}
    ]
    if not workers:
        if selection_policy == "reroute_only":
            raise RuntimeError("baseline_no_feasible_intervention")
        return None

    started = time.perf_counter()
    additional_candidates = []
    cluster_ids = {robot.robot_id for robot in cluster}
    external_protected_paths = _all_robot_protected_paths(
        dag_robots,
        excluded_ids=cluster_ids,
    )
    external_protected_paths.update(parallel_reserved_paths or {})
    # A direct dynamic reroute is a valid candidate for active-active as well
    # as active-idle contacts.  Holding the other contact participants while
    # one worker clears the component gives the evaluator and executor the
    # same conservative release semantics.
    for worker in workers:
        blockers = [robot for robot in cluster if robot is not worker]
        protected_paths = _all_robot_protected_paths(
            dag_robots,
            excluded_ids={worker.robot_id},
        )
        protected_paths.update(parallel_reserved_paths or {})
        clearance_cells = 2.0 * robot_radius / grid_size
        replanned = _astar_around_robot_bodies(
            navigation_cells, worker.cell, worker.goal,
            robot_centres=[cell for r in dag_robots if r.robot_id != worker.robot_id
                           for cell in _snapshot_remaining_path(r)],
            clearance_cells=clearance_cells,
        ) if selection_policy == "reroute_only" else select_dynamic_replan_path(
            navigation_cells,
            worker.cell,
            worker.goal,
            stationary_robot_centres=(robot.cell for robot in blockers),
            clearance_cells=clearance_cells,
            robot_speed=speed_cells_per_second,
            candidate_count=3,
        )
        if replanned is not None and _route_clears_protected_corridors(
            replanned,
            protected_paths,
            robot_radius,
            grid_size,
        ):
            additional_candidates.append(
                evaluate_reroute_plan(
                    worker_id=worker.robot_id,
                    route=replanned,
                    robots=dag_robots,
                    tasks=tasks,
                    task_status=task_status,
                    objects=objects,
                    navigation_cells=navigation_cells,
                    grid_size=grid_size,
                    speed_cells_per_second=speed_cells_per_second,
                    held_robot_ids=[r.robot_id for r in blockers],
                    robot_radius=robot_radius,
                    evaluate_dag=False,
                )
            )

    if selection_policy == "reroute_only":
        if not additional_candidates:
            raise RuntimeError("baseline_no_feasible_intervention")
        additional_candidates.sort(key=lambda c: (
            path_length(c["preferred_paths"][c["priority_robot"]], grid_size),
            c["priority_robot"], tuple(c["preferred_paths"][c["priority_robot"]])))
        selected = additional_candidates[0]
        for index, candidate in enumerate(additional_candidates):
            candidate["candidate_id"] = f"A{index:04d}"
            candidate["local_resolution_delay"] = _local_resolution_time(
                dict(candidate, strategy="service_reroute"), dag_robots,
                cluster_ids, navigation_cells, robot_radius, grid_size,
                speed_cells_per_second, objects)
            candidate["local_time_model"] = "current_execution_snapshot"
        selected["search_stats"] = {
            "candidate_evaluations": [_candidate_diagnostic(c, selected=c is selected)
                                      for c in additional_candidates],
            "generated_candidates": len(additional_candidates),
            "planning_wall_time_s": time.perf_counter() - started}
        return selected
    plan = search_relocation_plan(
        robots=cluster,
        dag_robots=dag_robots,
        priority_robot_ids=(robot.robot_id for robot in workers),
        participant_ids=(robot.robot_id for robot in cluster),
        tasks=tasks,
        task_status=task_status,
        objects=objects,
        navigation_cells=navigation_cells,
        robot_radius=robot_radius,
        grid_size=grid_size,
        speed_cells_per_second=speed_cells_per_second,
        additional_candidates=additional_candidates,
        external_protected_paths=external_protected_paths,
        selection_policy=selection_policy,
        planning_timeout_s=planning_timeout_s,
        dag_bound_pruning=False,
        local_time_evaluator=lambda candidate: _local_resolution_time(
            candidate, dag_robots, cluster_ids, navigation_cells,
            robot_radius, grid_size, speed_cells_per_second, objects),
        diagnostics=diagnostics,
        use_dag_priority=use_dag_priority,
        incremental_dag_evaluation=incremental_dag_evaluation,
        dag_candidate_refiner=(
            execution_candidate_refiner if execution_candidate_refiner is not None else
            _make_dag_refiner(dag_robots, tasks, task_status, objects,
                navigation_cells, robot_radius, grid_size,
                speed_cells_per_second, cluster_ids)
            if selection_policy == "dag_aware" and use_dag_lookahead else
            _make_release_refiner(dag_robots,tasks,task_status,objects,navigation_cells,
                robot_radius,grid_size,speed_cells_per_second,cluster_ids,release_prediction_model,
                anticipated_delay_horizon, verification_enabled=verify_release,
                hypothesis_workers=hypothesis_workers)
            if selection_policy == "dag_aware" and use_release_prediction
                and preserve_priority_corridor and not use_anticipated_delays else None
        ),
        anticipated_delay_horizon=(anticipated_delay_horizon
                                  if selection_policy == "dag_aware" and use_anticipated_delays else None),
        preserve_priority_corridor=preserve_priority_corridor,
    )
    plan["search_stats"]["planning_wall_time_s"] = time.perf_counter() - started
    return plan


def _make_release_refiner(robots,tasks,task_status,objects,navigation_cells,
                          robot_radius,grid_size,speed,participants,model="events",horizon=5.,
                          verification_enabled=True, hypothesis_workers=1):
    if model == "events":
        from .release_events import refine_release_events as refine_release_predictions
    elif model == "replay":
        from .dag_lookahead import refine_release_predictions
    elif model == "dynamic_events_verified":
        from .verified_release import refine_verified_release
        def refine_release_predictions(candidates, **kwargs):
            return refine_verified_release(candidates, **kwargs, horizon=horizon,
                                           verification_enabled=verification_enabled)
    elif model in {"dynamic_events", "dynamic_events_v1", "dynamic_events_guarded"}:
        from .dynamic_release import refine_dynamic_release_events
        def refine_release_predictions(candidates, **kwargs):
            return refine_dynamic_release_events(candidates, **kwargs, horizon=horizon,
                                                  use_release_guards=model == "dynamic_events_guarded",
                                                  hypothesis_workers=hypothesis_workers,
                                                  budget_s=5.0 if model == "dynamic_events_guarded" else .30)
    else:
        raise ValueError(f"Unknown release prediction model: {model}")
    def refine(candidates,stats,deadline,baseline_finishes,baseline_distance):
        return refine_release_predictions(candidates,robots=robots,tasks=tasks,
            task_status=task_status,objects=objects,navigation_cells=navigation_cells,
            robot_radius=robot_radius,grid_size=grid_size,speed=speed,participants=participants,
            stats=stats,search_deadline=deadline,baseline_finishes=baseline_finishes,
            baseline_distance=baseline_distance)
    return refine


def _make_dag_refiner(robots, tasks, task_status, objects, navigation_cells,
                      robot_radius, grid_size, speed, participants):
    from .dag_lookahead import refine_candidates, replay_candidate, successor_reservation_risk
    def refine(candidates, stats, search_deadline, baseline_finishes, baseline_distance):
        if not successor_reservation_risk(candidates, robots=robots, tasks=tasks,
            task_status=task_status, objects=objects, navigation_cells=navigation_cells,
            speed=speed, clearance=2*robot_radius/grid_size):
            stats["dag_lookahead"] = dict(enabled=True, status="skipped_no_successor_risk",
                                          wall_time_s=0.)
            return candidates
        remaining = (math.inf if search_deadline is None
                     else search_deadline-time.perf_counter())
        budget = min(5., max(0., remaining * .5))
        return refine_candidates(candidates, stable_key=_candidate_stable_key,
            select=lambda pool: _select_resolution_candidate(pool, "dag_aware"),
            stats=stats, budget_s=budget,
            baseline_finishes=baseline_finishes, baseline_distance=baseline_distance,
            evaluator=lambda candidate, deadline: replay_candidate(candidate,
                robots=robots, tasks=tasks, task_status=task_status, objects=objects,
                navigation_cells=navigation_cells, robot_radius=robot_radius,
                grid_size=grid_size, speed=speed, participants=participants,
                deadline=deadline))
    return refine


class StandaloneSimulation:
    def __init__(self, world_file, width=1400, height=760, speed=8.0,
                 selection_policy="dag_aware", async_planning=True,
                 initial_world=None, assignment_result=None,
                 use_dag_priority=True, incremental_dag_evaluation=True,
                 use_dag_lookahead=False, use_anticipated_delays=False,
                 anticipated_delay_horizon=5., preserve_priority_corridor=True,
                 use_release_prediction=True, release_prediction_model="events",
                 planner_workers=None, hypothesis_workers=1):
        if not isinstance(hypothesis_workers, int) or hypothesis_workers < 1:
            raise ValueError("hypothesis_workers must be a positive integer")
        self.hypothesis_workers = hypothesis_workers
        if planner_workers is not None and planner_workers < 1:
            raise ValueError("planner_workers must be positive")
        if selection_policy not in {"dag_aware", "local_distance", "local_delay", "reroute_only", "downstream_priority"}:
            raise ValueError(f"Unknown policy: {selection_policy}")
        self.selection_policy = selection_policy
        self.use_dag_priority = bool(use_dag_priority)
        self.incremental_dag_evaluation = bool(incremental_dag_evaluation)
        self.use_dag_lookahead = bool(use_dag_lookahead)
        self.use_anticipated_delays = bool(use_anticipated_delays)
        self.preserve_priority_corridor = bool(preserve_priority_corridor)
        self.use_release_prediction = bool(use_release_prediction)
        if release_prediction_model not in {"events", "replay", "dynamic_events", "dynamic_events_v1", "dynamic_events_guarded", "dynamic_events_verified"}:
            raise ValueError("Unknown release_prediction_model")
        if release_prediction_model.startswith("dynamic_events") and not 3 <= anticipated_delay_horizon <= 5:
            raise ValueError("Dynamic release horizon must be between 3 and 5 seconds")
        self.release_prediction_model = release_prediction_model
        self.anticipated_delay_horizon = float(anticipated_delay_horizon)
        if use_dag_lookahead and use_anticipated_delays:
            raise ValueError("Choose anticipated task delays OR executor replay, not both")
        if not 0 < self.anticipated_delay_horizon <= 5:
            raise ValueError("Anticipated reservation horizon must be in (0, 5] seconds")
        self.world = deepcopy(initial_world) if initial_world is not None else load_world(world_file)
        self.width = int(width)
        self.height = int(height)
        self.map_width = int(self.width * 0.70)
        self.dag_left = self.map_width
        self.steps_per_second = float(speed)
        self.show_paths = True
        self.paused = False
        self.hover_cell = None
        self.selected_cell = None
        self.margin = 28
        xs = [cell[0] for cell in self.world.free_cells]
        zs = [cell[1] for cell in self.world.free_cells]
        self.bounds = min(xs), max(xs), min(zs), max(zs)
        self.scale = min(
            (self.map_width - 2 * self.margin)
            / (self.bounds[1] - self.bounds[0] + 1),
            (self.height - 2 * self.margin) / (self.bounds[3] - self.bounds[2] + 1),
        )
        self.offset_x = (
            self.map_width
            - (self.bounds[1] - self.bounds[0] + 1) * self.scale
        ) / 2
        self.offset_y = (
            self.height - (self.bounds[3] - self.bounds[2] + 1) * self.scale
        ) / 2
        self.background = self._draw_map()
        if assignment_result is not None:
            self.assignment_result = deepcopy(assignment_result)
        elif self.world.tasks and all(
            len(task["candidate"]) == 1 for task in self.world.tasks
        ):
            # An all-singleton scene is an explicit experimental assignment,
            # including legitimate place-then-pick handoffs between robots.
            # Do not reject it through the optimizer's no-handoff search rule.
            self.assignment_result = {
                "assignment": {
                    task["id"]: task["candidate"][0]
                    for task in self.world.tasks
                },
                "makespan": math.nan,
                "total_distance": math.nan,
                "assignment_search": {
                    "fixed": True,
                    "chain_order": [[task["id"]] for task in self.world.tasks],
                    "raw_combinations": 1,
                    "chain_combinations": 1,
                    "expanded_nodes": 0,
                    "bound_pruned": 0,
                    "handoff_pruned": 0,
                },
            }
        else:
            self.assignment_result = optimize_task_assignments(
                list(self.world.tasks),
                GridDistanceOracle(self.world),
                robot_speed=self.world.grid_size * self.steps_per_second,
                service_time=None,
            )
        self.tasks = tuple(
            {
                **task,
                "robot": self.assignment_result["assignment"][task["id"]],
            }
            for task in self.world.tasks
        )
        self.robots = []
        self.objects = {}
        self.task_status = {}
        self.async_conflict_planning = async_planning
        self._planner_executor = ProcessPoolExecutor(
            max_workers=(planner_workers if planner_workers is not None
                         else max(1, len(self.world.robots))),
            mp_context=get_context("spawn"),
        )
        self._planning_generation = 0
        self.pending_conflict_plans = {}
        self._next_planning_id = 1
        self._print_assignment_summary()
        self.reset()

    def _pixel(self, cell):
        min_x, _, _, max_z = self.bounds
        return (
            int(round(self.offset_x + (cell[0] - min_x + 0.5) * self.scale)),
            int(round(self.offset_y + (max_z - cell[1] + 0.5) * self.scale)),
        )

    def _cell_from_pixel(self, pixel_x, pixel_y):
        """Convert a window pixel to the nearest JSON grid coordinate."""
        min_x, _, _, max_z = self.bounds
        cell_x = math.floor((pixel_x - self.offset_x) / self.scale) + min_x
        cell_z = max_z - math.floor((pixel_y - self.offset_y) / self.scale)
        return int(cell_x), int(cell_z)

    @staticmethod
    def _draw_dashed_path(frame, pixels, color, thickness=3, dash=8, gap=5):
        """Draw a polyline as evenly spaced anti-aliased dashes."""
        pattern = dash + gap
        travelled = 0.0
        for start, end in zip(pixels, pixels[1:]):
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length == 0.0:
                continue
            direction = vector / length
            offset = 0.0
            while offset < length:
                phase = (travelled + offset) % pattern
                step = min(length - offset, pattern - phase)
                if phase < dash:
                    draw_length = min(step, dash - phase)
                    point_a = tuple(np.rint(start + direction * offset).astype(int))
                    point_b = tuple(
                        np.rint(start + direction * (offset + draw_length)).astype(int)
                    )
                    cv2.line(
                        frame,
                        point_a,
                        point_b,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
                offset += step
            travelled += length

    def _on_mouse(self, event, pixel_x, pixel_y, _flags, _parameter):
        if pixel_x >= self.dag_left:
            self.hover_cell = None
            return
        cell = self._cell_from_pixel(pixel_x, pixel_y)
        self.hover_cell = cell
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.selected_cell = cell
        world_x = cell[0] * self.world.grid_size
        world_z = cell[1] * self.world.grid_size
        if cell in self.world.navigation_cells:
            state = "NAVIGABLE"
        elif cell in self.world.free_cells:
            state = "ROBOT_CLEARANCE"
        else:
            state = "BLOCKED"
        print(
            f"Selected map coordinate: cell=[{cell[0]}, {cell[1]}], "
            f"world_m=[{world_x:.2f}, {world_z:.2f}], state={state}"
        )

    def _draw_map(self):
        frame = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        for cell in self.world.free_cells:
            centre = self._pixel(cell)
            half = self.scale / 2
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
                half = self.scale / 2
                cv2.rectangle(
                    frame,
                    (int(centre[0] - half), int(centre[1] - half)),
                    (int(centre[0] + half), int(centre[1] + half)),
                    (220, 220, 220),
                    1,
                )
        return frame

    def reset(self):
        for pending in getattr(self, "pending_conflict_plans", {}).values():
            pending["future"].cancel()
        self._planning_generation = (
            getattr(self, "_planning_generation", 0) + 1
        )
        self.pending_conflict_plans = {}
        self._next_planning_id = 1
        self.paused = False
        self.active_resolutions = {}
        self._service_choice_cache = {}
        self._dag_lookahead_disabled = False
        self._release_prediction_disabled = False
        self._executor_verification_disabled = False
        self.service_decision_history = []
        self.resolution_history = []
        self.conflict_events = []
        self.planning_wall_time_s = 0.0
        # Simulated mission clock used only by benchmark diagnostics.
        self.simulation_time_s = 0.0
        self._next_resolution_id = 1
        self.objects = deepcopy(self.world.objects)
        for item in self.objects.values():
            item["visible"] = True
            item["is_on"] = False
        self.task_status = {task["id"]: "pending" for task in self.tasks}
        self.robots = []
        for index, definition in enumerate(self.world.robots):
            start = definition["start_cell"]
            color = tuple(definition.get("color_bgr", _DEFAULT_COLORS[index % 3]))
            self.robots.append(
                RobotState(definition["id"], start, color)
            )
        self._start_ready_tasks()

    def _print_assignment_summary(self):
        print("\n=== SHARED GLOBAL ASSIGNMENT SEARCH ===")
        stats = self.assignment_result["assignment_search"]
        if stats.get("fixed"):
            final = ", ".join(
                f"T{task_id}->{robot}"
                for task_id, robot in sorted(
                    self.assignment_result["assignment"].items()
                )
            )
            print("Fixed singleton assignment supplied by the scenario.")
            print(f"Final assignment: {final}\n")
            return
        chains = " | ".join(
            "->".join(f"T{task_id}" for task_id in chain)
            for chain in stats["chain_order"]
        )
        print(
            f"Chains: {chains}; raw={stats['raw_combinations']}, "
            f"chain-combinations={stats['chain_combinations']}, "
            f"expanded={stats['expanded_nodes']}, "
            f"bound-pruned={stats['bound_pruned']}, "
            f"handoff-pruned={stats['handoff_pruned']}"
        )
        print(
            f"Selected: makespan={self.assignment_result['makespan']:.2f}s, "
            f"distance={self.assignment_result['total_distance']:.2f}m"
        )
        final = ", ".join(
            f"T{task_id}->{robot}"
            for task_id, robot in sorted(
                self.assignment_result["assignment"].items()
            )
        )
        print(f"Final assignment: {final}")
        print("Executing the optimized assignment.\n")

    def _robot(self, robot_id):
        return next(robot for robot in self.robots if robot.robot_id == robot_id)

    @staticmethod
    def _destination_name(task):
        return task.get("target") if task["action"] == "place" else task["object"]

    def _start_ready_tasks(self):
        """Assign at most one dependency-ready task to every idle robot."""
        reserved = self._reserved_robot_ids()
        for robot in self.robots:
            if robot.phase != "idle" or robot.robot_id in reserved:
                continue
            task = next(
                (
                    task
                    for task in self.tasks
                    if task["robot"] == robot.robot_id
                    and self.task_status[task["id"]] == "pending"
                    and all(
                        self.task_status.get(dependency) == "completed"
                        for dependency in task["node"]
                    )
                ),
                None,
            )
            if task is None:
                continue
            destination = self.objects[self._destination_name(task)]
            path = preferred_path_to_object(
                self.world.navigation_cells,
                robot.cell,
                destination["cells"],
                interaction_cells=destination.get("interaction_cells"),
                robot_speed=self.steps_per_second,
                candidate_count=3,
            )
            if path is None:
                raise RuntimeError(
                    f"No A* path for {robot.robot_id} to "
                    f"{self._destination_name(task)}"
                )
            robot.current_task = task
            robot.goal = path[-1]
            robot.path = path
            robot.path_index = 0
            robot.phase = "preview"
            robot.preview_remaining = PATH_PREVIEW_SECONDS
            self.task_status[task["id"]] = "planned"
            print(
                f"Preferred path for T{task['id']} / {robot.robot_id}: "
                f"{len(path)} cells; previewing for "
                f"{PATH_PREVIEW_SECONDS:.1f}s"
            )

    def _reserved_robot_ids(self):
        active = set()
        for resolution in self.active_resolutions.values():
            participants = set(resolution["participants"])
            participants.difference_update(resolution.get("fixed_service_robots", ()))
            if resolution.get("kind") == "yield_wait":
                participants.discard(resolution.get("priority_robot"))
            active.update(participants)
        pending = {
            robot_id
            for plan in getattr(self, "pending_conflict_plans", {}).values()
            for robot_id in plan["participants"]
        }
        return active | pending

    def _overlapping_pending_plan_ids(self, robot_ids):
        requested = set(robot_ids)
        return [
            planning_id
            for planning_id, pending in getattr(
                self,
                "pending_conflict_plans",
                {},
            ).items()
            if pending["participants"] & requested
        ]

    def _cancel_pending_plans(self, planning_ids):
        participants = set()
        for planning_id in planning_ids:
            pending = self.pending_conflict_plans.pop(planning_id)
            participants.update(pending["participants"])
            pending["future"].cancel()
            self._restore_pending_phases(pending)
            print(f"Merged conflict cancels pending planner P{planning_id}")
        return participants

    def _overlapping_resolution_ids(self, robot_ids):
        return [
            resolution_id
            for resolution_id, resolution in self.active_resolutions.items()
            if resolution["participants"] & set(robot_ids)
        ]

    def _cancel_resolutions(self, resolution_ids):
        """Cancel interacting local plans and return their participant union."""
        participants = set()
        for resolution_id in resolution_ids:
            resolution = self.active_resolutions.pop(resolution_id)
            participants.update(resolution["participants"])
            for robot_id in resolution["participants"]:
                robot = self._robot(robot_id)
                robot.waiting_for = None
                robot.clearing_for = ()
                robot.clearing_started = False
                if robot.current_task is None:
                    robot.path = [robot.cell]
                    robot.path_index = 0
                    robot.goal = None
                    robot.phase = "idle"
                elif robot.phase == "acting":
                    robot.path = [robot.cell]
                    robot.path_index = 0
                else:
                    destination = self.objects[
                        self._destination_name(robot.current_task)
                    ]
                    route = preferred_path_to_object(
                        self.world.navigation_cells,
                        robot.cell,
                        destination["cells"],
                        interaction_cells=destination.get("interaction_cells"),
                        robot_speed=self.steps_per_second,
                        candidate_count=3,
                    )
                    if route is None:
                        raise RuntimeError(
                            f"Cannot restore task path for {robot.robot_id} "
                            "while merging conflict resolutions"
                        )
                    robot.path = list(route)
                    robot.path_index = 0
                    robot.goal = route[-1]
                    robot.phase = "moving"
                    self.task_status[robot.current_task["id"]] = "running"
            print(f"Merged conflict cancels local resolution C{resolution_id}")
        return participants

    def _begin_reroute_plan(self, plan, cluster_ids):
        """Execute the short-replan candidate selected by optimize()."""
        worker_id = plan["rerouted_agents"][0]
        worker = self._robot(worker_id)
        replanned = list(plan["preferred_paths"][worker_id])
        resolution_id = self._next_resolution_id
        self._next_resolution_id += 1
        participants = set(cluster_ids)
        held_phases = {
            robot_id: self._robot(robot_id).phase
            for robot_id in participants - {worker_id}
            if self._robot(robot_id).phase in {"moving", "preview"}
        }
        self.active_resolutions[resolution_id] = {
            "kind": "reroute",
            "steps": [],
            "index": 0,
            "preferred_paths": {worker.robot_id: list(replanned)},
            "participants": participants,
            "working_ids": {worker_id},
            "held_phases": held_phases,
            "original_goals": {
                robot_id: self._robot(robot_id).goal
                for robot_id in participants
            },
        }
        for robot_id in held_phases:
            blocker = self._robot(robot_id)
            blocker.phase = "waiting"
            blocker.waiting_for = f"reroute-{resolution_id}"
        if not hasattr(self, "resolution_history"):
            self.resolution_history = []
        self.resolution_history.append(
            {
                "resolution_id": resolution_id,
                "strategy": plan.get("strategy", "short_replan"),
                "intervention_type": plan.get("strategy", "short_replan"),
                "selected_candidate": plan.get("candidate_id"),
                "selected_hypothesis": plan["selected_hypothesis"],
                "estimated_makespan": plan["estimated_makespan"],
                "total_distance": plan["total_distance"],
                "additional_distance": plan.get(
                    "additional_distance",
                    plan["total_distance"],
                ),
                "total_delay": plan["total_delay"],
                "local_resolution_delay": plan.get(
                    "local_resolution_delay", 0.0
                ),
                "local_resolution_distance": plan.get(
                    "local_resolution_distance",
                    plan["total_distance"],
                ),
                "search_stats": plan.get("search_stats", {}),
            }
        )
        worker.path = list(replanned)
        worker.path_index = 0
        worker.phase = "moving"
        self.task_status[worker.current_task["id"]] = "running"
        self._print_resolution_candidates(resolution_id, plan)
        print(
            f"C{resolution_id} optimize selected short replan for "
            f"{worker_id}: makespan={plan['estimated_makespan']:.2f}s, "
            f"distance={plan['total_distance']:.2f}m, "
            f"delay={plan['total_delay']:.2f}s"
        )
        return resolution_id

    @staticmethod
    def _print_resolution_candidates(resolution_id, plan):
        """Expose local cost versus remaining-DAG cost for experiments."""
        candidates = plan.get("search_stats", {}).get(
            "candidate_evaluations",
            (),
        )
        for candidate in candidates:
            print(
                f"C{resolution_id} candidate "
                f"{candidate['hypothesis']}/{candidate['strategy']}: "
                f"local={candidate['local_resolution_distance']:.2f}m, "
                "local-delay="
                f"{candidate.get('local_resolution_delay', 0.0):.2f}s, "
                f"DAG-makespan={candidate['estimated_makespan']:.2f}s, "
                f"mission-distance={candidate['total_distance']:.2f}m, "
                "extra-distance="
                f"{candidate.get('additional_distance', candidate['total_distance']):.2f}m"
            )

    def _release_separated_reroutes(self):
        """Release reroutes after both bodies and remaining paths are clear.

        Keeping the resolution active suppresses duplicate hypotheses for the
        same encounter.  A held robot is released before the rerouted worker
        reaches its task whenever their remaining timed paths are already
        conflict-free.
        """
        clearance_cells = (
            2.0 * self.world.robot_radius / self.world.grid_size
        )
        completed = []
        for resolution_id, resolution in self.active_resolutions.items():
            if resolution.get("kind") != "reroute":
                continue
            participants = resolution["participants"]
            separated = all(
                math.dist(self._robot(worker_id).cell, self._robot(other_id).cell)
                > clearance_cells
                for worker_id in resolution["working_ids"]
                for other_id in participants - resolution["working_ids"]
            )
            remaining_paths_clear = all(
                not self._remaining_paths_conflict(
                    self._robot(worker_id),
                    self._robot(other_id),
                    clearance_cells,
                )
                for worker_id in resolution["working_ids"]
                for other_id in participants - resolution["working_ids"]
            )
            if (
                separated
                and not remaining_paths_clear
                and not resolution.get("path_hold_reported")
            ):
                resolution["path_hold_reported"] = True
                print(
                    f"C{resolution_id} bodies clear; reservation remains "
                    "active until remaining paths are conflict-free"
                )
            if separated and remaining_paths_clear:
                completed.append(resolution_id)
        for resolution_id in completed:
            resolution = self.active_resolutions.pop(resolution_id)
            for robot_id, phase in resolution.get("held_phases", {}).items():
                robot = self._robot(robot_id)
                if (
                    robot.phase == "waiting"
                    and robot.waiting_for == f"reroute-{resolution_id}"
                ):
                    robot.phase = phase
                    robot.waiting_for = None
            print(f"C{resolution_id} short replan cleared the contact cluster")
        if completed:
            self._start_ready_tasks()

    @staticmethod
    def _remaining_path(robot):
        """Return a live route beginning exactly at the robot's current cell."""
        if not robot.path:
            return [robot.cell]
        route = list(robot.path[robot.path_index:])
        if not route:
            return [robot.cell]
        if route[0] != robot.cell:
            route.insert(0, robot.cell)
        return route

    def _remaining_paths_conflict(self, first, second, clearance_cells):
        """Check overlap of full footprint-wide remaining path corridors.

        This is intentionally spatial rather than time-indexed: two robots
        are not released merely because they would reach the same narrow
        corridor at slightly different predicted times.
        """
        return _path_conflict(
            self._remaining_path(first),
            self._remaining_path(second),
            clearance_cells,
        )

    @staticmethod
    def _planning_signature(robot):
        task_id = None if robot.current_task is None else robot.current_task["id"]
        return robot.cell, task_id, robot.goal, robot.carrying

    def _submit_conflict_plan(self, cluster):
        """Freeze one contact component and plan for it off the main loop."""
        planning_id = self._next_planning_id
        self._next_planning_id += 1
        participants = {robot.robot_id for robot in cluster}
        signatures = {
            robot.robot_id: self._planning_signature(robot)
            for robot in cluster
        }
        original_phases = {
            robot.robot_id: robot.phase for robot in cluster
        }
        external_paths = self._parallel_protected_paths(participants)
        future = self._planner_executor.submit(
            _compute_conflict_plan,
            cluster=deepcopy(cluster),
            dag_robots=deepcopy(self.robots),
            tasks=deepcopy(self.tasks),
            task_status=deepcopy(self.task_status),
            objects=deepcopy(self.objects),
            navigation_cells=self.world.navigation_cells,
            robot_radius=self.world.robot_radius,
            grid_size=self.world.grid_size,
            speed_cells_per_second=self.steps_per_second,
            selection_policy=self.selection_policy,
            planning_timeout_s=getattr(self, "planning_timeout_s", None),
            use_dag_priority=getattr(self, "use_dag_priority", True),
            incremental_dag_evaluation=getattr(
                self, "incremental_dag_evaluation", True
            ),
            use_dag_lookahead=(getattr(self, "use_dag_lookahead", False)
                               and not getattr(self, "_dag_lookahead_disabled", False)),
            use_anticipated_delays=getattr(self, "use_anticipated_delays", False),
            anticipated_delay_horizon=getattr(self, "anticipated_delay_horizon", 5.),
            preserve_priority_corridor=getattr(self, "preserve_priority_corridor", True),
            use_release_prediction=(getattr(self, "use_release_prediction", True)
                                    and not getattr(self,"_release_prediction_disabled",False)),
            release_prediction_model=getattr(self, "release_prediction_model", "events"),
            hypothesis_workers=getattr(self, "hypothesis_workers", 1),
            verify_release=not getattr(self, "_executor_verification_disabled", False),
            parallel_reserved_paths={
                (rid, index): route for rid, routes in external_paths.items()
                for index, route in enumerate(routes)
            },
        )
        self.pending_conflict_plans[planning_id] = {
            "future": future,
            "participants": participants,
            "signatures": signatures,
            "original_phases": original_phases,
            "generation": self._planning_generation,
            "external_paths": external_paths,
        }
        for robot in cluster:
            if robot.current_task is not None and robot.phase != "acting":
                robot.phase = "waiting"
                robot.waiting_for = "conflict-planner"
        print(
            f"P{planning_id} planning asynchronously for "
            + ", ".join(sorted(participants))
        )
        return planning_id

    def _parallel_protected_paths(self, participants):
        """Reserve live routes and all unexecuted staging/continuation routes.

        A different component may have committed a repair while this worker
        was planning. Its future staging route matters, not just its body.
        """
        paths = {rid: [route] for rid, route in
                 _all_robot_protected_paths(self.robots, participants).items()}
        for resolution in getattr(self, "active_resolutions", {}).values():
            for rid, route in resolution.get("preferred_paths", {}).items():
                if rid not in participants:
                    paths.setdefault(rid, []).append(list(route))
            for step in resolution.get("steps", ())[resolution.get("index", 0):]:
                rid = step["robot_id"]
                if rid not in participants:
                    paths.setdefault(rid, []).append(list(step["path"]))
        return paths

    def _parallel_plan_is_current(self, plan, pending):
        """Reject newly occupied corridors without invalidating normal progress.

        Traversing an unchanged external route only shrinks its reservation;
        that must not starve an independent slow planner. New routes from
        other committed groups are checked before this result is installed.
        The ordinary executor collision checks remain authoritative.
        """
        baseline = pending.get("external_paths", {})
        current = self._parallel_protected_paths(pending["participants"])
        routes = list(plan.get("preferred_paths", {}).values())
        routes.extend(step["path"] for step in plan.get("relocations", ()))
        for rid, protected in current.items():
            def corridor(paths):
                cells = set()
                for path in paths:
                    cells.update(swept_robot_corridor(
                        path, self.world.robot_radius, self.world.robot_radius,
                        self.world.grid_size))
                return cells
            newly_reserved = corridor(protected) - corridor(baseline.get(rid, ()))
            if newly_reserved and any(
                    not _route_exits_without_reentry(route, newly_reserved)
                    for route in routes if route):
                return False
        return True

    def _restore_pending_phases(self, pending):
        for robot_id, original_phase in pending["original_phases"].items():
            robot = self._robot(robot_id)
            if robot.phase == "waiting" and robot.waiting_for == "conflict-planner":
                robot.phase = original_phase
                robot.waiting_for = None

    def _poll_conflict_plans(self):
        """Commit completed planner results on the simulation thread only."""
        pending_plans = getattr(self, "pending_conflict_plans", None)
        if not pending_plans:
            return
        completed_ids = sorted(
            planning_id
            for planning_id, pending in pending_plans.items()
            if pending["future"].done()
        )
        for planning_id in completed_ids:
            pending = pending_plans.pop(planning_id)
            if pending["generation"] != self._planning_generation:
                continue
            stale = any(
                self._planning_signature(self._robot(robot_id)) != signature
                for robot_id, signature in pending["signatures"].items()
            )
            if stale:
                self._restore_pending_phases(pending)
                print(f"P{planning_id} discarded because its snapshot is stale")
                continue
            try:
                plan = pending["future"].result()
            except RuntimeError as error:
                self._restore_pending_phases(pending)
                if any(self._robot(rid).phase == "acting" and
                       self._robot(rid).action_remaining > 0
                       for rid in pending["participants"]):
                    continue
                raise RuntimeError(
                    "Active-agent conflict has no feasible direct reroute or "
                    "BnB relocation resolution"
                ) from error
            if plan is None:
                self._restore_pending_phases(pending)
                continue
            self.planning_wall_time_s += plan.get("search_stats", {}).get("planning_wall_time_s", 0.0)
            if not self._parallel_plan_is_current(plan, pending):
                self._restore_pending_phases(pending)
                print(f"P{planning_id} discarded because another group's corridor changed")
                continue
            self._record_conflict(plan, pending["participants"])
            print(f"P{planning_id} finished; committing on simulation thread")
            # Reconstruct the live pre-planning phases before installing the
            # selected local barrier.  This lets reroute execution record and
            # later restore the held active workers exactly.
            self._restore_pending_phases(pending)
            if plan.get("strategy") == "short_replan":
                self._begin_reroute_plan(plan, pending["participants"])
            else:
                self._begin_resolution_plan(plan, pending["participants"])

    def _service_hold_estimate(self, robot):
        """Estimate consecutive service at this pose, up to the next trip.

        This is a forecast, not a promise that the route will then be clear.
        Live swept collision checks still control every movement edge.
        """
        duration = robot.action_remaining
        predicted = deepcopy(self.objects)
        done = {tid for tid, state in self.task_status.items() if state == "completed"}
        current = robot.current_task
        done.add(current["id"])
        if current["action"] == "place":
            target = predicted[current["target"]]
            predicted[current["object"]]["cells"] = frozenset({target["cell"]})
        for task in self.tasks:
            if task["robot"] != robot.robot_id or task["id"] in done:
                continue
            if not set(task.get("node", ())) <= done:
                break
            destination = predicted[self._destination_name(task)]
            route = preferred_path_to_object(
                self.world.navigation_cells, robot.cell, destination["cells"],
                interaction_cells=destination.get("interaction_cells"),
                robot_speed=self.steps_per_second, candidate_count=3,
            )
            if route is None or len(route) > 1:
                break
            duration += float(task.get("duration", 1.0))
            done.add(task["id"])
            if task["action"] == "place":
                target = predicted[task["target"]]
                predicted[task["object"]]["cells"] = frozenset({target["cell"]})
        return duration

    def _choose_service_wait_or_reroute(self, cluster):
        """Optimize a finite candidate set at service-state decision events."""
        signature = tuple(
            (r.robot_id, r.cell, r.phase,
             None if r.current_task is None else r.current_task["id"], r.goal,
             tuple(self._remaining_path(r)))
            for r in self.robots
        )
        cache = getattr(self, "_service_choice_cache", None)
        if cache is None:
            cache = self._service_choice_cache = {}
        key = tuple(sorted(r.robot_id for r in cluster))
        if cache.get(key) == signature:
            return
        cache[key] = signature
        acting = [r for r in cluster if r.phase == "acting"]
        workers = [
            r for r in cluster
            if r.phase in {"moving", "preview"} and r.goal is not None
        ]
        if not workers:
            return
        # Only the currently executing non-preemptive action is visible to
        # the local intervention model. Never inspect downstream task chains.
        hold = max(r.action_remaining for r in acting)
        clearance = 2 * self.world.robot_radius / self.world.grid_size
        candidates = []

        def evaluate(strategy, worker=None, route=None):
            delays = {r.robot_id: hold for r in workers if r is not worker}
            paths = {} if worker is None else {worker.robot_id: list(route)}
            extra = 0.0 if worker is None else max(0.0,
                path_length(route, self.world.grid_size)
                - path_length(self._remaining_path(worker), self.world.grid_size))
            candidates.append(dict(
                strategy=strategy,
                estimated_makespan=math.nan,
                total_distance=extra,
                additional_distance=extra,
                total_delay=0.0,
                local_resolution_distance=extra,
                local_resolution_delay=max(delays.values(), default=0.0),
                delay_by_robot=delays,
                preferred_paths=paths, planned_paths=paths,
                relocations=[], rerouted_agents=[] if worker is None else [worker.robot_id],
                selected_hypothesis="H(service completion)" if worker is None
                    else f"H({worker.robot_id} reroute during service)",
                priority_robot=None if worker is None else worker.robot_id,
            ))

        evaluate("wait")
        for worker in workers:
            others = [r for r in self.robots if r.robot_id != worker.robot_id]
            routes = dynamic_replan_candidates(
                self.world.navigation_cells, worker.cell, worker.goal,
                stationary_robot_centres=[r.cell for r in others],
                clearance_cells=clearance, robot_speed=self.steps_per_second,
                candidate_count=3,
            )
            for route in routes:
                if _route_clears_protected_corridors(
                    route, {r.robot_id: self._remaining_path(r) for r in others},
                    self.world.robot_radius, self.world.grid_size,
                ):
                    evaluate("service_reroute", worker, route)
        # Non-preemptive service is a feasibility constraint, not a priority rule.
        # The same selector compares all scored candidates and relocation leaves.
        plan = search_relocation_plan(
            robots=cluster, dag_robots=self.robots,
            priority_robot_ids=[r.robot_id for r in workers],
            participant_ids=key, tasks=self.tasks, task_status=self.task_status,
            objects=self.objects, navigation_cells=self.world.navigation_cells,
            robot_radius=self.world.robot_radius, grid_size=self.world.grid_size,
            speed_cells_per_second=self.steps_per_second,
            additional_candidates=candidates,
            fixed_robot_ids=[r.robot_id for r in acting],
            external_protected_paths={
                r.robot_id: self._remaining_path(r)
                for r in self.robots if r.robot_id not in key
            },
            # A forecast wait cost is not a certified feasible upper bound for
            # a complete collision-free mission: never use it for BnB pruning.
            dag_bound_pruning=False,
            selection_policy=getattr(self, "selection_policy", "dag_aware"),
            planning_timeout_s=getattr(self, "planning_timeout_s", None),
            local_time_evaluator=lambda candidate: _local_resolution_time(
                candidate, self.robots, set(key), self.world.navigation_cells,
                self.world.robot_radius, self.world.grid_size, self.steps_per_second, self.objects),
            diagnostics=getattr(self, "_service_search_diagnostics", None),
            anticipated_delay_horizon=(getattr(self,"anticipated_delay_horizon",5.)
                if self.selection_policy=="dag_aware" and getattr(self,"use_anticipated_delays",False)
                else None),
            preserve_priority_corridor=getattr(self,"preserve_priority_corridor",True),
            dag_candidate_refiner=(
                _make_dag_refiner(self.robots, self.tasks, self.task_status,
                    self.objects, self.world.navigation_cells,
                    self.world.robot_radius, self.world.grid_size,
                    self.steps_per_second, set(key))
                if self.selection_policy == "dag_aware"
                and getattr(self, "use_dag_lookahead", False)
                and not getattr(self, "_dag_lookahead_disabled", False) else
                _make_release_refiner(self.robots,self.tasks,self.task_status,
                    self.objects,self.world.navigation_cells,self.world.robot_radius,self.world.grid_size,
                    self.steps_per_second,set(key),getattr(self,"release_prediction_model","events"),
                    getattr(self,"anticipated_delay_horizon",5.),
                    verification_enabled=not getattr(self,"_executor_verification_disabled",False),
                    hypothesis_workers=getattr(self,"hypothesis_workers",1)) if self.selection_policy=="dag_aware"
                    and getattr(self,"use_release_prediction",True)
                    and not getattr(self,"_release_prediction_disabled",False)
                    and getattr(self,"preserve_priority_corridor",True)
                    and not getattr(self,"use_anticipated_delays",False)
                    and not getattr(self,"use_dag_lookahead",False) else None
            ),
            use_dag_priority=getattr(self, "use_dag_priority", True),
            incremental_dag_evaluation=getattr(
                self, "incremental_dag_evaluation", True
            ),
        )
        self._record_conflict(plan, key)
        stats = plan["search_stats"]
        audit = dict(
            participants=key, selected=plan["candidate_id"],
            strategy=plan["strategy"], service_hold_forecast_s=hold,
            fixed_service_robots=[r.robot_id for r in acting],
            optimality_scope="minimum predicted objective among evaluated candidates",
            wait_model="service completion forecast; subsequent conflicts reoptimized",
            search_stats=deepcopy(stats),
        )
        if not hasattr(self, "service_decision_history"):
            self.service_decision_history = []
        self.service_decision_history.append(audit)
        for candidate in stats["candidate_evaluations"]:
            print(
                f"Service candidate {candidate['candidate_id']}: "
                f"{candidate['strategy']} DAG={candidate['estimated_makespan']:.3f}s, "
                f"extra={candidate['additional_distance']:.3f}m, "
                f"delay={candidate['total_delay']:.3f}s"
            )
        print(f"Service optimize selected {plan['candidate_id']}")
        if plan["strategy"] == "service_reroute":
            worker = self._robot(plan["rerouted_agents"][0])
            worker.path = list(plan["preferred_paths"][worker.robot_id])
            worker.path_index = 0
            worker.movement_credit = 0.0
            worker.movement_edge = None
            worker.waiting_for = None
            worker.phase = "moving"
            self.task_status[worker.current_task["id"]] = "running"
        elif plan["strategy"] == "relocate":
            resolution_id = self._begin_resolution_plan(plan, set(key))
            if resolution_id in self.active_resolutions:
                self.active_resolutions[resolution_id]["fixed_service_robots"] = {
                    r.robot_id for r in acting
                }

    def _resolve_contact_clusters(self, clusters):
        """Start independent bounded local repairs; merge interacting plans."""
        activated = False
        for detected_cluster in clusters:
            cluster_ids = {robot.robot_id for robot in detected_cluster}
            pending_overlaps = self._overlapping_pending_plan_ids(cluster_ids)
            if pending_overlaps:
                existing_participants = set().union(
                    *(
                        self.pending_conflict_plans[value]["participants"]
                        for value in pending_overlaps
                    )
                )
                if (
                    len(pending_overlaps) == 1
                    and cluster_ids <= existing_participants
                ):
                    continue
                cluster_ids.update(
                    self._cancel_pending_plans(pending_overlaps)
                )
            overlaps = self._overlapping_resolution_ids(cluster_ids)
            if overlaps:
                existing_participants = set().union(
                    *(
                        self.active_resolutions[value]["participants"]
                        for value in overlaps
                    )
                )
                if len(overlaps) == 1 and cluster_ids <= existing_participants:
                    resolution = self.active_resolutions[overlaps[0]]
                    if resolution.get("kind") == "relocation":
                        # Expected contact while the selected mover is still
                        # executing the committed separation path.  Once that
                        # path has ended, the resolution becomes yield_wait;
                        # a later contact then represents a new horizon and
                        # must be optimized again instead of being suppressed
                        # forever by the old participant reservation.
                        continue
                cluster_ids.update(self._cancel_resolutions(overlaps))
            cluster = [self._robot(robot_id) for robot_id in sorted(cluster_ids)]
            if self.selection_policy == "reroute_only":
                plan = self._plan_snapshot(cluster)
                if plan is None:
                    raise RuntimeError("baseline_no_feasible_intervention")
                self._record_conflict(plan, cluster_ids)
                # Direct A* only: do not install a yielding/holding barrier.
                worker = self._robot(plan["rerouted_agents"][0])
                worker.path = list(plan["preferred_paths"][worker.robot_id])
                worker.path_index = 0
                worker.movement_credit = 0.0
                worker.movement_edge = None
                worker.phase = "moving"
                activated = True
                continue
            if any(
                robot.phase == "acting" and robot.action_remaining > 0
                for robot in cluster
            ):
                started = time.perf_counter()
                self._service_search_diagnostics = {}
                try:
                    self._choose_service_wait_or_reroute(cluster)
                except (RuntimeError, TimeoutError) as error:
                    error.stats = {**self._service_search_diagnostics, **getattr(error, "stats", {}),
                                   "planning_wall_time_s": time.perf_counter() - started}
                    self._record_failed_conflict(cluster, error)
                    raise
                finally:
                    self.planning_wall_time_s += time.perf_counter() - started
                # A selected reroute replaces the next edge in this same tick.
                activated = True
                continue
            workers = [
                robot
                for robot in cluster
                if robot.current_task is not None
                and robot.phase in {"preview", "moving", "waiting"}
            ]
            if not workers:
                continue
            if not getattr(self, "async_conflict_planning", False):
                plan = self._plan_snapshot(cluster)
                self._record_conflict(plan, cluster_ids)
                if plan["strategy"] == "short_replan":
                    self._begin_reroute_plan(plan, cluster_ids)
                else:
                    self._begin_resolution_plan(plan, cluster_ids)
                activated = True
                continue
            self._submit_conflict_plan(cluster)
            activated = True
        return activated

    def _plan_snapshot(self, cluster):
        started = time.perf_counter()
        diagnostics = {}
        try:
            return _compute_conflict_plan(
                cluster=cluster, dag_robots=self.robots, tasks=self.tasks,
                task_status=self.task_status, objects=self.objects,
                navigation_cells=self.world.navigation_cells,
                robot_radius=self.world.robot_radius, grid_size=self.world.grid_size,
                speed_cells_per_second=self.steps_per_second,
                selection_policy=self.selection_policy,
                planning_timeout_s=getattr(self, "planning_timeout_s", None),
                diagnostics=diagnostics,
                use_dag_priority=getattr(self, "use_dag_priority", True),
                incremental_dag_evaluation=getattr(
                    self, "incremental_dag_evaluation", True
                ),
                use_dag_lookahead=(getattr(self, "use_dag_lookahead", False)
                                   and not getattr(self, "_dag_lookahead_disabled", False)),
                use_anticipated_delays=getattr(self, "use_anticipated_delays", False),
                anticipated_delay_horizon=getattr(self, "anticipated_delay_horizon", 5.),
                preserve_priority_corridor=getattr(self, "preserve_priority_corridor", True),
                use_release_prediction=(getattr(self, "use_release_prediction", True)
                                        and not getattr(self,"_release_prediction_disabled",False)),
                release_prediction_model=getattr(self, "release_prediction_model", "events"),
                hypothesis_workers=getattr(self, "hypothesis_workers", 1),
                verify_release=not getattr(self, "_executor_verification_disabled", False),
            )
        except (RuntimeError, TimeoutError) as error:
            error.stats = {**diagnostics, **getattr(error, "stats", {}),
                           "planning_wall_time_s": time.perf_counter() - started}
            self._record_failed_conflict(cluster, error)
            raise
        finally:
            self.planning_wall_time_s += time.perf_counter() - started

    def _record_failed_conflict(self, cluster, error):
        self.conflict_events.append(dict(
            conflict_id=len(self.conflict_events) + 1,
            simulation_time_s=getattr(self, "simulation_time_s", 0.0),
            policy=self.selection_policy,
            participants=sorted(r.robot_id for r in cluster),
            selected_candidate=None, strategy="failure", relocations=0,
            failure_reason=str(error), search_stats=deepcopy(getattr(error, "stats", {})),
        ))

    def _record_conflict(self, plan, participants):
        if plan.get("search_stats", {}).get("executor_verification", {}).get("status") == "fallback":
            # Do not repeatedly spend a bounded replay budget on a snapshot
            # family requiring unsupported recourse. Keep guarded DAG scoring.
            self._executor_verification_disabled = True
        if plan.get("search_stats",{}).get("release_prediction",{}).get("status")=="fallback":
            self._release_prediction_disabled=True
        # One incomplete batch is enough to disable expensive refinement for
        # the remainder of this mission. Geometry/safety and original scoring
        # remain available; no repeated spending on unsupported snapshots.
        if plan.get("search_stats", {}).get("dag_lookahead", {}).get("status") == "fallback":
            self._dag_lookahead_disabled = True
        self.conflict_events.append(dict(
            conflict_id=len(self.conflict_events) + 1,
            simulation_time_s=getattr(self, "simulation_time_s", 0.0),
            policy=self.selection_policy, participants=sorted(participants),
            selected_candidate=plan.get("candidate_id"),
            strategy=plan["strategy"],
            relocations=len(plan.get("relocations", ())),
            search_stats=deepcopy(plan.get("search_stats", {})),
        ))

    def _begin_resolution_plan(self, plan, participant_ids):
        """Freeze affected workers and start an ordered relocation plan."""
        steps = list(plan["relocations"])
        if not steps:
            return None
        participants = set(participant_ids)
        if self._overlapping_resolution_ids(participants):
            raise RuntimeError("A robot cannot belong to two active resolutions")
        resolution_id = self._next_resolution_id
        self._next_resolution_id += 1
        self.active_resolutions[resolution_id] = {
            "kind": "relocation",
            "steps": steps,
            "index": 0,
            "preferred_paths": plan["preferred_paths"],
            "participants": participants,
            "priority_robot": plan.get("priority_robot"),
            # A joint candidate first moves the priority robot to create
            # space for a boxed-in blocker.  Its planned continuation is
            # valid only after every ordered staging step has completed, so
            # the ordinary one-sided look-ahead release is not applicable.
            "joint_staging": any(
                step["robot_id"] == plan.get("priority_robot")
                for step in steps
            ),
            "original_goals": {
                robot_id: self._robot(robot_id).goal
                for robot_id in participants
            },
        }
        if not hasattr(self, "resolution_history"):
            self.resolution_history = []
        self.resolution_history.append(
            {
                "resolution_id": resolution_id,
                "strategy": plan.get("strategy", "relocate"),
                "intervention_type": plan.get("strategy", "relocate"),
                "selected_candidate": plan.get("candidate_id"),
                "selected_hypothesis": plan["selected_hypothesis"],
                "estimated_makespan": plan["estimated_makespan"],
                "total_distance": plan["total_distance"],
                "additional_distance": plan.get(
                    "additional_distance",
                    plan["total_distance"],
                ),
                "total_delay": plan["total_delay"],
                "local_resolution_delay": plan.get(
                    "local_resolution_delay", 0.0
                ),
                "local_resolution_distance": plan.get(
                    "local_resolution_distance",
                    plan["total_distance"],
                ),
                "search_stats": plan["search_stats"],
            }
        )
        for robot_id in participants:
            robot = self._robot(robot_id)
            if robot.current_task is not None and robot.phase != "acting":
                robot.phase = "waiting"
                robot.waiting_for = None
        stats = plan["search_stats"]
        sequence = " -> ".join(
            f"{step['robot_id']}@{list(step['destination'])}" for step in steps
        )
        print(
            f"C{resolution_id} BnB resolution: "
            f"roots={stats['roots']}, expanded={stats['expanded']}, "
            f"bound-pruned={stats['bound_pruned']}, "
            f"feasible={stats['feasible']}"
        )
        self._print_resolution_candidates(resolution_id, plan)
        print(
            f"optimize -> makespan={plan['estimated_makespan']:.2f}s, "
            f"distance={plan['total_distance']:.2f}m, "
            "extra="
            f"{plan.get('additional_distance', plan['total_distance']):.2f}m, "
            f"delay={plan['total_delay']:.2f}s"
        )
        print(f"C{resolution_id} selected: {plan['selected_hypothesis']}")
        print(f"C{resolution_id} relocation plan: {sequence}")
        self._start_next_resolution_step(resolution_id)
        return resolution_id

    def _start_next_resolution_step(self, resolution_id):
        resolution = self.active_resolutions.get(resolution_id)
        if resolution is None:
            return
        index = resolution["index"]
        if index >= len(resolution["steps"]):
            self._finish_resolution_plan(resolution_id)
            return
        step = resolution["steps"][index]
        mover = self._robot(step["robot_id"])
        route = list(step["path"])
        if route[0] != mover.cell:
            raise RuntimeError(
                f"Resolution path for {mover.robot_id} starts at stale position"
            )
        mover.path = route
        mover.path_index = 0
        mover.goal = route[-1]
        if mover.phase == "acting":
            mover.interrupted_action_remaining = mover.action_remaining
        mover.phase = "parking"
        mover.clearing_for = tuple(step["protects"])
        mover.clearing_started = False
        print(
            f"C{resolution_id} relocation "
            f"{index + 1}/{len(resolution['steps'])}: "
            f"{mover.robot_id} follows {len(route)} A* cells to "
            f"{list(mover.goal)}"
        )

    def _complete_resolution_step(self, resolution_id, mover):
        resolution = self.active_resolutions[resolution_id]
        mover.path = [mover.cell]
        mover.path_index = 0
        mover.goal = None
        mover.phase = "waiting" if mover.current_task is not None else "idle"
        mover.waiting_for = None
        mover.clearing_for = ()
        mover.clearing_started = False
        resolution["index"] += 1
        self._start_next_resolution_step(resolution_id)

    def _parking_resolution_ids(self, robot_id):
        """Return resolutions whose live relocation step belongs to a robot.

        Completed relocation plans can remain active as ``yield_wait``
        reservations.  Their index is intentionally one past the final step,
        so they must not be treated as if they still had a current mover.
        """
        matches = []
        for resolution_id, resolution in self.active_resolutions.items():
            steps = resolution.get("steps", ())
            index = resolution.get("index", 0)
            if (
                0 <= index < len(steps)
                and steps[index].get("robot_id") == robot_id
            ):
                matches.append(resolution_id)
        return matches

    def _recover_orphaned_parking(self, robot):
        """Restore a mover whose resolution was cancelled on its arrival tick."""
        robot.path = [robot.cell]
        robot.path_index = 0
        robot.goal = None
        robot.waiting_for = None
        robot.clearing_for = ()
        robot.clearing_started = False
        if robot.current_task is None:
            robot.phase = "idle"
        else:
            destination = self.objects[
                self._destination_name(robot.current_task)
            ]
            route = preferred_path_to_object(
                self.world.navigation_cells,
                robot.cell,
                destination["cells"],
                interaction_cells=destination.get("interaction_cells"),
                robot_speed=self.steps_per_second,
                candidate_count=3,
            )
            if route is None:
                raise RuntimeError(
                    "Cannot restore task path after cancelled parking motion "
                    f"for {robot.robot_id}"
                )
            robot.path = list(route)
            robot.path_index = 0
            robot.goal = route[-1]
            robot.phase = "moving"
            self.task_status[robot.current_task["id"]] = "running"
        print(
            f"Recovered {robot.robot_id} at the endpoint of a cancelled "
            "parking motion"
        )

    def _finish_resolution_plan(self, resolution_id):
        resolution = self.active_resolutions[resolution_id]
        paths = resolution["preferred_paths"]
        priority_robot = resolution.get("priority_robot")
        for robot_id, planned_path in paths.items():
            if priority_robot is not None and robot_id != priority_robot:
                continue
            if (
                robot_id == priority_robot
                and resolution.get("priority_released_early")
            ):
                # Look-ahead already installed this route and the robot may
                # have advanced well beyond its original path start.
                continue
            robot = self._robot(robot_id)
            if robot.current_task is None or robot.phase == "acting":
                continue
            route = list(planned_path)
            if not route or route[0] != robot.cell:
                raise RuntimeError(
                    f"Final preferred path for {robot_id} has a stale start"
                )
            robot.path = route
            robot.path_index = 0
            robot.goal = route[-1]
            robot.waiting_for = None
            robot.phase = "moving"
            self.task_status[robot.current_task["id"]] = "running"
            if len(route) == 1:
                self._begin_action(robot)
        if priority_robot is not None:
            # Every yielder has reached its clearance pose. Generate a fresh
            # continuation from that actual pose to its live task goal; this
            # is the path corridor used by the release reservation below.
            for robot_id in resolution["participants"] - {priority_robot}:
                robot = self._robot(robot_id)
                if robot.current_task is None or robot.phase == "acting":
                    continue
                destination = self.objects[
                    self._destination_name(robot.current_task)
                ]
                continuation = preferred_path_to_object(
                    self.world.navigation_cells,
                    robot.cell,
                    destination["cells"],
                    interaction_cells=destination.get("interaction_cells"),
                    robot_speed=self.steps_per_second,
                    candidate_count=3,
                )
                if continuation is None:
                    raise RuntimeError(
                        f"No continuation from clearance pose for {robot_id}"
                    )
                paths[robot_id] = list(continuation)
            resolution["kind"] = "yield_wait"
            print(
                f"C{resolution_id} released {priority_robot}; yielding robots "
                "resume when their new path corridors are body-clear"
            )
            self._release_completed_priority_yields()
            return
        self.active_resolutions.pop(resolution_id)
        for robot_id in resolution["participants"]:
            robot = self._robot(robot_id)
            if robot.current_task is None and robot.phase == "waiting":
                robot.phase = "idle"
        print(
            f"C{resolution_id} complete; conflict-free preferred paths released"
        )
        self._start_ready_tasks()

    @staticmethod
    def _distance_limited_path(path, distance_limit):
        """Return a discrete route prefix spanning at least distance_limit."""
        return distance_prefix(list(path), distance_limit)

    def _release_priority_by_lookahead(self):
        """Release a priority robot while its yielder is still relocating.

        The resolution remains active, so the same contact cannot generate a
        second hypothesis. Release requires body separation plus a timed,
        swept-motion-safe horizon for the priority route and the active
        relocation route. Stationary participants must also be outside that
        look-ahead corridor.
        """
        clearance_cells = (
            2.0 * self.world.robot_radius / self.world.grid_size
        )
        lookahead_cells = max(
            2.0 * clearance_cells,
            self.steps_per_second,
        )
        for resolution_id, resolution in self.active_resolutions.items():
            if (
                resolution.get("kind") != "relocation"
                or resolution.get("priority_released_early")
                or resolution.get("joint_staging")
            ):
                continue
            priority_id = resolution.get("priority_robot")
            if priority_id is None:
                continue
            index = resolution["index"]
            if index >= len(resolution["steps"]):
                continue
            mover_id = resolution["steps"][index]["robot_id"]
            if mover_id == priority_id:
                continue
            mover = self._robot(mover_id)
            priority = self._robot(priority_id)
            if (
                mover.phase != "parking"
                or priority.current_task is None
                or priority.phase == "acting"
            ):
                continue
            separation = math.dist(mover.cell, priority.cell)
            if separation <= clearance_cells + 1e-9:
                continue

            priority_route = list(
                resolution["preferred_paths"].get(priority_id, ())
            )
            if not priority_route or priority_route[0] != priority.cell:
                continue
            mover_route = self._remaining_path(mover)
            if not priority_motion_is_clear(
                priority_route, mover_route,
                [self._robot(robot_id).cell for robot_id in resolution['participants']
                 if robot_id not in {priority_id, mover_id}],
                clearance_cells, self.steps_per_second):
                continue

            priority.path = priority_route
            priority.path_index = 0
            priority.goal = priority_route[-1]
            priority.waiting_for = None
            priority.phase = "moving"
            self.task_status[priority.current_task["id"]] = "running"
            resolution["priority_released_early"] = True
            resolution["lookahead_cells"] = lookahead_cells
            print(
                f"C{resolution_id} look-ahead released {priority_id} while "
                f"{mover_id} is still relocating: separation="
                f"{separation:.2f} cells, horizon={lookahead_cells:.2f} cells"
            )
            if len(priority_route) == 1:
                self._begin_action(priority)

    def _release_completed_priority_yields(self):
        """Replan and resume parked yielders from the observed live state.

        Candidate generation happens only at real/predicted contact. After a
        relocation, release one yielder at a time using dynamic A* around the
        robot bodies that actually remain.  Removing a released robot from the
        reservation lets it execute normally while later yielders stay parked.
        This is the receding-horizon half of park-pass-resume: a route that was
        blocked in the original snapshot is not treated as permanently
        impossible, and a stale static continuation is never forced through a
        robot that stopped at its goal.
        """
        clearance_cells = (
            2.0 * self.world.robot_radius / self.world.grid_size
        )
        def cached_route(key, compute):
            # Scoped to one immutable counterfactual batch, never the live
            # world. Cache every geometric input, including moved objects.
            cache=getattr(self,"_release_prediction_route_cache",None)
            if cache is None:
                return compute()
            if key not in cache:
                route=compute()
                cache[key]=None if route is None else tuple(route)
            return None if cache[key] is None else list(cache[key])
        completed = []
        for resolution_id, resolution in self.active_resolutions.items():
            if resolution.get("kind") != "yield_wait":
                continue
            priority_id = resolution["priority_robot"]
            priority = self._robot(priority_id)
            yielding_ids = sorted(
                resolution["participants"] - {priority_id}
            )
            separated = all(
                math.dist(priority.cell, self._robot(robot_id).cell)
                > clearance_cells
                for robot_id in yielding_ids
            )
            if priority.phase == "parking" or not separated:
                continue

            released = False
            for robot_id in yielding_ids:
                robot = self._robot(robot_id)
                if robot.current_task is None:
                    resolution["participants"].discard(robot_id)
                    continue
                if robot.phase == "acting":
                    resolution["participants"].discard(robot_id)
                    continue

                destination = self.objects[
                    self._destination_name(robot.current_task)
                ]
                original_goal = resolution.get("original_goals", {}).get(
                    robot_id
                )
                other_centres = tuple(
                    other.cell
                    for other in self.robots
                    if other.robot_id != robot_id
                )
                route, nominal, _ = choose_continuation_route(
                    self.world.navigation_cells, robot.cell, destination,
                    original_goal, other_centres, clearance_cells,
                    self.steps_per_second, cached_route)
                if route is None:
                    continue
                if nominal is not None and priority.current_task is not None:
                    priority_remaining = self._remaining_path(priority)
                    priority_travel = path_length(
                        priority_remaining
                    ) / self.steps_per_second
                    priority_service = (
                        priority.action_remaining
                        if priority.phase == "acting"
                        else float(priority.current_task.get("duration", 0.0))
                    )
                    reroute_now = path_length(route) / self.steps_per_second
                    wait_then_nominal = (
                        priority_travel
                        + priority_service
                        + path_length(nominal) / self.steps_per_second
                    )
                    if wait_for_nominal_is_better(
                        priority_remaining, priority_service, nominal, route,
                        robot.cell, clearance_cells, self.steps_per_second):
                        decision_key = f"wait-{robot_id}"
                        if decision_key not in resolution:
                            resolution[decision_key] = True
                            print(
                                f"C{resolution_id} continuation optimize keeps "
                                f"{robot_id} parked: wait+nominal="
                                f"{wait_then_nominal:.2f}s < reroute-now="
                                f"{reroute_now:.2f}s"
                            )
                        continue
                elif (
                    nominal is not None
                    and priority.current_task is None
                    and _path_conflict(
                        nominal,
                        [priority.cell],
                        clearance_cells,
                    )
                ):
                    # The former priority is now an idle blocker. Follow the
                    # shorter guarded nominal route until the swept detector
                    # opens a new optimization in which that idle robot can be
                    # relocated, instead of committing to a global detour.
                    route = nominal
                if self._is_navigating(priority):
                    if not continuation_motion_is_clear(
                        self._remaining_path(priority), route,
                        clearance_cells, self.steps_per_second):
                        # Keep the holding pose until the two observed motion
                        # prefixes admit a swept-safe simultaneous release.
                        continue
                resolution["preferred_paths"][robot_id] = list(route)
                robot.path = route
                robot.path_index = 0
                robot.goal = route[-1]
                robot.waiting_for = None
                robot.phase = "moving"
                self.task_status[robot.current_task["id"]] = "running"
                resolution["participants"].discard(robot_id)
                released = True
                print(
                    f"C{resolution_id} replanned and released {robot_id} "
                    f"from its holding pose ({len(route)} A* cells)"
                )
                if len(route) == 1:
                    self._begin_action(robot)
                # Keep the other yielders parked. Their paths are regenerated
                # from the next observed state instead of being committed as
                # simultaneous static corridors.
                break

            if resolution["participants"] <= {priority_id}:
                completed.append(resolution_id)
            elif not released and not resolution.get("path_hold_reported"):
                resolution["path_hold_reported"] = True
                print(
                    f"C{resolution_id} bodies clear; no body-safe yielder "
                    "continuation yet, keeping the holding reservation"
                )
        for resolution_id in completed:
            self.active_resolutions.pop(resolution_id)
            print(
                f"C{resolution_id} complete; parked continuations released"
            )
        if completed:
            self._start_ready_tasks()

    def _begin_action(self, robot):
        task = robot.current_task
        robot.phase = "acting"
        robot.action_remaining = (
            robot.interrupted_action_remaining
            if robot.interrupted_action_remaining is not None
            else float(task["duration"])
        )
        robot.interrupted_action_remaining = None
        self.task_status[task["id"]] = "acting"
        print(
            f"{robot.robot_id} begins {task['action']} for "
            f"{task['duration']:.1f}s"
        )

    def _finish_action(self, robot):
        task = robot.current_task
        action = task["action"]
        object_name = task["object"]
        if action == "pick":
            if robot.carrying is not None:
                raise RuntimeError(f"{robot.robot_id} is already carrying an object")
            self.objects[object_name]["visible"] = False
            for contained in self.objects.values():
                if contained.get("contained_by") == object_name:
                    contained["visible"] = False
            robot.carrying = object_name
        elif action == "place":
            if robot.carrying != object_name:
                raise RuntimeError(
                    f"{robot.robot_id} cannot place {object_name}; "
                    f"it carries {robot.carrying}"
                )
            target = self.objects[task["target"]]
            moved = self.objects[object_name]
            moved["cell"] = target["cell"]
            moved["cells"] = frozenset({target["cell"]})
            moved["visible"] = True
            moved["contained_by"] = task["target"]
            for contained in self.objects.values():
                if contained.get("contained_by") == object_name:
                    contained["cell"] = target["cell"]
                    contained["cells"] = frozenset({target["cell"]})
                    contained["visible"] = True
            robot.carrying = None
        elif action == "open":
            self.objects[object_name]["is_open"] = True
        elif action == "close":
            self.objects[object_name]["is_open"] = False
        elif action == "toggle_on":
            self.objects[object_name]["is_on"] = True
        elif action == "toggle_off":
            self.objects[object_name]["is_on"] = False
        elif action == "wash":
            self.objects[object_name]["washed"] = True

        self.task_status[task["id"]] = "completed"
        print(
            f"Completed T{task['id']}: {robot.robot_id} "
            f"{action} {object_name}"
        )
        robot.phase = "idle"
        robot.current_task = None
        robot.action_remaining = 0.0
        robot.goal = None
        robot.path = [robot.cell]
        robot.path_index = 0
        self._start_ready_tasks()

    def _update_actions(self, elapsed, eligible_actions=None):
        reserved = self._reserved_robot_ids()
        newly_acting = set()
        for robot in self.robots:
            if robot.phase != "preview":
                continue
            robot.preview_remaining = max(
                0.0,
                robot.preview_remaining - elapsed,
            )
            if robot.preview_remaining > 0.0:
                continue
            robot.phase = "moving"
            self.task_status[robot.current_task["id"]] = "running"
            print(
                f"Started T{robot.current_task['id']}: {robot.robot_id} "
                f"{robot.current_task['action']} "
                f"{robot.current_task['object']}"
            )
            if len(robot.path) == 1:
                self._begin_action(robot)
                newly_acting.add(robot.robot_id)

        completed = []
        for robot in self.robots:
            if (
                robot.phase != "acting"
                or robot.robot_id in newly_acting
                or robot.robot_id in reserved
                or eligible_actions is not None and robot.robot_id not in eligible_actions
            ):
                continue
            robot.action_remaining = max(0.0, robot.action_remaining - elapsed)
            if robot.action_remaining == 0.0:
                completed.append(robot)
        for robot in completed:
            self._finish_action(robot)

    @staticmethod
    def _is_navigating(robot):
        return robot.phase in {"moving", "parking"}

    def next_event_duration(self):
        """Seconds until the next action/metric edge completion, capped for UI."""
        self._update_actions(0.0)
        durations = [1.0 / self.steps_per_second]
        for robot in self.robots:
            if self._is_navigating(robot):
                target = robot.path[min(robot.path_index + 1, len(robot.path)-1)]
                edge = (robot.cell, target)
                credit = robot.movement_credit if robot.movement_edge == edge else 0.0
                durations.append(max(1e-9, (math.dist(*edge)-credit) / self.steps_per_second))
            elif robot.phase == "acting" and robot.robot_id not in self._reserved_robot_ids():
                durations.append(max(1e-9, robot.action_remaining))
        return min(durations)

    def step(self, elapsed):
        """Advance concurrent motion and pre-existing actions by elapsed seconds."""
        self.simulation_time_s = (
            getattr(self, "simulation_time_s", 0.0) + elapsed
        )
        self._update_actions(0.0)
        acting = {r.robot_id for r in self.robots if r.phase == "acting"}
        self._advance(elapsed)
        self._update_actions(elapsed, eligible_actions=acting)

    def _advance(self, elapsed=None):
        if elapsed is None:
            elapsed = 1.0 / self.steps_per_second
        self._poll_conflict_plans()
        self._release_separated_reroutes()
        self._release_priority_by_lookahead()
        self._release_completed_priority_yields()
        proposals = {
            robot.robot_id: robot.path[min(robot.path_index + 1, len(robot.path) - 1)]
            for robot in self.robots
            if self._is_navigating(robot)
        }
        contact_clusters = check_dynamic_robot_collisions(
            self.robots,
            self.world.robot_radius,
            self.world.grid_size,
            proposals=proposals,
        )
        if self._resolve_contact_clusters(contact_clusters):
            proposals = {
                robot.robot_id: robot.path[
                    min(robot.path_index + 1, len(robot.path) - 1)
                ]
                for robot in self.robots
                if self._is_navigating(robot)
            }
        allowed = set(proposals)
        for robot in self.robots:
            if robot.robot_id not in allowed:
                continue
            edge = (robot.cell, proposals[robot.robot_id])
            if robot.movement_edge != edge:
                robot.movement_edge = edge
                robot.movement_credit = 0.0
            robot.movement_credit += elapsed * self.steps_per_second
            if robot.movement_credit + 1e-9 < math.dist(*edge):
                allowed.remove(robot.robot_id)

        # Removing a move changes the trajectory seen by other robots; repeat
        # until every accepted simultaneous segment respects body clearance.
        clearance = 2.0 * self.world.robot_radius / self.world.grid_size
        while True:
            blocked = set()
            for index, first in enumerate(self.robots):
                for second in self.robots[index + 1:]:
                    a = proposals[first.robot_id] if first.robot_id in allowed else first.cell
                    b = proposals[second.robot_id] if second.robot_id in allowed else second.cell
                    minimum = swept_separation(first.cell, a, second.cell, b)
                    initial = math.dist(first.cell, second.cell)
                    # Legacy worlds may start touching: permit separation only.
                    safe = minimum >= clearance - 1e-9
                    if initial < clearance:
                        safe = minimum >= initial - 1e-9 and math.dist(a, b) > initial + 1e-9
                    if not safe:
                        blocked.update({first.robot_id, second.robot_id} & allowed)
            if not blocked:
                break
            for robot in self.robots:
                if robot.robot_id in blocked:
                    robot.movement_credit = 0.0
            allowed.difference_update(blocked)

        arrived = []
        for robot in self.robots:
            if robot.robot_id not in allowed or not self._is_navigating(robot):
                continue
            next_index = min(robot.path_index + 1, len(robot.path) - 1)
            next_cell = robot.path[next_index]
            self._commit_navigation_edge(robot, next_cell, next_index)
            if robot.cell == robot.goal:
                arrived.append(robot)
        # Action completion can install another robot's path. Dispatch only
        # after all movement from this tick has committed.
        for robot in arrived:
            if robot.phase == "parking":
                matches = self._parking_resolution_ids(robot.robot_id)
                if not matches:
                    # An asynchronous plan can be merged/cancelled in the
                    # same tick that its mover reaches the committed parking
                    # endpoint.  The physical move is already safe and must
                    # not be rolled back; restore the robot from the observed
                    # endpoint and let the normal detector replan next tick.
                    self._recover_orphaned_parking(robot)
                    continue
                if len(matches) != 1:
                    raise RuntimeError("Robot belongs to multiple active resolutions")
                self._complete_resolution_step(matches[0], robot)
            else:
                self._begin_action(robot)
        self._release_separated_reroutes()

    def _commit_navigation_edge(self, robot, next_cell, next_index):
        """Commit an already safety-checked edge (also used by fixed-plan execution)."""
        robot.travelled += math.dist(robot.cell, next_cell) * self.world.grid_size
        robot.movement_credit = 0.0
        robot.movement_edge = None
        robot.cell = next_cell
        robot.path_index = next_index

    def _draw(self):
        frame = self.background.copy()
        visible_objects = sorted(
            self.objects.values(),
            key=lambda value: bool(value.get("contained_by")),
        )
        contained_counts = {}
        for item in visible_objects:
            if not item.get("visible", True):
                continue
            color = tuple(item.get("color_bgr", (80, 180, 230)))
            if item.get("shape", "point") == "segment":
                start = self._pixel(item["start_cell"])
                end = self._pixel(item["end_cell"])
                thickness = max(
                    3,
                    int((2 * item.get("thickness_cells", 0) + 1) * self.scale),
                )
                cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
                centre = self._pixel(item["cell"])
                radius = thickness // 2
            else:
                display_cell = item["cell"]
                container = item.get("contained_by")
                if container:
                    index = contained_counts.get(container, 0)
                    contained_counts[container] = index + 1
                    display_cell = (
                        display_cell[0] + (-1 if index % 2 == 0 else 1),
                        display_cell[1] + index // 2,
                    )
                centre = self._pixel(display_cell)
                radius = max(5, int(item.get("radius_cells", 1) * self.scale))
                cv2.circle(frame, centre, radius, color, -1, cv2.LINE_AA)
            state_suffix = (
                " (OPEN)"
                if item.get("is_open")
                else " (ON)" if item.get("is_on") else ""
            )
            cv2.putText(
                frame,
                item["name"] + state_suffix,
                (centre[0] + radius + 3, centre[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                item["name"] + state_suffix,
                (centre[0] + radius + 3, centre[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        radius = max(
            7,
            int(self.world.robot_radius / self.world.grid_size * self.scale),
        )
        for robot in self.robots:
            if (
                self.show_paths
                and (self._is_navigating(robot) or robot.phase == "preview")
                and robot.path_index < len(robot.path) - 1
            ):
                pixels = np.array(
                    [self._pixel(cell) for cell in robot.path[robot.path_index:]],
                    dtype=np.int32,
                )
                if robot.phase == "preview":
                    self._draw_dashed_path(frame, pixels, robot.color)
                else:
                    cv2.polylines(
                        frame,
                        [pixels],
                        False,
                        robot.color,
                        2,
                        cv2.LINE_AA,
                    )
            centre = self._pixel(robot.cell)
            cv2.circle(frame, centre, radius, robot.color, -1, cv2.LINE_AA)
            cv2.circle(frame, centre, radius, (10, 10, 10), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                robot.robot_id,
                (centre[0] + radius + 4, centre[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )
            if robot.phase == "acting":
                cv2.putText(
                    frame,
                    f"{robot.current_task['action']} {robot.action_remaining:.1f}s",
                    (centre[0] + radius + 4, centre[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            elif robot.phase == "preview":
                cv2.putText(
                    frame,
                    f"preferred path {robot.preview_remaining:.1f}s",
                    (centre[0] + radius + 4, centre[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        self._draw_dag(frame)
        inspected_cell = self.selected_cell or self.hover_cell
        if inspected_cell is not None:
            centre = self._pixel(inspected_cell)
            if inspected_cell in self.world.navigation_cells:
                state = "NAVIGABLE"
            elif inspected_cell in self.world.free_cells:
                state = "ROBOT_CLEARANCE"
            else:
                state = "BLOCKED"
            world_x = inspected_cell[0] * self.world.grid_size
            world_z = inspected_cell[1] * self.world.grid_size
            cv2.drawMarker(
                frame,
                centre,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                max(12, int(self.scale * 2)),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                frame,
                (14, self.height - 42),
                (self.map_width - 14, self.height - 10),
                (15, 15, 15),
                -1,
            )
            cv2.putText(
                frame,
                (
                    f"cell=[{inspected_cell[0]}, {inspected_cell[1]}]  "
                    f"world=[{world_x:.2f}, {world_z:.2f}] m  {state}"
                ),
                (24, self.height - 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        status = "PAUSED" if self.paused else "RUNNING"
        cv2.putText(
            frame,
            f"{status} | Click: pin coordinate | Space: pause | R: reset | "
            "P: paths | Q/Esc: quit",
            (20, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _task_depths(self):
        tasks = {task["id"]: task for task in self.tasks}
        depths = {}

        def depth(task_id):
            if task_id in depths:
                return depths[task_id]
            dependencies = tasks[task_id]["node"]
            depths[task_id] = (
                max((depth(predecessor) for predecessor in dependencies), default=-1)
                + 1
            )
            return depths[task_id]

        for task_id in tasks:
            depth(task_id)
        return depths

    def _draw_dag(self, frame):
        """Draw the live dependency graph beside the occupancy map."""
        cv2.rectangle(
            frame,
            (self.dag_left, 0),
            (self.width - 1, self.height - 1),
            (24, 24, 24),
            -1,
        )
        cv2.line(
            frame,
            (self.dag_left, 0),
            (self.dag_left, self.height),
            (90, 90, 90),
            2,
        )
        cv2.putText(
            frame,
            "LIVE TASK DAG",
            (self.dag_left + 18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if not self.tasks:
            return

        depths = self._task_depths()
        robot_ids = [robot.robot_id for robot in self.robots]
        lane_width = (self.width - self.dag_left - 24) / max(1, len(robot_ids))
        max_depth = max(depths.values(), default=0)
        level_height = min(88.0, (self.height - 100) / max(1, max_depth + 1))
        centres = {}
        for task in self.tasks:
            lane = robot_ids.index(task["robot"])
            centres[task["id"]] = (
                int(self.dag_left + 12 + (lane + 0.5) * lane_width),
                int(70 + (depths[task["id"]] + 0.5) * level_height),
            )

        for task in self.tasks:
            end = centres[task["id"]]
            for predecessor in task["node"]:
                start = centres[predecessor]
                cv2.arrowedLine(
                    frame,
                    (start[0], start[1] + 22),
                    (end[0], end[1] - 22),
                    (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.12,
                )

        status_colors = {
            "pending": (75, 75, 75),
            "planned": (160, 80, 180),
            "running": (210, 120, 20),
            "acting": (20, 150, 240),
            "completed": (55, 170, 70),
        }
        node_half_width = max(42, int(lane_width / 2 - 7))
        for task in self.tasks:
            centre = centres[task["id"]]
            status = self.task_status.get(task["id"], "pending")
            color = status_colors.get(status, (75, 75, 75))
            cv2.rectangle(
                frame,
                (centre[0] - node_half_width, centre[1] - 21),
                (centre[0] + node_half_width, centre[1] + 21),
                color,
                -1,
            )
            cv2.rectangle(
                frame,
                (centre[0] - node_half_width, centre[1] - 21),
                (centre[0] + node_half_width, centre[1] + 21),
                (225, 225, 225),
                1,
            )
            cv2.putText(
                frame,
                f"T{task['id']} {task['robot']}",
                (centre[0] - node_half_width + 5, centre[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{task['action']} {task['object']}",
                (centre[0] - node_half_width + 5, centre[1] + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, self.width, self.height)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        last_step = time.monotonic()
        last_frame = last_step
        try:
            while True:
                now = time.monotonic()
                elapsed = now - last_frame
                last_frame = now
                if not self.paused:
                    remaining = elapsed
                    while remaining > 1e-9:
                        dt = min(remaining, self.next_event_duration())
                        self.step(dt)
                        remaining -= dt
                cv2.imshow(WINDOW_NAME, self._draw())
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return
                if key == ord(" "):
                    self.paused = not self.paused
                elif key == ord("r"):
                    self.reset()
                elif key == ord("p"):
                    self.show_paths = not self.show_paths
        finally:
            self.close()
            cv2.destroyWindow(WINDOW_NAME)

    def close(self):
        """Stop background planning workers without committing late results."""
        if getattr(self, "hypothesis_workers", 1) > 1:
            from .dynamic_release import close_hypothesis_pools
            close_hypothesis_pools()
        for pending in getattr(self, "pending_conflict_plans", {}).values():
            pending["future"].cancel()
        self.pending_conflict_plans = {}
        executor = getattr(self, "_planner_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._planner_executor = None


def _local_resolution_time(candidate, robots, participants, navigation_cells,
                           robot_radius, grid_size, speed, objects=None):
    """Replay only current motions/services with the actual release/guard code.

    No task graph, successor dispatch, object semantics, slack or downstream
    goals enter this model. Unknown future departures cannot be credited: a
    candidate that needs them has infinite (uncertified) local release time.
    Reroutes end their intervention at rejoining the original route suffix.
    """
    sim = StandaloneSimulation.__new__(StandaloneSimulation)
    sim.world = SimpleNamespace(navigation_cells=navigation_cells,
        grid_size=grid_size, robot_radius=robot_radius)
    sim.steps_per_second = speed
    sim.robots = deepcopy(robots)
    sim.objects = {}
    sim.tasks = ()
    sim.task_status = {}
    sim.pending_conflict_plans = {}
    sim.active_resolutions = {}
    sim.resolution_history = []
    sim._next_resolution_id = 1
    sim._start_ready_tasks = lambda: None
    sim._resolve_contact_clusters = lambda clusters: False
    sim._poll_conflict_plans = lambda: None
    for robot in sim.robots:
        if robot.current_task is not None:
            duration = float(robot.current_task.get("duration", 0.0))
            destination = StandaloneSimulation._destination_name(robot.current_task)
            robot.current_task = dict(id=robot.robot_id, action="toggle_on",
                object=robot.robot_id, duration=duration, node=[])
            sim.task_status[robot.robot_id] = "running"
            goal = robot.goal or robot.cell
            sim.objects[robot.robot_id] = deepcopy(objects[destination]) if objects is not None else dict(
                cell=goal, cells=frozenset({goal}), interaction_cells=[goal])
    plan = deepcopy(candidate)
    plan["search_stats"] = dict(roots=0, expanded=0, bound_pruned=0, feasible=1)
    rejoin = {}
    for rid in plan.get("rerouted_agents", ()):
        route = plan["preferred_paths"][rid]
        nominal = _snapshot_remaining_path(next(r for r in robots if r.robot_id == rid))
        suffix = 0
        while suffix < min(len(route), len(nominal)) and route[-1-suffix] == nominal[-1-suffix]:
            suffix += 1
        rejoin[rid] = route[len(route) - max(1, suffix)]
    remaining_rejoins = set(rejoin)
    total_length = sum(path_length(_snapshot_remaining_path(r)) for r in robots)
    total_length += sum(path_length(s["path"]) for s in plan.get("relocations", ()))
    total_length += sum(path_length(p) for p in plan.get("preferred_paths", {}).values())
    horizon = total_length / speed + sum(
        r.action_remaining + float((r.current_task or {}).get("duration", 0)) for r in robots) + 2 / speed
    elapsed = 0.0
    last_state = None
    stalled_steps = 0
    with redirect_stdout(io.StringIO()):
        if plan["strategy"] == "relocate":
            resolution_id = sim._begin_resolution_plan(plan, participants)
            if resolution_id in sim.active_resolutions:
                sim.active_resolutions[resolution_id]["fixed_service_robots"] = {
                    r.robot_id for r in robots if r.phase == "acting"}
        elif plan["strategy"] == "short_replan":
            sim._begin_reroute_plan(plan, participants)
        elif plan["strategy"] == "service_reroute":
            worker = sim._robot(plan["rerouted_agents"][0])
            worker.path = list(plan["preferred_paths"][worker.robot_id])
            worker.path_index = 0
            worker.movement_credit = 0.0
            worker.movement_edge = None
        while elapsed <= horizon:
            sim._release_separated_reroutes()
            sim._release_priority_by_lookahead()
            # Local-Delay ends when relocation has produced body-clear holding
            # poses and the priority robot has been released.  Continuation
            # A* computation and travel toward task goals happen after the
            # robots can resume nominal execution and therefore are not part
            # of immediate conflict-resolution time.
            yield_waits = [
                resolution
                for resolution in sim.active_resolutions.values()
                if resolution.get("kind") == "yield_wait"
            ]
            if yield_waits and len(yield_waits) == len(sim.active_resolutions):
                clearance = 2.0 * robot_radius / grid_size
                if all(
                    all(
                        math.dist(
                            sim._robot(resolution["priority_robot"]).cell,
                            sim._robot(robot_id).cell,
                        ) > clearance
                        for robot_id in (
                            resolution["participants"]
                            - {resolution["priority_robot"]}
                        )
                    )
                    for resolution in yield_waits
                ):
                    return elapsed
            sim._release_completed_priority_yields()
            remaining_rejoins = {rid for rid in remaining_rejoins if sim._robot(rid).cell != rejoin[rid]}
            proposals = {r.robot_id: r.path[min(r.path_index + 1, len(r.path) - 1)]
                         for r in sim.robots if sim._is_navigating(r) and r.path}
            conflicts = check_dynamic_robot_collisions(sim.robots, robot_radius, grid_size, proposals)
            blocked = any(participants.intersection(r.robot_id for r in group) for group in conflicts)
            if not sim.active_resolutions and not remaining_rejoins and not blocked:
                return elapsed
            state = tuple((r.cell, r.phase, r.path_index, r.action_remaining) for r in sim.robots)
            stalled_steps = stalled_steps + 1 if state == last_state else 0
            last_state = state
            if stalled_steps >= 3:
                return math.inf
            dt = sim.next_event_duration()
            if dt < 1e-8:
                dt = 1 / speed
            sim.step(dt)
            elapsed += dt
    return math.inf


def run_standalone_simulation(world_file, *, speed=8.0, selection_policy="dag_aware",
                              planner_workers=None, release_prediction_model="events",
                              hypothesis_workers=1):
    StandaloneSimulation(world_file, speed=speed, selection_policy=selection_policy,
                         planner_workers=planner_workers,
                         hypothesis_workers=hypothesis_workers,
                         release_prediction_model=release_prediction_model).run()
