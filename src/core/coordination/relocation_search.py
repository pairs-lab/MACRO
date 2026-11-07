"""Bounded DFS and joint-staging search for local multi-robot conflict repair.

The production pipeline disables DAG lower-bound pruning. Historical BnB
diagnostic labels are retained for archive compatibility.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

from core import (
    optimize,
    order_dag_priorities,
    clearance_yield_key,
)

from .dynamic_collision import swept_separation
from .planner import (
    path_length,
    preferred_path_to_object,
    select_dynamic_replan_path,
    select_preferred_path,
    shortest_path_distances,
    shortest_path_tree_to_any,
    with_geometry_cache,
)


RELOCATION_PATH_BUDGET = 3
DEFAULT_MAX_CANDIDATES_PER_ROOT = 12
DEFAULT_MAX_EXPANDED_PER_ROOT = 64
JOINT_SETUP_POOL_MULTIPLIER = 8
JOINT_ROUTE_SCAN_MULTIPLIER = 8


class PlanningTimeoutError(TimeoutError):
    """Raised when one runtime coordination search exceeds its wall budget."""

    def __init__(self, message, stats=None):
        super().__init__(message)
        self.stats = dict(stats or {})


def _coordinate_key(point):
    if isinstance(point, dict):
        return tuple(
            float(point[name]) for name in ("x", "y", "z") if name in point
        )
    return tuple(point)


def _candidate_stable_key(candidate):
    """Canonical non-DAG tie-break shared by the local baselines."""
    relocations = tuple(
        (
            str(step.get("robot_id", step.get("blocker_agent", ""))),
            _coordinate_key(
                step.get("destination", step.get("release_position", ()))
            ),
            tuple(
                _coordinate_key(point)
                for point in step.get("path", step.get("release_path", ()))
            ),
        )
        for step in candidate.get(
            "relocations", candidate.get("release_moves", ())
        )
    )
    planned_paths = tuple(
        (
            str(robot_id),
            tuple(_coordinate_key(point) for point in path),
        )
        for robot_id, path in sorted(
            candidate.get(
                "planned_paths", candidate.get("current_paths", {})
            ).items(),
            key=lambda item: str(item[0]),
        )
    )
    return (
        str(candidate.get("strategy", "")),
        str(candidate.get("priority_robot", candidate.get("priority_agent", ""))),
        tuple(str(value) for value in candidate.get("moved_agents", ())),
        tuple(str(value) for value in candidate.get("rerouted_agents", ())),
        relocations,
        planned_paths,
    )


def _select_spatially_diverse_steps(candidates, limit):
    """Select a deterministic k-center subset from a geometric move pool.

    The first representative is the fastest clearing move. Subsequent
    representatives maximize their minimum endpoint distance from the chosen
    set. This preserves short, long, lateral, and retreat alternatives without
    enumerating named maneuver cases or prescribed retreat distances.
    """
    ordered = sorted(candidates, key=clearance_yield_key)
    if limit is None or len(ordered) <= limit:
        return ordered
    selected = [ordered[0]]
    selected_keys = {
        (ordered[0]["destination"], tuple(ordered[0]["path"]))
    }
    while len(selected) < limit:
        remaining = [
            step for step in ordered
            if (step["destination"], tuple(step["path"])) not in selected_keys
        ]
        if not remaining:
            break
        indexed = list(enumerate(remaining))
        _, chosen = max(indexed, key=lambda item: (
            min(
                math.dist(item[1]["destination"], kept["destination"])
                for kept in selected
            ),
            -item[0],
        ))
        selected.append(chosen)
        selected_keys.add((chosen["destination"], tuple(chosen["path"])))
    return selected


def _select_resolution_candidate(candidates, selection_policy):
    """Select a resolution under the requested information boundary.

    The two local baselines are deliberately forbidden from using remaining-
    DAG makespan or propagated task-finish delay, including as tie-breakers.
    """
    if selection_policy == "dag_aware":
        return optimize(candidates, evaluator=lambda candidate: candidate)[
            "candidate"
        ]
    if selection_policy == "local_distance":
        return min(
            candidates,
            key=lambda candidate: (
                candidate["local_resolution_distance"],
                _candidate_stable_key(candidate),
            ),
        )
    if selection_policy == "local_delay":
        return min(
            candidates,
            key=lambda candidate: (
                candidate["local_resolution_delay"],
                _candidate_stable_key(candidate),
            ),
        )
    raise ValueError(f"Unknown selection policy: {selection_policy}")


def _select_downstream_candidate(candidates, priority_order):
    """Prefer longest downstream root, then local delay and geometric ties.

    A pure service wait has no priority root: compare it with the best available
    root by local delay instead of excluding non-preemptive service waiting.
    No candidate remaining-mission makespan enters this selector.
    """
    ranks = {robot_id: index for index, robot_id in enumerate(priority_order)}
    best = min((ranks[c['priority_robot']] for c in candidates
                if c.get('priority_robot') in ranks), default=0)
    return min(candidates, key=lambda c: (
        best if c.get('priority_robot') is None else ranks.get(c['priority_robot'], len(ranks)),
        c['local_resolution_delay'], _candidate_stable_key(c)))


def _candidate_diagnostic(candidate, selected=False):
    return {
        "candidate_id": candidate.get("candidate_id"),
        "strategy": candidate["strategy"],
        "moved_robot": candidate.get("rerouted_agents", []) or candidate.get("moved_agents", []),
        "geometry_key": repr(_candidate_stable_key(candidate)),
        "selected": selected,
        "local_time_model": candidate.get("local_time_model", "corridor_barrier"),
        "local_time_finite": math.isfinite(candidate["local_resolution_delay"]),
        "hypothesis": candidate["selected_hypothesis"],
        "estimated_makespan": candidate["estimated_makespan"],
        "total_distance": candidate["total_distance"],
        "additional_distance": candidate.get("additional_distance", candidate["total_distance"]),
        "total_delay": candidate["total_delay"],
        "local_resolution_delay": candidate["local_resolution_delay"],
        "local_resolution_distance": candidate.get("local_resolution_distance", candidate["total_distance"]),
        "nominal_estimated_makespan": candidate.get("nominal_estimated_makespan"),
        "lookahead_shortlisted": candidate.get("lookahead_shortlisted"),
        "lookahead": candidate.get("lookahead"),
        "anticipated_delay": candidate.get("anticipated_delay"),
        "protected_priority_corridor": candidate.get("protected_priority_corridor"),
        "release_prediction": candidate.get("release_prediction"),
    }


def _destination_name(task):
    return task.get("target") if task["action"] == "place" else task["object"]


def _priority_clearance_corridor(robot, current_path, tasks, task_status,
                                 objects, navigation_cells, speed):
    """Protect the unchanged current route and one predictable successor.

    This is a parking constraint, not a simultaneous-motion constraint on all
    future routes. A missing successor route is left unknown, never certified
    as clear. It does not inspect slack or any policy-specific DAG score.
    """
    corridor = list(current_path)
    current = robot.current_task
    if not corridor or current is None:
        return corridor
    ordered = [t for t in _topological_tasks(tasks)
               if t["robot"] == robot.robot_id and task_status.get(t["id"]) != "completed"]
    index = next((i for i,t in enumerate(ordered) if t["id"] == current["id"]), None)
    if index is None or index+1 >= len(ordered):
        return corridor
    destination = _destination_name(ordered[index+1])
    if current["action"] == "place" and destination == current["object"]:
        destination = current["target"]
    target = objects[destination]
    route = preferred_path_to_object(navigation_cells, corridor[-1], target["cells"],
        interaction_cells=target.get("interaction_cells"), robot_speed=speed, candidate_count=3)
    if route is not None:
        corridor.extend(route[1:])
    return corridor


def _topological_tasks(tasks):
    """Return a stable topological ordering of the task DAG."""
    by_id = {task["id"]: task for task in tasks}
    remaining = set(by_id)
    completed = set()
    ordered = []
    while remaining:
        ready = next(
            (
                task
                for task in tasks
                if task["id"] in remaining
                and set(task.get("node", ())) <= completed
            ),
            None,
        )
        if ready is None:
            raise ValueError("Task dependencies contain a cycle")
        ordered.append(ready)
        remaining.remove(ready["id"])
        completed.add(ready["id"])
    return ordered


def _remaining_dag_cost(
    *,
    tasks,
    task_status,
    robots_by_id,
    positions,
    preferred_paths,
    objects,
    navigation_cells,
    grid_size,
    speed_cells_per_second,
    relocation_delay_by_robot,
    include_travel,
    return_details=False,
    return_execution_durations=False,
    baseline_finish_times=None,
    return_evaluation=False,
    affected_robot_ids=(),
    baseline_evaluation=None,
    changed_task_ids=(),
    debug_logger=None,
    future_delay_by_task=None,
    timing_only=False,
):
    """Estimate the remaining DAG schedule from one hypothetical state.

    Relocation delay is charged to every robot affected by that relocation, so
    delaying a critical DAG branch costs more than delaying a non-critical one.
    The ``lower_bound`` travel mode adds only unavoidable current-goal travel.
    """
    remaining = [
        task for task in _topological_tasks(tasks)
        if task_status.get(task["id"]) != "completed"
    ]
    remaining_ids = {task["id"] for task in remaining}
    future_delay_by_task = dict(future_delay_by_task or {})
    if not set(future_delay_by_task) <= remaining_ids:
        raise ValueError("Future delays must refer to remaining task IDs")
    if any(not math.isfinite(float(v)) or v < 0 for v in future_delay_by_task.values()):
        raise ValueError("Future task delays must be finite and nonnegative")
    if timing_only and baseline_evaluation is None:
        raise ValueError("Timing-only propagation requires a baseline evaluation")
    finish_times = {
        task["id"]: 0.0
        for task in tasks
        if task["id"] not in remaining_ids
    }
    available = {
        robot_id: float(relocation_delay_by_robot.get(robot_id, 0.0))
        for robot_id in robots_by_id
    }
    predicted_positions = dict(positions)
    predicted_objects = {
        name: set(value["cells"])
        for name, value in objects.items()
    }
    predicted_interactions = {
        name: (
            set(value["interaction_cells"])
            if value.get("interaction_cells") is not None
            else None
        )
        for name, value in objects.items()
    }
    total_task_distance = 0.0
    predicted_centers = {name: value.get("cell") for name, value in objects.items()}
    physical_speed = speed_cells_per_second * grid_size
    execution_durations = {}
    start_times = {}
    task_distances = {}
    task_endpoints = {}
    task_paths = {}
    task_travel_times = {}

    affected_tasks = None
    directly_changed_tasks = set(changed_task_ids) | set(future_delay_by_task)
    if baseline_evaluation is not None:
        directly_changed_tasks.update(baseline_evaluation.get("future_delay_by_task", {}))
    if baseline_evaluation is not None:
        baseline_finish_times = baseline_evaluation["finish_times"]
    if baseline_finish_times is not None:
        affected_robot_ids = set(affected_robot_ids)
        successors = {task["id"]: set() for task in remaining}
        previous_by_robot = {}
        for task in remaining:
            task_id = task["id"]
            for predecessor in task.get("node", ()):
                if predecessor in successors:
                    successors[predecessor].add(task_id)
            previous = previous_by_robot.get(task["robot"])
            if previous is not None:
                successors[previous].add(task_id)
            previous_by_robot[task["robot"]] = task_id
        if affected_robot_ids:
            first_by_robot = {}
            for task in remaining:
                first_by_robot.setdefault(task["robot"], task["id"])
            directly_changed_tasks.update(
                first_by_robot[robot_id]
                for robot_id in affected_robot_ids
                if robot_id in first_by_robot
            )
        affected_tasks = set(directly_changed_tasks)
        frontier = list(affected_tasks)
        while frontier:
            for successor in successors[frontier.pop()]:
                if successor not in affected_tasks:
                    affected_tasks.add(successor)
                    frontier.append(successor)
        if debug_logger is not None:
            debug_logger({
                "remaining_tasks": len(remaining),
                "directly_changed_tasks": len(directly_changed_tasks),
                "affected_tasks": len(affected_tasks),
            })

    for task in remaining:
        task_id = task["id"]
        robot_id = task["robot"]
        robot = robots_by_id[robot_id]

        if (
            affected_tasks is not None
            and task_id not in affected_tasks
            and baseline_evaluation is not None
        ):
            start_times[task_id] = baseline_evaluation["start_times"][task_id]
            finish_times[task_id] = baseline_finish_times[task_id]
            execution_durations[task_id] = baseline_evaluation[
                "execution_durations"
            ][task_id]
            task_distance = baseline_evaluation["task_distances"][task_id]
            endpoint = baseline_evaluation["task_endpoints"][task_id]
            task_distances[task_id] = task_distance
            task_endpoints[task_id] = endpoint
            task_paths[task_id] = baseline_evaluation.get("task_paths", {}).get(task_id)
            task_travel_times[task_id] = baseline_evaluation.get("task_travel_times", {}).get(task_id, 0.)
            available[robot_id] = finish_times[task_id]
            predicted_positions[robot_id] = endpoint
            total_task_distance += task_distance
            if task["action"] == "place":
                center = predicted_centers[task["target"]]
                predicted_objects[task["object"]] = (
                    {center}
                    if center is not None
                    else set(predicted_objects[task["target"]])
                )
                predicted_centers[task["object"]] = center
            continue

        travel_distance = 0.0
        credited_time = 0.0
        endpoint = predicted_positions[robot_id]

        if include_travel == "lower_bound":
            if (
                robot.current_task is not None
                and robot.current_task["id"] == task_id
                and robot.goal is not None
            ):
                # Relocation path plus this straight-line remainder cannot be
                # greater than the corresponding executable A* sequence.
                travel_distance = (
                    math.dist(predicted_positions[robot_id], robot.goal)
                    * grid_size
                )
                endpoint = robot.goal
        elif include_travel and not timing_only:
            if robot.current_task is not None and robot.current_task["id"] == task_id:
                route = preferred_paths.get(robot_id)
                if route is None and robot.path:
                    route = robot.path[robot.path_index :]
            else:
                destination = _destination_name(task)
                route = preferred_path_to_object(
                    navigation_cells,
                    predicted_positions[robot_id],
                    predicted_objects[destination],
                    interaction_cells=predicted_interactions[destination],
                    robot_speed=speed_cells_per_second,
                    candidate_count=3,
                )
            if route is None:
                raise RuntimeError(f"No remaining DAG route for T{task_id}")
            travel_distance = path_length(route, grid_size)
            task_paths[task_id] = tuple(route)
            endpoint = route[-1]
            if len(route) >= 2 and getattr(robot, "movement_edge", None) == (route[0], route[1]) and positions[robot_id] == route[0]:
                credited_time = getattr(robot, "movement_credit", 0.0) / speed_cells_per_second

        if (
            robot.current_task is not None
            and robot.current_task["id"] == task_id
            and (robot.phase == "acting" or getattr(robot, "interrupted_action_remaining", None) is not None)
        ):
            interrupted = getattr(robot, "interrupted_action_remaining", None)
            action_duration = float(robot.action_remaining if interrupted is None else interrupted)
        else:
            action_duration = float(task.get("duration", 1.0))

        dependency_finish = max(
            (finish_times[dependency] for dependency in task.get("node", ())),
            default=0.0,
        )
        start = max(available[robot_id], dependency_finish)
        execution_duration = (
            max(0.0, travel_distance / physical_speed - credited_time)
            + action_duration
        )
        task_travel_times[task_id] = max(0., travel_distance / physical_speed - credited_time)
        if timing_only:
            travel_distance = baseline_evaluation["task_distances"][task_id]
            endpoint = baseline_evaluation["task_endpoints"][task_id]
            task_paths[task_id] = baseline_evaluation["task_paths"][task_id]
            task_travel_times[task_id] = baseline_evaluation["task_travel_times"][task_id]
            execution_duration = (baseline_evaluation["execution_durations"][task_id]
                - baseline_evaluation.get("future_delay_by_task", {}).get(task_id, 0.))
        execution_duration += float(future_delay_by_task.get(task_id, 0.))
        finish = start + execution_duration
        start_times[task_id] = start
        execution_durations[task_id] = execution_duration
        if (
            affected_tasks is not None
            and task_id not in affected_tasks
            and baseline_evaluation is None
        ):
            finish = baseline_finish_times[task_id]
        finish_times[task_id] = finish
        task_distances[task_id] = travel_distance
        task_endpoints[task_id] = endpoint
        available[robot_id] = finish
        predicted_positions[robot_id] = endpoint
        total_task_distance += travel_distance

        if task["action"] == "place":
            center = predicted_centers[task["target"]]
            predicted_objects[task["object"]] = (
                {center} if center is not None else set(predicted_objects[task["target"]])
            )
            predicted_centers[task["object"]] = center

    result = (max(finish_times.values(), default=0.0), total_task_distance)
    if return_details:
        details = (*result, finish_times)
        if return_execution_durations:
            details = (*details, execution_durations)
        if return_evaluation:
            return (*details, {
                "start_times": start_times,
                "finish_times": finish_times,
                "execution_durations": execution_durations,
                "task_distances": task_distances,
                "task_endpoints": task_endpoints,
                "task_paths": task_paths,
                "task_travel_times": task_travel_times,
                "future_delay_by_task": future_delay_by_task,
            })
        return details
    return result


def evaluate_reroute_plan(
    *,
    worker_id,
    route,
    robots,
    tasks,
    task_status,
    objects,
    navigation_cells,
    grid_size,
    speed_cells_per_second,
    held_robot_ids=(),
    robot_radius=None,
    evaluate_dag=True,
    dag_debug_logger=None,
):
    """Build one DAG-aware candidate that reroutes the working robot."""
    robots_by_id = {robot.robot_id: robot for robot in robots}
    positions = {robot.robot_id: robot.cell for robot in robots}
    planned_route = list(route)
    held_robot_ids = tuple(held_robot_ids)
    delay_by_robot = {}
    release_time = 0.0
    if held_robot_ids:
        clearance = 2 * robot_radius / grid_size
        release_index = next((i for i, cell in enumerate(planned_route)
                              if all(math.dist(cell, positions[r]) > clearance for r in held_robot_ids)), len(planned_route)-1)
        release_time = path_length(planned_route[:release_index+1], grid_size) / (speed_cells_per_second * grid_size)
        delay_by_robot = {r: release_time for r in held_robot_ids}
    nominal_route = select_preferred_path(
        navigation_cells,
        positions[worker_id],
        robots_by_id[worker_id].goal,
        robot_speed=speed_cells_per_second,
        candidate_count=3,
    )
    if nominal_route is None:
        raise RuntimeError(f"No nominal local route for {worker_id}")
    nominal_local_distance = path_length(nominal_route, grid_size)
    reroute_distance = path_length(planned_route, grid_size)
    local_resolution_distance = max(
        0.0,
        reroute_distance - nominal_local_distance,
    )
    # The executor releases held robots only when the remaining corridors are
    # disjoint, not merely when the moving body's centre has passed them.
    clearance = 2 * robot_radius / grid_size if robot_radius else 0.0
    release_index = next((i for i in range(len(planned_route)) if all(
        not _path_conflict(planned_route[i:],
                           list(robots_by_id[r].path[robots_by_id[r].path_index:])
                           or [positions[r]], clearance)
        for r in held_robot_ids)), None)
    local_resolution_delay = (math.inf if release_index is None else
        path_length(planned_route[:release_index + 1], grid_size)
        / (speed_cells_per_second * grid_size))
    if not evaluate_dag:
        return {
            "strategy": "short_replan", "estimated_makespan": math.nan,
            "estimated_remaining_DAG_makespan": math.nan,
            "total_distance": reroute_distance,
            "additional_distance": local_resolution_distance,
            "total_delay": local_resolution_delay,
            "local_resolution_distance": local_resolution_distance,
            "local_resolution_delay": local_resolution_delay,
            "dag_evaluation_s": 0.0, "relocations": [],
            "relocation_sequence": [],
            "preferred_paths": {worker_id: planned_route},
            "planned_paths": {worker_id: planned_route},
            "selected_hypothesis": f"H({worker_id} priority)",
            "priority_robot": worker_id, "moved_agents": [],
            "rerouted_agents": [worker_id], "final_positions": positions,
            "delay_by_robot": {r: local_resolution_delay for r in held_robot_ids},
        }
    dag_started = time.perf_counter()
    (baseline_makespan, baseline_distance, baseline_finishes,
     _, baseline_evaluation) = _remaining_dag_cost(
        tasks=tasks,
        task_status=task_status,
        robots_by_id=robots_by_id,
        positions=positions,
        preferred_paths={},
        objects=objects,
        navigation_cells=navigation_cells,
        grid_size=grid_size,
        speed_cells_per_second=speed_cells_per_second,
        relocation_delay_by_robot={},
        include_travel=True,
        return_details=True,
        return_execution_durations=True,
        return_evaluation=True,
    )
    makespan, total_distance, finish_times = _remaining_dag_cost(
        tasks=tasks,
        task_status=task_status,
        robots_by_id=robots_by_id,
        positions=positions,
        preferred_paths={worker_id: planned_route},
        objects=objects,
        navigation_cells=navigation_cells,
        grid_size=grid_size,
        speed_cells_per_second=speed_cells_per_second,
        relocation_delay_by_robot=delay_by_robot,
        include_travel=True,
        return_details=True,
        baseline_evaluation=baseline_evaluation,
        affected_robot_ids={worker_id, *held_robot_ids},
        debug_logger=dag_debug_logger,
    )
    propagated_delay = sum(
        max(0.0, finish - baseline_finishes.get(task_id, finish))
        for task_id, finish in finish_times.items()
    )
    dag_evaluation_s = time.perf_counter() - dag_started
    return {
        "strategy": "short_replan",
        "estimated_makespan": makespan,
        "estimated_remaining_DAG_makespan": makespan,
        "total_distance": total_distance,
        "additional_distance": max(0.0, total_distance - baseline_distance),
        "total_delay": propagated_delay,
        "nominal_makespan": baseline_makespan,
        # Purely local additional motion relative to the nominal current-task
        # route. The same delta definition is used by relocation leaves.
        "local_resolution_distance": local_resolution_distance,
        # Barrier duration plus extra local-motion duration; no DAG data.
        "local_resolution_delay": local_resolution_delay,
        "dag_evaluation_s": dag_evaluation_s,
        "relocations": [],
        "relocation_sequence": [],
        "preferred_paths": {worker_id: planned_route},
        "planned_paths": {worker_id: planned_route},
        "selected_hypothesis": f"H({worker_id} priority)",
        "priority_robot": worker_id,
        "moved_agents": [],
        "rerouted_agents": [worker_id],
        "final_positions": positions,
    }


def _preferred_paths(
    robots_by_id,
    positions,
    navigation_cells,
    speed,
    robot_ids=None,
):
    """Recompute static-only preferred paths at a hypothetical search node."""
    selected_ids = set(robots_by_id) if robot_ids is None else set(robot_ids)
    paths = {}
    for robot_id, robot in robots_by_id.items():
        if (
            robot_id not in selected_ids
            or robot.current_task is None
            or robot.goal is None
            or (robot.phase == "acting" and positions[robot_id] == robot.cell)
        ):
            continue
        route = select_preferred_path(
            navigation_cells,
            positions[robot_id],
            robot.goal,
            robot_speed=speed,
            candidate_count=3,
        )
        if route is None:
            raise RuntimeError(f"No preferred path for {robot_id}")
        paths[robot_id] = route
    return paths


def _path_conflict(first_path, second_path, clearance_cells):
    limit_squared = clearance_cells * clearance_cells
    return any(
        (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
        <= limit_squared
        for first in first_path
        for second in second_path
    )


def _distance_limited_path(path, distance_limit):
    """Return a route prefix spanning at least ``distance_limit`` cells."""
    route = list(path)
    if not route:
        return []
    prefix = [route[0]]
    travelled = 0.0
    for start, end in zip(route, route[1:]):
        prefix.append(end)
        travelled += math.dist(start, end)
        if travelled + 1e-9 >= distance_limit:
            break
    return prefix


def _timed_path_conflict(first_path, second_path, clearance_cells):
    """Check equal-speed paths at matching times, including swept crossings."""
    first = list(first_path)
    second = list(second_path)
    if not first or not second:
        return False
    first_index = second_index = 0
    first_progress = second_progress = 0.0
    epsilon = 1e-9

    def segment(path, index, progress):
        if index >= len(path) - 1:
            return path[-1], path[-1], math.inf
        start, end = path[index], path[index + 1]
        length = math.dist(start, end)
        if length <= epsilon:
            return start, end, 0.0
        ratio = progress / length
        point = (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )
        return point, end, length - progress

    while first_index < len(first) - 1 or second_index < len(second) - 1:
        a0, a_end, a_remaining = segment(first, first_index, first_progress)
        b0, b_end, b_remaining = segment(second, second_index, second_progress)
        duration = min(a_remaining, b_remaining)
        if not math.isfinite(duration):
            break
        if duration <= epsilon:
            if a_remaining <= epsilon:
                first_index += 1
                first_progress = 0.0
            if b_remaining <= epsilon:
                second_index += 1
                second_progress = 0.0
            continue

        def advance(point, end, remaining):
            if not math.isfinite(remaining) or remaining <= epsilon:
                return point
            ratio = duration / remaining
            return (
                point[0] + (end[0] - point[0]) * ratio,
                point[1] + (end[1] - point[1]) * ratio,
            )

        a1 = advance(a0, a_end, a_remaining)
        b1 = advance(b0, b_end, b_remaining)
        if swept_separation(a0, a1, b0, b1) <= clearance_cells + epsilon:
            return True
        if a_remaining <= duration + epsilon:
            first_index += 1
            first_progress = 0.0
        elif math.isfinite(a_remaining):
            first_progress += duration
        if b_remaining <= duration + epsilon:
            second_index += 1
            second_progress = 0.0
        elif math.isfinite(b_remaining):
            second_progress += duration
    return math.dist(first[-1], second[-1]) <= clearance_cells + epsilon


def _direction_group(start, destination):
    dx = destination[0] - start[0]
    dz = destination[1] - start[1]
    if dx == 0 and dz == 0:
        return None
    return int(round(math.atan2(dz, dx) / (math.pi / 4.0))) % 8


def _route_exits_without_reentry(route, forbidden):
    """Allow departure from an initial conflict, but never cross it again."""
    has_exited = route[0] not in forbidden
    for cell in route[1:]:
        if cell not in forbidden:
            has_exited = True
        elif has_exited:
            return False
    return has_exited


def _route_clears_robot_bodies(route, mover_id, positions, clearance_cells):
    """Permit separation from contact, but never move deeper through a robot."""
    epsilon = 1e-9
    for robot_id, centre in positions.items():
        if robot_id == mover_id:
            continue
        initial = math.dist(route[0], centre)
        previous = initial
        cleared = initial > clearance_cells
        for cell in route[1:]:
            distance = math.dist(cell, centre)
            if cleared:
                if distance <= clearance_cells + epsilon:
                    return False
            else:
                if distance + epsilon < previous:
                    return False
                if distance > clearance_cells + epsilon:
                    cleared = True
            previous = distance
    return True


def _clearance_tree_prefixes(parent, settled, start, forbidden, centres,
                             clearance_cells, grid_size):
    """Evaluate the original route predicates once per shared tree prefix."""
    from .native_geometry import backend
    from .clearance_geometry import context, MAX_BODY_TRANSITIONS
    cache = context()
    native = backend()
    if native is not None and len(centres) <= 63:
        native_cache = None
        if cache is not None:
            if 'native_clearance_cache' not in cache:
                cache['native_clearance_cache'] = native.new_clearance_cache()
                cache['native_clearance_stats_reader'] = native.clearance_cache_stats
            native_cache = cache['native_clearance_cache']
        return native.clearance_prefixes(parent, settled, start, forbidden,
                                          centres, clearance_cells, grid_size, native_cache)
    epsilon = 1e-9
    initial_mask = sum(1 << i for i, centre in enumerate(centres)
                       if math.dist(start, centre) > clearance_cells)
    # valid, has_exited, cleared-body mask, exact metric path length
    prefixes = {start: (True, start not in forbidden, initial_mask, 0.0)}
    transitions = cache.setdefault('clearance_body_transitions', {}) if cache is not None else None
    body_signature = (tuple(centres), clearance_cells)
    for cell in settled:
        if cell == start:
            continue
        previous_cell = parent[cell]
        valid, exited, mask, distance = prefixes[previous_cell]
        distance += math.hypot(cell[0] - previous_cell[0],
                               cell[1] - previous_cell[1]) * grid_size
        if cell in forbidden:
            valid = valid and not exited
        else:
            exited = True
        if valid:
            transition_key = (body_signature, previous_cell, cell, mask)
            if transitions is not None and transition_key in transitions:
                cache['clearance_stats']['body_hits'] += 1
                valid, mask = transitions[transition_key]
                prefixes[cell] = (valid, exited, mask, distance)
                continue
            if cache is not None:
                cache['clearance_stats']['body_misses'] += 1
            for i, centre in enumerate(centres):
                current_distance = math.dist(cell, centre)
                if mask & (1 << i):
                    if current_distance <= clearance_cells + epsilon:
                        valid = False
                        break
                else:
                    if current_distance + epsilon < math.dist(previous_cell, centre):
                        valid = False
                        break
                    if current_distance > clearance_cells + epsilon:
                        mask |= 1 << i
            if transitions is not None and len(transitions) < MAX_BODY_TRANSITIONS:
                transitions[transition_key] = (valid, mask)
        prefixes[cell] = (valid, exited, mask, distance)
    return prefixes


def _nearest_feasible_relocations(
    *,
    mover_id,
    positions,
    protected_paths,
    navigation_cells,
    robot_radius,
    grid_size,
    speed,
    goal=None,
    nominal_path=None,
    path_budget=RELOCATION_PATH_BUDGET,
    route_scan_budget=None,
):
    """Return short, direction-diverse clearing moves, never goal-ranked.

    Static continuation to the task goal remains a feasibility check only.
    Top-K uses clearance time/margin and angular coverage so forward endpoints
    cannot crowd out a feasible opposite-direction retreat before scoring.
    """
    start = positions[mover_id]
    from .clearance_geometry import protected_geometry, endpoint_margins
    from core.yielding import _select_clearance_ranked
    corridor_geometry = protected_geometry(tuple(protected_paths.values()), robot_radius, grid_size)
    forbidden = corridor_geometry[0]

    clearance_cells = 2.0 * robot_radius / grid_size
    from .dynamic_collision import _disk_offsets
    footprint = _disk_offsets(clearance_cells)
    occupied = set()
    protected_ids = set(protected_paths)
    for robot_id, cell in positions.items():
        if robot_id == mover_id or robot_id in protected_ids:
            continue
        occupied.update(
            (cell[0] + dx, cell[1] + dz)
            for dx, dz in footprint
        )

    candidates = []
    goals = navigation_cells - forbidden - occupied - {start}
    continuation_endpoints = None
    if goal is not None:
        continuation_endpoints = set(shortest_path_distances(navigation_cells, goal))
    scan_count = (
        len(goals)
        if route_scan_budget is None
        else min(
            len(goals),
            max(path_budget, int(route_scan_budget)),
        )
    )
    parent, endpoints, settled = shortest_path_tree_to_any(
        navigation_cells,
        start,
        goals,
        dynamic_obstacles=occupied,
        # Direction coverage is not monotone in release length, so ordinary
        # search evaluates every reachable goal. Ordered joint search passes
        # an explicit scan budget because its outer candidate pool is already
        # bounded; this avoids a full-grid scan at every counterfactual state.
        path_count=scan_count,
    )
    centres = tuple(cell for robot_id, cell in positions.items()
                    if robot_id != mover_id)
    prefixes = _clearance_tree_prefixes(
        parent, settled, start, forbidden, centres, clearance_cells, grid_size)
    eligible = []
    for endpoint in endpoints:
        valid, exited, _, distance = prefixes[endpoint]
        if endpoint == start or not valid or not exited:
            continue
        if continuation_endpoints is not None and endpoint not in continuation_endpoints:
            continue
        eligible.append(endpoint)
    margins = endpoint_margins(eligible, corridor_geometry)
    for endpoint, margin in zip(eligible, margins):
        if margin <= 1e-9:
            continue
        distance = prefixes[endpoint][3]
        group = _direction_group(start, endpoint)
        delay = distance / (speed * grid_size)
        # The old canonical key has no path yet at this point. One endpoint
        # identifies one tree path; preserve its exact float coordinate key.
        key = (float(delay), -float(margin), int(group),
               (float(endpoint[0]), float(endpoint[1])), ())
        candidates.append((key, (endpoint, distance, delay, group, margin), group))
    # Every endpoint has exactly one Dijkstra parent path, so destinations
    # uniquely determine path/deduplication ties. Rank metadata first and only
    # materialize the selected routes; angular diversity is unchanged.
    retained = _select_clearance_ranked(candidates, path_budget)
    selected = []
    protects = tuple(sorted(protected_paths))
    for endpoint, distance, delay, group, margin in retained:
        route = [endpoint]
        while route[-1] in parent:
            route.append(parent[route[-1]])
        route.reverse()
        selected.append(dict(robot_id=mover_id, start=start, destination=endpoint,
                             distance=distance, delay=delay, direction_group=group,
                             protects=protects, clearance_margin=margin, path=route))
    return selected


def _order_standalone_priorities(
    priority_robot_ids,
    robots_by_id,
    tasks,
    task_status,
    baseline_finishes,
    baseline_makespan,
    execution_durations,
    debug_logger=None,
):
    """Adapt a standalone snapshot to the shared DAG-priority rule."""
    completed_states = {"completed", "action_done", "failed"}
    remaining_tasks = [
        task for task in tasks
        if task_status.get(task["id"]) not in completed_states
    ]
    remaining_ids = {task["id"] for task in remaining_tasks}
    successors = {task_id: [] for task_id in remaining_ids}
    for task in remaining_tasks:
        for predecessor in task.get("node", ()):
            if predecessor in remaining_ids:
                successors[predecessor].append(task["id"])
    durations = {
        task["id"]: float(
            execution_durations.get(task["id"], task["duration"])
        )
        for task in remaining_tasks
    }
    for robot in robots_by_id.values():
        if (
            robot.current_task is not None
            and robot.current_task["id"] in durations
            and robot.phase == "acting"
        ):
            durations[robot.current_task["id"]] = max(
                0.0,
                float(robot.action_remaining),
            )
    active_task_by_robot = {
        robot_id: (
            robots_by_id[robot_id].current_task["id"]
            if robots_by_id[robot_id].current_task is not None
            else None
        )
        for robot_id in priority_robot_ids
    }
    return order_dag_priorities(
        priority_robot_ids,
        active_task_by_robot,
        successors,
        durations,
        nominal_finish_times=baseline_finishes,
        nominal_makespan=baseline_makespan,
        debug_logger=debug_logger,
    )


@with_geometry_cache
def search_relocation_plan(
    *,
    robots,
    dag_robots=None,
    root_mover_ids=None,
    priority_robot_ids=None,
    participant_ids=None,
    tasks,
    task_status,
    objects,
    navigation_cells,
    robot_radius,
    grid_size,
    speed_cells_per_second,
    max_depth=None,
    additional_candidates=(),
    external_protected_paths=None,
    selection_policy="dag_aware",
    exhaustive_candidates=False,
    max_candidates_per_root=DEFAULT_MAX_CANDIDATES_PER_ROOT,
    max_expanded_per_root=DEFAULT_MAX_EXPANDED_PER_ROOT,
    planning_timeout_s=None,
    dag_bound_pruning=False,
    fixed_robot_ids=(),
    local_time_evaluator=None,
    diagnostics=None,
    dag_debug_logger=None,
    use_dag_priority=True,
    incremental_dag_evaluation=True,
    dag_candidate_refiner=None,
    anticipated_delay_horizon=None,
    preserve_priority_corridor=True,
    allow_composed_clearance=True,
):
    """Return the best complete conflict-resolution plan.

    Search starts only after a real contact/predicted swept conflict. Each root
    is ``H(r_p)``: ``r_p`` is the priority robot whose preferred current-task
    path is preserved. Robots blocking that path may yield incrementally; the
    priority robot is never treated as the yielding mover under its own root.
    ``root_mover_ids`` is retained as a compatibility alias for callers that
    predate the explicit ``priority_robot_ids`` name.
    """
    search_started = time.perf_counter()
    fixed_robot_ids = frozenset(fixed_robot_ids)
    if max_candidates_per_root is not None and max_candidates_per_root < 1:
        raise ValueError("max_candidates_per_root must be positive or None")
    if max_expanded_per_root is not None and max_expanded_per_root < 1:
        raise ValueError("max_expanded_per_root must be positive or None")
    if planning_timeout_s is not None and planning_timeout_s <= 0:
        raise ValueError("planning_timeout_s must be positive or None")
    deadline = (
        None
        if planning_timeout_s is None
        else search_started + float(planning_timeout_s)
    )

    robots_by_id = {robot.robot_id: robot for robot in robots}
    dag_robots_by_id = {
        robot.robot_id: robot
        for robot in (robots if dag_robots is None else dag_robots)
    }
    requested_priorities = priority_robot_ids
    if requested_priorities is None:
        requested_priorities = root_mover_ids
    requested_priorities = (
        None
        if requested_priorities is None
        else tuple(dict.fromkeys(requested_priorities))
    )
    search_ids = (
        set(robots_by_id)
        if participant_ids is None
        else set(participant_ids)
    )
    unknown_ids = search_ids - set(robots_by_id)
    if unknown_ids:
        raise ValueError(f"Unknown conflict participants: {sorted(unknown_ids)}")
    if requested_priorities is not None and not set(requested_priorities) <= search_ids:
        raise ValueError("Every priority robot must be a contact participant")
    if speed_cells_per_second <= 0:
        raise ValueError("speed_cells_per_second must be positive")
    if max_depth is None:
        max_depth = max(2, len(search_ids))

    initial_positions = {
        robot_id: robot.cell for robot_id, robot in robots_by_id.items()
    }
    dag_initial_positions = {
        robot_id: robot.cell for robot_id, robot in dag_robots_by_id.items()
    }

    # Candidate feasibility is policy-independent. DAG-based root ordering and
    # incremental candidate evaluation are independent ablation switches. The
    # local policies retain ascending robot-ID order and never evaluate a DAG.
    evaluate_dag = selection_policy == "dag_aware"
    dag_bound_pruning = False
    baseline_makespan, baseline_task_distance, baseline_finishes = math.nan, 0.0, {}
    baseline_execution_durations = {}
    baseline_evaluation = None

    def dag_positions(local_positions):
        combined = dict(dag_initial_positions)
        combined.update(local_positions)
        return combined

    working_ids = {
        robot_id
        for robot_id, robot in robots_by_id.items()
        if (
            robot_id in search_ids
            and robot.current_task is not None
            and robot.goal is not None
            and robot.phase != "acting"
        )
    }
    if not working_ids:
        raise RuntimeError("No working robot path needs protection")

    if requested_priorities is None:
        requested_priorities = tuple(sorted(working_ids))
    else:
        requested_priorities = tuple(
            robot_id for robot_id in requested_priorities
            if robot_id in working_ids
        )
    if not requested_priorities:
        raise RuntimeError("Resolution search needs an active priority robot")

    priority_order = sorted(requested_priorities)

    clearance_cells = 2.0 * robot_radius / grid_size
    # Match the runtime release guard: protect at least two robot diameters
    # and at least one simulated second of travel.  Beyond this horizon the
    # executor observes the new state, replans continuations, and can trigger
    # another ordinary conflict event if necessary.
    lookahead_cells = max(2.0 * clearance_cells, speed_cells_per_second)
    initial_paths = _preferred_paths(
        robots_by_id,
        initial_positions,
        navigation_cells,
        speed_cells_per_second,
        search_ids,
    )
    external_protected_paths = {
        str(robot_id): list(route)
        for robot_id, route in (external_protected_paths or {}).items()
        if route
    }
    nominal_local_distance = sum(
        path_length(route, grid_size) for route in initial_paths.values()
    )
    feasible_leaves = [dict(candidate) for candidate in additional_candidates]
    metric_speed = speed_cells_per_second * grid_size
    clearance_corridors = {}
    def clearance_corridor(priority_id, route):
        key = (priority_id, tuple(route))
        if key not in clearance_corridors:
            clearance_corridors[key] = _priority_clearance_corridor(
                robots_by_id[priority_id], route, tasks, task_status, objects,
                navigation_cells, speed_cells_per_second)
        return clearance_corridors[key]
    for candidate in feasible_leaves:
        candidate.setdefault(
            "local_resolution_delay",
            candidate.get("local_resolution_distance", 0.0) / metric_speed,
        )
    visited = {}
    incumbent_makespan = min(
        (
            float(candidate["estimated_makespan"])
            for candidate in feasible_leaves
        ),
        default=math.inf,
    )
    stats = diagnostics if diagnostics is not None else {}
    stats.update({
        "roots": 0,
        "expanded": 0,
        "feasible": 0,
        "bound_pruned": 0,
        "duplicate_pruned": 0,
        "no_path_pruned": 0,
        "depth_pruned": 0,
        "nearest_feasible_pruned": 0,
        "additional_candidates": len(feasible_leaves),
        "remaining_dag_evaluation_s": sum(
            candidate.get("dag_evaluation_s", 0.0)
            for candidate in feasible_leaves
        ),
        "remaining_dag_evaluations": 0,
        "priority_order": list(priority_order),
        "candidate_limit_pruned": 0,
        "expanded_limit_pruned": 0,
        "max_candidates_per_root": max_candidates_per_root,
        "max_expanded_per_root": max_expanded_per_root,
        "planning_timeout_s": planning_timeout_s,
        "dag_bound_pruning": bool(dag_bound_pruning),
        "dag_priority": bool(evaluate_dag and use_dag_priority),
        "incremental_dag_evaluation": bool(
            evaluate_dag and incremental_dag_evaluation
        ),
        "lookahead_cells": lookahead_cells,
        "preserve_priority_corridor": bool(preserve_priority_corridor),
        "relocation_top_k_model": "clearance-time-margin-angular-diversity",
        "generated_candidates": len(feasible_leaves),
    })

    def check_deadline():
        if deadline is not None and time.perf_counter() >= deadline:
            stats["planning_wall_time_s"] = time.perf_counter() - search_started
            stats["candidate_evaluations"] = [_candidate_diagnostic(c) for c in feasible_leaves]
            raise PlanningTimeoutError(
                "Runtime conflict planning exceeded its wall-time budget",
                stats,
            )

    def timed_dag_cost(**kwargs):
        check_deadline()
        started = time.perf_counter()
        try:
            return _remaining_dag_cost(**kwargs)
        finally:
            stats["remaining_dag_evaluation_s"] += time.perf_counter() - started
            stats["remaining_dag_evaluations"] += 1

    if evaluate_dag or selection_policy == "downstream_priority":
        (
            baseline_makespan,
            baseline_task_distance,
            baseline_finishes,
            baseline_execution_durations,
            baseline_evaluation,
        ) = timed_dag_cost(
            tasks=tasks, task_status=task_status,
            robots_by_id=dag_robots_by_id,
            positions=dag_initial_positions, preferred_paths={},
            objects=objects, navigation_cells=navigation_cells,
            grid_size=grid_size,
            speed_cells_per_second=speed_cells_per_second,
            relocation_delay_by_robot={}, include_travel=True,
            return_details=True, return_execution_durations=True,
            return_evaluation=True,
        )
        if selection_policy == "downstream_priority":
            # Reuse the shared downstream-duration calculation; ignore its
            # slack order entirely. Candidate scores remain purely local.
            downstream_diagnostics = []
            _order_standalone_priorities(
                requested_priorities, robots_by_id, tasks, task_status,
                baseline_finishes, baseline_makespan,
                baseline_execution_durations,
                debug_logger=downstream_diagnostics.append)
            metrics = downstream_diagnostics[0]['robots']
            lengths = {item['robot_id']: item['downstream_duration'] for item in metrics}
            priority_order = sorted(requested_priorities, key=lambda rid: (-lengths[rid], rid))
            stats['downstream_priority_diagnostics'] = {
                'robots': [{'robot_id': rid, 'downstream_duration': lengths[rid]}
                           for rid in priority_order],
                'priority_order': list(priority_order),
                'selection': 'priority-rank-then-local-delay-then-stable-geometry',
                'uses_slack': False, 'uses_candidate_dag_scoring': False}
        elif use_dag_priority:
            priority_diagnostics = []

            def record_priority_diagnostics(payload):
                priority_diagnostics.append(payload)
                if dag_debug_logger is not None:
                    dag_debug_logger(payload)

            priority_order = list(_order_standalone_priorities(
                requested_priorities, robots_by_id, tasks, task_status,
                baseline_finishes, baseline_makespan,
                baseline_execution_durations,
                debug_logger=record_priority_diagnostics,
            ))
            stats["dag_priority_diagnostics"] = priority_diagnostics
        stats["priority_order"] = list(priority_order)

    def lower_bound(positions, relocations):
        delay_by_robot = defaultdict(float)
        for step in relocations:
            affected = set(step["protects"]) | {step["robot_id"]}
            for robot_id in affected:
                delay_by_robot[robot_id] += step["delay"]
        makespan, _ = timed_dag_cost(
            tasks=tasks,
            task_status=task_status,
            robots_by_id=dag_robots_by_id,
            positions=dag_positions(positions),
            preferred_paths={},
            objects=objects,
            navigation_cells=navigation_cells,
            grid_size=grid_size,
            speed_cells_per_second=speed_cells_per_second,
            relocation_delay_by_robot=delay_by_robot,
            include_travel="lower_bound",
        )
        return makespan

    def evaluate_leaf(priority_id, positions, paths, relocations):
        delay_by_robot = defaultdict(float)
        relocation_distance = 0.0
        for step in relocations:
            affected = set(step["protects"]) | {step["robot_id"]}
            for robot_id in affected:
                delay_by_robot[robot_id] += step["delay"]
            relocation_distance += step["distance"]
        # Yielding continuations start after the priority robot has traversed
        # its protected current-task corridor. They continue toward their
        # original goals from the clearance configurations; they are not sent
        # back to their pre-conflict poses.
        priority_wait = (
            path_length(paths[priority_id], grid_size) / metric_speed
        )
        for step in relocations:
            if step["robot_id"] != priority_id:
                delay_by_robot[step["robot_id"]] += priority_wait
        makespan, task_distance, finish_times = math.nan, sum(
            path_length(route, grid_size) for route in paths.values()), {}
        total_distance = relocation_distance + task_distance
        propagated_delay = sum(
            max(0.0, finish - baseline_finishes.get(task_id, finish))
            for task_id, finish in finish_times.items()
        )
        local_resolution_distance = max(
            0.0,
            relocation_distance
            + sum(path_length(route, grid_size) for route in paths.values())
            - nominal_local_distance,
        )
        barrier_duration = sum(step["delay"] for step in relocations)
        return {
            "estimated_makespan": makespan,
            "estimated_remaining_DAG_makespan": makespan,
            "total_distance": total_distance,
            "additional_distance": local_resolution_distance,
            "total_delay": propagated_delay,
            "nominal_makespan": baseline_makespan,
            "local_resolution_distance": local_resolution_distance,
            # Standalone search fallback: relocation barrier plus corridor
            # release time. Runtime callers replace this conservative estimate
            # with the shared current-execution replay (including look-ahead).
            "local_resolution_delay": barrier_duration + max((
                path_length(paths[priority_id][:i + 1], grid_size) / metric_speed
                for mover in {step["robot_id"] for step in relocations}
                for i in [next((index for index in range(len(paths[priority_id]))
                    if not _path_conflict(paths[priority_id][index:],
                        paths.get(mover, [positions[mover]]), clearance_cells)),
                    len(paths[priority_id]) - 1)]
            ), default=0.0),
            "delay_by_robot": dict(delay_by_robot),
        }

    def dfs(priority_id, priority_path, positions, relocations, depth):
        nonlocal incumbent_makespan, root_candidates, root_expanded
        check_deadline()
        if (
            max_candidates_per_root is not None
            and root_candidates >= max_candidates_per_root
        ):
            stats["candidate_limit_pruned"] += 1
            return
        moved_ids = frozenset(step["robot_id"] for step in relocations)
        state_key = (priority_id, tuple(sorted(positions.items())), moved_ids)
        committed_cost = (
            sum(step["delay"] for step in relocations),
            sum(step["distance"] for step in relocations),
        )
        if not exhaustive_candidates and state_key in visited and visited[state_key] <= committed_cost:
            stats["duplicate_pruned"] += 1
            return
        visited[state_key] = committed_cost

        # DAG lower bounds belong only to the proposed selector. Local
        # baselines must not use DAG information to shape their search space.
        if (
            dag_bound_pruning
            and selection_policy == "dag_aware"
            and not exhaustive_candidates
        ):
            bound = lower_bound(positions, relocations)
            if bound > incumbent_makespan:
                stats["bound_pruned"] += 1
                return
        try:
            paths = _preferred_paths(
                robots_by_id,
                positions,
                navigation_cells,
                speed_cells_per_second,
                search_ids,
            )
        except RuntimeError:
            stats["no_path_pruned"] += 1
            return

        # Legacy mode replans Pp around counterfactual holding poses. In
        # corridor-preserving mode a yielder must actually clear the frozen
        # route; a priority detour cannot make an inadequate bay look valid.
        if relocations and not preserve_priority_corridor:
            replanned_priority = select_dynamic_replan_path(
                navigation_cells,
                positions[priority_id],
                robots_by_id[priority_id].goal,
                stationary_robot_centres=(
                    positions[robot_id]
                    for robot_id in search_ids
                    if robot_id != priority_id
                ),
                clearance_cells=clearance_cells,
                robot_speed=speed_cells_per_second,
                candidate_count=3,
            )
            if replanned_priority is not None:
                priority_path = replanned_priority
        paths[priority_id] = priority_path
        priority_horizon = (clearance_corridor(priority_id, priority_path)
                            if preserve_priority_corridor else
                            _distance_limited_path(priority_path, lookahead_cells))

        priority_blockers = []
        for blocker_id in sorted(search_ids - {priority_id}):
            blocker_path = paths.get(blocker_id)
            if blocker_id in moved_ids and blocker_path is not None:
                # The mover is parked at its counterfactual holding pose.
                # Its continuation is generated only after the priority
                # horizon is released, so it is not active during this leaf.
                blocked = _path_conflict(
                    priority_horizon,
                    [positions[blocker_id]],
                    clearance_cells,
                )
            else:
                obstacle = (
                    _distance_limited_path(blocker_path, lookahead_cells)
                    if blocker_path is not None
                    else [positions[blocker_id]]
                )
                blocked = _path_conflict(
                    priority_horizon,
                    obstacle,
                    clearance_cells,
                )
                if preserve_priority_corridor:
                    blocked = blocked or _path_conflict(priority_horizon,
                        [positions[blocker_id]], clearance_cells)
            if blocked:
                priority_blockers.append(blocker_id)

        # A relocation hypothesis is complete once every yielding robot has a
        # body-clear holding pose outside the frozen current+next corridor
        # (or the short horizon in legacy mode). Do not require *all* routes to be
        # mutually disjoint here: those routes start at different release
        # times, are regenerated from the actual holding poses by the runtime,
        # and remain guarded by swept collision detection.  Requiring static
        # disjointness of all remaining routes made shared corridors falsely
        # infeasible even though a safe park-pass-resume schedule existed.
        #
        # This changes only geometric feasibility.  Every shared-candidate
        # policy receives the same pool and retains its own selector.
        if not priority_blockers:
            if not relocations:
                return
            try:
                metrics = evaluate_leaf(
                    priority_id,
                    positions,
                    paths,
                    relocations,
                )
            except RuntimeError:
                stats["no_path_pruned"] += 1
                return
            relocation_sequence = [dict(step) for step in relocations]
            planned_paths = {
                robot_id: list(route) for robot_id, route in paths.items()
            }
            leaf = {
                **metrics,
                "strategy": "relocate",
                "relocations": relocation_sequence,
                "relocation_sequence": relocation_sequence,
                "preferred_paths": planned_paths,
                "planned_paths": planned_paths,
                "selected_hypothesis": (
                    f"H({priority_id} priority)"
                ),
                "priority_robot": priority_id,
                "moved_agents": list(
                    dict.fromkeys(step["robot_id"] for step in relocations)
                ),
                "final_positions": dict(positions),
                "protected_priority_corridor": list(priority_horizon),
            }
            feasible_leaves.append(leaf)
            stats["generated_candidates"] = len(feasible_leaves)
            root_candidates += 1
            incumbent_makespan = min(incumbent_makespan, metrics["estimated_makespan"])
            stats["feasible"] += 1
            return

        if depth >= max_depth or (not allow_composed_clearance and relocations):
            stats["depth_pruned"] += 1
            return
        if (
            max_expanded_per_root is not None
            and root_expanded >= max_expanded_per_root
        ):
            stats["expanded_limit_pruned"] += 1
            return
        root_expanded += 1
        stats["expanded"] += 1
        # Branch over direct horizon blockers first, then over the remaining
        # participants as possible enabling moves. A direct blocker can be
        # boxed in by a robot that does not itself intersect the priority
        # horizon; relocating that support blocker first can open the only
        # valid ordered sequence. Existing depth/candidate/expansion budgets
        # keep this multi-agent permutation search bounded.
        support_movers = sorted(
            search_ids
            - {priority_id}
            - set(priority_blockers)
            - moved_ids
        )
        movers = tuple(priority_blockers) + tuple(support_movers)

        for mover_id in movers:
            if mover_id == priority_id or mover_id in fixed_robot_ids:
                continue
            # A relocation is selected to clear every currently protected
            # path. Moving the same robot twice is dominated by choosing its
            # final feasible destination directly at the earlier node.
            if mover_id in moved_ids:
                stats["duplicate_pruned"] += 1
                continue
            # A holding pose clears the active priority horizon. Other
            # participants are frozen while relocation steps execute, so
            # their bodies are handled by ``occupied`` and
            # ``_route_clears_robot_bodies`` rather than treating all future
            # task routes as simultaneous permanent obstacles.
            protected = {priority_id: priority_horizon}
            protected.update({
                robot_id: _distance_limited_path(route, lookahead_cells)
                for robot_id, route in external_protected_paths.items()
                if robot_id != mover_id
            })
            candidates = _nearest_feasible_relocations(
                mover_id=mover_id,
                positions=positions,
                protected_paths=protected,
                navigation_cells=navigation_cells,
                robot_radius=robot_radius,
                grid_size=grid_size,
                speed=speed_cells_per_second,
                goal=robots_by_id[mover_id].goal,
                nominal_path=initial_paths.get(mover_id),
            )
            if not candidates:
                stats["no_path_pruned"] += 1
                continue
            stats["nearest_feasible_pruned"] += max(
                0,
                len(navigation_cells) - len(candidates),
            )
            for step in candidates:
                next_positions = dict(positions)
                next_positions[mover_id] = step["destination"]
                dfs(
                    priority_id,
                    priority_path,
                    next_positions,
                    relocations + (step,),
                    depth + 1,
                )

    def ordered_joint_candidates(priority_id, *, fallback_full_scan=False):
        """Generate bounded ordered multi-robot candidates from one pool.

        Each sequence moves the priority robot to one diverse counterfactual
        state, then moves one conflicting participant relative to the newly
        planned priority horizon. No retreat length, direction, passing bay,
        or named maneuver is encoded. Feasibility and policy scoring decide
        which (if any) ordered sequence is useful.
        """
        if not allow_composed_clearance:
            return []
        joint = []
        setup_limit = (
            DEFAULT_MAX_CANDIDATES_PER_ROOT
            if max_candidates_per_root is None
            else max_candidates_per_root
        )
        pool_budget = (
            len(navigation_cells)
            if fallback_full_scan
            else min(
                len(navigation_cells),
                JOINT_SETUP_POOL_MULTIPLIER * setup_limit,
            )
        )
        # The blocker can be an idle robot whose last task ended at the next
        # fixture.  Restricting this joint staging search to ``working_ids``
        # made cul-de-sacs falsely infeasible: the active priority robot had
        # to back into a bay before the idle blocker could leave.  Every
        # non-fixed conflict participant is a valid geometric blocker; task
        # activity only determines which robots may be priority roots.
        for blocker_id in sorted(
            search_ids - {priority_id} - fixed_robot_ids
        ):
            setup_pool = _nearest_feasible_relocations(
                mover_id=priority_id,
                positions=initial_positions,
                protected_paths={
                    blocker_id: [initial_positions[blocker_id]]
                },
                navigation_cells=navigation_cells,
                robot_radius=robot_radius,
                grid_size=grid_size,
                speed=speed_cells_per_second,
                goal=robots_by_id[priority_id].goal,
                nominal_path=initial_paths.get(priority_id),
                path_budget=pool_budget,
                route_scan_budget=pool_budget,
            )
            for setup in _select_spatially_diverse_steps(
                setup_pool,
                setup_limit,
            ):
                check_deadline()
                staged = dict(initial_positions)
                staged[priority_id] = setup["destination"]
                try:
                    staged_paths = _preferred_paths(
                        robots_by_id,
                        staged,
                        navigation_cells,
                        speed_cells_per_second,
                        search_ids,
                    )
                except RuntimeError:
                    continue
                priority_path = staged_paths.get(priority_id)
                if priority_path is None:
                    continue
                # Joint staging exists for the case where the direct blocker
                # cannot leave until the priority robot first backs away.
                # Unlike an ordinary one-sided yield, an idle blocker has no
                # future task that will make it depart later. Park it outside
                # the complete current-task route, otherwise receding-horizon
                # search merely repeats the same two-step staging maneuver at
                # every look-ahead boundary.
                priority_horizon = (clearance_corridor(priority_id, priority_path)
                                    if preserve_priority_corridor else list(priority_path))
                parking_paths = _nearest_feasible_relocations(
                    mover_id=blocker_id,
                    positions=staged,
                    protected_paths={priority_id: priority_horizon},
                    navigation_cells=navigation_cells,
                    robot_radius=robot_radius,
                    grid_size=grid_size,
                    speed=speed_cells_per_second,
                    goal=robots_by_id[blocker_id].goal,
                    nominal_path=initial_paths.get(blocker_id),
                    route_scan_budget=(
                        len(navigation_cells)
                        if fallback_full_scan
                        else JOINT_ROUTE_SCAN_MULTIPLIER
                        * RELOCATION_PATH_BUDGET
                    ),
                )
                for parking in parking_paths:
                    final_positions = dict(staged)
                    final_positions[blocker_id] = parking["destination"]
                    try:
                        final_paths = _preferred_paths(
                            robots_by_id,
                            final_positions,
                            navigation_cells,
                            speed_cells_per_second,
                            search_ids,
                        )
                    except RuntimeError:
                        continue
                    final_priority_path = final_paths.get(priority_id)
                    if final_priority_path is None:
                        continue
                    # Priority staging may change the start, but the blocker
                    # must clear the route frozen BEFORE choosing its bay.
                    if preserve_priority_corridor:
                        final_paths[priority_id] = list(priority_path)
                        final_horizon = priority_horizon
                    else:
                        final_horizon = list(final_priority_path)
                    if any(
                        robot_id != priority_id
                        and _path_conflict(
                            final_horizon,
                            [final_positions[robot_id]],
                            clearance_cells,
                        )
                        for robot_id in search_ids
                    ):
                        continue
                    steps = (dict(setup), dict(parking))
                    metrics = evaluate_leaf(
                        priority_id,
                        final_positions,
                        final_paths,
                        steps,
                    )
                    joint.append({
                        **metrics,
                        "strategy": "relocate",
                        "relocations": list(steps),
                        "relocation_sequence": list(steps),
                        "preferred_paths": {
                            robot_id: list(route)
                            for robot_id, route in final_paths.items()
                        },
                        "planned_paths": {
                            robot_id: list(route)
                            for robot_id, route in final_paths.items()
                        },
                        "selected_hypothesis": (
                            f"H({priority_id} priority/joint)"
                        ),
                        "priority_robot": priority_id,
                        "moved_agents": [priority_id, blocker_id],
                        "final_positions": final_positions,
                        "protected_priority_corridor": list(final_horizon),
                    })
        joint.sort(key=lambda candidate: (
            sum(step["delay"] for step in candidate["relocations"]),
            -min(step["clearance_margin"] for step in candidate["relocations"]),
            sum(step["distance"] for step in candidate["relocations"]),
            tuple(
                (step["robot_id"], step["destination"])
                for step in candidate["relocations"]
            ),
        ))
        # Reserve the ordinary Top-K relocation slots and expose the rest of
        # the per-root budget as joint candidates. Geometry determines this
        # shared pool; DAG/local objectives are applied only afterward.
        joint_budget = max(
            1,
            setup_limit - RELOCATION_PATH_BUDGET,
        )
        return joint[:joint_budget]

    for priority_id in priority_order:
        check_deadline()
        priority_path = initial_paths.get(priority_id)
        if priority_path is None:
            continue
        root_candidates = 0
        root_expanded = 0
        stats["roots"] += 1
        dfs(priority_id, priority_path, initial_positions, (), 0)
        for candidate in ordered_joint_candidates(
            priority_id,
            fallback_full_scan=(root_candidates == 0),
        ):
            feasible_leaves.append(candidate)
            stats["generated_candidates"] = len(feasible_leaves)
            stats["feasible"] += 1

    if not feasible_leaves:
        raise RuntimeError(
            "Branch-and-Bound found no complete conflict-free relocation plan"
        )
    feasible_leaves.sort(key=_candidate_stable_key)
    for stable_index, candidate in enumerate(feasible_leaves):
        candidate.setdefault("candidate_id", f"C{stable_index:04d}:{candidate.get('priority_robot', 'none')}:{candidate['strategy']}")
    stats["candidate_evaluations"] = [_candidate_diagnostic(c) for c in feasible_leaves]
    if local_time_evaluator is not None:
        for index, candidate in enumerate(feasible_leaves):
            check_deadline()
            if candidate.get("strategy") == "relocate":
                # Relocation steps execute sequentially while the involved
                # robots are held.  Completion of that finite barrier is the
                # conservative time at which every participant can resume;
                # subsequent task-route travel is outside conflict resolution.
                candidate["local_resolution_delay"] = sum(
                    float(step["delay"])
                    for step in candidate.get("relocations", ())
                )
                candidate["local_time_model"] = (
                    "sequential_relocation_barrier"
                )
            else:
                candidate["local_resolution_delay"] = local_time_evaluator(
                    candidate
                )
                candidate["local_time_model"] = "current_execution_snapshot"
            stats["candidate_evaluations"][index] = _candidate_diagnostic(candidate)
    candidate_evaluations = {}
    if evaluate_dag:
        for candidate in feasible_leaves:
            if not math.isnan(candidate["estimated_makespan"]) and anticipated_delay_horizon is None:
                continue
            affected_robot_ids = (
                set(candidate.get("delay_by_robot", {}))
                | set(candidate.get("moved_agents", ()))
                | set(candidate.get("rerouted_agents", ()))
            )
            if candidate.get("priority_robot") is not None:
                affected_robot_ids.add(candidate["priority_robot"])

            evaluated = timed_dag_cost(
                tasks=tasks, task_status=task_status,
                robots_by_id=dag_robots_by_id,
                positions=dag_positions(candidate.get("final_positions", initial_positions)),
                preferred_paths=candidate["preferred_paths"], objects=objects,
                navigation_cells=navigation_cells, grid_size=grid_size,
                speed_cells_per_second=speed_cells_per_second,
                relocation_delay_by_robot=candidate.get("delay_by_robot", {}),
                include_travel=True, return_details=True,
                return_evaluation=anticipated_delay_horizon is not None,
                baseline_evaluation=(
                    baseline_evaluation
                    if incremental_dag_evaluation
                    else None
                ),
                affected_robot_ids=affected_robot_ids,
                debug_logger=dag_debug_logger,
            )
            makespan, distance, finishes = evaluated[:3]
            if anticipated_delay_horizon is not None:
                candidate_evaluations[id(candidate)] = evaluated[3]
            distance += sum(step["distance"] for step in candidate.get("relocations", ()))
            candidate.update(estimated_makespan=makespan,
                estimated_remaining_DAG_makespan=makespan, total_distance=distance,
                additional_distance=max(0.0, distance - baseline_task_distance),
                total_delay=sum(max(0.0, finish - baseline_finishes.get(tid, finish))
                                for tid, finish in finishes.items()))
    anticipated_applied = False
    if evaluate_dag and anticipated_delay_horizon is not None:
        from .anticipated_delay import estimate_future_delays
        if not 0 < anticipated_delay_horizon <= 5:
            raise ValueError("Anticipated reservation horizon must be in (0, 5] seconds")
        future_started = time.perf_counter()
        future_deadline = min(future_started+1., deadline if deadline is not None else math.inf)
        pending_updates = []
        try:
            for candidate in feasible_leaves:
                if time.perf_counter() >= future_deadline:
                    raise TimeoutError("anticipated delay budget exhausted")
                candidate_baseline = candidate_evaluations[id(candidate)]
                delays, diagnostics = estimate_future_delays(candidate=candidate,
                    evaluation=candidate_baseline, robots=list(dag_robots_by_id.values()),
                    tasks=tasks, speed=speed_cells_per_second, grid_size=grid_size,
                    robot_radius=robot_radius, horizon=anticipated_delay_horizon,
                    deadline=future_deadline)
                if delays:
                    ca_makespan, _, ca_finishes = timed_dag_cost(
                        tasks=tasks, task_status=task_status, robots_by_id=dag_robots_by_id,
                        positions=dag_positions(candidate.get("final_positions", initial_positions)),
                        preferred_paths=candidate["preferred_paths"], objects=objects,
                        navigation_cells=navigation_cells, grid_size=grid_size,
                        speed_cells_per_second=speed_cells_per_second,
                        relocation_delay_by_robot=candidate.get("delay_by_robot", {}),
                        include_travel=True, return_details=True,
                        future_delay_by_task=delays, changed_task_ids=set(delays),
                        baseline_evaluation=candidate_baseline if incremental_dag_evaluation else None,
                        timing_only=bool(incremental_dag_evaluation), debug_logger=dag_debug_logger)
                else:
                    ca_makespan=candidate["estimated_makespan"]
                    ca_finishes=candidate_baseline["finish_times"]
                pending_updates.append((candidate,ca_makespan,ca_finishes,diagnostics))
            if time.perf_counter() >= future_deadline:
                raise TimeoutError("anticipated delay budget exhausted")
            anticipated_applied = True
        except TimeoutError as error:
            stats["anticipated_delay"] = dict(status="fallback",reason=str(error))
        if anticipated_applied:
            for candidate,ca_makespan,ca_finishes,diagnostics in pending_updates:
                candidate["nominal_estimated_makespan"] = candidate["estimated_makespan"]
                candidate.update(estimated_makespan=ca_makespan,
                    estimated_remaining_DAG_makespan=ca_makespan,
                    total_delay=sum(max(0.,f-baseline_finishes.get(t,f)) for t,f in ca_finishes.items()),
                    anticipated_delay=diagnostics)
            stats["anticipated_delay"] = dict(status="applied",horizon_s=anticipated_delay_horizon,
                candidates_evaluated=len(pending_updates),candidates_with_delay=sum(
                    bool(d["delay_by_task"]) for _,_,_,d in pending_updates if "delay_by_task" in d))
        stats["anticipated_delay"]["wall_time_s"] = time.perf_counter()-future_started
    ranked_candidates = feasible_leaves
    if evaluate_dag and dag_candidate_refiner is not None:
        check_deadline()
        ranked_candidates = dag_candidate_refiner(feasible_leaves, stats, deadline,
                                                 baseline_finishes, baseline_task_distance)
        check_deadline()
    if selection_policy == "downstream_priority":
        selected = _select_downstream_candidate(ranked_candidates, priority_order)
    else:
        selected = (min(ranked_candidates,key=lambda c:(c["estimated_makespan"],
                c["additional_distance"],_candidate_stable_key(c))) if anticipated_applied
                else _select_resolution_candidate(ranked_candidates, selection_policy))
    stats["candidate_evaluations"] = [
        _candidate_diagnostic(candidate, selected=candidate is selected)
        for candidate in feasible_leaves
    ]
    stats["planning_wall_time_s"] = time.perf_counter() - search_started
    selected["search_stats"] = stats
    return selected
