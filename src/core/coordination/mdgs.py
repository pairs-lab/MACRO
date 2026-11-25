"""Multi-branch Disjunctive Graph Search (MDGS), Sec. VII of the paper.

Branch generation (VII-B): priority hypotheses H(R_h), Top-K clearance
configurations scored by C_clear (16), and composed clearance through the
enabling set (18), searched depth-first within B expansions and L branches per
hypothesis. Each branch is then scored by its predicted remaining makespan
(VII-C) and the best one is selected (VII-D). Runtime plans keep the
historical ``relocations`` key for the clearance sequence M^pre.
"""

from __future__ import annotations

import math
import time
import itertools
from collections import defaultdict

from core import (
    optimize,
    order_dag_priorities,
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


# K of the Top-K clearance selection and B of the per-hypothesis expansion
# budget. K = 5 was selected on the grid benchmark (the paper's Sec. VIII
# value is K = 3); B = 64.
TOP_K = 5
EXPANSION_BUDGET = 64
# Complete branches kept per hypothesis; the search of H(R_h) stops there.
BRANCH_LIMIT = 12
# lambda_tau of C_clear (16); lambda_xi = 1 - lambda_tau.
LAMBDA_TAU = 0.8


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


def _select_resolution_candidate(candidates, selection_policy):
    """Select a resolution under the requested information boundary.

    The two local baselines are deliberately forbidden from using remaining-
    DAG makespan or propagated task-finish delay, including as tie-breakers.
    """
    if selection_policy == "marco":
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
    """True if any cell of one path lies within ``clearance_cells`` of the other.

    Exact pairwise test; the second path is bucketed on a grid of side
    ``clearance_cells`` so only neighbouring buckets are compared.
    """
    limit_squared = clearance_cells * clearance_cells
    if len(first_path) * len(second_path) <= 64 or clearance_cells <= 0:
        return any(
            (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
            <= limit_squared
            for first in first_path
            for second in second_path
        )
    size = float(clearance_cells)
    buckets = {}
    for point in second_path:
        buckets.setdefault(
            (math.floor(point[0] / size), math.floor(point[1] / size)), []
        ).append(point)
    for point in first_path:
        bx = math.floor(point[0] / size)
        bz = math.floor(point[1] / size)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for other in buckets.get((bx + dx, bz + dz), ()):
                    if ((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2
                            <= limit_squared):
                        return True
    return False


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


def _clearance_candidates(
    *,
    mover_id,
    positions,
    protected_paths,
    navigation_cells,
    robot_radius,
    grid_size,
    speed,
    goal=None,
    path_budget=TOP_K,
    route_scan_budget=None,
    enabling_ids=(),
    cost_model=None,
):
    """Return the Top-K clearance configurations of one blocker (Sec. VII-B).

    Generate: one Dijkstra tree from the blocker reaches every cell outside
    the protected passages. Select: each endpoint is scored by C_clear (16)
    and Top-K keeps at most one endpoint per displacement direction. Static
    continuation to the task goal remains a feasibility check only.

    Robots in ``enabling_ids`` are not hard obstacles: an endpoint whose
    route passes through one of them is kept and reports, in ``enabling``,
    the robots of R_en (18) that must be cleared first. Every other robot
    body is a hard obstacle.
    """
    start = positions[mover_id]
    from .clearance_geometry import protected_geometry, endpoint_margins
    corridor_geometry = protected_geometry(tuple(protected_paths.values()), robot_radius, grid_size)
    forbidden = corridor_geometry[0]

    clearance_cells = 2.0 * robot_radius / grid_size
    from .dynamic_collision import _disk_offsets
    footprint = _disk_offsets(clearance_cells)
    occupied = set()
    protected_ids = set(protected_paths)
    enabling = tuple(sorted(
        robot_id for robot_id in enabling_ids
        if robot_id != mover_id and robot_id in positions
    ))
    enabling_set = set(enabling)
    for robot_id, cell in positions.items():
        if (robot_id == mover_id or robot_id in protected_ids
                or robot_id in enabling_set):
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
        # Direction coverage is not monotone in clearance length, so every
        # reachable goal is scanned unless a caller bounds the scan.
        path_count=scan_count,
    )
    centres = tuple(cell for robot_id, cell in positions.items()
                    if robot_id != mover_id and robot_id not in enabling_set)
    prefixes = _clearance_tree_prefixes(
        parent, settled, start, forbidden, centres, clearance_cells, grid_size)
    obstruction, penetration = _tree_obstruction_masks(
        parent, settled, start, [positions[rid] for rid in enabling],
        clearance_cells)
    if cost_model is None:
        cost_model = default_clearance_cost_model(robot_radius, grid_size, speed)
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
        mask = obstruction.get(endpoint, 0)
        # tau_clear is the time until the blocker reaches z_clr. An enabling
        # robot must first leave the route, i.e. travel at least its deepest
        # overlap with the route, which bounds the wait from below.
        enabling_wait = sum(
            penetration[endpoint][index] / speed
            for index in range(len(enabling)) if mask & (1 << index))
        cost = clearance_cost(delay + enabling_wait, margin, cost_model)
        blockers = tuple(sorted(
            (rid for index, rid in enumerate(enabling) if mask & (1 << index)),
            key=lambda rid: (math.dist(start, positions[rid]), rid),
        ))
        # One endpoint identifies one tree path, so the endpoint identifies the
        # deduplication key (positions 3 and 4 of the key).
        key = (float(cost), float(delay), int(group),
               (float(endpoint[0]), float(endpoint[1])), ())
        candidates.append((key, (endpoint, distance, delay, group, margin,
                                 cost, blockers), group))
    # Every endpoint has exactly one Dijkstra parent path, so destinations
    # uniquely determine path/deduplication ties. Rank metadata first and only
    # materialize the selected routes; angular diversity is unchanged.
    retained = _top_k_per_direction(candidates, path_budget)
    selected = []
    protects = tuple(sorted(protected_paths))
    for endpoint, distance, delay, group, margin, cost, blockers in retained:
        route = [endpoint]
        while route[-1] in parent:
            route.append(parent[route[-1]])
        route.reverse()
        selected.append(dict(robot_id=mover_id, start=start, destination=endpoint,
                             distance=distance, delay=delay, direction_group=group,
                             protects=protects, clearance_margin=margin,
                             clearance_cost=cost, enabling=blockers, path=route))
    return selected


def _top_k_per_direction(entries, count):
    """Top-K directional diversity of Fig. 4(b).

    Keep the best (lowest C_clear) cell of each displacement direction, then
    the ``count`` best of those directions, so adjacent near-duplicates of one
    cell never fill the budget. ``count=None`` returns every entry ranked.
    """
    ranked = sorted(entries, key=lambda item: item[0])
    if count is None:
        return [payload for _, payload, _ in ranked]
    best = {}
    for key, payload, direction in ranked:
        best.setdefault(direction, (key, payload))
    kept = sorted(best.values(), key=lambda item: item[0])[:max(0, int(count))]
    return [payload for _, payload in kept]


def default_clearance_cost_model(robot_radius, grid_size, speed,
                                 lambda_tau=LAMBDA_TAU):
    """Normalizers of (16): one robot diameter of travel time and of margin."""
    diameter = 2.0 * robot_radius
    return dict(lambda_tau=float(lambda_tau),
                tau_ref=diameter / (speed * grid_size),
                xi_ref=diameter)


def clearance_cost(clearance_time, clearance_margin, model):
    """C_clear of (16): weighted, normalized clearance time and margin."""
    lambda_tau = model["lambda_tau"]
    if not 0.0 <= lambda_tau <= 1.0:
        raise ValueError("lambda_tau must lie in [0, 1]")
    margin_term = 1.0 - min(max(0.0, clearance_margin) / model["xi_ref"], 1.0)
    return (lambda_tau * clearance_time / model["tau_ref"]
            + (1.0 - lambda_tau) * margin_term)


def _body_violation_update(cell, previous, centre, cleared, clearance_cells):
    """One step of the body rule; returns (violated, cleared).

    A robot that starts in contact may only move away until separated; once
    separated it may never come within ``clearance_cells`` again.
    """
    epsilon = 1e-9
    distance = math.dist(cell, centre)
    if cleared:
        return distance <= clearance_cells + epsilon, True
    if distance + epsilon < math.dist(previous, centre):
        return True, False
    return False, distance > clearance_cells + epsilon


def _tree_obstruction_masks(parent, settled, start, centres, clearance_cells):
    """Per tree cell: bitmask of the robots its route moves through, and the
    deepest overlap (cells) of the route with each robot's clearance disc."""
    if not centres:
        return {}, {}
    epsilon = 1e-9
    cleared = 0
    for index, centre in enumerate(centres):
        if math.dist(start, centre) > clearance_cells + epsilon:
            cleared |= 1 << index
    state = {start: (cleared, 0)}
    overlap = {start: tuple(0.0 for _ in centres)}
    for cell in settled:
        if cell == start or cell not in parent:
            continue
        previous = parent[cell]
        cleared, violated = state[previous]
        overlap[cell] = tuple(
            max(depth, clearance_cells - math.dist(cell, centre))
            for depth, centre in zip(overlap[previous], centres))
        for index, centre in enumerate(centres):
            bit = 1 << index
            if violated & bit:
                continue
            hit, now_cleared = _body_violation_update(
                cell, previous, centre, bool(cleared & bit), clearance_cells)
            if hit:
                violated |= bit
            elif now_cleared:
                cleared |= bit
        state[cell] = (cleared, violated)
    return {cell: masks[1] for cell, masks in state.items()}, overlap


def _route_obstructions(route, mover_id, positions, clearance_cells):
    """Robots (other than the mover) whose bodies a route moves through."""
    epsilon = 1e-9
    found = []
    for robot_id, centre in positions.items():
        if robot_id == mover_id:
            continue
        cleared = math.dist(route[0], centre) > clearance_cells + epsilon
        for previous, cell in zip(route, route[1:]):
            hit, cleared = _body_violation_update(
                cell, previous, centre, cleared, clearance_cells)
            if hit:
                found.append(robot_id)
                break
    return sorted(found, key=lambda rid: (math.dist(route[0], positions[rid]), rid))


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
def mdgs_resolve(
    *,
    robots,
    dag_robots=None,
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
    selection_policy="marco",
    branch_limit=BRANCH_LIMIT,
    expansion_budget=EXPANSION_BUDGET,
    planning_timeout_s=None,
    fixed_robot_ids=(),
    local_time_evaluator=None,
    diagnostics=None,
    dag_debug_logger=None,
    use_dag_priority=True,
    incremental_dag_evaluation=True,
    dag_candidate_refiner=None,
    preserve_priority_corridor=True,
    allow_composed_clearance=True,
    enabling_robots=(),
    lambda_tau=LAMBDA_TAU,
):
    """Return the best complete conflict-resolution plan.

    Search starts only after a real contact/predicted swept conflict. Each root
    is ``H(r_p)``: ``r_p`` is the priority robot whose preferred current-task
    path is preserved. The robots of R_c on that passage are cleared one at a
    time (MDGS, Sec. VII-B); a robot that obstructs a clearance move joins as
    the next blocker (enabling set (18)), which composes the clearances. The
    priority robot itself may be such an enabling robot, e.g. when it must
    back away so that a boxed-in blocker can leave.

    ``enabling_robots`` are stationary robots outside the contact group that
    may join R_c in this way.
    """
    search_started = time.perf_counter()
    fixed_robot_ids = frozenset(fixed_robot_ids)
    if branch_limit is not None and branch_limit < 1:
        raise ValueError("branch_limit must be positive or None")
    if expansion_budget is not None and expansion_budget < 1:
        raise ValueError("expansion_budget must be positive or None")
    if planning_timeout_s is not None and planning_timeout_s <= 0:
        raise ValueError("planning_timeout_s must be positive or None")
    deadline = (
        None
        if planning_timeout_s is None
        else search_started + float(planning_timeout_s)
    )

    robots_by_id = {robot.robot_id: robot for robot in robots}
    outsider_ids = set()
    for robot in enabling_robots:
        if robot.robot_id not in robots_by_id:
            robots_by_id[robot.robot_id] = robot
            outsider_ids.add(robot.robot_id)
    dag_robots_by_id = {
        robot.robot_id: robot
        for robot in (robots if dag_robots is None else dag_robots)
    }
    requested_priorities = (
        None
        if priority_robot_ids is None
        else tuple(dict.fromkeys(priority_robot_ids))
    )
    search_ids = (
        set(robots_by_id) - outsider_ids
        if participant_ids is None
        else set(participant_ids)
    )
    outsider_ids -= search_ids
    unknown_ids = search_ids - set(robots_by_id)
    if unknown_ids:
        raise ValueError(f"Unknown conflict participants: {sorted(unknown_ids)}")
    if requested_priorities is not None and not set(requested_priorities) <= search_ids:
        raise ValueError("Every priority robot must be a contact participant")
    if speed_cells_per_second <= 0:
        raise ValueError("speed_cells_per_second must be positive")
    # Robots that may be displaced. Each robot is displaced at most once.
    movable_ids = frozenset((search_ids | outsider_ids) - fixed_robot_ids)
    if max_depth is None:
        max_depth = max(1, len(movable_ids))
    cost_model = default_clearance_cost_model(
        robot_radius, grid_size, speed_cells_per_second, lambda_tau)

    initial_positions = {
        robot_id: robot.cell for robot_id, robot in robots_by_id.items()
    }
    dag_initial_positions = {
        robot_id: robot.cell for robot_id, robot in dag_robots_by_id.items()
    }

    # Candidate feasibility is policy-independent. DAG-based root ordering and
    # incremental candidate evaluation are independent ablation switches. The
    # local policies retain ascending robot-ID order and never evaluate a DAG.
    evaluate_dag = selection_policy == "marco"
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
    def reservation_owner(key):
        return key[0] if isinstance(key, tuple) else key

    # An enabling candidate is a body in the search state, not a reservation.
    external_protected_paths = {
        str(robot_id): list(route)
        for robot_id, route in (external_protected_paths or {}).items()
        if route and reservation_owner(robot_id) not in outsider_ids
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
        "branch_limit": branch_limit,
        "expansion_budget": expansion_budget,
        "planning_timeout_s": planning_timeout_s,
        "dag_priority": bool(evaluate_dag and use_dag_priority),
        "incremental_dag_evaluation": bool(
            evaluate_dag and incremental_dag_evaluation
        ),
        "lookahead_cells": lookahead_cells,
        "preserve_priority_corridor": bool(preserve_priority_corridor),
        "relocation_top_k_model": "weighted-clearance-cost-angular-diversity",
        "clearance_cost_model": dict(cost_model),
        "search_order": "depth-first-children-by-clearance-cost",
        "enabling_candidates": sorted(outsider_ids),
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

    def evaluate_leaf(priority_id, positions, paths, relocations, held_ids=()):
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
        for robot_id in held_ids:
            delay_by_robot[robot_id] += priority_wait
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
        barrier_duration = (sum(step["delay"] for step in relocations)
                            if relocations else priority_wait * bool(held_ids))
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

    def blocking_robots(priority_id, priority_horizon, positions, paths, moved_ids):
        """Robots of R_c on the protected passage, nearest first."""
        found = []
        for blocker_id in sorted(search_ids - {priority_id}):
            blocker_path = paths.get(blocker_id)
            if blocker_id in moved_ids or blocker_path is None:
                # A displaced robot is parked at its clearance configuration;
                # its continuation starts only after the passage is released.
                obstacle = [positions[blocker_id]]
                blocked = _path_conflict(priority_horizon, obstacle, clearance_cells)
            else:
                blocked = _path_conflict(
                    priority_horizon,
                    _distance_limited_path(blocker_path, lookahead_cells),
                    clearance_cells,
                )
                if preserve_priority_corridor:
                    blocked = blocked or _path_conflict(
                        priority_horizon, [positions[blocker_id]], clearance_cells)
            if blocked:
                found.append(blocker_id)

        def first_contact(robot_id):
            centre = positions[robot_id]
            for index, cell in enumerate(priority_horizon):
                if math.dist(cell, centre) <= clearance_cells + 1e-9:
                    return index
            return len(priority_horizon)
        return sorted(found, key=lambda rid: (first_contact(rid), rid))

    def record_leaf(priority_id, positions, paths, relocations, cost,
                    priority_horizon, held_ids=()):
        """Append one complete branch u_k to the candidate set U_c^B (19)."""
        try:
            metrics = evaluate_leaf(priority_id, positions, paths, relocations,
                                    held_ids=held_ids)
        except RuntimeError:
            stats["no_path_pruned"] += 1
            return False
        moved_ids = {step["robot_id"] for step in relocations}
        relocation_sequence = [dict(step) for step in relocations]
        planned_paths = {rid: list(route) for rid, route in paths.items()}
        joined = sorted(moved_ids - search_ids)
        feasible_leaves.append({
            **metrics,
            "strategy": "relocate",
            "relocations": relocation_sequence,
            "relocation_sequence": relocation_sequence,
            "preferred_paths": planned_paths,
            "planned_paths": planned_paths,
            "selected_hypothesis": (f"H({priority_id} priority/wait)" if held_ids
                                    else f"H({priority_id} priority)"),
            "priority_robot": priority_id,
            "moved_agents": list(dict.fromkeys(
                step["robot_id"] for step in relocations)),
            "held_agents": list(held_ids),
            "enabled_robots": joined,
            "participants": sorted(search_ids | set(joined)),
            "clearance_cost": cost,
            "final_positions": dict(positions),
            "protected_priority_corridor": list(priority_horizon),
        })
        stats["generated_candidates"] = len(feasible_leaves)
        stats["feasible"] += 1
        return True

    def expand_hypothesis(priority_id, root_priority_path):
        """Depth-first MDGS search of one hypothesis H(R_h) (Sec. VII-B).

        A state is the set of committed clearance maneuvers plus a stack of
        pending clearances: a blocker whose clearance route still meets a
        robot of its enabling set (18) waits on the stack while that robot is
        cleared first, which yields the edge (c', c_b) of a composed
        clearance. The search is depth-first and expands the children of a
        state in increasing C_clear, so a long composed clearance is reached
        within the budget; at most ``expansion_budget`` (B) expansions
        and ``branch_limit`` complete branches per hypothesis.
        """
        root_candidates = 0
        root_expanded = 0
        sequence = itertools.count()
        open_states = [(0.0, next(sequence), dict(initial_positions), (), ())]
        while open_states:
            check_deadline()
            if (branch_limit is not None
                    and root_candidates >= branch_limit):
                stats["candidate_limit_pruned"] += 1
                return
            cost, _, positions, relocations, pending = open_states.pop()
            positions = dict(positions)
            relocations = list(relocations)
            pending = list(pending)
            # Commit every pending clearance whose route has become free.
            obstructing = []
            while pending:
                step = pending[-1]
                obstructing = _route_obstructions(
                    step["path"], step["robot_id"], positions, clearance_cells)
                if obstructing:
                    break
                positions[step["robot_id"]] = step["destination"]
                relocations.append(pending.pop())
            moved_ids = frozenset(step["robot_id"] for step in relocations)
            waiting_ids = frozenset(step["robot_id"] for step in pending)
            state_key = (
                priority_id,
                tuple(sorted(positions.items())),
                moved_ids,
                tuple((step["robot_id"], step["destination"]) for step in pending),
            )
            if state_key in visited and visited[state_key] <= cost:
                stats["duplicate_pruned"] += 1
                continue
            visited[state_key] = cost

            if pending:
                # Expand: the nearest robot of R_en(c_b) becomes the next
                # blocker; it must clear the waiting blocker's route.
                enabling_id = obstructing[0]
                if (enabling_id not in movable_ids or enabling_id in moved_ids
                        or enabling_id in waiting_ids):
                    stats["no_path_pruned"] += 1
                    continue
                protected = {
                    step["robot_id"]: list(step["path"]) for step in pending
                }
                if enabling_id != priority_id:
                    protected[priority_id] = priority_horizon_of(
                        priority_id, positions, root_priority_path, moved_ids)
                mover_id = enabling_id
            else:
                try:
                    paths = _preferred_paths(
                        robots_by_id, positions, navigation_cells,
                        speed_cells_per_second, search_ids | moved_ids)
                except RuntimeError:
                    stats["no_path_pruned"] += 1
                    continue
                priority_path = priority_route_of(
                    priority_id, positions, root_priority_path, moved_ids, paths)
                if priority_path is None:
                    stats["no_path_pruned"] += 1
                    continue
                paths[priority_id] = priority_path
                priority_horizon = horizon_of(priority_id, priority_path)
                blockers = blocking_robots(
                    priority_id, priority_horizon, positions, paths, moved_ids)
                if not relocations:
                    # Re-orientation branch (L_k = 0): every other robot of
                    # R_c holds in place and passes after R_h. It exists only
                    # if no held body stands on the protected passage.
                    held = sorted(rid for rid in search_ids - {priority_id}
                                  if robots_by_id[rid].current_task is not None)
                    if held and not any(
                            _path_conflict(priority_horizon, [positions[rid]],
                                           clearance_cells)
                            for rid in search_ids - {priority_id}):
                        if record_leaf(priority_id, positions, paths, (), cost,
                                       priority_horizon, held_ids=held):
                            root_candidates += 1
                    if not blockers:
                        continue
                elif not blockers:
                    if record_leaf(priority_id, positions, paths, relocations,
                                   cost, priority_horizon):
                        root_candidates += 1
                    continue
                if relocations and not allow_composed_clearance:
                    stats["depth_pruned"] += 1
                    continue
                mover_id = blockers[0]
                if mover_id not in movable_ids or mover_id in moved_ids:
                    stats["no_path_pruned"] += 1
                    continue
                protected = {priority_id: priority_horizon}
            if len(relocations) + len(pending) >= max_depth:
                stats["depth_pruned"] += 1
                continue
            if (expansion_budget is not None
                    and root_expanded >= expansion_budget):
                stats["expanded_limit_pruned"] += 1
                return
            root_expanded += 1
            stats["expanded"] += 1
            protected.update({
                robot_id: _distance_limited_path(route, lookahead_cells)
                for robot_id, route in external_protected_paths.items()
                if robot_id != mover_id
            })
            enabling_ids = (
                movable_ids - moved_ids - waiting_ids - {mover_id}
                if allow_composed_clearance else ()
            )
            # Generate + Select (16): Top-K clearance configurations.
            candidates = _clearance_candidates(
                mover_id=mover_id,
                positions=positions,
                protected_paths=protected,
                navigation_cells=navigation_cells,
                robot_radius=robot_radius,
                grid_size=grid_size,
                speed=speed_cells_per_second,
                goal=robots_by_id[mover_id].goal,
                enabling_ids=enabling_ids,
                cost_model=cost_model,
            )
            if not candidates:
                stats["no_path_pruned"] += 1
                continue
            stats["nearest_feasible_pruned"] += max(
                0, len(navigation_cells) - len(candidates))
            # Candidates arrive in increasing C_clear; push them in reverse so
            # the cheapest child is expanded first (LIFO stack).
            for step in reversed(candidates):
                # Reservation keys may identify several paths for one external
                # robot (e.g. ('R3', 0)); they constrain geometry but are NOT
                # participants that this repair may hold or assign delays to.
                step = dict(step, protects=tuple(
                    rid for rid in step["protects"] if rid in search_ids))
                # Expand (17): insert the clearance node of this maneuver.
                open_states.append((
                    cost + step["clearance_cost"], next(sequence),
                    positions, tuple(relocations), tuple(pending) + (step,)))

    def priority_route_of(priority_id, positions, root_priority_path, moved_ids, paths):
        if priority_id in moved_ids:
            # The priority robot itself was cleared for an enabling move; its
            # passage is planned again from its clearance configuration.
            return paths.get(priority_id)
        if moved_ids and not preserve_priority_corridor:
            replanned = select_dynamic_replan_path(
                navigation_cells, positions[priority_id],
                robots_by_id[priority_id].goal,
                stationary_robot_centres=(
                    positions[rid] for rid in search_ids if rid != priority_id),
                clearance_cells=clearance_cells,
                robot_speed=speed_cells_per_second,
                candidate_count=3,
            )
            if replanned is not None:
                return replanned
        return root_priority_path

    def horizon_of(priority_id, priority_path):
        return (clearance_corridor(priority_id, priority_path)
                if preserve_priority_corridor else
                _distance_limited_path(priority_path, lookahead_cells))

    def priority_horizon_of(priority_id, positions, root_priority_path, moved_ids):
        if priority_id in moved_ids:
            route = select_preferred_path(
                navigation_cells, positions[priority_id],
                robots_by_id[priority_id].goal,
                robot_speed=speed_cells_per_second, candidate_count=3)
            if route is None:
                return [positions[priority_id]]
            return horizon_of(priority_id, route)
        return horizon_of(priority_id, root_priority_path)

    for priority_id in priority_order:
        check_deadline()
        priority_path = initial_paths.get(priority_id)
        if priority_path is None:
            continue
        stats["roots"] += 1
        expand_hypothesis(priority_id, priority_path)

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
                candidate["local_resolution_delay"] = (sum(
                    float(step["delay"])
                    for step in candidate["relocations"]
                ) if candidate["relocations"] else max(
                    candidate.get("delay_by_robot", {}).values(), default=0.0))
                candidate["local_time_model"] = (
                    "sequential_relocation_barrier"
                )
            else:
                candidate["local_resolution_delay"] = local_time_evaluator(
                    candidate
                )
                candidate["local_time_model"] = "current_execution_snapshot"
            stats["candidate_evaluations"][index] = _candidate_diagnostic(candidate)
    if evaluate_dag:
        for candidate in feasible_leaves:
            if not math.isnan(candidate["estimated_makespan"]):
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
                baseline_evaluation=(
                    baseline_evaluation
                    if incremental_dag_evaluation
                    else None
                ),
                affected_robot_ids=affected_robot_ids,
                debug_logger=dag_debug_logger,
            )
            makespan, distance, finishes = evaluated[:3]
            distance += sum(step["distance"] for step in candidate.get("relocations", ()))
            candidate.update(estimated_makespan=makespan,
                estimated_remaining_DAG_makespan=makespan, total_distance=distance,
                additional_distance=max(0.0, distance - baseline_task_distance),
                total_delay=sum(max(0.0, finish - baseline_finishes.get(tid, finish))
                                for tid, finish in finishes.items()))
    ranked_candidates = feasible_leaves
    if evaluate_dag and dag_candidate_refiner is not None:
        check_deadline()
        ranked_candidates = dag_candidate_refiner(feasible_leaves, stats, deadline,
                                                 baseline_finishes, baseline_task_distance)
        check_deadline()
    if selection_policy == "downstream_priority":
        selected = _select_downstream_candidate(ranked_candidates, priority_order)
    else:
        selected = _select_resolution_candidate(ranked_candidates, selection_policy)
    stats["candidate_evaluations"] = [
        _candidate_diagnostic(candidate, selected=candidate is selected)
        for candidate in feasible_leaves
    ]
    stats["planning_wall_time_s"] = time.perf_counter() - search_started
    selected["search_stats"] = stats
    return selected
