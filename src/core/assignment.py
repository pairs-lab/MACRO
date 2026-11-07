"""Search and evaluate multi-robot task assignments."""

import heapq
import json
import math
from collections.abc import Mapping
from pathlib import Path

DEFAULT_SERVICE_TIME = 1.0
MAKESPAN_TOLERANCE = 0.0
BOUND_FLOAT_TOLERANCE = 1e-9


def validate_tasks(tasks: list[dict]) -> None:
    """Validate the task fields required by the shared assignment core."""
    if not isinstance(tasks, list):
        raise ValueError("Tasks must be a JSON list.")
    seen_ids = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"Each task must be an object, got {task!r}.")
        task_id = task.get("id")
        if type(task_id) is not int or task_id <= 0:
            raise ValueError(f"Invalid task ID: {task_id!r}.")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task ID: {task_id}.")
        seen_ids.add(task_id)
        if not isinstance(task.get("action"), str):
            raise ValueError(f"Task {task_id} has an invalid action.")
        if not isinstance(task.get("object"), str) or not task["object"].strip():
            raise ValueError(f"Task {task_id} has an invalid object.")
        candidates = task.get("candidate")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not all(isinstance(robot, str) and robot for robot in candidates)
        ):
            raise ValueError(
                f"Task {task_id} must have a non-empty candidate list."
            )
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"Task {task_id} contains duplicate candidates.")


def _simulation_robot_name(robot: str) -> str:
    """Convert scheduler IDs such as R12 to simulation names such as robot12."""
    suffix = robot[1:] if robot.startswith("R") else ""
    return f"robot{suffix}" if suffix.isdigit() else robot


def _robot_value(value, robot: str, value_name: str) -> float:
    selected = value[robot] if isinstance(value, Mapping) else value
    selected = float(selected)
    if not math.isfinite(selected) or selected <= 0:
        raise ValueError(f"{value_name} for {robot} must be positive.")
    return selected


def _task_service_duration(task: dict, service_time, robot: str) -> float:
    """Resolve a task duration while keeping simulator policy outside core."""
    if service_time is None:
        value = task.get("duration", DEFAULT_SERVICE_TIME)
    elif callable(service_time):
        value = service_time(task, robot)
    else:
        value = (
            service_time[robot]
            if isinstance(service_time, Mapping)
            else service_time
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"service_time for task {task['id']} / {robot} must be positive."
        )
    return value


def _task_location(task: dict) -> str:
    """Return the object/receptacle location reached while executing a task."""
    if task["action"] == "place" and task.get("target"):
        return task["target"]
    return task["object"]


def _topological_tasks(tasks: list[dict]) -> list[dict]:
    """Return a stable topological order and reject malformed DAGs."""
    task_by_id = {task["id"]: task for task in tasks}
    input_order = {task["id"]: index for index, task in enumerate(tasks)}
    successors = {task_id: [] for task_id in task_by_id}
    indegree = {task_id: 0 for task_id in task_by_id}

    for task in tasks:
        dependencies = task.get("node")
        if not isinstance(dependencies, list):
            raise ValueError(f"Task {task['id']} node must be a list.")
        for predecessor in dependencies:
            if predecessor not in task_by_id:
                raise ValueError(
                    f"Task {task['id']} depends on missing task {predecessor}."
                )
            successors[predecessor].append(task["id"])
            indegree[task["id"]] += 1

    ready = [
        (input_order[task_id], task_id)
        for task_id, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    ordered = []
    while ready:
        _, task_id = heapq.heappop(ready)
        ordered.append(task_by_id[task_id])
        for successor in successors[task_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(
                    ready,
                    (input_order[successor], successor),
                )

    if len(ordered) != len(tasks):
        raise ValueError("Task dependencies contain a cycle.")
    return ordered


def _route(distance, robot, location, start_position):
    simulation_robot = _simulation_robot_name(robot)
    if hasattr(distance, "plan"):
        return distance.plan(
            simulation_robot,
            location,
            start_position=start_position,
        )

    # Compatibility with a simple distance(robot, object) callback. Such a
    # callback cannot update a hypothetical endpoint between consecutive tasks.
    return float(distance(simulation_robot, location)), start_position


def evaluate_plan(
    tasks: list[dict],
    assignment: dict[int, str],
    distance,
    robot_speed=1.0,
    service_time=DEFAULT_SERVICE_TIME,
) -> tuple[float, float, dict[str, list[int]]]:
    """Return makespan, total path distance, and per-robot task sequences."""
    validate_tasks(tasks)
    task_ids = {task["id"] for task in tasks}
    if hasattr(distance, "begin_prediction"):
        distance.begin_prediction()
    if set(assignment) != task_ids:
        raise ValueError("Assignment must contain every task ID exactly once.")

    robots = list(
        dict.fromkeys(robot for task in tasks for robot in task["candidate"])
    )
    robot_plans = {robot: [] for robot in robots}
    robot_available = {robot: 0.0 for robot in robots}
    robot_positions = {robot: None for robot in robots}
    finish_times = {}
    total_distance = 0.0

    for task in _topological_tasks(tasks):
        task_id = task["id"]
        robot = assignment[task_id]
        if robot not in task["candidate"]:
            raise ValueError(f"{robot} is not a candidate for task {task_id}.")

        travel_distance, endpoint = _route(
            distance,
            robot,
            _task_location(task),
            robot_positions[robot],
        )
        if not math.isfinite(travel_distance) or travel_distance < 0:
            raise ValueError(f"Invalid path distance for task {task_id}.")

        speed = _robot_value(robot_speed, robot, "robot_speed")
        duration = _task_service_duration(task, service_time, robot)
        dependency_time = max(
            (finish_times[dependency] for dependency in task["node"]),
            default=0.0,
        )
        finish_time = max(robot_available[robot], dependency_time) + travel_distance / speed + duration

        finish_times[task_id] = finish_time
        robot_available[robot] = finish_time
        robot_positions[robot] = endpoint
        robot_plans[robot].append(task_id)
        if hasattr(distance, "finish_predicted_task"):
            distance.finish_predicted_task(task)
        total_distance += travel_distance

    makespan = max(finish_times.values(), default=0.0)
    return makespan, total_distance, robot_plans


def optimize(
    candidates,
    evaluator,
    makespan_tolerance=MAKESPAN_TOLERANCE,
) -> dict:
    """Select a candidate by ``(makespan, extra distance, delay)``.

    This is the single selection policy shared by offline assignment and
    online motion scheduling. ``evaluator`` returns an estimated_makespan,
    total_distance, and optional total_delay for one candidate.  Online
    evaluators may additionally expose ``additional_distance``; this is the
    paper's :math:`\\Delta D` and is preferred to absolute remaining travel.

    ``makespan_tolerance`` is retained only for explicit ablations.  The
    paper-facing default is zero, hence exact lexicographic optimization.
    """
    if not 0.0 <= makespan_tolerance < 1.0:
        raise ValueError("makespan_tolerance must be in [0, 1).")

    evaluated = []
    for index, candidate in enumerate(candidates):
        try:
            metrics = dict(evaluator(candidate))
        except RuntimeError:
            continue
        makespan = float(
            metrics.get("estimated_makespan", metrics.get("makespan", math.nan))
        )
        distance = float(
            metrics.get(
                "additional_distance",
                metrics.get("total_distance", math.nan),
            )
        )
        delay = float(metrics.get("total_delay", 0.0))
        if any(
            math.isnan(value) or value < 0
            for value in (makespan, distance, delay)
        ):
            raise ValueError(
                "Candidate evaluation requires non-negative metrics."
            )
        if any(not math.isfinite(value) for value in (makespan, distance, delay)):
            continue
        metrics["estimated_makespan"] = makespan
        metrics["total_distance"] = float(
            metrics.get("total_distance", distance)
        )
        metrics["selection_distance"] = distance
        metrics["total_delay"] = delay
        evaluated.append((candidate, metrics, index))

    if not evaluated:
        raise RuntimeError("No feasible candidate plan was produced.")

    minimum_makespan = min(item[1]["estimated_makespan"] for item in evaluated)
    threshold = minimum_makespan * (1.0 + makespan_tolerance)
    near_optimal = [
        item
        for item in evaluated
        if item[1]["estimated_makespan"] <= threshold
    ]
    candidate, evaluation, _ = min(
        near_optimal,
        key=lambda item: (
            item[1]["selection_distance"],
            item[1]["total_delay"],
            item[1]["estimated_makespan"],
            item[2],
        ),
    )
    return {"candidate": candidate, "evaluation": evaluation}


def _linear_task_chains(
    tasks: list[dict],
    ordered_tasks: list[dict] | None = None,
    successors: dict[int, list[int]] | None = None,
) -> list[tuple[list[int], list[str]]]:
    """Build maximal one-in/one-out DAG chains with common capable robots."""
    ordered = ordered_tasks or _topological_tasks(tasks)
    task_by_id = {task["id"]: task for task in tasks}
    if successors is None:
        successors = {task["id"]: [] for task in tasks}
        for task in tasks:
            for predecessor in task["node"]:
                successors[predecessor].append(task["id"])

    visited = set()
    chains = []
    for task in ordered:
        task_id = task["id"]
        if task_id in visited:
            continue
        chain = [task_id]
        candidates = list(task["candidate"])
        visited.add(task_id)
        current = task_id
        while len(successors[current]) == 1:
            successor = successors[current][0]
            successor_task = task_by_id[successor]
            if successor in visited or len(successor_task["node"]) != 1:
                break
            common = [
                robot
                for robot in candidates
                if robot in successor_task["candidate"]
            ]
            if not common:
                break
            chain.append(successor)
            candidates = common
            visited.add(successor)
            current = successor
        chains.append((chain, candidates))
    return chains


def _assignment_search_metadata(tasks: list[dict], service_time) -> dict:
    """Precompute immutable DAG, chain, duration, and search-order metadata."""
    ordered_tasks = _topological_tasks(tasks)
    task_by_id = {task["id"]: task for task in tasks}
    successors = {task["id"]: [] for task in tasks}
    for task in tasks:
        for predecessor in task["node"]:
            successors[predecessor].append(task["id"])

    service_durations = {
        task["id"]: {
            robot: _task_service_duration(task, service_time, robot)
            for robot in task["candidate"]
        }
        for task in tasks
    }
    critical_depth = {}
    for task in reversed(ordered_tasks):
        task_id = task["id"]
        critical_depth[task_id] = 1 + max(
            (critical_depth[successor] for successor in successors[task_id]),
            default=0,
        )

    raw_chains = _linear_task_chains(
        tasks,
        ordered_tasks=ordered_tasks,
        successors=successors,
    )
    chains = []
    for original_index, (task_ids, candidates) in enumerate(raw_chains):
        chains.append(
            {
                "task_ids": task_ids,
                "tasks": [task_by_id[task_id] for task_id in task_ids],
                "candidates": candidates,
                "duration_by_robot": {
                    robot: sum(
                        service_durations[task_id][robot]
                        for task_id in task_ids
                    )
                    for robot in candidates
                },
                "criticality": critical_depth[task_ids[0]],
                "original_index": original_index,
            }
        )

    # Fail-first ordering: constrained chains first, then the longest/most
    # critical ones. Original indices remain the final deterministic tie-break.
    chains.sort(
        key=lambda chain: (
            len(chain["candidates"]),
            -chain["criticality"],
            -len(chain["task_ids"]),
            chain["original_index"],
        )
    )
    return {
        "ordered_tasks": ordered_tasks,
        "task_by_id": task_by_id,
        "successors": successors,
        "service_durations": service_durations,
        "chains": chains,
    }


def _chain_object_names(chain: dict) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            task["object"].strip().casefold()
            for task in chain["tasks"]
            if task["action"] in {"pick", "place"}
        )
    )


def _coalesce_owner_chains(metadata):
    """Branch once for chains already forced equal by pick/place ownership."""
    chains = metadata["chains"]
    parents = list(range(len(chains)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    owners = {}
    for i, chain in enumerate(chains):
        for name in _chain_object_names(chain):
            if name in owners:
                parents[root(i)] = root(owners[name])
            owners[name] = i
    groups = {}
    for i, chain in enumerate(chains):
        groups.setdefault(root(i), []).append(chain)
    result = []
    for group in groups.values():
        task_ids = {tid for chain in group for tid in chain["task_ids"]}
        candidates = [r for r in group[0]["candidates"]
                      if all(r in c["candidates"] for c in group)]
        if not candidates:
            raise RuntimeError("No eligible common robot for object-owner chains")
        group_tasks = [t for t in metadata["ordered_tasks"] if t["id"] in task_ids]
        result.append(dict(task_ids=[t["id"] for t in group_tasks], tasks=group_tasks,
            candidates=candidates,
            duration_by_robot={r:sum(metadata["service_durations"][t["id"]][r] for t in group_tasks)
                               for r in candidates},
            criticality=max(c["criticality"] for c in group),
            original_index=min(c["original_index"] for c in group)))
    result.sort(key=lambda c:(len(c["candidates"]),-c["criticality"],-len(c["task_ids"]),c["original_index"]))
    return result


def _greedy_assignment(metadata: dict) -> dict[int, str]:
    """Build a fast load-balanced feasible assignment to seed the incumbent."""
    assignment = {}
    robot_load = {}
    object_owners = {}
    for chain in metadata["chains"]:
        objects = _chain_object_names(chain)
        feasible = [
            robot
            for robot in chain["candidates"]
            if all(
                object_owners.get(object_name, robot) == robot
                for object_name in objects
            )
        ]
        if not feasible:
            raise RuntimeError("No handoff-free greedy chain assignment exists")
        candidate_order = {
            robot: index for index, robot in enumerate(chain["candidates"])
        }
        robot = min(
            feasible,
            key=lambda value: (
                robot_load.get(value, 0.0)
                + chain["duration_by_robot"][value],
                candidate_order[value],
            ),
        )
        assignment.update(
            {task_id: robot for task_id in chain["task_ids"]}
        )
        robot_load[robot] = (
            robot_load.get(robot, 0.0)
            + chain["duration_by_robot"][robot]
        )
        for object_name in objects:
            object_owners.setdefault(object_name, robot)
    return assignment


def optimize_task_assignments(
    tasks: list[dict],
    distance,
    robot_speed=1.0,
    service_time=DEFAULT_SERVICE_TIME,
    search_mode="chain_bnb",
    diagnostics=None,
) -> dict:
    """Optimal global task assignment with the C++ DFS Branch-and-Bound over task chains (Chain-BnB).

    search_mode="atomic_bnb" runs the per-task Atomic-BnB of the same native module (comparison only).
    Requires the native extension: python -m pip install ./native/chain_bnb
    """
    from .assignment_cpp import optimize_atomic_bnb_cpp, optimize_chain_bnb_cpp
    if search_mode in {"chain_bnb", "chain_bnb_cpp"}:
        solver = optimize_chain_bnb_cpp
    elif search_mode in {"atomic_bnb", "atomic_bnb_cpp", "task_bnb", "task_bnb_cpp"}:
        solver = optimize_atomic_bnb_cpp
    else:
        raise ValueError(f"Unknown assignment search mode: {search_mode}")
    return solver(tasks, distance, robot_speed=robot_speed, service_time=service_time, diagnostics=diagnostics)


def save_plan(plan: dict, output_file) -> Path:
    """Save an optimized plan as UTF-8 JSON."""
    path = Path(output_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
