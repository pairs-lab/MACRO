"""Search and evaluate multi-robot task assignments."""

import heapq
import json
import math
from functools import lru_cache
from collections.abc import Mapping
from pathlib import Path

DEFAULT_SERVICE_TIME = 1.0
MAKESPAN_TOLERANCE = 0.0
BOUND_FLOAT_TOLERANCE = 1e-9


def _assignment_bound_prunes(
    lower_bound: tuple[float, float],
    incumbent: tuple[float, float],
    tolerance: float = BOUND_FLOAT_TOLERANCE,
) -> bool:
    """Apply the assignment objective's tolerant lexicographic cutoff."""
    return (
        lower_bound[0] > incumbent[0] + tolerance
        or (
            abs(lower_bound[0] - incumbent[0]) <= tolerance
            and lower_bound[1] >= incumbent[1] - tolerance
        )
    )


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


def _assignment_lower_bound(
    ordered_tasks: list[dict],
    partial_assignment: dict[int, str],
    service_durations: dict[int, dict[str, float]],
    *,
    robot_speed=1.0,
    task_positions: dict[int, tuple] | None = None,
    initial_positions: dict[str, tuple] | None = None,
    chain_predecessors: dict[int, int] | None = None,
    distance_scale: float = 1.0,
) -> tuple[float, float]:
    """Return admissible ``(makespan, travel-distance)`` lower bounds.

    Spatial terms use straight-line distances only. The precedence bound
    charges navigation only where chain construction guarantees it: from a
    source robot pose into a source task, or between adjacent tasks in one
    handoff-free chain. The committed-workload bound additionally connects
    assigned tasks in stable topological order. Taking maxima of independently
    admissible bounds avoids double-counting. Missing geometry contributes
    zero rather than a fabricated path cost.
    """
    task_positions = task_positions or {}
    initial_positions = initial_positions or {}
    chain_predecessors = chain_predecessors or {}

    def set_distance(first_points, second_points):
        if not first_points or not second_points:
            return 0.0
        try:
            return min(
                math.dist(first, second) * distance_scale
                for first in first_points
                for second in second_points
            )
        except (TypeError, ValueError):
            return 0.0

    def guaranteed_navigation(task, robot):
        task_id = task["id"]
        destination = task_positions.get(task_id, ())
        predecessor = chain_predecessors.get(task_id)
        if predecessor is not None:
            return set_distance(task_positions.get(predecessor, ()), destination)
        if task.get("node"):
            return 0.0
        return set_distance(initial_positions.get(robot, ()), destination)

    finish_times = {}
    assigned_service = {}
    assigned_tasks = {}
    guaranteed_distances = []
    for task in ordered_tasks:
        task_id = task["id"]
        assigned_robot = partial_assignment.get(task_id)
        candidate_robots = (
            (assigned_robot,)
            if assigned_robot is not None
            else tuple(service_durations[task_id])
        )
        duration = min(
            service_durations[task_id][robot]
            + guaranteed_navigation(task, robot)
            / _robot_value(robot_speed, robot, "robot_speed")
            for robot in candidate_robots
        )
        dependency_finish = max(
            (finish_times[predecessor] for predecessor in task["node"]),
            default=0.0,
        )
        finish_times[task_id] = dependency_finish + duration
        if assigned_robot is not None:
            assigned_service[assigned_robot] = (
                assigned_service.get(assigned_robot, 0.0)
                + service_durations[task_id][assigned_robot]
            )
            assigned_tasks.setdefault(assigned_robot, []).append(task_id)
        predecessor = chain_predecessors.get(task_id)
        if predecessor is not None:
            guaranteed_distances.append(
                set_distance(
                    task_positions.get(predecessor, ()),
                    task_positions.get(task_id, ()),
                )
            )

    assigned_distance = {}
    for robot, assigned_task_ids in assigned_tasks.items():
        previous_points = initial_positions.get(robot, ())
        travel = 0.0
        for task_id in assigned_task_ids:
            current_points = task_positions.get(task_id, ())
            travel += set_distance(previous_points, current_points)
            if current_points:
                previous_points = current_points
        assigned_distance[robot] = travel

    source_distance = max(
        (
            min(
                set_distance(
                    initial_positions.get(robot, ()),
                    task_positions.get(task["id"], ()),
                )
                for robot in task["candidate"]
            )
            for task in ordered_tasks
            if not task.get("node")
        ),
        default=0.0,
    )
    distance_bound = max(
        max(guaranteed_distances, default=0.0),
        sum(assigned_distance.values()),
        source_distance,
    )
    workload_bound = max(
        (
            assigned_service[robot]
            + assigned_distance.get(robot, 0.0)
            / _robot_value(robot_speed, robot, "robot_speed")
            for robot in assigned_service
        ),
        default=0.0,
    )
    makespan_bound = max(
        max(finish_times.values(), default=0.0),
        workload_bound,
    )
    return makespan_bound, distance_bound


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


def _prepare_assignment_bound(metadata, geometry, robot_speed, chain_predecessors):
    """Compile the existing admissible bound's immutable geometry/durations.

    This computes the same bound as _assignment_lower_bound, without repeated
    point-set distance minimizations and robot parameter validation per node.
    """
    tasks = metadata["ordered_tasks"]
    positions = geometry.get("task_positions", {})
    starts = geometry.get("initial_positions", {})
    scale = float(geometry.get("distance_scale", 1.0))
    robots = list(dict.fromkeys(r for t in tasks for r in t["candidate"]))
    speeds = {r:_robot_value(robot_speed,r,"robot_speed") for r in robots}

    def separation(a,b):
        if not a or not b:
            return 0.0
        try:
            return min(math.dist(x,y)*scale for x in a for y in b)
        except (TypeError,ValueError):
            return 0.0

    pair = {(a["id"],b["id"]):separation(positions.get(a["id"]),positions.get(b["id"]))
            for a in tasks for b in tasks}
    first = {(r,t["id"]):separation(starts.get(r),positions.get(t["id"]))
             for r in robots for t in tasks}
    chain_distance = max((pair[p,tid] for tid,p in chain_predecessors.items()),default=0.0)
    source_distance = max((min(first[r,t["id"]] for r in t["candidate"])
                           for t in tasks if not t["node"]),default=0.0)
    compiled = []
    for t in tasks:
        tid = t["id"]
        duration = metadata["service_durations"][tid]
        pred = chain_predecessors.get(tid)
        bound_duration = {r:duration[r]+(pair[pred,tid] if pred is not None
                           else 0.0 if t["node"] else first[r,tid])/speeds[r]
                          for r in t["candidate"]}
        compiled.append((tid,t["node"],duration,bound_duration,min(bound_duration.values())))

    def bound(assignment):
        finish, service, travel, previous = {}, {}, {}, {}
        for tid, parents, durations, bound_durations, optimistic in compiled:
            robot = assignment.get(tid)
            finish[tid] = max((finish[p] for p in parents),default=0.0) + (
                optimistic if robot is None else bound_durations[robot])
            if robot is not None:
                service[robot] = service.get(robot,0.0)+durations[robot]
                origin = previous.get(robot)
                travel[robot] = travel.get(robot,0.0)+(first[robot,tid] if origin is None else pair[origin,tid])
                if positions.get(tid):
                    previous[robot] = tid
        workload = max((s+travel[r]/speeds[r] for r,s in service.items()),default=0.0)
        return max(max(finish.values(),default=0.0),workload), max(chain_distance,sum(travel.values()),source_distance)
    return bound


def _prepare_exact_prefix_bound(metadata, route, robot_speed):
    """Incrementally evaluate only the committed stable-topological prefix.

    Every completion retains this prefix's assignment, navigation and timing.
    Consequently its makespan/distance are admissible lower bounds. Tasks
    after the first unassigned task are deliberately not evaluated as a
    prefix: their previous robot endpoint may still change.
    """
    tasks = metadata["ordered_tasks"]
    robots = list(dict.fromkeys(r for t in tasks for r in t["candidate"]))
    index = {r:i for i,r in enumerate(robots)}
    task_index = {t["id"]:i for i,t in enumerate(tasks)}
    speeds = {r:_robot_value(robot_speed,r,"robot_speed") for r in robots}

    @lru_cache(maxsize=4096)
    def prefix_state(prefix):
        if not prefix:
            return (), (0.0,)*len(robots), (None,)*len(robots), 0.0, 0.0
        finish, available, positions, distance, makespan = prefix_state(prefix[:-1])
        task = tasks[len(prefix)-1]
        robot = prefix[-1]
        ri = index[robot]
        travel, endpoint = route(task["id"],robot,positions[ri])
        dependency = max((finish[task_index[p]] for p in task["node"]),default=0.0)
        f = max(available[ri],dependency) + travel/speeds[robot] + metadata["service_durations"][task["id"]][robot]
        available, positions = list(available), list(positions)
        available[ri], positions[ri] = f, endpoint
        return (*finish,f),tuple(available),tuple(positions),distance+travel,max(makespan,f)

    def bound(assignment):
        prefix = []
        for t in tasks:
            if t["id"] not in assignment:
                break
            prefix.append(assignment[t["id"]])
        try:
            state = prefix_state(tuple(prefix))
        except RuntimeError:
            return math.inf, math.inf
        return state[4],state[3]
    bound.cache_info = prefix_state.cache_info
    return bound


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


def _branch_and_bound_assignments(
    tasks: list[dict],
    evaluate_assignment,
    service_time,
    distance=None,
    robot_speed=1.0,
    search_mode="chain_bnb",
    diagnostics=None,
):
    """Yield globally assigned task chains using makespan lower-bound pruning."""
    metadata = _assignment_search_metadata(tasks, service_time)
    chains = metadata["chains"]
    original_chains = chains
    if search_mode not in {"chain_bnb", "task_bnb", "greedy", "exhaustive"}:
        raise ValueError(f"Unknown assignment search mode: {search_mode}")
    if search_mode == "task_bnb":
        chains = [{**chain, "task_ids": [task["id"]], "tasks": [task],
                   "duration_by_robot": metadata["service_durations"][task["id"]],
                   "group_key": ("chain", index)}
                  for index, chain in enumerate(original_chains) for task in chain["tasks"]]
    elif search_mode == "chain_bnb":
        chains = _coalesce_owner_chains(metadata)
    raw_combinations = math.prod(len(task["candidate"]) for task in tasks)
    chain_combinations = math.prod(
        len(chain["candidates"]) for chain in chains
    )
    stats = diagnostics if diagnostics is not None else {}
    stats.update({
        "task_chain_count": len(chains),
        "chain_order": [chain["task_ids"] for chain in chains],
        "raw_combinations": raw_combinations,
        "chain_combinations": chain_combinations,
        "expanded_nodes": 0,
        "bound_pruned": 0,
        "distance_bound_pruned": 0,
        "handoff_pruned": 0,
        "complete_evaluated": 0,
        "feasible_complete_evaluated": 0,
        "greedy_evaluated": 0,
        "duplicate_complete_skipped": 0,
    })
    assignment = {}
    assigned_load = {}
    object_owners = {}
    incumbent = (math.inf, math.inf)
    evaluated = []
    evaluated_keys = set()
    task_ids = [task["id"] for task in tasks]
    candidate_rank = {t["id"]:{r:i for i,r in enumerate(t["candidate"])} for t in tasks}
    best_score = None
    geometry = {}
    if distance is not None and hasattr(distance, "optimistic_geometry"):
        geometry = dict(distance.optimistic_geometry(metadata["ordered_tasks"]))
    chain_predecessors = {}
    for chain in original_chains:
        chain_predecessors.update(
            dict(zip(chain["task_ids"][1:], chain["task_ids"][:-1]))
        )
    prepared_bound = _prepare_assignment_bound(metadata, geometry, robot_speed, chain_predecessors)
    stats["original_chain_count"] = len(original_chains)
    stats["implementation_revision"] = "cached-bound-owner-groups-exact-prefix-v2"
    stats["owner_groups_merged"] = len(original_chains)-len(chains) if search_mode == "chain_bnb" else 0

    def evaluate_complete(candidate, *, greedy=False):
        nonlocal incumbent, best_score
        key = tuple(candidate[task_id] for task_id in task_ids)
        if key in evaluated_keys:
            stats["duplicate_complete_skipped"] += 1
            return
        # DFS visits each complete assignment once. Only heuristic seeds can
        # recur at a leaf; keeping every leaf key wastes O(leaves * tasks).
        if greedy:
            evaluated_keys.add(key)
        if greedy:
            stats["greedy_evaluated"] += 1
        else:
            stats["complete_evaluated"] += 1
        try:
            metrics = evaluate_assignment(candidate)
        except RuntimeError:
            return
        stats["feasible_complete_evaluated"] += 1
        cost = (
            float(metrics["estimated_makespan"]),
            float(metrics["total_distance"]),
        )
        if (
            cost[0] < incumbent[0] - BOUND_FLOAT_TOLERANCE
            or (
                abs(cost[0] - incumbent[0]) <= BOUND_FLOAT_TOLERANCE
                and cost[1] < incumbent[1] - BOUND_FLOAT_TOLERANCE
            )
        ):
            incumbent = cost
        score = (*cost, tuple(candidate_rank[tid][candidate[tid]] for tid in task_ids))
        if best_score is None or score < best_score:
            best_score = score
            evaluated[:] = [(candidate.copy(), metrics)]

    # A cheap complete solution makes pruning effective from the root rather
    # than waiting for the first depth-first leaf.
    try:
        greedy_assignment = _greedy_assignment(metadata)
    except RuntimeError:
        # Greedy is only an incumbent heuristic. A locally unfortunate object
        # owner must not prevent the exact DFS from finding a feasible plan.
        greedy_assignment = None
    if greedy_assignment is not None:
        evaluate_complete(greedy_assignment, greedy=True)

    def search(chain_index):
        if chain_index == len(chains):
            evaluate_complete(assignment)
            return

        chain = chains[chain_index]
        objects = _chain_object_names(chain)
        if "group_key" in chain:
            objects = (*objects, chain["group_key"])
        candidate_order = {
            robot: index for index, robot in enumerate(chain["candidates"])
        }
        candidates = sorted(
            chain["candidates"],
            key=lambda robot: (
                assigned_load.get(robot, 0.0)
                + chain["duration_by_robot"][robot],
                candidate_order[robot],
            ),
        )
        for robot in candidates:
            claimed_objects = []
            violates_handoff = False
            for object_name in objects:
                owner = object_owners.get(object_name)
                if owner is not None and owner != robot:
                    violates_handoff = True
                    break
                if owner is None:
                    object_owners[object_name] = robot
                    claimed_objects.append(object_name)
            if violates_handoff:
                stats["handoff_pruned"] += 1
                for object_name in claimed_objects:
                    object_owners.pop(object_name, None)
                continue

            assignment.update(
                {task_id: robot for task_id in chain["task_ids"]}
            )
            previous_load = assigned_load.get(robot, 0.0)
            assigned_load[robot] = (
                previous_load + chain["duration_by_robot"][robot]
            )
            stats["expanded_nodes"] += 1
            lower_bound = prepared_bound(assignment)
            if search_mode == "chain_bnb" and hasattr(evaluate_assignment, "prefix_bound"):
                prefix_bound = evaluate_assignment.prefix_bound(assignment)
                lower_bound = (max(lower_bound[0],prefix_bound[0]),max(lower_bound[1],prefix_bound[1]))
            prune = _assignment_bound_prunes(lower_bound, incumbent)
            if search_mode != "exhaustive" and prune:
                stats["bound_pruned"] += 1
                if (
                    abs(lower_bound[0] - incumbent[0])
                    <= BOUND_FLOAT_TOLERANCE
                    and lower_bound[1]
                    >= incumbent[1] - BOUND_FLOAT_TOLERANCE
                ):
                    stats["distance_bound_pruned"] += 1
            else:
                search(chain_index + 1)
            for task_id in chain["task_ids"]:
                assignment.pop(task_id, None)
            if previous_load:
                assigned_load[robot] = previous_load
            else:
                assigned_load.pop(robot, None)
            for object_name in claimed_objects:
                object_owners.pop(object_name, None)

    if search_mode != "greedy":
        search(0)
    if search_mode == "chain_bnb" and hasattr(evaluate_assignment, "prefix_bound"):
        cache_info = evaluate_assignment.prefix_bound.cache_info()
        stats["exact_prefix_cache_hits"] = cache_info.hits
        stats["exact_prefix_cache_misses"] = cache_info.misses
    return evaluated, stats


def optimize_task_assignments(
    tasks: list[dict],
    distance,
    robot_speed=1.0,
    service_time=DEFAULT_SERVICE_TIME,
    search_mode="chain_bnb",
    diagnostics=None,
) -> dict:
    """Globally assign task chains with DFS Branch-and-Bound."""
    if search_mode == "chain_bnb_cpp":
        from .assignment_cpp import optimize_chain_bnb_cpp
        return optimize_chain_bnb_cpp(
            tasks, distance, robot_speed=robot_speed,
            service_time=service_time, diagnostics=diagnostics)
    if search_mode in {"task_bnb_cpp", "atomic_bnb_cpp"}:
        from .assignment_cpp import optimize_atomic_bnb_cpp
        return optimize_atomic_bnb_cpp(
            tasks, distance, robot_speed=robot_speed,
            service_time=service_time, diagnostics=diagnostics)
    validate_tasks(tasks)
    fast_routes = None
    if hasattr(distance, "prepare_assignment_routes"):
        evaluation_metadata = _assignment_search_metadata(tasks, service_time)
        fast_routes = distance.prepare_assignment_routes(evaluation_metadata["ordered_tasks"])
        evaluation_robots = list(dict.fromkeys(r for t in tasks for r in t["candidate"]))
        evaluation_speeds = {r:_robot_value(robot_speed,r,"robot_speed") for r in evaluation_robots}

    def evaluate_assignment(assignment):
        if fast_routes is None:
            makespan, total_distance, robot_plans = evaluate_plan(
                tasks, assignment, distance, robot_speed=robot_speed, service_time=service_time)
        else:
            # Prediction evolves with the fixed topological task order, not
            # the assignment. Reuse frozen destinations and validated scalar
            # parameters; retain the original floating-point operation order.
            finish, available, positions = {}, {}, {}
            robot_plans = {r:[] for r in evaluation_robots}
            total_distance = 0.0
            for task in evaluation_metadata["ordered_tasks"]:
                tid = task["id"]
                robot = assignment[tid]
                travel, endpoint = fast_routes(tid,robot,positions.get(robot))
                if not math.isfinite(travel) or travel < 0:
                    raise ValueError(f"Invalid path distance for task {tid}")
                finish[tid] = max(available.get(robot,0.0),
                    max((finish[p] for p in task["node"]),default=0.0)) + travel/evaluation_speeds[robot] + evaluation_metadata["service_durations"][tid][robot]
                available[robot] = finish[tid]
                positions[robot] = endpoint
                robot_plans[robot].append(tid)
                total_distance += travel
            makespan = max(finish.values(),default=0.0)
        return {
            "estimated_makespan": makespan,
            "total_distance": total_distance,
            "total_delay": 0.0,
            "robot_plans": robot_plans,
        }

    if fast_routes is not None:
        evaluate_assignment.prefix_bound = _prepare_exact_prefix_bound(
            evaluation_metadata,fast_routes,robot_speed)

    evaluated, search_stats = _branch_and_bound_assignments(
        tasks,
        evaluate_assignment,
        service_time,
        distance=distance,
        robot_speed=robot_speed,
        search_mode=search_mode,
        diagnostics=diagnostics,
    )
    selected = optimize(
        evaluated,
        evaluator=lambda item: item[1],
    )
    selected_assignment, _ = selected["candidate"]
    evaluation = selected["evaluation"]

    return {
        "makespan": evaluation["estimated_makespan"],
        "total_distance": evaluation["total_distance"],
        "robot_plans": evaluation["robot_plans"],
        "assignment": selected_assignment,
        "assignment_search": search_stats,
        "tasks": [
            {
                "id": task["id"],
                "action": task["action"],
                "object": task["object"],
                "target": task.get("target", ""),
                "node": task["node"].copy(),
            }
            for task in tasks
        ],
    }


def save_plan(plan: dict, output_file) -> Path:
    """Save an optimized plan as UTF-8 JSON."""
    path = Path(output_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
