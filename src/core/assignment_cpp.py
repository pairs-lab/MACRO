"""C++ Chain-BnB (and Atomic-BnB, for comparison) task assignment with frozen endpoint-aware transitions.

Only the search is native. DAG/capability/ownership metadata and navigation are compiled here in Python, once, into
endpoint-state transition arcs; no Python navigation callbacks are made in the native DFS.
"""
from dataclasses import dataclass
import math
import time

from .assignment import (
    DEFAULT_SERVICE_TIME, _assignment_search_metadata, _coalesce_owner_chains, _chain_object_names,
    _greedy_assignment, _robot_value, _route, _task_location, evaluate_plan, validate_tasks,
)


@dataclass(frozen=True)
class NavigationArc:
    robot: str
    previous: tuple  # (previous task ID, endpoint), or (None, None)
    current: tuple   # (current task ID, endpoint)
    distance: float
    duration: float  # navigation + service


def compile_navigation(tasks, distance, robot_speed=1.0, service_time=None,
                       restrict_chain_transitions=False):
    """Enumerate all deterministic oracle transitions, without assigning tasks.

    Keeping endpoint state is essential: an interaction pose can depend on
    the robot's previous endpoint. A plain object-to-object distance matrix
    would silently give the MILP different navigation choices from BnB.
    """
    validate_tasks(tasks)
    metadata = _assignment_search_metadata(tasks, service_time)
    robots = list(dict.fromkeys(r for t in tasks for r in t["candidate"]))
    states = {r: [(None, None)] for r in robots}
    arcs = []
    # Optional native-backend preparation optimization. A group already has
    # one mandatory owner; never plan a transition that skips its intervening
    # tasks on that robot. MILP's default compiler remains unchanged.
    task_index = {t["id"]: i for i, t in enumerate(metadata["ordered_tasks"])}
    previous_required, next_required, eligible = {}, {}, {}
    if restrict_chain_transitions:
        for group in _coalesce_owner_chains(metadata):
            tids = group["task_ids"]
            for i, tid in enumerate(tids):
                previous_required[tid] = task_index[tids[i-1]] if i else -1
                next_required[tid] = task_index[tids[i+1]] if i+1 < len(tids) else len(tasks)
                eligible[tid] = group["candidates"]
    if hasattr(distance, "begin_prediction"):
        distance.begin_prediction()
    for task in metadata["ordered_tasks"]:
        for robot in task["candidate"]:
            if restrict_chain_transitions and robot not in eligible[task["id"]]:
                continue
            new_states = set()
            for previous in tuple(states[robot]):
                if restrict_chain_transitions:
                    previous_index = task_index[previous[0]] if previous[0] is not None else -1
                    if previous_index < previous_required[task["id"]]:
                        continue
                    if (previous[0] is not None
                            and next_required[previous[0]] < task_index[task["id"]]):
                        continue
                try:
                    cost, endpoint = _route(
                        distance, robot, _task_location(task), previous[1]
                    )
                except RuntimeError:
                    # BnB also excludes assignments whose navigation fails.
                    continue
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("Navigation requires finite nonnegative costs")
                if isinstance(endpoint, list):
                    endpoint = tuple(endpoint)
                current = (task["id"], endpoint)
                new_states.add(current)
                arcs.append(NavigationArc(
                    robot, previous, current, float(cost),
                    cost / _robot_value(robot_speed, robot, "robot_speed")
                    + metadata["service_durations"][task["id"]][robot],
                ))
            states[robot].extend(sorted(new_states, key=repr))
        if hasattr(distance, "finish_predicted_task"):
            distance.finish_predicted_task(task)
    return metadata, states, arcs


CPP_REVISION = "bnb-cpp-chain-atomic-v2"


def require_native():
    try:
        import _chain_bnb_native
    except ImportError as error:
        raise RuntimeError(
            "C++ BnB backend is not built for this Python. Run: "
            "python -m pip install ./native/chain_bnb"
        ) from error
    if _chain_bnb_native.revision != CPP_REVISION:
        raise RuntimeError("Native backend revision changed; rebuild with python -m pip install --force-reinstall ./native/chain_bnb")
    return _chain_bnb_native


def prepare_native_problem(tasks, distance, robot_speed=1.0,
                           service_time=DEFAULT_SERVICE_TIME, search_mode="chain_bnb"):
    if search_mode not in {"chain_bnb", "task_bnb"}:
        raise ValueError(f"Unknown native search mode: {search_mode}")
    metadata, states, arcs = compile_navigation(
        tasks, distance, robot_speed, service_time, restrict_chain_transitions=True)
    ordered = metadata["ordered_tasks"]
    task_index = {t["id"]: i for i, t in enumerate(ordered)}
    robots = list(states)
    robot_index = {r: i for i, r in enumerate(robots)}
    owner_keys, owner_index = [], {}
    if search_mode == "chain_bnb":
        groups = _coalesce_owner_chains(metadata)
        owner_keys = [[] for _ in groups]
    else:
        # Reproduce Atomic Python's task units, original-chain ordering,
        # capability intersections and deferred ownership checks exactly.
        # Navigation preparation is common to both native modes, but Atomic
        # does not contract ownership-linked decision variables.
        groups = []
        for ci, chain in enumerate(metadata["chains"]):
            for t in chain["tasks"]:
                unit = dict(chain, task_ids=[t["id"]], tasks=[t],
                            duration_by_robot=metadata["service_durations"][t["id"]])
                groups.append(unit)
                keys = [("object", name) for name in _chain_object_names(unit)] + [("chain", ci)]
                owner_keys.append([owner_index.setdefault(key, len(owner_index)) for key in keys])
    speeds = [_robot_value(robot_speed, r, "robot_speed") for r in robots]
    candidates = [[robot_index[r] for r in t["candidate"]] for t in ordered]
    services = [[metadata["service_durations"][t["id"]].get(r, 0.) for r in robots]
                for t in ordered]
    state_index = {r: {s: i for i, s in enumerate(states[r])} for r in robots}
    routes = [[[(-1., -1) for _ in states[r]] for _ in ordered] for r in robots]
    for arc in arcs:
        routes[robot_index[arc.robot]][task_index[arc.current[0]]][
            state_index[arc.robot][arc.previous]] = (
                arc.distance, state_index[arc.robot][arc.current])

    geometry = (distance.optimistic_geometry(ordered)
                if hasattr(distance, "optimistic_geometry") else {})
    positions = geometry.get("task_positions", {})
    starts = geometry.get("initial_positions", {})
    scale = float(geometry.get("distance_scale", 1.))

    def separation(a, b):
        if not a or not b:
            return 0.
        try:
            return min(math.dist(x, y) * scale for x in a for y in b)
        except (TypeError, ValueError):
            return 0.

    pair = [[separation(positions.get(a["id"]), positions.get(b["id"]))
             for b in ordered] for a in ordered]
    first = [[separation(starts.get(r), positions.get(t["id"])) for t in ordered]
             for r in robots]
    predecessors = {}
    for chain in metadata["chains"]:
        predecessors.update(zip(chain["task_ids"][1:], chain["task_ids"][:-1]))
    bound_durations = []
    for i, t in enumerate(ordered):
        pred = predecessors.get(t["id"])
        bound_durations.append([
            services[i][r] + (pair[task_index[pred]][i] if pred is not None
                             else 0. if t["node"] else first[r][i]) / speeds[r]
            for r in range(len(robots))])
    try:
        seed = _greedy_assignment(metadata)
    except RuntimeError:
        seed = None
    problem = dict(
        parents=[[task_index[p] for p in t["node"]] for t in ordered],
        candidates=candidates, services=services, speeds=speeds, routes=routes,
        groups=[[task_index[tid] for tid in c["task_ids"]] for c in groups],
        group_owner_keys=owner_keys, owner_count=len(owner_index),
        use_exact_prefix=search_mode == "chain_bnb",
        group_candidates=[[robot_index[r] for r in c["candidates"]] for c in groups],
        group_durations=[[c["duration_by_robot"].get(r, 0.) for r in robots] for c in groups],
        pair=pair, first=first, has_positions=[bool(positions.get(t["id"])) for t in ordered],
        bound_durations=bound_durations,
        optimistic=[min(bound_durations[i][r] for r in candidates[i]) for i in range(len(ordered))],
        chain_distance=max((pair[task_index[p]][task_index[t]] for t, p in predecessors.items()), default=0.),
        source_distance=max((min(first[r][i] for r in candidates[i])
                             for i, t in enumerate(ordered) if not t["node"]), default=0.),
        tie_order=[task_index[t["id"]] for t in tasks],
        seed_present=seed is not None,
        seed=[robot_index[seed[t["id"]]] for t in ordered] if seed is not None else [],
    )
    return problem, metadata, groups, robots


def _optimize_bnb_cpp(tasks, distance, robot_speed=1.0,
                      service_time=DEFAULT_SERVICE_TIME, diagnostics=None,
                      time_limit=None, search_mode="chain_bnb"):
    native = require_native()
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be finite and positive")
    started = time.perf_counter()
    problem, metadata, groups, robots = prepare_native_problem(
        tasks, distance, robot_speed, service_time, search_mode=search_mode)
    stats = diagnostics if diagnostics is not None else {}
    stats.update(
        implementation_revision=CPP_REVISION, backend="cpp", search_mode=search_mode,
        task_chain_count=len(groups), original_chain_count=len(metadata["chains"]),
        owner_groups_merged=len(metadata["chains"])-len(groups) if search_mode == "chain_bnb" else 0,
        exact_prefix_bound=search_mode == "chain_bnb",
        chain_order=[c["task_ids"] for c in groups],
        raw_combinations=math.prod(len(t["candidate"]) for t in tasks),
        chain_combinations=math.prod(len(c["candidates"]) for c in groups),
        navigation_states=sum(len(row[0]) for row in problem["routes"] if row),
        navigation_arcs=sum(arc[1] >= 0 for rows in problem["routes"] for row in rows for arc in row),
        preparation_time=time.perf_counter()-started,
    )
    remaining = None if time_limit is None else time_limit-(time.perf_counter()-started)
    if remaining is not None and remaining <= 0:
        stats.update(status="time_limit_before_search", solved=False, expanded_nodes=0)
        raise TimeoutError("assignment_time_limit")
    result = native.solve(problem, -1. if remaining is None else remaining, stats)
    stats.update(result["stats"])
    stats["native_search_time"] = time.perf_counter()-started-stats["preparation_time"]
    if result["timed_out"]:
        stats.update(status="assignment_time_limit", solved=False)
        raise TimeoutError("assignment_time_limit")
    if not result["found"]:
        raise RuntimeError("No feasible candidate plan was produced.")
    assignment = {t["id"]: robots[r] for t, r in zip(metadata["ordered_tasks"], result["assignment"])}
    makespan, total_distance, plans = evaluate_plan(
        tasks, assignment, distance, robot_speed=robot_speed, service_time=service_time)
    # Preserve arithmetic order, not merely a numerical closeness check.
    if makespan != result["T"] or total_distance != result["D"]:
        raise RuntimeError("Native objectives disagree with the shared Python evaluator")
    if time_limit is not None and time.perf_counter()-started > time_limit:
        stats.update(status="assignment_time_limit", solved=False)
        raise TimeoutError("assignment_time_limit")
    stats.update(status="optimal", solved=True)
    return dict(makespan=makespan, total_distance=total_distance, robot_plans=plans,
                assignment=assignment, assignment_search=stats,
                tasks=[dict(id=t["id"], action=t["action"], object=t["object"],
                            target=t.get("target", ""), node=t["node"].copy()) for t in tasks])


def optimize_chain_bnb_cpp(tasks, distance, robot_speed=1.0,
                           service_time=DEFAULT_SERVICE_TIME, diagnostics=None,
                           time_limit=None):
    return _optimize_bnb_cpp(tasks, distance, robot_speed, service_time,
                            diagnostics, time_limit, search_mode="chain_bnb")


def optimize_atomic_bnb_cpp(tasks, distance, robot_speed=1.0,
                            service_time=DEFAULT_SERVICE_TIME, diagnostics=None,
                            time_limit=None):
    return _optimize_bnb_cpp(tasks, distance, robot_speed, service_time,
                            diagnostics, time_limit, search_mode="task_bnb")
