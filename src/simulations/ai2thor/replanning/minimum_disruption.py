"""DAG-aware AI2-THOR feasibility restoration for robot conflicts."""

from __future__ import annotations

import math
import time
from itertools import islice

from core import optimize
from simulations.ai2thor.config import (
    ONLINE_RELEASE_CANDIDATE_COUNT,
    ONLINE_REPLANNING_ROBOT_SPEED,
)


DEFAULT_MAX_CANDIDATES_PER_ROOT = 12
DEFAULT_MAX_EXPANDED_PER_ROOT = 64


class PlanningTimeoutError(TimeoutError):
    """Raised when one AI2-THOR conflict search exceeds its wall budget."""

    def __init__(self, message, stats=None):
        super().__init__(message)
        self.stats = dict(stats or {})


def path_length(path):
    return sum(
        math.hypot(second["x"] - first["x"], second["z"] - first["z"])
        for first, second in zip(path, path[1:])
    )


def _point_key(point):
    return (
        round(float(point.get("x", 0.0)), 6),
        round(float(point.get("z", 0.0)), 6),
    )


def _candidate_stable_key(candidate):
    """Deterministic non-DAG tie-break shared by the local policies."""
    releases = tuple(
        (
            int(release["blocker_agent"]),
            _point_key(release["release_position"]),
            tuple(_point_key(point) for point in release["release_path"]),
        )
        for release in candidate.get("release_moves", ())
    )
    paths = tuple(
        (
            int(agent_id),
            tuple(_point_key(point) for point in path),
        )
        for agent_id, path in sorted(candidate.get("current_paths", {}).items())
    )
    return (
        str(candidate.get("strategy", "")),
        int(candidate.get("priority_agent", -1)),
        releases,
        paths,
    )


def _freeze_cache_value(value):
    """Return a deterministic hashable representation for route inputs."""
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_cache_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(item) for item in value))
    if isinstance(value, float):
        return round(value, 9)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _clone_route(route):
    """Protect cached A* geometry from accidental candidate mutation."""
    cloned = dict(route)
    cloned["path"] = [dict(point) for point in route.get("path", ())]
    return cloned


def _memoize_route_planner(route_planner):
    """Cache exact A* queries for one frozen conflict snapshot.

    The wrapper changes neither feasibility nor route ordering. Its lifetime is
    one call to ``select_minimum_disruption_plan``, so live Unity state can
    never leak into a later conflict.
    """
    cache = {}
    statistics = {"route_cache_hits": 0, "route_cache_misses": 0}

    def cached(agent_id, context, position_overrides=None, avoid=True):
        key = (
            int(agent_id),
            _freeze_cache_value(context),
            _freeze_cache_value(position_overrides or {}),
            bool(avoid),
        )
        if key in cache:
            statistics["route_cache_hits"] += 1
            cached_value = cache[key]
            if isinstance(cached_value, BaseException):
                raise type(cached_value)(str(cached_value))
            return _clone_route(cached_value)
        statistics["route_cache_misses"] += 1
        try:
            route = route_planner(
                agent_id,
                context,
                position_overrides or {},
                avoid,
            )
        except RuntimeError as error:
            cache[key] = error
            raise
        cache[key] = _clone_route(route)
        return _clone_route(route)

    cached.statistics = statistics
    return cached


def _select_candidate(candidates, selection_policy, optimizer=optimize):
    """Apply the same information boundaries as the standalone policies."""
    if selection_policy == "dag_aware":
        return optimizer(
            candidates,
            evaluator=lambda candidate: candidate["evaluation"],
        )["candidate"]
    if selection_policy == "local_distance":
        return min(
            candidates,
            key=lambda candidate: (
                candidate["evaluation"]["local_resolution_distance"],
                _candidate_stable_key(candidate),
            ),
        )
    if selection_policy == "local_delay":
        return min(
            candidates,
            key=lambda candidate: (
                candidate["evaluation"]["local_resolution_delay"],
                _candidate_stable_key(candidate),
            ),
        )
    if selection_policy == "reroute_only":
        reroutes = [
            candidate for candidate in candidates
            if candidate["strategy"] == "short_replan"
        ]
        if not reroutes:
            raise RuntimeError("baseline_no_feasible_intervention")
        return min(
            reroutes,
            key=lambda candidate: (
                candidate["priority_path_length"],
                _candidate_stable_key(candidate),
            ),
        )
    raise ValueError(f"Unknown selection policy: {selection_policy}")


def _candidate_diagnostic(candidate, selected=False):
    evaluation = candidate["evaluation"]
    return {
        "candidate_id": candidate.get("candidate_id"),
        "strategy": candidate["strategy"],
        "priority_agent": candidate["priority_agent"],
        "moved_agents": [
            release["blocker_agent"]
            for release in candidate.get("release_moves", ())
        ],
        "additional_distance": evaluation.get(
            "additional_distance",
            evaluation["total_distance"],
        ),
        "local_resolution_distance": evaluation[
            "local_resolution_distance"
        ],
        "local_resolution_delay": evaluation["local_resolution_delay"],
        "estimated_makespan": evaluation.get("estimated_makespan", math.nan),
        "total_delay": evaluation.get("total_delay", math.nan),
        "selected": bool(selected),
    }


def _local_metrics(candidate, contexts, robot_speed):
    release_distance = sum(
        release["release_distance"]
        for release in candidate["release_moves"]
    )
    current_path_lengths = candidate["current_path_lengths"]
    priority_distance = current_path_lengths[candidate["priority_agent"]]
    release_time = release_distance / robot_speed
    task_times = {
        agent_id: distance / robot_speed
        + contexts[agent_id].get("action_duration", 0.0)
        for agent_id, distance in current_path_lengths.items()
    }
    priority_time = task_times[candidate["priority_agent"]]
    concurrent_resume = candidate.get("concurrent_resume_paths") is not None
    protected_time = (
        path_length(candidate["priority_release_prefix"]) / robot_speed
        if concurrent_resume else priority_time
    )
    yielding_navigation_wait = (
        protected_time if concurrent_resume else priority_distance / robot_speed
    )
    conflict_horizon = release_time + protected_time
    task_finish_times = {
        agent_id: (
            release_time + priority_time
            if agent_id == candidate["priority_agent"]
            else conflict_horizon + duration
        )
        for agent_id, duration in task_times.items()
    }
    delay_by_agent = {
        agent_id: (
            release_time
            if agent_id == candidate["priority_agent"]
            else release_time + yielding_navigation_wait
        )
        for agent_id in current_path_lengths
    }
    nominal_distance = sum(
        candidate.get("nominal_path_lengths", current_path_lengths).values()
    )
    local_resolution_distance = max(
        0.0,
        release_distance + sum(current_path_lengths.values())
        - nominal_distance,
    )
    return {
        # DAG-less fallback only. The runtime hook replaces this with the full
        # remaining-DAG earliest-start estimate.
        "estimated_makespan": max(task_finish_times.values(), default=0.0),
        "total_distance": release_distance + sum(current_path_lengths.values()),
        "additional_distance": local_resolution_distance,
        "total_delay": sum(delay_by_agent.values()),
        "delay_by_agent": delay_by_agent,
        # These two scores deliberately contain no task-DAG information.
        "local_resolution_distance": local_resolution_distance,
        # Hold only until the certified prefix clears; use the complete route
        # for the serial fallback. Neither case uses downstream DAG data.
        "local_resolution_delay": release_time + yielding_navigation_wait,
        "nominal_local_distance": nominal_distance,
        "estimator": (
            "release-prefix-concurrent fallback" if concurrent_resume
            else "serial release-priority fallback"
        ),
    }


def _log_dag_estimate(candidate, evaluation):
    """Print the candidate's temporary graph and earliest-start schedule."""
    print(f"    Candidate: {candidate['strategy']}")
    print(f"    Priority: R{candidate['priority_agent'] + 1}")
    release_durations = evaluation.get("release_durations", {})
    if release_durations:
        release_text = ", ".join(
            f"{node.replace('RELEASE_', '').rsplit('_', 1)[0]} "
            f"{duration:.2f}s"
            for node, duration in release_durations.items()
        )
        print(f"    Release: {release_text}")
    elif candidate["release_moves"]:
        release_text = ", ".join(
            f"R{release['blocker_agent'] + 1} "
            f"{release['release_distance'] / ONLINE_REPLANNING_ROBOT_SPEED:.2f}s"
            for release in candidate["release_moves"]
        )
        print(f"    Release: {release_text}")
    else:
        print("    Release: none")

    precedence = evaluation.get("runtime_precedence", ())
    print("    Runtime precedence:")
    if precedence:
        for predecessor, successor in precedence:
            print(f"      {predecessor} -> {successor}")
    else:
        print("      none")

    starts = evaluation.get("task_start_times", {})
    finishes = evaluation.get("task_finish_times", {})
    if finishes:
        print("    DAG estimate:")
        for task_id, finish in finishes.items():
            print(
                f"      {task_id}: start={starts[task_id]:.2f}s, "
                f"finish={finish:.2f}s"
            )
        for robot, finish in evaluation.get("robot_finish_times", {}).items():
            print(f"      {robot} branch finish={finish:.2f}s")
        print(
            "      remaining makespan="
            f"{evaluation['estimated_makespan']:.2f}s"
        )
    print(
        f"    total distance={evaluation['total_distance']:.2f}m, "
        "additional distance="
        f"{evaluation.get('additional_distance', evaluation['total_distance']):.2f}m, "
        f"total delay={evaluation['total_delay']:.2f}s"
    )


def _complete_candidate(
    priority_agent,
    group,
    contexts,
    route_planner,
    priority_route,
    releases,
    overrides,
    strategy,
    remaining_estimator,
    robot_speed,
    candidate_validator,
    evaluate_dag=True,
    nominal_path_lengths=None,
):
    """Attach current-route and DAG estimates to one feasible path choice."""
    current_paths = {priority_agent: priority_route["path"]}
    current_path_lengths = {
        priority_agent: path_length(current_paths[priority_agent]),
    }
    for agent_id in group:
        if agent_id == priority_agent or contexts.get(agent_id) is None:
            continue
        try:
            route = route_planner(
                agent_id,
                contexts[agent_id],
                overrides,
                False,
            )
        except RuntimeError:
            return None
        current_paths[agent_id] = route["path"]
        current_path_lengths[agent_id] = path_length(route["path"])

    candidate = {
        "strategy": strategy,
        "priority_agent": priority_agent,
        "conflict_group": group,
        "release_moves": list(releases),
        "priority_path": priority_route["path"],
        "priority_target": priority_route.get("target_object"),
        "priority_path_length": current_path_lengths[priority_agent],
        "current_paths": current_paths,
        "current_path_lengths": current_path_lengths,
        "nominal_path_lengths": dict(nominal_path_lengths or {}),
        "position_overrides": dict(overrides),
        "feasible": True,
    }
    if candidate_validator is not None and not candidate_validator(candidate):
        return None
    metrics = _local_metrics(candidate, contexts, robot_speed)
    if evaluate_dag and remaining_estimator is not None:
        estimated = remaining_estimator(candidate, contexts, metrics.copy())
        if estimated is not None:
            metrics.update(estimated)
    candidate["evaluation"] = metrics
    return candidate


def build_priority_hypotheses(
    priority_agent,
    conflict_group,
    contexts,
    route_planner,
    blocker_ranker,
    release_planner,
    remaining_estimator=None,
    robot_speed=ONLINE_REPLANNING_ROBOT_SPEED,
    candidate_validator=None,
    selection_policy="dag_aware",
    max_candidates_per_root=DEFAULT_MAX_CANDIDATES_PER_ROOT,
    max_expanded_per_root=DEFAULT_MAX_EXPANDED_PER_ROOT,
    planning_timeout_s=None,
    diagnostics=None,
    joint_staging_planner=None,
):
    """Build direct and minimum-release path variants for one priority robot."""
    group = tuple(sorted(conflict_group))
    context = contexts[priority_agent]
    overrides = {}
    started = time.perf_counter()
    deadline = (
        None if planning_timeout_s is None
        else started + float(planning_timeout_s)
    )
    evaluate_dag = selection_policy == "dag_aware"
    stats = diagnostics if diagnostics is not None else {}
    stats.update({
        "expanded": 0,
        "feasible": 0,
        "duplicate_pruned": 0,
        "no_path_pruned": 0,
        "candidate_limit_pruned": 0,
        "expanded_limit_pruned": 0,
        # Kept explicit for parity with standalone: DAG-bound pruning is off.
        "bound_pruned": 0,
        "dag_bound_pruning": False,
    })

    def check_deadline():
        if deadline is not None and time.perf_counter() >= deadline:
            stats["planning_wall_time_s"] = time.perf_counter() - started
            raise PlanningTimeoutError(
                "AI2-THOR conflict planning exceeded its wall-time budget",
                stats,
            )

    # Static preferred path identifies which robots actually obstruct Hi. It
    # does not predict future trajectories and does not mutate live state.
    try:
        check_deadline()
        preferred = route_planner(
            priority_agent,
            context,
            overrides,
            False,
        )
    except RuntimeError:
        return []

    # Use one fixed nominal reference for every candidate generated under
    # this priority hypothesis. Local-Distance then measures added motion
    # instead of assigning zero cost to every direct reroute.
    nominal_path_lengths = {priority_agent: path_length(preferred["path"])}
    for agent_id in group:
        if agent_id == priority_agent or contexts.get(agent_id) is None:
            continue
        try:
            nominal_route = route_planner(
                agent_id,
                contexts[agent_id],
                {},
                False,
            )
        except RuntimeError:
            continue
        nominal_path_lengths[agent_id] = path_length(nominal_route["path"])

    candidates = []
    direct_route = None
    try:
        direct_route = route_planner(
            priority_agent,
            context,
            {},
            True,
        )
    except RuntimeError:
        pass
    if direct_route is not None:
        direct = _complete_candidate(
            priority_agent,
            group,
            contexts,
            route_planner,
            direct_route,
            [],
            {},
            "short_replan",
            remaining_estimator,
            robot_speed,
            candidate_validator,
            evaluate_dag=evaluate_dag,
            nominal_path_lengths=nominal_path_lengths,
        )
        if direct is not None:
            candidates.append(direct)

    if selection_policy == "reroute_only":
        stats["planning_wall_time_s"] = time.perf_counter() - started
        return candidates

    # Relocation uses the same DFS/Branch-and-Bound skeleton as standalone:
    # one move per robot, counterfactual positions, duplicate/no-path pruning,
    # and every complete leaf joins short_replan in one final optimize().
    relocation_candidates = []
    visited = {}
    root_candidates = 0

    def state_key(position_overrides, moved_agents):
        positions = tuple(
            sorted(
                (
                    agent_id,
                    round(position["x"], 3),
                    round(position["z"], 3),
                )
                for agent_id, position in position_overrides.items()
            )
        )
        return positions, tuple(sorted(moved_agents))

    def dfs(position_overrides, releases, moved_agents):
        nonlocal root_candidates
        check_deadline()
        if (
            max_candidates_per_root is not None
            and root_candidates >= max_candidates_per_root
        ):
            stats["candidate_limit_pruned"] += 1
            return
        key = state_key(position_overrides, moved_agents)
        committed_distance = sum(
            release["release_distance"] for release in releases
        )
        if key in visited and visited[key] <= committed_distance:
            stats["duplicate_pruned"] += 1
            return
        visited[key] = committed_distance

        # Recompute the priority route and blocker set after every tentative
        # clearance move, matching the standalone counterfactual search.
        try:
            current_preferred = route_planner(
                priority_agent,
                context,
                position_overrides,
                False,
            )
        except RuntimeError:
            stats["no_path_pruned"] += 1
            return
        if releases:
            try:
                released_route = route_planner(
                    priority_agent,
                    context,
                    position_overrides,
                    True,
                )
            except RuntimeError:
                released_route = None
            if released_route is not None:
                released = _complete_candidate(
                    priority_agent,
                    group,
                    contexts,
                    route_planner,
                    released_route,
                    releases,
                    position_overrides,
                    "relocate",
                    remaining_estimator,
                    robot_speed,
                    candidate_validator,
                    evaluate_dag=evaluate_dag,
                    nominal_path_lengths=nominal_path_lengths,
                )
                if released is not None:
                    relocation_candidates.append(released)
                    root_candidates += 1
                    stats["feasible"] += 1
                    return

        blockers = [
            blocker
            for blocker in blocker_ranker(
                priority_agent,
                group,
                current_preferred["path"],
                position_overrides,
            )
            if blocker not in moved_agents
        ]
        support_movers = [
            agent_id for agent_id in group
            if agent_id != priority_agent
            and agent_id not in blockers
            and agent_id not in moved_agents
        ]
        movers = blockers + support_movers
        if not movers:
            stats["no_path_pruned"] += 1
            return
        if (
            max_expanded_per_root is not None
            and stats["expanded"] >= max_expanded_per_root
        ):
            stats["expanded_limit_pruned"] += 1
            return
        stats["expanded"] += 1
        for blocker_agent in movers:
            release_options = release_planner(
                priority_agent,
                blocker_agent,
                current_preferred["path"],
                position_overrides,
                group,
                context,
                blocker_context=contexts.get(blocker_agent),
            )
            if release_options is None:
                stats["no_path_pruned"] += 1
                continue
            if isinstance(release_options, dict):
                release_options = [release_options]
            for release in islice(
                release_options,
                ONLINE_RELEASE_CANDIDATE_COUNT,
            ):
                next_overrides = dict(position_overrides)
                next_overrides[blocker_agent] = release[
                    "release_position"
                ].copy()
                dfs(
                    next_overrides,
                    [*releases, release],
                    moved_agents | {blocker_agent},
                )

    dfs({}, [], frozenset())
    if root_candidates == 0 and joint_staging_planner is not None:
        for releases, position_overrides in joint_staging_planner(
            priority_agent,
            group,
            context,
            contexts,
        ):
            check_deadline()
            if (
                max_candidates_per_root is not None
                and root_candidates >= max_candidates_per_root
            ):
                stats["candidate_limit_pruned"] += 1
                break
            try:
                staged_route = route_planner(
                    priority_agent,
                    context,
                    position_overrides,
                    True,
                )
            except RuntimeError:
                stats["no_path_pruned"] += 1
                continue
            staged = _complete_candidate(
                priority_agent,
                group,
                contexts,
                route_planner,
                staged_route,
                releases,
                position_overrides,
                "relocate",
                remaining_estimator,
                robot_speed,
                candidate_validator,
                evaluate_dag=evaluate_dag,
                nominal_path_lengths=nominal_path_lengths,
            )
            if staged is None:
                continue
            staged["joint_staging"] = True
            relocation_candidates.append(staged)
            root_candidates += 1
            stats["feasible"] += 1
    for candidate in relocation_candidates:
        candidate["search_stats"] = dict(stats)
    candidates.extend(relocation_candidates)
    stats["planning_wall_time_s"] = time.perf_counter() - started
    return candidates


def build_priority_hypothesis(*args, optimizer=optimize, **kwargs):
    """Compatibility helper returning the best variant for one priority."""
    candidates = build_priority_hypotheses(*args, **kwargs)
    if not candidates:
        return None
    return optimizer(
        candidates,
        evaluator=lambda candidate: candidate["evaluation"],
    )["candidate"]


def select_minimum_disruption_plan(
    conflict_group,
    contexts,
    route_planner,
    blocker_ranker,
    release_planner,
    remaining_estimator=None,
    optimizer=optimize,
    priority_agents=None,
    candidate_validator=None,
    selection_policy="dag_aware",
    planning_timeout_s=None,
    max_candidates_per_root=DEFAULT_MAX_CANDIDATES_PER_ROOT,
    max_expanded_per_root=DEFAULT_MAX_EXPANDED_PER_ROOT,
    diagnostics=None,
    joint_staging_planner=None,
):
    """Build all Hi, prune infeasible hypotheses, and apply shared optimize()."""
    planning_started = time.perf_counter()
    route_planner = _memoize_route_planner(route_planner)
    group = tuple(sorted(conflict_group))
    print("conflict_group=" + str([f"R{agent + 1}" for agent in group]))
    candidates = []
    all_root_stats = []
    priorities = tuple(group if priority_agents is None else priority_agents)
    if selection_policy != "dag_aware":
        # Local baselines must not receive DAG-derived priority ordering.
        priorities = tuple(sorted(priorities))
    for priority_agent in priorities:
        root_timeout_s = None
        if planning_timeout_s is not None:
            root_timeout_s = (
                float(planning_timeout_s)
                - (time.perf_counter() - planning_started)
            )
            if root_timeout_s <= 0:
                raise PlanningTimeoutError(
                    "AI2-THOR conflict planning exceeded its total "
                    "wall-time budget",
                    {"planning_wall_time_s": time.perf_counter() - planning_started},
                )
        print(f"priority hypothesis=H(R{priority_agent + 1})")
        root_diagnostics = {}
        try:
            variants = build_priority_hypotheses(
                priority_agent,
                group,
                contexts,
                route_planner,
                blocker_ranker,
                release_planner,
                remaining_estimator=remaining_estimator,
                candidate_validator=candidate_validator,
                selection_policy=selection_policy,
                planning_timeout_s=root_timeout_s,
                max_candidates_per_root=max_candidates_per_root,
                max_expanded_per_root=max_expanded_per_root,
                diagnostics=root_diagnostics,
                joint_staging_planner=joint_staging_planner,
            )
        except RuntimeError as error:
            print(f"  planning error={error}")
            variants = []
        all_root_stats.append(dict(root_diagnostics))
        if not variants:
            print("  candidate feasibility=INFEASIBLE -> pruned")
            continue
        for candidate in variants:
            for release in candidate["release_moves"]:
                position = release["release_position"]
                print(
                    f"  blocker=R{release['blocker_agent'] + 1}, "
                    f"release position=({position['x']:.2f}, "
                    f"{position['z']:.2f}), "
                    f"release distance={release['release_distance']:.2f}m"
                )
            evaluation = candidate["evaluation"]
            _log_dag_estimate(candidate, evaluation)
            print(
                f"  variant={candidate['strategy']}, "
                f"priority path length={candidate['priority_path_length']:.2f}m, "
                "candidate feasibility=FEASIBLE, "
                f"estimated makespan={evaluation['estimated_makespan']:.2f}s, "
                f"total distance={evaluation['total_distance']:.2f}m"
            )
            candidates.append(candidate)

    if not candidates:
        if selection_policy == "reroute_only":
            raise RuntimeError("baseline_no_feasible_intervention")
        raise RuntimeError("No feasible AI2-THOR conflict intervention")
    candidates.sort(key=_candidate_stable_key)
    for index, item in enumerate(candidates):
        item["candidate_id"] = f"C{index:04d}"
    candidate = _select_candidate(candidates, selection_policy, optimizer)
    evaluation = candidate["evaluation"]
    search_stats = diagnostics if diagnostics is not None else {}
    for field in (
        "expanded", "feasible", "bound_pruned", "duplicate_pruned",
        "no_path_pruned", "candidate_limit_pruned", "expanded_limit_pruned",
    ):
        search_stats[field] = sum(
            int(root.get(field, 0)) for root in all_root_stats
        )
    search_stats["dag_bound_pruning"] = False
    search_stats["generated_candidates"] = len(candidates)
    search_stats.update(route_planner.statistics)
    search_stats["planning_wall_time_s"] = (
        time.perf_counter() - planning_started
    )
    search_stats["candidate_evaluations"] = [
        _candidate_diagnostic(item, selected=item is candidate)
        for item in candidates
    ]
    candidate["search_stats"] = search_stats
    print(
        f"selected candidate=H(R{candidate['priority_agent'] + 1})/"
        f"{candidate['strategy']}, "
        f"makespan={evaluation['estimated_makespan']:.2f}s, "
        f"distance={evaluation['total_distance']:.2f}m, "
        "extra="
        f"{evaluation.get('additional_distance', evaluation['total_distance']):.2f}m, "
        f"delay={evaluation['total_delay']:.2f}s, "
        f"candidates={len(candidates)}, "
        f"A*-cache={search_stats['route_cache_hits']}/"
        f"{search_stats['route_cache_misses']} hits/misses, "
        f"planning={search_stats['planning_wall_time_s']:.3f}s"
    )
    return candidate
