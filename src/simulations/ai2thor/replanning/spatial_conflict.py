"""AI2-THOR spatial conflict detection and MoveAside planning."""

import math

from core import (
    distance_to_polyline,
    goal_directed_yield_key,
    select_goal_directed_top_k,
)

from simulations.ai2thor.navigation.grid_planner import (
    get_global_grid_map,
    iter_padded_astar_paths,
)
from simulations.ai2thor.config import (
    MULTI_AGENT_CLEARANCE,
    NAVIGATION_GRID_SIZE,
    NAVIGATION_PADDING,
    NAVIGATION_POSITION_TOLERANCE,
    ONLINE_ASIDE_MAX_DISTANCE,
    ONLINE_RELEASE_CANDIDATE_COUNT,
    SMOOTH_FINAL_POSITION_TOLERANCE,
    SMOOTH_MOVE_STEP,
    carried_object_safety_margin,
)


def _distance(first, second):
    return math.hypot(
        second["x"] - first["x"],
        second["z"] - first["z"],
    )


def _path_length(path):
    return sum(_distance(first, second) for first, second in zip(path, path[1:]))


def _position_at_distance(path, distance):
    if not path:
        raise ValueError("A candidate trajectory cannot be empty")
    remaining = max(0.0, float(distance))
    for first, second in zip(path, path[1:]):
        length = _distance(first, second)
        if remaining <= length and length > 1e-12:
            ratio = remaining / length
            return {
                "x": first["x"] + ratio * (second["x"] - first["x"]),
                "z": first["z"] + ratio * (second["z"] - first["z"]),
            }
        remaining -= length
    return {"x": path[-1]["x"], "z": path[-1]["z"]}


def _swept_distance(a0, a1, b0, b1):
    delta = (a0["x"] - b0["x"], a0["z"] - b0["z"])
    velocity = (
        a1["x"] - a0["x"] - b1["x"] + b0["x"],
        a1["z"] - a0["z"] - b1["z"] + b0["z"],
    )
    norm = velocity[0] ** 2 + velocity[1] ** 2
    ratio = 0.0 if norm <= 1e-12 else max(
        0.0,
        min(1.0, -sum(d * v for d, v in zip(delta, velocity)) / norm),
    )
    return math.hypot(
        delta[0] + ratio * velocity[0],
        delta[1] + ratio * velocity[1],
    )


def synchronized_paths_are_safe(paths, pair_clearance):
    """Check continuous equal-speed swept separation for every path pair."""
    ids = sorted(paths)
    lengths = {agent_id: _path_length(paths[agent_id]) for agent_id in ids}
    breakpoints = {0.0, *lengths.values()}
    for path in paths.values():
        elapsed = 0.0
        for first, second in zip(path, path[1:]):
            elapsed += _distance(first, second)
            breakpoints.add(elapsed)
    times = sorted(breakpoints)
    for index, first_id in enumerate(ids):
        for second_id in ids[index + 1:]:
            clearance = float(pair_clearance(first_id, second_id))
            initial_first = _position_at_distance(paths[first_id], 0.0)
            initial_second = _position_at_distance(paths[second_id], 0.0)
            previous = _distance(initial_first, initial_second)
            cleared = previous > clearance
            for start, finish in zip(times, times[1:]):
                a0 = _position_at_distance(paths[first_id], start)
                a1 = _position_at_distance(paths[first_id], finish)
                b0 = _position_at_distance(paths[second_id], start)
                b1 = _position_at_distance(paths[second_id], finish)
                swept = _swept_distance(a0, a1, b0, b1)
                current = _distance(a1, b1)
                if cleared:
                    if swept <= clearance:
                        return False
                else:
                    # A reported contact starts inside the clearance discs.
                    # Permit only monotone separation, never deeper overlap.
                    if swept + 1e-9 < previous or current + 1e-9 < previous:
                        return False
                    if current > clearance + 1e-9:
                        cleared = True
                previous = current
    return True


def _release_clears_positions(mover, path, positions, pair_clearance):
    """Allow separation from initial contact, but forbid deeper/re-entry motion."""
    epsilon = 1e-9
    for other, centre in positions.items():
        if other == mover:
            continue
        clearance = float(pair_clearance(mover, other))
        previous = _distance(path[0], centre)
        cleared = previous > clearance
        previous_point = path[0]
        for point in path[1:]:
            current = _distance(point, centre)
            swept = _point_segment_distance(centre, previous_point, point)
            if cleared:
                if swept <= clearance + epsilon:
                    return False
            else:
                if swept + epsilon < previous or current + epsilon < previous:
                    return False
                if current > clearance + epsilon:
                    cleared = True
            previous = current
            previous_point = point
    return True


def candidate_paths_are_safe(candidate, pair_clearance):
    """Validate the same release/priority/resume phases used at runtime."""
    current_paths = candidate["current_paths"]
    positions = {
        agent_id: path[0] for agent_id, path in current_paths.items()
    }
    # Current continuation paths start at final release poses. Restore each
    # blocker's pre-release pose before replaying the serial release phases.
    for release in candidate.get("release_moves", ()):
        positions[int(release["blocker_agent"])] = release["release_path"][0]
    for release in candidate.get("release_moves", ()):
        mover = int(release["blocker_agent"])
        if not _release_clears_positions(
            mover, release["release_path"], positions, pair_clearance
        ):
            return False
        positions[mover] = release["release_position"]

    if candidate.get("release_moves"):
        priority = int(candidate["priority_agent"])
        priority_phase=candidate.get("priority_release_prefix",candidate["priority_path"])
        phase = {
            agent_id: (
                priority_phase if agent_id == priority else [position]
            )
            for agent_id, position in positions.items()
        }
        if not synchronized_paths_are_safe(phase, pair_clearance):
            return False
        if candidate.get("concurrent_resume_paths") is not None:
            return synchronized_paths_are_safe(candidate["concurrent_resume_paths"],pair_clearance)
        resume = {
            agent_id: path
            for agent_id, path in current_paths.items()
            if agent_id != priority
        }
        resume[priority] = [candidate["priority_path"][-1]]
        return synchronized_paths_are_safe(resume, pair_clearance)
    return synchronized_paths_are_safe(current_paths, pair_clearance)


def certify_concurrent_resume(candidate,pair_clearance):
    """Find the earliest priority waypoint allowing safe concurrent resume.

    Release moves remain serial. Other robots wait only for the protected
    prefix, not the priority's full route/action. Live swept guards still own
    execution; this is not permission to force agents through each other.
    """
    if not candidate.get("release_moves"):
        return
    priority=int(candidate["priority_agent"])
    path=candidate["priority_path"]
    held={rid:route[0] for rid,route in candidate["current_paths"].items()}
    for release in candidate["release_moves"]:
        held[int(release["blocker_agent"])]=release["release_position"]
    for index in range(len(path)):
        resume={rid:route for rid,route in candidate["current_paths"].items()}
        resume.update({rid:[pose] for rid,pose in held.items() if rid not in resume})
        resume[priority]=path[index:]
        prefix={rid:(path[:index+1] if rid==priority else [pose]) for rid,pose in held.items()}
        if synchronized_paths_are_safe(prefix,pair_clearance) and synchronized_paths_are_safe(resume,pair_clearance):
            candidate["priority_release_prefix"]=path[:index+1]
            candidate["concurrent_resume_paths"]={rid:route for rid,route in resume.items()
                                                  if rid in candidate["current_paths"]}
            return


def _direction_group(start, destination):
    """Quantize a relocation destination into one of eight search roots."""
    dx = destination["x"] - start["x"]
    dz = destination["z"] - start["z"]
    if abs(dx) <= 1e-9 and abs(dz) <= 1e-9:
        return None
    return int(round(math.atan2(dz, dx) / (math.pi / 4.0))) % 8


def _point_segment_distance(point, first, second):
    dx = second["x"] - first["x"]
    dz = second["z"] - first["z"]
    length_squared = dx * dx + dz * dz
    if length_squared <= 1e-12:
        return _distance(point, first)
    ratio = (
        (point["x"] - first["x"]) * dx
        + (point["z"] - first["z"]) * dz
    ) / length_squared
    ratio = max(0.0, min(1.0, ratio))
    closest = {
        "x": first["x"] + ratio * dx,
        "z": first["z"] + ratio * dz,
    }
    return _distance(point, closest)


def _distance_to_path(point, path):
    if len(path) == 1:
        return _distance(point, path[0])
    return min(
        _point_segment_distance(point, first, second)
        for first, second in zip(path, path[1:])
    )


def _agent_positions(controller):
    events = getattr(controller.last_event, "events", None) or []
    return {
        agent_id: event.metadata["agent"]["position"].copy()
        for agent_id, event in enumerate(events)
    }


def rank_blockers_on_path(
    controller,
    priority_agent_id,
    conflict_group,
    preferred_path,
    position_overrides=None,
):
    """Return only group members intersecting the priority path, nearest first."""
    positions = _agent_positions(controller)
    positions.update(position_overrides or {})
    events = getattr(controller.last_event, "events", None) or []
    priority_margin = carried_object_safety_margin(events[priority_agent_id])
    blockers = []
    for blocker_id in conflict_group:
        if blocker_id == priority_agent_id:
            continue
        clearance = MULTI_AGENT_CLEARANCE + max(
            priority_margin,
            carried_object_safety_margin(events[blocker_id]),
        ) + SMOOTH_MOVE_STEP + SMOOTH_FINAL_POSITION_TOLERANCE
        distance = _distance_to_path(positions[blocker_id], preferred_path)
        if distance <= clearance:
            blockers.append((distance, blocker_id))
    blockers.sort()
    return [blocker_id for _, blocker_id in blockers]


def iter_minimum_release_motions(
    controller,
    priority_agent_id,
    blocker_agent_id,
    preferred_path,
    position_overrides=None,
    forbidden_regions=None,
    blocker_goal_path=None,
):
    """Yield deterministic Top-K releases ranked by progress to the goal.

    The primary score is release-path length plus optimistic continuation from
    the release pose to the blocker's original goal. Temporary backward moves
    remain legal, which is necessary in narrow corridors.
    """
    overrides = dict(position_overrides or {})
    positions = _agent_positions(controller)
    positions.update(overrides)
    blocker_position = positions[blocker_agent_id]
    events = getattr(controller.last_event, "events", None) or []
    blocker_margin = carried_object_safety_margin(events[blocker_agent_id])
    priority_margin = carried_object_safety_margin(events[priority_agent_id])
    path_clearance = (
        MULTI_AGENT_CLEARANCE
        + max(blocker_margin, priority_margin)
        + NAVIGATION_POSITION_TOLERANCE
        + NAVIGATION_GRID_SIZE
    )
    position_by_cell, free_cells = get_global_grid_map(
        controller,
        NAVIGATION_GRID_SIZE,
        NAVIGATION_PADDING,
        blocker_agent_id,
    )
    goals = []
    for cell in free_cells:
        position = position_by_cell[cell]
        move_distance = _distance(blocker_position, position)
        if move_distance <= 1e-9 or move_distance > ONLINE_ASIDE_MAX_DISTANCE:
            continue
        if _distance_to_path(position, preferred_path) <= path_clearance:
            continue
        if any(
            region["min_x"] - NAVIGATION_PADDING
            <= position["x"]
            <= region["max_x"] + NAVIGATION_PADDING
            and region["min_z"] - NAVIGATION_PADDING
            <= position["z"]
            <= region["max_z"] + NAVIGATION_PADDING
            for region in forbidden_regions or ()
        ):
            continue
        if any(
            other_id != blocker_agent_id
            and _distance(position, other_position) <= path_clearance
            for other_id, other_position in positions.items()
        ):
            continue
        direction = _direction_group(blocker_position, position)
        if direction is not None:
            goals.append((position, position))

    if not goals:
        return

    candidates_by_direction = {}
    worst_top_k_cost = math.inf
    try:
        paths = iter_padded_astar_paths(
            controller,
            blocker_position,
            goals,
            grid_size=NAVIGATION_GRID_SIZE,
            padding=NAVIGATION_PADDING,
            agent_id=blocker_agent_id,
            avoid_other_agents=True,
            # The blocker may start inside a future door-swing region and
            # must be allowed to leave it without teleporting.
            forbidden_regions=None,
            agent_position_overrides=overrides,
            # The release starts from an already reported contact. Removing
            # the priority robot's full static exclusion disc also removes
            # the blocker's start neighborhood and can make every escape
            # falsely unreachable. The continuous release validator below
            # still requires monotone separation and prevents re-entry.
            ignored_agent_ids={priority_agent_id},
            include_search_cost=True,
        )
        for path, position, release_lower_bound in paths:
            # Paths arrive in nondecreasing shortest-path distance. Since
            # Cgd = release_distance + nonnegative continuation, no unseen
            # endpoint can enter the current Top-K once this lower bound is
            # strictly worse than its largest retained score.
            if release_lower_bound > worst_top_k_cost + 1e-9:
                break
            direction = _direction_group(blocker_position, position)
            # Keep the snapped A* start cell. Dropping it and joining the live
            # pose directly to the next compressed corner creates an
            # unchecked chord that may cut toward the opposing robot.
            path = [blocker_position.copy(), *path]
            release_distance = _path_length(path)
            candidate = {
                "blocker_agent": blocker_agent_id,
                "release_position": position.copy(),
                "release_path": path,
                "release_distance": release_distance,
                "direction_group": direction,
                "goal_deviation": (
                    distance_to_polyline(position, blocker_goal_path)
                    if blocker_goal_path
                    else 0.0
                ),
                "goal_directed_cost": (
                    release_distance
                    + _distance(position, blocker_goal_path[-1])
                    if blocker_goal_path
                    else release_distance
                ),
            }
            direction_candidates = candidates_by_direction.setdefault(
                direction,
                [],
            )
            direction_candidates.append(candidate)
            direction_candidates.sort(key=goal_directed_yield_key)
            del direction_candidates[2:]
            # Preserve the best clearance in each geometric direction. This
            # prevents many nearly identical forward cells from exhausting
            # Top-K before a necessary backward-yield candidate is reached.
            if (
                len(candidates_by_direction) == 8
                and all(
                    len(items) == 2
                    for items in candidates_by_direction.values()
                )
            ):
                worst_top_k_cost = max(
                    item["goal_directed_cost"]
                    for items in candidates_by_direction.values()
                    for item in items
                )
    except RuntimeError:
        return

    yield from select_goal_directed_top_k(
        (
            item
            for items in candidates_by_direction.values()
            for item in items
        ),
        ONLINE_RELEASE_CANDIDATE_COUNT,
    )
