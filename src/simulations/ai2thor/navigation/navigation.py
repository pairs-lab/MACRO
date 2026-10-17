"""Padded grid navigation for controllable AI2-THOR robots."""

import math
import re
import time

from simulations.ai2thor.cameras import aim_robot_camera, move_robot
from simulations.ai2thor.config import (
    AGENT_RADIUS,
    NAVIGATION_GRID_SIZE,
    NAVIGATION_FINAL_ROTATION_STEP,
    NAVIGATION_MOVE_STEP,
    NAVIGATION_PADDING,
    NAVIGATION_PATH_PREVIEW_SECONDS,
    NAVIGATION_POSITION_TOLERANCE,
    NAVIGATION_ROTATION_STEP,
    INTERACTION_POSITION_OVERRIDES,
    interaction_distance_for,
    ONLINE_MAX_YIELDS_PER_NAVIGATION,
    SMOOTH_NAVIGATION_ENABLED,
    carried_object_safety_margin,
)
from simulations.ai2thor.controller import require_success

from .grid_planner import find_padded_astar_path, get_reachable_positions
from .door_rules import future_open_rule
from .smooth_path_follower import follow_smooth_path


class NavigationBlockedError(RuntimeError):
    """Raised when Unity reports an unexpected obstacle along a planned path."""


def _distance_xz(first, second):
    return math.hypot(second["x"] - first["x"], second["z"] - first["z"])


def _agent_event(event, agent_id):
    """Return metadata/frame event belonging to one real iTHOR agent."""
    events = getattr(event, "events", None)
    if events and len(events) > 1:
        return events[agent_id]
    return event


def _target_yaw(first, second):
    delta_x = second["x"] - first["x"]
    delta_z = second["z"] - first["z"]
    return math.degrees(math.atan2(delta_x, delta_z)) % 360


def _signed_yaw_difference(current_yaw, target_yaw):
    return (target_yaw - current_yaw + 180) % 360 - 180


def _distance_to_object_bounds(position, scene_object):
    """Return horizontal distance from a position to an object's bounding box."""
    bounds = scene_object.get("axisAlignedBoundingBox")
    if not bounds:
        return _distance_xz(position, scene_object["position"])

    center = bounds["center"]
    size = bounds["size"]
    min_x = center["x"] - size["x"] / 2
    max_x = center["x"] + size["x"] / 2
    min_z = center["z"] - size["z"] / 2
    max_z = center["z"] + size["z"] / 2

    delta_x = max(min_x - position["x"], 0, position["x"] - max_x)
    delta_z = max(min_z - position["z"], 0, position["z"] - max_z)
    return math.hypot(delta_x, delta_z)


def _interaction_goals(
    controller,
    object_type,
    agent_id=0,
    future_open_object=None,
    raw_pose_cache=None,
    timing=None,
):
    """Return every collision-free pose that can see a matching object."""
    exact_object_id = "|" in object_type
    objects = [
        obj
        for obj in controller.last_event.metadata.get("objects", [])
        if (
            obj.get("objectId") == object_type
            if exact_object_id
            else obj["objectType"].casefold() == object_type.casefold()
        )
    ]
    if not objects:
        raise ValueError(f"No object of type {object_type!r} exists")

    candidates = []
    diagnostics = []
    for scene_object in objects:
        interaction_limit = interaction_distance_for(scene_object)
        position = scene_object.get("position", {})
        bounds = scene_object.get("axisAlignedBoundingBox", {})
        centre = bounds.get("center", {})
        size = bounds.get("size", {})
        pose_key = (
            scene_object["objectId"],
            tuple(
                round(float(position.get(axis, 0.0)), 3)
                for axis in ("x", "y", "z")
            ),
            tuple(
                round(float(centre.get(axis, 0.0)), 3)
                for axis in ("x", "y", "z")
            ),
            tuple(
                round(float(size.get(axis, 0.0)), 3)
                for axis in ("x", "y", "z")
            ),
            bool(scene_object.get("isOpen", False)),
        )
        poses = raw_pose_cache.get(pose_key) if raw_pose_cache is not None else None
        if poses is None:
            query_started = time.perf_counter()
            positions = [
                point for point in get_reachable_positions(controller, NAVIGATION_GRID_SIZE, agent_id)
                if _distance_to_object_bounds(point, scene_object) <= interaction_limit - 0.04
            ]
            poses_event = require_success(
                controller.step(
                    action="GetInteractablePoses",
                    objectId=scene_object["objectId"],
                    **({"positions": positions} if positions else {}),
                    rotations=[0, 45, 90, 135, 180, 225, 270, 315],
                    horizons=[30],
                    standings=[True],
                    agentId=agent_id,
                    # Planning needs metadata only. Rendering each query causes
                    # an unnecessary Unity/GPU synchronization point.
                    renderImage=False,
                ),
                f"Get interactable poses for {scene_object['objectId']}",
            )
            poses = poses_event.metadata["actionReturn"]
            query_count = 1
            if not poses and scene_object.get("objectType", "").casefold() in {"faucet", "microwave", "bread"}:
                # Faucets, elevated microwaves and bread inside a microwave
                # may not be visible at the default downward camera horizon.
                # Ask Unity for additional genuine visible poses; never invent
                # a sink pose or bypass visibility/range checks for the faucet.
                fallback = require_success(
                    controller.step(
                        action="GetInteractablePoses",
                        objectId=scene_object["objectId"],
                        **({"positions": positions} if positions else {}),
                        rotations=[0, 45, 90, 135, 180, 225, 270, 315],
                        horizons=[0, 60],
                        standings=[True],
                        agentId=agent_id,
                        renderImage=False,
                    ),
                    f"Get alternative viewing poses for {scene_object['objectId']}",
                )
                poses = fallback.metadata["actionReturn"]
                query_count += 1
            if raw_pose_cache is not None:
                raw_pose_cache[pose_key] = [dict(pose) for pose in poses]
            if timing is not None:
                timing["unity_goal_queries"] = (
                    timing.get("unity_goal_queries", 0) + query_count
                )
                timing["unity_goal_time_s"] = (
                    timing.get("unity_goal_time_s", 0.0)
                    + time.perf_counter() - query_started
                )
        if not poses:
            diagnostics.append(f"{scene_object['objectId']}: Unity returned 0 visible poses")
            continue

        distances = {
            id(pose): _distance_to_object_bounds(pose, scene_object)
            for pose in poses
        }
        agent = _agent_event(controller.last_event, agent_id)
        payload_margin = carried_object_safety_margin(agent)
        minimum_payload_clearance = (
            AGENT_RADIUS + payload_margin if payload_margin else 0.0
        )
        safe_poses = [
            pose for pose in poses
            if distances[id(pose)] >= minimum_payload_clearance
            and (
                future_open_object is not None
                or distances[id(pose)] <= interaction_limit - 0.04
            )
        ]
        if not safe_poses:
            diagnostics.append(
                f"{scene_object['objectId']}: {len(poses)} visible poses rejected by range/payload clearance; "
                f"nearest={min(distances.values()):.3f}m, maximum={interaction_limit-.04:.3f}m, "
                f"minimum_payload_clearance={minimum_payload_clearance:.3f}m"
            )
            continue

        rule = (
            future_open_rule(future_open_object)
            if future_open_object
            else None
        )
        goal_filter = (rule or {}).get("goal_filter", {})
        safe_poses = [
            pose for pose in safe_poses
            if pose["x"] >= goal_filter.get("min_x", -math.inf)
            and pose["x"] <= goal_filter.get("max_x", math.inf)
            and pose["z"] >= goal_filter.get("min_z", -math.inf)
            and pose["z"] <= goal_filter.get("max_z", math.inf)
        ]
        if not safe_poses:
            diagnostics.append(f"{scene_object['objectId']}: visible poses rejected by door goal filter")
            continue

        requested_pose = (
            INTERACTION_POSITION_OVERRIDES.get(
                scene_object["objectId"].casefold()
            )
            or INTERACTION_POSITION_OVERRIDES.get(
                scene_object["objectType"].casefold()
            )
            or INTERACTION_POSITION_OVERRIDES.get(object_type.casefold())
        )
        if requested_pose is not None:
            if requested_pose.get("required", False):
                # A fixed interaction location is a hard spatial constraint,
                # not permission to invent a pose or bypass Unity visibility.
                tolerance = requested_pose.get("tolerance", .03)
                nearest = min(_distance_xz(pose, requested_pose) for pose in safe_poses)
                safe_poses = [pose for pose in safe_poses
                              if _distance_xz(pose, requested_pose) <= tolerance + 1e-9]
                if not safe_poses:
                    diagnostics.append(
                        f"{scene_object['objectId']}: required interaction point "
                        f"({requested_pose['x']:.3f}, {requested_pose['z']:.3f}) unavailable; "
                        f"nearest valid pose is {nearest:.3f}m away, tolerance={tolerance:.3f}m"
                    )
                    continue
            # A configured point is a preference, not evidence that Unity can
            # interact there. Retain real interactable poses near that point.
            safe_poses = sorted(safe_poses, key=lambda pose: _distance_xz(pose, requested_pose))

        nearby_poses = safe_poses
        for pose in nearby_poses:
            goal = {key: pose[key] for key in ("x", "y", "z")}
            goal["interaction_pose"] = True
            candidates.append((goal, scene_object))

    if not candidates:
        raise RuntimeError(f"No reachable pose can see {object_type!r}; " + "; ".join(diagnostics))

    return candidates


def _known_open_rule(controller, object_type):
    """Return a map rule only when this openable is already open."""
    rule = future_open_rule(object_type)
    if rule is None:
        return None
    return rule if any(
        obj.get("objectType", "").casefold() == object_type.casefold()
        and obj.get("isOpen", False)
        for obj in controller.last_event.metadata.get("objects", [])
    ) else None


def plan_realtime_route(
    controller,
    object_type,
    agent_id,
    future_open_object=None,
    agent_position_overrides=None,
    avoid_other_agents=True,
    interaction_goal_cache=None,
    raw_interaction_pose_cache=None,
    timing=None,
):
    """Plan one fresh A* route over the latest dynamic-agent overlay."""
    open_rule = (
        future_open_rule(future_open_object)
        if future_open_object
        else _known_open_rule(controller, object_type)
    )
    effective_open_object = (
        future_open_object
        or (object_type if open_rule is not None else None)
    )
    goal_key = (
        int(agent_id),
        object_type.casefold(),
        str(effective_open_object or "").casefold(),
    )
    goals = (
        interaction_goal_cache.get(goal_key)
        if interaction_goal_cache is not None
        else None
    )
    if goals is None:
        goal_started = time.perf_counter()
        goals = _interaction_goals(
            controller,
            object_type,
            agent_id,
            future_open_object=effective_open_object,
            raw_pose_cache=raw_interaction_pose_cache,
            timing=timing,
        )
        if interaction_goal_cache is not None:
            # Goal poses and object metadata are read-only during one frozen
            # conflict snapshot. The cache is discarded before motion resumes.
            interaction_goal_cache[goal_key] = goals
        if timing is not None:
            timing["interaction_goal_time_s"] = (
                timing.get("interaction_goal_time_s", 0.0)
                + time.perf_counter() - goal_started
            )
    live_start = _agent_event(
        controller.last_event,
        agent_id,
    ).metadata["agent"]["position"]
    start = (agent_position_overrides or {}).get(
        agent_id,
        live_start,
    ).copy()
    astar_started = time.perf_counter()
    path, target_object = find_padded_astar_path(
        controller,
        start,
        goals,
        grid_size=NAVIGATION_GRID_SIZE,
        padding=NAVIGATION_PADDING,
        agent_id=agent_id,
        avoid_other_agents=avoid_other_agents,
        forbidden_regions=(open_rule or {}).get("forbidden_regions"),
        agent_position_overrides=agent_position_overrides,
    )
    if timing is not None:
        timing["astar_time_s"] = (
            timing.get("astar_time_s", 0.0)
            + time.perf_counter() - astar_started
        )
        timing["astar_calls"] = timing.get("astar_calls", 0) + 1
    return {"path": path, "target_object": target_object}


def _notify_step(event, on_step, step_delay):
    if on_step is not None:
        on_step(event)
    if step_delay > 0:
        time.sleep(step_delay)


def _turn_toward(
    controller,
    robot_camera,
    waypoint,
    on_step,
    step_delay,
    agent_id=0,
    angle_tolerance=2.6,
    rotation_step=NAVIGATION_ROTATION_STEP,
    render_rotation=True,
):
    """Rotate in bounded increments until the robot faces a waypoint."""
    while True:
        agent = _agent_event(controller.last_event, agent_id).metadata["agent"]
        desired_yaw = _target_yaw(agent["position"], waypoint)
        yaw_difference = _signed_yaw_difference(
            agent["rotation"]["y"], desired_yaw
        )

        if abs(yaw_difference) <= angle_tolerance:
            return _agent_event(controller.last_event, agent_id)

        action = "RotateRight" if yaw_difference > 0 else "RotateLeft"
        event = move_robot(
            controller,
            robot_camera,
            action,
            report_failure=False,
            agent_id=agent_id,
            render_image=render_rotation,
            degrees=min(rotation_step, abs(yaw_difference)),
        )
        if event is None:
            message = controller.last_event.metadata.get(
                "errorMessage", "Unknown collision"
            )
            raise NavigationBlockedError(
                f"Robot could not execute {action}: {message}"
            )
        if render_rotation:
            _notify_step(event, on_step, step_delay)
        elif step_delay > 0:
            time.sleep(step_delay)


def _move_to_waypoint(
    controller,
    robot_camera,
    waypoint,
    on_step,
    step_delay,
    agent_id=0,
    position_tolerance=NAVIGATION_POSITION_TOLERANCE,
    move_step=NAVIGATION_MOVE_STEP,
    rotation_step=NAVIGATION_ROTATION_STEP,
):
    """Walk forward until the robot reaches one path waypoint."""
    previous_distance = math.inf

    while True:
        position = _agent_event(controller.last_event, agent_id).metadata["agent"][
            "position"
        ]
        distance = _distance_xz(position, waypoint)
        if distance <= position_tolerance:
            return _agent_event(controller.last_event, agent_id)

        # Correct the heading again at every step to avoid accumulated drift.
        _turn_toward(
            controller,
            robot_camera,
            waypoint,
            on_step,
            step_delay,
            agent_id,
            rotation_step=rotation_step,
            render_rotation=False,
        )

        move_magnitude = min(move_step, distance)
        event = move_robot(
            controller,
            robot_camera,
            "MoveAhead",
            report_failure=False,
            agent_id=agent_id,
            moveMagnitude=move_magnitude,
        )
        if event is None:
            message = controller.last_event.metadata.get(
                "errorMessage", "Unknown collision"
            )
            raise NavigationBlockedError(
                "Robot path is physically blocked; no object was removed. "
                f"Unity reported: {message}"
            )

        new_position = event.metadata["agent"]["position"]
        new_distance = _distance_xz(new_position, waypoint)
        if new_distance >= previous_distance - 1e-4:
            raise RuntimeError("Robot stopped making progress toward the waypoint")

        previous_distance = new_distance
        _notify_step(event, on_step, step_delay)


def _follow_navigation_path(
    controller,
    robot_camera,
    path,
    on_step,
    step_delay,
    agent_id,
    path_update_provider=None,
    observation_exempt_agent_ids=(),
):
    """Follow one already-planned path using smooth or legacy control."""
    if SMOOTH_NAVIGATION_ENABLED:
        return follow_smooth_path(
            controller,
            robot_camera,
            path,
            on_step=on_step,
            step_delay=step_delay,
            agent_id=agent_id,
            blocked_error_cls=NavigationBlockedError,
            path_update_provider=path_update_provider,
            observation_exempt_agent_ids=observation_exempt_agent_ids,
        )

    event = _agent_event(controller.last_event, agent_id)
    for waypoint in path[1:]:
        event = _move_to_waypoint(
            controller,
            robot_camera,
            waypoint,
            on_step,
            step_delay,
            agent_id,
        )
    return event


def _follow_path_waiting_for_robot(
    controller,
    robot_camera,
    path,
    on_step,
    step_delay,
    agent_id,
    conflict_resolver_provider=None,
    path_update_provider=None,
):
    """Execute one resolution, observe, and rebuild only on a new conflict."""
    current_path = path
    resolved_stationary_blocker = None
    for resolution_count in range(ONLINE_MAX_YIELDS_PER_NAVIGATION + 1):
        try:
            return _follow_navigation_path(
                controller,
                robot_camera,
                current_path,
                on_step,
                step_delay,
                agent_id,
                path_update_provider,
                (
                    {resolved_stationary_blocker}
                    if resolved_stationary_blocker is not None
                    else ()
                ),
            )
        except NavigationBlockedError as error:
            blocker = _blocking_agent_from_error(
                str(error),
                agent_id,
                controller=controller,
            )
            # A direct detour can pass the occupancy-grid checks yet still be
            # rejected by Unity because a held object's collision mesh is
            # larger than its conservative grid footprint.  The preceding
            # contact tells us which stationary robot the detour attempted to
            # pass.  Treat the failed physical execution as candidate
            # invalidation and re-optimize over the remaining interventions.
            direct_path_failed = (
                blocker is None
                and resolved_stationary_blocker is not None
                and "physically blocked" in str(error).casefold()
            )
            if direct_path_failed:
                blocker = resolved_stationary_blocker
            if blocker is None or conflict_resolver_provider is None:
                raise
            if resolution_count >= ONLINE_MAX_YIELDS_PER_NAVIGATION:
                raise NavigationBlockedError(
                    f"robot{agent_id + 1} exceeded the bounded number of "
                    "local conflict resolutions"
                ) from error

            # resolve() is a Condition barrier: every robot in the connected
            # conflict group stops before a fresh state snapshot is planned.
            replacement = conflict_resolver_provider(
                agent_id,
                blocker,
                direct_path_failed=direct_path_failed,
            )
            if not replacement:
                raise NavigationBlockedError(
                    f"No feasible local A* resolution exists for "
                    f"robot{agent_id + 1}; the robot remains stopped"
                ) from error
            resolved_stationary_blocker = blocker
            current_path = replacement
            print(
                f"robot{agent_id + 1} follows one selected realtime A* path"
            )

    raise NavigationBlockedError("Local conflict resolution did not converge")


def _blocking_agent_from_error(message, requesting_agent_id, controller=None):
    """Map an agent or carried-payload collision to the blocking agent."""
    match = re.search(
        r"Agent\s+(\d+)\s+is blocking Agent\s+(\d+)",
        message,
        flags=re.IGNORECASE,
    )
    if match is not None and int(match.group(2)) == requesting_agent_id:
        return int(match.group(1))

    if controller is None:
        return None
    payload_match = re.search(
        r"([A-Za-z][A-Za-z0-9]*)_[^\s]*\s+is blocking",
        message,
        flags=re.IGNORECASE,
    )
    if payload_match is None:
        return None
    blocking_type = payload_match.group(1).casefold()
    events = getattr(controller.last_event, "events", ())
    for agent_id, event in enumerate(events):
        if agent_id == requesting_agent_id:
            continue
        inventory = event.metadata.get("inventoryObjects") or ()
        if any(
            item.get("objectType", "").casefold() == blocking_type
            for item in inventory
        ):
            return agent_id
    return None


def navigate_to_object_type(
    controller,
    robot_camera,
    object_type,
    on_step=None,
    on_path_planned=None,
    step_delay=0.03,
    agent_id=0,
    preplanned_route=None,
    future_open_object=None,
    conflict_resolver_provider=None,
    path_update_provider=None,
    raw_interaction_pose_cache=None,
):
    """Move to the closest reachable point beside an object, then face it."""
    goals = _interaction_goals(
        controller,
        object_type,
        agent_id,
        future_open_object=future_open_object,
        raw_pose_cache=raw_interaction_pose_cache,
    )
    future_rule = (
        future_open_rule(future_open_object)
        if future_open_object
        else _known_open_rule(controller, object_type)
    )
    if future_rule is not None and future_open_object is None:
        goals = _interaction_goals(
            controller,
            object_type,
            agent_id,
            future_open_object=object_type,
            raw_pose_cache=raw_interaction_pose_cache,
        )

    if preplanned_route is not None:
        path = preplanned_route["path"]
        target_object = preplanned_route["target_object"]
        strategy = preplanned_route["strategy"]
        print(
            f"Optimizer selected {strategy} for robot{agent_id + 1} -> "
            f"{object_type}; following cached path with {len(path)} waypoints"
        )
        if on_path_planned is not None:
            on_path_planned(path)

        event = _follow_path_waiting_for_robot(
            controller,
            robot_camera,
            path,
            on_step,
            step_delay,
            agent_id,
            conflict_resolver_provider=conflict_resolver_provider,
            path_update_provider=path_update_provider,
        )
    else:
        start = _agent_event(controller.last_event, agent_id).metadata["agent"][
            "position"
        ].copy()
        path, target_object = find_padded_astar_path(
            controller,
            start,
            goals,
            grid_size=NAVIGATION_GRID_SIZE,
            padding=NAVIGATION_PADDING,
            agent_id=agent_id,
            avoid_other_agents=False,
            forbidden_regions=(future_rule or {}).get("forbidden_regions"),
        )

        if not path:
            raise RuntimeError(f"No path found to {object_type!r}")

        print(
            f"Navigating robot{agent_id + 1} to {object_type} with padded A*: "
            f"{len(path)} waypoints (robot radius {AGENT_RADIUS:.2f} m + "
            f"padding {NAVIGATION_PADDING:.2f} m; online agent conflicts)"
        )
        if on_path_planned is not None:
            on_path_planned(path)
            if NAVIGATION_PATH_PREVIEW_SECONDS > 0:
                time.sleep(NAVIGATION_PATH_PREVIEW_SECONDS)

        event = _follow_path_waiting_for_robot(
            controller,
            robot_camera,
            path,
            on_step,
            step_delay,
            agent_id,
            conflict_resolver_provider=conflict_resolver_provider,
            path_update_provider=path_update_provider,
        )

    # A coordination route can end at a temporary release pose. Rejoin the
    # task's validated interaction region before reporting navigation done.
    for arrival_attempt in range(3):
        observed = _agent_event(controller.last_event, agent_id)
        position = observed.metadata["agent"]["position"]
        if any(_distance_xz(position, goal) <= 0.04 for goal, _ in goals):
            break
        if arrival_attempt == 2:
            raise NavigationBlockedError(
                f"robot{agent_id + 1} did not reach the interaction region for {object_type}"
            )
        route, target_object = find_padded_astar_path(
            controller, position, goals,
            grid_size=NAVIGATION_GRID_SIZE, padding=NAVIGATION_PADDING,
            agent_id=agent_id, avoid_other_agents=False,
            forbidden_regions=(future_rule or {}).get("forbidden_regions"),
        )
        event = _follow_path_waiting_for_robot(
            controller, robot_camera, route, on_step, step_delay, agent_id,
            conflict_resolver_provider=conflict_resolver_provider,
            path_update_provider=path_update_provider,
        )

    event = _turn_toward(
        controller,
        robot_camera,
        target_object["position"],
        on_step,
        step_delay,
        agent_id,
        rotation_step=NAVIGATION_FINAL_ROTATION_STEP,
    )
    event = aim_robot_camera(controller, target_object["position"], agent_id)
    _notify_step(event, on_step, step_delay)

    position = _agent_event(event, agent_id).metadata["agent"]["position"]
    distance_to_target = _distance_to_object_bounds(position, target_object)
    print(
        f"Robot reached {object_type} at "
        f"x={position['x']:.2f}, z={position['z']:.2f}; "
        f"surface distance={distance_to_target:.2f} m"
    )
    return event
