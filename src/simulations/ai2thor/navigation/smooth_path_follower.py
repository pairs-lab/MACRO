"""Collision-checked AI2-THOR look-ahead path follower."""

import math
import threading
import time

from simulations.ai2thor.cameras import move_robot
from simulations.ai2thor.config import (
    ONLINE_CONFLICT_LOOKAHEAD_BUFFER,
    ROBOT_SAFETY_RADIUS,
    SMOOTH_HEADING_TOLERANCE,
    SMOOTH_FINAL_POSITION_TOLERANCE,
    SMOOTH_LOOKAHEAD_DISTANCE,
    SMOOTH_LARGE_ROTATION_STEP,
    SMOOTH_MAX_HEADING_ERROR,
    SMOOTH_MOVE_STEP,
    SMOOTH_ROTATION_STEP,
    SMOOTH_SAMPLE_SPACING,
    carried_object_safety_margin,
)


# AI2-THOR exposes one shared controller to all agent workers.  Serialize the
# final obstacle check with the following translation so another navigation
# thread cannot move between those two operations.
_MOTION_STEP_LOCK = threading.RLock()


def _distance(first, second):
    return math.hypot(second["x"] - first["x"], second["z"] - first["z"])


def _signed_angle(current, target):
    return (target - current + 180) % 360 - 180


def _yaw_to(first, second):
    return math.degrees(
        math.atan2(second["x"] - first["x"], second["z"] - first["z"])
    ) % 360


def _agent_event(event, agent_id):
    events = getattr(event, "events", None)
    if events and len(events) > 1:
        return events[agent_id]
    return event


def _is_carrying(event):
    """Return whether the agent has a physical payload in AI2-THOR."""
    return bool(event.metadata.get("inventoryObjects"))


def _tracking_limits(event, requested_lookahead):
    """Keep carried payloads on the collision-checked A* polyline.

    Ordinary navigation may use a longer pure-pursuit look-ahead to make the
    rendered motion smooth.  That look-ahead cuts the inside of a corner,
    however, so the agent body can remain clear while a held receptacle hits a
    wall.  A carrying agent therefore advances one densified path sample at a
    time.  This changes only path execution; it does not change A*, candidate
    feasibility, or DAG scoring.
    """
    if _is_carrying(event):
        return (
            min(requested_lookahead, SMOOTH_SAMPLE_SPACING),
            min(SMOOTH_MOVE_STEP, SMOOTH_SAMPLE_SPACING),
        )
    return requested_lookahead, SMOOTH_MOVE_STEP


def _bounded_move_magnitude(position, target, goal, move_step):
    """Bound a translation by both the next path sample and final goal."""
    return min(
        move_step,
        _distance(position, goal),
        _distance(position, target),
    )


def _agent_blocking_ahead(
    controller,
    agent_id,
    position,
    target,
    observation_exempt_agent_ids=(),
):
    """Detect another agent in the forward safety corridor before contact."""
    events = getattr(controller.last_event, "events", None) or []
    direction_x = target["x"] - position["x"]
    direction_z = target["z"] - position["z"]
    direction_length = max(math.hypot(direction_x, direction_z), 1e-9)
    moving_event = events[agent_id] if agent_id < len(events) else None
    moving_margin = (
        carried_object_safety_margin(moving_event) if moving_event else 0.0
    )
    for other_id, other_event in enumerate(events):
        if other_id == agent_id:
            continue
        payload_margin = max(
            moving_margin,
            carried_object_safety_margin(other_event),
        )
        observation_buffer = (
            0.0
            if other_id in observation_exempt_agent_ids
            else ONLINE_CONFLICT_LOOKAHEAD_BUFFER
        )
        protected_distance = 2 * ROBOT_SAFETY_RADIUS + payload_margin
        # Reserve a complete discrete step plus final-position tolerance so a
        # robot cannot jump across the protected boundary.
        stopping_distance = (
            protected_distance
            + SMOOTH_MOVE_STEP
            + SMOOTH_FINAL_POSITION_TOLERANCE
            + observation_buffer
        )
        other = other_event.metadata["agent"]["position"]
        offset_x = other["x"] - position["x"]
        offset_z = other["z"] - position["z"]
        distance = math.hypot(offset_x, offset_z)
        if distance > stopping_distance:
            continue
        forward = (
            offset_x * direction_x + offset_z * direction_z
        ) / direction_length
        lateral_squared = max(0.0, distance * distance - forward * forward)
        if forward > 0 and math.sqrt(lateral_squared) <= protected_distance:
            return other_id
    return None


def _interpolate_polyline(path, spacing):
    """Densify path segments without changing the collision-safe polyline."""
    if len(path) <= 1:
        return [point.copy() for point in path]

    samples = [path[0].copy()]
    for first, second in zip(path, path[1:]):
        length = _distance(first, second)
        count = max(1, int(math.ceil(length / spacing)))
        first_y = first.get("y", second.get("y", 0.0))
        second_y = second.get("y", first_y)
        for index in range(1, count + 1):
            ratio = index / count
            samples.append(
                {
                    "x": first["x"] + (second["x"] - first["x"]) * ratio,
                    "y": first_y + (second_y - first_y) * ratio,
                    "z": first["z"] + (second["z"] - first["z"]) * ratio,
                }
            )
    return samples


def _lookahead_index(samples, closest_index, lookahead):
    travelled = 0.0
    for index in range(closest_index + 1, len(samples)):
        travelled += _distance(samples[index - 1], samples[index])
        if travelled >= lookahead:
            return index
    return len(samples) - 1


def follow_smooth_path(
    controller,
    robot_camera,
    path,
    on_step=None,
    step_delay=0.0,
    agent_id=0,
    blocked_error_cls=RuntimeError,
    path_update_provider=None,
    lookahead_distance=SMOOTH_LOOKAHEAD_DISTANCE,
    observation_exempt_agent_ids=(),
):
    """Follow an existing path with small steering corrections and no replanning."""
    if not path:
        raise ValueError("A smooth path must contain at least one point")
    if len(path) == 1:
        return _agent_event(controller.last_event, agent_id)

    samples = _interpolate_polyline(path, SMOOTH_SAMPLE_SPACING)
    closest_index = 0
    path_length = sum(
        _distance(first, second) for first, second in zip(samples, samples[1:])
    )
    maximum_iterations = max(50, int(path_length / SMOOTH_MOVE_STEP) * 12)
    iteration = 0

    while iteration < maximum_iterations:
        iteration += 1
        if path_update_provider is not None:
            replacement_path = path_update_provider(agent_id)
            if replacement_path:
                samples = _interpolate_polyline(
                    replacement_path,
                    SMOOTH_SAMPLE_SPACING,
                )
                closest_index = 0
                replacement_length = sum(
                    _distance(first, second)
                    for first, second in zip(samples, samples[1:])
                )
                # A realtime A* replacement can be longer than the original
                # route. Give the follower a fresh iteration budget instead of
                # inheriting the almost exhausted budget of the old path.
                maximum_iterations = iteration + max(
                    50,
                    int(replacement_length / SMOOTH_MOVE_STEP) * 12,
                )
        event = _agent_event(controller.last_event, agent_id)
        agent = event.metadata["agent"]
        position = agent["position"]
        if _distance(position, samples[-1]) <= SMOOTH_FINAL_POSITION_TOLERANCE:
            if path_update_provider is not None:
                replacement_path = path_update_provider(
                    agent_id,
                    route_completed=True,
                )
                if replacement_path:
                    samples = _interpolate_polyline(
                        replacement_path,
                        SMOOTH_SAMPLE_SPACING,
                    )
                    closest_index = 0
                    replacement_length = sum(
                        _distance(first, second)
                        for first, second in zip(samples, samples[1:])
                    )
                    maximum_iterations = iteration + max(
                        50,
                        int(replacement_length / SMOOTH_MOVE_STEP) * 12,
                    )
                    continue
            return event

        # Keep progress monotonic, but resume correctly if execution was paused.
        closest_index = min(
            range(closest_index, len(samples)),
            key=lambda index: _distance(position, samples[index]),
        )
        effective_lookahead, move_step = _tracking_limits(
            event,
            lookahead_distance,
        )
        target_index = _lookahead_index(
            samples,
            closest_index,
            effective_lookahead,
        )
        target = samples[target_index]
        blocking_agent_id = _agent_blocking_ahead(
            controller,
            agent_id,
            position,
            target,
            observation_exempt_agent_ids,
        )
        if blocking_agent_id is not None:
            raise blocked_error_cls(
                "Robot path is occupied inside the safety corridor. "
                f"Unity reported: Agent {blocking_agent_id} is blocking "
                f"Agent {agent_id}."
            )
        desired_yaw = _yaw_to(position, target)
        yaw_error = _signed_angle(agent["rotation"]["y"], desired_yaw)

        if abs(yaw_error) > SMOOTH_HEADING_TOLERANCE:
            action = "RotateRight" if yaw_error > 0 else "RotateLeft"
            rotation_step = (
                SMOOTH_LARGE_ROTATION_STEP
                if abs(yaw_error) > SMOOTH_MAX_HEADING_ERROR
                else SMOOTH_ROTATION_STEP
            )
            rotated = move_robot(
                controller,
                robot_camera,
                action,
                report_failure=False,
                agent_id=agent_id,
                # Steering and translation form one visual control cycle.  Do
                # not render an intermediate frame in which the robot appears
                # to stop and rotate before every forward step.
                render_image=False,
                degrees=min(rotation_step, abs(yaw_error)),
            )
            if rotated is None:
                message = controller.last_event.metadata.get(
                    "errorMessage", "Unknown collision"
                )
                raise blocked_error_cls(
                    f"Robot could not steer along its path: {message}"
                )
        # At a sharp corner, finish the turn internally before translating.
        # The UI receives the next translated pose instead of several static
        # rotation frames, so the same A* path looks continuous.
        if abs(yaw_error) > SMOOTH_MAX_HEADING_ERROR:
            continue

        with _MOTION_STEP_LOCK:
            # Re-read every agent pose immediately before translation. Another
            # worker may have moved while this robot was steering.
            latest_event = _agent_event(controller.last_event, agent_id)
            latest_position = latest_event.metadata["agent"]["position"]
            blocking_agent_id = _agent_blocking_ahead(
                controller,
                agent_id,
                latest_position,
                target,
                observation_exempt_agent_ids,
            )
            if blocking_agent_id is not None:
                raise blocked_error_cls(
                    "Robot path is occupied inside the stopping corridor. "
                    f"Unity reported: Agent {blocking_agent_id} is blocking "
                    f"Agent {agent_id}."
                )

            # Never step past the current tracking target.  In particular, a
            # 0.10 m MoveAhead must not overshoot a 0.05 m A* corner sample and
            # turn the checked polyline into an unchecked diagonal chord.
            move_magnitude = _bounded_move_magnitude(
                latest_position,
                target,
                samples[-1],
                move_step,
            )
            if move_magnitude <= 1e-6:
                continue
            moved = move_robot(
                controller,
                robot_camera,
                "MoveAhead",
                report_failure=False,
                agent_id=agent_id,
                moveMagnitude=move_magnitude,
            )
        if moved is None:
            message = controller.last_event.metadata.get(
                "errorMessage", "Unknown collision"
            )
            raise blocked_error_cls(
                "Robot path is physically blocked; no object was removed. "
                f"Unity reported: {message}"
            )
        if on_step is not None:
            on_step(moved)
        if step_delay > 0:
            time.sleep(step_delay)

    raise blocked_error_cls("Smooth path follower did not converge")
