"""Shared DAG-guided ordering of robot-priority hypotheses."""

from __future__ import annotations

import math


def order_dag_priorities(
    participants,
    active_task_by_robot,
    successors,
    remaining_durations,
    *,
    nominal_finish_times=None,
    nominal_makespan=None,
    debug_logger=None,
):
    """Order robots by ``(slack, -downstream, robot_id)``.

    For active task ``i``, ``downstream`` is the longest remaining predicted
    execution-duration path strictly after ``i``. The score implements

    ``slack_i = T_hat_0 - (F_i + L_down_i)``.

    Navigation-aware nominal finish times may be supplied by either runtime.
    Callers supply navigation plus action time when available and a consistent
    optimistic fallback otherwise. This function only orders hypotheses; it
    never removes a feasible candidate.
    """
    participants = tuple(participants)
    finish_times = dict(nominal_finish_times or {})
    durations = {
        task_id: float(duration)
        for task_id, duration in remaining_durations.items()
    }
    successor_map = {
        task_id: tuple(successor for successor in successors.get(task_id, ())
                       if successor in durations)
        for task_id in durations
    }
    tail_cache = {}
    visiting = set()

    def tail(task_id):
        if task_id in tail_cache:
            return tail_cache[task_id]
        if task_id in visiting:
            raise ValueError("Task dependencies contain a cycle")
        visiting.add(task_id)
        value = durations[task_id] + max(
            (tail(successor) for successor in successor_map[task_id]),
            default=0.0,
        )
        visiting.remove(task_id)
        tail_cache[task_id] = value
        return value

    fallback_horizon = max((tail(task_id) for task_id in durations), default=0.0)
    horizon = (
        fallback_horizon
        if nominal_makespan is None
        else float(nominal_makespan)
    )
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError("nominal_makespan must be finite and non-negative")

    scored = []
    diagnostics = []
    for robot_id in participants:
        active_task = active_task_by_robot.get(robot_id)
        if active_task not in durations:
            downstream = 0.0
            slack = math.inf
        else:
            downstream = tail(active_task) - durations[active_task]
            finish = float(finish_times.get(active_task, durations[active_task]))
            slack = horizon - (finish + downstream)
        score = (slack, -downstream, robot_id)
        scored.append((score, robot_id))
        diagnostics.append({
            "robot_id": robot_id,
            "slack": slack,
            "downstream_duration": downstream,
        })
    ordered = tuple(robot_id for _, robot_id in sorted(scored))
    if debug_logger is not None:
        debug_logger({
            "robots": diagnostics,
            "priority_order": list(ordered),
        })
    return ordered
