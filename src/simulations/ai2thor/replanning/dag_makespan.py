"""AI2-THOR remaining-DAG estimation for online conflict candidates.

The estimator is deliberately independent of path generation and candidate
selection.  It overlays temporary release nodes on a snapshot of the runtime
DAG, computes an earliest-start schedule, and then discards that overlay.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from simulations.ai2thor.config import ONLINE_REPLANNING_ROBOT_SPEED


TaskId = int | str


@dataclass(frozen=True)
class DagScheduleEstimate:
    """Timing result for one candidate over the unfinished execution DAG."""

    makespan: float
    task_start_times: dict[TaskId, float]
    task_finish_times: dict[TaskId, float]
    robot_available_times: dict[str, float]
    robot_finish_times: dict[str, float]
    runtime_precedence: tuple[tuple[TaskId, TaskId], ...]
    release_durations: dict[str, float]
    total_travel_distance: float = 0.0
    affected_nodes: tuple[TaskId, ...] = ()
    reused_node_count: int = 0

    def as_metrics(self) -> dict:
        return {
            "estimated_makespan": self.makespan,
            "task_start_times": dict(self.task_start_times),
            "task_finish_times": dict(self.task_finish_times),
            "robot_available_times": dict(self.robot_available_times),
            "robot_finish_times": dict(self.robot_finish_times),
            "runtime_precedence": list(self.runtime_precedence),
            "release_durations": dict(self.release_durations),
            "predicted_total_distance": self.total_travel_distance,
            "affected_nodes": list(self.affected_nodes),
            "recomputed_node_count": len(self.affected_nodes),
            "reused_node_count": self.reused_node_count,
            "estimator": "remaining DAG earliest-start schedule",
        }


def _task_id(task: Mapping[str, Any]) -> int:
    value = task.get("task_id", task.get("id"))
    if type(value) is not int:
        raise ValueError(f"Task has an invalid ID: {value!r}")
    return value


def _simulation_robot(agent_id: int) -> str:
    return f"R{agent_id + 1}"


def _positive_speed(robot_speed, robot: str) -> float:
    speed = (
        robot_speed[robot]
        if isinstance(robot_speed, Mapping)
        else robot_speed
    )
    speed = float(speed)
    if speed <= 0:
        raise ValueError(f"robot speed for {robot} must be positive")
    return speed


def _topological_schedule(
    nodes: Sequence[TaskId],
    predecessors: Mapping[TaskId, set[TaskId]],
    durations: Mapping[TaskId, float],
    robots: Mapping[TaskId, str],
    *,
    baseline: DagScheduleEstimate | None = None,
    affected: set[TaskId] | None = None,
) -> DagScheduleEstimate:
    """Schedule graph nodes with dependency and unary robot constraints."""
    order = {node: index for index, node in enumerate(nodes)}
    successors = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for node in nodes:
        for predecessor in predecessors[node]:
            if predecessor not in successors:
                raise ValueError(
                    f"Remaining DAG references missing node {predecessor!r}"
                )
            if node not in successors[predecessor]:
                successors[predecessor].add(node)
                indegree[node] += 1

    ready = [
        (order[node], node) for node in nodes if indegree[node] == 0
    ]
    heapq.heapify(ready)
    start_times = {}
    finish_times = {}
    robot_available = {}
    processed = 0

    while ready:
        _, node = heapq.heappop(ready)
        robot = robots[node]
        can_reuse = (
            baseline is not None
            and affected is not None
            and node not in affected
            and node in baseline.task_finish_times
        )
        if can_reuse:
            start = baseline.task_start_times[node]
            finish = baseline.task_finish_times[node]
        else:
            dependency_finish = max(
                (finish_times[pred] for pred in predecessors[node]),
                default=0.0,
            )
            start = max(dependency_finish, robot_available.get(robot, 0.0))
            finish = start + durations[node]
        start_times[node] = start
        finish_times[node] = finish
        robot_available[robot] = finish
        processed += 1

        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, (order[successor], successor))

    if processed != len(nodes):
        raise ValueError(
            "Remaining DAG plus runtime precedence contains a cycle"
        )

    robot_finish = {}
    for node, finish in finish_times.items():
        robot = robots[node]
        robot_finish[robot] = max(robot_finish.get(robot, 0.0), finish)
    return DagScheduleEstimate(
        makespan=max(finish_times.values(), default=0.0),
        task_start_times=start_times,
        task_finish_times=finish_times,
        robot_available_times=robot_available,
        robot_finish_times=robot_finish,
        runtime_precedence=(),
        release_durations={},
        total_travel_distance=0.0,
        affected_nodes=tuple(
            nodes if affected is None else (
                node for node in nodes if node in affected
            )
        ),
        reused_node_count=(
            0 if affected is None else len(nodes) - len(affected)
        ),
    )


def estimate_remaining_dag_makespan(
    tasks: Sequence[Mapping[str, Any]],
    dag: Mapping[int, set[int] | Sequence[int]],
    robot_states: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    contexts: Mapping[int, Mapping[str, Any]] | None = None,
    robot_speed=ONLINE_REPLANNING_ROBOT_SPEED,
    task_duration_estimator: Callable[[Mapping[str, Any], str], float]
    | None = None,
    task_route_estimator: Callable[
        [Mapping[str, Any], str, Mapping[str, float] | None],
        tuple[float, Mapping[str, float]],
    ]
    | None = None,
    baseline: DagScheduleEstimate | None = None,
) -> DagScheduleEstimate:
    """Estimate one candidate on the remaining DAG without mutating it.

    ``robot_states`` accepts the executor's task-status snapshot and robot
    plans. ``task_duration_estimator`` supplies action time, while
    ``task_route_estimator`` supplies candidate-aware current and future
    navigation time from predicted robot positions.
    """
    contexts = contexts or {}
    statuses = robot_states.get("task_status", {})
    robot_plans = robot_states.get("robot_plans", {})
    agent_to_robot = robot_states.get("agent_to_robot", {})
    explicit_durations = robot_states.get("task_remaining_durations", {})
    completed_states = {"completed", "action_done"}

    task_by_id = {_task_id(task): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Remaining-DAG input contains duplicate task IDs")
    failed = [
        task_id
        for task_id in task_by_id
        if statuses.get(task_id) == "failed"
    ]
    if failed:
        raise RuntimeError(f"Cannot estimate a failed execution: {failed}")

    remaining = [
        task_id
        for task_id in task_by_id
        if statuses.get(task_id, "pending") not in completed_states
    ]
    remaining_set = set(remaining)
    predecessors = {
        task_id: {
            predecessor
            for predecessor in dag.get(task_id, ())
            if predecessor in remaining_set
        }
        for task_id in remaining
    }
    robots = {
        task_id: str(task_by_id[task_id]["robot"])
        for task_id in remaining
    }

    # Explicitly retain unary robot-resource order even if the caller supplied
    # only semantic JSON dependencies rather than the executor's enriched DAG.
    for robot, plan in robot_plans.items():
        robot_remaining = [task_id for task_id in plan if task_id in remaining_set]
        for predecessor, successor in zip(
            robot_remaining, robot_remaining[1:]
        ):
            predecessors[successor].add(predecessor)

    current_task_by_robot = {}
    for robot, plan in robot_plans.items():
        running = [
            task_id
            for task_id in plan
            if task_id in remaining_set
            and statuses.get(task_id) in {"running", "executing_action"}
        ]
        if running:
            current_task_by_robot[robot] = running[0]
        else:
            pending = [task_id for task_id in plan if task_id in remaining_set]
            if pending:
                current_task_by_robot[robot] = pending[0]

    candidate_path_lengths = candidate.get("current_path_lengths", {})
    candidate_paths = candidate.get("current_paths", {})
    durations = {}
    total_travel_distance = 0.0
    predicted_positions = dict(robot_states.get("robot_positions", {}))
    predicted_object_locations: dict[str, str] = {}
    for task_id in remaining:
        task = task_by_id[task_id]
        robot = robots[task_id]
        status = statuses.get(task_id, "pending")
        duration = explicit_durations.get(task_id)
        if task_duration_estimator is not None:
            duration = task_duration_estimator(task, status)
        if duration is None:
            duration = float(task.get("estimated_duration", 0.0))

        # A candidate supplies a fresh remaining route for each conflict robot.
        # It must override the generic current-task duration estimate.
        matching_agent = next(
            (
                agent_id
                for agent_id, mapped_robot in agent_to_robot.items()
                if mapped_robot == robot
            ),
            None,
        )
        if matching_agent is None and robot.startswith("R"):
            suffix = robot[1:]
            matching_agent = int(suffix) - 1 if suffix.isdigit() else None
        if (
            matching_agent in candidate_path_lengths
            and current_task_by_robot.get(robot) == task_id
            and status == "running"
        ):
            action_duration = float(
                contexts.get(matching_agent, {}).get("action_duration", 0.0)
            )
            duration = (
                float(candidate_path_lengths[matching_agent])
                / _positive_speed(robot_speed, robot)
                + action_duration
            )
            total_travel_distance += float(
                candidate_path_lengths[matching_agent]
            )
            path = candidate_paths.get(matching_agent)
            if path:
                predicted_positions[robot] = dict(path[-1])
        elif status in {"pending", "running"} and task_route_estimator is not None:
            route_task = task
            object_name = task.get("object")
            predicted_location = predicted_object_locations.get(object_name)
            if predicted_location and task.get("action") != "place":
                route_task = dict(task)
                route_task["navigate_to"] = predicted_location
            travel_distance, endpoint = task_route_estimator(
                route_task,
                robot,
                predicted_positions.get(robot),
            )
            travel_distance = float(travel_distance)
            if travel_distance < 0:
                raise ValueError(
                    f"Task {task_id} has a negative navigation distance"
                )
            duration += travel_distance / _positive_speed(robot_speed, robot)
            total_travel_distance += travel_distance
            predicted_positions[robot] = dict(endpoint)
        duration = float(duration)
        if duration < 0:
            raise ValueError(f"Task {task_id} has a negative duration")
        durations[task_id] = duration
        if task.get("action") == "place" and task.get("target"):
            predicted_object_locations[str(task.get("object"))] = str(
                task["target"]
            )

    nodes: list[TaskId] = list(remaining)
    runtime_precedence: list[tuple[TaskId, TaskId]] = []
    release_durations = {}
    previous_release = None
    for index, release in enumerate(candidate.get("release_moves", ()), 1):
        agent_id = int(release["blocker_agent"])
        robot = str(agent_to_robot.get(agent_id, _simulation_robot(agent_id)))
        node = f"RELEASE_{robot}_{index}"
        duration = float(release["release_distance"]) / _positive_speed(
            robot_speed, robot
        )
        total_travel_distance += float(release["release_distance"])
        nodes.append(node)
        predecessors[node] = set()
        robots[node] = robot
        durations[node] = duration
        release_durations[node] = duration
        if previous_release is not None:
            predecessors[node].add(previous_release)
            runtime_precedence.append((previous_release, node))
        previous_release = node

    # Releases are serial. Protect the certified prefix when concurrent
    # continuation is safe; otherwise retain the full-route serial fallback.
    if previous_release is not None:
        priority_agent = int(candidate["priority_agent"])
        priority_robot = str(
            agent_to_robot.get(
                priority_agent, _simulation_robot(priority_agent)
            )
        )
        priority_route_node = f"PRIORITY_ROUTE_{priority_robot}"
        priority_route_duration = float(
            candidate_path_lengths.get(
                priority_agent,
                candidate.get("priority_path_length", 0.0),
            )
        ) / _positive_speed(robot_speed, priority_robot)
        if candidate.get("concurrent_resume_paths") is not None:
            prefix = candidate["priority_release_prefix"]
            priority_route_duration = sum(
                math.hypot(second["x"] - first["x"], second["z"] - first["z"])
                for first, second in zip(prefix, prefix[1:])
            ) / _positive_speed(robot_speed, priority_robot)
        nodes.append(priority_route_node)
        predecessors[priority_route_node] = {previous_release}
        robots[priority_route_node] = priority_robot
        durations[priority_route_node] = priority_route_duration
        runtime_precedence.append((previous_release, priority_route_node))

        # The candidate current-task duration already contained this route.
        # Move it into the explicit runtime node to avoid double counting.
        priority_task = current_task_by_robot.get(priority_robot)
        if priority_task is not None:
            durations[priority_task] = max(
                0.0, durations[priority_task] - priority_route_duration
            )

        affected_agents = set(candidate_path_lengths)
        affected_agents.update(candidate.get("conflict_group", ()))
        affected_agents.add(priority_agent)
        affected_agents.update(
            int(release["blocker_agent"])
            for release in candidate.get("release_moves", ())
        )
        for agent_id in sorted(affected_agents):
            robot = str(
                agent_to_robot.get(
                    int(agent_id), _simulation_robot(int(agent_id))
                )
            )
            current_task = current_task_by_robot.get(robot)
            if current_task is None:
                continue
            predecessors[current_task].add(priority_route_node)
            edge = (priority_route_node, current_task)
            if edge not in runtime_precedence:
                runtime_precedence.append(edge)

    affected = None
    if baseline is not None:
        baseline_nodes = set(baseline.task_finish_times)
        changed = set(nodes) - baseline_nodes
        for node in set(nodes) & baseline_nodes:
            baseline_duration = (
                baseline.task_finish_times[node]
                - baseline.task_start_times[node]
            )
            if abs(durations[node] - baseline_duration) > 1e-9:
                changed.add(node)
        changed.update(successor for _, successor in runtime_precedence)

        successors = {node: set() for node in nodes}
        for node in nodes:
            for predecessor in predecessors[node]:
                successors[predecessor].add(node)
        affected = set(changed)
        frontier = list(changed)
        while frontier:
            node = frontier.pop()
            for successor in successors[node]:
                if successor not in affected:
                    affected.add(successor)
                    frontier.append(successor)

    estimate = _topological_schedule(
        nodes,
        predecessors,
        durations,
        robots,
        baseline=baseline,
        affected=affected,
    )
    return DagScheduleEstimate(
        makespan=estimate.makespan,
        task_start_times=estimate.task_start_times,
        task_finish_times=estimate.task_finish_times,
        robot_available_times=estimate.robot_available_times,
        robot_finish_times=estimate.robot_finish_times,
        runtime_precedence=tuple(runtime_precedence),
        release_durations=release_durations,
        total_travel_distance=total_travel_distance,
        affected_nodes=estimate.affected_nodes,
        reused_node_count=estimate.reused_node_count,
    )
