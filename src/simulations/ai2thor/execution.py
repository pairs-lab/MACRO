"""Convert a core DAG assignment into executable AI2-THOR actions."""

import heapq
import json
import math
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

from core import order_dag_priorities
from .replanning.dag_makespan import estimate_remaining_dag_makespan
from simulations.ai2thor.config import TASK_ACTION_DURATIONS


SUPPORTED_ACTIONS = {
    "pick",
    "place",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
    "wash",
    "wait",
}


def task_action_duration(task):
    duration = float(task.get("duration", TASK_ACTION_DURATIONS.get(task["action"], 1.0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Task duration must be finite and positive")
    return duration


def load_plan(plan_file) -> dict:
    """Load an optimized final-plan JSON file."""
    path = Path(plan_file).expanduser().resolve()
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Final plan not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid final-plan JSON in {path}: {error}") from error
    if not isinstance(plan, dict):
        raise ValueError("Final plan must be a JSON object.")
    return plan


def _simulation_robot_name(robot: str) -> str:
    suffix = robot[1:] if robot.startswith("R") else ""
    return f"robot{suffix}" if suffix.isdigit() else robot


def _assignment_from_robot_plans(robot_plans: dict) -> dict[int, str]:
    assignment = {}
    for robot, task_ids in robot_plans.items():
        if not isinstance(robot, str) or not isinstance(task_ids, list):
            raise ValueError("robot_plans must map robot names to task ID lists.")
        for task_id in task_ids:
            if type(task_id) is not int:
                raise ValueError(f"Invalid task ID in robot_plans: {task_id!r}.")
            if task_id in assignment:
                raise ValueError(f"Task {task_id} is assigned more than once.")
            assignment[task_id] = robot
    return assignment


def _execution_order(tasks: list[dict], robot_plans: dict) -> list[int]:
    """Topologically sort dependency edges plus per-robot ordering edges."""
    task_by_id = {task.get("id"): task for task in tasks}
    if len(task_by_id) != len(tasks) or None in task_by_id:
        raise ValueError("Final plan contains missing or duplicate task IDs.")

    input_order = {task_id: index for index, task_id in enumerate(task_by_id)}
    successors = {task_id: set() for task_id in task_by_id}
    indegree = {task_id: 0 for task_id in task_by_id}

    def add_edge(predecessor, successor):
        if predecessor not in task_by_id or successor not in task_by_id:
            raise ValueError(
                f"Plan edge references missing task: {predecessor} -> {successor}."
            )
        if successor not in successors[predecessor]:
            successors[predecessor].add(successor)
            indegree[successor] += 1

    for task in tasks:
        dependencies = task.get("node")
        if not isinstance(dependencies, list):
            raise ValueError(f"Task {task['id']} node must be a list.")
        for predecessor in dependencies:
            add_edge(predecessor, task["id"])

    for task_ids in robot_plans.values():
        for predecessor, successor in zip(task_ids, task_ids[1:]):
            add_edge(predecessor, successor)

    ready = [
        (input_order[task_id], task_id)
        for task_id, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    ordered = []
    while ready:
        _, task_id = heapq.heappop(ready)
        ordered.append(task_id)
        for successor in successors[task_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, (input_order[successor], successor))

    if len(ordered) != len(tasks):
        raise ValueError("DAG and robot task order together contain a cycle.")
    return ordered


def build_action_sequence(plan: dict) -> list[dict]:
    """Convert a final plan into ordered, serializable robot commands."""
    tasks = plan.get("tasks")
    robot_plans = plan.get("robot_plans")
    if not isinstance(tasks, list) or not isinstance(robot_plans, dict):
        raise ValueError("Final plan requires tasks and robot_plans.")

    assignment = _assignment_from_robot_plans(robot_plans)
    task_by_id = {task["id"]: task for task in tasks}
    if set(assignment) != set(task_by_id):
        missing = sorted(set(task_by_id) - set(assignment))
        extra = sorted(set(assignment) - set(task_by_id))
        raise ValueError(
            f"robot_plans do not match tasks; missing={missing}, extra={extra}."
        )

    commands = []
    for task_id in _execution_order(tasks, robot_plans):
        task = task_by_id[task_id]
        action = task.get("action")
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action in task {task_id}: {action!r}.")
        target = task.get("target", "")
        navigate_to = task.get("navigate_to") or (
            target
            if action in {"place", "wash"}
            else task.get("object", "")
        )
        if not isinstance(navigate_to, str) or not navigate_to:
            raise ValueError(f"Task {task_id} has no navigation destination.")

        commands.append(
            {
                "task_id": task_id,
                "robot": assignment[task_id],
                "navigate_to": navigate_to,
                "action": action,
                "object": task.get("object", ""),
                "target": target,
                "force_action": bool(task.get("force_action", False)),
                "duration": task_action_duration(task),
            }
        )
        if "wait_for_on_conflict" in task:
            wait_for = task["wait_for_on_conflict"]
            if (not isinstance(wait_for, list) or any(
                    not isinstance(rid, str) or rid not in robot_plans
                    or rid == assignment[task_id] for rid in wait_for)):
                raise ValueError("wait_for_on_conflict must list other assigned robot names")
            commands[-1]["wait_for_on_conflict"] = list(wait_for)
    return commands


class ExecutionPlanExecutor:
    """Dependency-aware executor with one event-driven worker per robot."""

    def __init__(self, plan: dict, navigation, actions):
        self.plan = plan
        self.navigation = navigation
        self.actions = actions
        self.task_pool = build_action_sequence(plan)
        self.task_by_id = {
            task["task_id"]: task for task in self.task_pool
        }
        self.robot_ids = {
            robot: _simulation_robot_name(robot)
            for robot in plan["robot_plans"]
        }
        self.dependency_map = self._build_dependency_map()

        self._condition = threading.Condition(threading.RLock())
        self._task_status = {
            task_id: "pending" for task_id in self.task_by_id
        }
        self._results = {}
        self._failure = None
        self._threads = []
        self._action_started_at = {}
        self._estimate_cache_lock = threading.Lock()
        self._dag_baseline_cache = {}
        self._route_estimate_cache = {}
        self.conflict_events = []
        self.planning_wall_time_s = 0.0
        self.navigation.conflict_runtime = self

    def record_conflict_candidate(self, candidate, participants) -> None:
        """Retain the same per-conflict audit fields as standalone."""
        stats = deepcopy(candidate.get("search_stats", {}))
        planning_time = float(stats.get("planning_wall_time_s", 0.0))
        with self._condition:
            self.planning_wall_time_s += planning_time
            self.conflict_events.append({
                "conflict_id": len(self.conflict_events) + 1,
                "policy": candidate.get("selection_policy", "dag_aware"),
                "participants": sorted(int(value) for value in participants),
                "selected_candidate": candidate.get("candidate_id"),
                "strategy": candidate.get("strategy"),
                "relocations": len(candidate.get("release_moves", ())),
                "search_stats": stats,
            })

    def _build_dependency_map(self) -> dict[int, set[int]]:
        dependencies = {
            task["id"]: set(task.get("node", []))
            for task in self.plan["tasks"]
        }
        # A robot executes its assigned list in optimizer order, even when two
        # consecutive tasks have no explicit edge in the source DAG.
        for task_ids in self.plan["robot_plans"].values():
            for predecessor, successor in zip(task_ids, task_ids[1:]):
                dependencies[successor].add(predecessor)
        return dependencies

    def estimate_conflict_candidate(
        self,
        candidate: dict,
        contexts: dict,
        local_metrics: dict,
    ) -> dict:
        """Evaluate a local motion candidate on the unfinished runtime DAG.

        Candidate-specific navigation is exact for the current conflict group.
        Pending navigation is replanned from predicted per-robot endpoints on
        the static AI2-THOR grid, matching the standalone remaining-DAG model.
        """
        with self._condition:
            statuses = self._task_status.copy()
            action_started_at = self._action_started_at.copy()
        snapshot_time = time.monotonic()

        def estimate_task_duration(task, status):
            duration = task_action_duration(task)
            if status == "executing_action":
                started = action_started_at.get(task["task_id"], snapshot_time)
                return max(0.0, duration - (snapshot_time - started))
            return duration

        agent_to_robot = {}
        for robot in self.robot_ids:
            suffix = robot[1:] if robot.startswith("R") else ""
            if suffix.isdigit():
                agent_to_robot[int(suffix) - 1] = robot

        robot_to_agent = {
            robot: agent_id for agent_id, robot in agent_to_robot.items()
        }
        route_estimator = getattr(self.navigation, "estimate_task_route", None)
        state_signature = tuple(sorted(statuses.items()))

        def estimate_task_route(task, robot, start_position):
            if route_estimator is None:
                raise RuntimeError(
                    "AI2-THOR future-route estimator is unavailable"
                )
            agent_id = robot_to_agent[robot]
            deferred_open = self._deferred_open_dependencies(task)
            start_key = (
                None
                if start_position is None
                else (
                    round(float(start_position.get("x", 0.0)), 4),
                    round(float(start_position.get("z", 0.0)), 4),
                )
            )
            route_key = (
                state_signature,
                agent_id,
                str(task["navigate_to"]).casefold(),
                start_key,
                str(task["target"]).casefold() if deferred_open else None,
            )
            with self._estimate_cache_lock:
                cached_route = self._route_estimate_cache.get(route_key)
            if cached_route is not None:
                distance, endpoint = cached_route
                return distance, dict(endpoint) if endpoint is not None else None
            route = route_estimator(
                agent_id,
                task["navigate_to"],
                start_position,
                task["target"] if deferred_open else None,
            )
            path = route["path"]
            distance = sum(
                math.hypot(
                    second["x"] - first["x"],
                    second["z"] - first["z"],
                )
                for first, second in zip(path, path[1:])
            )
            endpoint = path[-1] if path else start_position
            with self._estimate_cache_lock:
                if len(self._route_estimate_cache) >= 2048:
                    self._route_estimate_cache.pop(
                        next(iter(self._route_estimate_cache))
                    )
                self._route_estimate_cache[route_key] = (
                    distance,
                    dict(endpoint) if endpoint is not None else None,
                )
            return distance, endpoint

        positions_provider = getattr(self.navigation, "robot_positions", None)
        robot_positions = positions_provider() if positions_provider else {}

        estimator_arguments = (
            list(self.task_by_id.values()),
            self.dependency_map,
            {
                "task_status": statuses,
                "robot_plans": self.plan["robot_plans"],
                "agent_to_robot": agent_to_robot,
                "robot_positions": robot_positions,
            },
        )
        estimator_keywords = {
            "contexts": contexts,
            "task_duration_estimator": estimate_task_duration,
            "task_route_estimator": (
                estimate_task_route if route_estimator is not None else None
            ),
        }
        cache_key = (
            tuple(sorted(statuses.items())),
            tuple(
                sorted(
                    (
                        robot,
                        round(float(position.get("x", 0.0)), 4),
                        round(float(position.get("z", 0.0)), 4),
                    )
                    for robot, position in robot_positions.items()
                )
            ),
            tuple(sorted(int(agent) for agent in candidate.get("conflict_group", ()))),
        )
        with self._estimate_cache_lock:
            baseline = self._dag_baseline_cache.get(cache_key)
        baseline_cache_hit = baseline is not None
        if baseline is None:
            baseline = estimate_remaining_dag_makespan(
                *estimator_arguments,
                {
                    "strategy": "nominal",
                    "release_moves": (),
                    "current_path_lengths": {},
                    "current_paths": {},
                },
                **estimator_keywords,
            )
            with self._estimate_cache_lock:
                # Candidate batches are short; retaining only recent snapshots
                # prevents stale mission states from accumulating indefinitely.
                if len(self._dag_baseline_cache) >= 8:
                    self._dag_baseline_cache.pop(
                        next(iter(self._dag_baseline_cache))
                    )
                self._dag_baseline_cache[cache_key] = baseline
        estimate = estimate_remaining_dag_makespan(
            *estimator_arguments,
            candidate,
            baseline=baseline,
            **estimator_keywords,
        )
        propagated_delay = sum(
            max(
                0.0,
                finish - baseline.task_finish_times.get(task_id, finish),
            )
            for task_id, finish in estimate.task_finish_times.items()
            if isinstance(task_id, int)
        )
        additional_distance = max(
            0.0,
            estimate.total_travel_distance
            - baseline.total_travel_distance,
        )
        return {
            **local_metrics,
            **estimate.as_metrics(),
            "total_distance": estimate.total_travel_distance,
            "additional_distance": additional_distance,
            "total_delay": propagated_delay,
            "nominal_makespan": baseline.makespan,
            "nominal_total_distance": baseline.total_travel_distance,
            "baseline_cache_hit": baseline_cache_hit,
        }

    def order_conflict_priorities(
        self,
        participants,
        contexts=None,
    ) -> tuple[int, ...]:
        """Order runtime hypotheses by task slack and downstream work.

        This is the parameter-free ordering from Eq. (8).  Navigation is
        deliberately omitted here: this function only orders hypotheses; the
        complete candidate evaluator still computes candidate-specific A*
        travel and the full remaining-DAG makespan.
        """
        with self._condition:
            statuses = self._task_status.copy()
            action_started_at = self._action_started_at.copy()
        now = time.monotonic()
        completed = {"completed", "action_done", "failed"}
        remaining = {
            task_id
            for task_id, status in statuses.items()
            if status not in completed
        }
        successors = {task_id: set() for task_id in remaining}
        for task_id in remaining:
            for predecessor in self.dependency_map.get(task_id, ()):
                if predecessor in remaining:
                    successors[predecessor].add(task_id)

        durations = {}
        for task_id in remaining:
            duration = task_action_duration(self.task_by_id[task_id])
            if statuses.get(task_id) == "executing_action":
                duration = max(
                    0.0,
                    duration - (now - action_started_at.get(task_id, now)),
                )
            durations[task_id] = duration

        nominal_finishes = None
        horizon = None
        if contexts:
            try:
                nominal = self.estimate_conflict_candidate(
                    {
                        "strategy": "nominal",
                        "conflict_group": tuple(participants),
                        "release_moves": (),
                        "current_path_lengths": {},
                        "current_paths": {},
                    },
                    contexts,
                    {
                        "estimated_makespan": 0.0,
                        "total_distance": 0.0,
                        "total_delay": 0.0,
                    },
                )
                nominal_finishes = nominal["task_finish_times"]
                horizon = nominal["estimated_makespan"]
                durations.update(
                    {
                        task_id: (
                            nominal["task_finish_times"][task_id]
                            - nominal["task_start_times"][task_id]
                        )
                        for task_id in remaining
                        if task_id in nominal["task_finish_times"]
                    }
                )
            except RuntimeError:
                # Ordering is a search heuristic only. Candidate feasibility
                # and selection remain governed by the full evaluator.
                nominal_finishes = None
        active_task_by_agent = {}
        for agent_id in participants:
            robot = f"R{int(agent_id) + 1}"
            active_task_by_agent[agent_id] = next(
                (
                    task_id
                    for task_id in self.plan["robot_plans"].get(robot, ())
                    if task_id in remaining
                ),
                None,
            )
        return order_dag_priorities(
            participants,
            active_task_by_agent,
            successors,
            durations,
            nominal_finish_times=nominal_finishes,
            nominal_makespan=horizon,
        )

    def mark_task_completed(self, task_id: int, result) -> None:
        robot = self.task_by_id[task_id]["robot"]
        with self._condition:
            self._task_status[task_id] = "completed"
            self._results[task_id] = result
            self._condition.notify_all()
        print(
            f"Released workspace after T{task_id}; dependent robots may "
            "start moving"
        )
        task_finished = getattr(self.navigation, "task_finished", None)
        if task_finished is not None:
            task_finished(self.robot_ids[robot])

    def _get_next_available_task(self, agent: str):
        """Claim this agent's first pending task whose predecessors completed."""
        with self._condition:
            for task in self.task_pool:
                task_id = task["task_id"]
                if task["robot"] != agent:
                    continue
                if self._task_status[task_id] != "pending":
                    continue
                if not all(
                    self._task_status.get(dependency) == "completed"
                    for dependency in self._navigation_dependencies(task)
                ):
                    continue
                self._task_status[task_id] = "running"
                return task
        return None

    def _deferred_open_dependencies(self, task: dict) -> set[int]:
        """Allow carrier navigation and a different robot's open task to overlap."""
        if task["action"] != "place":
            return set()
        return {
            dependency
            for dependency in self.dependency_map[task["task_id"]]
            if self.task_by_id[dependency]["action"] == "open"
            and self.task_by_id[dependency]["object"].casefold()
            == task["target"].casefold()
            and self.task_by_id[dependency]["robot"] != task["robot"]
        }

    def _navigation_dependencies(self, task: dict) -> set[int]:
        return (
            self.dependency_map[task["task_id"]]
            - self._deferred_open_dependencies(task)
        )

    def _wait_for_action_dependencies(self, task: dict) -> None:
        """At the goal, wait until concurrent prerequisites have completed."""
        with self._condition:
            while not all(
                self._task_status.get(dependency) == "completed"
                for dependency in self.dependency_map[task["task_id"]]
            ):
                if self._failure is not None:
                    raise RuntimeError("A task dependency failed")
                self._condition.wait(timeout=0.05)

    def estimate_remaining_action_time(self, robot: str) -> float:
        """Estimate unfinished manipulation time from this robot's task chain."""
        with self._condition:
            total = 0.0
            now = time.monotonic()
            for task_id in self.plan["robot_plans"].get(robot, []):
                status = self._task_status[task_id]
                if status in {"completed", "action_done", "failed"}:
                    continue
                duration = task_action_duration(self.task_by_id[task_id])
                if status == "executing_action":
                    elapsed = now - self._action_started_at.get(task_id, now)
                    duration = max(0.0, duration - elapsed)
                total += duration
            return total

    def _wait_for_action_duration(self, task_id: int, duration: float) -> None:
        """Keep an atomic task active for its configured physical duration."""
        deadline = self._action_started_at[task_id] + duration
        with self._condition:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                if self._failure is not None:
                    raise RuntimeError("Another robot task failed")
                # Condition.wait releases the scheduler lock, so other robot
                # workers continue executing concurrently during this action.
                self._condition.wait(timeout=remaining)

    def wait_until_robot_available(self, robot: str) -> None:
        """Wait until the blocker has completed its current atomic action."""
        with self._condition:
            while any(
                self._task_status[task_id]
                in {"running", "executing_action", "action_done"}
                for task_id in self.plan["robot_plans"].get(robot, [])
            ):
                if self._failure is not None:
                    raise RuntimeError(
                        f"{robot} failed before becoming available"
                    )
                self._condition.wait(timeout=0.05)

    def wait_until_robot_finished(self, robot: str) -> None:
        """Scene-authored contact waits include gaps between atomic tasks."""
        with self._condition:
            while any(self._task_status[tid] != "completed"
                      for tid in self.plan["robot_plans"].get(robot, ())):
                if self._failure is not None:
                    raise RuntimeError(f"{robot} failed during contact wait")
                self._condition.wait(timeout=0.05)

    def _execute_task(self, task: dict, previous_result=None):
        """Navigate and execute one atomic task; previous_result is extensible."""
        del previous_result
        robot = self.robot_ids[task["robot"]]
        robot_locks = getattr(self.navigation, "robot_locks", {})
        with robot_locks.get(robot, nullcontext()):
            set_keyframe = getattr(self.navigation, "set_keyframe", None)
            if set_keyframe is not None:
                destination = (
                    f" -> {task['target']}" if task.get("target") else ""
                )
                set_keyframe(
                    f"T{task['task_id']}: {task['robot']} "
                    f"{task['action']} {task['object']}{destination}"
                )
            print(
                f"Executing T{task['task_id']}: {task['robot']} "
                f"{task['action']} {task['object']}"
            )
            deferred_open = self._deferred_open_dependencies(task)
            future_open_object = (
                task["target"] if deferred_open else None
            )
            if future_open_object:
                print(
                    f"Rule-based future obstacle: robot route reserves the "
                    f"open {future_open_object} door swing"
                )
            # Never infer position from the previous destination name. A robot
            # may have parked elsewhere to clear another agent's workspace.
            # A* cheaply returns a one-point path when it is genuinely still
            # at the correct interaction pose.
            stow = getattr(self.actions, "stow_for_navigation", None)
            if callable(stow):
                stow(robot)
            # Waiting is a temporal operation at the current pose, not a
            # fresh interaction trip (a just-closed door may hide its target).
            if task["action"] != "wait":
                self.navigation(
                    robot,
                    task["navigate_to"],
                    future_open_object=future_open_object,
                    action_duration=task_action_duration(task),
                    **({"wait_for_on_conflict": tuple(
                        self.robot_ids[rid] for rid in task["wait_for_on_conflict"]
                    )} if task.get("wait_for_on_conflict") else {}),
                )
            if set_keyframe is not None:
                set_keyframe(
                    f"T{task['task_id']}: {task['robot']} executes "
                    f"{task['action']} {task['object']}"
                )
            self._wait_for_action_dependencies(task)

            handlers = {
                "pick": lambda: self.actions.pick(
                    robot,
                    task["object"],
                    force_action=task.get("force_action", False),
                ),
                "place": lambda: self.actions.place(robot, task["target"]),
                "open": lambda: self.actions.open(robot, task["object"]),
                "close": lambda: self.actions.close(robot, task["object"]),
                "toggle_on": lambda: self.actions.switch_on(
                    robot, task["object"]
                ),
                "toggle_off": lambda: self.actions.switch_off(
                    robot, task["object"]
                ),
                "wash": lambda: self.actions.wash(robot, task["object"]),
                "wait": lambda: None,
            }
            with self._condition:
                self._task_status[task["task_id"]] = "executing_action"
                self._action_started_at[task["task_id"]] = time.monotonic()
                self._condition.notify_all()
            result = handlers[task["action"]]()
            action_duration = task_action_duration(task)
            self._wait_for_action_duration(
                task["task_id"],
                action_duration,
            )
            print(
                f"Completed action T{task['task_id']}: {task['robot']} "
                f"{task['action']} {task['object']} "
                f"({action_duration:.1f}s)"
            )
            if set_keyframe is not None:
                set_keyframe(
                    f"Completed T{task['task_id']}: {task['robot']} "
                    f"{task['action']} {task['object']}"
                )
            with self._condition:
                self._task_status[task["task_id"]] = "action_done"
                self._condition.notify_all()
            return result

    def _all_tasks_finished(self) -> bool:
        return all(
            status == "completed" for status in self._task_status.values()
        )

    def _agent_worker(self, agent: str) -> None:
        previous_result = None
        while True:
            with self._condition:
                if self._failure is not None or self._all_tasks_finished():
                    return

                task = self._get_next_available_task(agent)
                if task is None:
                    # notify_all() wakes this worker immediately when a
                    # predecessor completes; timeout also protects lost wakes.
                    self._condition.wait(timeout=0.05)
                    continue

            try:
                robot = self.robot_ids[agent]
                robot_locks = getattr(self.navigation, "robot_locks", {})
                # Keep ownership of this robot across the atomic transition:
                # action complete -> workspace released -> parking begins.
                # Other robots are woken by mark_task_completed and may start
                # their A* paths concurrently, but cannot issue a second move
                # command for this same robot.
                with robot_locks.get(robot, nullcontext()):
                    previous_result = self._execute_task(
                        task,
                        previous_result,
                    )
                    self.mark_task_completed(
                        task["task_id"],
                        previous_result,
                    )
            except BaseException as error:
                abort_replanning = getattr(
                    self.navigation,
                    "abort_replanning",
                    None,
                )
                if abort_replanning is not None:
                    abort_replanning()
                with self._condition:
                    self._task_status[task["task_id"]] = "failed"
                    if self._failure is None:
                        self._failure = error
                    self._condition.notify_all()
                return

    def run(self) -> list:
        """Start one worker thread per assigned robot and wait for completion."""
        prepare_2d_planning = getattr(
            self.navigation,
            "prepare_2d_planning",
            None,
        )
        if prepare_2d_planning is not None:
            prepare_2d_planning(
                task["navigate_to"] for task in self.task_pool
            )
        for agent in self.robot_ids:
            thread = threading.Thread(
                target=self._agent_worker,
                args=(agent,),
                name=f"executor-{agent}",
            )
            thread.start()
            self._threads.append(thread)

        pump_events = getattr(self.navigation, "pump_events", None)
        while any(thread.is_alive() for thread in self._threads):
            if pump_events is not None:
                pump_events()
            else:
                time.sleep(0.01)
            for thread in self._threads:
                thread.join(timeout=0.001)
        if pump_events is not None:
            pump_events()

        if self._failure is not None:
            raise RuntimeError("Execution plan failed") from self._failure
        if not self._all_tasks_finished():
            raise RuntimeError("Execution stopped before every task completed")
        return [
            self._results[task["task_id"]] for task in self.task_pool
        ]


def execute_plan(plan_or_file, navigation, actions) -> list:
    """Load a plan and execute it concurrently with dependency-aware workers."""
    plan = (
        load_plan(plan_or_file)
        if not isinstance(plan_or_file, dict)
        else plan_or_file
    )
    return ExecutionPlanExecutor(plan, navigation, actions).run()
