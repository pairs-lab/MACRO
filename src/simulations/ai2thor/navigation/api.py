"""AI2-THOR navigation command binding."""

import math
import threading
from itertools import islice

from ..replanning.multi_agent_scheduler import MultiAgentReplanScheduler
from ..replanning.minimum_disruption import select_minimum_disruption_plan
from simulations.ai2thor.config import (
    AGENT_COUNT,
    NAVIGATION_STEP_DELAY,
    ONLINE_RELEASE_CANDIDATE_COUNT,
    RUNTIME_CONFLICT_POLICY,
    RUNTIME_PLANNING_TIMEOUT,
    ROBOT_SAFETY_RADIUS,
    SMOOTH_FINAL_POSITION_TOLERANCE,
    SMOOTH_MOVE_STEP,
    SMOOTH_SAMPLE_SPACING,
    carried_object_safety_margin,
)

from .navigation import (
    NavigationBlockedError,
    _interaction_goals,
    navigate_to_object_type,
    plan_realtime_route,
)
from .door_rules import future_open_rule
from .smooth_path_follower import follow_smooth_path


# Names are generated from AGENT_COUNT; aliases keep existing user code valid.
robot1 = "robot1"
robot2 = "robot2"
robot3 = "robot3"

ROBOT_NAMES = tuple(f"robot{index + 1}" for index in range(AGENT_COUNT))
ROBOT_AGENT_IDS = {
    robot: agent_id for agent_id, robot in enumerate(ROBOT_NAMES)
}


def robot_name(agent_number: int) -> str:
    """Return the public robot name for a one-based agent number."""
    if not 1 <= agent_number <= AGENT_COUNT:
        raise ValueError(
            f"agent_number must be between 1 and {AGENT_COUNT}"
        )
    return ROBOT_NAMES[agent_number - 1]


def create_navigation_function(
    controller,
    robot_camera,
    on_step=None,
    on_path_planned=None,
):
    """Create navigation(robot, object_type) bound to the running simulation."""
    # Delay this import until both packages are initialized: spatial_conflict
    # reuses navigation.grid_planner and must not pull navigation.api in a cycle.
    from ..replanning.spatial_conflict import (
        candidate_paths_are_safe,
        iter_minimum_release_motions,
        rank_blockers_on_path,
    )

    agent_locks = {
        agent_id: threading.RLock() for agent_id in ROBOT_AGENT_IDS.values()
    }
    failed_release_directions = set()
    release_failures_lock = threading.RLock()
    planning_route_cache = threading.local()
    scene_interaction_pose_cache = {}

    def _freeze_route_input(value):
        if isinstance(value, dict):
            return tuple(
                (str(key), _freeze_route_input(item))
                for key, item in sorted(
                    value.items(), key=lambda pair: str(pair[0])
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(_freeze_route_input(item) for item in value)
        if isinstance(value, float):
            return round(value, 9)
        return value

    def _clone_route(route):
        cloned = dict(route)
        cloned["path"] = [dict(point) for point in route.get("path", ())]
        return cloned

    def _release_direction(release):
        if "direction_group" in release:
            return int(release["direction_group"])
        path = release["release_path"]
        if len(path) < 2:
            return 0
        dx = path[1]["x"] - path[0]["x"]
        dz = path[1]["z"] - path[0]["z"]
        return int(round(math.atan2(dz, dx) / (math.pi / 4))) % 8

    def _release_key(priority_id, blocker_id, context, release):
        return (
            priority_id,
            blocker_id,
            context["object_type"].casefold(),
            _release_direction(release),
        )

    def _plan_batch_route(
        agent_id,
        context,
        position_overrides=None,
        avoid_other_agents=True,
    ):
        cache = getattr(planning_route_cache, "routes", None)
        key = (
            int(agent_id),
            str(context["object_type"]).casefold(),
            str(context.get("future_open_object") or "").casefold(),
            _freeze_route_input(position_overrides or {}),
            bool(avoid_other_agents),
        )
        if cache is not None and key in cache:
            value = cache[key]
            if isinstance(value, RuntimeError):
                raise RuntimeError(str(value))
            return _clone_route(value)
        try:
            route = plan_realtime_route(
                controller,
                context["object_type"],
                agent_id,
                future_open_object=context.get("future_open_object"),
                agent_position_overrides=position_overrides,
                avoid_other_agents=avoid_other_agents,
                interaction_goal_cache=getattr(
                    planning_route_cache,
                    "goals",
                    None,
                ),
                raw_interaction_pose_cache=scene_interaction_pose_cache,
                timing=getattr(planning_route_cache, "timing", None),
            )
        except RuntimeError as error:
            if cache is not None:
                cache[key] = error
            raise
        if cache is not None:
            cache[key] = _clone_route(route)
        return _clone_route(route)

    def wait_for_stationary_blocker(blocking_agent_id):
        runtime = getattr(navigation, "conflict_runtime", None)
        if runtime is None:
            raise RuntimeError("Execution runtime is unavailable for waiting")
        runtime.wait_until_robot_available(f"R{blocking_agent_id + 1}")

    def wait_for_contact_blocker(blocking_agent_id):
        runtime = getattr(navigation, "conflict_runtime", None)
        if runtime is None:
            raise RuntimeError("Execution runtime is unavailable for contact waiting")
        runtime.wait_until_robot_finished(f"R{blocking_agent_id + 1}")

    def _rank_hypothesis_blockers(
        priority_id,
        group,
        preferred_path,
        position_overrides,
    ):
        return rank_blockers_on_path(
            controller,
            priority_id,
            group,
            preferred_path,
            position_overrides=position_overrides,
        )

    def _plan_minimum_release(
        priority_id,
        blocker_id,
        preferred_path,
        position_overrides,
        group,
        context,
        blocker_context=None,
    ):
        rule_object = context.get("future_open_object")
        if rule_object is None and any(
            scene_object.get("objectType", "").casefold()
            == context["object_type"].casefold()
            and scene_object.get("isOpen", False)
            for scene_object in controller.last_event.metadata.get("objects", [])
        ):
            rule_object = context["object_type"]
        open_rule = future_open_rule(rule_object) if rule_object else None
        remaining_blockers = [
            agent_id
            for agent_id in _rank_hypothesis_blockers(
                priority_id,
                group,
                preferred_path,
                position_overrides,
            )
            if agent_id != blocker_id
        ]
        feasible_releases = []
        blocker_goal_path = None
        if blocker_context is not None:
            try:
                blocker_goal_path = _plan_batch_route(
                    blocker_id,
                    blocker_context,
                    position_overrides,
                    False,
                )["path"]
            except RuntimeError:
                blocker_goal_path = None
        try:
            for release in iter_minimum_release_motions(
                controller,
                priority_id,
                blocker_id,
                preferred_path,
                position_overrides=position_overrides,
                forbidden_regions=(open_rule or {}).get("forbidden_regions"),
                blocker_goal_path=blocker_goal_path,
            ):
                release_key = _release_key(
                    priority_id,
                    blocker_id,
                    context,
                    release,
                )
                with release_failures_lock:
                    if release_key in failed_release_directions:
                        continue
                trial_overrides = dict(position_overrides)
                trial_overrides[blocker_id] = release["release_position"]
                try:
                    _plan_batch_route(
                        priority_id,
                        context,
                        trial_overrides,
                        True,
                    )
                    feasible_releases.append(release)
                    if (
                        len(feasible_releases)
                        >= ONLINE_RELEASE_CANDIDATE_COUNT
                    ):
                        break
                except RuntimeError:
                    # If another blocker remains, keep this as an incremental
                    # branch and let the caller compose the next release.
                    if remaining_blockers:
                        feasible_releases.append(release)
                        if (
                            len(feasible_releases)
                            >= ONLINE_RELEASE_CANDIDATE_COUNT
                        ):
                            break
        except RuntimeError:
            return None
        if feasible_releases:
            return feasible_releases
        return None

    def _estimate_remaining(candidate, contexts, local_metrics):
        runtime = getattr(navigation, "conflict_runtime", None)
        estimator = getattr(runtime, "estimate_conflict_candidate", None)
        if estimator is None:
            return local_metrics
        return estimator(candidate, contexts, local_metrics)

    def _plan_joint_staging(priority_id, group, context, contexts):
        """Yield priority first when a boxed-in blocker cannot depart."""
        events = getattr(controller.last_event, "events", None) or []
        positions = {
            agent_id: event.metadata["agent"]["position"].copy()
            for agent_id, event in enumerate(events)
        }
        try:
            preferred = _plan_batch_route(priority_id, context, {}, False)
        except RuntimeError:
            return []
        blockers = _rank_hypothesis_blockers(
            priority_id,
            group,
            preferred["path"],
            {},
        )
        sequences = []
        for blocker_id in blockers:
            blocker_position = positions[blocker_id]
            setup_paths = iter_minimum_release_motions(
                controller,
                blocker_id,
                priority_id,
                [blocker_position],
                position_overrides={},
                blocker_goal_path=preferred["path"],
            )
            for setup in islice(
                setup_paths or (),
                ONLINE_RELEASE_CANDIDATE_COUNT,
            ):
                staged = {priority_id: setup["release_position"].copy()}
                try:
                    staged_priority = _plan_batch_route(
                        priority_id,
                        context,
                        staged,
                        False,
                    )
                except RuntimeError:
                    continue
                parking_paths = _plan_minimum_release(
                    priority_id,
                    blocker_id,
                    staged_priority["path"],
                    staged,
                    group,
                    context,
                    blocker_context=contexts.get(blocker_id),
                )
                for parking in islice(
                    parking_paths or (),
                    ONLINE_RELEASE_CANDIDATE_COUNT,
                ):
                    final_positions = dict(staged)
                    final_positions[blocker_id] = parking[
                        "release_position"
                    ].copy()
                    sequences.append(
                        ([setup, parking], final_positions)
                    )
        return sequences

    def _validate_candidate(candidate):
        events = getattr(controller.last_event, "events", None) or []

        def pair_clearance(first_id, second_id):
            return (
                2 * ROBOT_SAFETY_RADIUS
                + max(
                    carried_object_safety_margin(events[first_id]),
                    carried_object_safety_margin(events[second_id]),
                )
                + SMOOTH_MOVE_STEP
                + SMOOTH_FINAL_POSITION_TOLERANCE
            )

        from ..replanning.spatial_conflict import certify_concurrent_resume
        certify_concurrent_resume(candidate,pair_clearance)
        return candidate_paths_are_safe(candidate, pair_clearance)

    def _publish_candidate_keyframe(candidate):
        setter = getattr(navigation, "set_keyframe", None)
        if setter is None:
            return
        priority = int(candidate["priority_agent"]) + 1
        releases = candidate.get("release_moves", ())
        if releases:
            moved = ", ".join(
                f"R{int(release['blocker_agent']) + 1}"
                for release in releases
            )
            setter(
                f"Conflict: relocate {moved} | priority R{priority}"
            )
        else:
            setter(f"Conflict: A* reroute R{priority}")

    def _plan_hypotheses(participants, contexts):
        # Every robot in the component is frozen at this point. Share exact A*
        # results across priority ordering, release feasibility, candidate
        # construction, and DAG scoring, then discard them before motion
        # resumes so no stale Unity state can cross conflict boundaries.
        planning_route_cache.routes = {}
        planning_route_cache.goals = {}
        planning_route_cache.timing = {}
        try:
            runtime = getattr(navigation, "conflict_runtime", None)
            orderer = getattr(runtime, "order_conflict_priorities", None)
            priority_agents = (
                orderer(participants, contexts)
                if orderer is not None and RUNTIME_CONFLICT_POLICY == "dag_aware"
                else tuple(sorted(participants))
            )
            candidate = select_minimum_disruption_plan(
                participants,
                contexts,
                _plan_batch_route,
                _rank_hypothesis_blockers,
                _plan_minimum_release,
                remaining_estimator=_estimate_remaining,
                priority_agents=priority_agents,
                candidate_validator=_validate_candidate,
                selection_policy=RUNTIME_CONFLICT_POLICY,
                planning_timeout_s=RUNTIME_PLANNING_TIMEOUT,
                joint_staging_planner=_plan_joint_staging,
            )
            timing = dict(planning_route_cache.timing)
        finally:
            planning_route_cache.routes = None
            planning_route_cache.goals = None
            planning_route_cache.timing = None
        candidate["search_stats"].update(timing)
        _publish_candidate_keyframe(candidate)
        print(
            "AI2-THOR planning breakdown: "
            f"Unity goals={timing.get('interaction_goal_time_s', 0.0):.3f}s/"
            f"{timing.get('unity_goal_queries', 0)} queries, "
            f"A*={timing.get('astar_time_s', 0.0):.3f}s/"
            f"{timing.get('astar_calls', 0)} calls"
        )
        recorder = getattr(runtime, "record_conflict_candidate", None)
        if recorder is not None:
            candidate["selection_policy"] = RUNTIME_CONFLICT_POLICY
            recorder(candidate, participants)
        return candidate

    def _resolve_stationary_conflict(
        requesting_agent_id,
        blocking_agent_id,
        context,
        reject_direct=False,
    ):
        """Compare direct detour and minimum release with the shared objective."""
        if reject_direct:
            print(
                "Direct A* detour disabled for this contact; "
                "optimizing the remaining relocation candidates"
            )

        def stationary_candidate_is_safe(candidate):
            if reject_direct and candidate["strategy"] == "short_replan":
                return False
            return _validate_candidate(candidate)

        for _ in range(ONLINE_RELEASE_CANDIDATE_COUNT + 1):
            # The stationary snapshot is just as immutable as a moving
            # conflict group. Reuse both exact A* answers and expensive Unity
            # interaction-pose queries while evaluating its candidates.
            planning_route_cache.routes = {}
            planning_route_cache.goals = {}
            planning_route_cache.timing = {}
            try:
                candidate = select_minimum_disruption_plan(
                    (requesting_agent_id, blocking_agent_id),
                    {requesting_agent_id: context},
                    _plan_batch_route,
                    _rank_hypothesis_blockers,
                    _plan_minimum_release,
                    remaining_estimator=_estimate_remaining,
                    priority_agents=(requesting_agent_id,),
                    candidate_validator=stationary_candidate_is_safe,
                    selection_policy=RUNTIME_CONFLICT_POLICY,
                    planning_timeout_s=RUNTIME_PLANNING_TIMEOUT,
                    joint_staging_planner=_plan_joint_staging,
                )
                timing = dict(planning_route_cache.timing)
            except RuntimeError as error:
                # A manipulation can still hold the task lock after navigation
                # unregisters. If this transient snapshot has no intervention,
                # use the existing availability wait and retry after the atomic
                # action. Do not mask timeouts or genuine idle-blocker failures.
                if str(error) == "No feasible AI2-THOR conflict intervention":
                    blocker_lock = agent_locks[blocking_agent_id]
                    if not blocker_lock.acquire(blocking=False):
                        return False
                    blocker_lock.release()
                raise
            finally:
                planning_route_cache.routes = None
                planning_route_cache.goals = None
                planning_route_cache.timing = None
            candidate["search_stats"].update(timing)
            _publish_candidate_keyframe(candidate)
            print(
                "AI2-THOR planning breakdown: "
                f"Unity goals="
                f"{timing.get('interaction_goal_time_s', 0.0):.3f}s/"
                f"{timing.get('unity_goal_queries', 0)} queries, "
                f"A*={timing.get('astar_time_s', 0.0):.3f}s/"
                f"{timing.get('astar_calls', 0)} calls"
            )
            runtime = getattr(navigation, "conflict_runtime", None)
            recorder = getattr(runtime, "record_conflict_candidate", None)
            if recorder is not None:
                candidate["selection_policy"] = RUNTIME_CONFLICT_POLICY
                recorder(candidate, (requesting_agent_id, blocking_agent_id))
            releases = candidate["release_moves"]
            if not releases:
                print(
                    f"Stationary conflict: robot{requesting_agent_id + 1} "
                    "keeps the selected direct A* detour"
                )
                return candidate["priority_path"]

            acquired_locks = []
            release_failed = False
            try:
                for release in releases:
                    blocker_id = release["blocker_agent"]
                    blocker_lock = agent_locks[blocker_id]
                    if not blocker_lock.acquire(blocking=False):
                        return False
                    acquired_locks.append(blocker_lock)
                    if on_path_planned is not None:
                        on_path_planned(blocker_id, release["release_path"])
                    try:
                        follow_smooth_path(
                            controller,
                            robot_camera,
                            release["release_path"],
                            on_step=on_step,
                            step_delay=NAVIGATION_STEP_DELAY,
                            agent_id=blocker_id,
                            blocked_error_cls=NavigationBlockedError,
                            lookahead_distance=SMOOTH_SAMPLE_SPACING,
                        )
                    except NavigationBlockedError as error:
                        with release_failures_lock:
                            failed_release_directions.add(
                                _release_key(
                                    requesting_agent_id,
                                    blocker_id,
                                    context,
                                    release,
                                )
                            )
                        print(
                            "Release path met an unmodelled dynamic object; "
                            f"trying another A* direction: {error}"
                        )
                        release_failed = True
                        break
                    destination = release["release_position"]
                    print(
                        f"Minimum release: robot{blocker_id + 1} parked at "
                        f"x={destination['x']:.2f}, z={destination['z']:.2f}"
                    )
            finally:
                for blocker_lock in reversed(acquired_locks):
                    blocker_lock.release()
            if not release_failed:
                return candidate["priority_path"]
        return None

    batch_scheduler = MultiAgentReplanScheduler(
        controller,
        _plan_batch_route,
        _resolve_stationary_conflict,
        wait_for_stationary_blocker,
        hypothesis_planner=_plan_hypotheses,
        contact_waiter=wait_for_contact_blocker,
    )

    def navigation(
        robot,
        object_type,
        *,
        future_open_object=None,
        action_duration=0.0,
        wait_for_on_conflict=(),
    ):
        if robot not in ROBOT_NAMES:
            raise ValueError(
                f"Unknown robot {robot!r}. Available robots: {list(ROBOT_NAMES)}"
            )
        agent_id = ROBOT_AGENT_IDS[robot]
        with release_failures_lock:
            failed_release_directions.difference_update(
                {
                    key
                    for key in failed_release_directions
                    if key[0] == agent_id
                    and key[2] == object_type.casefold()
                }
            )
        planning_open_object = future_open_object
        if planning_open_object is None and any(
            obj.get("objectType", "").casefold() == object_type.casefold()
            and obj.get("isOpen", False)
            for obj in controller.last_event.metadata.get("objects", [])
        ):
            planning_open_object = object_type
        batch_scheduler.register(
            agent_id,
            object_type,
            planning_open_object,
            action_duration,
            wait_for_on_conflict=tuple(ROBOT_AGENT_IDS[rid] for rid in wait_for_on_conflict),
        )

        def report_planned_path(path):
            if on_path_planned is not None:
                on_path_planned(agent_id, path)

        def provide_path_update(updated_agent_id, route_completed=False):
            path = batch_scheduler.wait_for_route(
                updated_agent_id,
                route_completed=route_completed,
            )
            if path and on_path_planned is not None:
                on_path_planned(updated_agent_id, path)
            return path

        try:
            with agent_locks[agent_id]:
                result = navigate_to_object_type(
                    controller,
                    robot_camera,
                    object_type,
                    on_step=on_step,
                    on_path_planned=report_planned_path,
                    step_delay=NAVIGATION_STEP_DELAY,
                    agent_id=agent_id,
                    future_open_object=planning_open_object,
                    conflict_resolver_provider=batch_scheduler.resolve,
                    path_update_provider=provide_path_update,
                    raw_interaction_pose_cache=scene_interaction_pose_cache,
                )
            batch_scheduler.navigation_arrived(agent_id)
            return result
        except BaseException:
            batch_scheduler.abort()
            raise
        finally:
            batch_scheduler.unregister(agent_id)

    navigation.agent_locks = agent_locks
    navigation.robot_locks = {
        robot: agent_locks[agent_id]
        for robot, agent_id in ROBOT_AGENT_IDS.items()
    }

    navigation.task_finished = lambda robot: batch_scheduler.task_finished(
        ROBOT_AGENT_IDS[robot]
    )

    def estimate_task_route(
        agent_id,
        object_type,
        start_position,
        future_open_object=None,
    ):
        """Plan a static-only route from a hypothetical future robot pose."""
        overrides = (
            {agent_id: dict(start_position)}
            if start_position is not None
            else None
        )
        return plan_realtime_route(
            controller,
            object_type,
            agent_id,
            future_open_object=future_open_object,
            agent_position_overrides=overrides,
            avoid_other_agents=False,
            raw_interaction_pose_cache=scene_interaction_pose_cache,
        )

    def robot_positions():
        events = getattr(controller.last_event, "events", None) or []
        return {
            f"R{agent_id + 1}": event.metadata["agent"]["position"].copy()
            for agent_id, event in enumerate(events)
        }

    navigation.estimate_task_route = estimate_task_route
    navigation.robot_positions = robot_positions
    navigation.abort_replanning = batch_scheduler.abort

    def prepare_2d_planning(object_types):
        """Project task interaction regions into the cached 2-D scene map."""
        requested = tuple(
            sorted({str(value) for value in object_types if value})
        )
        # All agents share one radius, camera model and reachable-position
        # lattice, so their raw interaction regions are identical. Payload
        # filtering remains agent-specific and is applied later in Python.
        for object_type in requested:
            _interaction_goals(
                controller,
                object_type,
                0,
                raw_pose_cache=scene_interaction_pose_cache,
            )
        print(
            "Prepared 2-D planning abstraction: "
            f"{len(scene_interaction_pose_cache)} interaction-region snapshots "
            f"for {len(requested)} task targets"
        )

    navigation.prepare_2d_planning = prepare_2d_planning
    return navigation
