"""Snapshot-only conflict generation and guarded mission evaluation.
Execution-specific local timing is supplied by the calling adapter.
No simulator, GUI, Unity or ROS imports are required.
"""
import time
from . import negotiation
from .dynamic_collision import swept_robot_corridor
from .planner import (
    _astar_around_robot_bodies, select_dynamic_replan_path, with_geometry_cache,
    path_length,
)
from .relocation_search import (
    _candidate_diagnostic, _route_exits_without_reentry,
    evaluate_reroute_plan, search_relocation_plan,
)

def _snapshot_remaining_path(robot):
    """Return the untraversed centre path from an immutable robot snapshot."""
    if not robot.path:
        return [robot.cell]
    route = list(robot.path[robot.path_index:])
    if not route:
        return [robot.cell]
    if route[0] != robot.cell:
        route.insert(0, robot.cell)
    return route


def _all_robot_protected_paths(robots, excluded_ids=()):
    """Protect every robot's full remaining path, or its stationary body."""
    excluded = set(excluded_ids)
    return {
        robot.robot_id: _snapshot_remaining_path(robot)
        for robot in robots
        if robot.robot_id not in excluded
    }


def _route_clears_protected_corridors(
    route,
    protected_paths,
    robot_radius,
    grid_size,
):
    """Require a route to leave every footprint-wide corridor without reentry."""
    forbidden = set()
    for protected in protected_paths.values():
        forbidden.update(
            swept_robot_corridor(
                protected,
                robot_radius,
                robot_radius,
                grid_size,
            )
        )
    return _route_exits_without_reentry(route, forbidden)


@with_geometry_cache
def compute_conflict_plan(
    *,
    cluster,
    dag_robots,
    tasks,
    task_status,
    objects,
    navigation_cells,
    robot_radius,
    grid_size,
    speed_cells_per_second,
    selection_policy="dag_aware",
    planning_timeout_s=None,
    diagnostics=None,
    use_dag_priority=True,
    incremental_dag_evaluation=True,
    anticipated_delay_horizon=5.,
    preserve_priority_corridor=True,
    use_release_prediction=True,
    release_prediction_model="dynamic_events_guarded",
    hypothesis_workers=1,
    negotiation_state=None,
    parallel_reserved_paths=None,
    execution_candidate_refiner=None,
    local_resolution_time_evaluator,
):
    """Plan from immutable snapshots without touching live simulation state.

    An execution adapter may replace the grid release evaluator with its own
    candidate evaluator *before* selection. None preserves the grid method.
    The adapter must return only executable, scored candidates and must fail
    closed if its evaluation budget is exhausted.
    """
    if execution_candidate_refiner is not None and selection_policy != "dag_aware":
        raise ValueError("An execution evaluator requires dag_aware selection")
    workers = [
        robot
        for robot in cluster
        if robot.current_task is not None
        and robot.phase in {"preview", "moving", "waiting"}
    ]
    if not workers:
        if selection_policy == "reroute_only":
            raise RuntimeError("baseline_no_feasible_intervention")
        return None

    started = time.perf_counter()
    additional_candidates = []
    cluster_ids = {robot.robot_id for robot in cluster}
    external_protected_paths = _all_robot_protected_paths(
        dag_robots,
        excluded_ids=cluster_ids,
    )
    external_protected_paths.update(parallel_reserved_paths or {})
    # A direct dynamic reroute is a valid candidate for active-active as well
    # as active-idle contacts.  Holding the other contact participants while
    # one worker clears the component gives the evaluator and executor the
    # same conservative release semantics.
    for worker in workers:
        blockers = [robot for robot in cluster if robot is not worker]
        protected_paths = _all_robot_protected_paths(
            dag_robots,
            excluded_ids={worker.robot_id},
        )
        protected_paths.update(parallel_reserved_paths or {})
        clearance_cells = 2.0 * robot_radius / grid_size
        replanned = _astar_around_robot_bodies(
            navigation_cells, worker.cell, worker.goal,
            robot_centres=[cell for r in dag_robots if r.robot_id != worker.robot_id
                           for cell in _snapshot_remaining_path(r)],
            clearance_cells=clearance_cells,
        ) if selection_policy == "reroute_only" else select_dynamic_replan_path(
            navigation_cells,
            worker.cell,
            worker.goal,
            stationary_robot_centres=(robot.cell for robot in blockers),
            clearance_cells=clearance_cells,
            robot_speed=speed_cells_per_second,
            candidate_count=3,
        )
        if replanned is not None and _route_clears_protected_corridors(
            replanned,
            protected_paths,
            robot_radius,
            grid_size,
        ):
            additional_candidates.append(
                evaluate_reroute_plan(
                    worker_id=worker.robot_id,
                    route=replanned,
                    robots=dag_robots,
                    tasks=tasks,
                    task_status=task_status,
                    objects=objects,
                    navigation_cells=navigation_cells,
                    grid_size=grid_size,
                    speed_cells_per_second=speed_cells_per_second,
                    held_robot_ids=[r.robot_id for r in blockers],
                    robot_radius=robot_radius,
                    evaluate_dag=False,
                )
            )

    if selection_policy == "reroute_only":
        if not additional_candidates:
            raise RuntimeError("baseline_no_feasible_intervention")
        additional_candidates.sort(key=lambda c: (
            path_length(c["preferred_paths"][c["priority_robot"]], grid_size),
            c["priority_robot"], tuple(c["preferred_paths"][c["priority_robot"]])))
        selected = additional_candidates[0]
        for index, candidate in enumerate(additional_candidates):
            candidate["candidate_id"] = f"A{index:04d}"
            candidate["local_resolution_delay"] = local_resolution_time_evaluator(
                dict(candidate, strategy="service_reroute"), dag_robots,
                cluster_ids, navigation_cells, robot_radius, grid_size,
                speed_cells_per_second, objects)
            candidate["local_time_model"] = "current_execution_snapshot"
        selected["search_stats"] = {
            "candidate_evaluations": [_candidate_diagnostic(c, selected=c is selected)
                                      for c in additional_candidates],
            "generated_candidates": len(additional_candidates),
            "planning_wall_time_s": time.perf_counter() - started}
        return selected
    plan = search_relocation_plan(
        robots=cluster,
        dag_robots=dag_robots,
        priority_robot_ids=(robot.robot_id for robot in workers),
        participant_ids=(robot.robot_id for robot in cluster),
        tasks=tasks,
        task_status=task_status,
        objects=objects,
        navigation_cells=navigation_cells,
        robot_radius=robot_radius,
        grid_size=grid_size,
        speed_cells_per_second=speed_cells_per_second,
        additional_candidates=additional_candidates,
        external_protected_paths=external_protected_paths,
        selection_policy=selection_policy,
        planning_timeout_s=planning_timeout_s,
        dag_bound_pruning=False,
        local_time_evaluator=lambda candidate: local_resolution_time_evaluator(
            candidate, dag_robots, cluster_ids, navigation_cells,
            robot_radius, grid_size, speed_cells_per_second, objects),
        diagnostics=diagnostics,
        use_dag_priority=use_dag_priority,
        incremental_dag_evaluation=incremental_dag_evaluation,
        dag_candidate_refiner=(
            execution_candidate_refiner if execution_candidate_refiner is not None else
            _make_release_refiner(dag_robots,tasks,task_status,objects,navigation_cells,
                robot_radius,grid_size,speed_cells_per_second,cluster_ids,release_prediction_model,
                anticipated_delay_horizon,
                hypothesis_workers=hypothesis_workers)
            if selection_policy == "dag_aware" and use_release_prediction
                and preserve_priority_corridor else None
        ),
        preserve_priority_corridor=preserve_priority_corridor,
        negotiation_selector=(negotiation.make_selector(
            dag_robots, negotiation_state, speed_cells_per_second, cluster_ids)
            if selection_policy == "negotiation_adapted" else None),
    )
    plan["search_stats"]["planning_wall_time_s"] = time.perf_counter() - started
    return plan


def _make_release_refiner(robots, tasks, task_status, objects, navigation_cells,
                          robot_radius, grid_size, speed, participants,
                          model="dynamic_events_guarded", horizon=5., hypothesis_workers=1):
    """Use the current guarded, candidate-conditioned timing predictor."""
    if model != "dynamic_events_guarded":
        raise ValueError(f"Unsupported release prediction model: {model}")
    from .dynamic_release import refine_dynamic_release_events

    def refine(candidates, stats, deadline, baseline_finishes, baseline_distance):
        return refine_dynamic_release_events(
            candidates, robots=robots, tasks=tasks, task_status=task_status,
            objects=objects, navigation_cells=navigation_cells,
            robot_radius=robot_radius, grid_size=grid_size, speed=speed,
            participants=participants, stats=stats, search_deadline=deadline,
            baseline_finishes=baseline_finishes, baseline_distance=baseline_distance,
            horizon=horizon, budget_s=5.,
            hypothesis_workers=hypothesis_workers)
    return refine
