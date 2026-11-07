"""Bounded reservation-aware refinement using the production grid executor.

Only DAG-Aware uses this layer. Candidate geometry/feasibility is unchanged.
Recourse in hypothetical execution uses Local-Delay (zero compute latency),
not recursively another look-ahead. No arbitrary collision penalty is added.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import math
import time
from types import SimpleNamespace


class LookaheadIncomplete(RuntimeError):
    pass


def successor_reservation_risk(candidates, *, robots, tasks, task_status,
                              objects, navigation_cells, speed, clearance):
    """Cheap geometric gate; no penalty and no feasibility rejection.

    A future task crossing a yielder's holding pose merits replay. Ambiguous
    object relocation or missing routes conservatively merits replay as well.
    """
    from .planner import preferred_path_to_object
    from .relocation_search import _path_conflict
    by_id = {r.robot_id:r for r in robots}
    future_routes = {}
    for candidate in candidates:
        priority = candidate.get("priority_robot")
        robot = by_id.get(priority)
        if robot is None or robot.current_task is None:
            continue
        owned = [t for t in tasks if t["robot"] == priority
                 and task_status.get(t["id"]) != "completed"]
        index = next((i for i,t in enumerate(owned)
                      if t["id"] == robot.current_task["id"]), None)
        if index is None or index+1 >= len(owned):
            continue
        task = owned[index+1]
        location = task.get("target") if task["action"] == "place" else task["object"]
        if any(t["action"] == "place" and t["object"] == location
               and task_status.get(t["id"]) != "completed" for t in tasks):
            return True
        key = (priority, task["id"], robot.goal)
        if key not in future_routes:
            item = objects[location]
            route = preferred_path_to_object(navigation_cells, robot.goal or robot.cell,
                item["cells"], interaction_cells=item.get("interaction_cells"),
                robot_speed=speed, candidate_count=3)
            if route is None:
                return True
            future_routes[key] = route
        poses = candidate.get("final_positions", {})
        for other in robots:
            if other.robot_id == priority:
                continue
            pose = poses.get(other.robot_id, other.cell)
            if _path_conflict(future_routes[key], [pose], clearance):
                return True
    return False


def refine_candidates(candidates, *, evaluator, stable_key, select, stats,
                      budget_s=5.0, baseline_finishes=None, baseline_distance=0.,
                      representatives_per_root=1, diagnostic_key="dag_lookahead"):
    """Atomically compare deterministic representatives on a common model.

    Shortlist: best old DAG candidate per priority, plus the global old DAG,
    distance and delay winners. Unreplayed candidates remain in diagnostics,
    but are not compared using optimistic old scores against replayed scores.
    Any incomplete replay discards ALL refinements, retaining the old ranking.
    """
    chosen = [select(candidates)]
    roots = sorted({str(c.get("priority_robot")) for c in candidates})
    for root in roots:
        remaining = [c for c in candidates if str(c.get("priority_robot")) == root]
        for _ in range(min(representatives_per_root, len(remaining))):
            representative = select(remaining)
            chosen.append(representative)
            remaining = [c for c in remaining if c is not representative]
    for field in ("local_resolution_distance", "local_resolution_delay"):
        chosen.append(min(candidates, key=lambda c: (c[field], stable_key(c))))
    shortlisted = sorted({id(c): c for c in chosen}.values(), key=stable_key)
    started = time.perf_counter()
    deadline = started + budget_s
    results = []
    info = dict(enabled=True, model="executor-current-plus-next-task",
                recourse_policy="local_delay", max_recourse_repairs=2,
                simulation_steps="production_next_event_duration",
                horizon_s=30.0, shortlist_size=len(shortlisted),
                original_candidates=len(candidates), budget_s=budget_s,
                status="fallback", evaluated=0)
    stats[diagnostic_key] = info
    if len(shortlisted) == 1:
        # Replaying the only representative cannot change the decision.
        # Do not spend GUI latency on a refinement with no ranking benefit.
        info.update(status="skipped_single_representative", wall_time_s=0.)
        return candidates
    try:
        for candidate in shortlisted:
            if time.perf_counter() >= deadline:
                raise LookaheadIncomplete("wall_budget")
            result = evaluator(candidate, deadline)
            if not all(math.isfinite(result[k]) for k in
                       ("estimated_makespan", "total_distance")):
                raise LookaheadIncomplete("nonfinite_forecast")
            results.append(result)
            info["evaluated"] += 1
        if time.perf_counter() >= deadline:
            raise LookaheadIncomplete("wall_budget")
    except (LookaheadIncomplete, TimeoutError) as error:
        info["fallback_reason"] = str(error)
        return candidates
    finally:
        info["wall_time_s"] = time.perf_counter() - started
    for candidate in candidates:
        candidate["lookahead_shortlisted"] = False
    for candidate, result in zip(shortlisted, results):
        candidate["nominal_estimated_makespan"] = candidate["estimated_makespan"]
        candidate["lookahead_shortlisted"] = True
        candidate["lookahead"] = result
        candidate.update(estimated_makespan=result["estimated_makespan"],
                         estimated_remaining_DAG_makespan=result["estimated_makespan"],
                         total_distance=result["total_distance"],
                         additional_distance=max(0., result["total_distance"]-baseline_distance))
        if baseline_finishes is not None:
            candidate["total_delay"] = sum(
                max(0., finish-baseline_finishes.get(task_id, finish))
                for task_id, finish in result["finish_times"].items())
    info["status"] = "applied"
    return shortlisted


def replay_candidate(candidate, *, robots, tasks, task_status, objects,
                     navigation_cells, robot_radius, grid_size, speed,
                     participants, deadline, horizon_s=30., max_repairs=2,
                     release_only=False, route_cache=None,
                     recourse_policy="local_delay",
                     recourse_release_prediction_model="dynamic_events_guarded"):
    # Lazy import avoids a cycle and guarantees shared movement/release code.
    from simulations.standalone.app import StandaloneSimulation, _compute_conflict_plan, _snapshot_remaining_path
    from .relocation_search import _remaining_dag_cost

    def check():
        if time.perf_counter() >= deadline:
            raise LookaheadIncomplete("wall_budget")
    check()
    if any(r.robot_id not in participants and
           (r.phase == "parking" or getattr(r, "waiting_for", None))
           for r in robots):
        # Immutable planner snapshots do not carry other components' live
        # reservation records. Never pretend those robots are free to depart.
        raise LookaheadIncomplete("external_intervention_not_snapshotted")
    sim = StandaloneSimulation.__new__(StandaloneSimulation)
    sim.world = SimpleNamespace(navigation_cells=navigation_cells,
                                grid_size=grid_size, robot_radius=robot_radius)
    sim.steps_per_second = speed
    if recourse_policy not in {"local_delay", "dag_aware"}:
        raise ValueError(f"Unsupported forecast recourse policy: {recourse_policy}")
    if recourse_policy == "dag_aware" and recourse_release_prediction_model != "dynamic_events_guarded":
        raise ValueError("DAG forecast recourse must use nonrecursive dynamic_events_guarded")
    sim.selection_policy = recourse_policy
    sim.async_conflict_planning = False
    sim.use_dag_priority = recourse_policy == "dag_aware"
    sim.incremental_dag_evaluation = True
    sim.use_dag_lookahead = False
    sim.use_release_prediction = recourse_policy == "dag_aware"
    sim.release_prediction_model = recourse_release_prediction_model
    sim._executor_verification_disabled = True
    sim._release_prediction_route_cache = route_cache
    sim.robots = deepcopy(robots)
    sim.objects = deepcopy(objects)
    sim.tasks = deepcopy(tasks)
    sim.task_status = dict(task_status)
    sim.pending_conflict_plans = {}
    sim.active_resolutions = {}
    sim.resolution_history = []
    sim.service_decision_history = []
    sim.conflict_events = []
    sim._service_choice_cache = {}
    sim._next_resolution_id = 1
    sim._planning_generation = 0
    sim.simulation_time_s = 0.
    sim.planning_wall_time_s = 0.
    targets = set()
    for robot in robots:
        if robot.robot_id not in participants:
            continue
        remaining = [t for t in tasks if t["robot"] == robot.robot_id
                     and task_status.get(t["id"]) != "completed"]
        if robot.current_task is not None:
            current = robot.current_task["id"]
            offset = next((i for i, t in enumerate(remaining) if t["id"] == current), 0)
            remaining = remaining[offset:]
        targets.update(t["id"] for t in remaining[:2])
    repairs = [0]
    def check_repair_limit():
        check()
        if repairs[0] >= max_repairs:
            raise LookaheadIncomplete("recourse_limit")
    original_record = sim._record_conflict
    def record_repair(plan, group):
        check_repair_limit()
        repairs[0] += 1
        original_record(plan, group)
    sim._record_conflict = record_repair
    def replan(cluster):
        check_repair_limit()
        return _compute_conflict_plan(
            cluster=cluster, dag_robots=sim.robots, tasks=sim.tasks,
            task_status=sim.task_status, objects=sim.objects,
            navigation_cells=navigation_cells, robot_radius=robot_radius,
            grid_size=grid_size, speed_cells_per_second=speed,
            selection_policy=sim.selection_policy,
            planning_timeout_s=max(.001, deadline-time.perf_counter()),
            use_dag_priority=sim.use_dag_priority,
            use_dag_lookahead=False,
            use_release_prediction=(sim.use_release_prediction
                                    and not getattr(sim, "_release_prediction_disabled", False)),
            release_prediction_model=sim.release_prediction_model,
            verify_release=False)
    sim._plan_snapshot = replan
    original_service = sim._choose_service_wait_or_reroute
    def service(cluster):
        # Repeated service cache hits are not new interventions. Count only
        # actual plans committed via _record_conflict, not detector calls.
        check()
        if release_only:
            raise LookaheadIncomplete("new_service_conflict")
        sim.planning_timeout_s = max(.001, deadline-time.perf_counter())
        return original_service(cluster)
    sim._choose_service_wait_or_reroute = service
    plan = deepcopy(candidate)
    plan["search_stats"] = dict(roots=0, expanded=0, bound_pruned=0, feasible=1)
    start_distance = sum(r.travelled for r in sim.robots)
    steps = 0
    completion_times = {t["id"]:0. for t in tasks
                        if task_status.get(t["id"]) == "completed"}
    with redirect_stdout(io.StringIO()):
        if plan["strategy"] == "relocate":
            resolution_id = sim._begin_resolution_plan(plan, participants)
            if resolution_id in sim.active_resolutions:
                sim.active_resolutions[resolution_id]["fixed_service_robots"] = {
                    r.robot_id for r in robots if r.phase == "acting"}
        elif plan["strategy"] == "short_replan":
            sim._begin_reroute_plan(plan, participants)
        elif plan["strategy"] == "service_reroute":
            robot = sim._robot(plan["rerouted_agents"][0])
            robot.path = list(plan["preferred_paths"][robot.robot_id])
            robot.path_index = 0
            robot.movement_credit = 0.
            robot.movement_edge = None
            robot.phase = "moving"
        elif plan["strategy"] != "wait":
            raise LookaheadIncomplete("unsupported_strategy")
        def prefix_complete():
            if release_only:
                # Stop only after the actual reservation has released all
                # yielders. Successor tasks can dispatch during this prefix.
                if plan["strategy"] == "wait":
                    return not any(r.robot_id in participants and r.phase == "acting"
                                   for r in sim.robots)
                return not sim.active_resolutions
            return all(sim.task_status[t] == "completed" for t in targets) and not sim.active_resolutions
        while not prefix_complete():
            check()
            if sim.simulation_time_s >= horizon_s - 1e-9 or steps >= 1801:
                raise LookaheadIncomplete("horizon_limit")
            # Event-sized steps avoid simulating display-only frames. This is
            # a forecast, not a claim of bit-identical GUI frame scheduling.
            dt = min(sim.next_event_duration(), horizon_s-sim.simulation_time_s)
            if dt <= 1e-9:
                # Flush a sub-nanosecond action residual using one bounded
                # frame; the executor's completion guard owns the transition.
                dt = min(1/60, horizon_s-sim.simulation_time_s)
            try:
                sim.step(dt)
            except LookaheadIncomplete:
                raise
            except RuntimeError as error:
                raise LookaheadIncomplete(f"uncertified_recourse:{error}") from error
            steps += 1
            for task_id, status in sim.task_status.items():
                if status == "completed" and task_id not in completion_times:
                    completion_times[task_id] = sim.simulation_time_s
    check()
    tail, tail_distance, tail_finishes = _remaining_dag_cost(
        tasks=sim.tasks, task_status=sim.task_status,
        robots_by_id={r.robot_id:r for r in sim.robots},
        positions={r.robot_id:r.cell for r in sim.robots},
        preferred_paths={r.robot_id:_snapshot_remaining_path(r) for r in sim.robots
                         if r.current_task is not None},
        objects=sim.objects, navigation_cells=navigation_cells, grid_size=grid_size,
        speed_cells_per_second=speed, relocation_delay_by_robot={},
        include_travel=True, return_details=True)
    completion_times.update({tid:sim.simulation_time_s+finish
                             for tid, finish in tail_finishes.items()
                             if sim.task_status.get(tid) != "completed"})
    return dict(estimated_makespan=sim.simulation_time_s+tail,
                total_distance=sum(r.travelled for r in sim.robots)-start_distance+tail_distance,
                replay_time_s=sim.simulation_time_s, tail_makespan_s=tail,
                recourse_repairs=repairs[0], steps=steps, target_task_ids=sorted(targets),
                recourse_policy=sim.selection_policy,
                recourse_release_prediction_model=(sim.release_prediction_model
                    if sim.use_release_prediction else None),
                nested_verification=False,
                finish_times=completion_times)


def refine_release_predictions(candidates, *, robots, tasks, task_status, objects,
                               navigation_cells, robot_radius, grid_size, speed,
                               participants, stats, search_deadline,
                               baseline_finishes, baseline_distance, budget_s=3.):
    """Common executor-prefix timing for ALL candidates; no recourse search.

    Apply transactionally. Unknown future conflicts or exhausted budgets restore
    the whole nominal batch, never mix certified and nominal timing predictions.
    This changes DAG scores only, not geometry or local-policy timing/ranking.
    """
    if len(candidates) < 2:
        stats["release_prediction"] = dict(status="skipped_single_candidate",wall_time_s=0.)
        return candidates
    started=time.perf_counter()
    deadline=min(started+budget_s,search_deadline if search_deadline is not None else math.inf)
    pending=[]
    route_cache={}
    try:
        for candidate in candidates:
            result=replay_candidate(candidate,robots=robots,tasks=tasks,task_status=task_status,
                objects=objects,navigation_cells=navigation_cells,robot_radius=robot_radius,
                grid_size=grid_size,speed=speed,participants=participants,deadline=deadline,
                max_repairs=0,release_only=True,route_cache=route_cache)
            pending.append((candidate,result))
        if time.perf_counter() >= deadline:
            raise LookaheadIncomplete("wall_budget")
    except LookaheadIncomplete as error:
        stats["release_prediction"] = dict(status="fallback",reason=str(error),
            candidates_completed=len(pending),wall_time_s=time.perf_counter()-started)
        return candidates
    for candidate,result in pending:
        candidate["nominal_estimated_makespan"]=candidate["estimated_makespan"]
        candidate.update(estimated_makespan=result["estimated_makespan"],
            estimated_remaining_DAG_makespan=result["estimated_makespan"],
            total_distance=result["total_distance"],
            additional_distance=max(0.,result["total_distance"]-baseline_distance),
            total_delay=sum(max(0.,f-baseline_finishes.get(t,f)) for t,f in result["finish_times"].items()),
            release_prediction=result)
    stats["release_prediction"]=dict(status="applied",candidates_evaluated=len(pending),
        wall_time_s=time.perf_counter()-started,model="executor-release-prefix-plus-remaining-DAG")
    return candidates
