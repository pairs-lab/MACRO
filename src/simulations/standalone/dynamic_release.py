"""Bounded moving-body/current-plus-successor timing for DAG candidates.

No global search or executor repair is run. Conflict exposure becomes a task
wait and propagates through the remaining DAG. Forecasts are not feasibility
certificates: stationary terminal bodies and unresolved encounters are censored
at the horizon, and the live executor retains every safety guard.
"""

import math
import time
import os
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import get_context

from .anticipated_delay import estimate_future_delays
from .relocation_search import _remaining_dag_cost
from .release_events import predict_release_events

_POOLS = {}


def close_hypothesis_pools():
    """Explicit lifecycle for pools owned by this process, not other planners."""
    for pool in list(_POOLS.values()):
        pool.shutdown(wait=True, cancel_futures=True)
    _POOLS.clear()


def _evaluate_hypothesis(candidate, kwargs):
    stats = {}
    result = refine_dynamic_release_events([candidate], stats=stats,
        hypothesis_workers=1, _allow_single=True, **kwargs)
    return result[0], stats['release_prediction']


def _parallel_refinement(candidates, kwargs, stats, deadline, started, workers):
    """Immutable process snapshots, stable result order, all-or-nothing commit."""
    from concurrent.futures.process import BrokenProcessPool
    key = (os.getpid(), workers)
    futures = []
    try:
        pool = _POOLS.get(key)
        if pool is None:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
            _POOLS[key] = pool
        futures = [pool.submit(_evaluate_hypothesis, c, kwargs) for c in candidates]
        _, pending = wait(futures, timeout=max(0., deadline-time.perf_counter()))
        if pending:
            raise TimeoutError('parallel_hypothesis_budget')
        results = [f.result() for f in futures]
        if any(info['status'] != 'applied' for _, info in results):
            raise TimeoutError('parallel_hypothesis_incomplete_batch')
        if time.perf_counter() >= deadline:
            raise TimeoutError('parallel_hypothesis_budget')
    except (TimeoutError, BrokenProcessPool) as error:
        for future in futures:
            future.cancel()
        if isinstance(error, BrokenProcessPool):
            _POOLS.pop(key, None)
            pool.shutdown(wait=False, cancel_futures=True)
        stats['release_prediction'] = dict(status='fallback', reason=str(error),
            model='parallel-dynamic-release', hypothesis_workers=workers,
            candidates_evaluated=0, wall_time_s=time.perf_counter()-started)
        return candidates
    for original, (evaluated, _) in zip(candidates, results):
        original.update(evaluated)
    info = dict(results[0][1])
    info.update(candidates_evaluated=len(results), hypothesis_workers=workers,
        evaluation_mode='process_parallel', wall_time_s=time.perf_counter()-started,
        worker_wall_time_sum_s=sum(r[1]['wall_time_s'] for r in results),
        censored_candidates=sum(r[1]['censored_candidates'] for r in results))
    stats['release_prediction'] = info
    return candidates


def stationary_service_chain(robots, tasks, evaluation):
    """Known consecutive service at the current goal; no future routing search.

    Stop at the first task whose cached path leaves the exact current goal.
    This extends occupancy, not the spatial look-ahead horizon.
    """
    result = {}
    for robot in robots:
        current = robot.current_task
        if current is None:
            continue
        goal = robot.goal or robot.cell
        ordered = sorted((t for t in tasks if t['robot'] == robot.robot_id
                          and t['id'] in evaluation['start_times']),
                         key=lambda t: (evaluation['start_times'][t['id']], t['id']))
        total = 0.
        begun = False
        for task in ordered:
            tid = task['id']
            if not begun:
                if tid != current['id']:
                    continue
                begun = True
            else:
                route = evaluation['task_paths'].get(tid) or (goal,)
                if any(math.dist(p, goal) > 1e-9 for p in route):
                    break
            total += (robot.action_remaining if tid == current['id'] and robot.phase == 'acting'
                      else float(task.get('duration', 0.)))
        result[robot.robot_id] = total
    return result


def refine_dynamic_release_events(candidates, *, robots, tasks, task_status,
                                  objects, navigation_cells, robot_radius,
                                  grid_size, speed, participants, stats,
                                  search_deadline, baseline_finishes,
                                  baseline_distance, horizon=5., budget_s=.30,
                                  use_release_guards=True, hypothesis_workers=1,
                                  _allow_single=False):
    """Compare the whole pool with one common, two-pass reservation model."""
    if not 3. <= horizon <= 5.:
        raise ValueError('Dynamic release horizon must be between 3 and 5 seconds')
    if not isinstance(hypothesis_workers, int) or hypothesis_workers < 1:
        raise ValueError('hypothesis_workers must be a positive integer')
    started = time.perf_counter()
    deadline = min(started+budget_s, search_deadline if search_deadline is not None else math.inf)
    model = 'guarded-release-checkpoints-stationary-chain-incremental-DAG-v5'
    if not use_release_guards:
        model = 'moving-reservations-current-plus-successor-incremental-DAG'
    def fallback(reason):
        stats['release_prediction'] = dict(status='fallback', reason=reason, model=model,
                                           wall_time_s=time.perf_counter()-started)
        return candidates
    if len(candidates) < 2 and not _allow_single:
        stats['release_prediction'] = dict(status='skipped_single_candidate', model=model, wall_time_s=0.)
        return candidates
    if time.perf_counter() >= deadline:
        return fallback('dynamic_event_budget')
    if any(r.robot_id not in participants and
           (r.phase == 'parking' or getattr(r, 'waiting_for', None)) for r in robots):
        return fallback('external_intervention_not_snapshotted')
    if hypothesis_workers > 1 and len(candidates) > 1:
        kwargs = dict(robots=robots, tasks=tasks, task_status=task_status,
            objects=objects, navigation_cells=navigation_cells,
            robot_radius=robot_radius, grid_size=grid_size, speed=speed,
            participants=participants, search_deadline=deadline,
            baseline_finishes=baseline_finishes, baseline_distance=baseline_distance,
            horizon=horizon, budget_s=budget_s, use_release_guards=use_release_guards)
        return _parallel_refinement(candidates, kwargs, stats, deadline, started,
                                    min(hypothesis_workers, len(candidates)))
    by_id = {r.robot_id: r for r in robots}
    positions = {r.robot_id: r.cell for r in robots}
    common = dict(tasks=tasks, task_status=task_status, robots_by_id=by_id,
                  objects=objects, navigation_cells=navigation_cells,
                  grid_size=grid_size, speed_cells_per_second=speed,
                  include_travel=True, return_details=True, return_evaluation=True)
    baseline = _remaining_dag_cost(**common, positions=positions, preferred_paths={},
                                   relocation_delay_by_robot={})[3]
    service_chains = stationary_service_chain(robots, tasks, baseline)
    route_cache = {}
    def cached_route(key, compute):
        if time.perf_counter() >= deadline:
            raise TimeoutError('dynamic_event_budget')
        if key not in route_cache:
            route = compute()
            route_cache[key] = None if route is None else tuple(route)
        route = route_cache[key]
        return None if route is None else list(route)
    pending = []
    for candidate in candidates:
        if time.perf_counter() >= deadline:
            return fallback('dynamic_event_budget')
        release_info = {}
        delays = dict(candidate.get('delay_by_robot', {}))
        if use_release_guards:
            try:
                delays, release_info = predict_release_events(
                    candidate, by_id, 2*robot_radius/grid_size, speed,
                    navigation=navigation_cells, objects=objects, deadline=deadline,
                    moving_body_continuation=True, cached_route=cached_route,
                    stationary_service_by_robot=service_chains)
                release_info['stationary_service_by_robot'] = service_chains
                release_info['continuation_model'] = 'body-aware-at-predicted-release-waypoint'
                release_info['executor_route_selection_replayed'] = False
            except TimeoutError:
                return fallback('dynamic_event_budget')
        forecast_paths = dict(candidate['preferred_paths'])
        forecast_paths.update(release_info.get('continuation_paths', {}))
        barrier = sum(float(step['delay']) for step in candidate.get('relocations', ()))
        if candidate['strategy'] == 'relocate':
            held = set(participants) | set(candidate.get('moved_agents', ()))
            if not use_release_guards:
                for robot_id in held:
                    delays[robot_id] = barrier
            # Match the executor's release/continuation guards before scanning
            # future encounters. Staging completion alone is NOT a release.
            # Fixed service robots keep executing during staging.
            for robot_id in list(delays):
                if (use_release_guards or robot_id in held) and by_id[robot_id].phase == 'acting':
                    delays.pop(robot_id, None)
        affected = set(delays) | set(candidate.get('rerouted_agents', ())) | set(candidate.get('moved_agents', ()))
        priority = candidate.get('priority_robot')
        if priority is not None:
            affected.add(priority)
        final_positions = dict(positions)
        final_positions.update(candidate.get('final_positions', {}))
        forecast = dict(candidate, delay_by_robot=delays, final_positions=final_positions,
                        preferred_paths=forecast_paths)
        initial = _remaining_dag_cost(
            **common, positions=final_positions, preferred_paths=forecast_paths,
            relocation_delay_by_robot=delays, baseline_evaluation=baseline,
            affected_robot_ids=affected)
        estimate = initial
        future_delays = {}
        passes = []
        # External robots are already running and cannot be ordered to yield
        # by this hypothetical repair. Priority then precedes its own yielders.
        order = {rid: index for index, rid in enumerate(
            ([priority] if priority is not None else []) + sorted(set(participants)-{priority}))}
        try:
            for _ in range(2):
                additional, info = estimate_future_delays(
                    candidate=forecast, evaluation=estimate[3], robots=robots, tasks=tasks,
                    speed=speed, grid_size=grid_size, robot_radius=robot_radius,
                    horizon=horizon, deadline=deadline, task_limit=2,
                    wait_before_navigation=True, robot_order=order, window_after_staging=True,
                    include_future_occupancy=True)
                passes.append(info)
                if not additional:
                    break
                for task_id, delay in additional.items():
                    future_delays[task_id] = future_delays.get(task_id, 0.) + delay
                estimate = _remaining_dag_cost(
                    **common, positions=final_positions, preferred_paths=forecast_paths,
                    relocation_delay_by_robot=delays, baseline_evaluation=initial[3],
                    timing_only=True, future_delay_by_task=future_delays)
                if time.perf_counter() >= deadline:
                    raise TimeoutError('dynamic_event_budget')
        except TimeoutError:
            return fallback('dynamic_event_budget')
        pending.append((candidate, delays, future_delays, passes, estimate, release_info, forecast_paths))
    if time.perf_counter() >= deadline:
        return fallback('dynamic_event_budget')
    # No mutation before all candidates have used the same forecast model.
    for candidate, delays, future_delays, passes, estimate, release_info, forecast_paths in pending:
        makespan, distance, finishes, _ = estimate
        distance += sum(step['distance'] for step in candidate.get('relocations', ()))
        candidate['nominal_estimated_makespan'] = candidate['estimated_makespan']
        candidate.update(estimated_makespan=makespan,
                         estimated_remaining_DAG_makespan=makespan, total_distance=distance,
                         additional_distance=max(0., distance-baseline_distance),
                         total_delay=sum(max(0., f-baseline_finishes.get(t, f)) for t, f in finishes.items()),
                         delay_by_robot=delays,
                         release_prediction=dict(model=model, horizon_s=horizon,
                             delay_by_robot=delays, future_delay_by_task=future_delays,
                             release_events=release_info, forecast_paths=forecast_paths,
                             reservation_passes=passes, estimated_makespan=makespan,
                             finish_times=finishes, censored=any(p['censored'] for p in passes),
                             successor_tasks_per_robot=1, global_repair=False))
    stats['release_prediction'] = dict(status='applied', model=model, horizon_s=horizon,
                                      candidates_evaluated=len(pending), wall_time_s=time.perf_counter()-started,
                                      censored_candidates=sum(any(p['censored'] for p in item[3]) for item in pending))
    return candidates
