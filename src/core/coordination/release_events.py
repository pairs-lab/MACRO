"""Bounded current-route release timing, without replaying a mission.

This is a prediction, not a feasibility certificate or a global planner.
Unknown successor conflicts remain explicitly unknown. The live executor
continues to own all body/swept collision checks and dynamic replanning.
"""

import math
import time

from .planner import path_length, preferred_path_to_object, _astar_around_robot_bodies, _interaction_goals
from .relocation_search import _remaining_dag_cost, _timed_path_conflict, _path_conflict


def distance_prefix(path, limit):
    result = list(path[:1])
    travelled = 0.0
    for first, second in zip(path, path[1:]):
        result.append(second)
        travelled += math.dist(first, second)
        if travelled + 1e-9 >= limit:
            break
    return result


def continuation_motion_is_clear(priority_path, continuation, clearance, speed):
    """The executor's bounded swept-motion guard, shared with prediction."""
    if not priority_path or not continuation:
        return False
    if math.dist(priority_path[0], continuation[0]) <= clearance:
        return False
    horizon = max(2.0 * clearance, speed)
    return not _timed_path_conflict(
        distance_prefix(priority_path, horizon),
        distance_prefix(continuation, horizon), clearance)


def priority_motion_is_clear(priority_path, mover_path, stationary, clearance, speed):
    """Shared guard for letting the priority move during a relocation."""
    if not continuation_motion_is_clear(priority_path, mover_path, clearance, speed):
        return False
    ahead = distance_prefix(priority_path, max(2.0*clearance, speed))
    return not any(_path_conflict(ahead, [position], clearance) for position in stationary)


def choose_continuation_route(navigation, start, destination, original_goal,
                              other_centres, clearance, speed, cached_route=None,
                              report_uncertainty=False):
    """Shared original-goal-first dynamic A* and guarded nominal fallback."""
    cached_route = cached_route or (lambda key, compute: compute())
    goals = (set(destination.get('interaction_cells', ())) & set(navigation)
             if destination.get('interaction_cells') is not None
             else _interaction_goals(navigation, destination['cells']))
    ordered = [original_goal] if original_goal in goals else []
    ordered.extend(sorted(goals-set(ordered), key=lambda goal: (math.dist(start, goal), goal))[:3])
    nominal = cached_route(('nominal', start, frozenset(destination['cells']),
                           None if destination.get('interaction_cells') is None else frozenset(destination['interaction_cells']), speed),
                          lambda: preferred_path_to_object(navigation, start, destination['cells'],
                              interaction_cells=destination.get('interaction_cells'), robot_speed=speed, candidate_count=3))
    routes = []
    for goal in ordered:
        route = cached_route(('dynamic', start, goal, tuple(other_centres), clearance),
                             lambda: _astar_around_robot_bodies(navigation, start, goal, other_centres, clearance))
        if route is not None:
            routes.append(route)
            if goal == original_goal:
                break
    if not routes:
        if nominal is None:
            return None, None, False
        routes.append(nominal)
    route = min(routes, key=lambda value: (path_length(value), tuple(value)))
    # A nominal fallback can encounter a future blocker; it is not certified.
    unknown = report_uncertainty and any(
        _path_conflict(route, [centre], clearance) for centre in other_centres)
    return route, nominal, unknown


def wait_for_nominal_is_better(priority_route, service_time, nominal, route,
                               yielder_position, clearance, speed):
    if nominal is None or route is None:
        return False
    waiting = path_length(priority_route)/speed + service_time + path_length(nominal)/speed
    detour = path_length(route)/speed
    return (not _path_conflict(priority_route, [yielder_position], clearance)
            and waiting + 1e-9 < detour)


def predict_release_events(candidate, robots, clearance, speed, *, navigation=None, objects=None,
                           deadline=math.inf, moving_body_continuation=False,
                           cached_route=None, stationary_service_by_robot=None):
    """Predict serial staging plus earliest bounded safe continuation.

    Current routes use metric s/v. Staging moves remain serial; safe early
    priority motion is credited for a single non-joint release. Unknown end-of-route
    clearance uses the old whole-route wait and is labelled, not declared
    conflict-free. All candidates use these same rules.
    """
    delays = dict(candidate.get('delay_by_robot', {}))
    if candidate['strategy'] != 'relocate':
        return delays, {'model': 'existing-current-intervention', 'unknown_agents': []}
    nominal_offsets = {}
    if moving_body_continuation:
        # First find nominal release waypoints using only shared geometry
        # guards. Body-aware route selection then runs once per yielder,
        # rather than an A* at every intervening priority waypoint.
        _, nominal_info = predict_release_events(candidate, robots, clearance, speed,
                                                  deadline=deadline)
        nominal_offsets = nominal_info['release_offsets_s']
    priority = candidate['priority_robot']
    paths = candidate['preferred_paths']
    priority_path = paths[priority]
    barrier = sum(float(step['delay']) for step in candidate['relocations'])
    held = set(delays) | set(candidate.get('moved_agents', ())) | {priority}
    delays = {robot_id: barrier for robot_id in held}
    priority_start = barrier
    steps = candidate['relocations']
    overlap_credited = False
    if len(steps) == 1 and steps[0]['robot_id'] != priority:
        mover = steps[0]['robot_id']
        stationary = [robot.cell for rid, robot in robots.items() if rid in held - {priority, mover}]
        elapsed = 0.0
        release_path = steps[0]['path']
        for index in range(len(release_path)):
            if index:
                elapsed += math.dist(release_path[index-1], release_path[index]) / speed
            if priority_motion_is_clear(priority_path, release_path[index:], stationary, clearance, speed):
                priority_start = min(barrier, elapsed)
                overlap_credited = priority_start < barrier
                break
    delays[priority] = priority_start
    unknown = []
    continuations = {}
    release_offsets = {priority: priority_start}
    def pose_on_path(route, elapsed):
        """Cell-based executor pose, not interpolation through occupied cells."""
        travelled = max(0., elapsed) * speed
        for index, (a, b) in enumerate(zip(route, route[1:])):
            distance = math.dist(a, b)
            if travelled + 1e-9 < distance:
                return route[index]
            travelled -= distance
        return route[-1]

    def centres_at(at, priority_pose, yielder):
        centres = []
        for rid, robot in robots.items():
            if rid == yielder:
                continue
            if rid == priority:
                pose = priority_pose
            elif rid in held:
                route = continuations.get(rid)
                pose = (pose_on_path(route, at-release_offsets[rid])
                        if route and at >= release_offsets[rid]
                        else candidate.get('final_positions', {}).get(rid, robot.cell))
            elif robot.phase == 'moving' and robot.path:
                route = list(robot.path[robot.path_index:])
                if not route or route[0] != robot.cell:
                    route.insert(0, robot.cell)
                credit = getattr(robot, 'movement_credit', 0.) / speed
                pose = pose_on_path(route, at+credit)
            else:
                pose = robot.cell
            centres.append(pose)
        return tuple(centres)
    for robot_id in sorted(held - {priority}):
        continuation = paths.get(robot_id)
        if not continuation:
            # An idle supporting robot has no task to delay.
            release_offsets[robot_id] = 0.0
            continue
        elapsed = 0.0
        found = False
        predicted_route = continuation
        nominal = continuation
        uncertain = False
        route_probes = 0
        last_centres = None
        if navigation is not None and objects is not None and not moving_body_continuation:
            task = robots[robot_id].current_task
            destination_name = task['target'] if task['action'] == 'place' else task['object']
            centres = tuple(priority_path[0] if rid == priority else
                            candidate.get('final_positions', {}).get(rid, robot.cell)
                            for rid, robot in robots.items() if rid != robot_id)
            predicted_route, nominal, uncertain = choose_continuation_route(
                navigation, continuation[0], objects[destination_name], robots[robot_id].goal,
                centres, clearance, speed, report_uncertainty=True)
        priority_task = getattr(robots.get(priority), 'current_task', None) or {}
        service = (robots[priority].action_remaining if getattr(robots.get(priority), 'phase', None) == 'acting'
                   else float(priority_task.get('duration', 0.)))
        if moving_body_continuation:
            service = (stationary_service_by_robot or {}).get(priority, service)
        for index in range(len(priority_path)):
            if time.perf_counter() >= deadline:
                raise TimeoutError('event_budget')
            if index:
                elapsed += math.dist(priority_path[index-1], priority_path[index]) / speed
            # A yielder cannot resume until every staging move has finished.
            # Evaluate only waypoints reached at or after that barrier.
            if priority_start + elapsed + 1e-9 < barrier:
                continue
            if moving_body_continuation and navigation is not None and objects is not None:
                if priority_start + elapsed + 1e-9 < nominal_offsets.get(robot_id, barrier):
                    continue
                # Recompute at bounded release checkpoints, never against the
                # priority's original snapshot pose.
                centres = centres_at(priority_start+elapsed, priority_path[index], robot_id)
                # Retry only after a material body displacement, at most three
                # times. A rejected route must not stay frozen while the body
                # that caused the detour moves away. Batch caching shares A*.
                changed = (last_centres is None or any(
                    math.dist(a, b) >= max(1., clearance / 2.)
                    for a, b in zip(centres, last_centres)))
                if route_probes < 3 and changed:
                    task = robots[robot_id].current_task
                    destination_name = task['target'] if task['action'] == 'place' else task['object']
                    predicted_route, nominal, uncertain = choose_continuation_route(
                        navigation, continuation[0], objects[destination_name], robots[robot_id].goal,
                        centres, clearance, speed, cached_route=cached_route,
                        report_uncertainty=True)
                    route_probes += 1
                    last_centres = centres
            if predicted_route is None or wait_for_nominal_is_better(
                priority_path[index:], service, nominal, predicted_route,
                continuation[0], clearance, speed):
                continue
            if continuation_motion_is_clear(priority_path[index:], predicted_route, clearance, speed):
                found = True
                if uncertain:
                    unknown.append(robot_id)
                break
        if not found:
            elapsed = path_length(priority_path) / speed
            if (moving_body_continuation and
                    _path_conflict(continuation, [priority_path[-1]], clearance)):
                # Arrival is not clearance: consecutive service at this pose
                # keeps the bottleneck occupied, even beyond the scan horizon.
                elapsed += service
            unknown.append(robot_id)
            predicted_route = continuation
        continuations[robot_id] = predicted_route
        # The bounded current-route forecast cannot time a successor crossing.
        # Also reject a continuation with a conflict beyond the swept horizon.
        protected = candidate.get('protected_priority_corridor', priority_path)
        future = protected[len(priority_path)-1:]
        if (len(future) > 1 and _path_conflict(future, predicted_route, clearance)
                or _timed_path_conflict(priority_path[index:] if found else priority_path,
                                        predicted_route, clearance)):
            if robot_id not in unknown:
                unknown.append(robot_id)
        release_offsets[robot_id] = max(barrier, priority_start + elapsed)
        delays[robot_id] = release_offsets[robot_id]
    return delays, {
        'model': 'serial-staging-bounded-release-events',
        'staging_time_s': barrier, 'release_offsets_s': release_offsets,
        'delay_by_robot': delays, 'unknown_agents': unknown,
        'continuation_paths': continuations,
        'early_staging_overlap_credited': overlap_credited,
        'future_conflicts_modelled': False,
        'moving_body_continuation': moving_body_continuation,
    }


def refine_release_events(candidates, *, robots, tasks, task_status, objects,
                          navigation_cells, robot_radius, grid_size, speed,
                          participants, stats, search_deadline,
                          baseline_finishes, baseline_distance, budget_s=.15):
    """Inject predicted waits into the same incremental remaining DAG."""
    started = time.perf_counter()
    deadline = min(started + budget_s, search_deadline if search_deadline is not None else math.inf)
    def fallback(reason):
        stats['release_prediction'] = dict(status='fallback', reason=reason,
                                           wall_time_s=time.perf_counter()-started,
                                           model='certified-bounded-release-events')
        return candidates
    # Single-candidate conflicts cannot gain ranking quality from refinement.
    if len(candidates) < 2:
        stats['release_prediction'] = dict(status='skipped_single_candidate', wall_time_s=0.)
        return candidates
    if time.perf_counter() >= deadline:
        return fallback('event_budget')
    by_id = {robot.robot_id: robot for robot in robots}
    positions = {robot.robot_id: robot.cell for robot in robots}
    kwargs = dict(tasks=tasks, task_status=task_status, robots_by_id=by_id,
                  objects=objects, navigation_cells=navigation_cells,
                  grid_size=grid_size, speed_cells_per_second=speed,
                  include_travel=True, return_details=True, return_evaluation=True)
    baseline = _remaining_dag_cost(**kwargs, positions=positions, preferred_paths={},
                                   relocation_delay_by_robot={})[3]
    pending = []
    for candidate in candidates:
        if time.perf_counter() >= deadline:
            return fallback('event_budget')
        try:
            delays, diagnostics = predict_release_events(candidate, by_id, 2*robot_radius/grid_size, speed,
                                                         navigation=navigation_cells, objects=objects, deadline=deadline)
        except TimeoutError:
            return fallback('event_budget')
        if diagnostics['unknown_agents']:
            return fallback('uncertified_continuation')
        changed = set(delays) | set(candidate.get('rerouted_agents', ())) | set(candidate.get('moved_agents', ()))
        if candidate.get('priority_robot') is not None:
            changed.add(candidate['priority_robot'])
        final = dict(positions)
        final.update(candidate.get('final_positions', {}))
        forecast_paths = dict(candidate['preferred_paths'])
        forecast_paths.update(diagnostics.get('continuation_paths', {}))
        estimate = _remaining_dag_cost(
            **kwargs, positions=final, preferred_paths=forecast_paths,
            relocation_delay_by_robot=delays, affected_robot_ids=changed,
            baseline_evaluation=baseline)
        pending.append((candidate, delays, diagnostics, estimate))
    if time.perf_counter() >= deadline:
        return fallback('event_budget')
    # Commit transactionally; never mix event timing with partially updated scores.
    for candidate, delays, diagnostics, estimate in pending:
        makespan, distance, finishes, evaluation = estimate
        distance += sum(step['distance'] for step in candidate.get('relocations', ()))
        candidate['nominal_estimated_makespan'] = candidate['estimated_makespan']
        candidate.update(estimated_makespan=makespan,
                         estimated_remaining_DAG_makespan=makespan,
                         total_distance=distance,
                         additional_distance=max(0., distance-baseline_distance),
                         total_delay=sum(max(0., f-baseline_finishes.get(t, f)) for t, f in finishes.items()),
                         delay_by_robot=delays,
                         release_prediction={**diagnostics, 'estimated_makespan': makespan,
                                             'directly_changed_robots': sorted(delays)})
    stats['release_prediction'] = dict(status='applied', model='bounded-release-events-plus-incremental-DAG',
                                      candidates_evaluated=len(pending), wall_time_s=time.perf_counter()-started)
    return candidates
