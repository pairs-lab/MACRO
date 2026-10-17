"""Small dependency-free A* planner for the standalone grid world."""

from __future__ import annotations

import heapq
import math
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

from core import optimize


_GEOMETRY_CACHE = ContextVar('standalone_geometry_cache', default=None)


def with_geometry_cache(function):
    """Scope memoization to one immutable planning snapshot, never live state."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _GEOMETRY_CACHE.get() is not None:
            return function(*args, **kwargs)
        cache = {'values': {}, 'sets': {}, 'hits': 0, 'misses': 0}
        import os
        cache['clearance_reuse'] = os.environ.get('MARCO_CLEARANCE_REUSE', '1') != '0'
        cache['clearance_stats'] = {'corridor_hits': 0, 'corridor_misses': 0,
                                    'margin_hits': 0, 'margin_misses': 0,
                                    'body_hits': 0, 'body_misses': 0}
        token = _GEOMETRY_CACHE.set(cache)
        try:
            result = function(*args, **kwargs)
            if isinstance(result, dict) and 'search_stats' in result:
                from .native_geometry import backend_info
                result['search_stats']['geometry_backend'] = backend_info()
                result['search_stats']['geometry_cache'] = {
                    'hits': cache['hits'], 'misses': cache['misses']}
                if 'native_grid_stats' in cache:
                    result['search_stats']['native_grid_reuse'] = dict(cache['native_grid_stats'])
                result['search_stats']['clearance_geometry_reuse'] = dict(
                    enabled=cache['clearance_reuse'], **cache['clearance_stats'])
                if 'native_clearance_cache' in cache:
                    result['search_stats']['clearance_geometry_reuse']['native_body'] = (
                        cache['native_clearance_stats_reader'](cache['native_clearance_cache']))
            return result
        finally:
            _GEOMETRY_CACHE.reset(token)
    return wrapped


def _copy_geometry_result(function_name, result):
    """Copy mutable containers, sharing immutable grid-coordinate tuples.

    The cached planners below return paths of (x, z) tuples, or a distance
    dictionary with tuple keys and scalar values. Copying every tuple with
    deepcopy is unnecessary; callers still receive independent path lists.
    Unknown result schemas retain the conservative deepcopy behavior.
    """
    if result is None:
        return None
    if function_name == 'shortest_paths_to_any':
        return [list(path) for path in result]
    if function_name == 'shortest_path_distances':
        return dict(result)
    if function_name == 'shortest_path_tree_to_any':
        parent, endpoints, settled = result
        return dict(parent), list(endpoints), list(settled)
    if function_name in {
        'astar', '_astar_around_robot_bodies', 'select_preferred_path',
        'preferred_path_to_object',
    }:
        return list(result)
    return deepcopy(result)


def _memoized_geometry(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        cache = _GEOMETRY_CACHE.get()
        if cache is None:
            return function(*args, **kwargs)
        def freeze(value):
            if isinstance(value, (set, frozenset)):
                return frozenset(value)
            if isinstance(value, (tuple, list)):
                return tuple(freeze(item) for item in value)
            if isinstance(value, dict):
                return tuple((key, freeze(item)) for key, item in sorted(value.items()))
            return value
        # Materialize iterators: robot-centre generators must not become
        # identity-based cache keys or be consumed differently on a hit.
        from collections.abc import Iterator
        args = tuple(tuple(value) if isinstance(value, Iterator) else value for value in args)
        kwargs = {key: tuple(value) if isinstance(value, Iterator) else value
                  for key, value in kwargs.items()}
        from .native_geometry import backend_info
        geometry_backend = backend_info()
        key = (function.__name__, geometry_backend['backend'],
               geometry_backend['revision'], geometry_backend['native_astar'],
               geometry_backend['native_grid_reuse'],
               freeze(args), freeze(kwargs))
        if key in cache['values']:
            cache['hits'] += 1
            return _copy_geometry_result(function.__name__, cache['values'][key])
        cache['misses'] += 1
        result = function(*args, **kwargs)
        if len(cache['values']) < 4096:
            cache['values'][key] = _copy_geometry_result(function.__name__, result)
        return result
    return wrapped


_MOVES = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def path_length(path, grid_size=1.0):
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) * grid_size
        for a, b in zip(path, path[1:])
    )


def _heuristic(cell, goal):
    dx = abs(goal[0] - cell[0])
    dz = abs(goal[1] - cell[1])
    return max(dx, dz) + (math.sqrt(2.0) - 1.0) * min(dx, dz)


@_memoized_geometry
def astar(free_cells, start, goal, dynamic_obstacles=()):
    """Return a shortest 8-connected path without cutting blocked corners."""
    blocked = set(dynamic_obstacles)
    blocked.discard(start)
    if start not in free_cells or goal not in free_cells or goal in blocked:
        return None
    queue = [(0.0, 0.0, start)]
    costs = {start: 0.0}
    parent = {}

    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dz, step_cost in _MOVES:
            neighbor = (current[0] + dx, current[1] + dz)
            if neighbor not in free_cells or neighbor in blocked:
                continue
            if dx and dz:
                sides = (
                    (current[0] + dx, current[1]),
                    (current[0], current[1] + dz),
                )
                if any(side not in free_cells or side in blocked for side in sides):
                    continue
            next_cost = cost + step_cost
            if next_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = next_cost
            parent[neighbor] = current
            heapq.heappush(
                queue,
                (next_cost + _heuristic(neighbor, goal), next_cost, neighbor),
            )
    return None


@_memoized_geometry
def _astar_around_robot_bodies(
    free_cells,
    start,
    goal,
    robot_centres,
    clearance_cells,
    extra_obstacles=(),
):
    """Plan around stationary robot bodies while allowing contact separation.

    A robot which starts inside another robot's clearance disc may move
    monotonically out of that disc. Once it has left, it may not enter again.
    This avoids the false ``no path`` produced by treating the entire disc as
    blocked while the planning robot is already touching it.
    """
    centres = tuple(robot_centres)
    from .native_geometry import backend, backend_info
    native = backend()
    if native is not None and backend_info()['native_astar'] and len(centres) <= 63:
        return native.astar_bodies(free_cells, start, goal, centres,
                                   clearance_cells, extra_obstacles)
    blocked = set(extra_obstacles)
    blocked.discard(start)
    if (
        start not in free_cells
        or goal not in free_cells
        or goal in blocked
        or clearance_cells <= 0
    ):
        return None

    all_cleared = (1 << len(centres)) - 1
    initial_mask = 0
    for index, centre in enumerate(centres):
        if math.dist(start, centre) > clearance_cells:
            initial_mask |= 1 << index

    initial_state = (start, initial_mask)
    queue = [(_heuristic(start, goal), 0.0, start, initial_mask)]
    costs = {initial_state: 0.0}
    parent = {}
    epsilon = 1e-9

    while queue:
        _, cost, current, cleared_mask = heapq.heappop(queue)
        state = (current, cleared_mask)
        if cost != costs.get(state):
            continue
        if current == goal and cleared_mask == all_cleared:
            path = [current]
            while state in parent:
                state = parent[state]
                path.append(state[0])
            path.reverse()
            return path

        for dx, dz, step_cost in _MOVES:
            neighbor = (current[0] + dx, current[1] + dz)
            if neighbor not in free_cells or neighbor in blocked:
                continue
            if dx and dz:
                sides = (
                    (current[0] + dx, current[1]),
                    (current[0], current[1] + dz),
                )
                if any(side not in free_cells or side in blocked for side in sides):
                    continue

            next_mask = cleared_mask
            valid = True
            for index, centre in enumerate(centres):
                bit = 1 << index
                next_distance = math.dist(neighbor, centre)
                if cleared_mask & bit:
                    if next_distance <= clearance_cells + epsilon:
                        valid = False
                        break
                    continue
                current_distance = math.dist(current, centre)
                if next_distance <= current_distance + epsilon:
                    valid = False
                    break
                if next_distance > clearance_cells + epsilon:
                    next_mask |= bit
            if not valid:
                continue

            next_state = (neighbor, next_mask)
            next_cost = cost + step_cost
            if next_cost >= costs.get(next_state, math.inf):
                continue
            costs[next_state] = next_cost
            parent[next_state] = state
            heapq.heappush(
                queue,
                (
                    next_cost + _heuristic(neighbor, goal),
                    next_cost,
                    neighbor,
                    next_mask,
                ),
            )
    return None


def dynamic_replan_candidates(
    free_cells,
    start,
    goal,
    *,
    stationary_robot_centres,
    clearance_cells,
    robot_speed=1.0,
    candidate_count=3,
):
    """Select a short A* replan around stationary robots.

    The caller decides whether the selected route is close enough to the
    original preferred path. This function only returns the best of up to
    three geometrically distinct feasible routes.
    """
    if robot_speed <= 0 or not math.isfinite(float(robot_speed)):
        raise ValueError("robot_speed must be positive")
    if not 1 <= int(candidate_count) <= 3:
        raise ValueError("candidate_count must be between 1 and 3")
    robot_centres = tuple(stationary_robot_centres)

    shortest = _astar_around_robot_bodies(
        free_cells,
        start,
        goal,
        robot_centres,
        clearance_cells,
    )
    if shortest is None:
        return []

    paths = [shortest]
    if int(candidate_count) == 1:
        return paths
    seen = {tuple(shortest)}
    interior = list(range(1, len(shortest) - 1))
    interior.sort(key=lambda index: (abs(index - len(shortest) / 2), index))
    for index in interior[: min(len(interior), 18)]:
        alternative = _astar_around_robot_bodies(
            free_cells,
            start,
            goal,
            robot_centres,
            clearance_cells,
            extra_obstacles={shortest[index]},
        )
        if alternative is None or tuple(alternative) in seen:
            continue
        paths.append(alternative)
        seen.add(tuple(alternative))
        if len(paths) == int(candidate_count):
            break

    return paths


def select_dynamic_replan_path(
    free_cells, start, goal, *, stationary_robot_centres, clearance_cells,
    robot_speed=1.0, candidate_count=3,
):
    paths = dynamic_replan_candidates(
        free_cells, start, goal,
        stationary_robot_centres=stationary_robot_centres,
        clearance_cells=clearance_cells, robot_speed=robot_speed,
        candidate_count=candidate_count,
    )
    if not paths:
        return None
    candidates = []
    for index, path in enumerate(paths, start=1):
        distance = path_length(path)
        candidates.append(
            {
                "id": index,
                "path": path,
                "estimated_makespan": distance / float(robot_speed),
                "total_distance": distance,
                "total_delay": 0.0,
            }
        )
    return optimize(candidates, evaluator=lambda candidate: candidate)[
        "candidate"
    ]["path"]


@_memoized_geometry
def shortest_paths_to_any(
    free_cells,
    start,
    goals,
    *,
    dynamic_obstacles=(),
    path_count=3,
):
    """Return up to ``path_count`` shortest paths to distinct valid goals."""
    parent, endpoints, _ = shortest_path_tree_to_any(
        free_cells, start, goals, dynamic_obstacles=dynamic_obstacles,
        path_count=path_count)
    paths = []
    for endpoint in endpoints:
        path = [endpoint]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path.reverse()
        paths.append(path)
    return paths


@_memoized_geometry
def shortest_path_tree_to_any(free_cells, start, goals, *,
                              dynamic_obstacles=(), path_count=3):
    """Same Dijkstra ordering as shortest_paths_to_any, without path copies.

    Settled cells are returned parent-before-child for prefix validation.
    No feasibility-based pruning or alternative-parent selection is applied.
    """
    from .native_geometry import backend
    native = backend()
    if native is not None:
        import os
        grid = free_cells
        cache = _GEOMETRY_CACHE.get()
        if (cache is not None and hasattr(native, 'prepare_grid') and
                os.environ.get('MARCO_NATIVE_GRID_REUSE', '1') != '0'):
            # Content-keyed, snapshot-local and bounded. Dynamic obstacles and
            # goals are deliberately NOT embedded in this immutable geometry.
            grids = cache.setdefault('native_navigation_grids', {})
            stats = cache.setdefault('native_grid_stats', {'hits': 0, 'builds': 0, 'uncached': 0})
            signature = frozenset(free_cells)
            if signature in grids:
                stats['hits'] += 1
                grid = grids[signature]
            elif len(grids) < 8:
                grid = native.prepare_grid(signature)
                grids[signature] = grid
                stats['builds'] += 1
            else:
                stats['uncached'] += 1
        return native.shortest_path_tree(grid, start, goals,
                                         dynamic_obstacles, float(path_count))
    blocked = set(dynamic_obstacles)
    blocked.discard(start)
    remaining_goals = set(goals) & set(free_cells) - blocked
    if start not in free_cells or not remaining_goals or path_count <= 0:
        return {}, [], []
    queue = [(0.0, start)]
    costs = {start: 0.0}
    parent = {}
    endpoints = []
    settled = []
    while queue and remaining_goals and len(endpoints) < path_count:
        cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        settled.append(current)
        if current in remaining_goals:
            endpoints.append(current)
            remaining_goals.remove(current)
        for dx, dz, step_cost in _MOVES:
            neighbor = (current[0] + dx, current[1] + dz)
            if neighbor not in free_cells or neighbor in blocked:
                continue
            if dx and dz:
                sides = (
                    (current[0] + dx, current[1]),
                    (current[0], current[1] + dz),
                )
                if any(side not in free_cells or side in blocked for side in sides):
                    continue
            next_cost = cost + step_cost
            if next_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = next_cost
            parent[neighbor] = current
            heapq.heappush(queue, (next_cost, neighbor))
    return parent, endpoints, settled


@_memoized_geometry
def shortest_path_distances(free_cells, start, *, dynamic_obstacles=()):
    """Return shortest grid distance from ``start`` to every reachable cell."""
    blocked = set(dynamic_obstacles)
    blocked.discard(start)
    if start not in free_cells:
        return {}
    queue = [(0.0, start)]
    costs = {start: 0.0}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        for dx, dz, step_cost in _MOVES:
            neighbor = (current[0] + dx, current[1] + dz)
            if neighbor not in free_cells or neighbor in blocked:
                continue
            if dx and dz:
                sides = (
                    (current[0] + dx, current[1]),
                    (current[0], current[1] + dz),
                )
                if any(
                    side not in free_cells or side in blocked for side in sides
                ):
                    continue
            next_cost = cost + step_cost
            if next_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = next_cost
            heapq.heappush(queue, (next_cost, neighbor))
    return costs


def _interaction_goals(free_cells, object_cells, slack_cells=2.0):
    """Return nearby poses while allowing a slightly better approach side."""
    targets = tuple(object_cells)
    distances = {
        cell: min(math.dist(cell, target) for target in targets)
        for cell in free_cells
    }
    if not distances:
        return set()
    minimum_distance = min(distances.values())
    return {
        cell
        for cell, distance in distances.items()
        if distance <= minimum_distance + float(slack_cells)
    }


@_memoized_geometry
def select_preferred_path(
    free_cells,
    start,
    goal,
    *,
    robot_speed=1.0,
    candidate_count=3,
):
    """Return the best of 2-3 static A* routes from ``start`` to ``goal``.

    Other robots are intentionally absent from this calculation. Alternative
    routes are generated by temporarily excluding different interior cells of
    the shortest route. The shared optimizer then applies its makespan-first,
    exact mission-makespan selection rule.
    """
    if robot_speed <= 0 or not math.isfinite(float(robot_speed)):
        raise ValueError("robot_speed must be positive")
    if not 1 <= int(candidate_count) <= 3:
        raise ValueError("candidate_count must be between 1 and 3")

    shortest = astar(free_cells, start, goal)
    if shortest is None:
        return None

    paths = [shortest]
    seen = {tuple(shortest)}
    interior = list(range(1, len(shortest) - 1))
    # Try central deviations first, followed by evenly distributed cells. This
    # normally produces distinct routes with only a few extra A* calls.
    interior.sort(key=lambda index: (abs(index - len(shortest) / 2), index))
    maximum_attempts = min(len(interior), 18)
    for index in interior[:maximum_attempts]:
        alternative = astar(
            free_cells,
            start,
            goal,
            dynamic_obstacles={shortest[index]},
        )
        if alternative is None or tuple(alternative) in seen:
            continue
        paths.append(alternative)
        seen.add(tuple(alternative))
        if len(paths) == int(candidate_count):
            break

    candidates = []
    for index, path in enumerate(paths, start=1):
        distance = path_length(path)
        candidates.append(
            {
                "id": index,
                "path": path,
                "estimated_makespan": distance / float(robot_speed),
                "total_distance": distance,
                "total_delay": 0.0,
            }
        )
    return optimize(candidates, evaluator=lambda candidate: candidate)[
        "candidate"
    ]["path"]


@_memoized_geometry
def preferred_path_to_object(
    free_cells,
    start,
    object_cells,
    *,
    interaction_cells=None,
    robot_speed=1.0,
    candidate_count=3,
):
    """Optimize 2-3 static A* routes and their object-approach poses."""
    goals = (
        set(interaction_cells) & set(free_cells)
        if interaction_cells is not None
        else _interaction_goals(free_cells, object_cells)
    )
    paths = shortest_paths_to_any(
        free_cells,
        start,
        goals,
        path_count=candidate_count,
    )
    if not paths:
        return None
    candidates = []
    for index, path in enumerate(paths, start=1):
        distance = path_length(path)
        candidates.append(
            {
                "id": index,
                "path": path,
                "estimated_makespan": distance / float(robot_speed),
                "total_distance": distance,
                "total_delay": 0.0,
            }
        )
    return optimize(candidates, evaluator=lambda candidate: candidate)[
        "candidate"
    ]["path"]
