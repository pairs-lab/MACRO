"""Compare lazy tree clearance against the original full-path algorithm."""
import math
import heapq
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core import distance_to_polyline, select_clearance_top_k
from simulations.standalone.dynamic_collision import swept_robot_corridor
from simulations.standalone.planner import path_length, _MOVES
from simulations.standalone.relocation_search import (
    _nearest_feasible_relocations, _route_exits_without_reentry,
    _route_clears_robot_bodies, _direction_group,
)


def original_shortest_paths(cells, start, goals, *, dynamic_obstacles, path_count):
    """Pre-optimization Dijkstra, independently reconstructing each goal."""
    blocked = set(dynamic_obstacles)
    blocked.discard(start)
    goals = set(goals) & set(cells) - blocked
    if start not in cells or not goals or path_count <= 0:
        return []
    queue = [(0., start)]
    costs = {start: 0.}
    parent = {}
    paths = []
    while queue and goals and len(paths) < path_count:
        cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        if current in goals:
            path = [current]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            paths.append(path[::-1])
            goals.remove(current)
        for dx, dy, step in _MOVES:
            other = current[0]+dx, current[1]+dy
            if other not in cells or other in blocked:
                continue
            if dx and dy and any(q not in cells or q in blocked
                                 for q in [(current[0]+dx, current[1]),
                                           (current[0], current[1]+dy)]):
                continue
            new_cost = cost+step
            if new_cost >= costs.get(other, math.inf):
                continue
            costs[other] = new_cost
            parent[other] = current
            heapq.heappush(queue, (new_cost, other))
    return paths


def full_path_reference(*, mover_id, positions, protected_paths,
                        navigation_cells, robot_radius, grid_size, speed,
                        goal=None, nominal_path=None, path_budget=3,
                        route_scan_budget=None):
    start = positions[mover_id]
    forbidden = set()
    for route in protected_paths.values():
        forbidden.update(swept_robot_corridor(
            route, robot_radius, robot_radius, grid_size))
    clearance = 2 * robot_radius / grid_size
    occupied = set()
    for rid, cell in positions.items():
        if rid == mover_id or rid in protected_paths:
            continue
        radius = math.ceil(clearance)
        occupied.update((cell[0]+dx, cell[1]+dy)
                        for dx in range(-radius, radius+1)
                        for dy in range(-radius, radius+1)
                        if math.hypot(dx, dy) <= clearance)
    goals = navigation_cells - forbidden - occupied - {start}
    scan_count = (len(goals) if route_scan_budget is None else
                  min(len(goals), max(path_budget, int(route_scan_budget))))
    from simulations.standalone.planner import shortest_path_distances
    reachable = (None if goal is None else
                 set(shortest_path_distances(navigation_cells, goal)))
    candidates = []
    for route in original_shortest_paths(navigation_cells, start, goals,
                                      dynamic_obstacles=occupied, path_count=scan_count):
        if (len(route) <= 1 or not _route_exits_without_reentry(route, forbidden)
                or not _route_clears_robot_bodies(route, mover_id, positions, clearance)):
            continue
        if reachable is not None and route[-1] not in reachable:
            continue
        margin = min((distance_to_polyline(route[-1], p)
                      for p in protected_paths.values()), default=math.inf)*grid_size-2*robot_radius
        if margin <= 1e-9:
            continue
        distance = path_length(route, grid_size)
        candidates.append(dict(robot_id=mover_id, start=start,
                               destination=route[-1], path=route,
                               distance=distance, delay=distance/(speed*grid_size),
                               direction_group=_direction_group(start, route[-1]),
                               protects=tuple(sorted(protected_paths)), clearance_margin=margin))
    return select_clearance_top_k(candidates, path_budget)


class ClearanceTreeEquivalenceTests(unittest.TestCase):
    def test_full_path_reference_matches_for_random_obstacles_and_contacts(self):
        for seed in range(60):
            rng = random.Random(seed)
            cells = {(x, y) for x in range(-5, 6) for y in range(-5, 6)
                     if rng.random() > .12}
            cells.add((0, 0))
            positions = {'R1': (-2, 0), 'R2': (0, 0),
                         'R3': (rng.randrange(-3, 4), rng.randrange(-3, 4))}
            grid_size = rng.choice([.05, .2, 1.])
            radius = grid_size*rng.choice([.2, .5, 1., 1.5])
            protected = {'R1': [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)]}
            arguments = dict(mover_id='R2', positions=positions,
                             protected_paths=protected, navigation_cells=cells,
                             robot_radius=radius, grid_size=grid_size, speed=8.,
                             goal=(4, 4) if seed % 2 else None)
            for budget, scan in [(3, None), (3, 12), (1, None), (10, 8), (None, None)]:
                with self.subTest(seed=seed, budget=budget, scan=scan):
                    kwargs = dict(arguments, path_budget=budget, route_scan_budget=scan)
                    self.assertEqual(_nearest_feasible_relocations(**kwargs),
                                     full_path_reference(**kwargs))


if __name__ == '__main__':
    unittest.main()
