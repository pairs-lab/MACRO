"""Dynamic-robot conflict detection for standalone preferred paths."""

from __future__ import annotations

import math
from functools import lru_cache


def swept_separation(a0, a1, b0, b1):
    """Minimum centre separation over simultaneous linear movement."""
    dx, dy = a0[0] - b0[0], a0[1] - b0[1]
    vx = a1[0] - a0[0] - b1[0] + b0[0]
    vy = a1[1] - a0[1] - b1[1] + b0[1]
    norm = vx * vx + vy * vy
    t = max(0.0, min(1.0, -(dx * vx + dy * vy) / norm)) if norm else 0.0
    return math.hypot(dx + t * vx, dy + t * vy)


@lru_cache(maxsize=64)
def _disk_offsets(clearance):
    """Pure immutable raster footprint; never caches obstacle-dependent checks."""
    extent = int(math.ceil(clearance))
    return tuple((dx, dy) for dx in range(-extent, extent + 1)
                 for dy in range(-extent, extent + 1)
                 if math.hypot(dx, dy) <= clearance)


def swept_robot_corridor(
    path,
    moving_robot_radius,
    other_robot_radius,
    grid_size,
):
    """Rasterize the full body-clearance corridor around a centre path."""
    if moving_robot_radius <= 0 or other_robot_radius <= 0 or grid_size <= 0:
        raise ValueError("robot radii and grid_size must be positive")
    clearance = (moving_robot_radius + other_robot_radius) / grid_size
    offsets = _disk_offsets(clearance)
    corridor = set()
    for centre_x, centre_z in path:
        corridor.update(
            (centre_x + dx, centre_z + dz)
            for dx, dz in offsets
        )
    return corridor


def check_dynamic_robot_collisions(
    robots,
    robot_radius,
    grid_size,
    proposals=None,
):
    """Return contact clusters, optionally including imminent step conflicts.

    Without proposals this checks current contact only. With proposals it also
    checks swept discs over the next simultaneous step. Transitive contacts merge:
    R1-R2 plus R2-R3 produces one ``[R1, R2, R3]`` cluster.
    """
    if robot_radius <= 0 or grid_size <= 0:
        raise ValueError("robot_radius and grid_size must be positive")
    clearance_cells = 2.0 * robot_radius / grid_size
    graph = {robot.robot_id: set() for robot in robots}
    by_id = {robot.robot_id: robot for robot in robots}
    for index, first in enumerate(robots):
        for second in robots[index + 1 :]:
            distance = math.dist(first.cell, second.cell)
            if proposals is not None:
                distance = swept_separation(first.cell, proposals.get(first.robot_id, first.cell), second.cell, proposals.get(second.robot_id, second.cell))
            if distance > clearance_cells:
                continue
            graph[first.robot_id].add(second.robot_id)
            graph[second.robot_id].add(first.robot_id)

    clusters = []
    visited = set()
    for robot_id, neighbors in graph.items():
        if robot_id in visited or not neighbors:
            continue
        stack = [robot_id]
        component = []
        visited.add(robot_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        clusters.append([by_id[value] for value in component])
    return clusters
