"""Environment-independent helpers for geometric yielding."""

from __future__ import annotations

import math


def _xz(point):
    if isinstance(point, dict):
        return float(point["x"]), float(point["z"])
    return float(point[0]), float(point[1])


def distance_to_polyline(point, path):
    """Return Euclidean distance from a point to a 2-D polyline."""
    px, pz = _xz(point)
    vertices = [_xz(vertex) for vertex in path]
    if not vertices:
        return math.inf
    if len(vertices) == 1:
        return math.hypot(px - vertices[0][0], pz - vertices[0][1])
    best = math.inf
    for (ax, az), (bx, bz) in zip(vertices, vertices[1:]):
        dx, dz = bx - ax, bz - az
        norm = dx * dx + dz * dz
        ratio = 0.0 if norm <= 1e-12 else (
            (px - ax) * dx + (pz - az) * dz
        ) / norm
        ratio = max(0.0, min(1.0, ratio))
        best = min(
            best,
            math.hypot(px - (ax + ratio * dx), pz - (az + ratio * dz)),
        )
    return best


def _prepare_polyline(path):
    """Immutable vertex/segment arithmetic reused for many endpoint queries."""
    vertices = tuple(_xz(vertex) for vertex in path)
    segments = []
    for (ax, az), (bx, bz) in zip(vertices, vertices[1:]):
        dx, dz = bx - ax, bz - az
        segments.append((ax, az, dx, dz, dx * dx + dz * dz))
    return vertices, tuple(segments)


def _distance_to_prepared_polyline(point, prepared):
    """Same operations and float order as distance_to_polyline, without parsing."""
    px, pz = _xz(point)
    vertices, segments = prepared
    if not vertices:
        return math.inf
    if len(vertices) == 1:
        return math.hypot(px - vertices[0][0], pz - vertices[0][1])
    best = math.inf
    for ax, az, dx, dz, norm in segments:
        ratio = 0.0 if norm <= 1e-12 else ((px - ax) * dx + (pz - az) * dz) / norm
        ratio = max(0.0, min(1.0, ratio))
        best = min(best, math.hypot(px - (ax + ratio * dx), pz - (az + ratio * dz)))
    return best


def goal_directed_yield_key(candidate):
    """Paper ordering: Cgd, release length, nominal-route deviation."""
    destination = candidate.get(
        "release_position", candidate.get("destination", {})
    )
    if isinstance(destination, dict):
        destination_key = (
            float(destination.get("x", 0.0)),
            float(destination.get("z", 0.0)),
        )
    else:
        destination_key = tuple(destination)
    return (
        float(candidate["goal_directed_cost"]),
        float(candidate.get("release_distance", candidate.get("distance", 0.0))),
        float(
            candidate.get(
                "nominal_route_deviation",
                candidate.get("goal_deviation", 0.0),
            )
        ),
        candidate.get("direction_group", -1),
        destination_key,
    )


def select_goal_directed_top_k(candidates, count):
    """Select the true Top-K after all feasible endpoints are evaluated."""
    if count <= 0:
        return []
    return sorted(candidates, key=goal_directed_yield_key)[: int(count)]


def clearance_yield_key(candidate):
    """Clearance time, safety margin, then canonical non-goal tie-breaks.

    Margin is a tie-break, not a weighted penalty. Goal distance, remaining
    tasks and DAG criticality deliberately do not participate in this key.
    """
    destination=candidate.get("destination",candidate.get("release_position"))
    direction=candidate.get("direction_group")
    return (float(candidate["delay"]),-float(candidate.get("clearance_margin",0.)),
            -1 if direction is None else int(direction),_xz(destination),
            tuple(_xz(p) for p in candidate.get("path",())))


def select_clearance_top_k(candidates, count):
    """Shortest clear move plus angularly diverse representatives.

    Eight geometric direction bins are used by the grid generator. After the
    shortest move, maximize angular separation from directions already kept;
    ties prefer the shortest clear move with greater body-clearance margin.
    Once direction coverage is exhausted, fill with remaining shortest moves.
    No orientation toward the task goal or arbitrary scalar weights are used.
    """
    # Reuse the geometry in the ranking key for deduplication. In particular,
    # do not walk every path a second time just to rebuild its coordinate key.
    return _select_clearance_ranked(
        ((clearance_yield_key(c), c, c.get('direction_group')) for c in candidates), count)


def _select_clearance_ranked(entries, count):
    """Shared exact selection over (canonical key, payload, direction) entries.

    Geometry generators can provide lightweight payloads before constructing
    dictionaries. The key and selection rule are identical to the public API.
    """
    ranked=sorted(entries,key=lambda item:item[0])
    ordered=[c for _,c,_ in ranked]
    if count is None:
        return ordered
    if count<=0 or not ordered:
        return []
    # Repeated endpoints/paths must not consume the diversity budget.
    unique=[];seen=set();representatives={}
    directions=[]
    for rank,c,direction in ranked:
        key=(rank[3],rank[4])
        if key not in seen:
            seen.add(key)
            if direction is not None and direction not in representatives:
                representatives[direction]=len(unique)
            unique.append(c)
            directions.append(direction)
    selected=[unique[0]];chosen={0};used={directions[0]};cursor=1
    while len(chosen)<len(unique) and len(selected)<int(count):
        # Gap depends only on the direction bin. Its earliest ranked member
        # wins every within-bin tie; original indices preserve remaining order.
        novel=[(index,direction) for direction,index in representatives.items()
               if direction not in used]
        if novel:
            def separation(item):
                i,direction=item
                gap=min((min(abs(direction-d),8-abs(direction-d))
                         for d in used if d is not None),default=0)
                return gap,-i
            index,_=max(novel,key=separation)
        else:
            while cursor in chosen:
                cursor+=1
            index=cursor
        chosen.add(index)
        selected.append(unique[index])
        used.add(directions[index])
    return selected
