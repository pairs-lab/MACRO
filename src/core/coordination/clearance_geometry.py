"""Bounded snapshot-local memoization of pure clearance geometry.

No cell is labeled globally safe: path-history-dependent body transitions
are keyed by the incoming cleared mask as well as the edge and all centres.
Corridor geometry and endpoint distance are independent of path history.
"""
from core.yielding import _prepare_polyline, _distance_to_prepared_polyline
from .dynamic_collision import swept_robot_corridor
from .planner import _GEOMETRY_CACHE

MAX_CORRIDORS = 64
MAX_MARGINS = 65536
MAX_BODY_TRANSITIONS = 32768


def context():
    cache = _GEOMETRY_CACHE.get()
    return cache if cache is not None and cache.get('clearance_reuse') else None


def protected_geometry(paths, radius, grid_size):
    cache = context()
    signature = (tuple(tuple(p) for p in paths), radius, grid_size)
    if cache is not None:
        stored = cache.setdefault('clearance_corridors', {})
        if signature in stored:
            cache['clearance_stats']['corridor_hits'] += 1
            return stored[signature]
        cache['clearance_stats']['corridor_misses'] += 1
    forbidden = set()
    for route in signature[0]:
        forbidden.update(swept_robot_corridor(route, radius, radius, grid_size))
    # Only immutable geometry is shared. Endpoint-distance maps are internal.
    cacheable = cache is not None and len(stored) < MAX_CORRIDORS
    prepared = tuple(_prepare_polyline(path) for path in signature[0])
    result = (frozenset(forbidden), signature, {} if cacheable else None, prepared)
    if cache is not None and len(stored) < MAX_CORRIDORS:
        stored[signature] = result
    return result


def endpoint_margin(endpoint, geometry):
    _, signature, distances, prepared = geometry
    _, radius, grid_size = signature
    cache = context()
    if cache is not None and distances is not None and endpoint in distances:
        cache['clearance_stats']['margin_hits'] += 1
        return distances[endpoint]
    value = min((_distance_to_prepared_polyline(endpoint, p) for p in prepared),
                default=float('inf')) * grid_size - 2 * radius
    if cache is not None:
        cache['clearance_stats']['margin_misses'] += 1
        count = cache.get('clearance_margin_count', 0)
        if distances is not None and count < MAX_MARGINS:
            distances[endpoint] = value
            cache['clearance_margin_count'] = count + 1
    return value


def endpoint_margins(endpoints, geometry):
    """Batch cache misses without changing rounding or endpoint order."""
    from .native_geometry import backend
    native = backend()
    if native is None or not hasattr(native, 'endpoint_margins'):
        return [endpoint_margin(endpoint, geometry) for endpoint in endpoints]
    _, signature, distances, prepared = geometry
    _, radius, grid_size = signature
    cache = context()
    result = []
    missing = []
    indices = []
    for endpoint in endpoints:
        if cache is not None and distances is not None and endpoint in distances:
            cache['clearance_stats']['margin_hits'] += 1
            result.append(distances[endpoint])
        else:
            indices.append(len(result))
            missing.append(endpoint)
            result.append(None)
    values = native.endpoint_margins(missing, prepared, radius, grid_size) if missing else []
    for index, endpoint, value in zip(indices, missing, values):
        result[index] = value
        if cache is not None:
            cache['clearance_stats']['margin_misses'] += 1
            count = cache.get('clearance_margin_count', 0)
            if distances is not None and count < MAX_MARGINS:
                distances[endpoint] = value
                cache['clearance_margin_count'] = count + 1
    return result
