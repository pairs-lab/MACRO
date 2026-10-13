"""A* planning on AI2-THOR reachable positions with clearance."""

import heapq
import gzip
import json
import math
import os
from pathlib import Path
import tempfile

from simulations.ai2thor.config import (
    AGENT_RADIUS,
    CARRIED_OBJECT_WALL_PADDING,
    ITHOR_SCENE,
    MULTI_AGENT_CLEARANCE,
    NAVIGATION_CACHE_ENABLED,
    SMOOTH_FINAL_POSITION_TOLERANCE,
    SMOOTH_MOVE_STEP,
    carried_object_safety_margin,
)
from simulations.ai2thor.controller import require_success


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIRECTORY = _PROJECT_ROOT / ".navigation_cache"
_MEMORY_POSITIONS = {}
_MEMORY_GLOBAL_MAPS = {}
_MEMORY_PAYLOAD_MAPS = {}


def _cache_path(grid_size):
    layout = os.environ.get("AI2THOR_LAYOUT_FINGERPRINT", "")
    filename = (
        f"{ITHOR_SCENE}{'_' + layout if layout else ''}_"
        f"grid_{grid_size:.3f}_radius_{AGENT_RADIUS:.3f}.json.gz"
    )
    return _CACHE_DIRECTORY / filename


def _global_cache_path(grid_size, padding):
    layout = os.environ.get("AI2THOR_LAYOUT_FINGERPRINT", "")
    filename = (
        f"{ITHOR_SCENE}{'_' + layout if layout else ''}_global_grid_{grid_size:.3f}_"
        f"radius_{AGENT_RADIUS:.3f}_padding_{padding:.3f}.json.gz"
    )
    return _CACHE_DIRECTORY / filename


def _load_cached_positions(grid_size):
    if not NAVIGATION_CACHE_ENABLED:
        return None

    path = _cache_path(grid_size)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as cache_file:
            positions = json.load(cache_file)
    except (OSError, ValueError):
        return None

    print(f"Loaded {len(positions)} navigation cells from {path.name}")
    return positions


def _save_cached_positions(positions, grid_size):
    if not NAVIGATION_CACHE_ENABLED:
        return

    _CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = _cache_path(grid_size)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=path.name,
            suffix=".tmp",
            dir=_CACHE_DIRECTORY,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        with gzip.open(temporary_path, "wt", encoding="utf-8") as cache_file:
            json.dump(positions, cache_file, separators=(",", ":"))
        os.replace(temporary_path, path)
        print(f"Saved {len(positions)} navigation cells to {path.name}")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def get_reachable_positions(controller, grid_size, agent_id=0):
    memory_key = (ITHOR_SCENE, grid_size, AGENT_RADIUS)
    if memory_key in _MEMORY_POSITIONS:
        return _MEMORY_POSITIONS[memory_key]

    positions = _load_cached_positions(grid_size)
    if positions is not None:
        _MEMORY_POSITIONS[memory_key] = positions
        return positions

    print("Scanning reachable positions for the navigation cache...")
    reachable_event = require_success(
        controller.step(action="GetReachablePositions", agentId=agent_id),
        "Get reachable positions for A*",
    )
    positions = reachable_event.metadata["actionReturn"]
    _save_cached_positions(positions, grid_size)
    _MEMORY_POSITIONS[memory_key] = positions
    return positions


def _cell(position, grid_size):
    return (
        int(round(position["x"] / grid_size)),
        int(round(position["z"] / grid_size)),
    )


def _distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _clearance_offsets(grid_size, padding):
    radius = int(math.ceil(padding / grid_size))
    return [
        (dx, dz)
        for dx in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if math.hypot(dx * grid_size, dz * grid_size) <= padding + 1e-9
    ]


def _erode_free_space(cells, grid_size, padding):
    """Remove cells too close to a wall/obstacle (configuration-space padding)."""
    offsets = _clearance_offsets(grid_size, padding)
    return {
        cell
        for cell in cells
        if all((cell[0] + dx, cell[1] + dz) in cells for dx, dz in offsets)
    }


def _payload_free_space(free_cells, grid_size, base_padding):
    """Return the immutable carried-object configuration space once."""
    key = (
        ITHOR_SCENE,
        float(grid_size),
        float(AGENT_RADIUS),
        float(base_padding),
        float(CARRIED_OBJECT_WALL_PADDING),
    )
    cached = _MEMORY_PAYLOAD_MAPS.get(key)
    if cached is None:
        cached = frozenset(
            _erode_free_space(
                free_cells,
                grid_size,
                CARRIED_OBJECT_WALL_PADDING,
            )
        )
        _MEMORY_PAYLOAD_MAPS[key] = cached
    return cached


def _agent_obstacles(
    controller,
    agent_id,
    grid_size,
    agent_position_overrides=None,
    ignored_agent_ids=None,
):
    """Return realtime circular footprints for all other agents.

    Use exactly the same stopping margin as the path follower. Otherwise A*
    can return a geometrically valid route which the controller immediately
    rejects, causing repeated steering at the same pose.
    """
    events = getattr(controller.last_event, "events", None) or []
    overrides = agent_position_overrides or {}
    ignored = set(ignored_agent_ids or ())
    moving_margin = (
        carried_object_safety_margin(events[agent_id])
        if agent_id < len(events)
        else 0.0
    )
    return [
        (
            overrides.get(other_id, event.metadata["agent"]["position"]),
            MULTI_AGENT_CLEARANCE
            + max(moving_margin, carried_object_safety_margin(event))
            + SMOOTH_MOVE_STEP
            + SMOOTH_FINAL_POSITION_TOLERANCE,
        )
        for other_id, event in enumerate(events)
        if other_id != agent_id and other_id not in ignored
    ]


def _remove_other_agents(
    free_cells,
    controller,
    agent_id,
    grid_size,
    agent_position_overrides=None,
    ignored_agent_ids=None,
):
    """Treat agents as circles while preserving an outward escape corridor.

    If two safety circles already overlap, deleting the moving robot's start
    cell makes A* snap to a distant cell and creates an invalid straight jump.
    Cells inside a circle are therefore retained only when they do not move
    closer to that blocker than the robot's current centre distance.
    """
    obstacles = [
        (_cell(position, grid_size), clearance / grid_size)
        for position, clearance in _agent_obstacles(
            controller,
            agent_id,
            grid_size,
            agent_position_overrides=agent_position_overrides,
            ignored_agent_ids=ignored_agent_ids,
        )
    ]
    events = getattr(controller.last_event, "events", None) or []
    moving_position = (agent_position_overrides or {}).get(
        agent_id,
        events[agent_id].metadata["agent"]["position"],
    )
    moving_cell = _cell(moving_position, grid_size)

    def clears_obstacle(cell, centre, clearance_cells):
        distance = _distance(cell, centre)
        if distance > clearance_cells:
            return True
        if cell == moving_cell:
            return True

        # When safety regions overlap, retain the half-plane that does not move
        # toward the other robot. Tangential cells are important in narrow
        # rooms: they let a blocker sidestep a newly moved object while the
        # radial escape edge is physically occupied.
        outward_x = moving_cell[0] - centre[0]
        outward_z = moving_cell[1] - centre[1]
        step_x = cell[0] - moving_cell[0]
        step_z = cell[1] - moving_cell[1]
        outward_progress = step_x * outward_x + step_z * outward_z
        return (
            distance >= _distance(moving_cell, centre) - 1e-9
            and outward_progress >= 0
        )

    return {
        cell
        for cell in free_cells
        if all(
            clears_obstacle(cell, centre, clearance_cells)
            for centre, clearance_cells in obstacles
        )
    }


def _load_global_grid_map(grid_size, padding):
    if not NAVIGATION_CACHE_ENABLED:
        return None

    path = _global_cache_path(grid_size, padding)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as cache_file:
            data = json.load(cache_file)
        positions = data["positions"]
        free_cells = {tuple(cell) for cell in data["freeCells"]}
    except (KeyError, OSError, TypeError, ValueError):
        return None

    print(
        f"Loaded global navigation map: {path.name} "
        f"({len(free_cells)} free cells)"
    )
    return positions, free_cells


def _save_global_grid_map(positions, free_cells, grid_size, padding):
    if not NAVIGATION_CACHE_ENABLED:
        return

    _CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = _global_cache_path(grid_size, padding)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=path.name,
            suffix=".tmp",
            dir=_CACHE_DIRECTORY,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        data = {
            "scene": ITHOR_SCENE,
            "gridSize": grid_size,
            "agentRadius": AGENT_RADIUS,
            "padding": padding,
            "positions": positions,
            "freeCells": [list(cell) for cell in sorted(free_cells)],
        }
        with gzip.open(temporary_path, "wt", encoding="utf-8") as cache_file:
            json.dump(data, cache_file, separators=(",", ":"))
        os.replace(temporary_path, path)
        print(f"Saved global navigation map: {path.name}")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def get_global_grid_map(controller, grid_size, padding, agent_id=0):
    """Load or build the immutable static map once per simulation process."""
    memory_key = (ITHOR_SCENE, grid_size, AGENT_RADIUS, padding)
    cached = _MEMORY_GLOBAL_MAPS.get(memory_key)
    if cached is not None:
        return cached

    loaded = _load_global_grid_map(grid_size, padding)
    if loaded is None:
        print("Building the persistent global navigation map once...")
        positions = get_reachable_positions(controller, grid_size, agent_id)
        position_by_cell = {
            _cell(position, grid_size): position for position in positions
        }
        free_cells = _erode_free_space(
            set(position_by_cell),
            grid_size,
            padding,
        )
        _save_global_grid_map(
            positions,
            free_cells,
            grid_size,
            padding,
        )
    else:
        positions, free_cells = loaded
        position_by_cell = {
            _cell(position, grid_size): position for position in positions
        }
        _MEMORY_POSITIONS[(ITHOR_SCENE, grid_size, AGENT_RADIUS)] = positions

    result = (position_by_cell, free_cells)
    _MEMORY_GLOBAL_MAPS[memory_key] = result
    return result


def prepare_global_grid_map(controller):
    """Preload the persistent global map during simulation startup."""
    from simulations.ai2thor.config import (
        NAVIGATION_GRID_SIZE,
        NAVIGATION_PADDING,
    )

    return get_global_grid_map(
        controller,
        NAVIGATION_GRID_SIZE,
        NAVIGATION_PADDING,
        agent_id=0,
    )


def _nearest_cell(position, cells, grid_size, max_distance=None):
    wanted = _cell(position, grid_size)
    if wanted in cells:
        return wanted

    candidates = cells
    if max_distance is not None:
        radius = int(math.ceil(max_distance / grid_size))
        candidates = {
            (wanted[0] + dx, wanted[1] + dz)
            for dx in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
            if (wanted[0] + dx, wanted[1] + dz) in cells
        }
        if not candidates:
            return None

    nearest = min(candidates, key=lambda candidate: _distance(wanted, candidate))
    distance_metres = _distance(wanted, nearest) * grid_size
    if max_distance is not None and distance_metres > max_distance:
        return None
    return nearest


def _neighbors(cell, free_cells):
    x, z = cell
    for dx, dz in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        neighbor = (x + dx, z + dz)
        if neighbor not in free_cells:
            continue
        # A diagonal step may not cut through an occupied corner.
        if dx and dz:
            if (x + dx, z) not in free_cells or (x, z + dz) not in free_cells:
                continue
        yield neighbor, math.hypot(dx, dz)


def _goal_heuristic(goal_cells):
    """Return an admissible octile-distance heuristic for multiple goals.

    Interaction targets normally expose only a small ring of poses, where an
    exact nearest-goal heuristic is cheap. Relocation search can expose
    thousands of possible parking cells, where evaluating a multi-goal
    heuristic costs more than it saves; that enumeration falls back to the
    zero admissible heuristic while ordinary routes keep true A*.
    """
    goal_cells = tuple(goal_cells)
    diagonal_cost = math.sqrt(2.0)

    def octile(dx, dz):
        diagonal = min(dx, dz)
        straight = max(dx, dz) - diagonal
        return diagonal * diagonal_cost + straight

    if len(goal_cells) <= 32:
        return lambda cell: min(
            octile(abs(cell[0] - goal[0]), abs(cell[1] - goal[1]))
            for goal in goal_cells
        )

    # Relocation exposes a broad set of parking goals whose bounding box often
    # contains the start. Its heuristic is then almost always zero but still
    # costs several Python operations per neighbor. Use the zero admissible
    # heuristic for this multi-goal enumeration; ordinary conflict reroutes,
    # priority routes, and continuation routes retain true octile A* above.
    return lambda _cell: 0.0


def _reconstruct_path(came_from, last):
    path = [last]
    while last in came_from:
        last = came_from[last]
        path.append(last)
    path.reverse()
    return path


def _compress_path(path):
    """Keep turns but discard redundant cells along the same straight segment."""
    if len(path) <= 2:
        return path

    result = [path[0]]
    old_direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for index in range(1, len(path) - 1):
        new_direction = (
            path[index + 1][0] - path[index][0],
            path[index + 1][1] - path[index][1],
        )
        if new_direction != old_direction:
            result.append(path[index])
        old_direction = new_direction
    result.append(path[-1])
    return result


def iter_padded_astar_paths(
    controller,
    start,
    goal_candidates,
    grid_size,
    padding,
    agent_id=0,
    avoid_other_agents=True,
    forbidden_regions=None,
    agent_position_overrides=None,
    ignored_agent_ids=None,
    include_search_cost=False,
):
    """Yield reachable goals in increasing padded-grid path cost.

    Position overrides support local counterfactuals such as "blocker R2 is at
    release pose q" without mutating the live Unity scene.
    """
    position_by_cell, global_free_cells = get_global_grid_map(
        controller,
        grid_size,
        padding,
        agent_id,
    )
    # Every overlay operation below returns a new set. Keep the immutable
    # cached set by reference until an overlay is actually required.
    free_cells = global_free_cells
    events = getattr(controller.last_event, "events", None) or []
    carrying = bool(
        agent_id < len(events)
        and events[agent_id].metadata.get("inventoryObjects")
    )
    if carrying:
        # ReachablePositions only accounts for the Unity agent capsule. Erode
        # the cached static grid one more ring so a carried bowl/object cannot
        # scrape a wall even though the robot body itself fits.
        free_cells = _payload_free_space(
            free_cells,
            grid_size,
            padding,
        )
    for region in forbidden_regions or ():
        clearance = region.get("padding", padding)
        free_cells = {
            cell
            for cell in free_cells
            if not (
                region["min_x"] - clearance
                <= position_by_cell[cell]["x"]
                <= region["max_x"] + clearance
                and region["min_z"] - clearance
                <= position_by_cell[cell]["z"]
                <= region["max_z"] + clearance
            )
        }
    if avoid_other_agents:
        free_cells = _remove_other_agents(
            free_cells,
            controller,
            agent_id,
            grid_size,
            agent_position_overrides=agent_position_overrides,
            ignored_agent_ids=ignored_agent_ids,
        )
    if not free_cells:
        raise RuntimeError("A* has no free cells after applying robot padding")

    start_cell = _nearest_cell(start, free_cells, grid_size)
    snapped_start = position_by_cell[start_cell]
    start_snap_distance = math.hypot(
        float(start["x"]) - float(snapped_start["x"]),
        float(start["z"]) - float(snapped_start["z"]),
    )
    goals = {}
    for position, payload in goal_candidates:
        goal_cell = _nearest_cell(
            position,
            free_cells,
            grid_size,
            # Interaction goals were validated by Unity. Moving them 30 cm
            # can lose visibility or reach; only permit grid rounding here.
            max_distance=(grid_size * 0.51 if position.get("interaction_pose")
                          else max(0.30, padding + grid_size)),
        )
        if goal_cell is not None:
            goals.setdefault(goal_cell, payload)

    if not goals:
        raise RuntimeError("No target pose has enough clearance for the robot")

    # True A*: octile distance is admissible for the 8-connected grid with
    # unit/cardinal and sqrt(2)/diagonal edge costs. The heuristic is computed
    # against the fixed full goal set, so every goal still leaves the queue in
    # nondecreasing optimal path cost when callers request more than one path.
    heuristic = _goal_heuristic(goals)
    queue = [(heuristic(start_cell), 0.0, start_cell)]
    cost = {start_cell: 0.0}
    came_from = {}
    remaining_goals = set(goals)

    while queue and remaining_goals:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost > cost[current]:
            continue
        if current in goals:
            remaining_goals.discard(current)
            cell_path = _compress_path(_reconstruct_path(came_from, current))
            path = [position_by_cell[cell].copy() for cell in cell_path]
            if include_search_cost:
                # The live continuous pose may be offset from its snapped
                # grid cell. Triangle inequality makes this a valid lower
                # bound for the corresponding continuous release path.
                lower_bound = max(
                    0.0,
                    current_cost * grid_size - start_snap_distance,
                )
                yield path, goals[current], lower_bound
            else:
                yield path, goals[current]

        for neighbor, step_cost in _neighbors(current, free_cells):
            new_cost = current_cost + step_cost
            if new_cost >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = new_cost
            came_from[neighbor] = current
            heapq.heappush(
                queue,
                (new_cost + heuristic(neighbor), new_cost, neighbor),
            )



def find_padded_astar_path(
    controller,
    start,
    goal_candidates,
    grid_size,
    padding,
    agent_id=0,
    avoid_other_agents=True,
    forbidden_regions=None,
    agent_position_overrides=None,
    ignored_agent_ids=None,
):
    """Return the least-cost padded A* path to any supplied goal."""
    paths = iter_padded_astar_paths(
        controller,
        start,
        goal_candidates,
        grid_size,
        padding,
        agent_id=agent_id,
        avoid_other_agents=avoid_other_agents,
        forbidden_regions=forbidden_regions,
        agent_position_overrides=agent_position_overrides,
        ignored_agent_ids=ignored_agent_ids,
    )
    try:
        return next(paths)
    except StopIteration as error:
        raise RuntimeError(
            "A* found no safe route to the target with "
            f"{padding:.2f} m robot padding; doors and objects were kept intact"
        ) from error
