"""AI2-THOR scene selection and multi-agent controller creation."""

import math
import threading

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from .config import (
    AGENT_COUNT,
    AGENT_RADIUS,
    CAMERA_FIELD_OF_VIEW,
    ITHOR_SCENE,
    NAVIGATION_GRID_SIZE,
    RENDER_QUALITY,
    VIEW_HEIGHT,
    VIEW_WIDTH,
)


def load_scene():
    """Return the configured iTHOR scene name."""
    print(f"Loading iTHOR scene {ITHOR_SCENE} with {AGENT_COUNT} agents...")
    return ITHOR_SCENE


def require_success(event, operation):
    """Raise a readable error when an AI2-THOR action fails."""
    if not event.metadata["lastActionSuccess"]:
        message = event.metadata.get("errorMessage", "Unknown AI2-THOR error")
        raise RuntimeError(f"{operation} failed: {message}")
    return event


def _distance_xz(first, second):
    return math.hypot(first["x"] - second["x"], first["z"] - second["z"])


def _synchronize_controller_steps(controller):
    """Serialize access to AI2-THOR's shared Unity request socket."""
    step_lock = threading.RLock()
    original_step = controller.step

    def synchronized_step(*args, **kwargs):
        with step_lock:
            return original_step(*args, **kwargs)

    controller.step = synchronized_step
    controller.step_lock = step_lock
    return controller


def _spread_agents(controller):
    """Place the real agents at distant valid points instead of one cluster."""
    reachable_event = require_success(
        controller.step(action="GetReachablePositions", agentId=0),
        "Find initial agent positions",
    )
    reachable = reachable_event.metadata["actionReturn"]
    if len(reachable) < AGENT_COUNT:
        raise RuntimeError("The iTHOR scene has too few reachable agent positions")

    # Farthest-point sampling gives deterministic, well-separated spawn poses.
    center_x = sum(point["x"] for point in reachable) / len(reachable)
    center_z = sum(point["z"] for point in reachable) / len(reachable)
    center = {"x": center_x, "z": center_z}
    selected = [max(reachable, key=lambda point: _distance_xz(point, center))]
    while len(selected) < AGENT_COUNT:
        selected.append(
            max(
                reachable,
                key=lambda point: min(
                    _distance_xz(point, chosen) for chosen in selected
                ),
            )
        )

    for agent_id, position in enumerate(selected):
        require_success(
            controller.step(
                action="TeleportFull",
                agentId=agent_id,
                x=position["x"],
                y=position["y"],
                z=position["z"],
                rotation={"x": 0, "y": (agent_id * 120) % 360, "z": 0},
                horizon=0,
                standing=True,
                forceAction=True,
            ),
            f"Place robot{agent_id + 1}",
        )


def _place_agents(controller, placements):
    """Apply editable robot poses from a map layout file."""
    if not placements:
        _spread_agents(controller)
        return
    if len(placements) < AGENT_COUNT:
        # Establish the same default starts, then override only named robots.
        _spread_agents(controller)

    reachable = require_success(
        controller.step(action="GetReachablePositions", agentId=0),
        "Find configured robot positions",
    ).metadata["actionReturn"]
    used = set()
    resolved = []
    for placement in placements:
        robot = placement.get("id", "")
        try:
            agent_id = int(robot.removeprefix("R")) - 1
        except ValueError as error:
            raise ValueError(f"Invalid robot id in layout: {robot!r}") from error
        if agent_id not in range(AGENT_COUNT) or agent_id in used:
            raise ValueError(f"Invalid or duplicate robot id in layout: {robot!r}")
        used.add(agent_id)
        requested = placement.get("position", {})
        if not all(axis in requested for axis in ("x", "z")):
            raise ValueError(f"{robot} requires position.x and position.z")
        requested = {
            "x": float(requested["x"]),
            "z": float(requested["z"]),
        }
        closest = min(
            reachable,
            key=lambda point: _distance_xz(point, requested),
        )
        snap_distance = _distance_xz(closest, requested)
        if snap_distance > 0.08:
            raise ValueError(
                f"{robot} position is not reachable; nearest grid point is "
                f"{snap_distance:.2f} m away"
            )
        resolved.append((
            agent_id,
            closest,
            float(placement.get("rotation", 0.0)) % 360,
        ))

    safety_positions = list(resolved)
    for agent_id in range(AGENT_COUNT):
        if agent_id not in used:
            agent = controller.last_event.events[agent_id].metadata["agent"]
            safety_positions.append((agent_id, agent["position"], agent["rotation"]["y"]))
    for index, (first_id, first, _) in enumerate(safety_positions):
        for second_id, second, _ in safety_positions[index + 1:]:
            if _distance_xz(first, second) < 2 * AGENT_RADIUS:
                raise ValueError(
                    f"R{first_id + 1} and R{second_id + 1} overlap in layout"
                )

    for agent_id, position, rotation in sorted(resolved):
        require_success(
            controller.step(
                action="TeleportFull",
                agentId=agent_id,
                x=position["x"],
                y=position["y"],
                z=position["z"],
                rotation={"x": 0, "y": rotation, "z": 0},
                horizon=0,
                standing=True,
                forceAction=True,
            ),
            f"Place robot{agent_id + 1} from layout",
        )


def _place_objects(controller, placements):
    """Move enabled pickupable objects from a map layout file."""
    for placement in placements or ():
        position = placement.get("position", {})
        if not all(axis in position for axis in ("x", "z")):
            raise ValueError(
                "Enabled object layout entries require position.x/z (y defaults to the current height)"
            )
        object_id = placement.get("object_id")
        objects = controller.last_event.metadata.get("objects", [])
        matches = [obj for obj in objects if obj.get("objectId") == object_id]
        if not matches:
            name = str(placement.get("name", ""))
            instance = int(placement.get("instance", 0))
            matches = sorted(
                (
                    obj for obj in objects
                    if obj.get("objectType", "").casefold() == name.casefold()
                ),
                key=lambda obj: obj["objectId"],
            )
            matches = matches[instance:instance + 1]
        if not matches:
            raise ValueError(
                f"Cannot find layout object {object_id or placement.get('name')!r}"
            )
        target = matches[0]
        if not target.get("pickupable", False):
            raise ValueError(
                f"{target['objectType']} is fixed and cannot be repositioned"
            )
        require_success(
            controller.step(
                action="PlaceObjectAtPoint",
                objectId=target["objectId"],
                position={
                    axis: float(position[axis] if axis in position else target["position"][axis])
                    for axis in ("x", "y", "z")
                },
                forceKinematic=bool(placement.get("force_kinematic", True)),
                renderImage=False,
            ),
            f"Place {target['objectType']} from layout",
        )


def _apply_initial_object_states(controller, states):
    """Set scene door states before navigation/mission execution, not as tasks."""
    for state in states or ():
        objects = controller.last_event.metadata.get("objects", [])
        if state.get("object_id"):
            matches = [o for o in objects if o.get("objectId") == state["object_id"]]
        else:
            matches = sorted([o for o in objects if o.get("objectType", "").casefold()
                              == state.get("name", "").casefold()], key=lambda o:o["objectId"])
            index = int(state.get("instance", 0))
            if index < 0:
                raise ValueError("Initial object state instance must be nonnegative")
            matches = matches[index:index+1]
        if not matches:
            raise ValueError(f"Cannot find initial-state object {state.get('name') or state.get('object_id')!r}")
        target = matches[0]
        if not target.get("openable", False):
            raise ValueError(f"{target['objectType']} is not openable")
        if type(state.get("is_open")) is not bool:
            raise ValueError("Initial is_open must be boolean")
        if target.get("isOpen", False) == state["is_open"]:
            continue
        require_success(controller.step(
            action="OpenObject" if state["is_open"] else "CloseObject",
            objectId=target["objectId"], agentId=0,
            # Explicit scene setup may operate off-camera. Runtime manipulation
            # still requires the real robot's visible interaction pose.
            forceAction=True, renderImage=False,
        ), f"Initialize {target['objectType']} door state")


def create_controller(
    scene,
    *,
    robot_placements=None,
    object_placements=None,
    initial_object_states=None,
):
    """Create the true multi-agent controller used by the simulation."""
    controller = Controller(
        scene=scene,
        platform=CloudRendering,
        quality=RENDER_QUALITY,
        width=VIEW_WIDTH,
        height=VIEW_HEIGHT,
        fieldOfView=CAMERA_FIELD_OF_VIEW,
        gridSize=NAVIGATION_GRID_SIZE,
        snapToGrid=False,
        rotateStepDegrees=5,
        agentRadius=AGENT_RADIUS,
        agentCount=AGENT_COUNT,
        makeAgentsVisible=True,
        headless=False,
    )
    _synchronize_controller_steps(controller)
    require_success(controller.last_event, "Create iTHOR scene")
    if len(controller.last_event.events) != AGENT_COUNT:
        raise RuntimeError(
            f"Expected {AGENT_COUNT} iTHOR agents, got "
            f"{len(controller.last_event.events)}"
        )
    _place_objects(controller, object_placements)
    _apply_initial_object_states(controller, initial_object_states)
    _place_agents(controller, robot_placements)
    return controller
