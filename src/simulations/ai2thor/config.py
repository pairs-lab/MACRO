"""Central configuration for the AI2-THOR simulation adapter."""

import json
import math
import os


def _positive_int_from_env(name, default):
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _positive_float_from_env(name, default):
    raw_value = os.environ.get(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


ITHOR_SCENE = os.environ.get("AI2THOR_SCENE", "FloorPlan1")
AGENT_COUNT = _positive_int_from_env("AI2THOR_AGENT_COUNT", 3)
RUNTIME_CONFLICT_POLICY = os.environ.get(
    "AI2THOR_RUNTIME_POLICY", "dag_aware"
).strip().lower()
if RUNTIME_CONFLICT_POLICY not in {
    "dag_aware", "local_distance", "local_delay", "reroute_only"
}:
    raise ValueError(
        "AI2THOR_RUNTIME_POLICY must be dag_aware, local_distance, "
        "local_delay, or reroute_only"
    )
RUNTIME_PLANNING_TIMEOUT = _positive_float_from_env(
    "AI2THOR_PLANNING_TIMEOUT", 30.0
)
INTERACTION_POSITION_OVERRIDES = {
    str(name).casefold(): {
        "x": float(position["x"]),
        "z": float(position["z"]),
        "required": bool(position.get("required", False)),
        "tolerance": float(position.get("tolerance", .03)),
    }
    for name, position in json.loads(
        os.environ.get("AI2THOR_INTERACTION_POSITIONS", "{}")
    ).items()
}

# Native 720p is the best quality/FPS point for three simultaneous agents on
# the RTX 4050. Displaying it 1:1 avoids the blur caused by OpenCV resampling.
RENDER_QUALITY = "Ultra"
VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
CAMERA_FIELD_OF_VIEW = 75

DISPLAY_WIDTH = VIEW_WIDTH
DISPLAY_HEIGHT = VIEW_HEIGHT
ROBOT_DISPLAY_WIDTH = VIEW_WIDTH
ROBOT_DISPLAY_HEIGHT = VIEW_HEIGHT

OVERHEAD_WINDOW = "AI2-THOR - Fixed overhead camera"
ROBOT_WINDOW_PREFIX = "AI2-THOR - Robot"
ROBOT_CAMERA_COUNT = AGENT_COUNT
ROBOT_CAMERA_PITCH = 60
AGENT_RADIUS = 0.05
# Symmetric local-planning footprint. Every robot owns the same circle; two
# robots conflict when their centre distance is less than twice this radius.
ROBOT_SAFETY_RADIUS = 0.12
# A carried object extends in front of the circular robot body. Apply this
# margin only while either robot in a pair has an inventory object.
CARRIED_OBJECT_SAFETY_MARGIN = 0.40
# A carried receptacle is compact and centred in front of the agent. It does
# not need the same conservative clearance as a loose item such as Apple.
CARRIED_OBJECT_SAFETY_MARGINS = {
    "bowl": 0.06,
    # Slender utensils have a substantially smaller horizontal footprint than
    # the generic loose-object bound.  Type-specific margins keep the swept
    # safety test conservative without making every narrow-kitchen release
    # pose infeasible.
    "spoon": 0.10,
    "butterknife": 0.12,
    "bread": 0.12,
}


def carried_object_safety_margin(event_or_metadata):
    """Return the configured robot-to-robot margin for held inventory."""
    metadata = getattr(event_or_metadata, "metadata", event_or_metadata)
    inventory = metadata.get("inventoryObjects") or ()
    margins = []
    position = metadata.get("agent", {}).get("position")
    objects = {obj.get("objectId"): obj for obj in metadata.get("objects", ())}
    for item in inventory:
        object_type = item.get("objectType")
        if not object_type:
            object_id = item.get("objectId", "")
            object_type = object_id.split("|", 1)[0].split("_", 1)[0]
        margins.append(
            CARRIED_OBJECT_SAFETY_MARGINS.get(
                object_type.casefold(),
                CARRIED_OBJECT_SAFETY_MARGIN,
            )
        )
        held = objects.get(item.get("objectId"), {})
        bounds = held.get("axisAlignedBoundingBox")
        if position and bounds:
            center, size = bounds["center"], bounds["size"]
            # Bound the observed payload about its owner's centre, including
            # hand extension. A type-only margin misses a long held utensil.
            extent = math.hypot(
                abs(center["x"] - position["x"]) + size["x"] / 2,
                abs(center["z"] - position["z"]) + size["z"] / 2,
            )
            # The type table is a fallback when Unity omits geometry. Keeping
            # its generic 40 cm bound as well would close valid passages for
            # a small retracted fruit even when its actual extent is known.
            margins[-1] = max(0.0, extent - ROBOT_SAFETY_RADIUS)
    return max(margins, default=0.0)
# Extra wall/furniture clearance while carrying an object. This is separate
# from inter-robot clearance: applying the larger robot-to-robot payload margin
# to walls would unnecessarily close the kitchen's narrow passages.
# The hand is retracted after pickup. A 7.5 cm static-obstacle ring keeps the
# stowed payload clear while preserving valid interaction poses close to a
# counter, such as the configured FloorPlan2 sink pose.
CARRIED_OBJECT_WALL_PADDING = 0.075

AGENT_ACTIONS = {
    "w": "MoveAhead",
    "s": "MoveBack",
    "a": "RotateLeft",
    "d": "RotateRight",
}

SIMULATION_SPEED = 2.0
ACTION_RATE_HZ = int(30 * SIMULATION_SPEED)
DISPLAY_RATE_HZ = 60
MANUAL_MOVE_STEP = 0.05 * SIMULATION_SPEED
MANUAL_ROTATION_STEP = 5 * SIMULATION_SPEED

NAVIGATION_STEP_DELAY = 0.0
NAVIGATION_MOVE_STEP = 0.10
NAVIGATION_ROTATION_STEP = 5 * SIMULATION_SPEED
NAVIGATION_GRID_SIZE = 0.05
NAVIGATION_POSITION_TOLERANCE = 0.075
# Extra free space required around the robot by the A* planner.  Together with
# AGENT_RADIUS this gives a planned collision radius of 0.13 m while still
# allowing routes close to walls and fixed furniture.
NAVIGATION_PADDING = 0.05
# Centre-to-centre clearance reserved around every other real agent. This is
# larger than two agent radii so A* does not merely graze a stationary robot.
MULTI_AGENT_CLEARANCE = 2 * ROBOT_SAFETY_RADIUS
NAVIGATION_REPLAN_ATTEMPTS = 3
NAVIGATION_CACHE_ENABLED = True
SHOW_NAVIGATION_PATH = False
NAVIGATION_PATH_PREVIEW_SECONDS = 0.0

# Smooth A* path following. The path itself is unchanged; these values only
# control look-ahead steering and the size of each rendered motion increment.
SMOOTH_NAVIGATION_ENABLED = True
SMOOTH_SAMPLE_SPACING = 0.05
SMOOTH_LOOKAHEAD_DISTANCE = 0.18
# Detect an approaching robot before payload meshes reach the hard clearance
# boundary, leaving enough room for a physically executable release motion.
ONLINE_CONFLICT_LOOKAHEAD_BUFFER = 0.20
SMOOTH_MOVE_STEP = NAVIGATION_MOVE_STEP
SMOOTH_ROTATION_STEP = 10.0
SMOOTH_LARGE_ROTATION_STEP = 90.0
# Ignore tiny look-ahead heading changes instead of showing a steer/move pulse
# at every navigation step.
SMOOTH_HEADING_TOLERANCE = 6.0
SMOOTH_MAX_HEADING_ERROR = 30.0
SMOOTH_FINAL_POSITION_TOLERANCE = 0.03
NAVIGATION_FINAL_ROTATION_STEP = 45.0

# Manipulation is allowed only from a nearby pose while the robot faces the
# target. These are semantic action constraints, separate from path padding.
MAXIMUM_INTERACTION_DISTANCE = 0.70
INTERACTION_DISTANCE_OVERRIDES = {
    str(name).casefold(): float(value)
    for name, value in json.loads(os.environ.get("AI2THOR_INTERACTION_DISTANCES", "{}")).items()
}
if any(not math.isfinite(v) or v <= .04 for v in INTERACTION_DISTANCE_OVERRIDES.values()):
    raise ValueError("Interaction distance overrides must be finite and greater than 0.04 m")


def interaction_distance_for(scene_object):
    """One shared range contract for route goals and live manipulation."""
    return INTERACTION_DISTANCE_OVERRIDES.get(
        scene_object.get("objectId", "").casefold(),
        INTERACTION_DISTANCE_OVERRIDES.get(
            scene_object.get("objectType", "").casefold(), MAXIMUM_INTERACTION_DISTANCE
        ),
    )


MAXIMUM_INTERACTION_YAW_ERROR = 10.0

# Shared manipulation duration used by both optimization and real execution.
TASK_ACTION_DURATIONS = {
    "pick": 1.0,
    "place": 1.0,
    "open": 1.0,
    "close": 1.0,
    "toggle_on": 1.0,
    "toggle_off": 1.0,
    "wash": 1.0,
}

ONLINE_REPLANNING_ROBOT_SPEED = 1.0
# Search a broad local area for parking when a narrow route has to be cleared.
# A* still minimizes distance, so this is only a feasibility bound.
ONLINE_ASIDE_MAX_DISTANCE = 2.50
ONLINE_RELEASE_CANDIDATE_COUNT = 16
ONLINE_MAX_YIELDS_PER_NAVIGATION = 2
