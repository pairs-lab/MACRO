"""Door-swing constraints for deterministic AI2-THOR scenes."""

from simulations.ai2thor.config import ITHOR_SCENE


# These are map facts, not a hard-coded route. A* remains responsible for
# selecting the path while treating the future open door as a static obstacle.
_RULES = {
    "FloorPlan1": {
        "fridge": {
            # The first rectangle is the physical swing. The second reserves
            # the pocket behind the open door as unusable parking space, so a
            # clearing robot cannot hide there and trap the incoming carrier.
            # Padding is applied by the grid planner.
            "forbidden_regions": [
                {"min_x": -1.72, "max_x": -0.75,
                 "min_z": 0.30, "max_z": 0.72},
                {"min_x": -1.72, "max_x": -0.75,
                 "min_z": 0.72, "max_z": 1.75},
            ],
            # Carrying robots approach from beyond the non-swinging end. This
            # keeps the payload clear of both the cabinet and the open door.
            # Stay beyond the open door, but close to the refrigerator body.
            # Without max_x, A* can choose x=-0.65 at the outer door edge,
            # which is visible to Unity but too far away for realistic use.
            "goal_filter": {"min_z": 1.90, "max_x": -1.20},
            "placement_point": {"x": -2.00, "y": 0.85, "z": 1.07},
        },
    },
}


def future_open_rule(object_type):
    """Return the known open-state navigation rule for an object, if any."""
    return _RULES.get(ITHOR_SCENE, {}).get(object_type.casefold())
