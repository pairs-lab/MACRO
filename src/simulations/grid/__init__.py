"""AI2-THOR-independent 2D simulation adapter."""

from .app import run_grid_simulation
from core.coordination.dynamic_collision import (
    check_dynamic_robot_collisions,
    swept_robot_corridor,
)
from core.coordination.planner import select_preferred_path
from core.coordination.mdgs import mdgs_resolve

__all__ = [
    "check_dynamic_robot_collisions",
    "run_grid_simulation",
    "mdgs_resolve",
    "select_preferred_path",
    "swept_robot_corridor",
]
