"""AI2-THOR-independent 2D simulation adapter."""

from .app import run_grid_simulation
from core.coordination.dynamic_collision import (
    check_dynamic_robot_collisions,
    swept_robot_corridor,
)
from core.coordination.planner import select_preferred_path
from core.coordination.relocation_search import search_relocation_plan

__all__ = [
    "check_dynamic_robot_collisions",
    "run_grid_simulation",
    "search_relocation_plan",
    "select_preferred_path",
    "swept_robot_corridor",
]
