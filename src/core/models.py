"""Shared robot state schema."""
from dataclasses import dataclass, field

@dataclass
class RobotState:
    robot_id: str
    cell: tuple[int, int]
    color: tuple[int, int, int]
    goal: tuple[int, int] | None = None
    path: list[tuple[int, int]] = field(default_factory=list)
    path_index: int = 0
    travelled: float = 0.0
    phase: str = "idle"
    current_task: dict | None = None
    action_remaining: float = 0.0
    interrupted_action_remaining: float | None = None
    preview_remaining: float = 0.0
    carrying: str | None = None
    waiting_for: str | None = None
    clearing_for: tuple[str, ...] = ()
    clearing_started: bool = False
    movement_credit: float = 0.0
    movement_edge: tuple | None = None

