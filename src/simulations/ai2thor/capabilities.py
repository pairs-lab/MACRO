"""AI2-THOR capability matching between JSON tasks and robots."""

import json
from pathlib import Path

from simulations.ai2thor.config import AGENT_COUNT


FULL_CAPABILITIES = frozenset(
    {
        "pick",
        "place",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
    }
)

# Every configured iTHOR agent currently supports the complete manipulation
# action set. This remains the single source of truth for task candidates.
ROBOT_CAPABILITIES = {
    f"R{index + 1}": FULL_CAPABILITIES
    for index in range(AGENT_COUNT)
}

ROBOT_TO_SIMULATION_NAME = {
    f"R{index + 1}": f"robot{index + 1}" for index in range(AGENT_COUNT)
}

ACTION_REQUIREMENTS = {
    action: frozenset({action}) for action in FULL_CAPABILITIES
}

DEFAULT_TASK_FILE = (
    Path(__file__).resolve().parents[1] / "scene" / "kitchen_1" / "tasks.json"
)


def load_tasks(task_file=DEFAULT_TASK_FILE) -> list[dict]:
    """Read a task list from JSON without changing the source file."""
    path = Path(task_file).expanduser().resolve()
    try:
        tasks = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Task file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid task JSON in {path}: {error}") from error

    if not isinstance(tasks, list):
        raise ValueError("Task JSON must contain a list.")
    return tasks


def get_required_capabilities(task: dict) -> frozenset[str]:
    """Return capabilities required by one task according to fixed rules."""
    action = task.get("action")
    if action not in ACTION_REQUIREMENTS:
        raise ValueError(f"Unsupported task action: {action!r}")
    return ACTION_REQUIREMENTS[action]


def robot_can_execute(
    robot_id: str,
    task: dict,
    robot_capabilities=None,
) -> bool:
    """Check whether one robot satisfies every capability required by a task."""
    capabilities = (
        ROBOT_CAPABILITIES
        if robot_capabilities is None
        else robot_capabilities
    )
    if robot_id not in capabilities:
        raise ValueError(f"Unknown robot: {robot_id!r}")
    required = get_required_capabilities(task)
    return required.issubset(capabilities[robot_id])


def get_candidate_robots(task: dict, robot_capabilities=None) -> list[str]:
    """Return all capable robot IDs in deterministic priority order."""
    capabilities = (
        ROBOT_CAPABILITIES
        if robot_capabilities is None
        else robot_capabilities
    )
    return [
        robot_id
        for robot_id in capabilities
        if robot_can_execute(robot_id, task, capabilities)
    ]


def add_candidate_robots(
    tasks: list[dict],
    robot_capabilities=None,
) -> list[dict]:
    """Return task copies enriched with their rule-based candidate robots."""
    result = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"Each task must be an object, got: {task!r}")
        enriched_task = task.copy()
        enriched_task["candidate"] = get_candidate_robots(
            task,
            robot_capabilities,
        )
        result.append(enriched_task)
    return result


def get_task_candidates(task_file=DEFAULT_TASK_FILE) -> list[dict]:
    """Load tasks and return candidate robots for every task."""
    return add_candidate_robots(load_tasks(task_file))


def to_simulation_robot_name(robot_id: str) -> str:
    """Translate an R1..RN scheduler ID into robot1..robotN."""
    try:
        return ROBOT_TO_SIMULATION_NAME[robot_id]
    except KeyError as error:
        raise ValueError(f"Unknown robot: {robot_id!r}") from error


def main() -> None:
    print(json.dumps(get_task_candidates(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
