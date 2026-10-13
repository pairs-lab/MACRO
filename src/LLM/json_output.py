"""Serialize validated task graphs as JSON."""

import json
from pathlib import Path


DEFAULT_OUTPUT_PATH = Path(__file__).with_name("tasks.json")


def tasks_to_json(tasks: list[dict]) -> str:
    """Return formatted UTF-8 JSON for a validated task list."""
    return json.dumps(tasks, indent=2, ensure_ascii=False)


def save_tasks(tasks: list[dict], output_path=DEFAULT_OUTPUT_PATH) -> Path:
    """Save tasks and return the resolved output path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tasks_to_json(tasks) + "\n", encoding="utf-8")
    return path
