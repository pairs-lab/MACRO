"""Parse and validate the task graph returned by the language model."""

import json

import networkx as nx
from simulations.ai2thor.capabilities import add_candidate_robots


ALLOWED_ACTIONS = {
    "pick",
    "place",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
}
REQUIRED_FIELDS = {"id", "action", "object", "target", "node"}


def parse_output(response_text: str) -> list[dict]:
    """Decode the model response into a Python task list."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("LLM returned an empty response.")

    text = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"LLM output is not valid JSON: {error}") from error


def validate_tasks(tasks: list[dict]) -> None:
    """Validate task fields, IDs, dependencies, and supported actions."""
    if not isinstance(tasks, list):
        raise ValueError("LLM output must be a JSON list.")

    ids = []
    for task in tasks:
        if not isinstance(task, dict) or set(task) != REQUIRED_FIELDS:
            raise ValueError(f"Invalid fields in task: {task!r}")
        if type(task["id"]) is not int:
            raise ValueError("Task ID must be an integer.")
        if task["action"] not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action: {task['action']!r}")
        if not isinstance(task["object"], str):
            raise ValueError("object must be a string.")
        if not isinstance(task["target"], str):
            raise ValueError("target must be a string.")
        if not isinstance(task["node"], list) or not all(
            type(dependency) is int for dependency in task["node"]
        ):
            raise ValueError("node must be a list of integer task IDs.")
        ids.append(task["id"])

    expected_ids = list(range(1, len(tasks) + 1))
    if ids != expected_ids:
        raise ValueError(
            f"Task IDs must be sequential. Expected {expected_ids}, got {ids}."
        )

    id_set = set(ids)
    for task in tasks:
        for dependency in task["node"]:
            if dependency not in id_set:
                raise ValueError(
                    f"Task {task['id']} depends on missing task {dependency}."
                )
            if dependency == task["id"]:
                raise ValueError(f"Task {task['id']} depends on itself.")


def build_dag(tasks: list[dict]) -> nx.DiGraph:
    """Build the dependency graph and reject cyclic model output."""
    graph = nx.DiGraph()
    for task in tasks:
        graph.add_node(
            task["id"],
            action=task["action"],
            object=task["object"],
            target=task["target"],
            candidate=task.get("candidate", []).copy(),
        )
        for predecessor in task["node"]:
            graph.add_edge(predecessor, task["id"])

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Generated graph contains a cycle.")
    return graph


def get_ready_tasks(tasks: list[dict], completed_tasks) -> list[dict]:
    """Return incomplete tasks whose predecessors are all complete."""
    completed = set(completed_tasks)
    return [
        task
        for task in tasks
        if task["id"] not in completed
        and all(dependency in completed for dependency in task["node"])
    ]


def process_output(response_text: str):
    """Parse and fully validate an LLM response."""
    tasks = parse_output(response_text)
    validate_tasks(tasks)
    tasks = add_candidate_robots(tasks)
    return tasks, build_dag(tasks)
