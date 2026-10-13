"""Command-line entry point for LLM task decomposition."""

from .gemini_client import decompose_instruction
from .json_output import save_tasks, tasks_to_json
from .process_output import get_ready_tasks, process_output


def main() -> None:
    instruction = input("Enter household instruction: ").strip()
    response_text = decompose_instruction(instruction)
    tasks, graph = process_output(response_text)
    output_path = save_tasks(tasks)
    ready_tasks = get_ready_tasks(tasks, completed_tasks=[])

    print("\nGENERATED TASKS")
    print(tasks_to_json(tasks))
    print("\nDAG EDGES")
    print(list(graph.edges()))
    print("\nREADY TASKS AT t=0")
    print(tasks_to_json(ready_tasks))
    print(f"\nSaved task graph to: {output_path}")


if __name__ == "__main__":
    main()
