"""End-to-end Gemini -> optimizer -> execution pipeline for Kitchen 1."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulations.ai2thor.capabilities import add_candidate_robots
from simulations.ai2thor.execution import execute_plan
from LLM import decompose_instruction, process_output
from core import optimize_task_assignments, save_plan
from simulations.ai2thor import run_simulation


SCENE_DIRECTORY = Path(__file__).resolve().parent
EXECUTION_PLAN_FILE = SCENE_DIRECTORY / "execution_plan.json"


def execute_tasks(navigation, actions, distance, *, tasks):
    """Optimize validated in-memory tasks and execute the saved plan."""
    tasks = add_candidate_robots(tasks)
    final_plan = optimize_task_assignments(tasks, distance, robot_speed=1.0)
    save_plan(final_plan, EXECUTION_PLAN_FILE)

    print("\nFINAL PLAN")
    print(json.dumps(final_plan, indent=2, ensure_ascii=False))
    execute_plan(EXECUTION_PLAN_FILE, navigation, actions)


def generate_tasks(instruction):
    """Use Gemini to turn one natural-language instruction into a task DAG."""
    print("Sending instruction to Gemini...")
    response_text = decompose_instruction(instruction)
    tasks, graph = process_output(response_text)
    if not tasks:
        raise ValueError("Gemini generated an empty task plan.")

    print("\nGENERATED TASKS")
    print(json.dumps(tasks, indent=2, ensure_ascii=False))
    print("DAG edges:", list(graph.edges()))
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the end-to-end Gemini kitchen pipeline."
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="show robot POV windows in addition to the always-on top view",
    )
    parser.add_argument(
        "instruction",
        nargs="*",
        help="optional household instruction; prompts when omitted",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    instruction = " ".join(args.instruction).strip()
    if not instruction:
        instruction = input("Enter household instruction: ").strip()
    if not instruction:
        raise SystemExit("Instruction cannot be empty.")

    try:
        generated_tasks = generate_tasks(instruction)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Pipeline stopped: {error}") from None

    run_simulation(
        partial(execute_tasks, tasks=generated_tasks),
        show_camera=args.camera,
    )
