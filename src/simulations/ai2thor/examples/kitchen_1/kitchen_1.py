import argparse
import sys
from functools import partial
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulations.ai2thor.execution import execute_plan, load_plan
from simulations.ai2thor import run_simulation


SCENE_DIRECTORY = Path(__file__).resolve().parent
EXECUTION_PLAN_FILE = SCENE_DIRECTORY / "execution_plan.json"


def execute_saved_plan(navigation, actions, _distance, *, plan):
    """Execute a plan loaded directly from execution_plan.json."""
    return execute_plan(plan, navigation, actions)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Kitchen 1 directly from execution_plan.json."
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="show robot POV windows in addition to the always-on top view",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        execution_plan = load_plan(EXECUTION_PLAN_FILE)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Invalid execution plan: {error}") from None
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None

    run_simulation(
        partial(execute_saved_plan, plan=execution_plan),
        show_camera=args.camera,
    )
