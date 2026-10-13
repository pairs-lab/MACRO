"""Run one JSON-defined AI2-THOR scene and execution plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from functools import partial
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"
MAP_DIRECTORY = PROJECT_ROOT / "maps_ai2thor"


def _map_directory(value: str) -> Path:
    requested = Path(value).expanduser()
    directory = requested if requested.is_dir() else MAP_DIRECTORY / value
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"AI2-THOR map directory not found: {directory}")
    return directory


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required JSON file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _merge_scene_placements(scene: dict, layout: dict) -> dict:
    """Scene placements override layout placements by application order."""
    placements = scene.get("object_placements", [])
    if not isinstance(placements, list):
        raise ValueError("scene.json object_placements must be a list")
    for placement in placements:
        if not isinstance(placement, dict):
            raise ValueError("Each object placement must be an object")
        position = placement.get("position", {})
        if (not isinstance(position, dict) or not all(axis in position for axis in ("x", "z"))
                or not (placement.get("name") or placement.get("object_id"))):
            raise ValueError("Object placements require name/object_id and position.x/z")
        try:
            if not all(math.isfinite(float(v)) for v in position.values()):
                raise ValueError("Object placement coordinates must be finite")
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid object placement coordinates") from error
    robot_placements = scene.get("robot_placements", [])
    if not isinstance(robot_placements, list):
        raise ValueError("robot_placements must be a list")
    seen = set()
    for placement in robot_placements:
        if not isinstance(placement, dict) or not isinstance(placement.get("position"), dict):
            raise ValueError("Robot placements require id and position.x/z")
        robot = placement.get("id")
        if robot not in {f"R{i+1}" for i in range(scene["agent_count"])} or robot in seen:
            raise ValueError("Invalid or duplicate robot placement id")
        seen.add(robot)
        try:
            if not all(math.isfinite(float(placement["position"][axis])) for axis in ("x", "z")):
                raise ValueError("Nonfinite robot position")
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Invalid robot placement coordinates") from error
    robots = [r for r in layout.get("robots", []) if r.get("id") not in seen]
    states = scene.get("initial_object_states", [])
    if not isinstance(states, list):
        raise ValueError("initial_object_states must be a list")
    for state in states:
        if (not isinstance(state, dict) or not (state.get("name") or state.get("object_id"))
                or type(state.get("is_open")) is not bool):
            raise ValueError("Initial object states require name/object_id and boolean is_open")
    return dict(layout, objects=[*layout.get("objects", []), *placements],
                robots=[*robots, *robot_placements], initial_object_states=states)


def _configure_environment(scene: dict, policy_override: str | None) -> None:
    scene_name = scene.get("scene")
    agent_count = scene.get("agent_count")
    if not isinstance(scene_name, str) or not scene_name:
        raise ValueError("scene.json requires a non-empty 'scene' string")
    if type(agent_count) is not int or agent_count <= 0:
        raise ValueError("scene.json requires a positive integer 'agent_count'")

    policy = policy_override or scene.get("runtime_policy", "dag_aware")
    if policy not in {
        "dag_aware", "local_distance", "local_delay", "reroute_only"
    }:
        raise ValueError(f"Unsupported runtime policy: {policy!r}")
    planning_timeout = float(scene.get("planning_timeout_s", 30.0))
    if planning_timeout <= 0:
        raise ValueError("planning_timeout_s must be positive")

    # These must be set before importing the AI2-THOR adapter because its
    # central config intentionally freezes one reproducible run configuration.
    os.environ["AI2THOR_SCENE"] = scene_name
    os.environ["AI2THOR_AGENT_COUNT"] = str(agent_count)
    os.environ["AI2THOR_RUNTIME_POLICY"] = policy
    os.environ["AI2THOR_PLANNING_TIMEOUT"] = str(planning_timeout)
    interaction_positions = scene.get("interaction_positions", {})
    if not isinstance(interaction_positions, dict):
        raise ValueError("interaction_positions must be an object")
    for name, position in interaction_positions.items():
        if (
            not isinstance(name, str)
            or not isinstance(position, dict)
            or not all(axis in position for axis in ("x", "z"))
        ):
            raise ValueError(
                "Each interaction position requires a name and numeric x/z"
            )
        try:
            float(position["x"])
            float(position["z"])
            tolerance = float(position.get("tolerance", .03))
            if not math.isfinite(tolerance) or not 0 <= tolerance <= .05:
                raise ValueError("Required-pose tolerance must be between 0 and 0.05 m")
            if type(position.get("required", False)) is not bool:
                raise ValueError("required must be boolean")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid interaction position for {name!r}"
            ) from error
    os.environ["AI2THOR_INTERACTION_POSITIONS"] = json.dumps(
        interaction_positions,
        separators=(",", ":"),
    )
    distances = scene.get("interaction_distances", {})
    if not isinstance(distances, dict):
        raise ValueError("interaction_distances must be an object")
    try:
        if any(not math.isfinite(float(v)) or float(v) <= .04 for v in distances.values()):
            raise ValueError("Distances must be finite and greater than 0.04 m")
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid interaction_distances") from error
    os.environ["AI2THOR_INTERACTION_DISTANCES"] = json.dumps(distances)


def _validate_robot_names(plan: dict, agent_count: int) -> None:
    robot_plans = plan.get("robot_plans", {})
    allowed = {f"R{index}" for index in range(1, agent_count + 1)}
    unknown = sorted(set(robot_plans) - allowed)
    if unknown:
        raise ValueError(
            f"actions.json uses unavailable robots {unknown}; allowed={sorted(allowed)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run maps_ai2thor/<map>/scene.json and actions.json."
    )
    parser.add_argument(
        "map",
        nargs="?",
        default="map1",
        help="map name such as map1, or a directory containing both JSON files",
    )
    parser.add_argument(
        "--policy",
        choices=("dag_aware", "local_distance", "local_delay", "reroute_only"),
        help="override runtime_policy from scene.json",
    )
    parser.add_argument("--camera", action="store_true", help="show robot POV windows")
    parser.add_argument("--grid-view", action="store_true", help="show occupancy-grid view")
    parser.add_argument("--task-notes", action="store_true", help="show task captions (hidden by default)")
    parser.add_argument("--record-video", type=Path, help="save clean overhead mission video as MP4")
    parser.add_argument("--exit-after-execution", action="store_true", help="close GUI after mission completion")
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="show the scene and spawned robots without loading or running tasks",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate both JSON files without launching Unity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        directory = _map_directory(args.map)
        scene = _load_json(directory / "scene.json")
        layout_path = directory / "layout.json"
        layout = _load_json(layout_path) if layout_path.exists() else {}
        layout = _merge_scene_placements(scene, layout)
        _configure_environment(scene, args.policy)
        placements = [item for item in layout.get("objects", ()) if item.get("apply", False)]
        initial_states = layout.get("initial_object_states", [])
        os.environ["AI2THOR_LAYOUT_FINGERPRINT"] = (
            hashlib.sha256(json.dumps(
                {"placements": placements, "initial_states": initial_states} if initial_states else placements,
                sort_keys=True).encode()).hexdigest()[:16]
            if placements or initial_states else ""
        )

        if str(SRC_DIRECTORY) not in sys.path:
            sys.path.insert(0, str(SRC_DIRECTORY))
        plan = None
        commands = []
        if not args.view_only:
            from simulations.ai2thor.execution import build_action_sequence, load_plan

            plan = load_plan(directory / "actions.json")
            _validate_robot_names(plan, int(scene["agent_count"]))
            commands = build_action_sequence(plan)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Invalid AI2-THOR map: {error}") from None

    print(
        f"AI2-THOR map={directory.name}, scene={scene['scene']}, "
        f"agents={scene['agent_count']}, tasks={len(commands)}, "
        f"mode={'view-only' if args.view_only else 'execute'}, "
        f"policy={os.environ['AI2THOR_RUNTIME_POLICY']}"
    )
    if args.validate_only:
        print("Validation passed; Unity was not launched.")
        return

    try:
        from simulations.ai2thor import run_simulation
    except ModuleNotFoundError as error:
        if error.name == "ai2thor":
            raise SystemExit(
                "The ai2thor package is not installed in this Python environment."
            ) from None
        raise

    on_start = None
    if not args.view_only:
        def execute_saved_plan(navigation, actions, _distance):
            from simulations.ai2thor.execution import execute_plan

            return execute_plan(plan, navigation, actions)

        on_start = partial(execute_saved_plan)

    run_simulation(
        on_start,
        show_camera=args.camera or bool(scene.get("show_camera", False)),
        grid_view=args.grid_view or bool(scene.get("grid_view", False)),
        layout=layout,
        show_task_notes=args.task_notes,
        video_file=args.record_video,
        exit_after_execution=args.exit_after_execution,
    )


if __name__ == "__main__":
    main()
