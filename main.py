"""Run the standalone 2D multi-robot simulation."""

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from simulations.standalone import run_standalone_simulation
from tools.maps.map_tags import scene_path


MAP_DIRECTORY = PROJECT_ROOT / "maps"
DEFAULT_MAP = "map1"


def scenario_in_directory(directory):
    canonical = directory / "scenario.json"
    if canonical.is_file():
        return canonical
    numbered_default = directory / "scenario_1.json"
    if numbered_default.is_file():
        return numbered_default
    alternatives = sorted(directory.glob("scenario*.json"))
    if len(alternatives) == 1:
        return alternatives[0]
    if len(alternatives) > 1:
        choices = ", ".join(path.name for path in alternatives)
        raise SystemExit(
            f"Multiple scenarios in {directory}: {choices}. "
            "Use --world with the exact scenario path."
        )
    return None


def resolve_map(name):
    """Resolve ``maps/mapN/scenario.json`` from a short map name."""
    requested = Path(name)
    direct = requested.expanduser()
    if direct.is_dir():
        scenario = scenario_in_directory(direct)
        if scenario is not None:
            return scenario.resolve()
    if direct.is_file():
        return direct.resolve()

    directory_names = [requested.name]
    if requested.name.startswith("map_"):
        directory_names.append("map" + requested.name[4:])
    elif requested.name.startswith("map"):
        suffix = requested.name[3:]
        if suffix:
            directory_names.append("map_" + suffix)
    for directory_name in directory_names:
        directory = MAP_DIRECTORY / directory_name
        scenario = scenario_in_directory(directory)
        if scenario is not None:
            return scenario

    # Backward compatibility for old single-file maps.
    if requested.suffix == ".json":
        exact = MAP_DIRECTORY / requested.name
        if exact.is_file():
            return exact
    matches = sorted(MAP_DIRECTORY.glob(f"{requested.name}*.json"))
    if not matches:
        available = ", ".join(
            path.parent.name
            for path in sorted(MAP_DIRECTORY.glob("*/scenario*.json"))
        )
        raise SystemExit(
            f"Unknown map {name!r}. Available maps: {available or 'none'}"
        )
    if len(matches) > 1:
        choices = ", ".join(path.stem for path in matches)
        raise SystemExit(f"Ambiguous map {name!r}; choose one of: {choices}")
    return matches[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the AI2-THOR-independent 2D core simulation."
    )
    parser.add_argument(
        "map_tag",
        nargs="?",
        help="short map tag, for example map1",
    )
    parser.add_argument(
        "scene_tag",
        nargs="?",
        help="short scene tag, for example scene1",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--map",
        default=None,
        help=(
            "map folder in maps/; for example map1 or map2 (default: map1)"
        ),
    )
    source.add_argument(
        "--world",
        type=Path,
        help="explicit JSON path (kept for backward compatibility)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--edit",
        action="store_true",
        help="open the selected map folder in the base-map editor",
    )
    mode.add_argument(
        "--map-only",
        action="store_true",
        help="show only the simulation map grid without robots or objects",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=8.0,
        help="robot speed in grid units per second; diagonal edges cost sqrt(2) units (default: 8)",
    )
    parser.add_argument("--policy", default="dag_aware", choices=(
        "dag_aware", "local_distance", "local_delay", "reroute_only", "downstream_priority"))
    parser.add_argument("--planner-workers", type=int, default=None,
                        help="CPU processes for independent moving conflict groups (default: robot count)")
    parser.add_argument("--hypothesis-workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="CPU processes for candidate scoring inside each conflict group")
    parser.add_argument("--release-prediction-model", choices=("events", "dynamic_events_guarded"),
                        default="dynamic_events_guarded", help="release prediction model (default: the paper method)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")
    if args.planner_workers is not None and args.planner_workers < 1:
        raise SystemExit("--planner-workers must be positive")
    if args.hypothesis_workers < 1:
        raise SystemExit("--hypothesis-workers must be positive")
    if args.world is not None and (args.map_tag or args.scene_tag):
        raise SystemExit("Do not combine positional tags with --world")
    if args.world is not None:
        world = args.world
    else:
        selected_map = args.map_tag or args.map or DEFAULT_MAP
        if args.scene_tag is not None:
            world = scene_path(selected_map, args.scene_tag)
            if not world.is_file():
                raise SystemExit(f"Unknown scene: {world}")
        else:
            world = resolve_map(selected_map)
    if args.edit:
        from tools.maps.create_map import GridMapCreator

        world_path = Path(world).expanduser().resolve()
        map_directory = (
            world_path.parent
            if world_path.suffix == ".json"
            else world_path
        )
        GridMapCreator(map_directory).run()
        return
    if args.map_only:
        from tools.maps.view_map import MapViewer

        MapViewer(world).run()
        return
    run_standalone_simulation(world, speed=args.speed, selection_policy=args.policy,
                             planner_workers=args.planner_workers,
                             hypothesis_workers=args.hypothesis_workers,
                             release_prediction_model=args.release_prediction_model)


if __name__ == "__main__":
    main()
