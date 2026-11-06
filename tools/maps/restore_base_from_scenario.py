"""Restore base-map geometry with backups; use tools.maps module invocation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def restore(scenario_file: Path, *, keep_scenario_geometry=False) -> Path:
    scenario_file = scenario_file.expanduser().resolve()
    scenario = json.loads(scenario_file.read_text(encoding="utf-8"))
    if "free_spans" not in scenario:
        raise ValueError(f"{scenario_file.name} does not contain free_spans")

    base_name = scenario.get("base_world", "map.json")
    base_file = (scenario_file.parent / base_name).resolve()
    base = (
        json.loads(base_file.read_text(encoding="utf-8"))
        if base_file.is_file()
        else {}
    )

    base_backup = base_file.with_name(f".{base_file.name}.before_scenario_restore")
    scenario_backup = scenario_file.with_name(
        f".{scenario_file.name}.before_base_restore"
    )
    if base_file.is_file() and not base_backup.exists():
        shutil.copy2(base_file, base_backup)
    if not scenario_backup.exists():
        shutil.copy2(scenario_file, scenario_backup)

    base.update(
        {
            "description": f"Base navigation grid restored from {scenario_file.name}.",
            "free_spans": scenario["free_spans"],
            "free_regions": [],
            "blocked_regions": [],
            "objects": [],
            "robots": [],
            "tasks": [],
        }
    )
    if not keep_scenario_geometry:
        scenario.pop("free_spans")

    base_temporary = base_file.with_name(f".{base_file.name}.restore.tmp")
    base_temporary.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    os.replace(base_temporary, base_file)
    if not keep_scenario_geometry:
        scenario_temporary = scenario_file.with_name(
            f".{scenario_file.name}.restore.tmp"
        )
        scenario_temporary.write_text(
            json.dumps(scenario, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(scenario_temporary, scenario_file)
    return base_file


def drop_geometry_override(scenario_file: Path) -> None:
    scenario_file = scenario_file.expanduser().resolve()
    scenario = json.loads(scenario_file.read_text(encoding="utf-8"))
    if "free_spans" not in scenario:
        return
    scenario.pop("free_spans")
    temporary = scenario_file.with_name(f".{scenario_file.name}.drop-grid.tmp")
    temporary.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, scenario_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move scenario free_spans back into its base map.json"
    )
    parser.add_argument("scenario", type=Path)
    parser.add_argument(
        "--keep-scenario-geometry",
        action="store_true",
        help="keep free_spans in the source scenario after restoring map.json",
    )
    parser.add_argument(
        "--drop-override-from",
        type=Path,
        help="remove stale free_spans from another scenario so it inherits map.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    restored = restore(
        arguments.scenario,
        keep_scenario_geometry=arguments.keep_scenario_geometry,
    )
    if arguments.drop_override_from is not None:
        drop_geometry_override(arguments.drop_override_from)
    print(f"Restored base map: {restored}")
