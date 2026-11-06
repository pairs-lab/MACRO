"""Export editable robot and object coordinates from one AI2-THOR map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_DIRECTORY = PROJECT_ROOT / "maps_ai2thor"


def distance_xz(first, second):
    return math.hypot(first["x"] - second["x"], first["z"] - second["z"])


def default_robot_positions(reachable, count):
    center = {
        "x": sum(point["x"] for point in reachable) / len(reachable),
        "z": sum(point["z"] for point in reachable) / len(reachable),
    }
    selected = [max(reachable, key=lambda point: distance_xz(point, center))]
    while len(selected) < count:
        selected.append(
            max(
                reachable,
                key=lambda point: min(
                    distance_xz(point, chosen) for chosen in selected
                ),
            )
        )
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export maps_ai2thor/<map>/layout.json from Unity metadata."
    )
    parser.add_argument("map", help="map name such as map2")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing layout.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    directory = (MAP_DIRECTORY / args.map).resolve()
    scene_path = directory / "scene.json"
    layout_path = directory / "layout.json"
    if layout_path.exists() and not args.force:
        raise SystemExit(
            f"{layout_path} already exists; use --force to overwrite edits"
        )
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    try:
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering
    except ModuleNotFoundError:
        raise SystemExit("Run this script inside the ai2thor environment") from None

    controller = Controller(
        scene=scene["scene"],
        platform=CloudRendering,
        quality="Low",
        width=640,
        height=480,
        gridSize=0.05,
        snapToGrid=False,
        agentRadius=0.05,
        agentCount=int(scene["agent_count"]),
        makeAgentsVisible=True,
        headless=True,
    )
    try:
        reachable_event = controller.step(
            action="GetReachablePositions", agentId=0, renderImage=False
        )
        if not reachable_event.metadata["lastActionSuccess"]:
            raise RuntimeError(reachable_event.metadata.get("errorMessage"))
        robot_positions = default_robot_positions(
            reachable_event.metadata["actionReturn"],
            int(scene["agent_count"]),
        )

        objects = sorted(
            controller.last_event.metadata.get("objects", []),
            key=lambda obj: (obj.get("objectType", ""), obj.get("objectId", "")),
        )
        instance_counts = {}
        object_rows = []
        for obj in objects:
            name = obj.get("objectType", "")
            instance = instance_counts.get(name, 0)
            instance_counts[name] = instance + 1
            position = obj.get("position", {})
            pickupable = bool(obj.get("pickupable", False))
            object_rows.append({
                "name": name,
                "instance": instance,
                "object_id": obj.get("objectId"),
                "position": {
                    axis: round(float(position.get(axis, 0.0)), 4)
                    for axis in ("x", "y", "z")
                },
                "pickupable": pickupable,
                "receptacle": bool(obj.get("receptacle", False)),
                "openable": bool(obj.get("openable", False)),
                "apply": False,
                "force_kinematic": True,
            })

        payload = {
            "description": (
                "Edit robot x/z directly. For a pickupable object, edit "
                "x/y/z and set apply=true. Fixed objects are listed for reference."
            ),
            "scene": scene["scene"],
            "robots": [
                {
                    "id": f"R{index + 1}",
                    "position": {
                        "x": round(float(position["x"]), 4),
                        "z": round(float(position["z"]), 4),
                    },
                    "rotation": (index * 120) % 360,
                    "apply": True,
                }
                for index, position in enumerate(robot_positions)
            ],
            "objects": object_rows,
        }
        layout_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved {layout_path}")
        print(
            f"Robots: {len(payload['robots'])}; "
            f"scene objects: {len(payload['objects'])}"
        )
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
