"""Canonical map/scene tag resolution shared by the command-line tools."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPS_DIRECTORY = PROJECT_ROOT / "maps"


def normalize_map_tag(value: str) -> str:
    tag = str(value).strip()
    match = re.fullmatch(r"map_?(\d+)", tag, flags=re.IGNORECASE)
    if match:
        return f"map{match.group(1)}"
    if tag.isdigit():
        return f"map{tag}"
    return tag


def normalize_scene_filename(value: str) -> str:
    name = Path(str(value)).name
    if name.endswith(".json"):
        name = name[:-5]
    match = re.fullmatch(
        r"(?:scene|scenario)_?(.*)",
        name,
        flags=re.IGNORECASE,
    )
    tag = match.group(1) if match else name
    if not tag:
        raise ValueError("Scene tag cannot be empty")
    return f"scenario_{tag}.json"


def map_directory(value) -> Path:
    raw = Path(value).expanduser()
    if raw.suffix == ".json":
        raw = raw.parent
    if raw.is_absolute():
        return raw.resolve()
    if len(raw.parts) == 1:
        return (MAPS_DIRECTORY / normalize_map_tag(raw.name)).resolve()
    return (PROJECT_ROOT / raw).resolve()


def scene_path(map_value, scene_value) -> Path:
    return map_directory(map_value) / normalize_scene_filename(str(scene_value))
