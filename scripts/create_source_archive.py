#!/usr/bin/env python3
"""Create a source archive while excluding credentials and generated data."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".lic"}


def _excluded(path: Path) -> bool:
    lower_name = path.name.casefold()
    parts = {part.casefold() for part in path.parts}
    return (
        path.parts[0] in {"benchmarks", "experiments", "docs", ".agents", ".codex", "results", "videos"}
        or path.parts[:2] == ("tests", "local")
        or path.as_posix() in {"plot.py", "run.sh"}
        or "__pycache__" in parts
        or ".pytest_cache" in parts
        or ".navigation_cache" in parts
        or lower_name == "api_key.txt"
        or lower_name == ".env"
        or lower_name.startswith(".env.")
        or "credential" in lower_name
        or "secret" in lower_name
        or path.suffix.casefold() in SENSITIVE_SUFFIXES
    )


def source_paths() -> list[Path]:
    """Return tracked/unignored source paths that pass the secret denylist."""
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = [
        Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    ]
    return sorted({path for path in paths if not _excluded(path)
                   and (ROOT / path).is_file() and not (ROOT / path).is_symlink()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List safe inputs without creating an archive.",
    )
    args = parser.parse_args()
    paths = source_paths()
    if args.list:
        for path in paths:
            print(path)
        return

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative in paths:
            unresolved = ROOT / relative
            source = unresolved.resolve()
            if (
                unresolved.is_symlink()
                or source == output
                or not source.is_file()
                or ROOT not in source.parents
            ):
                continue
            archive.add(source, arcname=Path(ROOT.name) / relative)
    print(f"Created credential-safe source archive: {output}")


if __name__ == "__main__":
    main()
