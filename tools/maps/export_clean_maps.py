"""Export map geometry only: white free space, dark obstacles, no grid."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from simulations.standalone.world import load_world


def export_map(source, output):
    world = load_world(source)
    metadata = json.loads(source.read_text(encoding="utf-8"))["editor_grid"]
    min_x, min_z = metadata["min_x"], metadata["min_z"]
    width, height = metadata["width"], metadata["height"]
    border = 4  # Display-only enclosing wall; navigation geometry is unchanged.
    side = max(width, height) + 2 * border
    outer_x = min_x - border - (max(width, height) - width) / 2
    outer_z = min_z - border - (max(width, height) - height) / 2
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_patch(Rectangle((outer_x, outer_z), side, side, color="#202020", linewidth=0))
    # Merge adjacent white cells into row rectangles; PDF/SVG remain vector.
    for z in range(min_z, min_z + height):
        xs = sorted(x for x, cell_z in world.free_cells
                    if cell_z == z and min_x <= x < min_x + width)
        if not xs:
            continue
        start = previous = xs[0]
        for x in xs[1:] + [min_x + width + 1]:
            if x != previous + 1:
                ax.add_patch(Rectangle((start, z), previous - start + 1, 1,
                                      facecolor="white", edgecolor="none", antialiased=False))
                start = x
            previous = x
    ax.set_xlim(outer_x, outer_x + side)
    ax.set_ylim(outer_z, outer_z + side)
    ax.set_aspect("equal")
    ax.set_axis_off()
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        target = output / f"{source.parent.name}_clean.{extension}"
        fig.savefig(target, dpi=600, facecolor="white", pad_inches=0)
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "maps" / "figures")
    args = parser.parse_args()
    for index in range(1, 5):
        export_map(ROOT / "maps" / f"map{index}" / "map.json", args.output)
