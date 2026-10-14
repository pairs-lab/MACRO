"""Create the planner's occupancy input from the same saved map used by AMCL."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def prepare_map(map_yaml, output):
    config = yaml.safe_load(map_yaml.read_text())
    image = cv2.imread(str(map_yaml.parent / config["image"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read occupancy image for {map_yaml}")
    if int(config.get("negate", 0)) != 0:
        raise ValueError("This saved-map adapter supports non-negated maps only")
    probability = 1. - np.flipud(image).astype(float) / 255.
    occupancy = np.full(image.shape, -1, dtype=int)
    occupancy[probability < float(config["free_thresh"])] = 0
    occupancy[probability > float(config["occupied_thresh"])] = 100
    grid = dict(width=int(image.shape[1]), height=int(image.shape[0]),
                resolution=float(config["resolution"]), origin=config["origin"][:2],
                data=occupancy.ravel().tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"map": grid}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_yaml", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    prepare_map(arguments.map_yaml, arguments.output)
