"""Convert measured occupancy grids into the shared planning input schema."""

import hashlib
import json
import math


def clean(value):
    """Convert planning dictionaries to strict JSON-compatible values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


class PlannerSession:
    """Cache static map geometry, never live robot states or interventions."""

    def __init__(self):
        self.key = None
        self.spans = None
        self.world = None
        self.audit = None

    def map_data(self, report, radius):
        raw = (report / "scan_audit.json").read_bytes()
        key = (hashlib.sha256(raw).hexdigest(), radius)
        if key != self.key:
            audit = json.loads(raw)
            grid = audit["map"]
            resolution = grid["resolution"]
            ox, oy = grid["origin"][:2]
            free = {
                (round((ox + (index % grid["width"] + .5) * resolution) / resolution),
                 round((oy + (index // grid["width"] + .5) * resolution) / resolution))
                for index, value in enumerate(grid["data"]) if value == 0
            }
            spans = {}
            for x, y in sorted(free):
                spans.setdefault(str(y), []).append([x, x])
            self.spans = spans
            self.world = None
            self.key = key
            self.audit = audit
        return self.audit, self.spans
