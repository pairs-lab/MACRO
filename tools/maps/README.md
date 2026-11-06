# Map tools

Run these commands from the project root. The editors require a graphical desktop.

| Tool | Command | Purpose |
| --- | --- | --- |
| Create | `python -m tools.maps.create_map maps/map_new` | Draw a new grid map. |
| Edit | `python -m tools.maps.change_map map1 scene1` | Edit scenario robots, objects and geometry overrides. |
| View | `python -m tools.maps.view_map maps/map1/scenario_1.json` | Display a scenario without running its mission. |
| Export grid images | `python -m tools.maps.export_clean_maps --output maps/figures` | Save map-only images. |
| Export AI2-THOR layout | `python -m tools.maps.export_ai2thor_layout map6` | Export scene geometry using Unity. |
| Restore base geometry | `python -m tools.maps.restore_base_from_scenario maps/map1/scenario_1.json` | Restore a base map from scenario geometry. |

Use `--help` for each tool's arguments. Creating or restoring maps writes JSON files; use a new map name to protect existing maps. The restore tool creates backups. AI2-THOR export requires the optional simulator dependencies and a working Unity environment.

Grid maps live in `maps/<name>/map.json`; missions live in `scenario_<number>.json`. Geometry, robot positions and object positions are separate from task definitions. Editing positions does not automatically update the mission DAG: update the scenario's `tasks` explicitly when changing the mission.

The tools do not modify the assignment or conflict-resolution algorithm.
