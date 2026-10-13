# MARCO: DAG-Aware Multi-Robot Coordination

A grid-based multi-robot simulator with task-DAG assignment and online conflict resolution. A separate AI2-THOR adapter supports interactive scene execution.

## Install

Python 3.10 or newer is recommended. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For optional C++ acceleration, install a C++17 compiler and build the native modules:

```bash
python -m pip install setuptools wheel pybind11
python -m pip install --no-build-isolation ./native/chain_bnb
python -m pip install --no-build-isolation ./native/grid_geometry
```

## Run a grid mission

```bash
python main.py map1 scene1 --policy dag_aware --release-prediction-model dynamic_events_guarded
```

This explicitly selects the guarded prediction configuration; existing command-line defaults remain unchanged. Robots execute their assigned tasks concurrently, respecting dependencies. When paths conflict, the coordinator generates bounded interventions, predicts their timing effects, and evaluates the remaining mission DAG. Online repairs are not globally optimal plans.

View a scenario or edit its base map:

```bash
python main.py map1 scene1 --map-only
python main.py map1 scene1 --edit
python main.py --help
```

For creating maps and changing robot/object positions, see [Map tools](tools/maps/README.md).

## AI2-THOR (optional)

Requires a graphical environment and a working Unity installation/runtime.

```bash
python -m pip install -r requirements-ai2thor.txt
python run_ai2thor.py map1 --validate-only
python run_ai2thor.py map6 --view-only
python run_ai2thor.py --help
```

This adapter uses simulator-specific navigation and execution; it is separate from the grid simulator. Optional language-model features require a locally configured API key. Never commit credentials.

## Project layout

| Directory | Contents |
| --- | --- |
| `src/core/` | Assignment, DAG scheduling and coordination primitives. |
| `src/simulations/standalone/` | Grid navigation, conflict repair and execution. |
| `src/simulations/ai2thor/` | AI2-THOR adapter. |
| `tools/maps/` | Map creation, editing and export utilities. |
| `maps/`, `maps_ai2thor/` | Example maps and mission configurations. |
| `native/` | Optional C++ acceleration modules. |
| `tests/` | Core regression tests. |

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Optional native dependencies must be built to exercise their tests. Generated outputs, local datasets and credentials are excluded by `.gitignore`.

## Source-only export

```bash
python scripts/create_source_archive.py dist/marco-source.tar.gz
```

The archive contains the publishable source tree, without Git history or local research artifacts. Review its contents before distribution.
