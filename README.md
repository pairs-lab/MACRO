# MARCO: Mission-Aware Runtime Coordination over Task Graphs

Multi-robot task execution with a task DAG and online conflict resolution, in a 2D grid simulator and in AI2-THOR.

## Install

Python 3.10 is recommended.

```bash
git clone https://github.com/pairs-lab/MACRO.git
cd MACRO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # 2D grid simulator
pip install -r requirements-ai2thor.txt    # AI2-THOR (optional)
```

Optional C++ acceleration (needs a C++17 compiler):

```bash
pip install setuptools wheel pybind11
pip install --no-build-isolation ./native/chain_bnb ./native/grid_geometry
```

## 2D grid scenarios

Maps are in `maps/` (`map1` to `map4`), each with scenarios `scene1` to `scene5`.

```bash
python main.py map1 scene1
python main.py map3 scene2
python main.py map1 scene1 --map-only     # show the map only
python main.py --help
```

## AI2-THOR scenarios

Maps are in `maps_ai2thor/` (`map1` to `map6`). AI2-THOR needs a desktop with a display; Unity is downloaded on the first run.

```bash
python run_ai2thor.py map1
python run_ai2thor.py map2 --grid-view    # also show the occupancy grid
python run_ai2thor.py map1 --view-only    # scene and robots only, no tasks
python run_ai2thor.py --help
```

## Layout

| Directory | Contents |
| --- | --- |
| `src/core/` | Task assignment, DAG scheduling, coordination |
| `src/simulations/standalone/` | 2D grid simulator and conflict resolution |
| `src/simulations/ai2thor/` | AI2-THOR environment |
| `maps/`, `maps_ai2thor/` | Scenarios |
| `tools/maps/` | Map editing tools |
| `native/` | Optional C++ modules |
