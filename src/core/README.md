# Shared algorithms

This package must not import a simulator, GUI, Unity, Nav2 or ROS middleware.

- `assignment.py`, `assignment_cpp.py`, `assignment_milp.py`: task assignment and scheduling backends.
- `models.py`: robot snapshot/state schema.
- `coordination/coordinator.py`: snapshot-based conflict candidate pipeline. The execution adapter supplies local resolution timing and may supply candidate-conditioned execution evaluation.
- `coordination/planner.py`: A*, shortest-path trees and scoped geometry reuse.
- `coordination/relocation_search.py`: bounded intervention search, candidate scoring and incremental remaining-DAG evaluation.
- `coordination/dynamic_collision.py`, `clearance_geometry.py`, `native_geometry.py`: swept geometry and optional C++ kernels.
- `coordination/release_events.py`, `anticipated_delay.py`, `dynamic_release.py`: guarded release and future reservation prediction.
- `coordination/task_timing.py`: callback-based task timing for interaction adapters.
- `coordination/interaction_search.py`: callback-based interaction repair search, without Unity imports.

Environment-specific occupancy conversion, interaction goals, controller adaptation and live execution belong in `simulations/`. A structural refactor must preserve search budgets, candidate ordering, timing inputs, tie-breaking and safety guards.
