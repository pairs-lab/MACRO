# T-junction grid demo

```bash
python main.py map5 scene1 --speed 1 --planner-workers 1
python main.py map5 scene2 --speed 1 --planner-workers 1
```

R1 (blue) and R2 (orange) start on opposite sides and exchange destinations. R3 (green) has no service task and initially occupies the southern passing branch. The one-cell-wide aisle cannot support side-by-side passing: cell width is 0.50 m and each robot has radius 0.23 m.

The mission uses the default guarded MARCO evaluator, swept-step detection and ordinary runtime repairs. Priority and relocations are selected by the algorithm, not scripted. The `open` tasks denote station service (8 s for R1, 1 s for R2); no Unity object interaction is involved. R3 moves only when needed as a clearance participant.

Scene 2 swaps the service durations (R1: 1 s, R2: 8 s), while keeping geometry, starts and goals unchanged.

At speed 1 grid unit/s, physical travel speed is 0.50 m/s. Press Space to pause, R to restart, P to toggle paths, or Q/Esc to close. The task DAG and robot states appear alongside the map.
