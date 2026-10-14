# Grid 2D adapter

Run from the repository root:

```bash
python main.py map1 scene1
```

`world.py` loads maps and missions. `navigation.py` supplies assignment route distances. `app.py` owns live robot execution and safety gates, and binds its local execution-time evaluator to `core.coordination.coordinator`. `renderer.py` owns visualization and mouse interaction.

Search, swept geometry, reservations and remaining-DAG scoring are implemented in `core/coordination`, not duplicated here. The guarded prediction model is the default.
