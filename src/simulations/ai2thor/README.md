# AI2-THOR adapter

```bash
python run_ai2thor.py map1 --validate-only
python run_ai2thor.py map6 --view-only
```

Install `requirements-ai2thor.txt` and provide a working graphical Unity runtime.

`controller.py` and `app.py` own simulator lifecycle; `navigation/` handles Unity poses, visibility and movement; `execution.py` dispatches tasks. `replanning/` contains Unity-specific geometry and live scheduling adapters. Callback-based repair search and DAG timing live in `core.coordination.interaction_search` and `core.coordination.task_timing`.

This adapter retains its existing Unity interaction semantics. It is not the grid executor running unchanged inside Unity.
