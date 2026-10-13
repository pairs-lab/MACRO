"""Public API for the AI2-THOR simulation adapter."""


def run_simulation(
    on_start=None,
    *,
    show_camera=False,
    grid_view=False,
    layout=None,
    show_task_notes=False,
    video_file=None,
    exit_after_execution=False,
):
    """Import the application lazily to avoid package import cycles."""
    from .app import run_simulation as start_simulation

    return start_simulation(
        on_start=on_start,
        show_camera=show_camera,
        grid_view=grid_view,
        layout=layout,
        show_task_notes=show_task_notes,
        video_file=video_file,
        exit_after_execution=exit_after_execution,
    )


__all__ = ["run_simulation"]
