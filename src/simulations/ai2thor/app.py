"""Top-level orchestration for the AI2-THOR simulation adapter."""

import cv2
import threading
from queue import Empty, Full, Queue

from .actions import create_action_function
from .cameras import setup_cameras
from .controller import create_controller, load_scene
from .controls import print_controls, run_interaction_loop
from .display import (
    close_camera_windows,
    create_camera_windows,
    refresh_camera_frames,
    get_third_party_frames,
    to_bgr,
)
from .config import (
    OVERHEAD_WINDOW,
    SHOW_NAVIGATION_PATH,
)
from .grid_display import GridMapRenderer
from .navigation.api import create_navigation_function
from .navigation.distance import create_distance_function
from .navigation.grid_planner import prepare_global_grid_map


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
    """Run with top view always visible and optional robot POV windows."""
    scene = load_scene()
    layout = layout or {}
    from .video import MissionVideoRecorder
    recorder = MissionVideoRecorder(video_file) if video_file is not None else None
    controller = create_controller(
        scene,
        robot_placements=[
            entry for entry in layout.get("robots", ())
            if entry.get("apply", True)
        ],
        object_placements=[
            entry for entry in layout.get("objects", ())
            if entry.get("apply", False)
        ],
        initial_object_states=layout.get("initial_object_states", []),
    )

    try:
        prepare_global_grid_map(controller)
        overhead_camera, robot_camera, event = setup_cameras(controller)
        create_camera_windows(
            overhead_camera,
            show_robot_cameras=show_camera,
        )
        grid_renderer = GridMapRenderer(controller) if grid_view else None
        if grid_renderer is not None:
            def grid_mouse(event, x, y, flags, parameter):
                grid_renderer.on_mouse(event, x, y, flags, parameter)
                overhead_camera.needs_redraw = True

            cv2.setMouseCallback(OVERHEAD_WINDOW, grid_mouse)
        navigation_paths = {}
        navigation_paths_lock = threading.RLock()
        keyframe_note = {"text": "Simulation initialized"}

        def raw_refresh(event):
            with navigation_paths_lock:
                visible_paths = {
                    agent_id: [point.copy() for point in path]
                    for agent_id, path in navigation_paths.items()
                }
                visible_note = keyframe_note["text"] if show_task_notes else None
            if recorder is not None:
                frames = get_third_party_frames(event)
                if frames:
                    recorder.update(to_bgr(frames[0]))
            if grid_renderer is None:
                refresh_camera_frames(
                    event,
                    overhead_camera=overhead_camera,
                    navigation_path=visible_paths,
                    show_robot_cameras=show_camera,
                    keyframe_note=visible_note,
                )
            else:
                cv2.imshow(
                    OVERHEAD_WINDOW,
                    grid_renderer.render(
                        event,
                        visible_paths,
                        keyframe_note=visible_note,
                    ),
                )
                cv2.waitKey(1)
        main_thread = threading.current_thread()
        frame_queue = Queue(maxsize=1)

        def refresh(event):
            if threading.current_thread() is main_thread:
                raw_refresh(event)
                return
            try:
                frame_queue.put_nowait(event)
            except Full:
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
                frame_queue.put_nowait(event)

        def pump_display():
            latest_event = None
            while True:
                try:
                    latest_event = frame_queue.get_nowait()
                except Empty:
                    break
            if latest_event is not None:
                raw_refresh(latest_event)
            elif recorder is not None:
                recorder.update()

        refresh(event)

        def show_planned_path(agent_id, path):
            with navigation_paths_lock:
                navigation_paths[agent_id] = [
                    point.copy() for point in path
                ]

        def set_keyframe_note(text):
            with navigation_paths_lock:
                keyframe_note["text"] = str(text)

        if on_start is None:
            event = controller.last_event
        else:
            navigation = create_navigation_function(
                controller,
                robot_camera,
                on_step=refresh,
                on_path_planned=(
                    show_planned_path
                    if SHOW_NAVIGATION_PATH
                    else None
                ),
            )
            navigation.pump_events = pump_display
            navigation.set_keyframe = set_keyframe_note
            actions = create_action_function(
                controller,
                robot_camera,
                on_step=refresh,
            )
            distance = create_distance_function(controller)
            on_start(navigation, actions, distance)
            refresh(controller.last_event)
            event = controller.last_event

        if recorder is not None:
            recorder.close()
        if exit_after_execution:
            return

        print_controls(event, show_robot_cameras=show_camera)
        run_interaction_loop(
            controller,
            overhead_camera,
            robot_camera,
            navigation_paths,
            show_robot_cameras=show_camera,
            keyframe_note_provider=(lambda: keyframe_note["text"]) if show_task_notes else None,
        )
    finally:
        if recorder is not None:
            recorder.close()
        close_camera_windows()
        controller.stop()
