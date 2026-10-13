"""Keyboard/mouse interaction loop for AI2-THOR."""

import select
import sys
import termios
import time
import tty

import cv2

from .cameras import move_robot, update_overhead_camera
from .config import (
    ACTION_RATE_HZ,
    AGENT_ACTIONS,
    DISPLAY_RATE_HZ,
    MANUAL_MOVE_STEP,
    MANUAL_ROTATION_STEP,
    AGENT_COUNT,
    ITHOR_SCENE,
    ROBOT_CAMERA_COUNT,
)
from .display import show_camera_frames
from .objects import get_addable_object_types


def print_controls(event, show_robot_cameras=False):
    addable = get_addable_object_types(event)
    print("AI2-THOR started!")
    print(f"Scene: iTHOR {ITHOR_SCENE}; real agents: {AGENT_COUNT}")
    print("Fixed top-down camera opened; the roof is hidden.")
    if show_robot_cameras:
        print(f"Opened {ROBOT_CAMERA_COUNT} independent robot POV cameras.")
    else:
        print("Robot POV cameras hidden; use --camera to show them.")
    robot_names = ", ".join(
        f"robot{agent_id + 1}" for agent_id in range(AGENT_COUNT)
    )
    print(f"Robots: {robot_names} (real AI2-THOR agents).")
    print("Addable object types:", ", ".join(addable) or "None")
    print(
        "Camera: scroll = zoom, left-drag = pan, "
        "middle-drag = rotate, right-click = reset."
    )
    print("Terminal: hold W/S = move, hold A/D = rotate, Q = quit.")


def read_pending_keys(timeout=0.02):
    """Read and coalesce terminal key-repeat events."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return []

    keys = [sys.stdin.read(1).lower()]
    while select.select([sys.stdin], [], [], 0)[0]:
        keys.append(sys.stdin.read(1).lower())
    return keys


def run_interaction_loop(
    controller,
    overhead_camera,
    robot_camera,
    navigation_path=None,
    show_robot_cameras=False,
    keyframe_note_provider=None,
):
    """Run mouse events, terminal controls, and camera refreshes."""
    stdin_fd = sys.stdin.fileno()
    original_terminal_mode = termios.tcgetattr(stdin_fd)
    last_action_time = 0.0
    last_display_time = time.monotonic()

    try:
        tty.setcbreak(stdin_fd)
        while True:
            cv2.waitKey(1)

            if overhead_camera.needs_update:
                event = update_overhead_camera(controller, overhead_camera)
                show_camera_frames(
                    event,
                    overhead_camera,
                    navigation_path,
                    show_robot_cameras,
                    (
                        keyframe_note_provider()
                        if keyframe_note_provider is not None
                        else None
                    ),
                )
                overhead_camera.needs_redraw = False
            elif overhead_camera.needs_redraw:
                show_camera_frames(
                    controller.last_event,
                    overhead_camera,
                    navigation_path,
                    show_robot_cameras,
                    (
                        keyframe_note_provider()
                        if keyframe_note_provider is not None
                        else None
                    ),
                )
                overhead_camera.needs_redraw = False

            keys = read_pending_keys()
            if not keys:
                continue
            if "q" in keys:
                break

            action = AGENT_ACTIONS.get(keys[-1])
            if action is None:
                continue

            now = time.monotonic()
            if now - last_action_time < 1 / ACTION_RATE_HZ:
                continue
            last_action_time = now

            action_parameters = {}
            if action in ("MoveAhead", "MoveBack"):
                action_parameters["moveMagnitude"] = MANUAL_MOVE_STEP
            elif action in ("RotateLeft", "RotateRight"):
                action_parameters["degrees"] = MANUAL_ROTATION_STEP

            event = move_robot(
                controller,
                robot_camera,
                action,
                **action_parameters,
            )
            if event is None:
                continue

            if time.monotonic() - last_display_time >= 1 / DISPLAY_RATE_HZ:
                show_camera_frames(
                    event,
                    overhead_camera,
                    navigation_path,
                    show_robot_cameras,
                    (
                        keyframe_note_provider()
                        if keyframe_note_provider is not None
                        else None
                    ),
                )
                last_display_time = time.monotonic()
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, original_terminal_mode)
