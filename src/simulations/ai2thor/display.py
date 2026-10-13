"""AI2-THOR OpenCV windows, frames, and navigation overlays."""

import math
import os

os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2

cv2.setUseOptimized(True)

from .config import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    OVERHEAD_WINDOW,
    ROBOT_CAMERA_COUNT,
    ROBOT_DISPLAY_HEIGHT,
    ROBOT_DISPLAY_WIDTH,
    ROBOT_WINDOW_PREFIX,
)


_LAST_ROBOT_FRAMES = {}
_LAST_OVERHEAD_FRAME = None
_PATH_COLORS = (
    (0, 255, 255),   # R1: yellow
    (255, 120, 0),   # R2: blue
    (255, 0, 255),   # R3: magenta
    (0, 220, 0),     # additional robots: green, then cycle
)
def draw_keyframe_note(frame, text):
    """Draw the current simulation keyframe caption in the upper-left."""
    if not text:
        return frame
    origin = (18, 31)
    # A thin dark outline keeps the requested white text readable over both
    # the RGB scene and the light occupancy-grid background.
    cv2.putText(
        frame,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_hover_coordinate(frame, overhead_camera):
    """Show the world-space x/z coordinate below the mouse cursor."""
    if overhead_camera is None or overhead_camera.cursor_world is None:
        return frame
    x, y = overhead_camera.cursor_pixel
    label = (
        f"x={overhead_camera.cursor_world['x']:.2f}, "
        f"z={overhead_camera.cursor_world['z']:.2f}"
    )
    text_size, _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
    )
    label_x = max(8, min(x + 14, frame.shape[1] - text_size[0] - 12))
    label_y = max(26, min(y - 12, frame.shape[0] - 8))
    cv2.rectangle(
        frame,
        (label_x - 6, label_y - text_size[1] - 6),
        (label_x + text_size[0] + 6, label_y + 6),
        (25, 25, 25),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def robot_window_name(agent_id):
    return f"{ROBOT_WINDOW_PREFIX} {agent_id + 1} camera"


def create_camera_windows(overhead_camera, show_robot_cameras=False):
    """Create the top-view window and optionally the robot POV windows."""
    window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(OVERHEAD_WINDOW, window_flags)
    cv2.resizeWindow(OVERHEAD_WINDOW, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.setMouseCallback(OVERHEAD_WINDOW, overhead_camera.on_mouse)

    if not show_robot_cameras:
        return

    for agent_id in range(ROBOT_CAMERA_COUNT):
        window_name = robot_window_name(agent_id)
        cv2.namedWindow(window_name, window_flags)
        cv2.resizeWindow(
            window_name,
            ROBOT_DISPLAY_WIDTH,
            ROBOT_DISPLAY_HEIGHT,
        )


def get_third_party_frames(event):
    """Support both Event and MultiAgentEvent camera frame layouts."""
    frames = event.third_party_camera_frames
    if not frames:
        frames = event.events[0].third_party_camera_frames
    return frames


def to_bgr(frame):
    """Convert AI2-THOR RGB/RGBA image frames for OpenCV display."""
    conversion = cv2.COLOR_RGBA2BGR if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR
    return cv2.cvtColor(frame, conversion)


def _world_to_overhead_pixel(point, overhead_camera, frame_shape):
    """Project a world point into an orthographic overhead-camera frame."""
    height, width = frame_shape[:2]
    yaw = math.radians(overhead_camera.rotation["y"])
    pitch = math.radians(overhead_camera.rotation["x"])
    delta = {
        axis: point[axis] - overhead_camera.position[axis]
        for axis in ("x", "y", "z")
    }

    right = (math.cos(yaw), 0.0, -math.sin(yaw))
    up = (
        math.sin(yaw) * math.sin(pitch),
        math.cos(pitch),
        math.cos(yaw) * math.sin(pitch),
    )
    horizontal = sum(delta[axis] * right[index] for index, axis in enumerate(("x", "y", "z")))
    vertical = sum(delta[axis] * up[index] for index, axis in enumerate(("x", "y", "z")))

    half_height = overhead_camera.orthographic_size
    half_width = half_height * width / height
    pixel_x = int(round((horizontal / half_width + 1.0) * width / 2.0))
    pixel_y = int(round((1.0 - vertical / half_height) * height / 2.0))
    return pixel_x, pixel_y


def draw_navigation_path(
    frame,
    path,
    overhead_camera,
    color=(0, 255, 255),
    label=None,
):
    """Draw the complete planned path before and during robot movement."""
    if overhead_camera is None or not path:
        return frame

    pixels = [
        _world_to_overhead_pixel(point, overhead_camera, frame.shape)
        for point in path
    ]
    for first, second in zip(pixels, pixels[1:]):
        cv2.line(frame, first, second, (20, 20, 20), 8, cv2.LINE_AA)
        cv2.line(frame, first, second, color, 4, cv2.LINE_AA)

    cv2.circle(frame, pixels[0], 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, pixels[0], 10, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, pixels[-1], 12, color, -1, cv2.LINE_AA)
    cv2.circle(frame, pixels[-1], 12, (255, 255, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(
            frame,
            label,
            (pixels[-1][0] + 10, pixels[-1][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_navigation_paths(frame, paths, overhead_camera):
    """Draw the latest selected path for every robot in a stable colour."""
    if isinstance(paths, dict):
        for agent_id, path in sorted(paths.items()):
            draw_navigation_path(
                frame,
                path,
                overhead_camera,
                color=_PATH_COLORS[agent_id % len(_PATH_COLORS)],
                label=f"R{agent_id + 1}",
            )
        return frame
    return draw_navigation_path(frame, paths, overhead_camera)


def show_camera_frames(
    event,
    overhead_camera=None,
    navigation_path=None,
    show_robot_cameras=False,
    keyframe_note=None,
):
    global _LAST_OVERHEAD_FRAME
    frames = get_third_party_frames(event)
    if frames:
        _LAST_OVERHEAD_FRAME = frames[0].copy()
    if _LAST_OVERHEAD_FRAME is None:
        return
    overhead_frame = to_bgr(_LAST_OVERHEAD_FRAME.copy())
    draw_navigation_paths(overhead_frame, navigation_path, overhead_camera)
    draw_keyframe_note(overhead_frame, keyframe_note)
    draw_hover_coordinate(overhead_frame, overhead_camera)
    cv2.imshow(OVERHEAD_WINDOW, overhead_frame)

    if not show_robot_cameras:
        return

    agent_events = getattr(event, "events", None)
    if agent_events:
        for agent_id, agent_event in enumerate(
            agent_events[:ROBOT_CAMERA_COUNT]
        ):
            if agent_event.frame is not None:
                _LAST_ROBOT_FRAMES[agent_id] = agent_event.frame
    else:
        active_event = getattr(event, "_active_event", event)
        if active_event.frame is not None:
            agent_id = active_event.metadata.get("agentId", 0)
            _LAST_ROBOT_FRAMES[agent_id] = active_event.frame

    for agent_id in range(ROBOT_CAMERA_COUNT):
        robot_frame = _LAST_ROBOT_FRAMES.get(agent_id)
        if robot_frame is not None:
            cv2.imshow(robot_window_name(agent_id), to_bgr(robot_frame))


def refresh_camera_frames(
    event,
    overhead_camera=None,
    navigation_path=None,
    show_robot_cameras=False,
    keyframe_note=None,
):
    """Show a frame and let OpenCV process window repaint events."""
    show_camera_frames(
        event,
        overhead_camera,
        navigation_path,
        show_robot_cameras,
        keyframe_note,
    )
    cv2.waitKey(1)


def close_camera_windows():
    cv2.destroyAllWindows()
