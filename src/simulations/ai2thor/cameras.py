"""AI2-THOR overhead and robot camera operations."""

import math
import os

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2

from .config import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    ROBOT_CAMERA_PITCH,
)
from .controller import require_success


class OverheadCamera:
    """Fixed map camera with mouse pan, zoom, tilt, and rotation."""

    def __init__(self, properties):
        self.position = properties["position"].copy()
        self.rotation = properties["rotation"].copy()
        self.orthographic_size = properties["orthographicSize"]

        self.initial_position = self.position.copy()
        self.initial_rotation = self.rotation.copy()
        self.initial_size = self.orthographic_size

        self.drag_start = None
        self.drag_position = None
        self.rotate_start = None
        self.rotate_rotation = None
        self.needs_update = False
        self.needs_redraw = False
        self.cursor_pixel = None
        self.cursor_world = None

    def parameters(self):
        return {
            "position": self.position,
            "rotation": self.rotation,
            "orthographic": True,
            "orthographicSize": self.orthographic_size,
        }

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEWHEEL:
            self._zoom(flags)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_position = self.position.copy()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.rotate_start = (x, y)
            self.rotate_rotation = self.rotation.copy()
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self._pan(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.rotate_start is not None:
            self._rotate(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.drag_position = None
        elif event == cv2.EVENT_MBUTTONUP:
            self.rotate_start = None
            self.rotate_rotation = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset()
        self._update_cursor(x, y)

    def _update_cursor(self, x, y):
        """Project the hovered overhead pixel onto the floor plane."""
        yaw = math.radians(self.rotation["y"])
        pitch = math.radians(self.rotation["x"])
        half_height = self.orthographic_size
        half_width = half_height * DISPLAY_WIDTH / DISPLAY_HEIGHT
        horizontal = (2.0 * x / DISPLAY_WIDTH - 1.0) * half_width
        vertical = (1.0 - 2.0 * y / DISPLAY_HEIGHT) * half_height
        right = (math.cos(yaw), 0.0, -math.sin(yaw))
        up = (
            math.sin(yaw) * math.sin(pitch),
            math.cos(pitch),
            math.cos(yaw) * math.sin(pitch),
        )
        forward = (
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        )
        origin = {
            axis: self.position[axis]
            + horizontal * right[index]
            + vertical * up[index]
            for index, axis in enumerate(("x", "y", "z"))
        }
        if abs(forward[1]) < 1e-6:
            world = None
        else:
            distance = -origin["y"] / forward[1]
            world = {
                "x": origin["x"] + distance * forward[0],
                "z": origin["z"] + distance * forward[2],
            }
        self.cursor_pixel = (int(x), int(y))
        self.cursor_world = world
        self.needs_redraw = True

    def _zoom(self, flags):
        delta = (flags >> 16) & 0xFFFF
        if delta >= 0x8000:
            delta -= 0x10000
        if not delta:
            return

        factor = 0.85 if delta > 0 else 1.18
        self.orthographic_size = max(
            0.8, min(12.0, self.orthographic_size * factor)
        )
        self.needs_update = True

    def _pan(self, x, y):
        dx = x - self.drag_start[0]
        dy = y - self.drag_start[1]
        aspect_ratio = DISPLAY_WIDTH / DISPLAY_HEIGHT
        self.position["x"] = self.drag_position["x"] - (
            dx * self.orthographic_size * aspect_ratio / DISPLAY_WIDTH
        )
        self.position["z"] = self.drag_position["z"] + (
            dy * self.orthographic_size / DISPLAY_HEIGHT
        )
        self.needs_update = True

    def _rotate(self, x, y):
        dx = x - self.rotate_start[0]
        dy = y - self.rotate_start[1]
        self.rotation["y"] = (self.rotate_rotation["y"] + dx * 0.3) % 360
        self.rotation["x"] = max(
            35, min(90, self.rotate_rotation["x"] - dy * 0.15)
        )
        self.needs_update = True

    def reset(self):
        self.position = self.initial_position.copy()
        self.rotation = self.initial_rotation.copy()
        self.orthographic_size = self.initial_size
        self.needs_update = True


class RobotCamera:
    """Third-party camera that follows the robot's eyes."""

    def __init__(self, event):
        agent_position = event.metadata["agent"]["position"]
        camera_position = event.metadata["cameraPosition"]
        self.eye_height = camera_position["y"] - agent_position["y"]
        self.field_of_view = event.metadata["fov"]

    def parameters(self, event, include_field_of_view=False):
        agent = event.metadata["agent"]
        return self.parameters_for_pose(
            agent["position"],
            agent["rotation"]["y"],
            include_field_of_view,
        )

    def parameters_for_pose(
        self,
        position,
        yaw,
        include_field_of_view=False,
        pitch=ROBOT_CAMERA_PITCH,
    ):
        """Build camera parameters for a real iTHOR agent pose."""
        parameters = {
            "position": {
                "x": position["x"],
                "y": position["y"] + self.eye_height,
                "z": position["z"],
            },
            "rotation": {
                "x": pitch,
                "y": yaw,
                "z": 0,
            },
        }
        if include_field_of_view:
            parameters["fieldOfView"] = self.field_of_view
        return parameters

def setup_cameras(controller):
    """Create the overhead camera; native agent frames provide robot POV."""
    properties_event = controller.step(action="GetMapViewCameraProperties")
    require_success(properties_event, "Get map camera properties")
    overhead_camera = OverheadCamera(properties_event.metadata["actionReturn"])

    event = controller.step(
        action="AddThirdPartyCamera",
        **overhead_camera.parameters(),
    )
    require_success(event, "Add overhead camera")

    robot_camera = RobotCamera(event)
    # GetMapViewCameraProperties already configures the overhead view correctly.
    # ToggleMapView must not be used here: it also converts the active agent's
    # native camera into an orthographic map camera and corrupts the robot POV.
    event = require_success(controller.step(action="Pass"), "Refresh cameras")
    return overhead_camera, robot_camera, event


def update_overhead_camera(controller, overhead_camera):
    event = controller.step(
        action="UpdateThirdPartyCamera",
        thirdPartyCameraId=0,
        **overhead_camera.parameters(),
    )
    overhead_camera.needs_update = False
    return require_success(event, "Update overhead camera")


def move_robot(
    controller,
    robot_camera,
    action,
    report_failure=True,
    agent_id=0,
    render_image=True,
    **action_parameters,
):
    """Move or rotate the robot and update its independent camera."""
    # Looking down also moves the agent's physical hand and any held object.
    # Level it before locomotion so a held object does not hit furniture.
    level_event = level_robot_camera(controller, agent_id, report_failure)
    if level_event is None:
        return None

    agent_event = controller.last_event.events[agent_id]
    is_carrying = bool(agent_event.metadata.get("inventoryObjects"))

    # A held object's bounding box can reject an otherwise collision-free
    # in-place turn. Translation remains collision checked by the nav planner.
    if is_carrying and action in {"RotateLeft", "RotateRight"}:
        action_parameters.setdefault("forceAction", True)

    # One rendered action provides both the overhead frame and native robot POV.
    # This avoids the old second UpdateThirdPartyCamera action on every step.
    event = controller.step(
        action=action,
        agentId=agent_id,
        renderImage=render_image,
        **action_parameters,
    )
    if not event.metadata["lastActionSuccess"]:
        if report_failure:
            print("\nAction failed:", event.metadata["errorMessage"])
        return None

    return event


def level_robot_camera(controller, agent_id=0, report_failure=True):
    """Return the physical agent camera/hand to a safe horizontal pose."""
    agent_event = controller.last_event.events[agent_id]
    horizon = round(agent_event.metadata["agent"]["cameraHorizon"], 1)
    if abs(horizon) < 0.1:
        return controller.last_event

    level_action = "LookUp" if horizon > 0 else "LookDown"
    event = controller.step(
        action=level_action,
        agentId=agent_id,
        degrees=abs(horizon),
        renderImage=False,
    )
    if not event.metadata["lastActionSuccess"]:
        if report_failure:
            print("\nAction failed:", event.metadata["errorMessage"])
        return None
    return event


def aim_robot_camera(controller, target, agent_id=0):
    """Aim the real agent camera vertically at a target and render its POV."""
    event = controller.last_event.events[agent_id]
    # Interaction requires looking at the target even while holding an item.
    # Unity still validates the look action against the physical payload.
    agent = event.metadata["agent"]
    camera_y = event.metadata["cameraPosition"]["y"]
    horizontal_distance = max(
        math.hypot(
            target["x"] - agent["position"]["x"],
            target["z"] - agent["position"]["z"],
        ),
        1e-6,
    )
    desired_horizon = max(
        -30,
        min(60, math.degrees(math.atan2(camera_y - target["y"], horizontal_distance))),
    )
    current_horizon = agent["cameraHorizon"]
    difference = round(desired_horizon - current_horizon, 1)
    if abs(difference) < 0.5:
        return controller.last_event

    action = "LookDown" if difference > 0 else "LookUp"
    return require_success(
        controller.step(
            action=action,
            agentId=agent_id,
            degrees=abs(difference),
            renderImage=True,
        ),
        f"Aim robot{agent_id + 1} camera",
    )
