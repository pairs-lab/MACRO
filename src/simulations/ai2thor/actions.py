"""High-level object actions for the AI2-THOR adapter."""

import math

from simulations.ai2thor.config import (
    AGENT_COUNT,
    AGENT_RADIUS,
    MAXIMUM_INTERACTION_DISTANCE,
    interaction_distance_for,
    MAXIMUM_INTERACTION_YAW_ERROR,
)
from simulations.ai2thor.controller import require_success
from .navigation.door_rules import future_open_rule


ROBOT_AGENT_IDS = {
    f"robot{agent_id + 1}": agent_id for agent_id in range(AGENT_COUNT)
}


class Actions:
    """Execute object actions for one of the real iTHOR agents."""

    def __init__(self, controller, robot_camera, on_step=None):
        self.controller = controller
        self.robot_camera = robot_camera
        self.on_step = on_step
        self.held_object_ids = {}
        self._stowed_object_ids = set()
        self._compound_payload_ids = set()
        # Unity can reject the second PutObject into a small receptacle and
        # lose its parentReceptacle metadata. Keep the intended containment
        # graph so compound payloads remain complete when moved later.
        self.receptacle_contents = {}

    def _agent_id(self, robot):
        try:
            return ROBOT_AGENT_IDS[robot]
        except KeyError as error:
            raise ValueError(
                f"Unknown robot {robot!r}. Available robots: "
                f"{list(ROBOT_AGENT_IDS)}"
            ) from error

    def _objects(self, agent_id):
        event = self.controller.last_event
        events = getattr(event, "events", None)
        if events and len(events) > agent_id:
            event = events[agent_id]
        return event.metadata.get("objects", []), event.metadata["agent"]

    def _find_object(self, robot, object_type, require_visible=False):
        agent_id = self._agent_id(robot)
        objects, agent = self._objects(agent_id)
        matches = [
            obj
            for obj in objects
            if obj.get("objectType", "").casefold() == object_type.casefold()
            and (not require_visible or obj.get("visible", False))
        ]
        if not matches:
            visibility = " visible" if require_visible else ""
            raise ValueError(f"No{visibility} object of type {object_type!r} exists")

        position = agent["position"]
        return min(
            matches,
            key=lambda obj: math.hypot(
                obj["position"]["x"] - position["x"],
                obj["position"]["z"] - position["z"],
            ),
        )

    @staticmethod
    def _surface_distance(position, scene_object):
        bounds = scene_object.get("axisAlignedBoundingBox")
        if not bounds:
            return math.hypot(
                scene_object["position"]["x"] - position["x"],
                scene_object["position"]["z"] - position["z"],
            )
        center = bounds["center"]
        size = bounds["size"]
        delta_x = max(
            center["x"] - size["x"] / 2 - position["x"],
            0.0,
            position["x"] - center["x"] - size["x"] / 2,
        )
        delta_z = max(
            center["z"] - size["z"] / 2 - position["z"],
            0.0,
            position["z"] - center["z"] - size["z"] / 2,
        )
        return math.hypot(delta_x, delta_z)

    def _require_interaction_pose(self, robot, scene_object):
        """Reject manipulation unless the robot is close and facing target."""
        agent_id = self._agent_id(robot)
        objects, agent = self._objects(agent_id)
        latest = next(
            (
                obj
                for obj in objects
                if obj["objectId"] == scene_object["objectId"]
            ),
            scene_object,
        )
        distance = self._surface_distance(agent["position"], latest)
        interaction_limit = interaction_distance_for(latest)
        if distance > interaction_limit:
            raise RuntimeError(
                f"{robot} is {distance:.2f} m from {latest['objectType']}; "
                f"must be within {interaction_limit:.2f} m"
            )

        delta_x = latest["position"]["x"] - agent["position"]["x"]
        delta_z = latest["position"]["z"] - agent["position"]["z"]
        target_yaw = math.degrees(math.atan2(delta_x, delta_z)) % 360
        yaw_error = (
            target_yaw - agent["rotation"]["y"] + 180
        ) % 360 - 180
        if abs(yaw_error) > MAXIMUM_INTERACTION_YAW_ERROR:
            raise RuntimeError(
                f"{robot} is not facing {latest['objectType']}; "
                f"yaw error is {abs(yaw_error):.1f} degrees"
            )
        if not latest.get("visible", False):
            raise RuntimeError(
                f"{latest['objectType']} is not visible from {robot}'s "
                "interaction pose"
            )
        return latest

    def _step(self, robot, action, **parameters):
        agent_id = self._agent_id(robot)
        event = self.controller.step(
            action=action,
            agentId=agent_id,
            renderImage=True,
            **parameters,
        )
        event = require_success(event, action)
        if self.on_step is not None:
            self.on_step(event)
        return event

    def _target_object(self, robot, object_type_or_id, require_visible=False):
        if "|" not in object_type_or_id:
            return self._find_object(
                robot,
                object_type_or_id,
                require_visible=require_visible,
            )
        agent_id = self._agent_id(robot)
        objects, _ = self._objects(agent_id)
        target = next(
            (
                obj
                for obj in objects
                if obj["objectId"] == object_type_or_id
            ),
            None,
        )
        if target is None:
            raise ValueError(f"Unknown objectId {object_type_or_id!r}")
        if require_visible and not target.get("visible", False):
            raise ValueError(f"Object {object_type_or_id!r} is not visible")
        return target

    def pick(self, robot, object_type_or_id, force_action=False):
        """Pick up the nearest object by objectType or objectId."""
        agent_id = self._agent_id(robot)
        target = self._target_object(robot, object_type_or_id)
        if force_action:
            objects, agent = self._objects(agent_id)
            target = next(
                (
                    obj for obj in objects
                    if obj["objectId"] == target["objectId"]
                ),
                target,
            )
            distance = self._surface_distance(agent["position"], target)
            if distance > MAXIMUM_INTERACTION_DISTANCE:
                raise RuntimeError(
                    f"{robot} is {distance:.2f} m from "
                    f"{target['objectType']}; must be within "
                    f"{MAXIMUM_INTERACTION_DISTANCE:.2f} m"
                )
        else:
            target = self._require_interaction_pose(robot, target)
        contained_object_ids = tuple(
            dict.fromkeys(
                [*(target.get("receptacleObjectIds") or ())]
                + list(self.receptacle_contents.get(target["objectId"], ()))
            )
        )
        event = self._step(
            robot,
            "PickupObject",
            objectId=target["objectId"],
            forceAction=force_action,
        )
        self.held_object_ids[agent_id] = target["objectId"]
        self._stowed_object_ids.discard(target["objectId"])
        # A filled receptacle needs rigid compound-payload handling; do not
        # separate its contents by shifting only the held parent.
        if not contained_object_ids:
            self._compound_payload_ids.discard(target["objectId"])
            event = self._retract_hand(robot, agent_id, event)
        else:
            self._compound_payload_ids.add(target["objectId"])
            print(
                f"{robot} picked {target['objectType']} with "
                f"{len(contained_object_ids)} contained object(s); "
                "held receptacle was not shifted"
            )
        return event

    def _stow_distance(self, agent_id, event):
        """Forward displacement to a compact pose, outside the agent body.

        Missing metadata falls back to one bounded retraction, never a
        cumulative shift each time navigation starts.
        """
        events = getattr(event, "events", None)
        if events and len(events) > agent_id:
            event = events[agent_id]
        metadata = getattr(event, "metadata", {})
        object_id = self.held_object_ids.get(agent_id)
        obj = next((obj for obj in metadata.get("objects", [])
                    if obj.get("objectId") == object_id), None)
        agent = metadata.get("agent", {})
        if not obj or "position" not in obj or "position" not in agent:
            return 0.0 if object_id in self._stowed_object_ids else 0.40
        position = agent["position"]
        yaw = math.radians(agent.get("rotation", {}).get("y", 0.0))
        forward = ((obj["position"]["x"] - position["x"]) * math.sin(yaw)
                   + (obj["position"]["z"] - position["z"]) * math.cos(yaw))
        bounds = obj.get("axisAlignedBoundingBox") or {}
        size = bounds.get("size", {})
        extent = 0.5 * max(size.get("x", 0.10), size.get("z", 0.10))
        target = AGENT_RADIUS + extent + 0.03
        return min(1.0, max(0.0, forward - target))

    def stow_for_navigation(self, robot):
        """Recheck the live payload pose before delivery without forceAction."""
        agent_id = self._agent_id(robot)
        object_id = self.held_object_ids.get(agent_id)
        if object_id is None or object_id in self._compound_payload_ids:
            return
        event = getattr(self.controller, "last_event", None)
        if event is not None:
            return self._retract_hand(robot, agent_id, event)

    def _retract_hand(self, robot, agent_id, pickup_event):
        """Stow an uncontained payload as far as collision checks allow.

        A coarse move can collide even when a shorter move remains possible.
        Never force the hand through geometry or turn a successful pickup into
        a failed task merely because the remaining retraction is obstructed.
        """
        event = pickup_event
        retracted = 0.0
        step_size = 0.05
        maximum = self._stow_distance(agent_id, pickup_event)
        while retracted < maximum - 1e-8:
            magnitude = min(step_size, maximum - retracted)
            attempt = self.controller.step(
                action="MoveHandBack", agentId=agent_id,
                moveMagnitude=magnitude, renderImage=True,
            )
            if not attempt.metadata["lastActionSuccess"]:
                step_size *= 0.5
                if step_size < 0.005:
                    print(
                        f"{robot} hand retraction stopped after {retracted:.2f}m: "
                        f"{attempt.metadata.get('errorMessage', 'blocked by physical collision')}"
                    )
                    break
                continue
            retracted += magnitude
            event = attempt
            if self.on_step is not None:
                self.on_step(event)
        if retracted >= maximum - 1e-8:
            self._stowed_object_ids.add(self.held_object_ids.get(agent_id))
        return event

    def place(self, robot, receptacle_type_or_id):
        """Place the held object into a receptacle by type or objectId."""
        agent_id = self._agent_id(robot)
        held_object_id = self.held_object_ids.get(agent_id)
        if held_object_id is None:
            raise RuntimeError(f"{robot} is not holding an object")

        objects, _ = self._objects(agent_id)
        if "|" in receptacle_type_or_id:
            receptacle = next(
                (obj for obj in objects if obj["objectId"] == receptacle_type_or_id),
                None,
            )
            if receptacle is None:
                raise ValueError(f"Unknown objectId {receptacle_type_or_id!r}")
        else:
            receptacle = self._find_object(robot, receptacle_type_or_id)
        receptacle = self._require_interaction_pose(robot, receptacle)
        if not receptacle.get("receptacle", False):
            raise ValueError(
                f"{receptacle['objectType']!r} is not a receptacle in this scene"
            )

        # Prefer the normal semantic action. Some small valid receptacles expose
        # no Unity placement slots for a larger held object, so fall back to the
        # receptacle's physical top-centre point.
        event = self.controller.step(
            action="PutObject",
            agentId=agent_id,
            objectId=receptacle["objectId"],
            # Contents of a receptacle must remain physical so they travel with
            # it later instead of remaining frozen at a world coordinate.
            placeStationary=False,
            renderImage=True,
        )
        if (
            not event.metadata["lastActionSuccess"]
            and future_open_rule(receptacle["objectType"]) is not None
        ):
            # Articulated receptacles can reject an otherwise valid slot when
            # the held compound object is close to the open door collider.
            # Interaction distance and heading were already validated above;
            # forceAction only relaxes Unity's slot-selection gate.
            event = self.controller.step(
                action="PutObject",
                agentId=agent_id,
                objectId=receptacle["objectId"],
                placeStationary=False,
                forceAction=True,
                renderImage=True,
            )
        if receptacle["objectType"].casefold() == "microwave":
            first_error = event.metadata.get("errorMessage", "")
            if not event.metadata["lastActionSuccess"]:
                # Retry semantic placement with a stationary payload. Do not
                # use an AABB top-centre teleport for an enclosed appliance.
                event = self.controller.step(
                    action="PutObject", agentId=agent_id,
                    objectId=receptacle["objectId"], placeStationary=True,
                    renderImage=True,
                )
            if not event.metadata["lastActionSuccess"]:
                raise RuntimeError(
                    f"PutObject in Microwave failed: {first_error}; stationary retry: "
                    f"{event.metadata.get('errorMessage', '')}"
                )
            latest = {o['objectId']:o for o in event.metadata.get('objects', [])}
            parent = latest.get(receptacle['objectId'], {})
            payload = latest.get(held_object_id, {})
            if (held_object_id not in (parent.get('receptacleObjectIds') or [])
                    and receptacle['objectId'] not in (payload.get('parentReceptacles') or [])):
                raise RuntimeError('Microwave placement succeeded but containment was not confirmed')
        if not event.metadata["lastActionSuccess"]:
            bounds = receptacle["axisAlignedBoundingBox"]
            point = bounds["center"].copy()
            point["y"] += bounds["size"]["y"] / 2 + 0.08
            event = require_success(
                self.controller.step(
                    action="PlaceObjectAtPoint",
                    agentId=agent_id,
                    objectId=held_object_id,
                    position=point,
                    renderImage=True,
                ),
                f"Place held object in {receptacle['objectType']}",
            )

        # Open articulated receptacles may make Unity choose a corner slot.
        # Fixed-scene rules can provide the physical centre of an interior
        # shelf; reposition only after semantic PutObject has succeeded so the
        # receptacle relationship is still established normally.
        rule = future_open_rule(receptacle["objectType"])
        placement_point = (rule or {}).get("placement_point")
        if placement_point is not None:
            tracked_content_ids = tuple(
                self.receptacle_contents.get(held_object_id, ())
            )
            current_objects = {
                obj["objectId"]: obj
                for obj in event.metadata.get("objects", [])
            }
            content_records = [
                current_objects[object_id]
                for object_id in tracked_content_ids
                if object_id in current_objects
            ]
            disabled_records = []
            try:
                # A loose or semantically unparented item can occupy the
                # container's destination and make Unity reject its move.
                # Disable only during this atomic reposition, then restore all
                # tracked contents relative to the container's new bounds.
                for content in content_records:
                    require_success(
                        self.controller.step(
                            action="DisableObject",
                            agentId=agent_id,
                            objectId=content["objectId"],
                            renderImage=False,
                        ),
                        f"Temporarily hold {content['objectType']}",
                    )
                    disabled_records.append(content)

                event = require_success(
                    self.controller.step(
                        action="PlaceObjectAtPoint",
                        agentId=agent_id,
                        objectId=held_object_id,
                        position=placement_point,
                        forceKinematic=False,
                        renderImage=True,
                    ),
                    f"Centre held object in {receptacle['objectType']}",
                )
                if disabled_records:
                    event = self._restore_container_contents(
                        agent_id,
                        held_object_id,
                        disabled_records,
                    )
            except BaseException:
                # Never leave a scene object disabled if centring fails.
                for content in disabled_records:
                    self.controller.step(
                        action="EnableObject",
                        agentId=agent_id,
                        objectId=content["objectId"],
                        renderImage=False,
                    )
                raise
            placed = next(
                obj
                for obj in event.metadata.get("objects", [])
                if obj["objectId"] == held_object_id
            )
            error = math.hypot(
                placed["position"]["x"] - placement_point["x"],
                placed["position"]["z"] - placement_point["z"],
            )
            if error > 0.10:
                raise RuntimeError(
                    f"{placed['objectType']} did not settle at the centre of "
                    f"{receptacle['objectType']}"
                )
            print(
                f"Centred {placed['objectType']} in "
                f"{receptacle['objectType']} at "
                f"x={placed['position']['x']:.2f}, "
                f"z={placed['position']['z']:.2f}"
            )

        contents = self.receptacle_contents.setdefault(
            receptacle["objectId"],
            [],
        )
        if held_object_id not in contents:
            contents.append(held_object_id)
        self.held_object_ids.pop(agent_id, None)
        if self.on_step is not None:
            self.on_step(event)
        return event

    def _restore_container_contents(
        self,
        agent_id,
        container_id,
        content_records,
    ):
        """Restore a tracked compound payload inside its moved container."""
        objects = {
            obj["objectId"]: obj
            for obj in self.controller.last_event.metadata.get("objects", [])
        }
        container = objects[container_id]
        bounds = container["axisAlignedBoundingBox"]
        center = bounds["center"]
        top_y = center["y"] + bounds["size"]["y"] / 2
        event = self.controller.last_event
        support_height = 0.0
        count = len(content_records)
        for index, content in enumerate(content_records):
            require_success(
                self.controller.step(
                    action="EnableObject",
                    agentId=agent_id,
                    objectId=content["objectId"],
                    renderImage=False,
                ),
                f"Restore {content['objectType']}",
            )
            size = content["axisAlignedBoundingBox"]["size"]
            # Stack contents from the bottom upward. A Bowl is too narrow for
            # Apple and Tomato to occupy two non-overlapping horizontal spawn
            # slots, so side-by-side placement makes Unity reject or eject one
            # item. Dynamic physics then lets this compact stack settle.
            direction = -1 if index % 2 == 0 else 1
            point = {
                "x": center["x"] + direction * min(
                    0.012,
                    bounds["size"]["x"] * 0.06,
                ),
                "y": top_y + support_height + size["y"] / 2 + 0.008,
                "z": center["z"],
            }
            event = self.controller.step(
                action="PlaceObjectAtPoint",
                agentId=agent_id,
                objectId=content["objectId"],
                position=point,
                forceKinematic=False,
                renderImage=index == count - 1,
            )
            if not event.metadata["lastActionSuccess"]:
                # A compound payload can fit geometrically but fail Unity's
                # dynamic "come to rest" test inside an already constrained
                # receptacle (for example, a Bowl on a Fridge shelf). Preserve
                # the intended containment at the same validated point with a
                # kinematic placement instead of aborting the whole mission.
                print(
                    f"{content['objectType']} could not settle dynamically; "
                    "restoring it at the validated container-relative pose"
                )
                event = self.controller.step(
                    action="PlaceObjectAtPoint",
                    agentId=agent_id,
                    objectId=content["objectId"],
                    position=point,
                    forceKinematic=True,
                    renderImage=index == count - 1,
                )
            event = require_success(
                event,
                f"Restore {content['objectType']} inside container",
            )
            support_height += size["y"] + 0.008
        print(
            f"Restored {count} contained object(s) inside "
            f"{container['objectType']}"
        )
        return event

    def switch_on(self, robot, object_type_or_id):
        """Turn on a toggleable object if it is currently off."""
        return self._set_toggle(robot, object_type_or_id, enabled=True)

    def switch_off(self, robot, object_type_or_id):
        """Turn off a toggleable object if it is currently on."""
        return self._set_toggle(robot, object_type_or_id, enabled=False)

    def wash(self, robot, object_type_or_id):
        """Validate washing while holding an object under a running faucet."""
        agent_id = self._agent_id(robot)
        event = self.controller.last_event.events[agent_id]
        inventory = event.metadata.get("inventoryObjects") or []
        if not any(
            item.get("objectType", "").casefold()
            == object_type_or_id.casefold()
            or item.get("objectId") == object_type_or_id
            for item in inventory
        ):
            raise RuntimeError(
                f"{robot} must hold {object_type_or_id} while washing"
            )
        objects, _ = self._objects(agent_id)
        if not any(
            obj.get("objectType") == "Faucet" and obj.get("isToggled", False)
            for obj in objects
        ):
            raise RuntimeError("Faucet must be on while washing")
        # Washing is represented by the executor's configured one-second
        # action duration.  Emit a rendered no-op so the overhead display has
        # a fresh multi-agent frame while that semantic action is recorded.
        return self._step(robot, "Pass")

    def _set_toggle(self, robot, object_type_or_id, enabled):
        target = self._target_object(
            robot,
            object_type_or_id,
            require_visible=True,
        )
        target = self._require_interaction_pose(robot, target)
        if target.get("isToggled") == enabled:
            return self.controller.last_event
        action = "ToggleObjectOn" if enabled else "ToggleObjectOff"
        return self._step(robot, action, objectId=target["objectId"])

    def open(self, robot, object_type_or_id):
        """Open an openable object."""
        target = self._target_object(
            robot,
            object_type_or_id,
            require_visible=True,
        )
        target = self._require_interaction_pose(robot, target)
        return self._step(robot, "OpenObject", objectId=target["objectId"])

    def close(self, robot, object_type_or_id):
        """Close a closable object."""
        target = self._target_object(
            robot,
            object_type_or_id,
            require_visible=True,
        )
        target = self._require_interaction_pose(robot, target)
        return self._step(robot, "CloseObject", objectId=target["objectId"])

def create_action_function(controller, robot_camera, on_step=None):
    """Return an Actions object bound to the running simulation."""
    return Actions(controller, robot_camera, on_step=on_step)
