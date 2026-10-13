"""Concurrent conflict scheduler for the AI2-THOR adapter."""

from __future__ import annotations

import math
import threading

from simulations.ai2thor.config import (
    ROBOT_SAFETY_RADIUS,
    SMOOTH_FINAL_POSITION_TOLERANCE,
    SMOOTH_MOVE_STEP,
    carried_object_safety_margin,
)


class MultiAgentReplanScheduler:
    """Run one resolution per disjoint local contact cluster.

    Navigation workers remain ordinary executor threads. This object only
    provides a per-cluster barrier and dispatches the selected A* release/task
    routes. Disjoint clusters therefore progress independently, while clusters
    that later touch are cancelled and rebuilt as one merged local problem.
    """

    def __init__(
        self,
        controller,
        route_planner,
        stationary_resolver=None,
        blocker_waiter=None,
        hypothesis_planner=None,
        contact_waiter=None,
    ):
        self.controller = controller
        self.route_planner = route_planner
        self.stationary_resolver = stationary_resolver
        self.blocker_waiter = blocker_waiter
        self.contact_waiter = contact_waiter or blocker_waiter
        self.hypothesis_planner = hypothesis_planner
        self._condition = threading.Condition(threading.RLock())
        self._contexts = {}
        self._resolutions = {}
        self._agent_resolution = {}
        self._next_resolution_id = 1

    def register(
        self,
        agent_id,
        object_type,
        future_open_object=None,
        action_duration=0.0,
        wait_for_on_conflict=(),
    ):
        with self._condition:
            self._contexts[agent_id] = {
                "object_type": object_type,
                "future_open_object": future_open_object,
                "action_duration": float(action_duration),
                "moving": True,
                "wait_for_on_conflict": tuple(wait_for_on_conflict),
            }
            self._condition.notify_all()

    def navigation_arrived(self, agent_id):
        with self._condition:
            context = self._contexts.get(agent_id)
            if context is not None:
                context["moving"] = False

    def unregister(self, agent_id):
        with self._condition:
            if agent_id not in self._agent_resolution:
                self._contexts.pop(agent_id, None)
            self._condition.notify_all()

    def _agent_position(self, agent_id):
        events = getattr(self.controller.last_event, "events", None) or []
        return events[agent_id].metadata["agent"]["position"]

    def _pair_clearance(self, first_id, second_id):
        events = getattr(self.controller.last_event, "events", None) or []
        payload_margin = max(
            carried_object_safety_margin(events[first_id]),
            carried_object_safety_margin(events[second_id]),
        )
        return (
            2 * ROBOT_SAFETY_RADIUS
            + payload_margin
            + SMOOTH_MOVE_STEP
            + SMOOTH_FINAL_POSITION_TOLERANCE
        )

    def _conflict_component(self, first_id, second_id):
        """Return only the transitive current-contact component."""
        moving = {
            agent_id
            for agent_id, context in self._contexts.items()
            if context["moving"]
        }
        if first_id not in moving or second_id not in moving:
            return set()

        component = {first_id, second_id}
        changed = True
        while changed:
            changed = False
            for candidate in moving - component:
                candidate_position = self._agent_position(candidate)
                if any(
                    math.hypot(
                        candidate_position["x"]
                        - self._agent_position(member)["x"],
                        candidate_position["z"]
                        - self._agent_position(member)["z"],
                    )
                    <= self._pair_clearance(candidate, member)
                    for member in component
                ):
                    component.add(candidate)
                    changed = True
        return component

    def _delete_resolution_locked(self, resolution_id):
        resolution = self._resolutions.pop(resolution_id, None)
        if resolution is None:
            return set()
        for agent_id in resolution["participants"]:
            if self._agent_resolution.get(agent_id) == resolution_id:
                self._agent_resolution.pop(agent_id, None)
        return set(resolution["participants"])

    def _create_resolution_locked(self, participants, initially_paused):
        resolution_id = self._next_resolution_id
        self._next_resolution_id += 1
        participants = set(participants)
        self._resolutions[resolution_id] = {
            "participants": participants,
            "paused": set(initially_paused) & participants,
            "status": "collecting",
            "queue": [],
            "current_agent": None,
            "current_route": None,
            "current_kind": None,
            "route_delivered": False,
            "resume_routes": {},
            "resumed": set(),
        }
        for agent_id in participants:
            self._agent_resolution[agent_id] = resolution_id
        return resolution_id

    def _join_or_create_resolution_locked(self, component, reporter):
        overlapping_ids = {
            self._agent_resolution[agent_id]
            for agent_id in component
            if agent_id in self._agent_resolution
        }
        if len(overlapping_ids) == 1:
            resolution_id = next(iter(overlapping_ids))
            resolution = self._resolutions[resolution_id]
            if component <= resolution["participants"]:
                resolution["paused"].add(reporter)
                return resolution_id

        participants = set(component)
        paused = {reporter}
        for resolution_id in sorted(overlapping_ids):
            old = self._resolutions[resolution_id]
            participants.update(old["participants"])
            paused.update(old["paused"])
            self._delete_resolution_locked(resolution_id)
            print(f"Merged contact cancels AI2-THOR resolution C{resolution_id}")
        resolution_id = self._create_resolution_locked(participants, paused)
        print(
            f"AI2-THOR local resolution C{resolution_id}: stopping "
            + ", ".join(f"R{value + 1}" for value in sorted(participants))
        )
        return resolution_id

    @staticmethod
    def _activate_step_locked(resolution):
        step = resolution["queue"][0]
        resolution["current_agent"] = step["agent_id"]
        resolution["current_kind"] = step["kind"]
        resolution["current_route"] = step["path"]
        resolution["route_delivered"] = False

    def _install_candidate(self, resolution_id, candidate, contexts):
        queue = [
            {
                "agent_id": release["blocker_agent"],
                "kind": "release",
                "path": release["release_path"],
            }
            for release in candidate["release_moves"]
        ]
        resume_routes = {
            int(agent_id): path
            for agent_id, path in candidate.get("current_paths", {}).items()
        }
        priority_agent = int(candidate["priority_agent"])
        concurrent=candidate.get("concurrent_resume_paths")
        if queue and concurrent is not None:
            prefix=candidate["priority_release_prefix"]
            if len(prefix)>1:
                queue.append(dict(agent_id=priority_agent,kind="priority",path=prefix))
            resume_routes={int(rid):route for rid,route in concurrent.items()}
        elif queue:
            # Match Algorithm 1 and the grid-cell executor: after the serial
            # releases, protect the recomputed priority corridor; only then
            # let yielding robots continue to their original goals.
            queue.append(
                {
                    "agent_id": priority_agent,
                    "kind": "priority",
                    "path": candidate["priority_path"],
                }
            )
            resume_routes.pop(priority_agent, None)
        else:
            resume_routes[priority_agent] = candidate["priority_path"]
        with self._condition:
            resolution = self._resolutions.get(resolution_id)
            if resolution is None:
                return
            resolution["queue"] = queue
            resolution["resume_routes"] = resume_routes
            resolution["contexts"] = dict(contexts)
            # Relocation continuations are regenerated from the observed
            # clearance poses. Direct reroutes remain exactly the selected
            # paths because no robot was displaced first.
            resolution["replan_on_resume"] = bool(queue) and concurrent is None
            resolution["resumed"] = set()
            if queue:
                resolution["status"] = "executing"
                self._activate_step_locked(resolution)
            else:
                resolution["status"] = "resuming"
            self._condition.notify_all()

    def _fail_resolution(self, resolution_id, error):
        print(f"AI2-THOR local resolution C{resolution_id} failed: {error}")
        with self._condition:
            resolution = self._resolutions.get(resolution_id)
            if resolution is None:
                return
            resolution["status"] = "failed"
            resolution["error"] = str(error)
            self._condition.notify_all()

    def _plan_resolution(self, resolution_id, participants, contexts):
        if self.hypothesis_planner is None:
            self._fail_resolution(
                resolution_id,
                "priority-hypothesis planner is unavailable",
            )
            return
        try:
            candidate = self.hypothesis_planner(participants, contexts)
        except (RuntimeError, TimeoutError) as error:
            self._fail_resolution(resolution_id, error)
            return
        self._install_candidate(resolution_id, candidate, contexts)

    def trigger(self, first_id, second_id, restart=False):
        """Freeze, snapshot and plan one local contact component."""
        del restart  # A fresh contact automatically merges/rebuilds its cluster.
        with self._condition:
            component = self._conflict_component(first_id, second_id)
            if len(component) < 2:
                return None
            resolution_id = self._join_or_create_resolution_locked(
                component,
                first_id,
            )
            resolution = self._resolutions[resolution_id]
            self._condition.notify_all()
            while (
                resolution_id in self._resolutions
                and not resolution["participants"] <= resolution["paused"]
            ):
                self._condition.wait(timeout=0.05)
            if resolution_id not in self._resolutions:
                # A merged component replaces this barrier, it does not mean
                # that navigation failed. Join the replacement resolution.
                return self._agent_resolution.get(first_id)

            should_plan = resolution["status"] == "collecting"
            if should_plan:
                resolution["status"] = "planning"
                participants = sorted(resolution["participants"])
                contexts = {
                    agent_id: self._contexts.get(agent_id)
                    for agent_id in participants
                }
                print(
                    f"AI2-THOR resolution C{resolution_id} snapshot: "
                    + ", ".join(
                        (
                            f"R{agent_id + 1}="
                            f"({self._agent_position(agent_id)['x']:.2f},"
                            f" {self._agent_position(agent_id)['z']:.2f})"
                        )
                        for agent_id in participants
                    )
                )
            else:
                participants = None
                contexts = None

        if should_plan:
            self._plan_resolution(resolution_id, participants, contexts)
        return resolution_id

    def resolve(
        self,
        requesting_id,
        blocking_id,
        *,
        direct_path_failed=False,
    ):
        """Resolve moving clusters or a working-vs-stationary contact."""
        with self._condition:
            requesting = self._contexts.get(requesting_id)
            blocking = self._contexts.get(blocking_id)
            both_navigating = bool(
                requesting
                and requesting["moving"]
                and blocking
                and blocking["moving"]
            )

        # Optional scene-authored contact protocol, not a policy score or DAG
        # predecessor: travel proceeds until a real collision is reported.
        if requesting and blocking_id in requesting.get("wait_for_on_conflict", ()):
            if self.contact_waiter is None or self.stationary_resolver is None:
                raise RuntimeError("Contact wait requires a completion waiter and clearance planner")
            print(f"R{requesting_id + 1} waits on contact for R{blocking_id + 1} to finish")
            self.contact_waiter(blocking_id)
            # Once its work is done, require an actual clearance move rather
            # than sending the waiting robot on a detour around the blocker.
            return self.stationary_resolver(
                requesting_id, blocking_id, requesting, reject_direct=True)

        if (both_navigating and blocking
                and requesting_id in blocking.get("wait_for_on_conflict", ())):
            # The priority robot can report the same contact first. Do not
            # freeze it in a mutual barrier while its peer awaits completion.
            # Replan around the peer's stationary body; live swept checks
            # still reject any edge that cannot pass safely.
            return self.route_planner(requesting_id, requesting, {}, True)["path"]

        if both_navigating:
            resolution_id = self.trigger(requesting_id, blocking_id)
            if resolution_id is None:
                return None
            return self.wait_for_route(requesting_id)

        if requesting is None or self.stationary_resolver is None:
            return None
        for _ in range(3):
            route = self.stationary_resolver(
                requesting_id,
                blocking_id,
                requesting,
                reject_direct=direct_path_failed,
            )
            if route is False:
                if self.blocker_waiter is None:
                    return None
                self.blocker_waiter(blocking_id)
                continue
            return route
        return None

    def wait_for_route(self, agent_id, route_completed=False):
        """Run releases, protected priority route, then yielding continuations."""
        with self._condition:
            resolution_id = self._agent_resolution.get(agent_id)
            if resolution_id is None:
                return None
            resolution = self._resolutions.get(resolution_id)
            if resolution is None:
                return None
            resolution["paused"].add(agent_id)
            self._condition.notify_all()

            if (
                route_completed
                and agent_id == resolution["current_agent"]
                and resolution["route_delivered"]
            ):
                resolution["queue"].pop(0)
                if resolution["queue"]:
                    self._activate_step_locked(resolution)
                else:
                    resolution["status"] = "resuming"
                    resolution["current_agent"] = None
                    resolution["current_kind"] = None
                    resolution["current_route"] = None
                    resolution["route_delivered"] = False
                self._condition.notify_all()
                if (
                    resolution["status"] == "resuming"
                    and agent_id not in resolution["resume_routes"]
                ):
                    resolution["paused"].discard(agent_id)
                    return None

            while (
                resolution_id in self._resolutions
                and resolution["status"] != "failed"
                and not (
                    resolution["status"] == "executing"
                    and resolution["current_agent"] == agent_id
                    or resolution["status"] == "resuming"
                    and agent_id in resolution["resume_routes"]
                )
            ):
                self._condition.wait(timeout=0.05)
            if resolution_id not in self._resolutions:
                replacement = self._agent_resolution.get(agent_id)
                if replacement is not None and replacement != resolution_id:
                    return self.wait_for_route(agent_id)
                return None
            if resolution["status"] == "failed":
                raise RuntimeError(
                    "AI2-THOR conflict resolution failed: "
                    + resolution.get("error", "unknown planning error")
                )
            if resolution["status"] == "resuming":
                if agent_id in resolution["resumed"]:
                    return None
                path = None
                if resolution.get("replan_on_resume"):
                    context = resolution.get("contexts", {}).get(agent_id)
                    if context is not None:
                        try:
                            route = self.route_planner(
                                agent_id,
                                context,
                                {},
                                True,
                            )
                            path = [
                                point.copy() for point in route["path"]
                            ]
                            resolution["resume_routes"][agent_id] = path
                            print(
                                f"AI2-THOR C{resolution_id}: replanned "
                                f"R{agent_id + 1} from its observed "
                                "clearance pose"
                            )
                        except RuntimeError:
                            # Keep the selected nominal continuation guarded by
                            # the normal swept collision detector. A new contact
                            # will create a fresh resolution instead of forcing
                            # an unsafe stale route through another body.
                            path = None
                if path is None:
                    path = [
                        point.copy()
                        for point in resolution["resume_routes"][agent_id]
                    ]
                resolution["resumed"].add(agent_id)
                resolution["paused"].discard(agent_id)
                if set(resolution["resume_routes"]) <= resolution["resumed"]:
                    self._delete_resolution_locked(resolution_id)
                self._condition.notify_all()
                return path
            if resolution["route_delivered"]:
                return None
            resolution["route_delivered"] = True
            resolution["paused"].discard(agent_id)
            return [point.copy() for point in resolution["current_route"]]

    def task_finished(self, agent_id):
        """Discard stale bookkeeping if a task ends during cancellation."""
        with self._condition:
            resolution_id = self._agent_resolution.get(agent_id)
            if resolution_id is None:
                return
            resolution = self._resolutions[resolution_id]
            resolution["participants"].discard(agent_id)
            resolution["paused"].discard(agent_id)
            resolution["resume_routes"].pop(agent_id, None)
            resolution["resumed"].discard(agent_id)
            self._agent_resolution.pop(agent_id, None)
            self._contexts.pop(agent_id, None)
            if not resolution["participants"]:
                self._delete_resolution_locked(resolution_id)
            self._condition.notify_all()

    def abort(self):
        with self._condition:
            self._resolutions.clear()
            self._agent_resolution.clear()
            self._condition.notify_all()
