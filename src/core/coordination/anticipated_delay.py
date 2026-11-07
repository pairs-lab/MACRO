"""Short-horizon conflict exposure expressed as task delays, not penalties.

Reservations are immutable nominal candidate trajectories (one pass). Moving
segments wait for earlier-task reservations; stationary bodies take precedence.
An unresolved reservation is charged ONLY up to the horizon and reported as
censored, not certified feasible or solved. This is a predictor, not MAPF.
"""
from dataclasses import dataclass
import math
import os
import time
from .dynamic_collision import swept_separation


@dataclass(frozen=True)
class Segment:
    robot: str
    task: int | None
    start: float
    end: float
    first: tuple
    last: tuple
    task_start: float
    anchored: bool = False

    @property
    def moving(self):
        return self.first != self.last

    def position(self, at):
        fraction = max(0., min(1., (at-self.start)/(self.end-self.start)))
        if len(self.first) == 2 and len(self.last) == 2:
            return (self.first[0]+fraction*(self.last[0]-self.first[0]),
                    self.first[1]+fraction*(self.last[1]-self.first[1]))
        return tuple(a+fraction*(b-a) for a,b in zip(self.first,self.last))


class ReservationTable:
    """Time buckets index continuous swept segments, including edge swaps."""
    def __init__(self, segments, slot_s, horizon, robot_order=None):
        self.slot_s=slot_s; self.horizon=horizon; self.buckets={}
        self.robot_order = robot_order
        # Experimental: focused checks improve, but end-to-end pilots have not
        # established a gain. Keep the stable execution path as the default.
        self._reuse_positions = os.environ.get('MARCO_RESERVATION_POSITION_CACHE', '0') == '1'
        self._positions = {}
        self.position_cache_hits = 0
        self.position_cache_misses = 0
        for segment in segments:
            for slot in self.slots(segment.start, segment.end):
                self.buckets.setdefault(slot, []).append(segment)

    def slots(self, start, end):
        return range(max(0,math.floor(start/self.slot_s)),
                     math.ceil(min(end,self.horizon)/self.slot_s))

    def blockers(self, segment, clearance):
        seen=set(); hits=[]
        order = self.robot_order
        own_key = ((segment.task_start, segment.robot) if order is None else
                   (order.get(segment.robot, -1), segment.task_start, segment.robot))
        # Only immutable reservation segments are cached across queries. Query
        # segments are short-lived: cache their interpolation locally by time.
        own_position = segment.position
        if self._reuse_positions:
            query_positions = {}
            def own_position(at):
                if at not in query_positions:
                    query_positions[at] = segment.position(at)
                return query_positions[at]
        for slot in self.slots(segment.start,segment.end):
            for other in self.buckets.get(slot, ()):
                if id(other) in seen or other.robot == segment.robot:
                    continue
                seen.add(id(other))
                begin=max(segment.start,other.start)
                end=min(segment.end,other.end,self.horizon)
                if end-begin <= 1e-9:
                    continue
                # Initial bodies cannot be wished away. Future occupancy is
                # ordered like its approaching task when explicitly enabled;
                # the lower-priority task must then delay its arrival.
                if other.moving or (self.robot_order is not None and not other.anchored):
                    if self.robot_order is None:
                        other_key = (other.task_start, other.robot)
                    else:
                        other_key = (self.robot_order.get(other.robot, -1), other.task_start, other.robot)
                    if other_key > own_key:
                        continue
                if swept_separation(own_position(begin),own_position(end),
                                    (self._reservation_position(other, begin) if self._reuse_positions else other.position(begin)),
                                    (self._reservation_position(other, end) if self._reuse_positions else other.position(end))) < clearance-1e-8:
                    hits.append(other)
        return hits

    def _reservation_position(self, segment, at):
        # Every queried reservation remains strongly held by self.buckets for
        # this table's lifetime, so its id cannot be recycled while cached.
        key = (id(segment), at)
        if key in self._positions:
            self.position_cache_hits += 1
            return self._positions[key]
        self.position_cache_misses += 1
        value = segment.position(at)
        if len(self._positions) < 4096:
            self._positions[key] = value
        return value


def project_trajectories(*, evaluation, robots, tasks, positions, speed,
                         grid_size, horizon=5., window_start=0., task_limit=None,
                         wait_before_navigation=False, terminal_occupancy_task=False):
    """Reuse cached task paths; no A*, executor rollout or recursive repair."""
    segments=[]
    ordered_tasks = sorted(tasks, key=lambda t:(evaluation["start_times"].get(t["id"],math.inf),t["id"]))
    for robot in robots:
        cursor=window_start
        pose=positions.get(robot.robot_id,robot.cell)
        projected = 0
        anchored = True
        last_task = None
        for task in ordered_tasks:
            tid=task["id"]
            if task["robot"] != robot.robot_id or tid not in evaluation["start_times"]:
                continue
            if task_limit is not None and projected >= task_limit:
                break
            projected += 1
            last_task = tid
            start=evaluation["start_times"][tid]
            finish=evaluation["finish_times"][tid]
            route=evaluation["task_paths"].get(tid) or (pose,)
            if start > cursor+1e-9:
                segments.append(Segment(robot.robot_id,tid,cursor,min(start,horizon),pose,pose,start,anchored))
            travel=evaluation["task_travel_times"][tid]
            lengths=[math.dist(a,b) for a,b in zip(route,route[1:])]
            # Movement credit shortens only the first edge, not the geometry.
            credit=max(0.,sum(lengths)/speed-travel)
            at=start
            waiting = (evaluation.get('future_delay_by_task', {}).get(tid, 0.)
                       if wait_before_navigation else 0.)
            if waiting > 0. and at < horizon:
                segments.append(Segment(robot.robot_id, tid, at, min(at+waiting, horizon),
                                        tuple(route[0]), tuple(route[0]), start, anchored))
                at += waiting
            for a,b,length in zip(route,route[1:],lengths):
                duration=length/speed
                consumed=min(credit,duration);duration-=consumed;credit-=consumed
                if duration > 1e-9 and at < horizon:
                    segments.append(Segment(robot.robot_id,tid,at,at+duration,a,b,start))
                if a != b:
                    anchored = False
                at+=duration
            pose=tuple(route[-1])
            if finish > at+1e-9 and at < horizon:
                segments.append(Segment(robot.robot_id,tid,at,min(finish,horizon),pose,pose,start,anchored))
            cursor=finish
            if cursor >= horizon:
                break
        if cursor < horizon-1e-9:
            segments.append(Segment(robot.robot_id,last_task if terminal_occupancy_task else None,
                                    cursor,horizon,pose,pose,cursor,anchored))
    # Existing intervention timing is already charged in Delta_c; exclude its
    # relocation barrier from the anticipated-conflict scan.
    return [s for s in segments if s.end > window_start+1e-9
            and s.start < horizon and s.end > s.start+1e-9]


def estimate_future_delays(*, candidate, evaluation, robots, tasks, speed,
                           grid_size, robot_radius, horizon=5., deadline=None,
                           task_limit=None, wait_before_navigation=False,
                           robot_order=None, window_after_staging=False,
                           include_future_occupancy=False,
                           censored_policy='horizon_wait'):
    # Experimental scoring policy, not a feasibility relaxation. An unresolved
    # encounter has no predicted release time. Keep its exposure in diagnostics
    # instead of treating the end of the observation window as a release event.
    # The stable default and all live execution guards remain unchanged.
    if censored_policy not in {'horizon_wait', 'resolved_only', 'resolved_boundary'}:
        raise ValueError('Unknown censored delay policy')
    started=time.perf_counter()
    barrier=sum(float(s["delay"]) for s in candidate.get("relocations",()))
    window_end = barrier + horizon if window_after_staging else horizon
    if barrier >= window_end:
        return {},dict(horizon_s=horizon,status="outside_horizon",wall_time_s=0.,
                       delay_by_task={},observations=[],censored=False)
    positions=dict(candidate.get("final_positions",{}))
    segments=project_trajectories(evaluation=evaluation,robots=robots,tasks=tasks,
        positions=positions,speed=speed,grid_size=grid_size,horizon=window_end,window_start=barrier,
        task_limit=task_limit, wait_before_navigation=wait_before_navigation,
        terminal_occupancy_task=include_future_occupancy)
    table=ReservationTable(segments,1/speed,window_end,robot_order=robot_order)
    delays={}; observations=[]
    for robot_id in sorted({r.robot_id for r in robots}):
        shift=0.
        for segment in sorted((s for s in segments if s.robot==robot_id and
                               (s.moving or include_future_occupancy and not s.anchored and s.task is not None)),
                              key=lambda s:(s.start,s.end)):
            at=max(segment.start,barrier)+shift
            if at >= window_end-1e-9:
                break
            original_at=at
            clipped_start=max(segment.start,barrier)
            duration=segment.end-clipped_start
            blocked_by=set();censored=False
            last_blocked_at = None
            while at < window_end-1e-9:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("anticipated delay budget exhausted")
                moved=Segment(segment.robot,segment.task,at,at+duration,
                              segment.position(clipped_start),segment.last,segment.task_start+shift)
                blockers=table.blockers(moved,2*robot_radius/grid_size)
                if not blockers:
                    break
                last_blocked_at = at
                blocked_by.update(b.robot for b in blockers)
                # Advance to the latest occupied segment boundary, recheck all
                # reservations. Simultaneous blockers are not summed twice.
                release=min(window_end,max(b.end for b in blockers))
                if release <= at+1e-9:
                    release=min(window_end,at+1/speed)
                at=release
            if at >= window_end-1e-9 and blocked_by:
                censored=True
            if (censored_policy == 'resolved_boundary' and not censored
                    and last_blocked_at is not None):
                # A blocker segment's END is a conservative departure time,
                # not the first collision-free departure: the query robot may
                # reach the occupied location only partway along its edge.
                # Refine only an already-resolved encounter. Maintain a swept-
                # checked safe upper endpoint; never extrapolate past H or
                # infer feasibility for a censored encounter. Non-monotonic
                # reservations may hide an earlier safe interval; this is NOT
                # claimed to be globally earliest departure.
                low, high = last_blocked_at, at
                for _ in range(10):
                    if deadline is not None and time.perf_counter() >= deadline:
                        raise TimeoutError('anticipated delay budget exhausted')
                    middle = (low + high) / 2.
                    query = Segment(segment.robot, segment.task, middle, middle+duration,
                        segment.position(clipped_start), segment.last, segment.task_start+shift)
                    if table.blockers(query, 2*robot_radius/grid_size):
                        low = middle
                    else:
                        high = middle
                at = high
            observed_delta=max(0.,at-original_at)
            delta = 0. if censored and censored_policy == 'resolved_only' else observed_delta
            if observed_delta:
                if delta:
                    delays[segment.task]=delays.get(segment.task,0.)+delta
                    shift+=delta
                observations.append(dict(robot_id=robot_id,task_id=segment.task,
                    delay_s=delta,observed_delay_s=observed_delta,
                    blocked_by=sorted(blocked_by),censored=censored))
            if censored and censored_policy == 'resolved_only':
                # Do not forecast downstream motion past an unresolved contact.
                break
    return delays,dict(horizon_s=horizon,status="estimated",observations=observations,
        censored=any(o["censored"] for o in observations),
        delay_by_task=delays,wall_time_s=time.perf_counter()-started,
        reservation_segments=len(segments),model="continuous-swept-immutable-reservations-one-pass",
        window_start_s=barrier, window_end_s=window_end, task_limit=task_limit,
        wait_before_navigation=wait_before_navigation, censored_policy=censored_policy)
