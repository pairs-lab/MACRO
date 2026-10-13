"""Event-time CBS adapter for the production eight-neighbor executor.

Fixed assignment, original first-ready dispatch, exact Euclidean edge times,
atomic service and idle staging. Event-aligned safe holding intervals and
positive-edge reservations avoid repeatedly resolving a single timestamp.
This adaptation is NOT a complete/optimal continuous-time PC-CBS solver.
"""
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
import heapq
import itertools
import math
import time

from .pc_cbs import SearchTimeout, separation

REVISION = "pc-cbs-legacy8-event-v3"
EPS = 1e-9
MOVES = tuple((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y)


@dataclass(frozen=True)
class Task:
    id: int
    robot: str
    goals: frozenset
    duration: float
    predecessors: tuple = ()


@dataclass(frozen=True)
class Problem:
    cells: frozenset
    starts: dict
    tasks: tuple
    clearance: float
    horizon: float
    speed: float


@dataclass
class Plan:
    path: tuple
    times: tuple
    starts: dict
    finishes: dict
    services: dict


def point(plan, at):
    # Production robot.cell stays at the source until the edge completes.
    return plan.path[max(0, bisect_right(plan.times, at + EPS)-1)]


def neighbors(problem, p):
    for dx, dy in MOVES:
        q = p[0]+dx, p[1]+dy
        if q not in problem.cells:
            continue
        if dx and dy and ((p[0]+dx, p[1]) not in problem.cells or
                          (p[0], p[1]+dy) not in problem.cells):
            continue
        yield q, math.hypot(dx, dy)/problem.speed


def task_order(problem):
    if not all(math.isfinite(v) and v > 0 for v in
               (problem.speed, problem.clearance, problem.horizon)):
        raise ValueError("Invalid speed/clearance/horizon")
    tasks = {t.id: t for t in problem.tasks}
    if len(tasks) != len(problem.tasks):
        raise ValueError("Duplicate task IDs")
    for t in problem.tasks:
        if (t.robot not in problem.starts or not t.goals or not t.goals <= problem.cells
                or not math.isfinite(t.duration) or t.duration <= 0
                or any(p not in tasks for p in t.predecessors)):
            raise ValueError("Invalid task/assignment/service")
    if not set(problem.starts.values()) <= problem.cells:
        raise ValueError("Invalid starts")
    if any(math.dist(a,b) < problem.clearance-EPS for a,b in
           itertools.combinations(problem.starts.values(),2)):
        raise ValueError("Initially overlapping robots are unsupported")
    ordered, done = [], set()
    while len(done) < len(tasks):
        ready = [t for t in problem.tasks if t.id not in done and set(t.predecessors) <= done]
        if not ready:
            raise ValueError("Cyclic task graph")
        ordered.append(ready[0])
        done.add(ready[0].id)
    return ordered


def distances(problem, goals, check):
    table = {p: 0. for p in goals}
    queue = [(0., p) for p in sorted(goals)]
    heapq.heapify(queue)
    while queue:
        check()
        d,p = heapq.heappop(queue)
        if d > table[p]+EPS:
            continue
        for q,cost in neighbors(problem,p):
            nd = d+cost
            if nd < table.get(q,math.inf)-EPS:
                table[q] = nd
                heapq.heappush(queue,(nd,q))
    return table


def allowed(constraints, a, b, start, end):
    for kind,at,x,y in constraints:
        if kind == 'edge' and abs(at-end) <= EPS and a == x and b == y:
            return False
        if kind == 'vertex':
            # A stationary interval occupies its cell including both endpoints.
            occupied = a if at < end-EPS else b
            if start-EPS <= at <= end+EPS and occupied == x:
                return False
        if kind == 'positive_edge':
            # A transition crossing a required completion must be exactly the
            # anchored edge, not a wait/service or another navigation edge.
            if start < at-EPS and end >= at-EPS:
                if abs(end-at) > EPS or a != x or b != y:
                    return False
        if kind == 'guard':
            begin,source,clearance = x
            destination = y
            # The anchored robot holds source on [begin, at), then completes
            # source->destination at at. The other robot holds a until end.
            if max(start,begin) < min(end,at)-EPS and math.dist(a,source) < clearance-EPS:
                return False
            if begin+EPS < end < at-EPS and separation(a,b,source,source) < clearance-EPS:
                return False
            # At begin we do NOT assume a swept stationary anchor: its previous
            # edge could finish simultaneously. Only its endpoint is fixed.
            if abs(end-begin) <= EPS and math.dist(b,source) < clearance-EPS:
                return False
            if start < at-EPS <= end:
                other_end = b if abs(end-at) <= EPS else a
                if separation(a,other_end,source,destination) < clearance-EPS:
                    return False
            if abs(start-at) <= EPS and math.dist(a,destination) < clearance-EPS:
                return False
    return True


def positive_satisfied(plan, constraint):
    kind,at,a,b = constraint
    if kind != 'positive_edge':
        return True
    return any(abs(end-at) <= EPS and x == a and y == b
               for x,y,end in zip(plan.path,plan.path[1:],plan.times[1:]))


def split_collision(problem, constraints, violation):
    """Disjoint absent/present branches; propagate the forced edge's interval.

    No arbitrary window around the conflict is forbidden. The reservation is
    implied by a required edge with its exact legacy traversal duration.
    """
    robot,forbidden = next(entry for entry in violation if entry[1][0] == 'edge')
    _,at,source,destination = forbidden
    absent = dict(constraints)
    absent[robot] = absent[robot]|{forbidden}
    present = dict(constraints)
    present[robot] = present[robot]|{('positive_edge',at,source,destination)}
    begin = at-math.dist(source,destination)/problem.speed
    guard = ('guard',at,(begin,source,problem.clearance),destination)
    for other in present:
        if other != robot:
            present[other] = present[other]|{guard}
    return absent,present


def plan_reservations(problem, plan):
    """Exact held-cell/swept-event reservations of a fixed timed trajectory."""
    guards = {('guard',end,(start,a,problem.clearance),b)
              for a,b,start,end in zip(plan.path,plan.path[1:],plan.times,plan.times[1:])}
    guards.add(('guard',problem.horizon,(plan.times[-1],plan.path[-1],problem.clearance),plan.path[-1]))
    return frozenset(guards)


def park_idle(problem, plan, guards, check):
    """Warm-start staging after service; never changes a task finish time.

    Prefer a nearby open cell outside every task's interaction region. Without
    this, a prioritized seed can freeze an early-finishing robot on another
    robot's service pose forever. Failure simply retains the original plan.
    """
    source,at = plan.path[-1],plan.times[-1]
    goals = frozenset(p for t in problem.tasks for p in t.goals)
    queue = [(0.,source)]
    costs,parents = {source:0.},{source:None}
    for _ in range(1000):
        check()
        if not queue:
            break
        cost,p = heapq.heappop(queue)
        if cost > costs[p]+EPS:
            continue
        if (all(math.dist(p,g) >= problem.clearance+1. for g in goals)
                and len(tuple(neighbors(problem,p))) >= 6
                and allowed(guards,p,p,at+cost,problem.horizon)):
            chain = []
            q = p
            while q != source:
                chain.append(q); q = parents[q]
            chain.reverse()
            return Plan(plan.path+tuple(chain),plan.times+tuple(at+costs[q] for q in chain),
                        dict(plan.starts),dict(plan.finishes),dict(plan.services))
        for q,edge in neighbors(problem,p):
            end = cost+edge
            if end < costs.get(q,math.inf)-EPS and at+end <= problem.horizon:
                if allowed(guards,p,q,at+cost,at+end):
                    costs[q],parents[q] = end,p
                    heapq.heappush(queue,(end,q))
    return plan


def low_level(problem, robot, releases, constraints, tables, stats, check):
    tasks = [t for t in problem.tasks if t.robot == robot]
    owned = {t.id for t in tasks}
    all_done = frozenset(owned)
    by_id = {t.id:t for t in tasks}
    # position, time, completed IDs, active ID (or None).
    initial = problem.starts[robot], 0., frozenset(), None
    serial = itertools.count()
    parents, events = {initial:None}, {}
    index = {t.id:i for i,t in enumerate(tasks)}
    goal_distances = tuple(tuple(min((tables[b.id].get(p,math.inf) for p in a.goals),
                                    default=math.inf) for b in tasks) for a in tasks)
    external = {pred for t in tasks for pred in t.predecessors if pred not in owned}
    tail_labels = {}
    # Idle navigation between timed events uses earliest-arrival labels rather
    # than distinct states for every loop's accumulated floating-point time.
    # Edge-forbidden completion times also induce departure-time boundaries.
    timed_boundaries = {releases[pred] for pred in external}
    for kind,at,a,b in constraints:
        timed_boundaries.update((at,at+1/problem.speed))
        if kind in ('edge','positive_edge'):
            timed_boundaries.add(at-math.dist(a,b)/problem.speed)
        elif kind == 'guard':
            timed_boundaries.update((a[0],a[0]+1/problem.speed))
    # Arrivals and service finishes on either side of an event can have
    # different feasibility/readiness. Expose their departure/service times.
    endpoints = tuple(timed_boundaries)
    for at in endpoints:
        for duration in (1/problem.speed,math.sqrt(2)/problem.speed,
                         *(t.duration for t in tasks)):
            timed_boundaries.update((at-duration,at-duration+1/problem.speed))
    timed_boundaries = tuple(sorted(v for v in timed_boundaries if v >= 0.))

    @lru_cache(maxsize=None)
    def cell_boundaries(p):
        boundaries = set()
        for kind,at,a,b in constraints:
            if kind == 'vertex' and p == a:
                boundaries.update((at,at+1/problem.speed))
            elif kind == 'guard':
                begin,source,clearance = a
                if math.dist(p,source) < clearance-EPS:
                    boundaries.update((begin,at,at+1/problem.speed))
                elif separation(p,p,source,b) < clearance-EPS:
                    boundaries.update((at,at+1/problem.speed))
            elif kind == 'positive_edge':
                boundaries.add(at)
        return tuple(sorted(boundaries))

    @lru_cache(maxsize=None)
    def dispatch_boundaries(done):
        boundaries = {releases[pred] for t in tasks if t.id not in done
                      for pred in t.predecessors if pred in external}
        for at in tuple(boundaries):
            boundaries.update(at-t.duration for t in tasks if t.id not in done)
        return tuple(sorted(boundaries))

    def epoch(p,at,done):
        return (bisect_right(cell_boundaries(p),at+EPS),
                bisect_right(dispatch_boundaries(done),at+EPS))

    @lru_cache(maxsize=None)
    def route_tail(first, mask):
        """Relaxed open tour: goal-set distances, no release/collision constraints.

        Distances use independently chosen poses at each end, so this is a
        lower bound even when consecutive tasks share multiple interaction poses.
        For large assignments use an MST relaxation to avoid exponential DP.
        """
        check()
        if not mask:
            return 0.
        if len(tasks) > 12:
            nodes = [first]+[i for i in range(len(tasks)) if mask & (1 << i)]
            reached = {nodes[0]}
            total = 0.
            while len(reached) < len(nodes):
                check()
                cost,node = min((goal_distances[a][b],b) for a in reached
                                for b in nodes if b not in reached)
                total += cost
                reached.add(node)
            return total
        return min(goal_distances[first][i]+route_tail(i,mask ^ (1 << i))
                   for i in range(len(tasks)) if mask & (1 << i))

    @lru_cache(maxsize=None)
    def remaining_info(done):
        remaining = tuple(t for t in tasks if t.id not in done)
        mask = sum(1 << index[t.id] for t in remaining)
        # Once every relevant timed event has passed, earlier arrival dominates
        # later arrival at the same (cell, completed tasks, active task).
        return remaining,mask,sum(t.duration for t in remaining)

    def estimate(s):
        p,at,done,active = s
        remaining,mask,service = remaining_info(done)
        choices = (by_id[active],) if active is not None else remaining
        travel = min((tables[t.id].get(p,math.inf)+route_tail(index[t.id],mask ^ (1 << index[t.id]))
                      for t in choices),default=0.)
        bound = at+service+travel
        # A dependent task cannot finish before its external release plus its
        # own service; ignoring travel and other service remains admissible.
        bound = max([bound]+[releases[pred]+t.duration for t in remaining
                             for pred in t.predecessors if pred in external])
        return bound

    def dominated(state, travel, record=False):
        p,at,done,active = state
        # No pending task becomes newly ready inside an idle epoch. Labels
        # retain a Pareto frontier of arrival and distance; an earlier label
        # must also be able to hold its cell safely until the later arrival.
        signature = p,done,active,epoch(p,at,done)
        labels = tail_labels.setdefault(signature,[])
        if any(t <= at+EPS and d <= travel+EPS and (t < at-EPS or d < travel-EPS)
               and allowed(constraints,p,p,t,at) for t,d in labels):
            stats['dominance_pruned'] = stats.get('dominance_pruned',0)+1
            return True
        if record:
            labels[:] = [(t,d) for t,d in labels if not(at <= t+EPS and travel <= d+EPS)]
            labels.append((at,travel))
        return False

    def key(s):
        # Only collapse floating-point accumulation noise, never quantize edges.
        return s[0], round(s[1],12), s[2], s[3]

    queue = [(estimate(initial),0.,next(serial),initial)]
    seen = {key(initial):0.}
    dominated(initial,0.,record=True)
    while queue:
        check()
        _,travel,_,state = heapq.heappop(queue)
        if travel > seen.get(key(state),math.inf)+EPS:
            continue
        if dominated(state,travel):
            continue
        stats['low_level_expanded'] += 1
        p,at,done,active = state
        if at > problem.horizon+EPS or estimate(state) > problem.horizon+EPS:
            continue
        if (done == all_done and allowed(constraints,p,p,at,problem.horizon)
                and all(kind != 'positive_edge' or t < at+EPS for kind,t,x,y in constraints)):
            chain = []
            while state is not None:
                chain.append(state)
                state = parents[state]
            chain.reverse()
            path, times = [problem.starts[robot]], [0.]
            starts,services,finishes = {},{},{}
            for child in chain[1:]:
                if child[1] > times[-1]+EPS:
                    path.append(child[0]); times.append(child[1])
                event = events.get(child)
                if event:
                    kind,tid,start,end = event
                    if kind == 'start':
                        starts[tid] = start
                    else:
                        services[tid],finishes[tid] = start,end
            plan = Plan(tuple(path),tuple(times),starts,finishes,services)
            if not all(positive_satisfied(plan,c) for c in constraints):
                continue
            return plan
        children = []
        if active is None and done != all_done:
            # Exactly the executor's input-order first-ready rule, not a new
            # sequence of added DAG edges. Skip earlier blocked tasks.
            ready = next((t for t in tasks if t.id not in done and
                          all((pred in done if pred in owned else
                               releases[pred] <= at+EPS) for pred in t.predecessors)),None)
            if ready is not None:
                children.append(((p,at,done,ready.id),('start',ready.id,at,at),at))
        if not children:
            departures = (at,)+tuple(t for t in timed_boundaries if t > at+EPS)
            if active is not None:
                task = by_id[active]
                if p in task.goals:
                    accepted = set()
                    for departure in departures:
                        check()
                        end = departure+task.duration
                        group = epoch(p,end,done|{active})
                        if group in accepted:
                            continue
                        if not allowed(constraints,p,p,at,departure):
                            break
                        if allowed(constraints,p,p,departure,end):
                            accepted.add(group)
                            children.append(((p,end,done|{active},None),
                                             ('service',active,departure,end),departure))
            # Idle relocations are allowed, including before dependencies and
            # after completion. Service intervals remain non-preemptive.
            for q,cost in neighbors(problem,p):
                accepted = set()
                for departure in departures:
                    check()
                    end = departure+cost
                    group = epoch(q,end,done)
                    if group in accepted:
                        continue
                    if not allowed(constraints,p,p,at,departure):
                        break
                    if active is None and any(at+EPS < t <= departure+EPS for t in dispatch_boundaries(done)):
                        # Dispatch at a release event first; do not postpone it
                        # merely to create an idle navigation macro.
                        continue
                    if allowed(constraints,p,q,departure,end):
                        accepted.add(group)
                        children.append(((q,end,done,active),None,departure))
            wait_ends = {t for t in dispatch_boundaries(done) if t > at+EPS} if active is None else set()
            for end in sorted(wait_ends):
                if allowed(constraints,p,p,at,end):
                    children.append(((p,end,done,active),None,at))
        for child,event,departure in children:
            check()
            nt = travel+math.dist(p,child[0])
            k = key(child)
            if k in seen and seen[k] <= nt+EPS:
                continue
            if child[1] > problem.horizon+EPS or estimate(child) > problem.horizon+EPS:
                continue
            if dominated(child,nt,record=True):
                continue
            seen[k] = nt
            parent = state
            if departure > at+EPS:
                # Keep a real wait in the output trajectory. The synthetic
                # bridge is not searched and cannot erase a dispatch event.
                parent = (p,departure,done,active,('wait',next(serial)))
                parents[parent],events[parent] = state,None
            parents[child],events[child] = parent,event
            heapq.heappush(queue,(estimate(child),nt,next(serial),child))
    return None


def collision_count(problem, plans):
    """Focal tie guidance only; never replaces the final feasibility audit."""
    before = {r:p.path[0] for r,p in plans.items()}
    count = 0
    for at in sorted({t for p in plans.values() for t in p.times[1:]}):
        after = {r:point(p,at) for r,p in plans.items()}
        count += sum(separation(before[a],after[a],before[b],after[b]) < problem.clearance-EPS
                     for a,b in itertools.combinations(sorted(plans),2))
        before = after
    return count


def conflict(problem, plans):
    # Sweep precisely the source/destination cell transitions completed at each
    # event. Other robots retain their source cell, as in app._advance.
    event_times = sorted({at for p in plans.values() for at in p.times[1:]})
    before = {r:p.path[0] for r,p in plans.items()}
    for at in event_times:
        after = {r:point(p,at) for r,p in plans.items()}
        for a,b in itertools.combinations(sorted(plans),2):
            if separation(before[a],after[a],before[b],after[b]) < problem.clearance-EPS:
                entries = []
                for r in (a,b):
                    kind = 'edge' if before[r] != after[r] else 'vertex'
                    entries.append((r,(kind,at,before[r],after[r])))
                return tuple(entries)
        before = after
    return None


def validate(problem, plans):
    task_order(problem)
    if set(plans) != set(problem.starts):
        raise ValueError('Missing robot plan')
    finishes = {tid:at for p in plans.values() for tid,at in p.finishes.items()}
    for robot,plan in plans.items():
        owned = [t for t in problem.tasks if t.robot == robot]
        ids = {t.id for t in owned}
        if any(set(v) != ids for v in (plan.starts,plan.services,plan.finishes)):
            raise ValueError('Missing/extra task events')
        if (len(plan.times) != len(plan.path) or not plan.path or plan.times[0] != 0
                or plan.path[0] != problem.starts[robot] or plan.times[-1] > problem.horizon+EPS):
            raise ValueError('Invalid path/horizon')
        for a,b,start,end in zip(plan.path,plan.path[1:],plan.times,plan.times[1:]):
            if end <= start or a not in problem.cells or b not in problem.cells:
                raise ValueError('Invalid timed edge')
            if a != b and (b not in dict(neighbors(problem,a)) or
                           abs(end-start-math.dist(a,b)/problem.speed) > EPS):
                raise ValueError('Invalid edge duration/corner cutting')
        previous = 0.
        done = set()
        for task in sorted(owned,key=lambda t:plan.starts[t.id]):
            start,service,end = plan.starts[task.id],plan.services[task.id],plan.finishes[task.id]
            if not previous-EPS <= start <= service+EPS or abs(end-service-task.duration) > EPS:
                raise ValueError('Invalid service/robot ownership interval')
            if end > plan.times[-1]+EPS:
                raise ValueError('Service finishes outside timed path')
            if any(finishes[p] > start+EPS for p in task.predecessors):
                raise ValueError('Precedence before navigation violated')
            expected = next((t for t in owned if t.id not in done and
                             all(finishes[p] <= start+EPS for p in t.predecessors)),None)
            if expected is None or expected.id != task.id:
                raise ValueError('Input-order first-ready dispatch violated')
            pose = point(plan,service)
            if pose not in task.goals or point(plan,end) != pose or any(
                    point(plan,t) != pose for t in plan.times if service-EPS <= t <= end+EPS):
                raise ValueError('Wrong goal or preempted service')
            if any(a != b and edge_start < end-EPS and edge_end > service+EPS
                   for a,b,edge_start,edge_end in zip(plan.path,plan.path[1:],plan.times,plan.times[1:])):
                raise ValueError('Navigation overlaps atomic service')
            previous = end
            done.add(task.id)
    if conflict(problem,plans):
        raise ValueError('Swept-disc collision violation')
    return True


def solve(problem, time_limit=30., focal_weight=1.25, warm_start=True):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('Positive finite budget required')
    if not math.isfinite(focal_weight) or focal_weight < 1.:
        raise ValueError('Focal weight must be finite and >= 1')
    started = time.perf_counter()
    stats = dict(expanded=0,low_level_expanded=0,collision_splits=0,
                 low_level_calls=0,dispatch_iterations=0,cache_hits=0,dominance_pruned=0,
                 positive_branches=0,interval_reservations=0,warm_start_attempts=0)
    incumbent = None
    incumbent_source = None

    def check():
        if time.perf_counter()-started >= time_limit:
            raise SearchTimeout('pc_cbs_time_limit')

    def result(status,plans=None):
        return dict(status=status,plans=plans,stats=stats.copy(),revision=REVISION,
                    planning_wall_time_s=time.perf_counter()-started,
                    makespan_s=max((v for p in plans.values() for v in p.finishes.values()),default=0.) if plans else None,
                    focal_weight=focal_weight,optimality_claim=False,
                    suboptimality_bound_claim=False,solution_source=incumbent_source,
                    warm_start=bool(warm_start))

    try:
        ordered = task_order(problem)
        tables = {t.id:distances(problem,t.goals,check) for t in ordered}
        initial_releases = {}
        for t in ordered:
            initial_releases[t.id] = max((initial_releases[p] for p in t.predecessors),default=0.)+t.duration
        cache = {}
        external_by_robot = {r:tuple(sorted({pred for t in ordered if t.robot == r
            for pred in t.predecessors if next(v.robot for v in ordered if v.id == pred) != r}))
            for r in problem.starts}

        def plan_all(constraints):
            releases = dict(initial_releases)
            seen_releases = set()
            while True:
                check()
                stats['dispatch_iterations'] += 1
                signature = tuple((k,round(v,12)) for k,v in sorted(releases.items()))
                if signature in seen_releases:
                    return None  # Adapted dispatch fixed point failed; NOT infeasible.
                seen_releases.add(signature)
                plans = {}
                for r in sorted(problem.starts):
                    check()
                    # Own-task releases do not enter readiness checks; unrelated
                    # robot finishes must not invalidate a cached subproblem.
                    key = r,tuple((pred,releases[pred]) for pred in external_by_robot[r]),constraints[r]
                    if key in cache:
                        stats['cache_hits'] += 1
                        p = cache[key]
                    else:
                        stats['low_level_calls'] += 1
                        p = low_level(problem,r,releases,constraints[r],tables,stats,check)
                        cache[key] = p
                    if p is None:
                        return None
                    plans[r] = p
                updated = {k:v for p in plans.values() for k,v in p.finishes.items()}
                if all(abs(updated[k]-releases[k]) <= EPS for k in updated):
                    return plans
                releases = updated

        constraints = {r:frozenset() for r in problem.starts}
        plans = plan_all(constraints)
        if plans is None:
            return result('no_plan_found_within_adapter')
        if warm_start and conflict(problem,plans) is not None:
            # Prioritized reservations supply a FEASIBLE incumbent only. They
            # do not replace CBS constraints or impose an order on CBS search.
            # Its cost/time are included in the same planning budget.
            seed_deadline = min(started+time_limit,time.perf_counter()+min(3.,time_limit*.2))

            def seed_check():
                check()
                if time.perf_counter() >= seed_deadline:
                    raise SearchTimeout('warm_start_budget')

            try:
                robot_order = sorted(problem.starts,key=lambda r:(-sum(t.duration for t in ordered if t.robot == r),r))
                orders = itertools.islice(itertools.permutations(robot_order),6)
                for robot_order in orders:
                    seed_check()
                    stats['warm_start_attempts'] += 1
                    releases = {k:v for p in plans.values() for k,v in p.finishes.items()}
                    for _ in range(max(2,len(problem.tasks))):
                        seed_check()
                        seeded,guards = {},frozenset()
                        for robot in robot_order:
                            stats['low_level_calls'] += 1
                            p = low_level(problem,robot,releases,guards,tables,stats,seed_check)
                            if p is None:
                                break
                            if robot != robot_order[-1]:
                                p = park_idle(problem,p,guards,seed_check)
                            seeded[robot] = p
                            guards = guards|plan_reservations(problem,p)
                        if len(seeded) != len(problem.starts):
                            break
                        updated = {k:v for p in seeded.values() for k,v in p.finishes.items()}
                        if all(abs(updated[k]-releases[k]) <= EPS for k in updated):
                            try:
                                validate(problem,seeded)
                            except ValueError:
                                break
                            if (incumbent is None or max(updated.values(),default=0.) <
                                    max(v for p in incumbent.values() for v in p.finishes.values())):
                                incumbent,incumbent_source = seeded,'prioritized_warm_start'
                            break
                        releases = updated
            except SearchTimeout:
                check()  # A seed cutoff is not the overall solver cutoff.
        queue,serial,seen = [],itertools.count(),set()

        def push(c,p):
            signature = tuple((r,tuple(sorted(v))) for r,v in sorted(c.items()))
            if signature not in seen:
                seen.add(signature)
                heapq.heappush(queue,(max((v for plan in p.values() for v in plan.finishes.values()),default=0.),
                                      collision_count(problem,p),next(serial),c,p))

        push(constraints,plans)
        while queue:
            check()
            # Within a stated focal window, prefer fewer remaining collisions.
            # w=1 restores best-makespan order. The adapter's low-level search
            # lacks an optimality proof: w is NOT a certified solution bound.
            ceiling = queue[0][0]*focal_weight+EPS
            chosen = min((i for i,v in enumerate(queue) if v[0] <= ceiling),
                         key=lambda i:(queue[i][1],queue[i][0],queue[i][2]))
            node = queue[chosen]
            queue[chosen] = queue[-1]
            queue.pop()
            if queue:
                heapq.heapify(queue)
            _,_,_,constraints,plans = node
            stats['expanded'] += 1
            violation = conflict(problem,plans)
            if violation is None:
                validate(problem,plans)
                if incumbent is not None:
                    cost = max((v for p in plans.values() for v in p.finishes.values()),default=0.)
                    old_cost = max((v for p in incumbent.values() for v in p.finishes.values()),default=0.)
                    if old_cost < cost:
                        return result('solved',incumbent)
                incumbent_source = 'cbs'
                return result('solved',plans)
            stats['collision_splits'] += 1
            stats['positive_branches'] += 1
            stats['interval_reservations'] += len(problem.starts)-1
            for child in split_collision(problem,constraints,violation):
                p = plan_all(child)
                if p is not None:
                    push(child,p)
        return result('feasible_search_exhausted',incumbent) if incumbent else result('no_plan_found_within_adapter')
    except SearchTimeout:
        return result('feasible_time_limit',incumbent) if incumbent else result('timeout')
