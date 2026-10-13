"""PC-CBS adaptation for fixed-order, single-robot service tasks.

Discrete cardinal moves, non-preemptive service, precedence before navigation,
and swept-disc collision constraints. This is not the original authors' code.
Search is bounded by an explicit time horizon; no continuous-runtime optimality
claim is made.
"""
from dataclasses import dataclass
from collections import deque
import heapq
import itertools
import math
import time


REVISION = "pc-cbs-service-cardinal-v1"
MOVES = ((-1, 0), (0, -1), (0, 1), (1, 0))


@dataclass(frozen=True)
class Task:
    id: int
    robot: str
    goals: frozenset
    duration: int
    predecessors: tuple = ()


@dataclass(frozen=True)
class Problem:
    cells: frozenset
    starts: dict
    tasks: tuple
    clearance: float
    horizon: int


@dataclass
class Plan:
    path: tuple
    starts: dict
    finishes: dict
    services: dict


class SearchTimeout(TimeoutError):
    pass


def separation(a0, a1, b0, b1):
    delta = (a0[0]-b0[0], a0[1]-b0[1])
    velocity = (a1[0]-a0[0]-b1[0]+b0[0], a1[1]-a0[1]-b1[1]+b0[1])
    norm = sum(v*v for v in velocity)
    t = max(0., min(1., -sum(d*v for d, v in zip(delta, velocity))/norm)) if norm else 0.
    return math.hypot(*(d+t*v for d, v in zip(delta, velocity)))


def task_order(problem):
    tasks = {t.id: t for t in problem.tasks}
    if len(tasks) != len(problem.tasks):
        raise ValueError("Duplicate task IDs")
    for t in problem.tasks:
        if t.robot not in problem.starts or not t.goals or not t.goals <= problem.cells:
            raise ValueError("Invalid robot or task goals")
        if type(t.duration) is not int or t.duration <= 0:
            raise ValueError("Service durations must be positive integer ticks")
        if any(p not in tasks for p in t.predecessors):
            raise ValueError("Unknown predecessor")
    if problem.horizon < 1 or not math.isfinite(problem.clearance) or problem.clearance <= 0:
        raise ValueError("Invalid horizon/clearance")
    if not set(problem.starts.values()) <= problem.cells:
        raise ValueError("Invalid starts")
    for a, b in itertools.combinations(problem.starts.values(), 2):
        if math.dist(a, b) < problem.clearance - 1e-9:
            raise ValueError("Initially overlapping robots")
    result, done = [], set()
    while len(result) < len(tasks):
        ready = [t for t in problem.tasks if t.id not in done and set(t.predecessors) <= done]
        if not ready:
            raise ValueError("Cyclic task graph")
        # Input order is the fixed per-robot task order, not a search decision.
        t = ready[0]
        result.append(t)
        done.add(t.id)
    for robot in problem.starts:
        sequence = [t for t in problem.tasks if t.robot == robot]
        for first, second in zip(sequence, sequence[1:]):
            if first.id not in second.predecessors:
                raise ValueError("Explicit per-robot sequence edges are required")
    return result


def propagate(problem, bounds, ordered):
    """Propagate start/finish intervals without assuming collision-free travel."""
    bounds = {k: list(v) for k, v in bounds.items()}
    successors = {t.id: [] for t in ordered}
    for t in ordered:
        for p in t.predecessors:
            successors[p].append(t.id)
        b = bounds[t.id]  # earliest start, latest start, earliest finish, latest finish
        b[0] = max([b[0]] + [bounds[p][2] for p in t.predecessors])
        b[2] = max(b[2], b[0] + t.duration)
    for t in reversed(ordered):
        b = bounds[t.id]
        b[3] = min([b[3]] + [bounds[s][1] for s in successors[t.id]])
        b[1] = min(b[1], b[3] - t.duration)
    if any(b[0] > b[1] or b[2] > b[3] for b in bounds.values()):
        return None
    return {k: tuple(v) for k, v in bounds.items()}


def distances(cells, goals, check):
    distance = {p: 0 for p in goals}
    queue = deque(sorted(goals))
    while queue:
        check()
        p = queue.popleft()
        for dx, dy in MOVES:
            q = p[0]+dx, p[1]+dy
            if q in cells and q not in distance:
                distance[q] = distance[p]+1
                queue.append(q)
    return distance


def point(plan, tick):
    return plan.path[min(tick, len(plan.path)-1)]


def low_level(problem, robot, bounds, constraints, tables, stats, check, others=()):
    sequence = [t for t in problem.tasks if t.robot == robot]
    constraints = frozenset(constraints)

    def allowed(tick, a, b):
        return (tick, a, b) not in constraints

    def goal_safe(p, tick):
        return all(a != p or b != p for t, a, b in constraints if t >= tick)

    # State = position, elapsed ticks, task index, phase, service ticks left.
    # Phase 0 has not dispatched navigation; phase 1 navigates; phase 2 services.
    initial = (problem.starts[robot], 0, 0, 0, 0)
    parents = {initial: None}
    events = {}
    serial = itertools.count()
    suffix = [0] * (len(sequence)+1)
    for i in range(len(sequence)-1, -1, -1):
        travel = 0
        if i+1 < len(sequence):
            travel = min((tables[sequence[i+1].id].get(p, math.inf)
                          for p in sequence[i].goals), default=math.inf)
        suffix[i] = sequence[i].duration + travel + suffix[i+1]

    def estimate(state):
        p, tick, i, phase, remaining = state
        if i == len(sequence):
            return tick
        task = sequence[i]
        travel = tables[task.id].get(p, math.inf) if phase != 2 else 0
        service = task.duration if phase != 2 else remaining
        tail = suffix[i] - task.duration
        return max(tick + travel + service + tail, bounds[task.id][2] + tail)

    queue = [(estimate(initial), 0, 0, next(serial), initial)]
    best_ties = {initial: (0, 0)}
    while queue:
        check()
        _, conflicts, moves, _, state = heapq.heappop(queue)
        if best_ties.get(state) != (conflicts, moves):
            continue
        stats["low_level_expanded"] += 1
        p, tick, i, phase, remaining = state
        if i == len(sequence) and goal_safe(p, tick):
            chain = []
            while state is not None:
                chain.append(state)
                state = parents[state]
            chain.reverse()
            path = [problem.starts[robot]]
            starts, finishes, services = {}, {}, {}
            for s in chain[1:]:
                if s[1] == len(path):
                    path.append(s[0])
                event = events.get(s)
                if event:
                    kind, task_id, at = event
                    {"start": starts, "finish": finishes, "service": services}[kind][task_id] = at
            return Plan(tuple(path), starts, finishes, services)
        if tick > problem.horizon or estimate(state) > problem.horizon:
            continue
        next_states = []
        if i < len(sequence):
            task = sequence[i]
            lo, hi, end_lo, end_hi = bounds[task.id]
            if phase == 0 and lo <= tick <= hi:
                next_states.append(((p, tick, i, 1, 0), ("start", task.id, tick)))
            if phase == 1 and p in task.goals and end_lo <= tick+task.duration <= end_hi:
                next_states.append(((p, tick, i, 2, task.duration), ("service", task.id, tick)))
            if phase == 2 and tick < problem.horizon and allowed(tick, p, p):
                if remaining == 1:
                    next_states.append(((p, tick+1, i+1, 0, 0), ("finish", task.id, tick+1)))
                else:
                    next_states.append(((p, tick+1, i, 2, remaining-1), None))
        if tick < problem.horizon and (i == len(sequence) or phase != 2):
            # Before dispatch, wait only: no anticipatory task navigation.
            choices = ((0, 0),) if i < len(sequence) and phase == 0 else ((0, 0),) + MOVES
            for dx, dy in choices:
                q = p[0]+dx, p[1]+dy
                if q in problem.cells and allowed(tick, p, q):
                    next_states.append(((q, tick+1, i, phase, 0), None))
        for child, event in next_states:
            nt = child[1]
            extra = sum(separation(p, child[0], point(other, tick), point(other, nt))
                        < problem.clearance-1e-9 for other in others) if nt > tick else 0
            ties = conflicts+extra, moves+int(p != child[0])
            if child in best_ties and best_ties[child] <= ties:
                continue
            best_ties[child] = ties
            parents[child] = state
            events[child] = event
            heapq.heappush(queue, (estimate(child), *ties, next(serial), child))
    return None


def conflict(problem, plans, ordered):
    for task in ordered:
        start = plans[task.robot].starts[task.id]
        ends = [plans[next(t.robot for t in ordered if t.id == p)].finishes[p]
                for p in task.predecessors]
        if ends and start < max(ends):
            return "precedence", task.id, max(ends)
    horizon = max(len(p.path)-1 for p in plans.values())
    for tick in range(horizon):
        for a, b in itertools.combinations(sorted(plans), 2):
            a0, a1 = point(plans[a], tick), point(plans[a], tick+1)
            b0, b1 = point(plans[b], tick), point(plans[b], tick+1)
            if separation(a0, a1, b0, b1) < problem.clearance-1e-9:
                return "collision", (a, (tick, a0, a1)), (b, (tick, b0, b1))
    return None


def validate(problem, plans):
    """Independent feasibility audit of paths, task order, service and clearance."""
    ordered = task_order(problem)
    if set(plans) != set(problem.starts):
        raise ValueError("Missing robot plan")
    for robot, plan in plans.items():
        if not plan.path or plan.path[0] != problem.starts[robot]:
            raise ValueError("Wrong initial pose")
        if len(plan.path)-1 > problem.horizon:
            raise ValueError("Horizon exceeded")
        for a, b in zip(plan.path, plan.path[1:]):
            if a not in problem.cells or b not in problem.cells or sum(abs(x-y) for x, y in zip(a, b)) > 1:
                raise ValueError("Invalid navigation edge")
        previous_finish = 0
        for task in (t for t in problem.tasks if t.robot == robot):
            start, service, finish = plan.starts[task.id], plan.services[task.id], plan.finishes[task.id]
            if not previous_finish <= start <= service or finish != service+task.duration:
                raise ValueError("Invalid task interval")
            if finish >= len(plan.path):
                raise ValueError("Task finishes outside path")
            if any(point(plan, t) != point(plan, previous_finish) for t in range(previous_finish, start+1)):
                raise ValueError("Anticipatory navigation")
            if point(plan, service) not in task.goals:
                raise ValueError("Wrong service pose")
            if any(point(plan, t) != point(plan, service) for t in range(service, finish+1)):
                raise ValueError("Preempted service")
            previous_finish = finish
    if conflict(problem, plans, ordered) is not None:
        raise ValueError("Precedence or swept-disc collision violation")
    return True


def solve(problem, time_limit=30.):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Positive finite time limit required")
    started = time.perf_counter()
    deadline = started+time_limit
    stats = {"expanded": 0, "low_level_expanded": 0, "precedence_splits": 0,
             "collision_splits": 0, "low_level_calls": 0, "cache_hits": 0}

    def check():
        if time.perf_counter() >= deadline:
            raise SearchTimeout("pc_cbs_time_limit")

    def result(status, plans=None):
        return {"status": status, "plans": plans, "stats": dict(stats),
                "planning_wall_time_s": time.perf_counter()-started,
                "makespan_ticks": max((v for p in plans.values() for v in p.finishes.values()), default=0) if plans else None,
                "plan_horizon_ticks": max(len(p.path)-1 for p in plans.values()) if plans else None,
                "optimality_claim": False, "revision": REVISION}

    try:
        ordered = task_order(problem)
        tables = {t.id: distances(problem.cells, t.goals, check) for t in ordered}
        cache = {}

        def plan_all(bounds, constraints):
            plans = {}
            # Long-workload-first changes only low-level tie guidance.
            robots = sorted(problem.starts, key=lambda r: (-sum(t.duration for t in ordered if t.robot == r), r))
            for robot in robots:
                check()
                ids = [t.id for t in problem.tasks if t.robot == robot]
                key = robot, tuple((i, bounds[i]) for i in ids), tuple(sorted(constraints[robot]))
                if key in cache:
                    stats["cache_hits"] += 1
                    plan = cache[key]
                else:
                    stats["low_level_calls"] += 1
                    plan = low_level(problem, robot, bounds, constraints[robot], tables, stats, check, tuple(plans.values()))
                    cache[key] = plan
                if plan is None:
                    return None
                plans[robot] = plan
            return plans

        bounds = propagate(problem, {t.id: (0, problem.horizon, t.duration, problem.horizon) for t in ordered}, ordered)
        if bounds is None:
            return result("infeasible_within_horizon")
        constraints = {r: frozenset() for r in problem.starts}
        plans = plan_all(bounds, constraints)
        if plans is None:
            return result("infeasible_within_horizon")
        serial = itertools.count()
        queue = []
        seen = set()

        def push(b, c, p):
            signature = tuple(sorted(b.items())), tuple((r, tuple(sorted(v))) for r, v in sorted(c.items()))
            if signature in seen:
                return
            seen.add(signature)
            heapq.heappush(queue, (max(len(v.path)-1 for v in p.values()), next(serial), b, c, p))

        push(bounds, constraints, plans)
        while queue:
            check()
            _, _, bounds, constraints, plans = heapq.heappop(queue)
            stats["expanded"] += 1
            violation = conflict(problem, plans, ordered)
            if violation is None:
                validate(problem, plans)
                return result("solved", plans)
            kind, a, b = violation
            children = []
            if kind == "precedence":
                stats["precedence_splits"] += 1
                task_id, split = a, min(b, bounds[a][1]+1)
                for side in ("early", "late"):
                    child = dict(bounds)
                    lo, hi, end_lo, end_hi = child[task_id]
                    child[task_id] = (lo, min(hi, split-1), end_lo, end_hi) if side == "early" else (max(lo, split), hi, end_lo, end_hi)
                    child = propagate(problem, child, ordered)
                    if child is not None and child != bounds:
                        children.append((child, constraints))
            else:
                stats["collision_splits"] += 1
                for robot, forbidden in (a, b):
                    child = dict(constraints)
                    child[robot] = child[robot] | {forbidden}
                    children.append((bounds, child))
            for child_bounds, child_constraints in children:
                p = plan_all(child_bounds, child_constraints)
                if p is not None:
                    push(child_bounds, child_constraints, p)
        return result("infeasible_within_horizon")
    except SearchTimeout:
        return result("timeout")
