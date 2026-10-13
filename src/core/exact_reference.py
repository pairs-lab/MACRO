"""Exact lower/upper-bound certification for small legacy8 problems.

This is NOT a complete joint continuous-time MAPF solver. Independent exhaustive
robot tours give a relaxation lower bound. If their joint execution is feasible,
that bound is attained and is an exact optimum of the compiled fixed-assignment
problem. Otherwise a PC-CBS incumbent supplies an upper bound only.

Distances and services use Q(sqrt(2)), not rounded ticks or float comparisons.
"""
from dataclasses import dataclass
from fractions import Fraction
import heapq
import itertools
import math
import time

from . import pc_cbs_legacy as legacy


@dataclass(frozen=True)
class ExactTime:
    rational: Fraction = Fraction(0)
    diagonal: Fraction = Fraction(0)

    def __add__(self, other):
        return ExactTime(self.rational + other.rational, self.diagonal + other.diagonal)

    def __sub__(self, other):
        return ExactTime(self.rational - other.rational, self.diagonal - other.diagonal)

    def __lt__(self, other):
        a, b = (self - other).rational, (self - other).diagonal
        if a == 0:
            return b < 0
        if b == 0 or (a < 0) == (b < 0):
            return a < 0
        # sqrt(2) is irrational: equality here cannot occur for nonzero a,b.
        return a*a < 2*b*b if a > 0 else a*a > 2*b*b

    def __float__(self):
        return float(self.rational) + float(self.diagonal)*math.sqrt(2)

    def as_dict(self):
        return dict(rational=str(self.rational), sqrt2_coefficient=str(self.diagonal))


ZERO = ExactTime()


def scalar(value):
    return ExactTime(Fraction(str(value)))


def edge_time(first, second, speed):
    unit = Fraction(1)/Fraction(str(speed))
    return ExactTime(diagonal=unit) if first[0] != second[0] and first[1] != second[1] else ExactTime(unit)


def independent_tour(problem, robot, check, stats):
    """Dijkstra over (cell, completed subset); external precedence relaxed.

    Internal precedence remains. First-ready input order is relaxed too. Idle
    motion before dispatch is permitted by the executor. Positive service and
    movement costs make earliest-arrival dominance exact in this relaxation.
    """
    tasks = [t for t in problem.tasks if t.robot == robot]
    index = {t.id:i for i,t in enumerate(tasks)}
    full = (1 << len(tasks))-1
    initial = (problem.starts[robot], 0)
    best, parent = {initial:ZERO}, {initial:None}
    queue, serial = [(ZERO, 0, initial)], itertools.count(1)
    while queue:
        check()
        cost, _, state = heapq.heappop(queue)
        if best[state] != cost:
            continue
        stats['expanded'] += 1
        cell, done = state
        if done == full:
            actions = []
            while parent[state] is not None:
                previous, action = parent[state]
                actions.append(action)
                state = previous
            actions.reverse()
            return cost, actions
        children = [((q,done), edge_time(cell,q,problem.speed), ('move',q))
                    for q,_ in legacy.neighbors(problem,cell)]
        for i, task in enumerate(tasks):
            if done & (1 << i) or cell not in task.goals:
                continue
            if any(p in index and not done & (1 << index[p]) for p in task.predecessors):
                continue
            children.append(((cell,done | (1 << i)), scalar(task.duration), ('service',task.id)))
        for child, duration, action in children:
            new = cost + duration
            if child not in best or new < best[child]:
                best[child] = new
                parent[child] = state, action
                heapq.heappush(queue,(new,next(serial),child))
    return None, None


def make_plan(problem, robot, actions):
    """Preserve exact event times alongside the legacy executable plan."""
    tasks = {t.id:t for t in problem.tasks}
    cell, at, dispatch = problem.starts[robot], ZERO, ZERO
    path, times = [cell], [at]
    starts, services, finishes = {}, {}, {}
    for kind, value in actions:
        if kind == 'move':
            at += edge_time(cell,value,problem.speed)
            cell = value
        else:
            starts[value], services[value] = dispatch, at
            at += scalar(tasks[value].duration)
            finishes[value] = at
            dispatch = at
        path.append(cell)
        times.append(at)
    exact = dict(path=path,times=times,starts=starts,services=services,finishes=finishes)
    plan = legacy.Plan(tuple(path),tuple(map(float,times)),
        {k:float(v) for k,v in starts.items()}, {k:float(v) for k,v in finishes.items()},
        {k:float(v) for k,v in services.items()})
    return exact, plan


def separation_squared(a,b,c,d):
    """Exact rational swept separation of simultaneous cell commits."""
    delta = tuple(x-y for x,y in zip(a,c))
    velocity = tuple(x-y-z+w for x,y,z,w in zip(b,a,d,c))
    norm = sum(v*v for v in velocity)
    fraction = max(Fraction(0),min(Fraction(1),-Fraction(sum(x*v for x,v in zip(delta,velocity)),norm))) if norm else Fraction(0)
    return sum((x+fraction*v)**2 for x,v in zip(delta,velocity))


def certify_joint(problem, exact):
    """Check original dispatch, precedence and discrete-pose swept safety.

    The production cell stays at an edge source until its completion. All
    exact completion events are checked, including simultaneous commits and
    terminal idle bodies. No epsilon is used for this proof check.
    """
    finishes = {tid:at for p in exact.values() for tid,at in p['finishes'].items()}
    horizon = scalar(problem.horizon)
    for robot, plan in exact.items():
        done = set()
        owned = [t for t in problem.tasks if t.robot == robot]
        for tid,start in sorted(plan['starts'].items(),key=lambda item:item[1]):
            expected = next((t for t in owned if t.id not in done and
                             all(not start < finishes[p] for p in t.predecessors)),None)
            if expected is None or expected.id != tid:
                return False, 'precedence_or_first_ready'
            done.add(tid)
        if horizon < plan['times'][-1]:
            return False, 'horizon'
    events = sorted({t for p in exact.values() for t in p['times']})
    positions = dict(problem.starts)
    pointers = {r:0 for r in exact}
    clearance = Fraction(str(problem.clearance))**2
    for at in events:
        proposal = dict(positions)
        for r,p in exact.items():
            i = pointers[r]
            while i+1 < len(p['times']) and not at < p['times'][i+1]:
                i += 1
            proposal[r] = p['path'][i]
            pointers[r] = i
        for a,b in itertools.combinations(sorted(exact),2):
            if separation_squared(positions[a],proposal[a],positions[b],proposal[b]) < clearance:
                return False, 'swept_collision'
        positions = proposal
    return True, None


def solve(problem, time_limit=30., incumbent_time_limit=0.):
    """Certify optimum when the exact relaxation is attained; else bound gap.

    A bound mismatch or timeout is explicitly NOT an optimal result. The
    incumbent budget is separate and optional, and cannot certify optimality.
    """
    legacy.task_order(problem)
    if time_limit <= 0 or not math.isfinite(time_limit) or incumbent_time_limit < 0 or not math.isfinite(incumbent_time_limit):
        raise ValueError('Invalid reference budgets')
    started = time.perf_counter()
    stats = dict(expanded=0,completed_robot_bounds=0)
    exact, plans, bounds = {}, {}, {}
    def check():
        if time.perf_counter()-started >= time_limit:
            raise legacy.SearchTimeout('reference_bound_timeout')
    status, reason = 'bounded', None
    try:
        for robot in sorted(problem.starts):
            cost, actions = independent_tour(problem,robot,check,stats)
            if actions is None:
                status, reason = 'infeasible', 'disconnected_relaxation'
                break
            bounds[robot] = cost
            exact[robot], plans[robot] = make_plan(problem,robot,actions)
            stats['completed_robot_bounds'] += 1
        else:
            feasible, reason = certify_joint(problem,exact)
            if feasible:
                legacy.validate(problem,plans)
                status = 'optimal'
    except legacy.SearchTimeout:
        status, reason = 'bound_timeout', 'reference_bound_timeout'
    bound_time = time.perf_counter()-started
    lower = max(bounds.values(),default=ZERO)
    upper, source, incumbent_stats = None, None, None
    if status == 'optimal':
        upper, source = float(lower), 'exact_relaxation_attained'
    else:
        plans = None
        if status != 'infeasible' and incumbent_time_limit > 0:
            incumbent = legacy.solve(problem,incumbent_time_limit,focal_weight=1.,warm_start=True)
            incumbent_stats = {k:v for k,v in incumbent.items() if k != 'plans'}
            if incumbent['plans'] is not None:
                legacy.validate(problem,incumbent['plans'])
                upper, plans, source = incumbent['makespan_s'], incumbent['plans'], 'pc_cbs_feasible_only'
    return dict(status=status,optimality_claim=status=='optimal',
        certificate='exact_Q_sqrt2_relaxation_attained' if status=='optimal' else None,
        certificate_scope='compiled_legacy8_fixed_assignment_original_dispatch',
        complete_joint_solver=False, unresolved_reason=reason,
        lower_bound_s=float(lower), lower_bound_exact=lower.as_dict(),
        lower_bound_complete=stats['completed_robot_bounds']==len(problem.starts),
        upper_bound_s=upper, gap_s=upper-float(lower) if upper is not None else None,
        per_robot_lower_bounds={r:cost.as_dict() for r,cost in bounds.items()},
        bound_wall_time_s=bound_time, planning_wall_time_s=time.perf_counter()-started,
        stats=stats,solution_source=source,incumbent=incumbent_stats,plans=plans)
