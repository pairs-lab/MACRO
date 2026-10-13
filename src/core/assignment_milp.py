"""Atomic MILP for precisely the fixed-order assignment space of assignment.py.

Navigation is compiled into endpoint-state arcs, not optimized by the MILP.
The oracle must obey the core contract: deterministic plan(robot, location,
start_position), with object prediction updated in stable topological order.
"""

from collections import defaultdict
from dataclasses import dataclass
import math
import time

from .assignment import (
    _assignment_search_metadata, _coalesce_owner_chains, _robot_value, _route, _task_location,
    evaluate_plan, validate_tasks,
)


def _record_node_count(model, stats):
    """An optimization rejected by a license has no NodeCount attribute.

    Metric collection must never replace the actual solver exception.
    """
    try:
        count = int(model.NodeCount)
    except Exception:
        return
    stats["expanded_nodes"] += count


@dataclass(frozen=True)
class NavigationArc:
    robot: str
    previous: tuple  # (previous task ID, endpoint), or (None, None)
    current: tuple   # (current task ID, endpoint)
    distance: float
    duration: float  # navigation + service


def compile_navigation(tasks, distance, robot_speed=1.0, service_time=None,
                       restrict_chain_transitions=False):
    """Enumerate all deterministic oracle transitions, without assigning tasks.

    Keeping endpoint state is essential: an interaction pose can depend on
    the robot's previous endpoint. A plain object-to-object distance matrix
    would silently give the MILP different navigation choices from BnB.
    """
    validate_tasks(tasks)
    metadata = _assignment_search_metadata(tasks, service_time)
    robots = list(dict.fromkeys(r for t in tasks for r in t["candidate"]))
    states = {r: [(None, None)] for r in robots}
    arcs = []
    # Optional native-backend preparation optimization. A group already has
    # one mandatory owner; never plan a transition that skips its intervening
    # tasks on that robot. MILP's default compiler remains unchanged.
    task_index = {t["id"]: i for i, t in enumerate(metadata["ordered_tasks"])}
    previous_required, next_required, eligible = {}, {}, {}
    if restrict_chain_transitions:
        for group in _coalesce_owner_chains(metadata):
            tids = group["task_ids"]
            for i, tid in enumerate(tids):
                previous_required[tid] = task_index[tids[i-1]] if i else -1
                next_required[tid] = task_index[tids[i+1]] if i+1 < len(tids) else len(tasks)
                eligible[tid] = group["candidates"]
    if hasattr(distance, "begin_prediction"):
        distance.begin_prediction()
    for task in metadata["ordered_tasks"]:
        for robot in task["candidate"]:
            if restrict_chain_transitions and robot not in eligible[task["id"]]:
                continue
            new_states = set()
            for previous in tuple(states[robot]):
                if restrict_chain_transitions:
                    previous_index = task_index[previous[0]] if previous[0] is not None else -1
                    if previous_index < previous_required[task["id"]]:
                        continue
                    if (previous[0] is not None
                            and next_required[previous[0]] < task_index[task["id"]]):
                        continue
                try:
                    cost, endpoint = _route(
                        distance, robot, _task_location(task), previous[1]
                    )
                except RuntimeError:
                    # BnB also excludes assignments whose navigation fails.
                    continue
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("Navigation requires finite nonnegative costs")
                if isinstance(endpoint, list):
                    endpoint = tuple(endpoint)
                current = (task["id"], endpoint)
                new_states.add(current)
                arcs.append(NavigationArc(
                    robot, previous, current, float(cost),
                    cost / _robot_value(robot_speed, robot, "robot_speed")
                    + metadata["service_durations"][task["id"]][robot],
                ))
            states[robot].extend(sorted(new_states, key=repr))
        if hasattr(distance, "finish_predicted_task"):
            distance.finish_predicted_task(task)
    return metadata, states, arcs


def require_gurobi():
    try:
        import gurobipy as gp
    except ImportError as error:
        raise RuntimeError(
            "Atomic MILP requires gurobipy and a valid Gurobi license. "
            "Install with: python -m pip install gurobipy"
        ) from error
    return gp


def optimize_atomic_milp(
    tasks, distance, robot_speed=1.0, service_time=None,
    time_limit=120.0, threads=1, seed=0, diagnostics=None,
    log_file=None, environment=None,
):
    """Minimize T, then D; report solved only after both optimality proofs.

    Model construction, oracle compilation, both optimization passes and
    common-evaluator validation all consume the same assignment budget.
    No return-to-depot, arbitrary task sequencing, anticipatory travel, or
    extra runtime collision constraints are introduced.
    """
    if time_limit <= 0 or threads < 1:
        raise ValueError("Positive time limit and threads required")
    started = time.perf_counter()
    gp = require_gurobi()
    metadata, states, arcs = compile_navigation(
        tasks, distance, robot_speed, service_time
    )
    stats = diagnostics if diagnostics is not None else {}
    stats.update(expanded_nodes=0, status="building", solved=False)
    with gp.Model("atomic_fixed_order_assignment", env=environment) as model:
        model.Params.OutputFlag = int(log_file is not None)
        if log_file is not None:
            model.Params.LogFile = str(log_file)
            model.Params.LogToConsole = 0
        model.Params.Threads = threads
        model.Params.Seed = seed
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.0
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        x = {(t["id"], r): model.addVar(vtype=gp.GRB.BINARY,
                                      name=f"x_{t['id']}_{r}")
             for t in tasks for r in t["candidate"]}
        y = [model.addVar(vtype=gp.GRB.BINARY, name=f"arc_{k}")
             for k in range(len(arcs))]
        finish = {t["id"]: model.addVar(lb=0, name=f"F_{t['id']}")
                  for t in tasks}
        makespan = model.addVar(lb=0, name="T")
        incoming, outgoing, assigned = defaultdict(list), defaultdict(list), defaultdict(list)
        for k, arc in enumerate(arcs):
            incoming[arc.robot, arc.current].append(y[k])
            outgoing[arc.robot, arc.previous].append(y[k])
            assigned[arc.current[0], arc.robot].append(y[k])
            # Travel starts AFTER both previous robot service and all DAG
            # predecessors, exactly as evaluate_plan's max(...) + travel.
            previous_finish = 0 if arc.previous[0] is None else finish[arc.previous[0]]
            model.addGenConstrIndicator(
                y[k], True, finish[arc.current[0]] >= previous_finish + arc.duration
            )
            task = metadata["task_by_id"][arc.current[0]]
            for parent in task["node"]:
                model.addGenConstrIndicator(
                    y[k], True, finish[task["id"]] >= finish[parent] + arc.duration
                )
        for robot, robot_states in states.items():
            for state_index, state in enumerate(robot_states):
                # An idle robot can go directly from source to sink.
                sink = model.addVar(vtype=gp.GRB.BINARY, name=f"sink_{robot}_{state_index}")
                model.addConstr(
                    gp.quicksum(outgoing[robot, state]) + sink
                    == (1 if state[0] is None else gp.quicksum(incoming[robot, state]))
                )
        for task in tasks:
            tid = task["id"]
            model.addConstr(gp.quicksum(x[tid, r] for r in task["candidate"]) == 1)
            model.addConstr(makespan >= finish[tid])
            for robot in task["candidate"]:
                model.addConstr(x[tid, robot] == gp.quicksum(assigned[tid, robot]))
        # Both existing BnB modes enforce these same-chain equalities. Atomic
        # branching does NOT permit handoff within a collapsed linear chain.
        groups = [chain["task_ids"] for chain in metadata["chains"]]
        owners = defaultdict(list)
        for task in tasks:
            if task["action"] in {"pick", "place"}:
                owners[task["object"].strip().casefold()].append(task["id"])
        groups.extend(owners.values())
        for group in groups:
            if len(group) < 2:
                continue
            for tid in group[1:]:
                for robot in sorted(set(metadata["task_by_id"][tid]["candidate"]) | set(metadata["task_by_id"][group[0]]["candidate"])):
                    model.addConstr(x.get((tid, robot), 0) == x.get((group[0], robot), 0))
        total_distance = gp.quicksum(a.distance * v for a, v in zip(arcs, y))

        def solve_phase(objective, label):
            remaining = time_limit - (time.perf_counter() - started)
            if remaining <= 0:
                stats["status"] = f"time_limit_before_{label}"
                return False
            model.Params.TimeLimit = remaining
            model.setObjective(objective, gp.GRB.MINIMIZE)
            try:
                model.optimize()
            finally:
                # Retain search work even if the outer assignment watchdog
                # interrupts Python as the solver returns from a timeout.
                _record_node_count(model, stats)
            stats["status"] = f"{label}:{model.Status}"
            return model.Status == gp.GRB.OPTIMAL

        first_optimal = solve_phase(makespan, "T")
        if not first_optimal:
            return {"solved": False, "assignment_search": stats}
        def selected_assignment():
            return {t["id"]: next(r for r in t["candidate"] if x[t["id"], r].X > 0.5)
                    for t in tasks}

        # A continuous solver F/T value can be slightly below the physical
        # recurrence within feasibility tolerances. Pinning that raw value
        # can exclude other assignments with EXACTLY the same evaluated T
        # in the distance pass. Canonicalize it through the shared evaluator;
        # do not add a positive allowed objective degradation.
        raw_t = float(model.ObjVal)
        optimal_t, _, _ = evaluate_plan(
            tasks, selected_assignment(), distance, robot_speed, service_time
        )
        if not math.isclose(raw_t, optimal_t, rel_tol=1e-8, abs_tol=1e-7):
            raise RuntimeError("MILP primary objective disagrees with the common BnB evaluator")
        stats["primary_model_T"] = raw_t
        stats["primary_evaluated_T"] = optimal_t
        model.addConstr(makespan <= optimal_t, name="fix_optimal_T")
        if not solve_phase(total_distance, "D"):
            return {"solved": False, "assignment_search": stats}
        assignment = selected_assignment()
        predicted_t, predicted_d, plans = evaluate_plan(
            tasks, assignment, distance, robot_speed, service_time
        )
        model_d = float(total_distance.getValue())
        if (not math.isclose(predicted_t, optimal_t, rel_tol=1e-8, abs_tol=1e-7)
                or not math.isclose(predicted_d, model_d, rel_tol=1e-8, abs_tol=1e-7)):
            raise RuntimeError("MILP objectives disagree with the common BnB evaluator")
        stats.update(status="optimal", solved=True)
        return dict(solved=True, makespan=predicted_t, total_distance=predicted_d,
                    assignment=assignment, robot_plans=plans, assignment_search=stats)
