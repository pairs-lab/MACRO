"""Nominal ROS approach preview using the grid executor's swept primitive.

Unlike a discrete grid tick, ROS needs a configurable braking/replanning
preview. This is not the candidate-conditioned DAG prediction horizon.
Idle robots keep a constant position and participate in every pair check.
"""
import math

from simulations.ros2.marco_geometry import ROBOT_RADIUS, remaining_route, route_position, swept_separation


def predicted_contacts(poses, routes, pairs, speeds, horizon_s=3., step_s=.1,
                       radius_m=ROBOT_RADIUS):
    if not all(math.isfinite(v) and v > 0 for v in (horizon_s, step_s, radius_m)):
        raise ValueError('Horizon, step and radius must be finite and positive')
    if horizon_s > 10 or math.ceil(horizon_s / step_s) > 1000:
        raise ValueError('Approach preview exceeds bounded sampling budget')
    names={r for pair in pairs for r in pair}
    remaining={r:remaining_route(poses[r],routes.get(r,[])) for r in names}
    velocities={r:float(speeds.get(r,0.)) for r in names}
    if any(not math.isfinite(v) or v < 0 for v in velocities.values()):
        raise ValueError('Speeds must be finite and nonnegative')
    before={r:route_position(remaining[r],0.) for r in names}
    hits={}
    previous_t=0.
    for index in range(1,math.ceil(horizon_s/step_s)+1):
        t=min(index*step_s,horizon_s)
        after={r:route_position(remaining[r],velocities[r]*t) for r in names}
        for a,b in pairs:
            distance=swept_separation(before[a],after[a],before[b],after[b])
            if distance <= 2*radius_m and (a,b) not in hits:
                hits[a,b]={'robots':[a,b],'predicted_separation_m':distance,
                           'interval_s':[previous_t,t]}
        before=after;previous_t=t
    return list(hits.values())
