"""A-priori ROS execution contract and bounded controller-inspired prediction.

Not a Nav2/costmap or Gazebo physics clone. The approximation uses the actual
launched RPP settings and SDF wheel acceleration, never measured trial times.
The T-junction mission has two independent navigation goals and no service.
"""
import hashlib
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from simulations.ros2.marco_geometry import remaining_route, routes_are_disjoint, passage_is_clear

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_profile():
    runtime = Path(os.environ.get('DEMO_REPORT_DIR', str(HERE)))
    configs = [yaml.safe_load((runtime/f'capture_tb{i}.yaml').read_text()) for i in (1, 2, 3)]
    profiles = []
    for config in configs:
        controller = config['controller_server']['ros__parameters']
        profiles.append({'frequency_hz': controller['controller_frequency'],
            'controllers': {k: controller[k] for k in ('CorridorFollow', 'TurnFollow', 'ResumeFollow')},
            'checkers': {k: controller[k] for k in ('clearance_goal_checker', 'general_goal_checker')},
            'smoother': config['velocity_smoother']['ros__parameters']})
    if any(p != profiles[0] for p in profiles[1:]):
        raise RuntimeError('Prediction requires identical fleet controller profiles')
    model = Path(os.environ['MARCO_ROS_MODEL_PACKAGE'])/'models/turtlebot3_waffle/model.sdf'
    sdf = ET.parse(model)
    plugin = next(p for p in sdf.iter('plugin') if p.find('wheel_diameter') is not None)
    profile = {**profiles[0], 'wheel_radius_m': float(plugin.findtext('wheel_diameter'))/2,
        'wheel_separation_m': float(plugin.findtext('wheel_separation')),
        # Gazebo diff-drive limits circumferential wheel speed in m/s,
        # then divides by wheel radius when writing the joint velocity.
        'wheel_acceleration_m_s2': float(plugin.findtext('max_wheel_acceleration')),
        'dt_s': .05, 'max_action_s': 120., 'settle_s': .5,
        'handoff_s': 2/float(profiles[0]['frequency_hz']),
        'release_margin_m': .10, 'clearance_margin_m': .11,
        'settle_clock': 'ROS simulation time',
        'service_duration_s': 0., 'scope': 'three_robot_t navigation-only Gazebo pilot'}
    if min(profile[k] for k in ('wheel_radius_m', 'wheel_separation_m',
                                'wheel_acceleration_m_s2', 'frequency_hz')) <= 0:
        raise ValueError('Non-positive controller/physics limits')
    return profile


def action_contract(plan, profile):
    """The executor consumes these exact paths, plugins and goal checkers."""
    clearance = []
    for step in plan.get('relocation_sequence', []):
        robot = step['robot_id']
        if step.get('entry_path'):
            clearance.append(dict(robot=robot, kind='entry', path=step['entry_path'],
                controller='CorridorFollow', checker='clearance_goal_checker',
                settle=True, acceptance_m=.035))
        clearance.append(dict(robot=robot, kind='holding', path=step['path'],
            controller='CorridorFollow' if robot == 'R3' else 'TurnFollow',
            checker='clearance_goal_checker', settle=True,
            acceptance_m=.05 if robot == 'R3' else .035))
    resume = {}
    for robot, path in plan['planned_paths'].items():
        if not path:
            continue
        parts = plan.get('resumption_segments', {}).get(robot) or [
            dict(path=path, controller='CorridorFollow')]
        resume[robot] = [dict(robot=robot, kind='resume', path=part['path'],
            controller=part['controller'], settle=False,
            checker='clearance_goal_checker' if i < len(parts)-1 else 'general_goal_checker',
            acceptance_m=.035 if i < len(parts)-1 else .12)
            for i, part in enumerate(parts)]
    return dict(grid_size=plan['grid_size'], priority_robot=plan['priority_robot'],
        protected_priority_corridor=plan.get('protected_priority_corridor'),
        clearance=clearance, resume=resume, profile=profile,
        task_semantics='two independent navigation goals; idle R3; zero service')


def verify_contract(plan):
    contract = plan['execution_contract']
    if digest(contract) != plan['execution_contract_sha256']:
        raise RuntimeError('Execution contract hash mismatch')
    if digest(action_contract(plan, contract['profile'])) != digest(contract):
        raise RuntimeError('Paths/controllers changed after candidate evaluation')
    if digest(load_profile()) != digest(contract['profile']):
        raise RuntimeError('Launched controller/physics configuration changed')
    return contract


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value, low, high):
    return max(low, min(high, value))


def rollout(action, initial, profile, resolution, deadline=None, protected=None):
    """Approximate RPP geometry, curvature/approach regulation and wheel limits.

    Stop tolerance matches the action checker (holding may be cancelled by the
    executor earlier). Costmap collision exceptions and AMCL are not simulated.
    Trace times are relative to dispatch, including nominal action handoff.
    """
    p = profile['controllers'][action['controller']]
    if p.get('use_velocity_scaled_lookahead_dist') or p.get('use_cost_regulated_linear_velocity_scaling'):
        raise RuntimeError('Unsupported RPP configuration; refuse unmatched prediction')
    path = [[x*resolution, y*resolution] for x, y in action['path']]
    if not path or any(not math.isfinite(v) for point in path for v in point):
        raise ValueError('Invalid execution path')
    # Interpolate the identical polyline so nearest-pose pruning remains
    # meaningful even for sparse waypoint paths (no geometric alteration).
    dense = [path[0]]
    for a,b in zip(path,path[1:]):
        count=max(1,math.ceil(math.dist(a,b)/.025))
        dense.extend([[a[j]+(b[j]-a[j])*i/count for j in (0,1)]
                      for i in range(1,count+1)])
    path=dense
    cusps=[i for i in range(1,len(path)-1)
           if sum((path[i][j]-path[i-1][j])*(path[i+1][j]-path[i][j])
                  for j in (0,1)) < 0.]
    suffix = [0.]*len(path)
    for i in range(len(path)-2, -1, -1):
        suffix[i] = suffix[i+1]+math.dist(path[i], path[i+1])
    x, y, yaw = initial[:3]
    left = right = 0.
    dt = profile['dt_s']; t = profile['handoff_s']
    trace = [[t, x, y, yaw]]; nearest = 0
    radius = profile['wheel_radius_m']; separation = profile['wheel_separation_m']
    max_delta = profile['wheel_acceleration_m_s2']/radius*dt
    tolerance = (action['acceptance_m'] if action['kind']=='holding' else
                 min(profile['checkers'][action['checker']]['xy_goal_tolerance'], action['acceptance_m']))
    sm = profile['smoother']; command_v = command_w = 0.
    for iteration in range(math.ceil(profile['max_action_s']/dt)):
        if iteration % 20 == 0 and deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('ROS prediction budget exhausted')
        goal_reached=math.hypot(x-path[-1][0], y-path[-1][1]) <= tolerance
        passage_clear=(not goal_reached or action['kind']!='holding' or protected is None or
                       passage_is_clear([x,y,yaw],protected,profile['clearance_margin_m']))
        if goal_reached and passage_clear:
            break
        # Monotone local pruning avoids jumping to a later crossing on a path.
        end = min(len(path), nearest+40)
        nearest = min(range(nearest, end), key=lambda i: math.hypot(path[i][0]-x, path[i][1]-y))
        carrot = path[-1]
        lookahead = float(p['lookahead_dist'])
        if p['allow_reversing']:
            cusp=next((i for i in cusps if i>nearest),None)
            if cusp is not None:
                lookahead=min(lookahead,math.hypot(path[cusp][0]-x,path[cusp][1]-y))
        for i in range(nearest, len(path)):
            if math.hypot(path[i][0]-x, path[i][1]-y) >= lookahead:
                carrot = path[i]
                if i > nearest:
                    a, b = path[i-1], path[i]
                    dx, dy = b[0]-a[0], b[1]-a[1]
                    aa = dx*dx+dy*dy
                    bb = 2*((a[0]-x)*dx+(a[1]-y)*dy)
                    cc = (a[0]-x)**2+(a[1]-y)**2-lookahead**2
                    if aa and bb*bb-4*aa*cc >= 0:
                        f = _clamp((-bb+math.sqrt(bb*bb-4*aa*cc))/(2*aa), 0., 1.)
                        carrot = [a[0]+f*dx, a[1]+f*dy]
                break
        dx, dy = carrot[0]-x, carrot[1]-y
        bx = math.cos(yaw)*dx+math.sin(yaw)*dy
        by = -math.sin(yaw)*dx+math.cos(yaw)*dy
        curvature = 2*by/(bx*bx+by*by) if bx*bx+by*by > .001 else 0.
        angle = math.atan2(by, bx)
        v = float(p['desired_linear_vel'])
        if p['use_rotate_to_heading'] and abs(angle) > p['rotate_to_heading_min_angle']:
            v = 0.
            omega = math.copysign(min(p['rotate_to_heading_angular_vel'],
                math.sqrt(2*p['max_angular_accel']*abs(angle))), angle)
            current_omega = radius*(right-left)/separation
            omega = _clamp(omega, current_omega-p['max_angular_accel']*dt,
                           current_omega+p['max_angular_accel']*dt)
        else:
            if p['use_regulated_linear_velocity_scaling'] and abs(curvature) > 1e-9:
                v *= min(1., 1/(abs(curvature)*p['regulated_linear_scaling_min_radius']))
            v = max(v, p['regulated_linear_scaling_min_speed'])
            remaining = suffix[nearest]+math.dist((x, y), path[nearest])
            if remaining < p['approach_velocity_scaling_dist']:
                v = min(v, max(p['min_approach_linear_velocity'],
                    p['desired_linear_vel']*math.dist((x, y), path[-1])/p['approach_velocity_scaling_dist']))
            if p['allow_reversing'] and bx < 0:
                v = -v
            omega = v*curvature
        v = _clamp(v, sm['min_velocity'][0], sm['max_velocity'][0])
        omega = _clamp(omega, sm['min_velocity'][2], sm['max_velocity'][2])
        command_v += _clamp(v-command_v, sm['max_decel'][0]*dt, sm['max_accel'][0]*dt)
        command_w += _clamp(omega-command_w, sm['max_decel'][2]*dt, sm['max_accel'][2]*dt)
        v, omega = command_v, command_w
        wanted_left = (v-omega*separation/2)/radius
        wanted_right = (v+omega*separation/2)/radius
        left += _clamp(wanted_left-left, -max_delta, max_delta)
        right += _clamp(wanted_right-right, -max_delta, max_delta)
        v = radius*(left+right)/2; omega = radius*(right-left)/separation
        mid = yaw+omega*dt/2
        x += v*math.cos(mid)*dt; y += v*math.sin(mid)*dt
        yaw = _wrap(yaw+omega*dt); t += dt
        trace.append([t, x, y, yaw])
    else:
        raise RuntimeError('Controller-inspired rollout did not reach action goal within 120 s')
    if action['settle']:
        # Nominal braking duration; holding feedback/AMCL settling not cloned.
        t += max(abs(left), abs(right))*radius/profile['wheel_acceleration_m_s2']+profile['settle_s']
    trace.append([t, x, y, yaw])
    return dict(duration_s=t, final_pose=[x, y, yaw], trace=trace)


def predict(contract, poses, deadline=None):
    profile = contract['profile']; res = contract['grid_size']
    current = {f'R{i}': list(poses[f'tb{i}']) for i in (1, 2, 3)}
    t = 0.; clearance = []; actions = []
    protected_cells=contract['protected_priority_corridor'] or contract['resume'][contract['priority_robot']][0]['path']
    protected=[[x*res,y*res] for x,y in protected_cells]
    for action in contract['clearance']:
        result = rollout(action, current[action['robot']], profile, res, deadline,protected)
        current[action['robot']] = result['final_pose']
        t += result['duration_s']
        actions.append(dict(robot=action['robot'], kind=action['kind'],
            controller=action['controller'], duration_s=result['duration_s'], completion_s=t))
        if action['kind'] == 'holding':
            clearance.append(dict(robot='tb'+action['robot'][1:], predicted_completion_s=t))
    trajectories = {}; duration = {}
    for robot, parts in contract['resume'].items():
        trace = []; elapsed = 0.; state = current[robot]
        for action in parts:
            result = rollout(action, state, profile, res, deadline)
            trace += [[elapsed+row[0], *row[1:]] for row in result['trace']]
            elapsed += result['duration_s']; state = result['final_pose']
            actions.append(dict(robot=robot, kind='resume', controller=action['controller'],
                                duration_s=result['duration_s']))
        trajectories[robot] = trace; duration[robot] = elapsed
    priority = contract['priority_robot']; finishes = {priority: t+duration[priority]}
    paths = {r: [[x*res, y*res] for part in parts for x, y in part['path']]
             for r, parts in contract['resume'].items()}
    releases = {priority: t}
    for robot in paths:
        if robot == priority:
            continue
        waiting = remaining_route(current[robot], paths[robot])
        release = None
        # Same remaining-corridor gate as the executor, not a distance/v gate.
        for elapsed, x, y, yaw in trajectories[priority][::2]:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('ROS release prediction budget exhausted')
            if routes_are_disjoint(remaining_route([x, y, yaw], paths[priority]),
                                   waiting, profile['release_margin_m']):
                release = elapsed; break
        if release is None:
            raise RuntimeError('Predicted priority still blocks resumption at final pose')
        releases[robot] = t+release; finishes[robot] = t+release+duration[robot]
    return dict(model='controller-inspired buffered-path prediction v1 (not Nav2/Gazebo clone)',
        clearance=clearance, goal_completion_s={'tb'+r[1:]: v for r, v in finishes.items()},
        release_s={'tb'+r[1:]: v for r, v in releases.items()}, actions=actions,
        remaining_navigation_s=max(finishes.values()), profile_sha256=digest(profile),
        predictor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        excludes=['startup', 'planning', 'AMCL noise', 'costmap controller exceptions',
                  'contact/friction physics', 'holding feedback dynamics', 'recovery actions'],
        service_duration_s=0., fitted_to_execution_data=False)
