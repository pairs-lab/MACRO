"""Adapt and predict EVERY ROS-feasible candidate before MARCO selection.

Grid search/default evaluator are unchanged; this T-junction adapter replaces
only candidate evaluation with the explicit ROS navigation execution contract.
"""
import os
import sys
import json
import time
import math
import traceback
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from simulations.ros2.map_adapter import PlannerSession, clean
from simulations.grid.app import GridSimulation
from simulations.grid.world import load_world
from core.coordination.planner import astar, shortest_path_tree_to_any, path_length
from simulations.ros2.marco_geometry import corridor_clear
from simulations.grid.app import _compute_conflict_plan
from simulations.ros2.three_robot_t.execution_prediction import load_profile, action_contract, digest, predict


def smooth_south_entry(start,target,res):
    """Cubic curve leaving ``start`` southward and arriving southward at ``target`` (about one cell between points)."""
    (x0,y0),(x1,y1)=start,target
    drop=abs(y1-y0)
    if y1>=y0 or drop<1:return None
    handle=.45*drop
    p=[(x0,y0),(x0,y0-handle),(x1,y1+handle),(x1,y1)]
    curve=[]
    for i in range(201):
        t=i/200;u=1-t
        curve.append((u**3*p[0][0]+3*u*u*t*p[1][0]+3*u*t*t*p[2][0]+t**3*p[3][0],
                      u**3*p[0][1]+3*u*u*t*p[1][1]+3*u*t*t*p[2][1]+t**3*p[3][1]))
    route=[tuple(start)]
    for c in curve[1:-1]:
        if math_distance(c,route[-1])>=1.0:route.append(c)
    route.append(tuple(target))
    return route


def adapt_candidate(abstract,sim,poses,res,goal_adjustments,turn_safe,geometry=None):
    if abstract.get('priority_robot') not in ('R1','R2') or not abstract.get('planned_paths'):
        raise RuntimeError('Candidate has no executable navigation priority/path contract')
    execution=deepcopy({k:abstract[k] for k in ('strategy','priority_robot','candidate_id',
        'protected_priority_corridor','relocation_sequence','planned_paths') if k in abstract})
    corridor_cells=abstract.get('protected_priority_corridor') or abstract['planned_paths'][abstract['priority_robot']]
    corridor=[[x*res,y*res] for x,y in corridor_cells]
    free=set(sim.world.navigation_cells)
    # Corridor clearance depends only on (corridor, cell): reuse it across relocation steps and candidates
    # that share a priority corridor (exactly the same sets, computed once per request).
    geometry=geometry if geometry is not None else {}
    corridor_key=tuple(map(tuple,corridor))
    clear_of_corridor=geometry.get(('clear',corridor_key))
    if clear_of_corridor is None:
        clear_of_corridor=geometry[('clear',corridor_key)]={c for c in free if corridor_clear([c[0]*res,c[1]*res],corridor)}
    def clear_with_margin(c,value):
        key=('margin',corridor_key,value,c)
        if key not in geometry:geometry[key]=corridor_clear([c[0]*res,c[1]*res],corridor,value)
        return geometry[key]
    # Disc navigation clearance alone is insufficient for a rotating
    # rectangular Waffle. Buffer holding goals for all yaw orientations.
    margin=.10;reserve=.02
    for robot in sim.robots:
        if robot.current_task and robot.robot_id not in execution['planned_paths']:
            execution['planned_paths'][robot.robot_id]=list(robot.path)
    positions={f'R{i}':tuple(round(v/res) for v in poses[f'tb{i}'][:2]) for i in range(1,4)}
    refinements=[]
    def arc_cells(origin, heading, turn, radius=.25, reverse=False):
        # A differential-drive quarter turn, tangent to the incoming
        # heading. Translation prevents the rear corner sweeping into
        # the north wall during an in-place turn at y=0.
        ox,oy=[v*res for v in origin];cells=[]
        for j in range(25):
            angle=j*math.pi/48
            forward=(-1 if reverse else 1)*radius*math.sin(angle)
            lateral=turn*radius*(1-math.cos(angle))
            x=ox+forward*math.cos(heading)-lateral*math.sin(heading)
            y=oy+forward*math.sin(heading)+lateral*math.cos(heading)
            # Keep sub-cell geometry for Nav2 curvature. Rounding the
            # arc to 2.5 cm creates stair steps and false sharp turns.
            c=(x/res,y/res)
            if not cells or cells[-1]!=c:cells.append(c)
        return cells
    def obstacles(robot):
        others=[(r,p) for r,p in positions.items() if r!=robot]
        return {c for c in free if any(((c[0]-p[0])*res)**2+((c[1]-p[1])*res)**2<(.56 if r=='R3' else .46)**2 for r,p in others)}
    for step in execution.get('relocation_sequence',[]):
        robot=step['robot_id'];endpoint=tuple(step['path'][-1])
        blocked=obstacles(robot)
        outside=clear_of_corridor-blocked
        goals={c for c in outside & turn_safe
               if clear_with_margin(c,margin+reserve)
               and abs(c[0]*res)<=.25+1e-8
               # The idle support must clear the turning arc, not only
               # the straight protected passage. Reserve the deeper bay.
               and (robot!='R3' or c[1]*res<=-.925+1e-8)
               and all(math_distance(c,p)*res>=.58 for r,p in positions.items() if r!=robot)}
        if endpoint not in goals:
            root=endpoint if endpoint in outside else positions[robot]
            space=outside if endpoint in outside else free
            parent,ends,_=shortest_path_tree_to_any(space,root,goals,
                dynamic_obstacles=blocked,path_count=1)
            if not ends:raise RuntimeError(f'No buffered holding pose for {robot}')
            target=ends[0];extension=[target]
            while extension[-1]!=root:extension.append(parent[extension[-1]])
            extension.reverse()
            step['path']=list(step['path'])+extension[1:]
        else:target=endpoint;extension=[endpoint]
        # Rebuild staging from the observed start against the other held bodies.
        # This is explicit ROS adaptation, not an extra MARCO candidate search.
        if robot in ('R1','R2') and abs(positions[robot][1]*res)<.08:
            heading=0. if robot=='R1' else math.pi
            entry=(round((-.25 if robot=='R1' else .25)/res),positions[robot][1])
            entry_path=astar(free,positions[robot],entry,dynamic_obstacles=blocked)
            radius=.06
            prefix=arc_cells(entry,heading,-1 if robot=='R1' else 1,radius,reverse=True)
            if not entry_path or any((round(c[0]),round(c[1])) not in free or (round(c[0]),round(c[1])) in blocked for c in prefix):
                raise RuntimeError(f'No collision-free staging turn for {robot}')
            step['entry_path']=entry_path
            tail=astar(free,tuple(round(v) for v in prefix[-1]),target,dynamic_obstacles=blocked)
            route=prefix+tail[1:] if tail else None
            refinements.append({'robot':robot,'staging_turn_arc':prefix,'radius_m':radius,
                                'reverse':True,'entry_path':entry_path})
        elif robot=='R3' and positions[robot][1]*res>-.65:
            # Leave the mouth in the existing southward heading before attempting a lateral turn near either
            # waiting robot. A single curve tangent to that heading keeps the property without a stop at the mouth;
            # if it is not collision-free (or DEMO_R3_ENTRY=two_stage) use the two-stage entry.
            smooth=smooth_south_entry(positions[robot],target,res)
            if os.environ.get('DEMO_R3_ENTRY','smooth')!='two_stage' and smooth and all(
                    (round(c[0]),round(c[1])) in free and (round(c[0]),round(c[1])) not in blocked for c in smooth):
                route=smooth
                refinements.append({'robot':robot,'single_smooth_entry':True,'points':len(smooth)})
            else:
                entry=(positions[robot][0],round(-.65/res))
                entry_path=astar(free,positions[robot],entry,dynamic_obstacles=blocked)
                route=astar(free,entry,target,dynamic_obstacles=blocked)
                if not entry_path:raise RuntimeError('R3 cannot enter the lower pocket')
                step['entry_path']=entry_path
                refinements.append({'robot':robot,'southward_entry_path':entry_path})
        else:route=astar(free,positions[robot],target,dynamic_obstacles=blocked)
        if not route:raise RuntimeError(f'Buffered staging blocked for {robot}')
        step.update(path=route,destination=target,distance=path_length(route,res))
        positions[robot]=target
        refinements.append({'robot':robot,'original_endpoint':endpoint,'buffered_endpoint':target,
                            'extension':extension,'body_aware_path':route})
    # Idle support robot stays parked. Its pose must constrain resumption.
    execution['resumption_segments']={}
    for robot,path in list(execution['planned_paths'].items()):
        if robot=='R3':execution['planned_paths'].pop(robot);continue
        if not path:continue
        blockers=[p for r,p in positions.items() if r=='R3' and r!=robot]
        blocked={c for c in free if any(((c[0]-p[0])*res)**2+((c[1]-p[1])*res)**2<(.56)**2 for p in blockers)}
        goal=tuple(path[-1])
        # Differential-drive turning needs a wide-pocket waypoint before
        # entering the one-robot throat. This map-specific ROS adaptation
        # is logged and excluded from abstract-prediction validation.
        if robot!=abstract['priority_robot'] and positions[robot][1]<round(-.3/res):
            waypoint=(positions[robot][0],round(-.25/res))
            first=astar(free,positions[robot],waypoint,dynamic_obstacles=blocked)
            arc=arc_cells(waypoint,math.pi/2,-1 if robot=='R1' else 1)
            if any((round(c[0]),round(c[1])) not in free or (round(c[0]),round(c[1])) in blocked for c in arc):
                raise RuntimeError(f'No collision-free resumption turn for {robot}')
            second=astar(free,tuple(round(v) for v in arc[-1]),goal,dynamic_obstacles=blocked)
            route=first+arc[1:]+second[1:] if first and second else None
            execution['resumption_segments'][robot]=[
                {'path':first,'controller':'CorridorFollow'},
                {'path':arc,'controller':'ResumeFollow'},
                {'path':second,'controller':'CorridorFollow'}]
            refinements.append({'robot':robot,'resumption_turning_waypoint':waypoint,
                                'resumption_turn_arc':arc,'radius_m':.25})
        else:route=astar(free,positions[robot],goal,dynamic_obstacles=blocked)
        if not route:raise RuntimeError(f'Resumption blocked for {robot}')
        execution['planned_paths'][robot]=route
    execution['grid_size']=res
    execution['goal_adjustments']=goal_adjustments
    execution['ros_execution']={'tracking_margin_m':margin,'endpoint_reserve_m':reserve,
        'refinements':refinements,'evaluated_before_selection':True,
        'idle_support_robot':'R3', 'scope':'Gazebo simulation, not physical experiment'}
    return execution


def plan(session, request):
    report = Path(os.environ['DEMO_REPORT_DIR'])
    start = time.perf_counter()
    audit, spans = session.map_data(report, .22)
    res = audit['map']['resolution']
    poses = request['poses']
    goal_x=float(os.environ.get('DEMO_GOAL_X','2.7'))
    payload = {'grid_size':res, 'robot_radius':.22, 'free_spans':spans,
        'robots':[{'id':f'R{i}', 'start_cell':[round(v/res) for v in poses[f'tb{i}'][:2]]}
                  for i in range(1,4)],
        'objects':[{'name':f'goal{i}', 'cell':[round(x/res),0],
                    'interaction_cells':[[round(x/res),0]], 'blocking':False}
                   for i,x in [(1,goal_x),(2,-goal_x)]],
        'tasks':[{'id':i,'candidate':[f'R{i}'],'action':'toggle_on',
                  'object':f'goal{i}','duration':0.,'node':[]} for i in (1,2)]}
    world_path=report/'scanned_world.json'
    geometry={**payload,'objects':[],'tasks':[]}
    world_path.write_text(json.dumps(geometry))
    navigable=load_world(world_path).navigation_cells
    goal_adjustments=[]
    for obj in payload['objects']:
        original=tuple(obj['cell'])
        snapped=min(navigable,key=lambda c:(((c[0]-original[0])**2+(c[1]-original[1])**2),c))
        if math_distance(original,snapped)*res>.15:
            raise RuntimeError('Measured map has no safe goal within 15cm of requested goal')
        obj['cell']=snapped;obj['interaction_cells']=[snapped]
        goal_adjustments.append({'name':obj['name'],'requested_cell':original,'safe_cell':snapped})
    world_path.write_text(json.dumps(payload))
    sim=GridSimulation(world_path,speed=.16/res,async_planning=False,
        selection_policy='dag_aware',release_prediction_model='dynamic_events_guarded')
    sim.planning_timeout_s=30.
    try:
        if os.environ.get('DEMO_CONTROLLER','hybrid')!='hybrid' or os.environ.get('DEMO_RESUME_POLICY','phase_specific')!='phase_specific':
            raise RuntimeError('Matched prediction requires hybrid/phase_specific execution')
        sim._start_ready_tasks()
        if request.get('preferred'):
            result={'grid_size':res,'paths':{r.robot_id:list(r.path) for r in sim.robots if r.current_task}}
            (report/'marco_preferred_paths.json').write_text(json.dumps(clean(result),indent=2))
            return {'request_s':time.perf_counter()-start}
        for r in sim.robots:
            if r.current_task:r.phase='moving'
        profile=load_profile()
        raw_free=set(sim.world.free_cells);free=set(sim.world.navigation_cells)
        extent=int(.30/res)+1
        offsets=[(x,y) for x in range(-extent,extent+1) for y in range(-extent,extent+1)
                 if (x*x+y*y)*res*res<=.30*.30]
        turn_safe={c for c in free if all((c[0]+x,c[1]+y) in raw_free for x,y in offsets)}
        evaluations=[];executions={};geometry={}
        def evaluate(candidates,stats,deadline,baseline_finishes,baseline_distance):
            phase_start=time.perf_counter();ranked=[];cache={}
            for candidate in candidates:
                if deadline is not None and time.perf_counter()>=deadline:
                    raise TimeoutError('ROS candidate evaluation budget exhausted')
                item={'candidate_id':candidate['candidate_id'],
                      'core_nominal_makespan_s':candidate['estimated_makespan']}
                try:
                    execution=adapt_candidate(candidate,sim,poses,res,goal_adjustments,turn_safe,geometry)
                    contract=action_contract(execution,profile)
                    key=digest(contract)
                    prediction=cache.get(key)
                    if prediction is None:
                        prediction=predict(contract,poses,deadline);cache[key]=prediction
                    execution.update(execution_contract=contract,execution_contract_sha256=key,
                                     timing_prediction=prediction)
                    candidate['core_nominal_estimated_makespan']=candidate['estimated_makespan']
                    candidate['estimated_makespan']=prediction['remaining_navigation_s']
                    candidate['estimated_remaining_DAG_makespan']=candidate['estimated_makespan']
                    candidate['evaluation_model']='ROS controller-inspired independent-goal timing'
                    distance=sum(path_length(a['path'],res) for a in contract['clearance'])
                    distance+=sum(path_length(a['path'],res) for parts in contract['resume'].values() for a in parts)
                    candidate['additional_distance']=max(0.,distance-baseline_distance)
                    candidate['total_distance']=distance;candidate['total_delay']=0.
                    executions[candidate['candidate_id']]=execution;ranked.append(candidate)
                    item.update(status='evaluated',predicted_ros_s=candidate['estimated_makespan'],
                                ros_travel_m=distance,contract_sha256=key)
                except (RuntimeError,ValueError) as error:
                    item.update(status='rejected',reason=str(error))
                evaluations.append(item)
            stats['ros_execution_evaluation']={'evaluated':len(ranked),'rejected':len(candidates)-len(ranked),
                                               'wall_time_s':time.perf_counter()-phase_start}
            if not ranked:raise RuntimeError('No ROS-executable candidate; no unmatched grid fallback')
            return ranked
        try:
            abstract=_compute_conflict_plan(cluster=sim.robots,dag_robots=sim.robots,
                tasks=sim.tasks,task_status=sim.task_status,objects=sim.objects,
                navigation_cells=sim.world.navigation_cells,robot_radius=sim.world.robot_radius,
                grid_size=res,speed_cells_per_second=sim.steps_per_second,
                selection_policy='dag_aware',planning_timeout_s=30.,
                release_prediction_model='dynamic_events_guarded',execution_candidate_refiner=evaluate)
        finally:
            (report/'ros_candidate_evaluations.json').write_text(json.dumps(clean(evaluations),indent=2))
        execution=executions[abstract['candidate_id']]
        (report/'marco_plan.json').write_text(json.dumps(clean(abstract),indent=2))
        (report/'marco_execution_plan.json').write_text(json.dumps(clean(execution),indent=2))
        print('Selected',abstract['candidate_id'],'sequence',[(s['robot_id'],s['destination']) for s in execution['relocation_sequence']],file=sys.stderr,flush=True)
        return {'request_s':time.perf_counter()-start}
    finally:sim.close()


def math_distance(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**.5


if __name__=='__main__':
    session=PlannerSession()
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('shutdown'):break
            with redirect_stdout(sys.stderr):timings=plan(session,request)
            reply={'ok':True,'timings':timings}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply={'ok':False,'error':str(error)}
        print(json.dumps(reply),flush=True)
