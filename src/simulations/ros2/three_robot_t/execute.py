"""Execute the measured-map three-robot intervention via Nav2 actions.

Bounded, fail-closed experimental coordinator; no physical robot support.
"""
import json
import math
import os
import sys
import threading
import time
from itertools import combinations
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0,str(HERE.parent))
from simulations.ros2.navigation import Audit, pose
from simulations.ros2.marco_geometry import padded_bounds_overlap, passage_is_clear
from simulations.ros2.planner_worker import PlannerWorker
from simulations.ros2.marco_geometry import ROBOT_RADIUS, swept_separation, remaining_route, routes_are_disjoint
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter, parameter_value_to_python
from rcl_interfaces.srv import GetParameters
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from visualization_msgs.msg import MarkerArray
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from simulations.ros2.three_robot_t.timing_bounds import BOUND_RADIUS, publish_bounds
from simulations.ros2.three_robot_t.execution_prediction import verify_contract, digest, load_profile
from simulations.ros2.three_robot_t.swept_trigger import predicted_contacts


def main():
    if os.environ.get('ROS_DOMAIN_ID')!='73':raise RuntimeError('Requires isolated ROS domain 73')
    report=Path(os.environ['DEMO_REPORT_DIR'])
    robots=('tb1','tb2','tb3');pairs=list(combinations(robots,2))
    rclpy.init();audit=Audit(robots);nav={r:BasicNavigator(namespace=r) for r in robots}
    bound_pub=audit.create_publisher(MarkerArray,'/marco/conflict_bounds',10)
    bound_timer=audit.create_timer(.1,lambda:publish_bounds(audit,bound_pub))
    speed_pubs={r:audit.create_publisher(SpeedLimit,f'/{r}/speed_limit',10) for r in robots}
    costmaps={}
    costmap_subs=[audit.create_subscription(OccupancyGrid,f'/{r}/local_costmap/costmap',
        lambda m,r=r:costmaps.update({r:m}),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)) for r in robots]
    def capture_costmaps(label):
        grids={}
        for r,m in list(costmaps.items()):
            origin=m.info.origin.position
            try:
                tf=audit.buffers[r].lookup_transform(m.header.frame_id,'base_footprint',Time())
                q=tf.transform.rotation;p=tf.transform.translation
                actual=[p.x,p.y,math.atan2(2*q.w*q.z,1-2*q.z*q.z)]
            except Exception:actual=None
            grids[r]={'frame':m.header.frame_id,'robot_pose_in_grid_frame':actual,
                      'resolution':m.info.resolution,'width':m.info.width,'height':m.info.height,
                      'origin':[origin.x,origin.y],'data':list(m.data)}
        (report/f'costmaps_{label}.json').write_text(json.dumps(grids))
    def speed_limit(value):
        for pub in speed_pubs.values():
            m=SpeedLimit();m.header.stamp=audit.get_clock().now().to_msg()
            m.percentage=False;m.speed_limit=value;pub.publish(m)
    for n in nav.values():n.set_parameters([Parameter('use_sim_time',value=True)])
    executor=SingleThreadedExecutor();executor.add_node(audit)
    receiver=threading.Thread(target=executor.spin,daemon=True);receiver.start()
    record={'success':False,'scope':'Gazebo + Nav2 + AMCL on measured SLAM map',
        'physical_robot_experiment':False,'support_robot':'tb3 (idle task, physically relocated)',
        'controller':os.environ.get('DEMO_CONTROLLER','hybrid'),
        'resume_policy':os.environ.get('DEMO_RESUME_POLICY','phase_specific'),
        'gazebo_contact_profile':'original SDF (unmodified contact physics)',
        'stages':[],'min_swept_separation_m':None,'positions':[],
        'prediction_contract_deviations':[]}
    trigger_horizon=float(os.environ.get('DEMO_TRIGGER_HORIZON_S','3.0'))
    record['display_bound_radius_m']=BOUND_RADIUS
    record['conflict_trigger']={'model':'nominal simultaneous swept route segments, all three robots',
        'horizon_s':trigger_horizon,'sampling_step_s':.1,'robot_radius_m':ROBOT_RADIUS,
        'centre_clearance_m':2*ROBOT_RADIUS,'moving_speed_m_s':.16,
        'idle_speed_m_s':0.,'independent_of_display_radius':True}
    worker=None;start=time.monotonic();held=set();last=None;protected=[];execution_epoch=None
    def event(stage,**kwargs):
        e={'stage':stage,'wall_s':time.monotonic()-start,
           'ros_s':audit.get_clock().now().nanoseconds/1e9,**kwargs}
        record['stages'].append(e);print(json.dumps(e),flush=True)
    def guard():
        nonlocal last
        if any(time.monotonic()-audit.pose_times.get(r,0)>1.5 for r in robots):
            raise RuntimeError('Stale map-frame TF')
        current={r:audit.poses[r][:2] for r in robots}
        for a,b in pairs:
            d=swept_separation((last or current)[a],current[a],(last or current)[b],current[b])
            record['min_swept_separation_m']=min(record['min_swept_separation_m'] or d,d)
            if d<=.44 or padded_bounds_overlap(audit.poses[a],audit.poses[b]):
                raise RuntimeError(f'Live swept/footprint guard violated: {a}/{b}, {d:.3f}m')
        for r in held:
            if not passage_is_clear(audit.poses[r],protected,.08):
                raise RuntimeError(f'{r} lost protected passage clearance')
            v,w=audit.twists.get(r,(0.,0.))
            if abs(w)>=.03 and not staging_idle_safe(r,v,w):
                raise RuntimeError(f'{r} lost all-yaw holding envelope: v={v:.4f}, w={w:.4f}')
        last=current
        record['positions'].append({'wall_s':time.monotonic()-start,'poses':{r:audit.poses[r][:] for r in robots}})
    def pump():audit.pump();guard()
    def send(r,cells,stage,controller_override=None,precise_goal=False,contract_action=None):
        path=audit.publish(r,stage,[[x*res,y*res] for x,y in cells])
        checker='clearance_goal_checker' if stage=='clearance' or precise_goal else 'general_goal_checker'
        audit.holding.discard(r)
        controller='CorridorFollow' if stage=='resume' and os.environ.get('DEMO_CONTROLLER','hybrid')=='hybrid' else 'FollowPath'
        if stage=='resume' and execution_epoch is not None and os.environ.get('DEMO_CONTROLLER','hybrid')=='hybrid':
            if os.environ.get('DEMO_RESUME_POLICY','phase_specific')=='legacy_rpp':
                controller='ResumeFollow'
            else:controller='CorridorFollow' if r==priority_name else 'FollowPath'
        if stage=='clearance' and r in ('tb1','tb2') and os.environ.get('DEMO_CONTROLLER','hybrid')=='hybrid':
            controller='TurnFollow'
        if stage=='clearance' and r=='tb3' and os.environ.get('DEMO_CONTROLLER','hybrid')=='hybrid':
            controller='CorridorFollow'
        controller=controller_override or controller
        if contract_action is not None:
            if digest(contract)!=record['execution_contract_sha256']:
                raise RuntimeError('Execution contract mutated after selection')
            if digest(cells)!=digest(contract_action['path']):
                raise RuntimeError('Attempt to execute a path different from the scored candidate')
            controller=contract_action['controller'];checker=contract_action['checker']
        if not nav[r].followPath(path,controller_id=controller,goal_checker_id=checker):
            raise RuntimeError(f'FollowPath rejected for {r}')
        event(stage+'_started',robot=r,controller=controller,path_points=len(cells),goal_m=[x*res for x in cells[-1]])
    def verify_live_profile(profile):
        """Read actual ROS node parameters, not just the generated YAML."""
        values={}
        for robot in robots:
            controller_expected={'controller_frequency':profile['frequency_hz']}
            for plugin,params in {**profile['controllers'],**profile['checkers']}.items():
                controller_expected.update({f'{plugin}.{key}':value for key,value in params.items()})
            for node,expected in [('controller_server',controller_expected),
                                  ('velocity_smoother',profile['smoother'])]:
                client=audit.create_client(GetParameters,f'/{robot}/{node}/get_parameters')
                try:
                    if not client.wait_for_service(timeout_sec=5.):
                        raise RuntimeError(f'{robot}/{node} parameters unavailable')
                    request=GetParameters.Request();request.names=list(expected)
                    future=client.call_async(request);end=time.monotonic()+5
                    while not future.done():
                        audit.pump()
                        if time.monotonic()>end:raise TimeoutError('Live profile verification timeout')
                    actual={name:parameter_value_to_python(value)
                            for name,value in zip(request.names,future.result().values)}
                    for name,value in expected.items():
                        if actual.get(name)!=value:
                            raise RuntimeError(f'Unmatched controller parameter {robot}/{node}/{name}')
                    values[f'{robot}/{node}']=actual
                finally:audit.destroy_client(client)
        record['live_controller_profile']={'verified':True,'profile_sha256':digest(profile),'values':values}
    try:
        audit.holding.update(robots)
        worker=PlannerWorker(HERE.parent,report,.22,HERE/'planner.py')
        deadline=time.monotonic()+45
        while len(audit.odom)<3 or any(n.get_clock().now().nanoseconds==0 for n in nav.values()):
            for n in nav.values():rclpy.spin_once(n,timeout_sec=.02)
            audit.pump()
            if time.monotonic()>deadline:raise TimeoutError('No startup odometry / clock')
        key='initial_poses' if os.environ.get('DEMO_START_MODE')=='approach' else 'snapshot_poses'
        record['start_mode']=key
        initial=json.loads((HERE/'layout.json').read_text())[key]
        if key=='initial_poses':
            initial['R1'][0]=-float(os.environ.get('DEMO_GOAL_X','2.7'))
            initial['R2'][0]=float(os.environ.get('DEMO_GOAL_X','2.7'))
        for i,r in enumerate(robots,1):
            p=pose(*initial[f'R{i}']);p.header.stamp=nav[r].get_clock().now().to_msg()
            nav[r].setInitialPose(p)
        for n in nav.values():n.waitUntilNav2Active()
        deadline=time.monotonic()+30
        while len(audit.poses)<3:
            audit.pump()
            if time.monotonic()>deadline:raise TimeoutError('Missing AMCL TF')
        for _ in range(40):audit.pump()
        record['initial_poses']={r:audit.poses[r][:] for r in robots}
        if key=='initial_poses':
            worker.request(record['initial_poses'],preferred=True)
            preferred=json.loads((report/'marco_preferred_paths.json').read_text())
            res=preferred['grid_size'];preview={}
            for r in ('tb1','tb2'):
                cells=preferred['paths']['R'+r[-1]]
                metric=[[x*res,y*res] for x,y in cells]
                preview[r]=metric
                send(r,cells,'resume')
            deadline=time.monotonic()+120
            while True:
                pump()
                hits=predicted_contacts({r:audit.poses[r][:] for r in robots},preview,pairs,
                                        {'tb1':.16,'tb2':.16},horizon_s=trigger_horizon)
                if hits:break
                for r in ('tb1','tb2'):
                    if nav[r].isTaskComplete():raise RuntimeError(f'{r} nominal action ended before conflict trigger')
                if time.monotonic()>deadline:raise TimeoutError('No swept conflict trigger')
            event('swept_conflict_predicted',contacts=hits,
                  poses={r:audit.poses[r][:] for r in robots})
            for r in ('tb1','tb2'):nav[r].cancelTask()
            audit.holding.update(robots)
            quiet=None;deadline=time.monotonic()+12
            while time.monotonic()<deadline:
                pump()
                if all(abs(v)<.01 and abs(w)<.03 for v,w in audit.twists.values()):
                    quiet=quiet or time.monotonic()
                    if time.monotonic()-quiet>=.5:break
                else:quiet=None
            else:raise RuntimeError('Approaching robots did not settle')
            speed_limit(.16)
            event('swept_conflict_stopped',poses={r:audit.poses[r][:] for r in robots})
        record['conflict_poses']={r:audit.poses[r][:] for r in robots}
        verify_live_profile(load_profile())
        record['planning_timings']=worker.request(record['conflict_poses'])
        plan=json.loads((report/'marco_execution_plan.json').read_text());record['plan']=plan
        contract=verify_contract(plan)
        record['execution_contract_sha256']=plan['execution_contract_sha256']
        res=plan['grid_size'];priority=plan['priority_robot'];priority_name='tb'+priority[1:]
        world=json.loads((report/'scanned_world.json').read_text())
        raw_free={(x,int(y)) for y,spans in world['free_spans'].items() for a,b in spans for x in range(a,b+1)}
        def staging_idle_safe(r,v,w):
            # Residual slow yaw is not a passage conflict when the entire
            # rotation envelope is free. Do not weaken the approach/entry
            # stop criterion; this applies only inside the holding bay.
            if abs(v)>=.01:return False
            if abs(w)<.03:return True
            if abs(w)>=.25:return False
            x,y=audit.poses[r][:2]
            # Both padded rectangular bodies fit in radius 0.264 m discs.
            # Reserve >0.55 m between centres before allowing residual
            # yaw; this covers every peer heading, not only the current SAT.
            if any(math.dist((x,y),p[:2])<=.55 for other,p in audit.poses.items() if other!=r):
                return False
            for cx in range(math.floor((x-.29)/res),math.ceil((x+.29)/res)+1):
                for cy in range(math.floor((y-.29)/res),math.ceil((y+.29)/res)+1):
                    if math.hypot(cx*res-x,cy*res-y)<=.29 and (cx,cy) not in raw_free:
                        return False
            return True
        record['staging_idle_gate']={'linear_limit_m_s':.01,'ordinary_angular_limit_rad_s':.03,
            'bounded_angular_limit_rad_s':.25,'all_yaw_envelope_m':.29,
            'all_yaw_peer_separation_m':.55,
            'condition':'bounded residual yaw allowed only if entire disc is static-free; live peer/passage guards remain active'}
        protected=[[x*res,y*res] for x,y in (plan.get('protected_priority_corridor') or plan['planned_paths'][priority])]
        event('plan_ready',candidate=plan['candidate_id'],priority=priority,
              sequence=[s['robot_id'] for s in plan.get('relocation_sequence',[])])
        record['timing_prediction']=plan['timing_prediction']
        abstract=json.loads((report/'marco_plan.json').read_text())
        record['core_prediction_reference']={'estimated_makespan_s':abstract.get('core_nominal_estimated_makespan'),
            'comparable_to_ros_execution':False,
            'reason':'Unadapted grid timing is reference only; selection uses frozen ROS contract, with zero task service'}
        execution_epoch=audit.get_clock().now().nanoseconds/1e9
        record['execution_epoch_ros_s']=execution_epoch
        event('execution_epoch')
        guard()
        for step in plan.get('relocation_sequence',[]):
            capture_costmaps('before_'+step['robot_id'])
            if step.get('entry_path'):
                r='tb'+step['robot_id'][1:];target=[v*res for v in step['entry_path'][-1]]
                entry_action=next(a for a in contract['clearance'] if a['robot']==step['robot_id'] and a['kind']=='entry')
                send(r,step['entry_path'],'clearance',contract_action=entry_action);deadline=time.monotonic()+30
                while not nav[r].isTaskComplete():
                    pump()
                    if time.monotonic()>deadline:raise TimeoutError(f'{r} pocket entry timeout')
                if nav[r].getResult()!=TaskResult.SUCCEEDED or math.dist(audit.poses[r][:2],target)>.035:
                    raise RuntimeError(f'{r} pocket entry failed')
                audit.holding.add(r);quiet=None;deadline=time.monotonic()+12
                while time.monotonic()<deadline:
                    pump();v,w=audit.twists.get(r,(1.,1.))
                    if abs(v)<.01 and abs(w)<.03:
                        quiet=quiet or audit.get_clock().now().nanoseconds/1e9
                        if audit.get_clock().now().nanoseconds/1e9-quiet>=contract['profile']['settle_s']:break
                    else:quiet=None
                else:raise RuntimeError(f'{r} pocket entry did not settle')
                event('pocket_entry_completed',robot=r,pose=audit.poses[r][:])
            r='tb'+step['robot_id'][1:]
            holding_action=next(a for a in contract['clearance'] if a['robot']==step['robot_id'] and a['kind']=='holding')
            send(r,step['path'],'clearance',contract_action=holding_action)
            target=[x*res for x in step['destination']];deadline=time.monotonic()+100;corrections=0
            # R3 parks deeper by another 5 cm. A 5 cm tracking tolerance
            # avoids DWB chasing a sub-cell pose after it has safely left
            # the mouth. Actual passage/rotation/peer gates remain decisive.
            holding_tolerance=holding_action['acceptance_m']
            while True:
                pump();done=nav[r].isTaskComplete()
                error=math.dist(audit.poses[r][:2],target)
                if passage_is_clear(audit.poses[r],protected,contract['profile']['clearance_margin_m']) and error<holding_tolerance:
                    nav[r].cancelTask();audit.holding.add(r)
                    stop_by=time.monotonic()+12;quiet=None
                    while time.monotonic()<stop_by:
                        pump();v,w=audit.twists.get(r,(1.,1.))
                        if staging_idle_safe(r,v,w):
                            quiet=quiet or audit.get_clock().now().nanoseconds/1e9
                            if audit.get_clock().now().nanoseconds/1e9-quiet>=contract['profile']['settle_s']:break
                        else:quiet=None
                    else:raise RuntimeError(f'{r} did not settle')
                    if not passage_is_clear(audit.poses[r],protected,.10):
                        raise RuntimeError(f'{r} unstable clearance')
                    held.add(r);event('clearance_completed',robot=r,pose=audit.poses[r][:],goal_error_m=error,
                                      holding_goal_tolerance_m=holding_tolerance);break
                if done:
                    if nav[r].getResult()==TaskResult.SUCCEEDED and corrections<2:
                        corrections+=1;audit.holding.add(r)
                        record['prediction_contract_deviations'].append({'robot':r,'kind':'staging_pose_correction'})
                        quiet=None;stop_by=time.monotonic()+12
                        while time.monotonic()<stop_by:
                            pump();v,w=audit.twists.get(r,(1.,1.))
                            if staging_idle_safe(r,v,w):
                                quiet=quiet or audit.get_clock().now().nanoseconds/1e9
                                if audit.get_clock().now().nanoseconds/1e9-quiet>=contract['profile']['settle_s']:break
                            else:quiet=None
                        else:raise RuntimeError(f'{r} pose correction did not settle')
                        event('staging_pose_correction',robot=r,attempt=corrections,
                              map_goal_error_m=math.dist(audit.poses[r][:2],target),
                              criterion='Nav2 succeeded but live map-frame clearance/goal gate not satisfied')
                        correction=[[v/res for v in audit.poses[r][:2]],step['destination']]
                        send(r,correction,'clearance')
                        continue
                    raise RuntimeError(f'{r} clearance ended {nav[r].getResult()}, error={error:.3f}')
                if time.monotonic()>deadline:raise TimeoutError(f'{r} clearance timeout')
        pending=[r for r in plan['planned_paths'] if r!=priority and plan['planned_paths'][r]]
        active={};times={};retries={r:0 for r in robots};segments={};segment_index={}
        segmented=(os.environ.get('DEMO_CONTROLLER','hybrid')=='hybrid' and
                   os.environ.get('DEMO_RESUME_POLICY','phase_specific')=='phase_specific')
        def send_segment(r):
            i=segment_index[r];part=segments[r][i]
            send(r,part['path'],'resume',controller_override=part.get('controller'),
                 precise_goal=i<len(segments[r])-1,contract_action=part)
            event('resumption_segment_started',robot=r,index=i,total=len(segments[r]))
        def resume(robot):
            r='tb'+robot[1:];held.discard(r)
            if not segmented:raise RuntimeError('Executor mode does not match prediction contract')
            segments[r]=contract['resume'][robot]
            segment_index[r]=0;active[r]=robot;times[r]=time.monotonic();send_segment(r)
        resume(priority)
        while active or pending:
            pump()
            first=remaining_route(audit.poses[priority_name],[[x*res,y*res] for x,y in plan['planned_paths'][priority]])
            for robot in list(pending):
                r='tb'+robot[1:]
                second=remaining_route(audit.poses[r],[[x*res,y*res] for x,y in plan['planned_paths'][robot]])
                if routes_are_disjoint(first,second,contract['profile']['release_margin_m']):
                    event('concurrent_release',robot=r,priority_still_active=priority_name in active)
                    resume(robot);pending.remove(robot)
            for r in list(active):
                if nav[r].isTaskComplete():
                    if nav[r].getResult()!=TaskResult.SUCCEEDED:
                        if retries[r]>=2:raise RuntimeError(f'{r} resume failed after bounded controller retries')
                        retries[r]+=1;nav[r].cancelTask();audit.holding.add(r)
                        record['prediction_contract_deviations'].append({'robot':r,'kind':'DWB_controller_recovery'})
                        event('controller_retry',robot=r,attempt=retries[r],
                              result=str(nav[r].getResult()),controller='FollowPath (DWB)',
                              criterion='same remaining planned path, no new candidate')
                        deadline=time.monotonic()+12;quiet=None
                        while time.monotonic()<deadline:
                            pump();v,w=audit.twists.get(r,(1.,1.))
                            if abs(v)<.01 and abs(w)<.03:
                                quiet=quiet or audit.get_clock().now().nanoseconds/1e9
                                if audit.get_clock().now().nanoseconds/1e9-quiet>=contract['profile']['settle_s']:break
                            else:quiet=None
                        else:raise RuntimeError(f'{r} controller recovery did not settle')
                        part=segments[r][segment_index[r]]
                        path_m=[[x*res,y*res] for x,y in part['path']]
                        rest=remaining_route(audit.poses[r],path_m)
                        cells=[(round(x/res),round(y/res)) for x,y in rest]
                        send(r,cells,'resume',controller_override='FollowPath',
                             precise_goal=segment_index[r]<len(segments[r])-1)
                        continue
                    goal=[x*res for x in segments[r][segment_index[r]]['path'][-1]]
                    error=math.dist(audit.poses[r][:2],goal)
                    if segment_index[r]<len(segments[r])-1:
                        if error>.035:raise RuntimeError(f'{r} intermediate resumption pose outside tolerance')
                        event('resumption_segment_completed',robot=r,index=segment_index[r],goal_error_m=error)
                        segment_index[r]+=1;send_segment(r);continue
                    if error>.12:raise RuntimeError(f'{r} final pose outside tolerance')
                    event('goal_completed',robot=r,pose=audit.poses[r][:],goal_error_m=error)
                    audit.holding.add(r);active.pop(r)
                elif time.monotonic()-times[r]>120:raise TimeoutError(f'{r} resumption timeout')
            if pending and not active:raise RuntimeError('Completed priority still blocks resumption')
        record['success']=True;event('mission_success')
        actual=audit.get_clock().now().nanoseconds/1e9-execution_epoch
        prediction=record['timing_prediction']['remaining_navigation_s']
        record['timing_comparison']={'predicted_s':prediction,'executed_ros_s':actual,
            'matched_action_contract':not record['prediction_contract_deviations'],
            'prediction_minus_execution_s':prediction-actual,
            'absolute_relative_error_pct':100*abs(prediction-actual)/actual,
            'goals':[{'robot':e['robot'],'predicted_s':record['timing_prediction']['goal_completion_s'][e['robot']],
                      'executed_ros_s':e['ros_s']-execution_epoch}
                     for e in record['stages'] if e['stage']=='goal_completed']}
    except Exception as error:
        capture_costmaps('failure')
        record['error']=str(error);event('failed',error=str(error));raise
    finally:
        for n in nav.values():n.cancelTask()
        audit.holding.update(robots);audit.hold_commands()
        if worker:worker.close()
        record.update(elapsed_s=time.monotonic()-start,final_poses=audit.poses,final_twists=audit.twists,
                      samples=audit.samples,nav2_paths=audit.paths)
        (report/'execution_audit.json').write_text(json.dumps(record,indent=2))
        executor.shutdown(timeout_sec=2);receiver.join(timeout=2)
        for n in nav.values():n.destroy_node()
        audit.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
