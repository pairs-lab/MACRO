"""Three-robot Gazebo/Nav2/AMCL view on the recorded SLAM map.

Default direct launch spawns the conflict snapshot; run.sh selects approach.
"""
from pathlib import Path
import json
import os
import yaml
from launch import LaunchDescription
from launch.actions import ExecuteProcess, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marco_controllers import add_hybrid_controllers


def generate_launch_description():
    here = Path(__file__).resolve().parent
    root = here.parent
    runtime = Path(os.environ.get('DEMO_REPORT_DIR', str(here)))
    runtime.mkdir(parents=True, exist_ok=True)
    package = Path(os.environ['MARCO_ROS_MODEL_PACKAGE'])
    model_path=package/'models/turtlebot3_waffle/model.sdf'
    from ament_index_python.packages import get_package_share_directory
    vendor = Path(get_package_share_directory('nav2_bringup'))/'launch'
    key='initial_poses' if os.environ.get('DEMO_START_MODE')=='approach' else 'snapshot_poses'
    poses = json.loads((here/'layout.json').read_text())[key]
    if key=='initial_poses':
        # Keep the full padded body away from the measured end-wall cells.
        poses['R1'][0]=-float(os.environ.get('DEMO_GOAL_X','1.2'))
        poses['R2'][0]=float(os.environ.get('DEMO_GOAL_X','1.2'))
    urdf = (package/'urdf/turtlebot3_waffle.urdf').read_text().replace(
        'package://turtlebot3_description/meshes/', 'file://'+str(package/'urdf/meshes')+'/')
    config = yaml.safe_load((root/'fleet.rviz').read_text())
    displays = config['Visualization Manager']['Displays']
    displays[:] = [d for d in displays if d['Class'] not in (
        'rviz_default_plugins/MarkerArray', 'rviz_default_plugins/LaserScan')]
    for d in displays:
        if 'global inflation' in d['Name']:
            d['Enabled'] = True
            d['Alpha'] = .55 if 'tb1' in d['Name'] else .15
        if 'local inflation' in d['Name']:
            d['Enabled'] = False
    displays.extend(yaml.safe_load(yaml.safe_dump(d).replace('tb2','tb3'))
                    for d in list(displays) if 'tb2' in d['Name'])
    displays.append({'Class':'rviz_default_plugins/MarkerArray','Name':'Conflict bounds (radius 0.25 m)',
                     'Enabled':True,'Value':True,'Topic':{'Value':'/marco/conflict_bounds',
                     'Depth':10,'Reliability Policy':'Reliable','Durability Policy':'Volatile'}})
    config['Visualization Manager']['Views']['Current'].update(Scale=float(os.environ.get('DEMO_RVIZ_SCALE','105')), X=0., Y=float(os.environ.get('DEMO_RVIZ_Y','-1')))
    if os.environ.get('DEMO_CLEAN_VIEW'):
        # paper figures: no big AMCL pose arrow (covariance ellipse stays), one sharp path colour per robot, large borderless view
        colours = {'tb1': '0; 110; 255', 'tb2': '255; 120; 0', 'tb3': '40; 190; 60'}
        for d in displays:
            robot = next((r for r in colours if d['Name'].startswith(r)), None)
            if d['Class'] == 'rviz_default_plugins/Path' and robot:
                d.update({'Color': colours[robot], 'Line Width': 0.035, 'Line Style': 'Billboards', 'Alpha': 1.0,
                          'Offset': {'X': 0.0, 'Y': {'tb1': 0.035, 'tb2': -0.035}.get(robot, 0.0), 'Z': 0.0}})
            if d['Class'] == 'rviz_default_plugins/PoseWithCovariance':
                d.update({'Shape': 'Axes', 'Axes Length': 0.001, 'Axes Radius': 0.001})
                d.setdefault('Covariance', {}).setdefault('Orientation', {})['Value'] = False      # no yaw-covariance wedge
        config['Window Geometry'].update({'Width': 1900, 'Height': 1130, 'X': 0, 'Y': 0, 'Hide Left Dock': True, 'Hide Right Dock': True})
    (runtime/'capture.rviz').write_text(yaml.safe_dump(config))
    actions = [ExecuteProcess(cmd=['gzserver', '--verbose', str(here/'three_robot_t.world'),
                                  '-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so'], output='screen'),
               ExecuteProcess(cmd=['/usr/bin/python3', str(root/'fleet_bridge.py')],
                              additional_env={'FLEET_ROBOTS':'tb1 tb2 tb3'}, output='screen'),
               ExecuteProcess(cmd=['/usr/bin/python3', str(root/'fleet_obstacles.py')],
                              additional_env={'FLEET_ROBOTS':'tb1 tb2 tb3'}, output='screen')]
    for index, robot in enumerate(['tb1','tb2','tb3'], 1):
        x, y, yaw = poses[f'R{index}']
        params = yaml.safe_load((root/'nav2_tb1.yaml').read_text().replace('/tb1/',f'/{robot}/'))
        params['amcl']['ros__parameters']['initial_pose'].update(x=float(x),y=float(y),yaw=float(yaw))
        if os.environ.get('DEMO_CONTROLLER','hybrid')=='hybrid':
            add_hybrid_controllers(params)
        path = runtime/f'capture_{robot}.yaml'
        path.write_text(yaml.safe_dump(params))
        actions.extend([
            Node(package='robot_state_publisher', executable='robot_state_publisher',namespace=robot,
                 parameters=[{'robot_description':urdf,'use_sim_time':True}],
                 remappings=[('/tf','tf'),('/tf_static','tf_static')]),
            Node(package='gazebo_ros', executable='spawn_entity.py',arguments=[
                 '-file',str(model_path), '-entity',robot,
                 '-robot_namespace',f'/{robot}','-x',str(x),'-y',str(y),'-Y',str(yaw),'-z','.01']),
        ])
        common={'namespace':robot,'use_sim_time':'true','autostart':'true',
                'use_composition':'False','params_file':str(path)}
        actions.append(GroupAction([PushRosNamespace(robot),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(vendor/'localization_launch.py')),
                launch_arguments={**common,'map':str(here/'measured_images/scanned_map.yaml')}.items()),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(vendor/'navigation_launch.py')),
                launch_arguments=common.items())]))
    actions.append(Node(package='rviz2',executable='rviz2',name='rviz_three_robot_capture',
        arguments=['-d',str(runtime/'capture.rviz')],parameters=[{'use_sim_time':True}],
        remappings=[('/tf','/fleet/tf'),('/tf_static','/fleet/tf_static')]))
    return LaunchDescription(actions)
