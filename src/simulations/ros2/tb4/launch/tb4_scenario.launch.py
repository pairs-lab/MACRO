"""Three TurtleBot 4 robots in the MARCO T-junction: Ignition world, Nav2 + AMCL on the recorded map, RViz fleet view.

Counterpart of three_robot_t/rviz_capture.launch.py (TurtleBot 3 / Gazebo Classic). No charging dock is spawned.
Environment: DEMO_REPORT_DIR (run directory), DEMO_START_MODE (approach|snapshot), DEMO_GOAL_X, DEMO_CONTROLLER,
MARCO_TB4_HEADLESS (true: Ignition server only, RViz is the display), MARCO_TB4_MODEL (lite|standard).
"""
import json
import math
import os
import sys
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace

HERE = Path(__file__).resolve().parent
TB4 = HERE.parent
ROS2 = TB4.parent
SCENARIO = ROS2 / 'three_robot_t'
sys.path.insert(0, str(ROS2))
from marco_controllers import add_hybrid_controllers  # noqa: E402

BODY_RADIUS = .171          # Create 3 base collision radius (irobot_create_description)
FOOTPRINT_PADDING = .0      # the 0.5 m T-junction corridor leaves +-0.079 m for the centre; any padding closes the local costmap
PEER_MARGIN = .01           # peers are injected into each other's maps with a 1 cm margin


def circle_footprint(radius, sides=16):
    return str([[round(radius * math.cos(2 * math.pi * i / sides), 4), round(radius * math.sin(2 * math.pi * i / sides), 4)]
                for i in range(sides)])


def tb4_nav2_params(text, robot, pose):
    """nav2_tb1.yaml (TurtleBot 3 waffle) adapted to the TurtleBot 4: base_link frame, round footprint."""
    text = text.replace('/tb1/', f'/{robot}/').replace('base_footprint', 'base_link')
    params = yaml.safe_load(text)
    for costmap in ('local_costmap', 'global_costmap'):
        node = params[costmap][costmap]['ros__parameters']
        node['footprint'] = circle_footprint(BODY_RADIUS)
        node['footprint_padding'] = FOOTPRINT_PADDING
    params['amcl']['ros__parameters']['initial_pose'].update(x=float(pose[0]), y=float(pose[1]), yaw=float(pose[2]))
    if os.environ.get('DEMO_CONTROLLER', 'hybrid') == 'hybrid':
        add_hybrid_controllers(params)
    return params


def generate_launch_description():
    runtime = Path(os.environ.get('DEMO_REPORT_DIR', str(HERE)))
    runtime.mkdir(parents=True, exist_ok=True)
    nav2_launch = Path(get_package_share_directory('nav2_bringup')) / 'launch'
    key = 'initial_poses' if os.environ.get('DEMO_START_MODE') == 'approach' else 'snapshot_poses'
    poses = json.loads((SCENARIO / 'layout.json').read_text())[key]
    goal_x = float(os.environ.get('DEMO_GOAL_X', '2.7'))
    if key == 'initial_poses':
        poses['R1'][0], poses['R2'][0] = -goal_x, goal_x

    config = yaml.safe_load((ROS2 / 'fleet.rviz').read_text())
    displays = config['Visualization Manager']['Displays']
    displays[:] = [d for d in displays if d['Class'] not in ('rviz_default_plugins/MarkerArray', 'rviz_default_plugins/LaserScan')]
    for d in displays:
        if 'global inflation' in d['Name']:
            d['Enabled'] = True
            d['Alpha'] = .55 if 'tb1' in d['Name'] else .15
        if 'local inflation' in d['Name']:
            d['Enabled'] = False
    displays.extend(yaml.safe_load(yaml.safe_dump(d).replace('tb2', 'tb3')) for d in list(displays) if 'tb2' in d['Name'])
    displays.append({'Class': 'rviz_default_plugins/MarkerArray', 'Name': 'Conflict bounds (radius 0.25 m)',
                     'Enabled': True, 'Value': True, 'Topic': {'Value': '/marco/conflict_bounds', 'Depth': 10,
                     'Reliability Policy': 'Reliable', 'Durability Policy': 'Volatile'}})
    config['Visualization Manager']['Views']['Current'].update(Scale=float(os.environ.get('DEMO_RVIZ_SCALE', '105')), X=0., Y=float(os.environ.get('DEMO_RVIZ_Y', '-1')))
    if os.environ.get('DEMO_CLEAN_VIEW'):
        # paper figures: no big AMCL pose arrow, one sharp path colour per robot, large borderless view
        colours = {'tb1': '0; 110; 255', 'tb2': '255; 120; 0', 'tb3': '40; 190; 60'}
        for d in displays:
            robot = next((r for r in colours if d['Name'].startswith(r)), None)
            if d['Class'] == 'rviz_default_plugins/Path' and robot:
                d.update({'Color': colours[robot], 'Line Width': 0.035, 'Line Style': 'Billboards', 'Alpha': 1.0,
                          'Offset': {'X': 0.0, 'Y': {'tb1': 0.035, 'tb2': -0.035}.get(robot, 0.0), 'Z': 0.0}})
            if d['Class'] == 'rviz_default_plugins/PoseWithCovariance':
                d.update({'Shape': 'Axes', 'Axes Length': 0.001, 'Axes Radius': 0.001})
                d.setdefault('Covariance', {}).setdefault('Orientation', {})['Value'] = False
        config['Window Geometry'].update({'Width': 1900, 'Height': 1130, 'X': 0, 'Y': 0, 'Hide Left Dock': True, 'Hide Right Dock': True})
    (runtime / 'capture.rviz').write_text(yaml.safe_dump(config))

    fleet_env = {'FLEET_ROBOTS': 'tb1 tb2 tb3', 'FLEET_FOOTPRINT_RADIUS': str(BODY_RADIUS + PEER_MARGIN)}
    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(HERE / 'tb4_world.launch.py')),
            launch_arguments={'model': os.environ.get('MARCO_TB4_MODEL', 'lite'),
                              'headless': os.environ.get('MARCO_TB4_HEADLESS', 'true'),
                              'start_mode': 'approach' if key == 'initial_poses' else 'snapshot',
                              'goal_x': str(goal_x)}.items()),
        ExecuteProcess(cmd=['/usr/bin/python3', str(ROS2 / 'fleet_bridge.py')], additional_env=fleet_env, output='screen'),
        ExecuteProcess(cmd=['/usr/bin/python3', str(ROS2 / 'fleet_obstacles.py')], additional_env=fleet_env, output='screen'),
    ]
    template = (ROS2 / 'nav2_tb1.yaml').read_text()
    for index, robot in enumerate(['tb1', 'tb2', 'tb3'], 1):
        params = tb4_nav2_params(template, robot, poses[f'R{index}'])
        path = runtime / f'capture_{robot}.yaml'
        path.write_text(yaml.safe_dump(params))
        common = {'namespace': robot, 'use_sim_time': 'true', 'autostart': 'true',
                  'use_composition': 'False', 'params_file': str(path)}
        actions.append(GroupAction([PushRosNamespace(robot),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(nav2_launch / 'localization_launch.py')),
                launch_arguments={**common, 'map': str(SCENARIO / 'measured_images/scanned_map.yaml')}.items()),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(nav2_launch / 'navigation_launch.py')),
                launch_arguments=common.items())]))
    actions.append(Node(package='rviz2', executable='rviz2', name='rviz_three_robot_capture',
        arguments=['-d', str(runtime / 'capture.rviz')], parameters=[{'use_sim_time': True}],
        remappings=[('/tf', '/fleet/tf'), ('/tf_static', '/fleet/tf_static')]))
    return LaunchDescription(actions)
