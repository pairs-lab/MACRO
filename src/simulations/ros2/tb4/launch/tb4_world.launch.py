"""Three TurtleBot 4 robots (tb1, tb2, tb3) in the MARCO T-junction, Ignition Gazebo, no charging dock.

Needs the TurtleBot 4 simulator packages (turtlebot4_ignition_bringup, irobot_create_*): source an environment that
provides them, e.g. the `environment.sh` of a TurtleBot 4 workspace. Robots are spawned with the official bringup
copied without the dock (turtlebot4_spawn_no_dock.launch.py).
Arguments: model:=lite|standard  headless:=true|false  start_mode:=approach|snapshot  goal_x:=2.7
"""
import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
TB4 = HERE.parent
LAYOUT = TB4.parent / 'three_robot_t' / 'layout.json'


def launch_setup(context):
    share = lambda name: Path(get_package_share_directory(name))
    model = LaunchConfiguration('model').perform(context)
    headless = LaunchConfiguration('headless').perform(context) == 'true'
    key = 'initial_poses' if LaunchConfiguration('start_mode').perform(context) == 'approach' else 'snapshot_poses'
    goal_x = float(LaunchConfiguration('goal_x').perform(context))
    poses = json.loads(LAYOUT.read_text())[key]
    if key == 'initial_poses':           # keep the padded body away from the end walls, as in the Classic launch
        poses['R1'][0], poses['R2'][0] = -goal_x, goal_x
    world = TB4 / 'worlds' / 'three_robot_t.sdf'
    actions = [
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', ':'.join([
            str(TB4 / 'worlds'), str(share('turtlebot4_ignition_bringup') / 'worlds'),
            str(share('irobot_create_ignition_bringup') / 'worlds'),
            str(share('turtlebot4_description').parent), str(share('irobot_create_description').parent)])),
        SetEnvironmentVariable('IGN_GUI_PLUGIN_PATH', ':'.join([
            str(share('turtlebot4_ignition_gui_plugins') / 'lib'),
            str(share('irobot_create_ignition_plugins') / 'lib')])),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share('ros_ign_gazebo') / 'launch/ign_gazebo.launch.py')),
            launch_arguments={'ign_args': f'{world} -r -v 2' + (' -s' if headless else f' --gui-config {TB4 / "config/gui.config"}')}.items()),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='clock_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'], output='screen'),
    ]
    wanted = [int(v) for v in LaunchConfiguration('robots').perform(context).split(',')]
    stagger = float(LaunchConfiguration('stagger').perform(context))
    actions_spawned = []
    for index, robot in ((1, 'R1'), (2, 'R2'), (3, 'R3')):
        if index not in wanted:
            continue
        x, y, yaw = poses[robot]
        spawn = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(HERE / 'tb4_robot.launch.py')),
            launch_arguments={'model': model, 'namespace': f'tb{index}', 'x': str(x), 'y': str(y), 'z': '0.0',
                              'yaw': str(yaw)}.items())
        # Creating the GPU sensors of several robots at once is fragile in Ignition; spawn them one by one.
        actions.append(TimerAction(period=stagger * len(actions_spawned), actions=[spawn]) if stagger > 0 else spawn)
        actions_spawned.append(index)
    scans = [f'/world/three_robot_t/model/tb{i}/turtlebot4/link/rplidar_link/sensor/rplidar/scan'
             '@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan' for i in wanted]
    actions.append(Node(package='ros_gz_bridge', executable='parameter_bridge', name='scan_bridge', output='screen',
                        arguments=scans, parameters=[{'use_sim_time': True}],
                        remappings=[(f'/world/three_robot_t/model/tb{i}/turtlebot4/link/rplidar_link/sensor/rplidar/scan',
                                     f'/tb{i}/scan') for i in wanted]))
    actions.append(ExecuteProcess(cmd=['/usr/bin/python3', str(TB4 / 'cmd_vel_relay.py')],
                                  additional_env={'FLEET_ROBOTS': ' '.join(f'tb{i}' for i in wanted)}, output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='lite', choices=['standard', 'lite']),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('start_mode', default_value='approach', choices=['approach', 'snapshot']),
        DeclareLaunchArgument('stagger', default_value='5', description='seconds between robot spawns'),
        DeclareLaunchArgument('robots', default_value='1,2,3', description='subset of robots to spawn (testing)'),
        DeclareLaunchArgument('goal_x', default_value=os.environ.get('DEMO_GOAL_X', '2.7')),
        OpaqueFunction(function=launch_setup),
    ])
