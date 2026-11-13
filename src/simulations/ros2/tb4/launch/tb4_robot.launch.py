"""One MARCO TurtleBot 4 in the running Ignition world, minimal node set and no charging dock.

Per robot: robot_state_publisher, the entity (ros_ign_gazebo create), the two ros2_control spawners (joint state
broadcaster, diff-drive controller) and the two static frame aliases the Create 3 stack uses (odom, base_link).
The official bringup starts ~65 processes per robot (one ros_gz bridge per topic, Create 3 hazard/IR/UI/dock nodes);
three of them saturate a laptop CPU, so only what Nav2/AMCL and the MARCO executor consume is started. The lidar scan
and /clock are bridged once for all robots by tb4_world.launch.py; /cmd_vel by cmd_vel_relay.py.
Arguments: namespace model x y z yaw
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

SCRIPT = Path(__file__).resolve().parents[1] / 'make_robot_description.py'


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    remap_tf = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    control = str(Path(get_package_share_directory('irobot_create_control')) / 'config' / 'control.yaml')

    state = Node(package='robot_state_publisher', executable='robot_state_publisher', name='robot_state_publisher',
                 output='screen', remappings=remap_tf,
                 parameters=[{'use_sim_time': True, 'robot_description': Command(
                     ['python3 ', str(SCRIPT), ' ', LaunchConfiguration('model'), ' ', namespace])}])
    create = Node(package='ros_ign_gazebo', executable='create', output='screen',
                  arguments=['-name', [namespace, '/turtlebot4'], '-x', LaunchConfiguration('x'),
                             '-y', LaunchConfiguration('y'), '-z', LaunchConfiguration('z'),
                             '-Y', LaunchConfiguration('yaw'), '-topic', 'robot_description'])
    diffdrive = Node(package='controller_manager', executable='spawner', namespace=namespace,
                     parameters=[control], arguments=['diffdrive_controller', '-c', 'controller_manager'], output='screen')
    joint_state = Node(package='controller_manager', executable='spawner',
                       arguments=['joint_state_broadcaster', '-c', 'controller_manager'], output='screen')
    odom_alias = Node(package='tf2_ros', executable='static_transform_publisher', name='tf_namespaced_odom_publisher',
                      arguments=['0', '0', '0', '0', '0', '0', 'odom', [namespace, '/odom']], remappings=remap_tf)
    base_alias = Node(package='tf2_ros', executable='static_transform_publisher', name='tf_namespaced_base_link_publisher',
                      arguments=['0', '0', '0', '0', '0', '0', [namespace, '/base_link'], 'base_link'], remappings=remap_tf)
    return LaunchDescription([
        DeclareLaunchArgument('namespace'), DeclareLaunchArgument('model', default_value='lite'),
        DeclareLaunchArgument('x', default_value='0.0'), DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.0'), DeclareLaunchArgument('yaw', default_value='0.0'),
        GroupAction([PushRosNamespace(namespace), state, create, joint_state, odom_alias, base_alias,
                     # the diff-drive controller is loaded after the joint state broadcaster, as in the official bringup
                     RegisterEventHandler(OnProcessExit(target_action=joint_state, on_exit=[diffdrive]))]),
    ])
