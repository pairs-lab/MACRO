#!/usr/bin/env python3
"""TurtleBot 4 URDF for the MARCO multi-robot Ignition world.

usage: make_robot_description.py MODEL NAMESPACE   (prints the URDF)

Starts from the official turtlebot4_description xacro (gazebo:=ignition) and removes what a multi-robot MARCO run does
not use and what breaks several robots in one world:
  * the per-robot Ignition `Sensors` and `Contact` system plugins (the world declares the Sensors system once;
    one instance per robot makes the render scene collide and Ignition crashes with two or more robots),
  * the Create 3 cliff and IR gpu_lidar sensors and the OAK-D RGBD camera (11 + 1 render sensors per robot).
The RPLIDAR (scan), wheel odometry, IMU, bumper contact and the ros2_control diff-drive stay.
"""
import subprocess
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory

KEEP_LIDAR = {'rplidar'}
DROP_PLUGINS = ('sensors-system', 'contact-system')


def main(model, namespace):
    xacro = f"{get_package_share_directory('turtlebot4_description')}/urdf/{model}/turtlebot4.urdf.xacro"
    text = subprocess.run(['xacro', xacro, 'gazebo:=ignition', f'namespace:={namespace}'],
                          check=True, capture_output=True, text=True).stdout
    root = ET.fromstring(text)
    removed = {'sensor': 0, 'plugin': 0}
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == 'sensor':
                kind, name = child.get('type'), child.get('name')
                if (kind == 'gpu_lidar' and name not in KEEP_LIDAR) or kind in ('rgbd_camera', 'camera', 'depth_camera'):
                    parent.remove(child); removed['sensor'] += 1
            elif child.tag == 'plugin' and any(key in (child.get('filename') or '') for key in DROP_PLUGINS):
                parent.remove(child); removed['plugin'] += 1
    for sensor in root.iter('sensor'):
        if sensor.get('name') in KEEP_LIDAR and sensor.find('ignition_frame_id') is None:
            frame = ET.SubElement(sensor, 'ignition_frame_id'); frame.text = 'rplidar_link'   # scan frame = URDF link
    # launch's Command substitution fails on any stderr output, so nothing is printed there
    assert removed['sensor'] > 0, 'unexpected TurtleBot 4 description: nothing was removed'
    sys.stdout.write(ET.tostring(root, encoding='unicode'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
