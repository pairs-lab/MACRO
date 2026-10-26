#!/usr/bin/env python3
"""/<robot>/cmd_vel -> /<robot>/diffdrive_controller/cmd_vel_unstamped for all robots in ONE process.

Replaces the Create 3 `motion_control` node (one process per robot; its reflexes need the cliff/IR/bumper sensors that
the MARCO TurtleBot 4 description does not simulate). Velocity limits stay with Nav2's velocity smoother and the
diff-drive controller (irobot_create_control/config/control.yaml).
"""
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Relay(Node):
    def __init__(self):
        super().__init__('tb4_cmd_vel_relay')
        self.pubs = {}
        for robot in os.environ.get('FLEET_ROBOTS', 'tb1 tb2 tb3').split():
            pub = self.create_publisher(Twist, f'/{robot}/diffdrive_controller/cmd_vel_unstamped', 10)
            self.pubs[robot] = pub
            self.create_subscription(Twist, f'/{robot}/cmd_vel', pub.publish, 10)


if __name__ == '__main__':
    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
