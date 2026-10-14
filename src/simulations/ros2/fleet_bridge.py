"""Visualization-only namespaced TF/URDF bridge; does not change Nav2 frames."""
import copy
import os
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate

def frame(robot,value):
    value=value.lstrip('/')
    return value if value=='map' else robot+'/'+value

def prefixed_urdf(robot,value):
    root=ET.fromstring(value)
    for element in root.iter():
        if element.tag in ('link','joint') and 'name' in element.attrib:
            element.set('name',robot+'/'+element.get('name'))
        for key in ('link','joint'):
            if key in element.attrib:element.set(key,robot+'/'+element.get(key))
    return ET.tostring(root,encoding='unicode')

class FleetBridge(Node):
    def __init__(self):
        super().__init__('fleet_visualization_bridge')
        self.subs=[];self.static={};self.pubs={}
        transient=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sensor=QoSProfile(depth=20,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.dynamic_pub=self.create_publisher(TFMessage,'/fleet/tf',100)
        self.static_pub=self.create_publisher(TFMessage,'/fleet/tf_static',transient)
        for robot in os.environ.get('FLEET_ROBOTS', 'tb1 tb2').split():
            for static,topic,qos in [(False,'tf',sensor),(True,'tf_static',transient)]:
                self.subs.append(self.create_subscription(TFMessage,f'/{robot}/{topic}',
                    lambda m,r=robot,s=static:self.transforms(r,s,m),qos))
            pub=self.create_publisher(String,f'/fleet/{robot}/robot_description',transient)
            self.subs.append(self.create_subscription(String,f'/{robot}/robot_description',
                lambda m,r=robot,p=pub:p.publish(String(data=prefixed_urdf(r,m.data))),transient))
            for suffix,typ,qos in [('scan',LaserScan,sensor),
                    ('local_costmap/costmap',OccupancyGrid,transient),
                    ('local_costmap/costmap_updates',OccupancyGridUpdate,10)]:
                pub=self.create_publisher(typ,f'/fleet/{robot}/{suffix}',qos)
                self.pubs[robot,suffix]=pub
                self.subs.append(self.create_subscription(typ,f'/{robot}/{suffix}',
                    lambda m,r=robot,p=pub:self.relay(r,p,m),qos))
    def relay(self,robot,pub,message):
        message=copy.deepcopy(message);message.header.frame_id=frame(robot,message.header.frame_id)
        pub.publish(message)
    def transforms(self,robot,static,message):
        result=copy.deepcopy(message)
        for transform in result.transforms:
            transform.header.frame_id=frame(robot,transform.header.frame_id)
            transform.child_frame_id=frame(robot,transform.child_frame_id)
            if static:self.static[transform.child_frame_id]=transform
        if static:self.static_pub.publish(TFMessage(transforms=list(self.static.values())))
        else:self.dynamic_pub.publish(result)

if __name__=='__main__':
    rclpy.init();node=FleetBridge()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.try_shutdown()
