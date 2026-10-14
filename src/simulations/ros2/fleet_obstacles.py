"""Inject known peer padded footprints into local maps; AMCL map unchanged."""
import copy
import math
import os
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped

class FleetObstacles(Node):
    def __init__(self):
        super().__init__('fleet_footprint_obstacles',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.map=None;self.poses={};self.subs=[];self.pubs={}
        q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subs.append(self.create_subscription(OccupancyGrid,'/tb1/map',self.receive_map,q))
        self.robots=os.environ.get('FLEET_ROBOTS','tb1 tb2').split()
        for r in self.robots:
            self.subs.append(self.create_subscription(PoseWithCovarianceStamped,f'/{r}/amcl_pose',
                lambda m,r=r:self.poses.update({r:m.pose.pose}),q))
            self.pubs[r]=self.create_publisher(OccupancyGrid,f'/fleet/{r}/obstacle_map',q)
        self.create_timer(.2,self.publish)
    def receive_map(self,message):self.map=message
    def publish(self):
        if self.map is None:return
        for own in self.robots:
            message=copy.deepcopy(self.map);message.header.stamp=self.get_clock().now().to_msg()
            for other in self.robots:
                if other==own or other not in self.poses:continue
                pose=self.poses[other];x,y=pose.position.x,pose.position.y;q=pose.orientation
                yaw=math.atan2(2*q.w*q.z,1-2*q.z*q.z);c,s=math.cos(yaw),math.sin(yaw)
                # Exact rectangular envelope includes padding, not just lidar cylinder.
                res=message.info.resolution;ox=message.info.origin.position.x;oy=message.info.origin.position.y
                w,h=message.info.width,message.info.height
                for row in range(max(0,int((y-.35-oy)/res)),min(h,int((y+.35-oy)/res)+1)):
                    for col in range(max(0,int((x-.35-ox)/res)),min(w,int((x+.35-ox)/res)+1)):
                        dx,dy=ox+(col+.5)*res-x,oy+(row+.5)*res-y
                        u,v=c*dx+s*dy,-s*dx+c*dy
                        # Half-cell raster padding encloses boundary cells.
                        if -.207-res/2<=u<=.079+res/2 and abs(v)<=.164+res/2:
                            message.data[row*w+col]=100
            self.pubs[own].publish(message)

if __name__=='__main__':
    rclpy.init();node=FleetObstacles()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.try_shutdown()
