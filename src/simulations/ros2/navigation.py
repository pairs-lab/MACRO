"""Observed ROS robot state, motor holds and planned-path visualization."""
import math
import time
from rclpy.parameter import Parameter
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path as RosPath, Odometry
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from rclpy.time import Time

def pose(x,y,yaw=0.):
    p=PoseStamped();p.header.frame_id='map';p.pose.position.x=float(x);p.pose.position.y=float(y)
    p.pose.orientation.z=math.sin(yaw/2);p.pose.orientation.w=math.cos(yaw/2)
    return p

class Audit(Node):
    def __init__(self, robots=('tb1','tb2')):
        super().__init__('nav2_conflict_audit',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.poses={};self.pose_times={};self.odom={};self.samples=[];self.paths={};self.subs=[];self.pubs={}
        self.amcl_poses={};self.twists={};self.buffers={r:Buffer() for r in robots}
        self.holding=set();self.stop_pubs={};self.command_samples=[];self.hold_integrals={}
        for robot in robots:
            for topic in ['cmd_vel','cmd_vel_nav']:
                self.stop_pubs[robot,topic]=self.create_publisher(Twist,f'/{robot}/{topic}',10)
                self.subs.append(self.create_subscription(Twist,f'/{robot}/{topic}',
                    lambda m,r=robot,t=topic:self.command_samples.append(
                        {'robot':r,'topic':t,'linear':m.linear.x,'angular':m.angular.z,'wall_time':time.time()}),10))
            self.subs.append(self.create_subscription(PoseWithCovarianceStamped,f'/{robot}/amcl_pose',
                lambda m,r=robot:self.amcl(r,m),
                QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)))
            self.subs.append(self.create_subscription(Odometry,f'/{robot}/odom',
                lambda m,r=robot:self.odometry(r,m),10))
            self.subs.append(self.create_subscription(TFMessage,f'/{robot}/tf',
                lambda m,r=robot:self.transforms(r,m),100))
            self.subs.append(self.create_subscription(TFMessage,f'/{robot}/tf_static',
                lambda m,r=robot:self.transforms(r,m,True),
                QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)))
            self.subs.append(self.create_subscription(RosPath,f'/{robot}/plan',
                lambda m,r=robot:self.paths.setdefault(r,[]).append(
                    [[p.pose.position.x,p.pose.position.y] for p in m.poses]),10))
            for stage in ['clearance','resume']:
                self.pubs[robot,stage]=self.create_publisher(RosPath,f'/{robot}/marco_{stage}',
                    QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(.05,self.hold_commands)
    def hold_commands(self):
        # A cancelled action is not a motor hold. Measured residual yaw motion
        # can remain with zero wheel commands in this Gazebo model. Apply a
        # small bounded velocity-feedback brake, without translating the robot.
        # Remove it from holding BEFORE starting any new controller action.
        for robot in tuple(self.hold_integrals):
            if robot not in self.holding:self.hold_integrals.pop(robot)
        for robot in tuple(self.holding):
            command=Twist()
            angular=self.twists.get(robot,(0.,0.))[1]
            integral=max(-.10,min(.10,self.hold_integrals.get(robot,0.)+.05*angular))
            self.hold_integrals[robot]=integral
            command.angular.z=max(-.10,min(.10,-.5*angular-.5*integral))
            for topic in ['cmd_vel_nav','cmd_vel']:
                self.stop_pubs[robot,topic].publish(command)
    def amcl(self,r,m):
        p,q=m.pose.pose.position,m.pose.pose.orientation
        self.amcl_poses[r]=[p.x,p.y,math.atan2(2*q.w*q.z,1-2*q.z*q.z)]
        self.samples.append({'robot':r,'pose':self.amcl_poses[r], 'wall_time':time.time(),
            'covariance_xy':[m.pose.covariance[0],m.pose.covariance[7]],
            'odom':self.odom.get(r)})
    def odometry(self,r,m):
        self.odom[r]=(m.pose.pose.position.x,m.pose.pose.position.y)
        self.twists[r]=(m.twist.twist.linear.x,m.twist.twist.angular.z)
    def transforms(self,r,m,static=False):
        buffer=self.buffers[r]
        for transform in m.transforms:
            if static:buffer.set_transform_static(transform,'fleet_adapter')
            else:buffer.set_transform(transform,'fleet_adapter')
        try:
            transform=buffer.lookup_transform('map','base_footprint',Time())
        except Exception:
            return
        p,q=transform.transform.translation,transform.transform.rotation
        self.poses[r]=[p.x,p.y,math.atan2(2*q.w*q.z,1-2*q.z*q.z)]
        self.pose_times[r]=time.monotonic()
    def pump(self):
        time.sleep(.02)  # Dedicated executor receives both robots independently.
    def publish(self,r,stage,path):
        message=RosPath();message.header.frame_id='map';message.header.stamp=self.get_clock().now().to_msg()
        for i,(x,y) in enumerate(path):
            nxt=path[min(i+1,len(path)-1)]
            direction=(nxt[0]-x,nxt[1]-y)
            if i==len(path)-1 and i>0:
                direction=(x-path[i-1][0],y-path[i-1][1])
            p=pose(x,y,math.atan2(direction[1],direction[0]));p.header=message.header;message.poses.append(p)
        self.pubs[r,stage].publish(message)
        return message
