"""RViz bound markers; prediction lives in execution_prediction.py."""
import math
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

BOUND_RADIUS = .25

def publish_bounds(audit, publisher):
    markers = MarkerArray()
    for i, (robot, p) in enumerate(sorted(audit.poses.items())):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = audit.get_clock().now().to_msg()
        m.ns = 'conflict_trigger_bounds'; m.id = i
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.pose.orientation.w = 1.
        m.scale.x = .018
        m.color.r, m.color.g, m.color.b = [(0.1, .5, 1.), (1., .45, .1), (.2, .8, .3)][i]
        m.color.a = 1.
        m.points = [Point(x=p[0]+BOUND_RADIUS*math.cos(j*math.tau/64),
                          y=p[1]+BOUND_RADIUS*math.sin(j*math.tau/64), z=.08)
                    for j in range(65)]
        markers.markers.append(m)
    publisher.publish(markers)
