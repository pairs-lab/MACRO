"""ROS-side metric guards reusing MARCO's exact swept-separation primitive.

Load the dependency-free source module, not standalone's GUI package facade.
The disc radius must be identical to the radius supplied to MARCO.
"""
import math
from core.coordination.dynamic_collision import swept_separation
ROBOT_RADIUS = .22


def padded_bounds_overlap(a,b,margin=.02):
    """Separating-axis test on the same padded SDF envelope shown in RViz.

    Requires at least 2 cm separation on a separating axis, in addition to
    the footprint's 1 cm padding. AMCL uncertainty is not a safety proof.
    """
    def corners(p):
        x,y,yaw=p;c,s=math.cos(yaw),math.sin(yaw)
        return [(x+c*u-s*v,y+s*u+c*v) for u,v in
                [(-.207,-.164),(-.207,.164),(.079,.164),(.079,-.164)]]
    polygons=[corners(a),corners(b)]
    for polygon in polygons:
        for i in range(4):
            x,y=polygon[i];u,v=polygon[(i+1)%4]
            nx,ny=-(v-y),u-x;norm=math.hypot(nx,ny);nx/=norm;ny/=norm
            projections=[[nx*x+ny*y for x,y in p] for p in polygons]
            if max(projections[0])+margin<min(projections[1]) or max(projections[1])+margin<min(projections[0]):
                return False
    return True

def passage_is_clear(actual,protected,tracking_margin):
    """Release condition depends on the protected passage, not goal error."""
    if not corridor_clear(actual,protected,tracking_margin):return False
    for i,(x,y) in enumerate(protected):
        a=protected[max(0,i-1)];b=protected[min(len(protected)-1,i+1)]
        yaw=math.atan2(b[1]-a[1],b[0]-a[0])
        if padded_bounds_overlap(actual,(x,y,yaw),margin=.02):return False
    return True


def corridor_distance(position, path):
    """Continuous centre-to-polyline distance, including between grid cells."""
    if not path:
        raise ValueError('A protected corridor cannot be empty')
    if len(path) == 1:
        return math.dist(position[:2], path[0][:2])
    return min(swept_separation(position[:2], position[:2], a[:2], b[:2])
               for a, b in zip(path, path[1:]))


def corridor_clear(position, path, margin=0.):
    return corridor_distance(position, path) > 2 * ROBOT_RADIUS + margin


def remaining_route(position, path):
    if not path:
        return [list(position[:2])]
    index = min(range(len(path)), key=lambda i: math.dist(position[:2], path[i][:2]))
    return [list(position[:2])] + [list(p[:2]) for p in path[index + 1:]]


def routes_are_disjoint(first, second, margin=.10):
    """Conservative spatial gate over both complete remaining swept corridors.

    Unlike equal-time preview, this remains safe under unequal progress.
    """
    if not first or not second:
        return False
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    segments=lambda route: list(zip(route, route[1:])) or [(route[0],route[0])]
    threshold=2*ROBOT_RADIUS+margin
    for a,b in segments(first):
        for c,d in segments(second):
            if (max(a[0],b[0])+threshold<min(c[0],d[0]) or
                max(c[0],d[0])+threshold<min(a[0],b[0]) or
                max(a[1],b[1])+threshold<min(c[1],d[1]) or
                max(c[1],d[1])+threshold<min(a[1],b[1])):
                continue
            # Proper intersections have zero separation; collinear overlap
            # and endpoint contacts are covered by point-to-segment distances.
            if cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0:
                return False
            distance=min(corridor_distance(p,segment) for p,segment in
                         [(a,[c,d]),(b,[c,d]),(c,[a,b]),(d,[a,b])])
            if distance<=threshold:return False
    return True


def route_position(route, distance):
    for a, b in zip(route, route[1:]):
        length = math.dist(a, b)
        if length and distance <= length:
            return [a[j] + (b[j] - a[j]) * distance / length for j in range(2)]
        distance -= length
    return route[-1]


