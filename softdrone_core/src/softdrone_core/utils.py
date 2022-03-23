import rospy
from visualization_msgs.msg import Marker, MarkerArray

def get_trajectory_viz_markers(pts, ns, ix, color, frame_id="map"):
    strip_marker = Marker()
    strip_marker.header.frame_id = frame_id
    strip_marker.header.stamp = rospy.Time.now()
    strip_marker.ns = ns + '_line'
    strip_marker.id = ix
    strip_marker.type = Marker.LINE_STRIP
    strip_marker.action = Marker.ADD
    strip_marker.pose.orientation.w = 1.0
    strip_marker.scale.x = 0.05
    strip_marker.color.a = 0.5
    strip_marker.color.r = color[0]
    strip_marker.color.g = color[1]
    strip_marker.color.b = color[2]
    strip_marker.points = pts

    sphere_marker = Marker()
    sphere_marker.header.frame_id = "map"
    sphere_marker.header.stamp = rospy.Time.now()
    sphere_marker.ns = ns + '_pts'
    sphere_marker.id = ix + 1
    sphere_marker.type = Marker.SPHERE_LIST
    sphere_marker.action = Marker.ADD
    sphere_marker.pose.orientation.w = 1.0
    sphere_marker.scale.x = 0.1
    sphere_marker.scale.y = 0.1
    sphere_marker.scale.z = 0.1
    sphere_marker.color.a = 0.5
    sphere_marker.color.r = color[0]
    sphere_marker.color.g = color[1]
    sphere_marker.color.b = color[2]
    sphere_marker.points = pts

    ma = MarkerArray()
    ma.markers.append(strip_marker)
    ma.markers.append(sphere_marker)
    return ma
