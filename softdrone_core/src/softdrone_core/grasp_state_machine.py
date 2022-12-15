"""Custom state machine with slightly simplified logic."""
from softdrone.python.control.find_trajectory import find_polynomials, PolynomialInfo, LengthInfo
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf

from softdrone_core.msg import PolynomialTrajectory, LengthInfoMsg, GraspTrajectory, GraspCommand
from softdrone_core.utils import get_trajectory_viz_markers
from softdrone_core import InterpTrajectoryTracker
from softdrone_core.srv import SendGraspCommand, SendGraspCommandRequest

from nav_msgs.msg import Odometry
import geometry_msgs.msg
import mavros_msgs.msg
import mavros_msgs.srv
from std_msgs.msg import Bool, Int8
import std_srvs.srv
import std_msgs.msg
import numpy as np
import importlib
import rospy
import enum
import sys

def target_body_pva_to_global(b_pos, b_vel, b_acc, target_position, target_rotation, target_vel, target_omegas, target_acc=None):

    # rotate position to align with global frame
    g_pos = np.dot(target_rotation, b_pos)

    g_rel_pos = g_pos
    g_pos = g_pos + target_position

    # rotate velocity to align with global frame
    g_vel = np.dot(target_rotation, b_vel)
    # correct for rotating frame
    g_rel_vel = g_vel
    g_vel = g_rel_vel + np.cross(np.dot(target_rotation, target_omegas), g_rel_pos) + np.dot(target_rotation, target_vel)

    g_acc = np.dot(target_rotation, b_acc)

    # correct for centripetal and coriolis acceleration
    g_acc = g_acc + np.cross(target_omegas, np.cross(target_omegas, g_rel_pos)) + 2 * np.cross(target_omegas, g_rel_vel)

    # correct for einstein acceleration
    if target_acc is not None:
        g_acc = g_acc + target_acc

    return g_pos, g_vel, g_acc


class GraspDroneState(enum.Enum):
    """Current state of the drone."""

    WAITING_FOR_HOME = 0
    WAITING_FOR_ARM = 1
    WAITING_FOR_OFFBOARD = 2
    TAKEOFF = 3
    HOVER = 4
    MOVING_TO_START = 5
    SETTLE_BEFORE_ALIGNMENT = 6
    MOVING_TO_ALIGNED_START = 7
    SETTLE_BEFORE_GRASP = 8
    EXECUTING_MISSION = 9
    SETTLE_AFTER = 10
    RISE = 11
    MOVING_TO_DROP = 12
    DROP = 13
    MOVING_TO_HOME = 14
    HOVER_BEFORE_LAND = 15
    LAND = 16
    IDLE = 17
    HANDLE_SWITCH_TO_MANUAL = 18
    HANDLE_DISARM = 19


class SetpointType(enum.IntEnum):
    """Offboard setpoint types."""

    # default type (corresponds to full setpoint despite the name)
    POSITION = 0x0000
    # DO NOT USE (identical to POSITION but different name semantics)
    VELOCITY = 0x1000
    # mostly untested
    LOITER = 0x2000
    # suggested for use (tested)
    TAKEOFF = 0x3000
    # suggested for use (tested)
    LAND = 0x4000
    # DO NOT USE(works as intended, but corresponds to falling out of the sky)
    IDLE = 0x5000
    # suggested for use (is identical to POSITION)
    OFFBOARD = 0x6000
    # DO NOT USE (unhandled, falls out of the sky)
    FOLLOW_TARGET = 0x7000
    # identical to POSITION, but use of ground effect term is possible
    GRASP_TRAJECTORY = 0x8000
    # identical to POSITION, but ground effect is disabled
    GRASP_TRAJECTORY_NO_GE = 0x9000


def get_polynomial(
    polynomial_wrapper, planner, start, end, average_vel, safety_factor=1.4
):
    """Get the polynomial."""
    distance = np.linalg.norm(start - end)
    total_time = distance / average_vel * safety_factor
    polynomials = planner(start, end, time=total_time)
    return polynomial_wrapper(polynomials, total_time)


class GraspStateMachine:
    """Class for tracking a trajectory."""

    def __init__(self, mission_manager):
        """Set everything up."""
        self._mission_manager = mission_manager
        self._state = GraspDroneState.WAITING_FOR_HOME
        self._is_armed = False
        self._arm_time = None
        self._current_position = None
        self._home_position = None
        self._land_position = None

        self._current_waypoint_polynomial = None
        self._grasp_trajectory_tracker = None

        self._last_received_waypoint_msg = None
        self._last_received_trajectory_msg = None

        # Used to synchronize polynomial planning
        self._waiting_on_waypoint_request = False
        self._waiting_on_grasp_trajectory_request = False

        self._last_waypoint_polynomial_recv_from_planner = rospy.Time.now().to_sec()
        self._last_grasp_polynomial_recv_from_planner = rospy.Time.now().to_sec()
        ## Note these next two times are curently set in the callbacks when we receive
        # the polynomials, but they should probably be set by timestamps from the 
        # trajectory generator node
        self._last_waypoint_polynomial_update = None
        self._last_grasp_trajectory_update = rospy.Time.now().to_sec()
        self._grasp_attempted = False
        self._target_position = np.array([3.,3.,0.])
        self._target_position_fixed = None
        self._stop_updating_target_position = False

        self._target_vel = np.zeros(3)
        self._target_omegas = np.zeros(3)

        self._target_cov = np.eye(6)

        self._target_yaw = 0.0
        self._target_yaw_fixed = 0.0
        self._target_rotation = np.eye(3)
        self._target_rotation_fixed = np.eye(3)
        self._settle_after_pos = np.array([0.,0.,0.])

        self._current_grasp_command = None

        self._target_grasp_angle = rospy.get_param("~target_grasp_angle")
        self._grasp_start_horz_offset = rospy.get_param("~grasp_start_horz_offset")
        self._grasp_start_vert_offset = rospy.get_param("~grasp_start_vert_offset")
        self._fixed_grasp_start_point = rospy.get_param("~fixed_grasp_start_point")
        self._start_pos = np.array(rospy.get_param("~start_pos"))
        self._start_theta = rospy.get_param("~start_theta")

        self._land_offset = np.array(rospy.get_param("~land_offset"))

        self._grasp_start_distance = rospy.get_param("~grasp_start_distance")

        self._trajectory_settle_time = rospy.get_param("~trajectory_settle_time")

        self._drop_position = np.array(rospy.get_param("~drop_position"))

        self._state_durations = {}
        self._start_times = {}

        self._register_handlers()
        self._register_state_durations()
        self._register_transitions()

        self._takeoff_offset = rospy.get_param("~takeoff_offset")
        self._rise_offset = np.array(rospy.get_param("~rise_offset"))
        self._open_lengths = rospy.get_param("~open_lengths", [190, 190, 208, 208])
        self._land_threshold = rospy.get_param("~land_threshold", 0.001)
        self._dist_threshold = rospy.get_param("~dist_threshold", 0.2)
        self._replan_during_stages = rospy.get_param("~replan_during_stages", False)
        self._replan_during_grasp_trajectory = rospy.get_param("~replan_during_grasp_trajectory", False)
        self._grasp_attempted_tolerance = rospy.get_param("~grasp_attempted_tolerance", 1)
        self._average_polynomial_velocity = rospy.get_param(
            "~average_polynomial_velocity", 0.8
        )
        self._pose_counter = 0
        self._desired_yaw = rospy.get_param("~desired_yaw", 0.0)
        self._ground_effect_stop_time = rospy.get_param("~ground_effect_stop_time", 10.0)
    
        self._start_feedforward_z_acc = False
        self._feedforward_z_acc_start_time = None
        self._feedforward_z_acc_duration = rospy.get_param("~feedforward_z_acc_duration", 3.0)
        self._feedforward_z_acc = rospy.get_param("~feedforward_z_acc", 6.0)

        require_grasp_confirmation = rospy.get_param("~require_grasp_confirmation", False)
        if require_grasp_confirmation:
            self._grasp_start_ok = False
        else:
            self._grasp_start_ok = True

        #TODO(jared) combine gpio/serial nodes, setting to false doesn't enable serial without running launch for serial node
        self._enable_gpio_grasp = rospy.get_param("~enable_gpio_grasp")

        if self._enable_gpio_grasp:
            #rospy.wait_for_service('cmd_gripper')
            #self._gripper_client = rospy.ServiceProxy('cmd_gripper', SendGraspCommand)
            self._gripper_pub = rospy.Publisher('cmd_gripper_sub', Int8)

        rospy.Subscriber("~state", mavros_msgs.msg.State, self._state_callback, queue_size=10)
        rospy.Subscriber("~pose", PoseStamped, self._pose_callback, queue_size=10)

        rospy.Subscriber("~waypoint_polynomial", PolynomialTrajectory, self._waypoint_trajectory_cb)
        rospy.Subscriber("~grasp_trajectory", GraspTrajectory, self._grasp_trajectory_cb)

        rospy.Subscriber("~grasp_target", Odometry, self._target_pose_cb)

        #TODO organize vision / mocap logic better
        #rospy.Subscriber("~grasp_target_mocap", PoseStamped, self._target_pose_mocap_cb)

        rospy.Subscriber("~do_grasp", Bool, self._do_grasp_cb)

        self._target_pub = rospy.Publisher(
            "~target", mavros_msgs.msg.PositionTarget, queue_size=10
        )
        self._lengths_pub = rospy.Publisher(
            "~lengths", std_msgs.msg.Int64MultiArray, queue_size=10
        )

        self._tracker_reset_pub = rospy.Publisher(
                "~tracker_reset", Bool, queue_size=1
        )

        self._waypoint_request_pub = rospy.Publisher('~waypoint_trajectory_request', Bool, queue_size=1, latch=True)
        self._waypoint_request_target_frame_pub = rospy.Publisher('~waypoint_trajectory_target_frame_request', Bool, queue_size=1, latch=True)
        self._grasp_request_pub = rospy.Publisher('~grasp_trajectory_request', Bool, queue_size=1, latch=True)

        self.test_pub = rospy.Publisher("/global_target_odom", Odometry, queue_size=10)

        self._waypoint_pub = rospy.Publisher("~waypoint", PoseStamped, queue_size=1)

        control_rate = rospy.get_param("~control_rate", 100)
        if control_rate < 20.0:
            rospy.logfatal("Invalid control rate, must be above 20 hz")
            rospy.signal_shutdown("invalid control rate")
            return

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / control_rate), self._timer_callback
        )

    def _request_wp_once(self):
        if not self._waiting_on_waypoint_request:
            bool_msg = Bool()
            bool_msg.data = True
            self._waypoint_request_pub.publish(bool_msg)
            self._waiting_on_waypoint_request = True

    def _request_wp_target_frame_once(self):
        if not self._waiting_on_waypoint_request:
            bool_msg = Bool()
            bool_msg.data = True
            self._waypoint_request_target_frame_pub.publish(bool_msg)
            self._waiting_on_waypoint_request = True

    def _request_grasp_trajectory_once(self):
        if not self._waiting_on_grasp_trajectory_request:
            bool_msg = Bool()
            bool_msg.data = True
            self._grasp_request_pub.publish(bool_msg)
            self._waiting_on_grasp_trajectory_request = True

    def _do_grasp_cb(self, msg):
        self._grasp_start_ok = True

   # def _target_pose_mocap_cb(self, msg):
   #     self._target_position[0] = msg.pose.position.x
   #     self._target_position[1] = msg.pose.position.y
   #     self._target_position[2] = msg.pose.position.z

   #     # Hack to make coords align with d455 version
   #     quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
   #     q_rot = tf.transformations.quaternion_from_euler(np.pi, 0, 0)
   #     quat = tf.transformations.quaternion_multiply(q_rot, quat)
   #     
   #     
   #     (r, p, y) = tf.transformations.euler_from_quaternion(quat)

   #     self._target_yaw = y
   #     self._target_rotation = tf.transformations.quaternion_matrix(quat)[:3,:3]
   #     self._target_vel[0] = 0
   #     self._target_vel[1] = 0
   #     self._target_vel[2] = 0
   #     self._target_omegas[0] = 0
   #     self._target_omegas[1] = 0
   #     self._target_omegas[2] = 0

    def _target_pose_cb(self, msg):

        if not self._stop_updating_target_position:
            self._target_position[0] = msg.pose.pose.position.x
            self._target_position[1] = msg.pose.pose.position.y
            self._target_position[2] = msg.pose.pose.position.z
            quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            # TODO: Remove this hack once cad frames in proper frame.
            #q_rot = tf.transformations.quaternion_from_euler(np.pi, 0, 0)
            #quat = tf.transformations.quaternion_multiply(q_rot, quat)
            (r, p, y) = tf.transformations.euler_from_quaternion(quat)

            self._target_cov = np.array(msg.pose.covariance).reshape((6,6))
            #print('Target cov det: %f\n\n' % (np.linalg.det(self._target_cov)*1e6))
            #print('Target cov : {}\n\n'.format(self._target_cov))
            #print('\n\nTarget cov ros: {}\n\n'.format(msg.pose.covariance))

            self._target_yaw = y
            self._target_rotation = tf.transformations.quaternion_matrix(quat)[:3,:3]

            self._target_vel[0] = msg.twist.twist.linear.x
            self._target_vel[1] = msg.twist.twist.linear.y
            self._target_vel[2] = msg.twist.twist.linear.z
            self._target_omegas[0] = msg.twist.twist.angular.x
            self._target_omegas[1] = msg.twist.twist.angular.y
            self._target_omegas[2] = msg.twist.twist.angular.z

            theta_approach = self._target_grasp_angle 
            offset_vector = np.zeros(3)
            offset_vector[0] = np.cos(theta_approach) * self._grasp_start_horz_offset
            offset_vector[1] = np.sin(theta_approach) * self._grasp_start_horz_offset
            # Negateive here bc z points down
            offset_vector[2] = -self._grasp_start_vert_offset

            #self._loiter_at_point(settle_pos[0], settle_pos[1], settle_pos[2], yaw=self._desired_yaw)
            #g_vel = self._target_rotation.dot(self._target_vel)
            g_pos, g_vel, g_acc = target_body_pva_to_global(offset_vector, np.zeros(3), np.zeros(3), self._target_position, self._target_rotation, self._target_vel, self._target_omegas)
            global_pose = Odometry()
            global_pose.header = msg.header
            global_pose.pose.pose.position.x = g_pos[0]
            global_pose.pose.pose.position.y = g_pos[1]
            global_pose.pose.pose.position.z = g_pos[2]
            global_pose.twist.twist.linear.x = g_vel[0]
            global_pose.twist.twist.linear.y = g_vel[1]
            global_pose.twist.twist.linear.z = g_vel[2]
            self.test_pub.publish(global_pose)

    def _update_grasp_start_point(self):
        #if not self._fixed_grasp_start_point:
        # TODO: update for nonplanar target?
        theta_approach = self._target_yaw + self._target_grasp_angle
        offset_vector = np.zeros(3)
        offset_vector[0] = np.cos(theta_approach) * self._grasp_start_horz_offset
        offset_vector[1] = np.sin(theta_approach) * self._grasp_start_horz_offset
        offset_vector[2] = self._grasp_start_vert_offset
        grasp_start_pos = self._target_position + offset_vector
        self._grasp_start_pos = grasp_start_pos
        self._grasp_start_theta = theta_approach + np.pi
        self._desired_yaw = self._grasp_start_theta
        self._update_waypoint(self._grasp_start_pos, self._grasp_start_theta)

    def _update_waypoint(self, pos, yaw):
        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self._waypoint_pub.publish(msg)

    def _update_waypoint_trajectory(self):
        msg = self._last_received_waypoint_msg
        if msg is None:
            return
        poly_x = np.polynomial.polynomial.Polynomial(msg.coeffs_x)
        poly_y = np.polynomial.polynomial.Polynomial(msg.coeffs_y)
        poly_z = np.polynomial.polynomial.Polynomial(msg.coeffs_z)
        polys = [poly_x, poly_y, poly_z]
        polynomial_info = PolynomialInfo(polys, msg.time_end - msg.time_start)
        self._last_waypoint_polynomial_update = rospy.Time.now().to_sec()
        self._current_waypoint_polynomial = polynomial_info
        self._last_received_waypoint_msg = None

    def _waypoint_trajectory_cb(self, msg):
        self._waiting_on_waypoint_request = False
        self._last_waypoint_polynomial_recv_from_planner = rospy.Time.now().to_sec()
        self._last_received_waypoint_msg = msg

    def _grasp_trajectory_cb(self, msg):
        self._waiting_on_grasp_trajectory_request = False
        self._last_grasp_polynomial_recv_from_planner = rospy.Time.now().to_sec()
        self._last_received_trajectory_msg = msg

    def _update_grasp_trajectory(self):
        msg = self._last_received_trajectory_msg
        if msg is None:
            return
        if self._grasp_attempted:
            return
        poly_msg = msg.polynomial

        dt = poly_msg.time_end - poly_msg.time_start
        poly_msg.time_start = rospy.Time.now().to_sec()
        poly_msg.time_end = poly_msg.time_start + dt

        poly_x = np.polynomial.Polynomial(poly_msg.coeffs_x)
        poly_y = np.polynomial.Polynomial(poly_msg.coeffs_y)
        poly_z = np.polynomial.Polynomial(poly_msg.coeffs_z)
        polynomial = PolynomialInfo([poly_x, poly_y, poly_z], poly_msg.time_end - poly_msg.time_start)

        lng = msg.lengths
        init_lengths = np.array(lng.init_lengths).reshape((4, 4))
        open_lengths = np.array(lng.open_lengths).reshape((4, 4))
        grasp_lengths = np.array(lng.grasp_lengths).reshape((4, 4))
        grasp_lengths = LengthInfo(init_lengths, open_lengths, grasp_lengths, lng.open_time, lng.grasp_time)
        gripper_latency = lng.gripper_latency

        trajectory_tracker = InterpTrajectoryTracker(polynomial, grasp_lengths, gripper_latency=gripper_latency, settle_time=self._trajectory_settle_time, alpha=1.)

        self._last_grasp_trajectory_update = poly_msg.time_start
        self._grasp_trajectory_tracker = trajectory_tracker

    def _register_handlers(self):
        """Register default handlers and replace with a couple new ones."""
        self._state_handlers = {
            GraspDroneState.WAITING_FOR_HOME: self._handle_waiting_for_home,
            GraspDroneState.WAITING_FOR_ARM: self._handle_waiting_for_arm,
            GraspDroneState.WAITING_FOR_OFFBOARD: self._handle_offboard,
            GraspDroneState.TAKEOFF: self._handle_takeoff,
            GraspDroneState.HOVER: self._handle_hover,
            GraspDroneState.MOVING_TO_START: self._handle_moving_to_start,
            GraspDroneState.SETTLE_BEFORE_ALIGNMENT: self._handle_settle_before_alignment,
            GraspDroneState.MOVING_TO_ALIGNED_START: self._handle_moving_to_aligned_start,
            GraspDroneState.SETTLE_BEFORE_GRASP: self._handle_settle_before_grasp,
            GraspDroneState.EXECUTING_MISSION: self._handle_executing_mission,
            GraspDroneState.SETTLE_AFTER: self._handle_settle_after,
            GraspDroneState.RISE: self._handle_rise,
            GraspDroneState.MOVING_TO_DROP: self._handle_moving_to_drop,
            GraspDroneState.DROP: self._handle_drop,
            GraspDroneState.MOVING_TO_HOME: self._handle_moving_to_home,
            GraspDroneState.HOVER_BEFORE_LAND: self._handle_hover_before_land,
            GraspDroneState.LAND: self._handle_land,
            GraspDroneState.IDLE: self._handle_idle,
            GraspDroneState.HANDLE_SWITCH_TO_MANUAL: self._handle_switch_to_manual,
            GraspDroneState.HANDLE_DISARM: self._handle_disarm,
        }

    def _register_state_durations(self):
        # TODO(nathan) refactor with map from state to param name and default
        for state in GraspDroneState:
            if state == GraspDroneState.WAITING_FOR_HOME:
                continue  # this shouldn't ever time out
            if state == GraspDroneState.WAITING_FOR_ARM:
                continue  # this shouldn't ever time out
            if state == GraspDroneState.MOVING_TO_START:
                continue  # this shouldn't ever time out
            if state == GraspDroneState.EXECUTING_MISSION:
                continue  # this shouldn't ever time out
            if state == GraspDroneState.MOVING_TO_DROP:
                continue  # this shouldn't ever time out
            if state == GraspDroneState.MOVING_TO_HOME:
                continue  # this shouldn't ever time out
            if state == GraspDroneState.HANDLE_DISARM:
                continue  # this shouldn't ever time out
            if state == GraspDroneState.WAITING_FOR_OFFBOARD:
                self._state_durations[state] = rospy.Duration(
                    rospy.get_param("~offboard_wait_duration", 0.2)
                )
                continue
            if state == GraspDroneState.SETTLE_BEFORE_ALIGNMENT:
                self._state_durations[state] = rospy.Duration(
                    rospy.get_param("~mission_settle_duration", 5.)
                )
                continue
            if state == GraspDroneState.SETTLE_BEFORE_GRASP:
                self._state_durations[state] = rospy.Duration(
                    rospy.get_param("~mission_settle_duration", 5.)
                )
                continue
            if state == GraspDroneState.SETTLE_AFTER:
                self._state_durations[state] = rospy.Duration(
                    rospy.get_param("~mission_settle_duration", 2.0)
                )
                continue
            if state == GraspDroneState.DROP:
                self._state_durations[state] = rospy.Duration(
                    rospy.get_param("~drop_duration", 4.0)
                )
                continue
            if state == GraspDroneState.LAND:
                self._state_durations[state] = rospy.Duration(
                    rospy.get_param("~land_duration", 4.0)
                )
                continue

            self._state_durations[state] = rospy.Duration(
                rospy.get_param("~default_timeout", 3.0)
            )

    def _register_transitions(self):
        """Set transitions up."""
        # TODO(nathan) consider pairwise iteration instead
        self._state_transitions = {
            GraspDroneState(i): GraspDroneState(i + 1)
            for i in range(len(GraspDroneState) - 1)
        }
        self._state_transitions[GraspDroneState.SETTLE_AFTER] = GraspDroneState.MOVING_TO_DROP
        self._state_transitions[GraspDroneState.HANDLE_DISARM] = None

    def _send_lengths(self, lengths, scale=True):
        """Set a length target to the gripper."""
        msg = std_msgs.msg.Int64MultiArray()
        if scale:
            msg.data = [int(1000 * length) for length in lengths]
        else:
            msg.data = [int(length) for length in lengths]

        msg_dim = std_msgs.msg.MultiArrayDimension()
        msg_dim.label = "data"
        msg_dim.size = 4
        msg_dim.stride = 4

        msg.layout.dim.append(msg_dim)
        msg.layout.data_offset = 0
        self._lengths_pub.publish(msg)

    def _send_target(
        self, position, yaw=0.0, velocity=None, acceleration=None, is_grasp=False, use_ground_effect=True
    ):
        """Send a waypoint target to mavros."""
        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED

        if is_grasp:
            if use_ground_effect:
                msg.type_mask |= SetpointType.GRASP_TRAJECTORY
            else:
                msg.type_mask |= SetpointType.GRASP_TRAJECTORY_NO_GE

        # TODO(nathan) should enable commanding yaw rate, but not important yet
        msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_YAW_RATE

        msg.position.x = position[0]
        msg.position.y = position[1]
        msg.position.z = position[2]
        msg.yaw = yaw

        if velocity is not None:
            msg.velocity.x = velocity[0]
            msg.velocity.y = velocity[1]
            msg.velocity.z = velocity[2]
        else:
            msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_VX
            msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_VY
            msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_VZ

        if acceleration is not None:
            msg.acceleration_or_force.x = acceleration[0]
            msg.acceleration_or_force.y = acceleration[1]
            msg.acceleration_or_force.z = acceleration[2]
        else:
            msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_AFX
            msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_AFY
            msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_AFZ

        self._target_pub.publish(msg)

    def _timer_callback(self, msg):
        """Handle timer."""
        if self._state not in self._state_handlers:
            rospy.logfatal("State {} is not handled".format(self._state))
            rospy.signal_shutdown("unhandled state")
            return

        rospy.logwarn_throttle(1.0, "Current state: {}".format(self._state))
        if self._state_handlers[self._state]():
            if self._state_transitions[self._state] is None:
                rospy.logwarn("State machine finished at: {}".format(self._state))
                rospy.loginfo("Exiting!")
                self._timer.shutdown()
                rospy.signal_shutdown("finished executing")

            self._state = self._state_transitions[self._state]

    def _state_callback(self, msg):
        """Handle state callback."""
        if (not self._is_armed) and msg.armed:
            self._arm_time = rospy.Time.now()
        self._is_armed = msg.armed

    def _pose_callback(self, msg):
        """Handle pose callback."""
        self._current_position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ]
        )
        if np.all(self._current_position):
            self._pose_counter += 1
        if self._home_position is None and self._pose_counter > 400:
            self._home_position = self._current_position.copy()
            print("HOME")
            print(self._home_position)
            self._land_position = self._home_position + self._land_offset

    def _loiter_at_point(self, x, y, z, yaw=None):
        """Send a loiter command."""
        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LOITER
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        msg.yaw = self._desired_yaw if yaw is None else yaw

        self._target_pub.publish(msg)


    def _get_elapsed(self, state):
        """Check if a state has finished by time."""
        curr_time = rospy.Time.now()
        if state not in self._start_times:
            self._start_times[state] = curr_time
            return rospy.Duration(0.0)

        return curr_time - self._start_times[state]

    def _get_elapsed_ratio(self, state):
        """Check if a state has finished by time."""
        if state not in self._state_durations:
            rospy.logerror("State {} should not check time elapsed!".format(state))
            return False

        curr_time = rospy.Time.now()
        if state not in self._start_times:
            self._start_times[state] = curr_time
            return 0.0
        elapsed = (curr_time - self._start_times[state]).to_sec()
        return elapsed / self._state_durations[state].to_sec()

    def _has_elapsed(self, state):
        """Check if a state has finished by time."""
        if state not in self._state_durations:
            rospy.logerror("State {} should not check time elapsed!".format(state))
            return False

        curr_duration = self._get_elapsed(state)
        return curr_duration > self._state_durations[state]

    def _handle_waiting_for_home(self):
        """Wait for a valid home position from the drone, and a valid trajectory having been received."""
            
        self._update_grasp_trajectory()
        self._update_waypoint_trajectory()
        self._waiting_on_waypoint_request = False
        self._request_wp_once()
        if self._last_waypoint_polynomial_update is None:
            rospy.logwarn_throttle(1.0, 'Missing waypoint trajectory or target trajectory')
            return False

        return self._home_position is not None

    def _handle_waiting_for_arm(self):
        """Wait for the drone to be armed."""

        self._update_grasp_trajectory()
        self._update_waypoint_trajectory()

        #self._update_grasp_start_point()
        if self._arm_time is None:
            return False
        else:
            return self._is_armed and (rospy.Time.now() - self._arm_time > rospy.Duration(5))

    def _handle_offboard(self):
        """Publish to offboard and switch to offboard mode."""

        self._update_grasp_trajectory()
        self._update_waypoint_trajectory()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.IDLE

        self._target_pub.publish(msg)

        # this check is crucial. we need a valid target before switch to offboard
        if self._has_elapsed(GraspDroneState.WAITING_FOR_OFFBOARD):
            return False

        set_mode = rospy.ServiceProxy("mavros/set_mode", mavros_msgs.srv.SetMode)
        result = set_mode(0, "offboard")
        return result.mode_sent

    def _handle_takeoff(self):
        """Use the takeoff command to go to the right height."""

        self._update_grasp_trajectory()
        self._update_waypoint_trajectory()

        absolute_takeoff_height = self._home_position[2] + self._takeoff_offset

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.TAKEOFF
        msg.position.z = absolute_takeoff_height
        print("ABSOLUTE")
        print(absolute_takeoff_height)

        self._target_pub.publish(msg)

        diff = abs(absolute_takeoff_height - self._current_position[2])
        return diff < self._dist_threshold or self._has_elapsed(GraspDroneState.TAKEOFF)

    def _handle_hover(self):
        """Use loiter command to hang out at hover setpoint."""

        self._update_grasp_trajectory()
        self._update_waypoint(self._start_pos, self._start_theta)
        self._loiter_at_point(
            self._home_position[0],
            self._home_position[1],
            self._home_position[2] + self._takeoff_offset,
            yaw=self._desired_yaw * self._get_elapsed_ratio(GraspDroneState.HOVER)
        )

        req_traj = self._has_elapsed(GraspDroneState.HOVER)
        if req_traj:
            self._request_wp_once()
            self._update_waypoint_trajectory()
        proceed = req_traj and (rospy.Time.now().to_sec() - self._last_waypoint_polynomial_recv_from_planner) < .1
        return proceed

    def _handle_moving_to_start(self):
        """Move from the takeoff position to arbritrary start point."""

        if self._replan_during_stages:
            self._update_grasp_trajectory()
            self._update_waypoint_trajectory()

        #self._update_grasp_start_point()
        elapsed = rospy.Time.now().to_sec() - self._last_waypoint_polynomial_update
        curr_poly = self._current_waypoint_polynomial
        if elapsed >= curr_poly._total_time:
            reset_msg = Bool()
            reset_msg.data = True
            self._tracker_reset_pub.publish(reset_msg)
            return True

        pos, vel, acc = curr_poly.interp(elapsed)
        yaw_scale = min(elapsed / curr_poly._total_time, 1.0)
        self._send_target(pos, yaw=self._start_theta * yaw_scale, velocity=vel, acceleration=acc)
        return False

    def _handle_settle_before_alignment(self):
        """Use loiter command to stop before executing the grasp."""
        self._update_grasp_trajectory()
        self._update_waypoint_trajectory()
        #self._update_waypoint(self._land_position, 0)
        self._update_grasp_start_point()
        #settle_pos = self._grasp_start_pos
        #self._loiter_at_point(settle_pos[0], settle_pos[1], settle_pos[2])
        self._loiter_at_point(self._start_pos[0], self._start_pos[1], self._start_pos[2], yaw=self._start_theta)
        req_traj = self._has_elapsed(GraspDroneState.SETTLE_BEFORE_ALIGNMENT)
        if req_traj:
            self._request_wp_target_frame_once()
            self._update_waypoint_trajectory()
        proceed = req_traj and (rospy.Time.now().to_sec() - self._last_waypoint_polynomial_recv_from_planner) < .1
        return proceed
        #req_traj = self._has_elapsed(GraspDroneState.SETTLE_BEFORE) and self._grasp_start_ok
        #if req_traj:
        #    self._request_grasp_trajectory_once()
        #proceed = req_traj and (rospy.Time.now().to_sec() - self._last_grasp_polynomial_recv_from_planner) < .1 and self._grasp_trajectory_tracker is not None
        #self._target_position_fixed = self._target_position.copy()
        #self._target_yaw_fixed = self._target_yaw
        #self._target_rotation_fixed = self._target_rotation
        #grasp_cmd = Int8()
        #grasp_cmd.data = GraspCommand.OPEN_PARTIAL
        #self._gripper_pub.publish(grasp_cmd)
        #if proceed and self._enable_gpio_grasp:
        #    grasp_cmd.data = GraspCommand.OPEN_ASYMMETRIC
        #    try:
        #        rospy.logwarn("Called open gripper!")
        #        self._gripper_pub.publish(grasp_cmd)
        #    except rospy.ServiceException as e:
        #        print("Service call failed to open gripper: %s"%e)
        #else:
        #    grasp_cmd.data = GraspCommand.OPEN_PARTIAL
        #
        #return proceed

    def _handle_moving_to_aligned_start(self):
        elapsed = rospy.Time.now().to_sec() - self._last_waypoint_polynomial_update
        curr_poly = self._current_waypoint_polynomial
        if elapsed >= curr_poly._total_time:
            #reset_msg = Bool()
            #reset_msg.data = True
            #self._tracker_reset_pub.publish(reset_msg)
            return True

        pos, vel, acc = curr_poly.interp(elapsed)
        g_pos, g_vel, g_acc = target_body_pva_to_global(pos, vel, acc, self._target_position, self._target_rotation, self._target_vel, self._target_omegas)
        drone_wrt_target = self._current_position - self._target_position
        yaw = np.arctan2(drone_wrt_target[1], drone_wrt_target[0])
        self._send_target(g_pos, yaw=yaw + np.pi, velocity=g_vel, acceleration=g_acc)
        return False

    def _handle_settle_before_grasp(self):
        """Maintain grasp start position before executing."""
        self._update_grasp_trajectory()
        self._update_waypoint_trajectory()
        self._update_grasp_start_point()
        settle_pos = self._grasp_start_pos

        theta_approach = self._target_grasp_angle 
        offset_vector = np.zeros(3)
        offset_vector[0] = np.cos(theta_approach) * self._grasp_start_horz_offset
        offset_vector[1] = np.sin(theta_approach) * self._grasp_start_horz_offset
        # Negateive here bc z points down
        offset_vector[2] = -self._grasp_start_vert_offset

        #self._loiter_at_point(settle_pos[0], settle_pos[1], settle_pos[2], yaw=self._desired_yaw)
        #g_vel = self._target_rotation.dot(self._target_vel)
        g_pos, g_vel, g_acc = target_body_pva_to_global(offset_vector, np.zeros(3), np.zeros(3), self._target_position, self._target_rotation, self._target_vel, self._target_omegas)
        #self._send_target(settle_pos, velocity=g_vel, yaw=self._desired_yaw)
        self._send_target(g_pos, yaw=self._desired_yaw, velocity=g_vel, acceleration=g_acc)
        req_traj = self._has_elapsed(GraspDroneState.SETTLE_BEFORE_GRASP) and self._grasp_start_ok
        if req_traj:
            self._request_grasp_trajectory_once()
        proceed = req_traj and (rospy.Time.now().to_sec() - self._last_grasp_polynomial_recv_from_planner) < .1 and self._grasp_trajectory_tracker is not None
        self._target_position_fixed = self._target_position.copy()
        self._target_yaw_fixed = self._target_yaw
        self._target_rotation_fixed = self._target_rotation
        grasp_cmd = Int8()
        grasp_cmd.data = GraspCommand.OPEN_PARTIAL
        self._gripper_pub.publish(grasp_cmd)
        return proceed

    def _handle_executing_mission(self):
        """State handler for EXECUTING_MISSION."""

        if not self._grasp_attempted:
            theta_approach = self._target_yaw + self._target_grasp_angle
            #theta_approach = self._target_yaw_fixed + self._target_grasp_angle
            self._desired_yaw = theta_approach + np.pi

            # Hack for mocap moving target
            grasp_cmd = Int8()
            grasp_cmd.data = GraspCommand.OPEN_ASYMMETRIC
            self._gripper_pub.publish(grasp_cmd)


        #if not self._replan_during_grasp_trajectory:
            # results in not replanning after starting to follow the grasp trajectory
        #    self._grasp_attempted = True

        elapsed = rospy.Time.now().to_sec() - self._last_grasp_trajectory_update
        result = self._grasp_trajectory_tracker._run_normal(elapsed)
        # For fixed target
        #g_pos, g_vel, g_acc = target_body_pva_to_global(result.position, result.velocity, result.acceleration, self._target_position_fixed, self._target_rotation_fixed, np.zeros(3), np.zeros(3))
        # For moving target
        g_pos, g_vel, g_acc = target_body_pva_to_global(result.position, result.velocity, result.acceleration, self._target_position, self._target_rotation, self._target_vel, self._target_omegas)

        self._settle_after_pos = g_pos

        if self._enable_gpio_grasp:
            #lat_target_dist = np.linalg.norm(self._target_position_fixed[:2] - self._current_position[:2])
            #lat_target_dist = np.linalg.norm(self._target_position_fixed[0] - self._current_position[0])
            #lat_target_dist = self._target_position_fixed[0] - self._current_position[0]
            # TODO: Replace with tf tree
            R = self._target_rotation
            t = self._target_position.reshape(3,1)
            tf = np.hstack((R, t))
            tf = np.vstack((tf, [0,0,0,1]))
            pos = np.append(self._current_position, 1)
            lat_target_dist = np.linalg.inv(tf).dot(pos)[0]
            if lat_target_dist < self._grasp_attempted_tolerance and not self._grasp_attempted:
                self._current_grasp_command = GraspCommand.OPEN_ASYMMETRIC
                grasp_cmd = Int8()
                grasp_cmd.data = GraspCommand.OPEN_ASYMMETRIC
                self._gripper_pub.publish(grasp_cmd)
                rospy.loginfo("Called open asymmetric gripper!")
                self._grasp_attempted = True
                self._stop_updating_target_position = True
                #self._current_grasp_command = GraspCommand.OPEN_ASYMMETRIC
            #if np.linalg.norm(self._target_position_fixed[:2] - self._current_position[:2]) < self._grasp_attempted_tolerance:
                #if not self._grasp_attempted:
                    #self._current_grasp_command = GraspCommand.OPEN_ASYMMETRIC
                    #grasp_cmd = Int8()
                    #grasp_cmd.data = GraspCommand.OPEN_ASYMMETRIC
                    #self._gripper_pub.publish(grasp_cmd)
                    #rospy.loginfo("Called open asymmetric gripper!")
                # stop updating the grasp trajectory after we attempt the grasp. If we didn't do this, we would
                # keep going back to the grasp point
                #self._grasp_attempted = True
                #self._stop_updating_target_position = True

                #lat_target_dist = np.linalg.norm(self._target_position[:2] - self._current_position[:2])
                #lat_target_dist = np.linalg.norm(self._target_position_fixed[:2] - self._current_position[:2])
            if lat_target_dist < self._grasp_start_distance:
                self._current_grasp_command = GraspCommand.CLOSE
                grasp_cmd = Int8()
                grasp_cmd.data = GraspCommand.CLOSE
                self._gripper_pub.publish(grasp_cmd)
                rospy.loginfo("Called close gripper!")

                if not self._start_feedforward_z_acc:
                    self._start_feedforward_z_acc = True
                    self._feedforward_z_acc_start_time = rospy.Time.now().to_sec() + 0.3
                    #self._feedforward_z_acc_start_time = rospy.Time.now().to_sec() + 0.7 #slow vision

            #grasp_cmd = Int8()
            #grasp_cmd.data = self._current_grasp_command
            #self._gripper_pub.publish(grasp_cmd)
            

        feedforward_z_acc = 0
        if self._start_feedforward_z_acc:
            if rospy.Time.now().to_sec() < self._feedforward_z_acc_start_time:
                feedforward_z_acc = 0
            elif rospy.Time.now().to_sec() - self._feedforward_z_acc_start_time < self._feedforward_z_acc_duration:
                feedforward_z_acc = self._feedforward_z_acc
            

        self._send_target(
            g_pos,
            yaw=self._desired_yaw,
            velocity=g_vel,
            acceleration=g_acc + np.array([0., 0., feedforward_z_acc]),
            is_grasp=True,
            use_ground_effect=elapsed < self._ground_effect_stop_time,
        )

        return result.finished

    def _handle_settle_after(self):
        """Use loiter command to stop after executing the grasp."""

        self._update_grasp_trajectory()

        self._update_waypoint(self._drop_position, 0)
        settle_pos = self._settle_after_pos
        self._loiter_at_point(settle_pos[0], settle_pos[1], settle_pos[2])
        req_traj = self._has_elapsed(GraspDroneState.SETTLE_AFTER)
        if req_traj:
            self._request_wp_once()
            self._update_waypoint_trajectory()
        proceed = req_traj and (rospy.Time.now().to_sec() - self._last_waypoint_polynomial_recv_from_planner) < .1
        return proceed

    def _handle_rise(self):
        """Use loiter command to hang out at hover setpoint."""

        self._update_grasp_trajectory()
        self._update_waypoint_trajectory()

        rise_pos = self._grasp_trajectory_tracker.get_end() + self._rise_offset
        self._loiter_at_point(rise_pos[0], rise_pos[1], rise_pos[2])
        return self._has_elapsed(GraspDroneState.RISE)

    def _handle_moving_to_drop(self):
        """Move to the drop position"""

        if self._replan_during_stages:
            self._update_grasp_trajectory()
            self._update_waypoint_trajectory()

        self._update_waypoint(self._drop_position, 0)
        elapsed = rospy.Time.now().to_sec() - self._last_waypoint_polynomial_update
        curr_poly = self._current_waypoint_polynomial
        if elapsed >= curr_poly._total_time:
            return True

        pos, vel, acc = curr_poly.interp(elapsed)
        self._send_target(pos, yaw=self._desired_yaw, velocity=vel, acceleration=acc)
        return False

    def _handle_drop(self):
        """Wait while we drop the target."""
        self._update_grasp_trajectory()
        self._update_waypoint(self._land_position, 0)
        drop_pos = self._drop_position
        self._loiter_at_point(drop_pos[0], drop_pos[1], drop_pos[2])
        self._send_lengths(self._open_lengths, scale=False)
        if self._enable_gpio_grasp:
            try:
                #grasp_cmd = GraspCommand()
                #grasp_cmd.cmd = GraspCommand.DEFAULT
                grasp_cmd = Int8()
                grasp_cmd.data = GraspCommand.CLOSE
                #self._gripper_client(grasp_cmd)
                self._gripper_pub.publish(grasp_cmd)

            except rospy.ServiceException as e:
                print("Service call failed to open gripper: %s"%e)
        req_traj = self._has_elapsed(GraspDroneState.DROP)
        if req_traj:
            self._request_wp_once()
            self._update_waypoint_trajectory()
        proceed = req_traj and (rospy.Time.now().to_sec() - self._last_waypoint_polynomial_recv_from_planner) < .1
        return proceed

    def _handle_moving_to_home(self):
        """Move from dropping the target to hovering above home position."""

        if self._replan_during_stages:
            self._update_grasp_trajectory()
            self._update_waypoint_trajectory()

        self._update_waypoint(self._land_position, 0)
        elapsed = rospy.Time.now().to_sec() - self._last_waypoint_polynomial_update
        curr_poly = self._current_waypoint_polynomial
        if elapsed >= curr_poly._total_time:
            return True

        pos, vel, acc = curr_poly.interp(elapsed)
        self._send_target(pos, yaw=self._desired_yaw, velocity=vel, acceleration=acc)
        return False

    def _handle_hover_before_land(self):
        """Use loiter command to hang out at hover setpoint."""
        #yaw_scale = 0.8 - self._get_elapsed_ratio(GraspDroneState.HOVER_BEFORE_LAND)
        #yaw_scale = 0.0 if yaw_scale < 0.0 else yaw_scale
        yaw_scale = 1.
        self._loiter_at_point(
            self._land_position[0],
            self._land_position[1],
            self._land_position[2] + self._takeoff_offset,
            yaw=self._desired_yaw * yaw_scale,
        )
        return self._has_elapsed(GraspDroneState.HOVER_BEFORE_LAND)

    def _handle_land(self):
        """Use land command to descend safely."""
        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LAND
        msg.position.x = self._land_position[0]
        msg.position.y = self._land_position[1]
        msg.position.z = self._land_position[2]  # this gets ignored
        self._target_pub.publish(msg)
        # diff = abs(self._land_position[2] - self._current_position[2])
        # return diff < self._land_threshold or self._has_elapsed(GraspDroneState.LAND)
        return self._has_elapsed(GraspDroneState.LAND)

    def _handle_idle(self):
        """Turn motors to idle to fully land."""
        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.type_mask |= SetpointType.IDLE
        self._target_pub.publish(msg)
        return self._has_elapsed(GraspDroneState.IDLE)

    def _handle_switch_to_manual(self):
        """Switch out of offboard."""
        set_mode = rospy.ServiceProxy("mavros/set_mode", mavros_msgs.srv.SetMode)
        result = set_mode(0, "manual")
        return result.mode_sent

    def _handle_disarm(self):
        """Disarm the drone."""
        set_mode = rospy.ServiceProxy("mavros/cmd/arming", mavros_msgs.srv.CommandBool)
        result = set_mode(False)
        return result.success
