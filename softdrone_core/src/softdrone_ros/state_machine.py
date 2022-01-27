"""Main state machine for testing."""
import rospy
import enum

import numpy as np

import geometry_msgs.msg
import mavros_msgs.msg
import std_msgs.msg


class DroneState(enum.Enum):
    """Current state of the drone."""

    WAITING_FOR_HOME = 0
    WAITING_FOR_ARM = 1
    TAKEOFF = 2
    MOVING_TO_START = 3
    EXECUTING_MISSION = 4
    MOVING_TO_HOME = 5
    FINISHED = 6


class MissionRunResult:
    """Quick class to hold some info together."""

    def __init__(
        self,
        position,
        yaw=0.0,
        velocity=None,
        acceleration=None,
        lengths=None,
        finished=False,
    ):
        """Set everything up."""
        self.position = position
        self.yaw = yaw
        self.velocity = velocity
        self.acceleration = acceleration
        self.lengths = lengths
        self.finished = finished


class StateMachine(object):
    """Class for tracking a trajectory."""

    def __init__(self, mission_manager, initial_state=None):
        """Set everything up."""
        control_rate = rospy.get_param("~control_rate", 100)
        if control_rate < 20.0:
            rospy.logfatal("Invalid control rate, must be above 20 hz")
            rospy.signal_shutdown("invalid control rate")
            return

        self._dist_threshold = rospy.get_param("~dist_threshold", 0.2)
        self._register_handlers()
        self._register_transitions()

        self._mission_manager = mission_manager

        if initial_state is None:
            self._state = DroneState.WAITING_FOR_HOME
        else:
            self._state = initial_state

        self._is_armed = False
        self._current_position = None
        self._home_position = None
        self._hover_position = None
        self._settle_start_time = None
        self._settle_duration = rospy.Duration(
            rospy.get_param("~start_settle_duration", 2.0)
        )

        self._state_sub = rospy.Subscriber(
            "state", mavros_msgs.msg.State, self._state_callback, queue_size=10
        )
        self._pose_sub = rospy.Subscriber(
            "pose", geometry_msgs.msg.PoseStamped, self._pose_callback, queue_size=10
        )
        self._target_pub = rospy.Publisher(
            "target", mavros_msgs.msg.PositionTarget, queue_size=10
        )
        self._lengths_pub = rospy.Publisher(
            "lengths", std_msgs.msg.Int64MultiArray, queue_size=10
        )

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / control_rate), self._timer_callback
        )

    def _register_handlers(self):
        """
        Set handlers.

        Mainly here to override later if we decide to subclass this.
        """
        self._state_handlers = {
            DroneState.WAITING_FOR_HOME: self._handle_waiting_for_home,
            DroneState.WAITING_FOR_ARM: self._handle_waiting_for_arm,
            DroneState.TAKEOFF: self._handle_takeoff,
            DroneState.MOVING_TO_START: self._handle_moving_to_start,
            DroneState.EXECUTING_MISSION: self._handle_executing_mission,
            DroneState.MOVING_TO_HOME: self._handle_moving_to_home,
            DroneState.FINISHED: self._handle_finished,
        }

    def _register_transitions(self):
        """
        Set transitions.

        Mainly here to override later if we decide to subclass this.
        """
        self._state_transitions = {
            DroneState(i): DroneState(i + 1) for i in range(len(DroneState) - 1)
        }
        self._state_transitions[DroneState.FINISHED] = DroneState.FINISHED

    def _handle_waiting_for_home(self):
        """State handler for WAITING_FOR_HOME."""
        return self._home_position is not None

    def _handle_waiting_for_arm(self):
        """State handler for WAITING_FOR_ARM."""
        return self._is_armed

    def _handle_takeoff(self):
        """State handler for TAKEOFF."""
        self._send_target(self._hover_position)
        return self._check_pose(self._hover_position)

    def _handle_moving_to_start(self):
        """State handle for MOVING_TO_START."""
        start = self._mission_manager.get_start()
        self._send_target(start)
        if self._check_pose(start) and self._settle_start_time is None:
            self._settle_start_time = rospy.Time.now()

        if self._settle_start_time is not None:
            diff = rospy.Time.now() - self._settle_start_time
            return diff > self._settle_duration

        return False

    def _handle_executing_mission(self):
        """State handler for EXECUTING_MISSION."""
        result = self._mission_manager.run(self._current_position)
        self._send_target(
            result.position,
            yaw=result.yaw,
            velocity=result.velocity,
            acceleration=result.acceleration,
        )

        if result.lengths is not None:
            self._send_lengths(result.lengths)

        return result.finished

    def _handle_moving_to_home(self):
        """State handle for MOVING_TO_START."""
        self._send_target(self._hover_position)
        return self._check_pose(self._hover_position)

    def _handle_finished(self):
        self._send_target(self._hover_position)
        return False

    def _send_target(self, position, yaw=0.0, velocity=None, acceleration=None):
        """Send a waypoint target to mavros."""
        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED

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

    def _timer_callback(self, msg):
        """Handle timer."""
        if self._state not in self._state_handlers:
            rospy.logfatal("State {} is not handled".format(self._state))
            rospy.signal_shutdown("unhandled state")
            return

        rospy.logwarn_throttle(1.0, "Current state: {}".format(self._state))
        if self._state_handlers[self._state]():
            self._state = self._state_transitions[self._state]

    def _state_callback(self, msg):
        """Handle state callback."""
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

        if self._home_position is None:
            self._home_position = self._current_position.copy()
            self._hover_position = self._home_position + np.array([0.0, 0.0, 1.0])

    def _check_pose(self, position):
        """Check if we're close to a requested position."""
        dist = np.linalg.norm(np.array(position) - self._current_position)
        return dist < self._dist_threshold
