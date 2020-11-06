"""Custom state machine with slightly simplified logic."""
from intel_aero_ros.state_machine import StateMachine
import mavros_msgs.msg
import mavros_msgs.srv
import numpy as np
import rospy
import enum


# TODO(nathan) this is ugly
class LandingDroneState(enum.Enum):
    """Current state of the drone."""

    WAITING_FOR_HOME = 0
    WAITING_FOR_ARM = 1
    OFFBOARD_WAIT = 2
    TAKEOFF = 3
    HOVER = 4
    MOVE_TO_START = 5
    LAND = 6
    DROP = 7


class SetpointType(enum.IntEnum):
    """Offboard setpoint binary mask."""

    TAKEOFF = 0x1000
    LAND = 0x2000
    LOITER = 0x4000
    IDLE = 0x8000


class LandingStateMachine(StateMachine):
    """Class for tracking a trajectory."""

    def __init__(self):
        """Set everything up."""
        self._start_times = {}

        self._offboard_wait = rospy.Duration(rospy.get_param("~offboard_wait", 0.2))
        self._hover_duration = rospy.Duration(rospy.get_param("~hover_duration", 6.0))
        self._land_duration = rospy.Duration(rospy.get_param("~land_duration", 1.0))
        self._start_settle_duration = rospy.Duration(
            rospy.get_param("~start_settle_duration", 10.0)
        )

        self._start_position = np.array(
            rospy.get_param("~start_position", [0.0, 0.0, 2.0])
        )
        self._land_velocity = np.array(
            rospy.get_param("~land_velocity", [0.0, 0.0, 0.0])
        )

        self._lengths = rospy.get_param("~lengths", [121, 121, 138, 138])

        self._override_pub = rospy.Publisher(
            "rc_override", mavros_msgs.msg.OverrideRCIn, queue_size=10
        )

        super(LandingStateMachine, self).__init__(
            None, initial_state=LandingDroneState.WAITING_FOR_HOME
        )

    def _register_handlers(self):
        """Register default handlers and replace with a couple new ones."""
        self._state_handlers = {
            LandingDroneState.WAITING_FOR_HOME: self._handle_waiting_for_home,
            LandingDroneState.WAITING_FOR_ARM: self._handle_waiting_for_arm,
            LandingDroneState.OFFBOARD_WAIT: self._handle_offboard,
            LandingDroneState.TAKEOFF: self._handle_takeoff,
            LandingDroneState.HOVER: self._handle_hover,
            LandingDroneState.MOVE_TO_START: self._handle_moving_to_start,
            LandingDroneState.LAND: self._handle_land,
            LandingDroneState.DROP: self._handle_drop,
        }

    def _register_transitions(self):
        """Set transitions up."""
        self._state_transitions = {
            LandingDroneState(i): LandingDroneState(i + 1)
            for i in range(len(LandingDroneState) - 1)
        }
        self._state_transitions[LandingDroneState.DROP] = LandingDroneState.DROP

    def _handle_waiting_for_arm(self):
        """State handler for WAITING_FOR_ARM."""
        return self._is_armed

    def _handle_offboard(self):
        """Publish to offboard and switch to offboard mode."""
        if LandingDroneState.OFFBOARD_WAIT not in self._start_times:
            self._start_times[LandingDroneState.OFFBOARD_WAIT] = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.IDLE

        self._target_pub.publish(msg)

        # ideally we'd be checking something, but we can't...
        elapsed = rospy.Time.now() - self._start_times[LandingDroneState.OFFBOARD_WAIT]
        if elapsed < self._offboard_wait:
            return False

        # set_mode = rospy.ServiceProxy("mavros/set_mode", mavros_msgs.srv.SetMode)
        # result = set_mode(0, "offboard")
        # return result.mode_sent
        return True

    def _handle_takeoff(self):
        """Use takeoff setpoint to command setpoint."""
        if LandingDroneState.TAKEOFF not in self._start_times:
            self._start_times[LandingDroneState.TAKEOFF] = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.TAKEOFF
        msg.position.z = self._hover_position[2]

        self._target_pub.publish(msg)

        diff = rospy.Time.now() - self._start_times[LandingDroneState.TAKEOFF]
        if diff > self._hover_duration:
            return True

        diff = abs(self._hover_position[2] - self._current_position[2])
        return diff < self._dist_threshold

    def _handle_hover(self):
        """Use loiter command to hang out at hover setpoint."""
        if LandingDroneState.HOVER not in self._start_times:
            self._start_times[LandingDroneState.HOVER] = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LOITER
        msg.position.x = self._hover_position[0]
        msg.position.y = self._hover_position[1]
        msg.position.z = self._hover_position[2]

        self._target_pub.publish(msg)

        elapsed = rospy.Time.now() - self._start_times[LandingDroneState.HOVER]
        return elapsed > self._hover_duration

    def _handle_moving_to_start(self):
        """State handle for MOVE_TO_START."""
        if LandingDroneState.MOVE_TO_START not in self._start_times:
            self._start_times[LandingDroneState.MOVE_TO_START] = rospy.Time.now()

        self._send_target(self._start_position)
        self._send_lengths(self._lengths, scale=False)

        elapsed = rospy.Time.now() - self._start_times[LandingDroneState.MOVE_TO_START]
        return elapsed > self._start_settle_duration

    def _handle_land(self):
        """Agressive landing."""
        if LandingDroneState.LAND not in self._start_times:
            self._start_times[LandingDroneState.LAND] = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_PX
        msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_PY
        msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_PZ
        msg.velocity.x = self._land_velocity[0]
        msg.velocity.y = self._land_velocity[1]
        msg.velocity.z = self._land_velocity[2]

        self._target_pub.publish(msg)

        elapsed = rospy.Time.now() - self._start_times[LandingDroneState.LAND]
        return elapsed > self._land_duration

    def _handle_drop(self):
        """Agressive landing."""
        # TODO(nathan) check this very carefully
        msg = mavros_msgs.msg.OverrideRCIn()
        msg.channels = [1200] * 7 + [2000]
        self._override_pub.publish(msg)
        return False
