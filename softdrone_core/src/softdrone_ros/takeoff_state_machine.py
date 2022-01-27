"""Custom state machine with slightly simplified logic."""
from softdrone_ros.state_machine import StateMachine
import mavros_msgs.msg
import mavros_msgs.srv
import numpy as np
import rospy
import enum


# TODO(nathan) this is ugly
class TakeoffDroneState(enum.Enum):
    """Current state of the drone."""

    WAITING_FOR_HOME = 0
    WAITING_FOR_ARM = 1
    WAITING_FOR_OFFBOARD = 2
    TAKEOFF = 3
    HOVER = 4
    MOVING_TO_DROP = 5
    DROP = 6
    MOVING_TO_HOME = 7
    HOVER_AGAIN = 8
    LAND = 9


class SetpointType(enum.IntEnum):
    """Offboard setpoint binary mask."""

    TAKEOFF = 0x1000
    LAND = 0x2000
    LOITER = 0x4000
    IDLE = 0x8000


class TakeoffStateMachine(StateMachine):
    """Class for tracking a trajectory."""

    def __init__(self):
        """Set everything up."""
        self._offboard_start_time = None
        self._offboard_wait_duration = rospy.Duration(
            rospy.get_param("~offboard_wait_duration", 0.2)
        )
        self._wait_for_arm = rospy.get_param("~wait_for_arm", True)
        self._hover_duration = rospy.Duration(rospy.get_param("~hover_duration", 6.0))
        self._hover_start_time = None
        self._hover2_start_time = None
        self._grasp_settle_start_time = None
        self._rise_start_time = None
        self._drop_offset = np.array(rospy.get_param("~drop_offset", [1.5, 1.0, 0.5]))
        self._drop_position = None

        self._start_times = {}
        self._settle_velocity = rospy.get_param("~end_velocity", [0, 0.5, -0.05])

        self._land_separate = rospy.get_param("~land_separate", False)
        if self._land_separate:
            self._land_position = rospy.get_param("~land_position", [-2.5, -3.25, 2.0])
        else:
            self._land_position = None

        self._drop_start = None
        self._drop_duration = rospy.Duration(rospy.get_param("~drop_duration", 8.0))
        self._open_lengths = rospy.get_param("~open_lengths", [121, 121, 138, 138])

        super(TakeoffStateMachine, self).__init__(None, initial_state=TakeoffDroneState.WAITING_FOR_HOME
        )

    def _register_handlers(self):
        """Register default handlers and replace with a couple new ones."""
        self._state_handlers = {
            TakeoffDroneState.WAITING_FOR_HOME: self._handle_waiting_for_home,
            TakeoffDroneState.WAITING_FOR_ARM: self._handle_waiting_for_arm,
            TakeoffDroneState.WAITING_FOR_OFFBOARD: self._handle_offboard,
            TakeoffDroneState.TAKEOFF: self._handle_takeoff,
            TakeoffDroneState.HOVER: self._handle_hover,
            TakeoffDroneState.MOVING_TO_DROP: self._handle_moving_to_drop,
            TakeoffDroneState.DROP: self._handle_drop,
            TakeoffDroneState.MOVING_TO_HOME: self._handle_moving_to_home,
            TakeoffDroneState.HOVER_AGAIN: self._handle_hover_again,
            TakeoffDroneState.LAND: self._handle_land,
        }

    def _register_transitions(self):
        """Set transitions up."""
        self._state_transitions = {
            TakeoffDroneState(i): TakeoffDroneState(i + 1)
            for i in range(len(TakeoffDroneState) - 1)
        }
        self._state_transitions[TakeoffDroneState.LAND] = TakeoffDroneState.LAND

    def _handle_waiting_for_arm(self):
        """State handler for WAITING_FOR_ARM."""
        if not self._wait_for_arm:
            return True

        return self._is_armed

    def _handle_offboard(self):
        """Publish to offboard and switch to offboard mode."""
        if self._offboard_start_time is None:
            self._offboard_start_time = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.IDLE

        self._target_pub.publish(msg)

        # ideal we'd be checking something, but we can't...
        if rospy.Time.now() - self._offboard_start_time < self._offboard_wait_duration:
            return False

        set_mode = rospy.ServiceProxy("mavros/set_mode", mavros_msgs.srv.SetMode)
        result = set_mode(0, "offboard")
        return result.mode_sent

    def _handle_takeoff(self):
        """Use takeoff setpoint to command setpoint."""
        if TakeoffDroneState.TAKEOFF not in self._start_times:
            self._start_times[TakeoffDroneState.TAKEOFF] = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.TAKEOFF
        msg.position.z = self._hover_position[2]

        self._target_pub.publish(msg)

        diff = rospy.Time.now() - self._start_times[TakeoffDroneState.TAKEOFF]
        if diff > self._hover_duration:
            return True

        diff = abs(self._hover_position[2] - self._current_position[2])
        return diff < self._dist_threshold

    def _handle_hover(self):
        """Use loiter command to hang out at hover setpoint."""
        if self._hover_start_time is None:
            self._hover_start_time = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LOITER
        msg.position.x = self._hover_position[0]
        msg.position.y = self._hover_position[1]
        msg.position.z = self._hover_position[2]

        self._target_pub.publish(msg)

        return rospy.Time.now() - self._hover_start_time > self._hover_duration

    def _handle_hover_again(self):
        """Use loiter command to hang out at hover setpoint."""
        if self._hover2_start_time is None:
            self._hover2_start_time = rospy.Time.now()

        if self._land_position is None:
            self._land_position = self._hover_position.copy()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LOITER
        msg.position.x = self._land_position[0]
        msg.position.y = self._land_position[1]
        msg.position.z = self._land_position[2]

        self._target_pub.publish(msg)

        return rospy.Time.now() - self._hover2_start_time > self._hover_duration

    def _handle_moving_to_drop(self):
        """State handle for MOVING_TO_START."""
        if self._drop_position is None:
            self._drop_position = self._hover_position + self._drop_offset
        self._send_target(self._drop_position)
        return self._check_pose(self._drop_position)

    def _handle_drop(self):
        """State handle for MOVING_TO_START."""
        if self._drop_start is None:
            self._drop_start = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LOITER
        msg.position.x = self._drop_position[0]
        msg.position.y = self._drop_position[1]
        msg.position.z = self._drop_position[2]

        self._target_pub.publish(msg)
        self._send_lengths(self._open_lengths, scale=False)

        return rospy.Time.now() - self._drop_start > self._drop_duration

    def _handle_moving_to_home(self):
        """State handle for MOVING_TO_START."""
        if self._land_position is None:
            self._land_position = self._hover_position.copy()

        if TakeoffDroneState.MOVING_TO_HOME not in self._start_times:
            self._start_times[TakeoffDroneState.MOVING_TO_HOME] = rospy.Time.now()

        diff = rospy.Time.now() - self._start_times[TakeoffDroneState.MOVING_TO_HOME]
        if diff > self._hover_duration:
            return True

        self._send_target(self._land_position)
        return self._check_pose(self._land_position)

    def _handle_land(self):
        """Use land command to descend safely."""
        if self._land_position is None:
            self._land_position = self._hover_position.copy()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LAND
        msg.position.x = self._land_position[0]
        msg.position.y = self._land_position[1]
        msg.position.z = self._land_position[2]

        self._target_pub.publish(msg)
        return False
