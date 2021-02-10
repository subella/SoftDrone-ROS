"""Custom state machine with slightly simplified logic."""
from intel_aero_ros.state_machine import StateMachine
import mavros_msgs.msg
import mavros_msgs.srv
import numpy as np
import rospy
import enum


# TODO(nathan) this is ugly
class TuningDroneState(enum.Enum):
    """Current state of the drone."""

    WAITING_FOR_HOME = 0
    WAITING_FOR_ARM = 1
    WAITING_FOR_OFFBOARD = 2
    TAKEOFF = 3
    HOVER = 4
    MOVING_TO_START = 5
    EXECUTING_MISSION = 6
    SETTLE = 7
    RISE = 8
    MOVING_TO_DROP = 9
    DROP = 10
    MOVING_TO_HOME = 11
    HOVER_AGAIN = 12
    LAND = 13


class SetpointType(enum.IntEnum):
    """Offboard setpoint binary mask."""

    TAKEOFF = 0x1000
    LAND = 0x2000
    LOITER = 0x4000
    IDLE = 0x8000


class GainTuningStateMachine(StateMachine):
    """Class for tracking a trajectory."""

    def __init__(self, mission_manager):
        """Set everything up."""
        self._offboard_start_time = None
        self._offboard_wait_duration = rospy.Duration(
            rospy.get_param("~offboard_wait_duration", 0.2)
        )
        self._wait_for_arm = rospy.get_param("~wait_for_arm", True)
        self._hover_duration = rospy.Duration(rospy.get_param("~hover_duration", 6.0))
        self._mission_settle_time = rospy.Duration(rospy.get_param("~mission_settle_time", 1.0))
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
        self._open_lengths = rospy.get_param("~open_lengths", [190, 190, 208, 208])

        super(GainTuningStateMachine, self).__init__(
            mission_manager, initial_state=TuningDroneState.WAITING_FOR_HOME
        )

    def _register_handlers(self):
        """Register default handlers and replace with a couple new ones."""
        self._state_handlers = {
            TuningDroneState.WAITING_FOR_HOME: self._handle_waiting_for_home,
            TuningDroneState.WAITING_FOR_ARM: self._handle_waiting_for_arm,
            TuningDroneState.WAITING_FOR_OFFBOARD: self._handle_offboard,
            TuningDroneState.TAKEOFF: self._handle_takeoff,
            TuningDroneState.HOVER: self._handle_hover,
            TuningDroneState.MOVING_TO_START: self._handle_moving_to_start,
            TuningDroneState.EXECUTING_MISSION: self._handle_executing_mission,
            TuningDroneState.SETTLE: self._handle_settle,
            TuningDroneState.RISE: self._handle_rise,
            TuningDroneState.MOVING_TO_DROP: self._handle_moving_to_drop,
            TuningDroneState.DROP: self._handle_drop,
            TuningDroneState.MOVING_TO_HOME: self._handle_moving_to_home,
            TuningDroneState.HOVER_AGAIN: self._handle_hover_again,
            TuningDroneState.LAND: self._handle_land,
        }

    def _register_transitions(self):
        """Set transitions up."""
        self._state_transitions = {
            TuningDroneState(i): TuningDroneState(i + 1)
            for i in range(len(TuningDroneState) - 1)
        }
        self._state_transitions[TuningDroneState.LAND] = TuningDroneState.LAND

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
        if TuningDroneState.TAKEOFF not in self._start_times:
            self._start_times[TuningDroneState.TAKEOFF] = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.TAKEOFF
        msg.position.z = self._hover_position[2]

        self._target_pub.publish(msg)

        diff = rospy.Time.now() - self._start_times[TuningDroneState.TAKEOFF]
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

    def _handle_moving_to_start(self):
        """State handle for MOVING_TO_START."""
        start = self._mission_manager.get_start()
        self._send_target(start)

        if TuningDroneState.MOVING_TO_START not in self._start_times:
            self._start_times[TuningDroneState.MOVING_TO_START] = rospy.Time.now()

        diff = rospy.Time.now() - self._start_times[TuningDroneState.MOVING_TO_START]
        if diff > self._hover_duration and self._settle_start_time is None:
            self._settle_start_time = rospy.Time.now()

        if self._check_pose(start) and self._settle_start_time is None:
            self._settle_start_time = rospy.Time.now()

        if self._settle_start_time is not None:
            diff = rospy.Time.now() - self._settle_start_time
            return diff > self._settle_duration

        return False

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

    def _handle_settle(self):
        """Use loiter command to hang out at hover setpoint."""
        settle_pos = self._mission_manager.get_end()
        settle_pos[1] += 0.1
        settle_pos[2] += 0.0

        if self._grasp_settle_start_time is None:
            self._grasp_settle_start_time = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_PX
        msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_PY
        msg.type_mask |= mavros_msgs.msg.PositionTarget.IGNORE_PZ
        msg.position.x = settle_pos[0]
        msg.position.y = settle_pos[1]
        msg.position.z = settle_pos[2]
        msg.velocity.x = self._settle_velocity[0]
        msg.velocity.y = self._settle_velocity[1]
        msg.velocity.z = self._settle_velocity[2]

        self._target_pub.publish(msg)

        if self._grasp_settle_start_time is None:
            return False

        return rospy.Time.now() - self._grasp_settle_start_time > self._mission_settle_time

    def _handle_rise(self):
        """Use loiter command to hang out at hover setpoint."""
        settle_pos = self._mission_manager.get_end()
        settle_pos[1] += 0.0
        settle_pos[2] += 1.0

        if self._rise_start_time is None:
            self._rise_start_time = rospy.Time.now()

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LOITER
        msg.position.x = settle_pos[0]
        msg.position.y = settle_pos[1]
        msg.position.z = settle_pos[2]

        self._target_pub.publish(msg)

        return rospy.Time.now() - self._rise_start_time > (2 * self._hover_duration)

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

        if TuningDroneState.MOVING_TO_HOME not in self._start_times:
            self._start_times[TuningDroneState.MOVING_TO_HOME] = rospy.Time.now()

        diff = rospy.Time.now() - self._start_times[TuningDroneState.MOVING_TO_HOME]
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
