"""Custom state machine with slightly simplified logic."""
import geometry_msgs.msg
import mavros_msgs.msg
import mavros_msgs.srv
import std_msgs.msg
import numpy as np
import importlib
import pathlib
import rospy
import enum
import sys


class GraspDroneState(enum.Enum):
    """Current state of the drone."""

    WAITING_FOR_HOME = 0
    WAITING_FOR_ARM = 1
    WAITING_FOR_OFFBOARD = 2
    TAKEOFF = 3
    HOVER = 4
    MOVING_TO_START = 5
    SETTLE_BEFORE = 6
    EXECUTING_MISSION = 7
    SETTLE_AFTER = 8
    RISE = 9
    MOVING_TO_DROP = 10
    DROP = 11
    MOVING_TO_HOME = 12
    HOVER_BEFORE_LAND = 13
    LAND = 14
    IDLE = 15
    HANDLE_SWITCH_TO_MANUAL = 16
    HANDLE_DISARM = 17


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
        self._current_position = None
        self._home_position = None

        self._state_durations = {}
        self._start_times = {}
        self._state_polynomials = {}

        self._register_handlers()
        self._register_state_durations()
        self._register_transitions()

        self._takeoff_offset = rospy.get_param("~takeoff_offset", 1.0)
        self._rise_offset = np.array(rospy.get_param("~rise_offset", [0.0, 0.0, 1.0]))
        self._open_lengths = rospy.get_param("~open_lengths", [190, 190, 208, 208])
        self._land_threshold = rospy.get_param("~dist_threshold", 0.001)
        self._dist_threshold = rospy.get_param("~dist_threshold", 0.2)
        self._average_polynomial_velocity = rospy.get_param(
            "~average_polynomial_velocity", 0.8
        )
        self._desired_yaw = rospy.get_param("~desired_yaw", 0.0)

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

        control_rate = rospy.get_param("~control_rate", 100)
        if control_rate < 20.0:
            rospy.logfatal("Invalid control rate, must be above 20 hz")
            rospy.signal_shutdown("invalid control rate")
            return

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / control_rate), self._timer_callback
        )

    def _register_handlers(self):
        """Register default handlers and replace with a couple new ones."""
        self._state_handlers = {
            GraspDroneState.WAITING_FOR_HOME: self._handle_waiting_for_home,
            GraspDroneState.WAITING_FOR_ARM: self._handle_waiting_for_arm,
            GraspDroneState.WAITING_FOR_OFFBOARD: self._handle_offboard,
            GraspDroneState.TAKEOFF: self._handle_takeoff,
            GraspDroneState.HOVER: self._handle_hover,
            GraspDroneState.MOVING_TO_START: self._handle_moving_to_start,
            GraspDroneState.SETTLE_BEFORE: self._handle_settle_before,
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
            if state == GraspDroneState.SETTLE_BEFORE:
                self._state_durations[state] = rospy.Duration(
                    rospy.get_param("~mission_settle_duration", 2.0)
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

    def _setup_intermediate_waypoints(self):
        """Plan polynomial trajectories between states."""
        if self._home_position is None:
            raise RuntimeError("home position not received! Exiting!")

        hover_position = self._home_position + np.array(
            [0.0, 0.0, self._takeoff_offset]
        )
        start_position = self._mission_manager.get_start()
        rise_position = self._mission_manager.get_end()

        package_dir = pathlib.Path(__file__).resolve().parent.parent.parent
        soft_drone_pydir = package_dir.parent / "python"
        sys.path.append(str(soft_drone_pydir))
        planner = importlib.import_module(
            "soft_drone_python.find_trajectory"
        ).find_polynomials
        polynomial_wrapper = importlib.import_module(
            "soft_drone_python.find_trajectory"
        ).PolynomialInfo

        rospy.loginfo(
            "Planing from home ({}) to grasp start ({})".format(
                np.squeeze(hover_position), np.squeeze(start_position)
            )
        )
        self._state_polynomials[GraspDroneState.MOVING_TO_START] = get_polynomial(
            polynomial_wrapper,
            planner,
            hover_position,
            start_position,
            self._average_polynomial_velocity,
        )
        rospy.loginfo(
            "Planing from grasp end ({}) to drop position ({})".format(
                np.squeeze(rise_position), np.squeeze(start_position)
            )
        )
        self._state_polynomials[GraspDroneState.MOVING_TO_DROP] = get_polynomial(
            polynomial_wrapper,
            planner,
            rise_position,
            start_position,
            self._average_polynomial_velocity,
        )
        rospy.loginfo(
            "Planing from drop position ({}) to home ({})".format(
                np.squeeze(start_position), np.squeeze(hover_position)
            )
        )
        self._state_polynomials[GraspDroneState.MOVING_TO_HOME] = get_polynomial(
            polynomial_wrapper,
            planner,
            start_position,
            hover_position,
            self._average_polynomial_velocity,
        )

    def _get_elapsed(self, state):
        """Check if a state has finished by time."""
        curr_time = rospy.Time.now()
        if state not in self._start_times:
            self._start_times[state] = curr_time
            return rospy.Duration(0.0)

        return curr_time - self._start_times[state]

    def _has_elapsed(self, state):
        """Check if a state has finished by time."""
        if state not in self._state_durations:
            rospy.logerror("State {} should not check time elapsed!".format(state))
            return False

        curr_duration = self._get_elapsed(state)
        return curr_duration > self._state_durations[state]

    def _handle_waiting_for_home(self):
        """Wait for a valid home position from the drone."""
        if self._home_position is not None:
            self._setup_intermediate_waypoints()
        return self._home_position is not None

    def _handle_waiting_for_arm(self):
        """Wait for the drone to be armed."""
        return self._is_armed

    def _handle_offboard(self):
        """Publish to offboard and switch to offboard mode."""
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
        absolute_takeoff_height = self._home_position[2] + self._takeoff_offset

        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.TAKEOFF
        msg.position.z = absolute_takeoff_height

        self._target_pub.publish(msg)

        diff = abs(absolute_takeoff_height - self._current_position[2])
        return diff < self._dist_threshold or self._has_elapsed(GraspDroneState.TAKEOFF)

    def _handle_hover(self):
        """Use loiter command to hang out at hover setpoint."""
        self._loiter_at_point(
            self._home_position[0],
            self._home_position[1],
            self._home_position[2] + self._takeoff_offset,
        )
        return self._has_elapsed(GraspDroneState.HOVER)

    def _handle_moving_to_start(self):
        """Move from the takeoff position to the start of the grasp."""
        elapsed = self._get_elapsed(GraspDroneState.MOVING_TO_START).to_sec()
        curr_poly = self._state_polynomials[GraspDroneState.MOVING_TO_START]
        if elapsed >= curr_poly._total_time:
            return True

        pos, vel, acc = curr_poly.interp(elapsed)
        self._send_target(pos, yaw=self._desired_yaw, velocity=vel, acceleration=acc)
        return False

    def _handle_settle_before(self):
        """Use loiter command to stop after executing the grasp."""
        settle_pos = self._mission_manager.get_start()
        self._loiter_at_point(settle_pos[0], settle_pos[1], settle_pos[2])
        return self._has_elapsed(GraspDroneState.SETTLE_BEFORE)

    def _handle_executing_mission(self):
        """State handler for EXECUTING_MISSION."""
        # elapsed = self._get_elapsed(GraspDroneState.EXECUTING_MISSION).to_sec()
        result = self._mission_manager.run(self._current_position)
        self._send_target(
            result.position,
            yaw=self._desired_yaw,  # TODO(nathan) we should move this upstream
            velocity=result.velocity,
            acceleration=result.acceleration,
            is_grasp=True,
            use_ground_effect=True
        )

        if result.lengths is not None:
            self._send_lengths(result.lengths)

        return result.finished

    def _handle_settle_after(self):
        """Use loiter command to stop after executing the grasp."""
        settle_pos = self._mission_manager.get_end()
        self._loiter_at_point(settle_pos[0], settle_pos[1], settle_pos[2])
        return self._has_elapsed(GraspDroneState.SETTLE_AFTER)

    def _handle_rise(self):
        """Use loiter command to hang out at hover setpoint."""
        rise_pos = self._mission_manager.get_end() + self._rise_offset
        self._loiter_at_point(rise_pos[0], rise_pos[1], rise_pos[2])
        return self._has_elapsed(GraspDroneState.RISE)

    def _handle_moving_to_drop(self):
        """Move from the rise setpoint to the state of the grasp trajectory."""
        elapsed = self._get_elapsed(GraspDroneState.MOVING_TO_DROP).to_sec()
        curr_poly = self._state_polynomials[GraspDroneState.MOVING_TO_DROP]
        if elapsed >= curr_poly._total_time:
            return True

        pos, vel, acc = curr_poly.interp(elapsed)
        self._send_target(pos, yaw=self._desired_yaw, velocity=vel, acceleration=acc)
        return False

    def _handle_drop(self):
        """Wait while we drop the target."""
        drop_pos = self._mission_manager.get_start()
        self._loiter_at_point(drop_pos[0], drop_pos[1], drop_pos[2])
        self._send_lengths(self._open_lengths, scale=False)
        return self._has_elapsed(GraspDroneState.DROP)

    def _handle_moving_to_home(self):
        """Move from dropping the target to hovering above home position."""
        elapsed = self._get_elapsed(GraspDroneState.MOVING_TO_HOME).to_sec()
        curr_poly = self._state_polynomials[GraspDroneState.MOVING_TO_HOME]
        if elapsed >= curr_poly._total_time:
            return True

        pos, vel, acc = curr_poly.interp(elapsed)
        self._send_target(pos, yaw=self._desired_yaw, velocity=vel, acceleration=acc)
        return False

    def _handle_hover_before_land(self):
        """Use loiter command to hang out at hover setpoint."""
        self._loiter_at_point(
            self._home_position[0],
            self._home_position[1],
            self._home_position[2] + self._takeoff_offset,
            yaw=0.0,
        )
        return self._has_elapsed(GraspDroneState.HOVER_BEFORE_LAND)

    def _handle_land(self):
        """Use land command to descend safely."""
        msg = mavros_msgs.msg.PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = mavros_msgs.msg.PositionTarget.FRAME_LOCAL_NED
        msg.type_mask |= SetpointType.LAND
        msg.position.x = self._home_position[0]
        msg.position.y = self._home_position[1]
        msg.position.z = self._home_position[2]  # this gets ignored
        self._target_pub.publish(msg)
        # diff = abs(self._home_position[2] - self._current_position[2])
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
