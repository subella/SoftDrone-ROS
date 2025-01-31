"""Trajectory tracking node."""
from softdrone_core.state_machine import MissionRunResult
import rospy


class InterpTrajectoryTracker:
    """Class for tracking a trajectory."""

    def __init__(
        self,
        polynomial,
        lengths,
        gripper_latency=0.0,
        alpha=1,
        settle_time=None,
    ):
        """Set everything up."""
        self._polynomial = polynomial
        self._lengths = lengths
        self._total_time = self._polynomial._total_time
        self._gripper_latency = gripper_latency
        self.alpha = alpha
        if settle_time is not None:
            self._final_time = self._total_time + settle_time
        else:
            self._final_time = self._total_time

        self._start_time = None

    def get_start(self):
        """Get the start position for the mission."""
        return self._polynomial.interp(0.0)[0]

    def get_end(self):
        """Get the start position for the mission."""
        return self._polynomial.interp(self._total_time)[0]

    def _run_normal(self, t):
        """Interpolate the trajectory."""
        scaled_t = self.alpha * t
        if scaled_t > self._total_time:
            pos, vel, acc = self._polynomial.interp(self._total_time)
            lengths = self._lengths.interp(self._total_time)
        else:
            pos, vel, acc = self._polynomial.interp(scaled_t)
            lengths = self._lengths.interp(scaled_t + self.alpha * self._gripper_latency)
        vel *= self.alpha
        acc *= self.alpha**2

        return MissionRunResult(
            pos,
            yaw=0.0,
            velocity=vel,
            acceleration=acc,
            finished=(scaled_t > self._final_time),
            lengths=lengths,
        )

    def run(self, current_position):
        """State handler for EXECUTING_MISSION."""
        if self._start_time is None:
            self._start_time = rospy.Time.now()

        t = (rospy.Time.now() - self._start_time).to_sec()
        return self._run_normal(t)
