"""Trajectory tracking node."""
from intel_aero_ros.state_machine import MissionRunResult
import rospy


class InterpTrajectoryTracker:
    """Class for tracking a trajectory."""

    def __init__(
        self, polynomial, lengths, gripper_latency=0.0, start_gripper_early=False
    ):
        """Set everything up."""
        self._polynomial = polynomial
        self._lengths = lengths
        self._total_time = self._polynomial._total_time
        self._gripper_latency = gripper_latency
        self._start_gripper_early = start_gripper_early

        self._start_time = None

    def get_start(self):
        """Get the start position for the mission."""
        return self._polynomial.interp(0.0)[0]

    def get_end(self):
        """Get the start position for the mission."""
        return self._polynomial.interp(self._total_time)[0]

    def _run_normal(self, t):
        """Don't bother trying to do the full gripper trajectory."""
        if t > self._total_time:
            return MissionRunResult(
                self._polynomial.interp(self._total_time)[0],
                lengths=self._lengths.interp(self._total_time),
                finished=True,
            )

        pos, vel, acc = self._polynomial.interp(t)
        return MissionRunResult(
            pos,
            yaw=0.0,
            velocity=vel,
            acceleration=acc,
            lengths=self._lengths.interp(t + self._gripper_latency),
            finished=False,
        )

    def _run_full_gripper(self, t):
        """Run the full gripper trajectory."""
        if t > self._total_time + self._gripper_latency:
            return MissionRunResult(
                self._polynomial.interp(self._total_time)[0],
                lengths=self._lengths.interp(self._total_time),
                finished=True,
            )

        if t <= self._gripper_latency:
            pos, vel, acc = self._polynomial.interp(0)
        else:
            pos, vel, acc = self._polynomial.interp(t - self._gripper_latency)

        return MissionRunResult(
            pos,
            yaw=0.0,
            velocity=vel,
            acceleration=acc,
            lengths=self._lengths.interp(t),
            finished=False,
        )

    def run(self, current_position):
        """State handler for EXECUTING_MISSION."""
        if self._start_time is None:
            self._start_time = rospy.Time.now()

        t = (rospy.Time.now() - self._start_time).to_sec()
        if self._start_gripper_early:
            return self._run_full_gripper(t)
        else:
            return self._run_normal(t)
