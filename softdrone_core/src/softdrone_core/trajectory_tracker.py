"""Trajectory tracking node."""
from softdrone_core.state_machine import MissionRunResult
import rospy
import math


class TrajectoryTracker:
    """Class for tracking a trajectory."""

    def __init__(
        self, positions, velocities, lengths, frame_duration, accelerations=None
    ):
        """Set everything up."""
        assert len(positions) > 0
        assert len(positions) == len(velocities)
        assert len(positions) == len(lengths)
        assert accelerations is None or len(positions) == len(accelerations)

        self._start_time = None
        self._frame_duration = frame_duration

        self._positions = positions
        self._velocities = velocities
        self._accels = accelerations
        self._lengths = lengths

    def get_start(self):
        """Get the start position for the mission."""
        return self._positions[0]

    def run(self, current_position):
        """State handler for EXECUTING_MISSION."""
        if self._start_time is None:
            self._start_time = rospy.Time.now()

        # TODO(nathan) just interpolate the polynomials / finger lengths on demand
        diff = (rospy.Time.now() - self._start_time).to_sec()
        index = int(math.ceil(diff / self._frame_duration))

        if index >= len(self._positions):
            return MissionRunResult(
                self._positions[-1], lengths=self._lengths[-1], finished=True
            )

        return MissionRunResult(
            self._positions[index],
            velocity=self._velocities[index],
            acceleration=self._accels[index] if self._accels is not None else None,
            lengths=self._lengths[index],
            finished=False,
        )
