"""Trajectory tracking node."""
from intel_aero_ros.state_machine import MissionRunResult
import numpy as np
import rospy


class WaypointTracker:
    """Class for tracking some waypoints."""

    def __init__(self, positions, distance_threshold, lengths=None):
        """Set everything up."""
        assert len(positions) > 0

        self._distance_threshold = distance_threshold
        self._positions = positions
        self._lengths = lengths
        self._index = 0

    def get_start(self):
        """Get the start position for the mission."""
        return self._positions[0]

    def run(self, curr_position):
        """State handler for EXECUTING_MISSION."""
        if self._index < len(self._positions):
            dist = np.linalg.norm(np.array(self._positions[self._index]) - curr_position)
            if dist < self._distance_threshold:
                self._index += 1

        if self._index >= len(self._positions):
            return MissionRunResult(
                self._positions[-1],
                lengths=self._lengths[-1] if self._lengths is not None else None,
                finished=True,
            )

        rospy.logwarn_throttle(0.5, "Moving to waypoint {}".format(self._index))

        return MissionRunResult(
            self._positions[self._index],
            lengths=self._lengths[self._index] if self._lengths is not None else None,
            finished=False,
        )
