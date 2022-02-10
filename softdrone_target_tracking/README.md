# SoftDrone-Target-Tracking
Package for target tracking

## Requirements
Tested in Ubuntu 18.04 with [ROS Melodic](http://wiki.ros.org/melodic). Requires [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page), [MRPT](https://docs.mrpt.org/reference/latest/download-mrpt.html), and [GoogleTest](https://github.com/google/googletest). The ROS packages for MRPT are not needed.

## Usage
The tracker takes as input the agent pose in the global frame and the target pose in the agent frame (i.e., target relative to agent). The tracker outputs the target pose in the global frame by fusing the observations of the target relative to the agent using an EKF.

The topics for the inputs/outputs are defined by a .yaml. For the agent pose, the expected message is nav_msgs/Odometry, and for the target pose, the expected message is geometry_msgs/PoseWithCovarianceStamped. In the .yaml file defining the topics for the inputs/outputs, do not use a leading "/" for the topics to be published in the namespace of the tracker. 

## Notes:
To run the tracker with simulated data:
```
roslaunch target_tracking dummy_tracker.launch
``` 

To view the inputs/outputs:
```
roscd target_tracking
rviz -d rviz/dummy_tracker.rviz
```
