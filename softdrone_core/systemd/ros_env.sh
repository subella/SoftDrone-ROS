#!/bin/sh
# also from https://blog.roverrobotics.com/how-to-run-ros-on-startup-bootup/
export ROS_HOSTNAME=$(hostname).localexport ROS_MASTER_URI=http://$ROS_HOSTNAME:11311
