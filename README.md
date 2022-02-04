Installation
============

First you need to install the softdrone trajectory optimization python package:
```
cd $SoftDrone_ROS_DIR/SoftDrone-TrajOpt
pip2 install -e .
```

Next, build the softdrone core package:
```
catkin build softdrone_core
```

Simulation
==========

To launch the ROS simulation, in one terminal run:
```
roslaunch softdrone_core master.launch simulation_ros:=true
```
This should result in a new RVIZ window.

In another terminal, run
```
roslaunch softdrone_core setpoint_cli_node.launch send_to_mavros:=false
```
We use this window to (among other things) mimic arming the drone from the RC controller.
Type ARM into the prompt and press enter. After 5 seconds the drone should take off and
start following the red trajectory.
