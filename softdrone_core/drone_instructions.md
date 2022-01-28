### Setting up the highbay

Pretty straightforward, but:

  - Plug in router (mocap ethernet cord goes to one of the switch ports, the white internet cable goes to the gateway port)
  - Turn on the mocap cameras (surge protector on the mocap server rack)
  - Log into the mocap computer (no password for the spark user) and after a minute or so, start motive
  - Once motive starts, make sure you see 43 cameras reporting. If you don't, try loading a few different calibrations (wait for about 15 seconds in between).
    - If that doesn't work, restart motive
    - If that doesn't work, exit motive, power off mocap, and restart the mocap computer and try again
    - If nothing works, call John
  - Grab props, a controller and the drone
  - Set the drone with the realsense pointing towards the wall with the battery safe. Make sure the drone is lined up with one of the stripes on the net (to prevent any yaw errors).
  - Make a rigid body called **spark_sdrone**
  - If you're doing grasping, set up the grasp object and call it **spark_grasp_target**

### Interacting with the drone

I tend to have three terminals up whenever interacting with the drone, as well as QGroundControl (which can be found [here](https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html)).  One terminal is responsible for building the firmware / having a reference as to the firmware source (probably not necessary), one terminal is for handling looking at autopilot logs and running jupyter notebook, and one terminal is for an ssh session on the drone.

On the drone:

- Run tmux to avoid multiple ssh sessions (unless you really want to)
  - Quick tmux guide can be found [here](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/).
  - The drone tmux configuration uses `Ctrl-a` as the leader key for all commands instead of `Ctrl-b`.  You can get vim-like movement between panes with `Alt+[h,j,k,l]`.
- First window is to edit any code on the drone (`roscd softdrone_core`)
- Second window is to flash the firmware (you can skip this if you want)
- Third window is split into two panes: `roscore` and `roslaunch softdrone_core mocap_translation.launch gcs_ip:=YOUR_IP`
- Fourth window tends to also be split into two panes: `roslaunch softdrone_core SOME_LAUNCH_FILE` and a debug shell into the firmware. At this point, I mainly use it to set parameters, which you can do from QGroundControl.
- Fifth window is for copying logs: `cd /var/lib/mavlink-router`.  I tend to rsync logs to a directory, but whatever works.

Other things to remember:

- Reboot the drone in between trials to get unique logs
- Make sure the right rigid body names show up for the mocap translation

### Actually flying

Any code that you run will hang out and wait for you to arm the drone (throttle stick low and to the right for a few seconds).  Before arming, make sure that the kill switch is disabled and the mode switch has the drone in position mode (and the drone accepted the position mode request).  Once armed, the drone will switch to listening to whatever ROS tells it and nothing else.

Once you want to stop the drone (either in an emergency or during normal operation):
- Flip the kill switch to stop the motors
- Switch the mode switch to manual and disarm (throttle down and to the left)

### Important code:

Grasping: `roslaunch softdrone_core trajectory_follower.launch use_gripper:=true`

Involves (files relative to `softdrone_core`):
 - `launch/trajectory_follower.launch`
 - `bin/trajectory_tracking_node`
 - `src/softdrone_core/gain_tuning_state_machine.py`

Landing: `roslaunch softdrone_core landing_experiments.launch use_gripper:=true naive:=false`

Involves (files relative to `softdrone_core`):
 - `launch/landing_experiments.launch`
 - `bin/landing_node`
 - `src/softdrone_core/landing_state_machine.py`
