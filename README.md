# Overview
This is the core repository for the Soft Drone grasping pipeline.

* `SoftDrone-TrajOpt` contains the implementation of the polynomial trajectory planner.
* `softdrone_core` is the core package and contains the flight state machine and entry points.
* `gtsam_tracker` contains the implementation of the fixed lag smoother
* `softdrone_target_pose_estimator` contains the keypoint detector and TEASER++
* `softdrone_target_tracking` is not used.


# Tutorial


[Fabricating the Gripper Base](#fabricating-the-gripper-base)

[Fabricating the Fingers](#fabricating-the-fingers)

[Flashing the ESCs](#flashing-the-escs)

[Flying the Drone](#flying-the-drone)


# Fabricating the Gripper Base
The easiest way is to just print out the sliced file already in the 3D printer. On the printer, click Build > USB > Gripper Redesign. On the very bottom right is a file called ReinforcedGripperBase. To print it, first take out the tray, clean it with water and paper towel, dry it, then cover it with a light layer of gluestick. Then hit build. This file fails a lot tbh becuase its printing in higher quality and that jams it sometimes. You might want to slice a low quality version of it from the CAD repo https://github.mit.edu/SPARK/softDrone_cad/tree/master/PrestressedGripper/AdjWidthBase/ReinforcedBase

Ok, if its printed, then scrape it out and remove all the support. Unfortunately theres still a bit of work. First you need to tap the holes for the pcb. In the gray storage next to the printers one of the containers has a tap and drill bit size M2.5x0.45. I think this is the right one but don't quote me on it, try it on just one hole first. First drill out the holes to be a bit bigger using the bit. Then tap it. Migrate the pcb over and screw it in.

The slots for the fingers need to be widened too, usually. Just mess around with it with drills bits and stuff until the finger can slide smoothly in and out. Do not try to force a finger into it because it will get stuck and it will be frustrating.

Assembling the rest should be straight forward, just copy over the parts from the broken gripper base to the new one. Don't lose or break the winches though because those are elaborate to reconstruct.

Also, the hole spacing is acutally just wrong so when you have trouble connecting it to the drone thats why. Just losen the standoffs a bit first so they can fit in the holes and then tighten everything.
 
# Fabricating the Fingers
First you need to find the mold. It got moved around so idk ask Aaron. Its dimensions should make fingers that are 31.6x31.6mm (meausre it with calipers) and it should look well used. Print a new one if you have to: https://github.mit.edu/SPARK/softDrone_cad/tree/master/PrestressedGripper/ConstantRadiusFinger/Prestress65/SquareAR. Same goes for the eyelets and finger base connector. The printer should have this stuff loaded in, its 4x_eyelets_316 and 4x_eyeletsEnd_316 and 4x_fingerBaseConn I think.

Put cardboard down and put the mold onto of it. Then cover the inside and top where the holes are with the Ease Release 200. Use a brush. Now put an empty cup on the scale and tare it. You want to use Flex it Foam X!. Read the manual it will tell you the percentage of parts A to B. Put about 50g of part A in and then put whatever it wants for part B using the ratio. Mix well with a popsicle stick and pour into the bottom half of the mold. Then put the top half on and wrap duck tape around the whole thing in 3 different spots but dont cover up the holes. It will cure for about 45 mins. Then take it out carefully. If it doesn't come out easily you either didn't put enough ease release on it on the mold had too much residue from previous castings. Clean up the edges and stuff with the pliers that are good at cutting. 

The eyelet holes need to be widened a bit. Use a small allen wrench and poke it through all the holes. Study how the end eyelet is supposed to work by looking at the other fingers. You will need to remove the support there so that the cable can wrap around like that. I usually used an xacto knife for that but cut myself once and bled a lot so be careful.

After that just super glue everything in place. Take the cable from an existing finger or cut a new one.

The existing cables should have some black markings on where it should be lined up initially in order to match our fem. I would just play around with initial positions until it looks what you would expect (refer to the paper). You can use the test script to test it: https://github.mit.edu/SPARK/SoftDrone-Hardware/blob/master/arduino/tests_PCB_rev2/test_grasp_rev2/test_grasp/test_grasp.ino

The script that acutally should be ran in flight is this one: https://github.mit.edu/SPARK/SoftDrone-Hardware/blob/master/arduino/gripper_software_v1/gripper_software_v1.ino

There is a 3 wire jumper cable that needs to be connected from the xavier to the pcb. I dont remember which orientation it should be in so just try them both. Also make sure to connect the pink 5v logic supply from the xavier.

# Flashing the ESCs
The ESCs we were using shipped with wrong firmware so we need to change it. First, when you are making a new ESC make sure that all the cables match what the ESC before had. The wire colors don't necessarily match the motor phases (the current ESC does match). 

When its done, in the top drawer of my desk theres an Arduino Uno with a brown and red wire. The brown wire should go to GND on the arduino and the red to pin 11. The brown wire should be on the very left pin of the connector when viewing it where you can see the metal crimps exposed. Move the red wire into index 1 (opposite side of the brown wire). Plug in the white connector to the ESC.

You will need to be on Windows and run BLHeliSuite32. Click read setup and follow the instructions. If it looks good go to ESC flash tab and flash the ESC to version 3.9. Disconnect and power off ESC. Then move the red wire to the next pin index (its a 4 in 1 ESC and each pin writes to a different ESC). Repeat the process until all 4 ESCs are flashed.
# Flying the Drone

## Mocap
1. Reserve the flight space by emailing lukedc@mit.edu
2. Take the soft drone bin of stuff into the mocap room
3. Turn on the mocap cameras by flipping the switch
4. Plug in the router to power. Connect the yellow ethernet cable from optitrack into the router and connect another ethernet cable from our router to your computer.
5. Start charging the batteries. You just have to plug the charger into power and then plug the batteries in and they will already start charging. Be careful because the bottom half of the charger is falling off.
6. Login into the mocap computer. Username `Sparklab`, password is `spark` (or `sparklab` idr).
7. Ensure the net is all tight
8. Position the drone and target like in the videos. It is important you align the drone and target as best you can with the mocap axis. Launch motive and create rigid bodies. The drone should be named `spark_sdrone` and the target named `spark_grasp_tar`.
9. Plug battery into drone. It will automically connect to router.
10. Bind the transmitter to the drone by holding the `bind` (top left) button down and then flipping the power switch on the transmitter. Continue holding the button until you get the positive beeps. If you get the angry beeps, you can keep trying and get closer to the receiver. If its always the angry beeps, then unplug and replug the drone battery and try again.
11. We need to do an annoying step now to sync times because the router doesn't have internet. While connected to internet, type `sudo ntpdate -s time-a-g.nist.gov` on your personal computer.
12. Wait for the drone to boot. Then ssh with `ssh sparklab@spark-xavier-5`. password is sparklab
13. Connect an ethernet cable that is connected to internet to the drone. Then type `sudo ntpdate -s time-a-g.nist.gov` on the drone.
14. Unplug the ethernet cable. We are good to go now.
15. Ssh into the drone again. Type `zsh`. Then, crtl + R and start typing mavproxy. You will need to add your ip address to this mavproxy command. Now launch the command.
16. On your computer, open up QGroundControl. It should now automatically connect.
17. Make sure the drone is on a reasonable commit, like one of the ones that say "This commit grasped x for x in mocap".
18. Now, reboot the drone in QGroundControl. Do not touch the drone after rebooting, you may mess up its internal EKF.
19. Launch `roslaunch softdrone_core full_drone.launch` on the drone.
20. On your LOCAL computer, install this repo and optitrack repo. Source the catkin_ws, and run `export ROS_MASTER_URI=http://192.168.1.129:11311/`. Also, run `export ROS_IP=192.168.1.XXX`. 
21. In the mocap translator, ensure that these lines are changed to be like this. This is another unfortunate hack for time syncing:
  ```
      def _timer_callback(self, event):
          """Send latest pose periodically if it exists."""
          if self._latest_pose is not None:
              self._latest_pose.header.stamp = rospy.Time.now()
              self._pose_pub.publish(self._latest_pose)

          if self._latest_pose_wc is not None:
              self._latest_pose_wc.header.stamp = rospy.Time.now()
              self._pose_wc_pub.publish(self._latest_pose_wc)

          if self._latest_target_pose is not None:
              self._latest_target_pose.header.stamp = rospy.Time.now()
              self._target_pose_pub.publish(self._latest_target_pose)
   ```
23. Now launch `roslaunch optitrack optitrack.launch` and `roslaunch softdrone_core mocap_translator.launch`. You should see that optitrack detects the drone and target rigid bodies.
24. Some good things should now be happening in the terminal you launched `full_drone.launch` in. It should be saying "WAITING FOR HOME" and eventually it will find home and spit out an array like this [-1.33, -2.2, 1.0]. Copy this value and shut down all your ros nodes. Go into `softdrone_core/config/grasp_state_machine_node.yaml` and change the `start_position` to the value you copied. Add about 1m to the z (3rd coordinate.)
25. Reboot the drone again in QGroundControl.
26. Launch `full_drone.launch`, then launch optitrack and mocap_translator again on your local computer.
27. Evenutally, the output will switch to "WAITING FOR ARM". Make sure the transmitter buttons are in these states: Flight mode: 0, AUX: 1, FLAP: 0, RATE: HI, MOTOR: ARM. The MOTOR is the kill switch, be ready to flip it if something goes bad.
28. Its time. Press the throttle (left joystick) all the way down and to the right. The propellers will start spinning. Then the drone will take off and do its thing. 

## Vision
1. Reserve the flight space by emailing lukedc@mit.edu
2. Take the soft drone bin of stuff into the mocap room
4. Plug in the router to power. Connect connect an ethernet cable from our router to your computer.
5. Start charging the batteries. You just have to plug the charger into power and then plug the batteries in and they will already start charging. Be careful because the bottom half of the charger is falling off.
7. Ensure the net is all tight
8. Position the drone and target like in the videos. 
9. Plug battery into drone. It will automically connect to router.
10. Bind the transmitter to the drone by holding the `bind` (top left) button down and then flipping the power switch on the transmitter. Continue holding the button until you get the positive beeps. If you get the angry beeps, you can keep trying and get closer to the receiver. If its always the angry beeps, then unplug and replug the drone battery and try again.
12. Wait for the drone to boot. Then ssh with `ssh sparklab@spark-xavier-5`. password is sparklab
15. Ssh into the drone again. Type `zsh`. Then, crtl + R and start typing mavproxy. You will need to add your ip address to this mavproxy command. Now launch the command.
16. On your computer, open up QGroundControl. It should now automatically connect.
17. Make sure the drone is on a reasonable commit, like one of the ones that say "This commit grasped x for x in vision".
18. Now, reboot the drone in QGroundControl. Do not touch the drone after rebooting, you may mess up its internal EKF.
19. Launch the keypoint detector. Open a new ssh terminal, then type `roscd softdrone_target_pose_estimator`, then `cd src`, then `python3 keypoint_server.py`. This can be running the entire time the drone is on you never should need to restart it. It needs to be running before you run anything else though.
20. Launch `roslaunch softdrone_core full_drone.launch` on the drone.
21. Probably, it will yell at you that the T265 isn't found, so unplug it and plug it back in and try again.
25. Reboot the drone again from QGroundControl.
27. Evenutally, the output will switch to "WAITING FOR ARM". Make sure the transmitter buttons are in these states: Flight mode: 0, AUX: 1, FLAP: 0, RATE: HI, MOTOR: ARM. The MOTOR is the kill switch, be ready to flip it if something goes bad.
28. Its time. Press the throttle (left joystick) all the way down and to the right. The propellers will start spinning. Then the drone will take off and do its thing. 

