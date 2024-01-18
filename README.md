# Overview
This is the core repository for the Soft Drone grasping pipeline.

* `SoftDrone-TrajOpt` contains the implementation of the polynomial trajectory planner.
* `softdrone_core` is the core package and contains the flight state machine and entry points.
* `gtsam_tracker` contains the implementation of the fixed lag smoother
* `softdrone_target_pose_estimator` contains the keypoint detector and TEASER++
* `softdrone_target_tracking` is not used.
