// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    tracker_node.hpp
 * @author  Jared Strader
 * 
 * @brief Runs an EKF for estimating the pose of the target (in the global 
 *        frame). Subscribes to the Odometry topic for accessing the agent pose
 *        (in the global frame) and the PoseWithCovarianceStamped topic for the 
 *        target (relative to the agent).
 */
//-----------------------------------------------------------------------------

#include <ros/ros.h>
#include <target_tracking/tracker_ros.hpp>

std::string agent_topic_;
std::string target_rel_topic_;
std::string target_topic_;

int main(int argc, char **argv)
{
	ros::init(argc, argv, "tracker_node");
	ros::NodeHandle nh;

	ROS_INFO("tracker_node running...");

	sdrone::TrackerROS tracker(nh);
	ros::spin();

	return 0;
}

