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

void load_params(const ros::NodeHandle &nh)
{
  nh.getParam("tracker_params/agent_topic", agent_topic_);
  nh.getParam("tracker_params/target_relative_topic", target_rel_topic_);
  nh.getParam("tracker_params/target_topic", target_topic_);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "tracker_node");
	ros::NodeHandle nh;

	ROS_INFO("tracker_node running...");

	load_params(nh);
	sdrone::TrackerROS tracker(nh, agent_topic_, target_rel_topic_, target_topic_);
	ros::spin();

	return 0;
}

