// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    plotter_node.cpp
 * @author  Samuel Ubellacker
 * 
 * @brief Plots relevant information.
 */
//-----------------------------------------------------------------------------

#include <ros/ros.h>
#include <target_pose_estimating/plotter_ros.hpp>

std::string rgb_img_sub_topic_;
std::string cad_keypoints_sub_topic_;
std::string estimated_pose_sub_topic_;
std::string reprojected_cad_keypoints_img_pub_topic_;

void load_params(const ros::NodeHandle& nh)
{
  nh.getParam("plotter_params/rgb_img_sub_topic", rgb_img_sub_topic_);
  nh.getParam("plotter_params/cad_keypoints_sub_topic", cad_keypoints_sub_topic_);
  nh.getParam("plotter_params/estimated_pose_sub_topic", estimated_pose_sub_topic_);
  nh.getParam("plotter_params/reprojected_cad_keypoints_img_pub_topic", reprojected_cad_keypoints_img_pub_topic_);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "plotter_node");
	ros::NodeHandle nh;

	ROS_INFO("plotter_node running...");


	load_params(nh);
	sdrone::PlotterROS plotter(nh,
                             rgb_img_sub_topic_,
                             cad_keypoints_sub_topic_,
                             estimated_pose_sub_topic_,
                             reprojected_cad_keypoints_img_pub_topic_);
	ros::spin();


	return 0;
}
