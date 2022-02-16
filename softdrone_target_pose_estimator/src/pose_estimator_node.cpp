// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    pose_estimator_node.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief Given detected 3D keypoints, estimates the pose of the 
 *        target using point cloud registration with Teaser.
 */
//-----------------------------------------------------------------------------

#include <ros/ros.h>
#include <target_pose_estimating/pose_estimator_ros.hpp>

std::string depth_img_topic_;
std::string pose_topic_;
std::string cad_frame_file_name_;

void load_params(const ros::NodeHandle& nh)
{
  nh.getParam("keypoint_detector_params/depth_image_topic", depth_img_topic_);
  nh.getParam("keypoint_detector_params/pose_topic", pose_topic_);
  nh.getParam("keypoint_detector_params/cad_frame_file_name", cad_frame_file_name_);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "pose_estimator_node");
	ros::NodeHandle nh;

	ROS_INFO("pose_estimator_node running...");


	load_params(nh);
	softdrone::PoseEstimatorROS pose_estimator(nh, 
                                             depth_img_topic_,
                                             pose_topic_,
                                             cad_frame_file_name);
	ros::spin();

	return 0;
}