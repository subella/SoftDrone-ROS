// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    reproject_keypoints_node.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief Reprojects 2D keypoint pixel locations
 *        to 3D coordinates using depth image and
 *        camera intrinsics.
 */
//-----------------------------------------------------------------------------

#include <ros/ros.h>
#include <target_pose_estimating/reproject_keypoints_ros.hpp>

std::string keypoints_2D_sub_topic_;
std::string keypoints_3D_sub_topic_;
std::string rgb_cam_info_sub_topic_;
std::string depth_img_sub_topic_;

std::string keypoints_2D_pub_topic_;
std::string keypoints_3D_pub_topic_;

void load_params(const ros::NodeHandle& nh)
{
  nh.getParam("reproject_keypoints_params/keypoints_2D_sub_topic", keypoints_2D_sub_topic_);
  nh.getParam("reproject_keypoints_params/keypoints_3D_sub_topic", keypoints_3D_sub_topic_);
  nh.getParam("reproject_keypoints_params/rgb_cam_info_sub_topic", rgb_cam_info_sub_topic_);
  nh.getParam("reproject_keypoints_params/depth_img_sub_topic",    depth_img_sub_topic_);

  nh.getParam("reproject_keypoints_params/keypoints_2D_pub_topic", keypoints_2D_pub_topic_);
  nh.getParam("reproject_keypoints_params/keypoints_3D_pub_topic", keypoints_3D_pub_topic_);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "reproject_keypoints_node");
	ros::NodeHandle nh;

	ROS_INFO("reproject_keypoints_node running...");

	load_params(nh);
	sdrone::ReprojectKeypointsROS reproject_keypoints(nh,
                                                    keypoints_2D_sub_topic_,
                                                    keypoints_3D_sub_topic_,
                                                    rgb_cam_info_sub_topic_,
                                                    depth_img_sub_topic_,
                                                    keypoints_2D_pub_topic_,
                                                    keypoints_3D_pub_topic_);
	ros::spin();

	return 0;
}