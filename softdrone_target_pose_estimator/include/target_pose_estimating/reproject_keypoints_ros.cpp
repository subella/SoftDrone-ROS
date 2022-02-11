// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    reproject_keypoints_ros.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/reproject_keypoints_ros.hpp>

namespace sdrone
{

ReprojectKeypointsROS::
ReprojectKeypointsROS(const ros::NodeHandle &nh)
  : nh_(nh), 
    it_(nh_),
    keypoints_2D_sub_(nh_, "", 1),
    depth_img_sub_(it_, "", 1),
    sync_(SyncPolicy(10), keypoints_2D_sub_, depth_img_sub_)
{
  is_initialized_ = false;
};

ReprojectKeypointsROS::
ReprojectKeypointsROS(const ros::NodeHandle& nh,
                      const std::string&     keypoints_2D_topic,
                      const std::string&     keypoints_3D_topic,
                      const std::string&     rgb_cam_info_topic,
                      const std::string&     depth_img_topic)
  : nh_(nh), 
    it_(nh_),
    keypoints_2D_sub_(nh_, keypoints_2D_topic, 1),
    depth_img_sub_(it_, depth_img_topic, 1),
    sync_(SyncPolicy(10), keypoints_2D_sub_, depth_img_sub_)
{
  sync_.registerCallback(boost::bind(&ReprojectKeypointsROS::syncCallback, this, _1, _2));
  keypoints_3D_pub_ = nh_.advertise<Keypoints3D>(keypoints_3D_topic,  1);
};


void ReprojectKeypointsROS::
syncCallback(const Keypoints2D::ConstPtr& keypoints_2D, const ImageMsg::ConstPtr& depth_img)
{
  ROS_INFO_STREAM("Callback called.");
};

}; //namespace sdrone