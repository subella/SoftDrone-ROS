// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    pose_estimator_ros.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/pose_estimator_ros.hpp>

namespace softdrone
{

PoseEstimatorROS::
PoseEstimatorROS(const ros::NodeHandle &nh)
  : nh_(nh), 
    it_(nh_),
    depth_img_sub_(it_, "", 1),
    keypoints_sub_(nh_, "", 1),
    sync_(SyncPolicy(10), depth_img_sub_, keypoints_sub_)
{
  is_initialized_ = false;
};

PoseEstimatorROS::
PoseEstimatorROS(const ros::NodeHandle& nh,
              const std::string&     depth_img_topic,
              const std::string&     keypoint_topic,
              const std::string&     pose_topic)
  : nh_(nh), 
    it_(nh_),
    depth_img_sub_(it_, depth_img_topic, 1),
    keypoints_sub_(nh_, keypoint_topic, 1),
    sync_(SyncPolicy(10), depth_img_sub_, keypoints_sub_)
{
  is_initialized_ = false;

  pose_pub_ = nh_.advertise<PoseWCov>(pose_topic,  1);
  sync_.registerCallback(boost::bind(&PoseEstimatorROS::syncCallback, this, _1, _2));
};


void PoseEstimatorROS::
syncCallback(const ImageMsg::ConstPtr& depth_img, const Keypoints::ConstPtr& keypoints)
{
  ROS_INFO_STREAM("Callback called.");

};


}; //namespace soft