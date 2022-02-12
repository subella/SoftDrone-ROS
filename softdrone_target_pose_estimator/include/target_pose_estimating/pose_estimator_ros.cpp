// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    pose_estimator_ros.cpp
 * @author  Samuel Ubellacker
 */
//-----------------------------------------------------------------------------

#include <target_pose_estimating/pose_estimator_ros.hpp>

namespace sdrone
{

PoseEstimatorROS::
PoseEstimatorROS(const ros::NodeHandle &nh)
  : nh_(nh),
    PoseEstimator()
{
};

PoseEstimatorROS::
PoseEstimatorROS(const ros::NodeHandle& nh,
                 const std::string&     keypoints_3D_topic,
                 const std::string&     pose_topic,
                 const std::string&     cad_frame_file_name,
                 const TeaserParams&    params)
  : nh_(nh),
    PoseEstimator(cad_frame_file_name, params)
{
  keypoints_sub_ = nh_.subscribe(keypoints_3D_topic, 1, &PoseEstimatorROS::keypoints3DCallback, this);
  pose_pub_ = nh_.advertise<PoseWCov>(pose_topic,  1);
};

void PoseEstimatorROS::
keypoints3DCallback(const Keypoints3D& keypoints_3D)
{

  ROS_INFO_STREAM("called");
  Eigen::Matrix3Xd keypoints_3D_mat;
  keypoints3DToEigen(keypoints_3D, keypoints_3D_mat);

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  ROS_INFO_STREAM("HERE");
  int success = solveTransformation(keypoints_3D_mat, R, t);
  ROS_INFO_STREAM(success);

  if (success)
  {
    PoseWCov pose;
    eigenToPoseWCov(R, t, pose);
    pose_pub_.publish(pose);
  }
}

void PoseEstimatorROS::
keypoints3DToEigen(const Keypoints3D& keypoints_3D, Eigen::Matrix3Xd& keypoints_3D_mat)
{

}

void PoseEstimatorROS::
eigenToPoseWCov(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, PoseWCov& pose)
{

}

}; //namespace soft