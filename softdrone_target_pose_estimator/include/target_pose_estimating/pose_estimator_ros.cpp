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
keypoints3DCallback(const Keypoints3DMsg& keypoints_3D_msg)
{

  auto keypoints_3D = keypoints_3D_msg.keypoints_3D;
  int num_kpts = keypoints_3D.size();
  Eigen::Matrix3Xd keypoints_3D_mat(3, num_kpts);
  keypoints3DToEigen(keypoints_3D, keypoints_3D_mat);

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  int success = solveTransformation(keypoints_3D_mat, R, t);

  if (success)
  {
    PoseWCov pose_cov;
    eigenToPoseWCov(R, t, pose_cov);
    pose_pub_.publish(pose_cov);
  }
}

void PoseEstimatorROS::
keypoints3DToEigen(const std::vector<Keypoint3DMsg> keypoints_3D, Eigen::Matrix3Xd& keypoints_3D_mat)
{
  for (int i=0; i < keypoints_3D.size(); i++)
  {
    Keypoint3DMsg keypoint_3D = keypoints_3D.at(i);
    keypoints_3D_mat.col(i) << keypoint_3D.x, keypoint_3D.y, keypoint_3D.z;
  }  
}

void PoseEstimatorROS::
eigenToPoseWCov(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, PoseWCov& pose_cov)
{
  pose_cov.pose.position.x = t[0];
  pose_cov.pose.position.y = t[1];
  pose_cov.pose.position.z = t[2];
  
  Eigen::Quaterniond q(R);
  pose_cov.pose.orientation.w = q.w();
  pose_cov.pose.orientation.x = q.x();
  pose_cov.pose.orientation.y = q.y();
  pose_cov.pose.orientation.z = q.z();
}

}; //namespace sdrone