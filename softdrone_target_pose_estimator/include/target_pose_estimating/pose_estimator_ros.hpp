// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    pose_estimator_ros.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief ROS wrapper for interfacing pose estimator with ROS.
 */
//-----------------------------------------------------------------------------

#ifndef Pose_Estimator_ROS_HPP
#define Pose_Estimator_ROS_HPP

#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <target_pose_estimating/pose_estimator.hpp>
#include "softdrone_target_pose_estimator/Keypoints3D.h"

namespace sdrone
{

class PoseEstimatorROS : public PoseEstimator {
  public:
    typedef softdrone_target_pose_estimator::Keypoints3D Keypoints3D;
    typedef geometry_msgs::PoseWithCovariance PoseWCov;

    PoseEstimatorROS(const ros::NodeHandle &nh);

    PoseEstimatorROS(const ros::NodeHandle& nh,
                     const std::string&     keypoints_3D_topic,
                     const std::string&     pose_topic,
                     const std::string&     cad_frame_file_name,
                     const TeaserParams&    params);

    ~PoseEstimatorROS() = default;


  private:

    ros::NodeHandle nh_;

    ros::Time time_stamp_;

    std::string frame_id_;

    ros::Subscriber keypoints_sub_;

    ros::Publisher pose_pub_;

    void keypoints3DCallback(const Keypoints3D& keypoints_3D);

    void keypoints3DToEigen(const Keypoints3D& keypoints_3D, Eigen::Matrix3Xd& keypoints_3D_mat);

    void eigenToPoseWCov(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, PoseWCov& pose);

};

}; //namespace softdrone

#endif // Tracker_ROS_HPP