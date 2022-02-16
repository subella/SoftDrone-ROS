// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoint_detector_ros.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief ROS wrapper for interfacing pose estimator with ROS.
 */
//-----------------------------------------------------------------------------

#ifndef Pose_Estimator_ROS_HPP
#define Pose_Estimator_ROS_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <target_pose_estimating/pose_estimator.hpp>
#include "softdrone_target_pose_estimator/Keypoints.h"

namespace softdrone
{

class PoseEstimatorROS : public PoseEstimator {
  public:
    typedef sensor_msgs::Image ImageMsg;
    typedef softdrone_target_pose_estimator::Keypoints Keypoints;
    typedef geometry_msgs::PoseWithCovariance PoseWCov;
    typedef message_filters::sync_policies::ApproximateTime<ImageMsg, Keypoints> SyncPolicy;

    PoseEstimatorROS(const ros::NodeHandle &nh);

    PoseEstimatorROS(const ros::NodeHandle &nh,
                     const std::string     &depth_img_topic,
                     const std::string     &keypoint_topic,
                     const std::string     &pose_topic);

    ~PoseEstimatorROS() = default;


  private:

    ros::NodeHandle nh_;

    image_transport::ImageTransport it_;

    ros::Time time_stamp_;

    std::string frame_id_;

    image_transport::SubscriberFilter depth_img_sub_;

    message_filters::Subscriber<Keypoints> keypoints_sub_;

    message_filters::Synchronizer<SyncPolicy> sync_;

    ros::Publisher pose_pub_;

    void syncCallback(const ImageMsg::ConstPtr& depth_img, const Keypoints::ConstPtr& keypoints);
    
};

}; //namespace softdrone

#endif // Tracker_ROS_HPP