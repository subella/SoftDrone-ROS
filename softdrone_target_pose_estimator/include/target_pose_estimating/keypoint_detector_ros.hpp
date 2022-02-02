// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    keypoint_detector_ros.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief ROS wrapper for interfacing keypoint detector with ROS.
 */
//-----------------------------------------------------------------------------

#ifndef Keypoint_Detector_ROS_HPP
#define Keypoint_Detector_ROS_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <target_pose_estimating/keypoint_detector.hpp>

#include "softdrone_target_pose_estimator/Keypoints.h"

namespace softdrone
{

class KeypointDetectorROS : public KeypointDetector {
  public:
    typedef sensor_msgs::ImageConstPtr ImageMsg;
    typedef softdrone_target_pose_estimator::Keypoints Keypoints;

    KeypointDetectorROS(const ros::NodeHandle &nh);

    KeypointDetectorROS(const ros::NodeHandle &nh,
                        const std::string     &rgb_image_topic,
                        const std::string     &keypoints_topic,
                        const std::string     &model_file_name);

    ~KeypointDetectorROS() = default;


  private:

    ros::NodeHandle nh_;

    ros::Time time_stamp_;

    image_transport::Subscriber rgb_image_sub_;

    ros::Publisher keypoints_pub_;

    void rgbImageCallback(const ImageMsg& rgb_image);

    static Keypoints formatKeypoints(torch::Tensor tensor_kpts);
    
};

}; //namespace softdrone

#endif // Tracker_ROS_HPP