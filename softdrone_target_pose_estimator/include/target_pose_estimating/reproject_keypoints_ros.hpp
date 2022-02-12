// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    reproject_keypoints_ros.hpp
 * @author  Samuel Ubellacker
 * 
 * @brief ROS wrapper for reprojecting 2D keypoints to 3D.
 */
//-----------------------------------------------------------------------------

#ifndef REPROJECT_KEYPOINTS_ROS_HPP
#define REPROJECT_KEYPOINTS_ROS_HPP


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <target_pose_estimating/reproject_keypoints.hpp>

#include "softdrone_target_pose_estimator/Keypoints2D.h"
#include "softdrone_target_pose_estimator/Keypoints3D.h"

namespace sdrone
{

class ReprojectKeypointsROS : public ReprojectKeypoints {

  public:
    
    typedef softdrone_target_pose_estimator::Keypoint2D Keypoint2D;
    typedef softdrone_target_pose_estimator::Keypoints2D Keypoints2D;
    typedef softdrone_target_pose_estimator::Keypoint3D Keypoint3D;
    typedef softdrone_target_pose_estimator::Keypoints3D Keypoints3D;
    typedef sensor_msgs::Image ImageMsg;
    typedef sensor_msgs::CameraInfo CameraInfoMsg;
    typedef message_filters::sync_policies::ApproximateTime<Keypoints2D, ImageMsg> SyncPolicy;

    ReprojectKeypointsROS(const ros::NodeHandle &nh);

    ReprojectKeypointsROS(const ros::NodeHandle& nh,
                          const std::string&     keypoints_2D_topic,
                          const std::string&     keypoints_3D_topic,
                          const std::string&     rgb_cam_info_topic,
                          const std::string&     depth_img_topic);

    ~ReprojectKeypointsROS() = default;

  private:

    ros::NodeHandle nh_;

    ros::Time time_stamp_;

    std::string frame_id_;

    image_transport::ImageTransport it_;

    message_filters::Subscriber<Keypoints2D> keypoints_2D_sub_;

    ros::Subscriber rgb_cam_info_sub_;

    image_transport::SubscriberFilter depth_img_sub_;

    message_filters::Synchronizer<SyncPolicy> sync_;

    ros::Publisher keypoints_3D_pub_;

    void rgbCamInfoCallback(const CameraInfoMsg& camera_info_msg);

    void syncCallback(const Keypoints2D::ConstPtr& keypoints_2D_msg, const ImageMsg::ConstPtr& depth_img_msg);

    void makePxPyZ(const std::vector<Keypoint2D>& keypoints_2D, const cv::Mat& depth_img, 
                   Eigen::MatrixX2i& px_py_mat, Eigen::VectorXd& z_vec);

    void eigenToKeypoints3DMsg(Eigen::MatrixX3d& keypoints_3D_mat, Keypoints3D& keypoints_3D_msg);

};

}; //namespace sdrone

#endif // REPROJECT_KEYPOINT_ROS_HPP