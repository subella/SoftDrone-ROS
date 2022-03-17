// -*-c++-*-
//-----------------------------------------------------------------------------
/**
 * @file    tracker_ros.hpp
 * @author  Jared Strader
 * 
 * @brief ROS wrapper for interfacing tracker woth ROS
 */
//-----------------------------------------------------------------------------

#ifndef Tracker_ROS_HPP
#define Tracker_ROS_HPP

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Odometry.h>

#include <target_tracking/tracker.hpp>

namespace sdrone
{

class TrackerROS : public Tracker {
  public:
    typedef geometry_msgs::Pose Pose;
    typedef geometry_msgs::PoseWithCovariance PoseWCov;
    typedef geometry_msgs::PoseWithCovarianceStamped PoseWCovStamp;
    typedef nav_msgs::Odometry Odom;
    typedef message_filters::sync_policies::ApproximateTime<Odom, PoseWCovStamp> SyncPolicy;


    /** \brief */
    TrackerROS(const ros::NodeHandle &nh);

    /** \brief  */
    ~TrackerROS() = default;

    /** \brief  The mean is the 7D pose [x, y, z, qw, qx, qy, qz] and the
     * covariance is with respect to the 6D pose [x, y, z, thetax, thetay,
     * thetaz] where order is ZYX */
    static Pose samplePose(const Eigen::VectorXd &mu,
                           const Eigen::MatrixXd &cov);

  private:
    /** \brief */
    ros::NodeHandle nh_;

    /** \brief */
    std::string frame_id_;

    /** \brief */
    ros::Time time_stamp_;

    /** \brief Subscriber to agent odometry */
    message_filters::Subscriber<Odom> agent_sub_;

    /** \brief Subscriber to target pose */
    message_filters::Subscriber<PoseWCovStamp> target_rel_sub_;

    /** \brief Synchronizer for agent and target data */
    message_filters::Synchronizer<SyncPolicy> sync_;

    /** \brief */
    ros::Publisher target_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    ros::Timer publish_timer_;

    /** \brief */
    void syncCallback(const Odom::ConstPtr &odom, const PoseWCovStamp::ConstPtr &pwcs);

    /** \brief */
    void publishResults7D();
    void publishResults6D();

    void ekfOutputCallback();

    /** \brief */
    static Belief7D belief7DFromPoseWCov(const PoseWCov &pwc);
    static Belief6D belief6DFromPoseWCov(const PoseWCov &pwc);

    /** \brief */
    static PoseWCov poseWCovFromBelief(const Belief7D &b);

    /** \brief */
    static PoseWCov poseWCovFromBelief(const Belief6D &b);

    /** \brief */
    static Pose6D getPose6DFromPoseWCov(const PoseWCov &pwc);

    /** \brief */
    static CMat66 getCov6DFromPoseWCov(const PoseWCov &pwc);

    /** \brief */
    static Pose getPoseFromPose7D(const Pose7D &pose);

    //265
    
};

}; //namespace sdrone

#endif // Tracker_ROS_HPP
