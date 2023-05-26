#pragma once

#include "mrpt_utils.hpp"

#include <gtsam/navigation/NavState.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_msgs/Bool.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <ros/console.h>
#include <nav_msgs/Odometry.h>


#include <Eigen/Dense>

#include "gtsam_estimator.hpp"

namespace sdrone {
class GTSAMNode {
    public:
        typedef geometry_msgs::PoseWithCovarianceStamped PoseWCovStamped;
        typedef geometry_msgs::PoseStamped PoseStamped;
        typedef std_msgs::Bool Bool;
        typedef nav_msgs::Odometry Odom;
        typedef message_filters::sync_policies::ApproximateTime<Odom, PoseWCovStamped> SyncPolicy;
        typedef message_filters::sync_policies::ApproximateTime<PoseWCovStamped, PoseWCovStamped> SyncPolicyPose;

        GTSAMNode(const ros::NodeHandle &nh);

    private:
        ros::NodeHandle nh_;

        std::string frame_id_;

        ros::Subscriber reset_sub_;
        message_filters::Subscriber<Odom> agent_sub_;
        message_filters::Subscriber<PoseWCovStamped> agent_sub_pose_;
        message_filters::Subscriber<PoseWCovStamped> target_rel_sub_;
        
        message_filters::Synchronizer<SyncPolicy> sync_;
        message_filters::Synchronizer<SyncPolicyPose> sync_pose_;

        GTSAMEstimator estimator_;

        ros::Publisher target_pose_pub_;
        ros::Publisher target_odom_pub_;
        ros::Publisher teaser_global_ps_pub_;

        bool is_initialized_;
        double lag_;

        std::string target_frame_id_;
        double process_covariance_trans_;
        double process_covariance_rot_;
        double velocity_prior_var_;
        double angular_velocity_prior_var_;
        double transition_position_prior_cov_;
        double transition_rotation_prior_cov_;
        double transition_velocity_prior_cov_;
        double transition_angular_velocity_prior_cov_;

        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;
        tf2_ros::TransformBroadcaster transform_broadcaster_;

        void reset_callback(const Bool::ConstPtr &bool_msg);
        void syncCallback(const Odom::ConstPtr &odom, const PoseWCovStamped::ConstPtr &pwcs);
        void syncCallbackPose(const PoseWCovStamped::ConstPtr &pwcs_drone, const PoseWCovStamped::ConstPtr &pwcs_target);
        void publishTargetEstimate(gtsam::NavState ns, gtsam::Matrix ns_cov, gtsam::Vector3 omegas, gtsam::Matrix omegas_cov);
        void updateFromRos(Belief6D obs, double time);

}; // class GTSAMNode
} // namespace sdrone


