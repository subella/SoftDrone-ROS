#include "gtsam_tracker/gtsam_estimator_ros.hpp"

namespace sdrone {

GTSAMNode::GTSAMNode(const ros::NodeHandle &nh) : nh_(nh), agent_sub_(nh_, "agent_odom", 1), target_rel_sub_(nh_, "estimated_relative_pose", 1), 
sync_(SyncPolicy(1000), agent_sub_, target_rel_sub_), tf_listener_(tf_buffer_) {
    is_initialized_ = false;
    target_pose_pub_ = nh_.advertise<PoseWCovStamped>("target_global_pose_estimate", 1);
    target_odom_pub_ = nh_.advertise<Odom>("target_global_odom_estimate", 1);
    sync_.registerCallback(boost::bind(&GTSAMNode::syncCallback, this, _1, _2));
    nh_.getParam("velocity_prior_var", velocity_prior_var_);
    nh_.getParam("angular_velocity_prior_var", angular_velocity_prior_var_);
    nh_.getParam("transition_position_prior_cov", transition_position_prior_cov_);
    nh_.getParam("transition_rotation_prior_cov", transition_rotation_prior_cov_);
    nh_.getParam("transition_velocity_prior_cov", transition_velocity_prior_cov_);
    nh_.getParam("transition_angular_velocity_prior_cov", transition_angular_velocity_prior_cov_);
    nh_.getParam("target_frame_id", target_frame_id_);
    double lag;
    nh_.getParam("smoother_lag", lag);
    estimator_ = GTSAMEstimator(lag);
}

void GTSAMNode::syncCallback(const Odom::ConstPtr &odom, const PoseWCovStamped::ConstPtr &pwcs) {
    frame_id_ = odom->header.frame_id;
    double time = odom->header.stamp.toSec();

    geometry_msgs::TransformStamped optical_to_body_tf = tf_buffer_.lookupTransform("base_link", pwcs->header.frame_id, ros::Time(0), ros::Duration(1.0));
    geometry_msgs::PoseWithCovarianceStamped pwcs_body;
    tf2::doTransform(*pwcs, pwcs_body, optical_to_body_tf);

    if (odom->child_frame_id != pwcs_body.header.frame_id) {
        ROS_ERROR_STREAM("Error: odom->child_frame_id (" << odom->child_frame_id << ") != pwcs.header.frame_id (" << pwcs_body.header.frame_id << ")");
    }

    Belief6D b_agent = belief6DFromPoseWCov(odom->pose);
    Belief6D b_target_rel = belief6DFromPoseWCov(pwcs_body.pose);
    Belief6D b_target_meas = b_agent + b_target_rel;
    updateFromRos(b_target_meas, time);
    estimator_.update_solution();
    if (estimator_.result_ready_) {
        gtsam::NavState ns = estimator_.current_nav_state_;
        gtsam::Matrix ns_cov = estimator_.current_nav_cov_;
        gtsam::Vector3 omegas = estimator_.current_angle_vel_;
        gtsam::Matrix omegas_cov = estimator_.current_angle_vel_cov_;
        publishTargetEstimate(ns, ns_cov, omegas, omegas_cov);
    }
}

void GTSAMNode::publishTargetEstimate(gtsam::NavState ns, gtsam::Matrix ns_cov, gtsam::Vector3 omegas, gtsam::Matrix omegas_cov) {
    geometry_msgs::PoseWithCovarianceStamped pwcs;

    pwcs.pose.pose.position.x = ns.position().x();
    pwcs.pose.pose.position.y = ns.position().y();
    pwcs.pose.pose.position.z = ns.position().z();

    for (int ix = 0; ix < 6; ++ix) {
        for (int jx = 0; jx < 6; ++jx) {
            pwcs.pose.covariance[6*ix + jx] = ns_cov(ix, jx);
        }
    }

    gtsam::Vector quat = ns.pose().rotation().quaternion();

    pwcs.pose.pose.orientation.w = quat(0);
    pwcs.pose.pose.orientation.x = quat(1);
    pwcs.pose.pose.orientation.y = quat(2);
    pwcs.pose.pose.orientation.z = quat(3);

    pwcs.header.stamp = ros::Time::now(); // TODO: change this to when the observation was made
    pwcs.header.frame_id = frame_id_;
    target_pose_pub_.publish(pwcs);

    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time::now(); // TODO change this to when observation was made
    odom_msg.header.frame_id = frame_id_;
    odom_msg.child_frame_id = target_frame_id_;
    odom_msg.pose = pwcs.pose; // the odom message has the same pose as our previous message
    gtsam::Vector3 b_vel = ns.bodyVelocity();
    odom_msg.twist.twist.linear.x = b_vel(0);
    odom_msg.twist.twist.linear.y = b_vel(1);
    odom_msg.twist.twist.linear.z = b_vel(2);
    odom_msg.twist.twist.angular.x = omegas(0);
    odom_msg.twist.twist.angular.y = omegas(1);
    odom_msg.twist.twist.angular.z = omegas(2);

    for (int ix = 0; ix < 3; ++ix) {
        for (int jx = 0; jx < 3; ++jx) {
            odom_msg.twist.covariance[ix*6 + jx] = ns_cov(ix, jx);
            odom_msg.twist.covariance[(3 + ix)*6 + jx + 3] = omegas_cov(ix, jx); // covariance[ix + 3][jx + 3]
        }
    }
    
    target_odom_pub_.publish(odom_msg);

    geometry_msgs::TransformStamped transformStamped;
    
    transformStamped.header.stamp = ros::Time::now(); // TODO change to time observation was made
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "target";
    transformStamped.transform.translation.x = pwcs.pose.pose.position.x;
    transformStamped.transform.translation.y = pwcs.pose.pose.position.y;
    transformStamped.transform.translation.z = pwcs.pose.pose.position.z;

    tf2::Quaternion q;
    //std::cout << "tf yaw from gtsam: " << ns.pose().rotation().yaw() << std::endl;
    //q.setRPY(ns.pose().rotation.roll(), ns.pose().rotation().pitch(), ns.pose().rotation().yaw());
    //transformStamped.transform.rotation.x = q.x();
    //transformStamped.transform.rotation.y = q.y();
    //transformStamped.transform.rotation.z = q.z();
    //transformStamped.transform.rotation.w = q.w();
    transformStamped.transform.rotation.w = quat(0);
    transformStamped.transform.rotation.x = quat(1);
    transformStamped.transform.rotation.y = quat(2);
    transformStamped.transform.rotation.z = quat(3);

    //transformStamped.transform.rotation.x = pwcs.pose.pose.orientation.x;
    //transformStamped.transform.rotation.y = pwcs.pose.pose.orientation.y;
    //transformStamped.transform.rotation.z = pwcs.pose.pose.orientation.z;
    //transformStamped.transform.rotation.w = pwcs.pose.pose.orientation.w;
    
    transform_broadcaster_.sendTransform(transformStamped);

}

void GTSAMNode::updateFromRos(Belief6D obs, double time) {

    Belief7D obs_quat(obs);
    // Extract position and orientation from observation vector
    Eigen::Vector3d position;
    position << obs.mean.m_coords[0], obs.mean.m_coords[1], obs.mean.m_coords[2];
    Eigen::Vector4d quat;
    quat << obs_quat.mean.m_quat.r(), obs_quat.mean.m_quat.x(), obs_quat.mean.m_quat.y(), obs_quat.mean.m_quat.z();

    // velocity priors are 0-mean with extremely large variance
    Eigen::Vector3d velocity;
    velocity << 0.0, 0.0, 0.0;
    Eigen::Vector3d angular_velocity;
    angular_velocity << 0.0, 0.0, 0.0;

    // covariances:
    // position/orientation -- 6x6 (dense)
    // velocity -- 3x3 (diagonal)
    // angular velocity -- 3x3 (diagonal)
    // transition prior -- 12x12 (diagonal)
   
    Eigen::Matrix<double, 6, 6> temp = obs.cov.asEigen();
    Eigen::Matrix<double, 6, 6> pos_rot_cov = obs.cov.asEigen();

    const int map[6] = {0,1,2,5,4,3};
    for (int i=0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            pos_rot_cov(i, j) = temp(map[i], map[j]);
        }
    }

    Eigen::Vector3d velocity_cov(velocity_prior_var_, velocity_prior_var_, velocity_prior_var_);
    Eigen::Vector3d angular_velocity_cov(angular_velocity_prior_var_, angular_velocity_prior_var_, angular_velocity_prior_var_);
    // position, rotation, velocity, angular_velocity

    Eigen::Matrix<double, 12, 1> transition_prior_cov;
    transition_prior_cov <<
        transition_position_prior_cov_, transition_position_prior_cov_, transition_position_prior_cov_,
        transition_rotation_prior_cov_, transition_rotation_prior_cov_, transition_rotation_prior_cov_,
        transition_velocity_prior_cov_, transition_velocity_prior_cov_, transition_velocity_prior_cov_, 
        transition_angular_velocity_prior_cov_, transition_angular_velocity_prior_cov_, transition_angular_velocity_prior_cov_;

    estimator_.add_new_factors(position, velocity, quat, angular_velocity, time, pos_rot_cov, velocity_cov, angular_velocity_cov, transition_prior_cov);
}

} // namespace sdrone
