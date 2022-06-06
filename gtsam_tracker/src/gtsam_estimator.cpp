#include "gtsam_tracker/gtsam_estimator.hpp"

namespace sdrone {

GTSAMEstimator::GTSAMEstimator() : GTSAMEstimator::GTSAMEstimator(10.0) {}

GTSAMEstimator::GTSAMEstimator(bool lag) : lag_(lag) {
    result_ready_ = false;
    current_key_index_ = 0;

   // The Incremental version uses iSAM2 to perform the nonlinear optimization
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.0; // Set the relin threshold to zero such that the batch estimate is recovered
    parameters.relinearizeSkip = 1; // Relinearize every time
    smoother_ = gtsam::IncrementalFixedLagSmoother(lag_, parameters);

}

void GTSAMEstimator::insert_factors(gtsam::Key current_key, double time, double dt, gtsam::NavState nav_state, 
                                    gtsam::noiseModel::Gaussian::shared_ptr nav_noise, 
                                    gtsam::Vector3 angular_velocity, 
                                    gtsam::noiseModel::Diagonal::shared_ptr angular_vel_noise,
                                    gtsam::noiseModel::Diagonal::shared_ptr transition_noise) {

    new_factors_.addPrior(current_key, nav_state, nav_noise);
    new_values_.insert(current_key, nav_state);
    new_timestamps_[current_key] = time;

    new_factors_.addPrior(current_key + 1, angular_velocity, angular_vel_noise);
    new_values_.insert(current_key + 1, angular_velocity);
    new_timestamps_[current_key + 1] = time;

    if (current_key >= 2) {
        ConstantVelocityCustomFactor transition_factor(current_key - 2, current_key - 1, current_key, current_key + 1, dt, 
                transition_noise);
        new_factors_.push_back(transition_factor);
    }
}

void GTSAMEstimator::add_new_factors(Eigen::Vector3d target_position, Eigen::Vector3d target_velocity,
                                     Eigen::Vector4d target_quat, Eigen::Vector3d target_angular_velocity, double time,
                                     Eigen::Matrix<double, 6,6> pos_rot_cov, Eigen::Vector3d velocity_cov_diagonal,
                                     Eigen::Vector3d angular_velocity_cov_diagonal, 
                                     Eigen::Matrix<double, 12, 1> transition_prior_cov) {
    // target_quat is w,x,y,z

    gtsam::Rot3 rotation(target_quat(0), target_quat(1), target_quat(2), target_quat(3));
    gtsam::Point3 point(target_position(0), target_position(1), target_position(2));
    gtsam::Pose3 pose(rotation, point);
    gtsam::Vector3 velocity(target_velocity(0), target_velocity(1), target_velocity(2));
    gtsam::NavState nav_state(pose, velocity);

    Eigen::Matrix<double, 9, 9> nav_state_cov = Eigen::Matrix<double, 9, 9>::Zero();
    nav_state_cov.block(0,0,6,6) = pos_rot_cov;
    nav_state_cov(6,6) = velocity_cov_diagonal(0);
    nav_state_cov(7,7) = velocity_cov_diagonal(1);
    nav_state_cov(8,8) = velocity_cov_diagonal(2);

    gtsam::noiseModel::Gaussian::shared_ptr nav_noise = gtsam::noiseModel::Gaussian::Covariance(nav_state_cov);

    gtsam::Vector3 angular_velocity(target_angular_velocity(0), target_angular_velocity(1), target_angular_velocity(2));
    gtsam::noiseModel::Diagonal::shared_ptr angular_velocity_noise = gtsam::noiseModel::Diagonal::Sigmas(angular_velocity_cov_diagonal);

    gtsam::noiseModel::Diagonal::shared_ptr transition_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(transition_prior_cov);

    insert_factors(current_key_index_, time, time - time_last_, nav_state, nav_noise, angular_velocity, angular_velocity_noise, transition_prior_noise);
    time_last_ = time;

    current_key_index_ += 2;

}

void GTSAMEstimator::update_solution() {

    smoother_.update(new_factors_, new_values_, new_timestamps_);
    for (int ix=0; ix < 10; ++ix) {
        smoother_.update();
    }

    if (current_key_index_ > 0) {
        gtsam::Values result = smoother_.calculateEstimate();
        current_nav_state_ = result.at<gtsam::NavState>(current_key_index_ - 2);
        current_angle_vel_ = result.at<gtsam::Vector3>(current_key_index_ - 1);
        gtsam::Marginals marginals(smoother_.getFactors(), result);
        current_nav_cov_ = marginals.marginalCovariance(current_key_index_ - 2);
        current_angle_vel_cov_ = marginals.marginalCovariance(current_key_index_ - 1);
        result_ready_ = true;
    }

    // Clear containerss for the next iteration
    new_timestamps_.clear();
    new_values_.clear();
    new_factors_.resize(0);

}

} // namespace sdrone
