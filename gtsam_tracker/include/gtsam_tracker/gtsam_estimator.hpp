#pragma once

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/inference/Key.h>


#include "gtsam_tracker/ConstantVelocityCustomFactor.hpp"
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/navigation/NavState.h>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace sdrone {
class GTSAMEstimator {
    public:
        // add new timestep
        
        void add_new_factors(Eigen::Vector3d target_position, Eigen::Vector3d target_velocity,
                             Eigen::Vector4d target_quat, Eigen::Vector3d target_angular_velocity, double time,
                             Eigen::Matrix<double, 6,6> pos_rot_cov, Eigen::Vector3d velocity_cov_diagonal,
                             Eigen::Vector3d angular_velocity_cov_diagonal,
                             Eigen::Matrix<double, 12, 1> transition_prior_cov);

        void insert_factors(gtsam::Key current_key, double time, double dt, gtsam::NavState nav_state,
                            gtsam::noiseModel::Gaussian::shared_ptr nav_noise,
                            gtsam::Vector3 angular_velocity,
                            gtsam::noiseModel::Diagonal::shared_ptr angular_vel_noise,
                            gtsam::noiseModel::Diagonal::shared_ptr transition_noise);
        void update_solution();

        GTSAMEstimator(bool lag);
        GTSAMEstimator();

        gtsam::NavState current_nav_state_;
        gtsam::Vector3 current_angle_vel_;

        gtsam::Matrix current_nav_cov_;
        gtsam::Matrix current_angle_vel_cov_;
        bool result_ready_;

    private:
        double lag_;
        double time_last_;
        gtsam::noiseModel::Diagonal::shared_ptr transition_noise_prior_;
        gtsam::IncrementalFixedLagSmoother smoother_;
        gtsam::NonlinearFactorGraph new_factors_;
        gtsam::Values new_values_;
        gtsam::FixedLagSmoother::KeyTimestampMap new_timestamps_;
        gtsam::Key current_key_index_;


}; // class GTSAMEstimator
} // namespace sdrone
