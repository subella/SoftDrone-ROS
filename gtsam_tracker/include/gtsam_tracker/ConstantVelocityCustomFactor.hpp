#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>

class ConstantVelocityCustomFactor: public gtsam::NoiseModelFactor4<gtsam::NavState, gtsam::Vector3, gtsam::NavState, gtsam::Vector3> {
    double dt_;

public:
  ConstantVelocityCustomFactor(gtsam::Key i, gtsam::Key j, gtsam::Key k, gtsam::Key l, double dt, const gtsam::SharedNoiseModel& model):
    gtsam::NoiseModelFactor4<gtsam::NavState, gtsam::Vector3, gtsam::NavState, gtsam::Vector3>(model, i, j, k, l), dt_(dt) {}

    gtsam::Vector evaluateError(const gtsam::NavState& ns1, const gtsam::Vector3& omegas1, const gtsam::NavState& ns2, const gtsam::Vector3& omegas2,
                       boost::optional<gtsam::Matrix&> H1 = boost::none,
                       boost::optional<gtsam::Matrix&> H2 = boost::none,
                       boost::optional<gtsam::Matrix&> H3 = boost::none,
                       boost::optional<gtsam::Matrix&> H4 = boost::none) const {

        static const gtsam::Vector3 b_accel{0.0, 0.0, 0.0}; // TODO: even for "constant velocity", this is not 0, it should be a vector perpendicular to the target's motion with magnitude depending on omega (curature)

        gtsam::Matrix99 predicted_H_x1;
        gtsam::Matrix93 predicted_H_x2;
        gtsam::NavState predicted = ns1.update(b_accel, omegas1, dt_, H1 ? &predicted_H_x1 : nullptr, {}, H2 ? &predicted_H_x2 : nullptr);

        gtsam::Matrix99 error_H_predicted;
        gtsam::Matrix h3(9, 3);
        gtsam::Vector9 error_nav = predicted.localCoordinates(ns2, H1 ? &error_H_predicted : nullptr, &h3);

        if (H1) { // 12 x 9, last 3 rows are 0
            *H1 = (gtsam::Matrix(12,9) << error_H_predicted * predicted_H_x1, Eigen::MatrixXd::Zero(3,9)).finished();
        }

        if (H3) {
            *H3 = (gtsam::Matrix(12,9) << h3, Eigen::MatrixXd::Zero(3,9)).finished();
        }

        gtsam::Vector3 error_omegas = omegas2 - omegas1;

        if (H2) { // 12 x 3, last three rows are -identity
            *H2 = (gtsam::Matrix(12, 3) << error_H_predicted * predicted_H_x2, -Eigen::MatrixXd::Identity(3,3)).finished();
        }

        if (H4) {
            *H4 = (gtsam::Matrix(12,3) << Eigen::MatrixXd::Zero(9,3), Eigen::MatrixXd::Identity(3,3)).finished();
        }

        return (gtsam::Vector(12) << error_nav, error_omegas).finished();
  }
};
